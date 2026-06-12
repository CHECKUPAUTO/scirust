//! # OpGraph — Graphe de dépendance des opérations
//!
//! Représente le graphe de calcul forward comme un DAG (Directed Acyclic Graph).
//! Chaque nœud est une opération avec ses inputs (indices dans le graphe).

/// Type d'opération supporté par le graphe de fusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpKind {
    // Matrix operations
    MatMul,
    MatMulGpu,
    Linear,

    // Activation functions
    ReLU,
    SiLU,
    Gelu,
    GeluApprox,
    Sigmoid,
    Tanh,
    Softmax,
    LogSoftmax,
    Identity,

    // Normalization
    LayerNorm,
    LayerNormFused, // LayerNorm avec bias/scale intégré
    BatchNorm,
    RMSNorm, // RMSNorm (simplifié, pas de mean)

    // Element-wise
    Add,
    Sub,
    Mul,
    Div,
    Scale,
    Relu,

    // Reshape / layout
    Reshape,
    Transpose,
    View,
    Flatten,
    Concat,
    Slice,

    // Loss
    CrossEntropy,
    MseLoss,
    NllLoss,

    // Pooling / other
    MaxPool2d,
    AvgPool2d,
    Dropout,
    Dropout2d,

    // Mamba/SSM specific
    SsmInit,
    SsmStep,
    SsmScan,

    // Quantization
    Quantize,
    Dequantize,

    // Input / output markers
    Input,
    Output,
}

/// Une opération dans le graphe de fusion.
#[derive(Debug, Clone)]
pub struct FusedOp {
    /// Type d'opération.
    pub kind: OpKind,
    /// Indices des inputs (nœuds parents dans le DAG).
    pub inputs: Vec<usize>,
    /// Valeur constante associée (si applicable, ex: eps pour LayerNorm).
    pub constant: Option<FusionConstant>,
    /// Nœud de sortie dans le graphe — None si c'est une source (Input).
    pub output: Option<usize>,
}

/// Valeur constante pour les opérations.
#[derive(Debug, Clone)]
pub enum FusionConstant {
    F32(f32),
    U32(u32),
    Bool(bool),
    Axes(Vec<u8>),
    Shape(Vec<usize>),
}

impl FusedOp {
    /// Crée une nouvelle opération sans inputs (source).
    pub fn new(kind: OpKind, constant: Option<FusionConstant>) -> Self {
        Self {
            kind,
            inputs: Vec::new(),
            constant,
            output: None,
        }
    }

    /// Crée une opération avec un input.
    pub fn with_input(kind: OpKind, input: usize, constant: Option<FusionConstant>) -> Self {
        Self {
            kind,
            inputs: vec![input],
            constant,
            output: None,
        }
    }

    /// Crée une opération avec deux inputs.
    pub fn with_inputs(
        kind: OpKind,
        input_a: usize,
        input_b: usize,
        constant: Option<FusionConstant>,
    ) -> Self {
        Self {
            kind,
            inputs: vec![input_a, input_b],
            constant,
            output: None,
        }
    }

    /// Vérifie si cette opération peut être fusionnée avec une autre.
    pub fn can_fuse_with(&self, other: &OpKind) -> bool {
        use OpKind::*;

        // Patterns de fusion autorisés
        matches!(
            (self.kind, other),
            // MatMul peut être fusionné avec activation
            (MatMul | Linear, ReLU | SiLU | Gelu | GeluApprox | Sigmoid | Tanh)
            // Activation peut suivre MatMul
            | (ReLU | SiLU | Gelu | GeluApprox | Sigmoid | Tanh, MatMul | Linear)
            // MatMul peut enchaîner avec un autre MatMul (MLP)
            | (MatMul | Linear, MatMul | Linear)
            // LayerNorm peut fusionner avec activation
            | (LayerNorm | LayerNormFused, ReLU | SiLU | Gelu | GeluApprox | Sigmoid | Tanh)
            // Activation peut précéder LayerNorm
            | (ReLU | SiLU | Gelu | GeluApprox | Sigmoid | Tanh, LayerNorm | LayerNormFused)
            // RMSNorm peut fusionner avec activation
            | (RMSNorm, ReLU | SiLU | Gelu | GeluApprox | Sigmoid | Tanh)
            // SSM step peut se chaîner avec lui-même
            | (SsmStep, SsmStep)
            // Element-wise entre résultats de même shape
            | (Add | Sub | Mul | Div, Add | Sub | Mul | Div)
            // Element-wise entre MatMul et scalaire
            | (MatMul | Linear, Scale)
            | (Scale, MatMul | Linear)
        )
    }
}

/// Graphe de dépendance pour la fusion.
///
/// Structure de données simple: liste de nœuds avec les indices de dépendance.
pub struct OpGraph {
    /// Liste des opérations.
    ops: Vec<FusedOp>,
    /// Topological sort order (mise à jour lors de la construction).
    pub topo_order: Vec<usize>,
}

impl OpGraph {
    /// Crée un graphe vide.
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            topo_order: Vec::new(),
        }
    }

    /// Ajoute une opération source (Input).
    pub fn add_input(&mut self, kind: OpKind, constant: Option<FusionConstant>) -> usize {
        let idx = self.ops.len();
        self.ops.push(FusedOp::new(kind, constant));
        idx
    }

    /// Ajoute une opération avec un input.
    pub fn add_unary(
        &mut self,
        kind: OpKind,
        input: usize,
        constant: Option<FusionConstant>,
    ) -> usize {
        let idx = self.ops.len();
        self.ops.push(FusedOp::with_input(kind, input, constant));
        self.ops[idx].output = Some(idx);
        idx
    }

    /// Ajoute une opération avec deux inputs.
    pub fn add_binary(
        &mut self,
        kind: OpKind,
        input_a: usize,
        input_b: usize,
        constant: Option<FusionConstant>,
    ) -> usize {
        let idx = self.ops.len();
        self.ops
            .push(FusedOp::with_inputs(kind, input_a, input_b, constant));
        self.ops[idx].output = Some(idx);
        idx
    }

    /// Ajoute une opération avec plusieurs inputs.
    pub fn add_nary(
        &mut self,
        kind: OpKind,
        inputs: Vec<usize>,
        constant: Option<FusionConstant>,
    ) -> usize {
        let idx = self.ops.len();
        self.ops.push(FusedOp {
            kind,
            inputs,
            constant,
            output: Some(idx),
        });
        idx
    }

    /// Exécute le topological sort pour déterminer l'ordre d'exécution.
    pub fn compute_topo_order(&mut self) {
        let n = self.ops.len();
        let mut in_degree = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![vec![]; n];

        // Construire le graphe d'adjacence
        for (i, op) in self.ops.iter().enumerate()
        {
            for &input in &op.inputs
            {
                adj[input].push(i);
                in_degree[i] += 1;
            }
        }

        // Kahn's algorithm
        let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        self.topo_order.clear();

        while let Some(node) = queue.pop()
        {
            self.topo_order.push(node);
            for &next in &adj[node]
            {
                in_degree[next] -= 1;
                if in_degree[next] == 0
                {
                    queue.push(next);
                }
            }
        }

        if self.topo_order.len() != n
        {
            panic!("OpGraph: cycle detected — not a DAG!");
        }
    }

    /// Retourne les opérations triées topologiquement.
    pub fn sorted_ops(&self) -> &[FusedOp] {
        // Mapper les indices topo_order vers les ops
        &self.ops
    }

    /// Retourne le nœud d'entrée donné par l'index.
    pub fn op(&self, idx: usize) -> &FusedOp {
        &self.ops[idx]
    }

    /// Retourne le nombre d'opérations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Vérifie si le graphe est vide.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Retourne les opérations dans un sous-graph (du premier au dernier).
    pub fn slice_ops(&self, start: usize, end: usize) -> &[FusedOp] {
        &self.ops[start..end]
    }

    /// Construit un sous-graphe à partir d'un chemin de sortie.
    /// Retourne les indices des nœuds nécessaires au calcul de `root`.
    pub fn get_subgraph_root(&self, root: usize) -> Vec<usize> {
        let mut visited = std::collections::HashSet::new();
        let mut result = Vec::new();
        self._collect_deps(root, &mut visited, &mut result);
        result.sort();
        result.dedup();
        result
    }

    fn _collect_deps(
        &self,
        idx: usize,
        visited: &mut std::collections::HashSet<usize>,
        result: &mut Vec<usize>,
    ) {
        if !visited.insert(idx)
        {
            return;
        }
        result.push(idx);
        for &input in &self.ops[idx].inputs
        {
            self._collect_deps(input, visited, result);
        }
    }
}

impl Default for OpGraph {
    fn default() -> Self {
        Self::new()
    }
}
