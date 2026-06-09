//! # FusionPipeline — Pipeline de détection et génération de kernels fusionnés
//!
//! Le pipeline prend un `OpGraph` (graphe de dépendance) et:
//! 1. Détecte les motifs de fusion canoniques
//! 2. Regroupe les nœuds compatibles
//! 3. Génère un `FusedKernel` pour chaque groupe

use crate::graph::{FusedOp, OpGraph, OpKind};
use crate::kernel::FusedKernel;
use crate::patterns::FusionPatterns;

/// Pipeline de fusion.
pub struct FusionPipeline {
    /// Motifs de fusion supportés.
    patterns: FusionPatterns,
}

impl FusionPipeline {
    /// Crée un nouveau pipeline avec les motifs par défaut.
    pub fn new() -> Self {
        Self {
            patterns: FusionPatterns::default(),
        }
    }

    /// Exécute la fusion sur un graphe.
    ///
    /// Retourne une liste de kernels fusionnés si des motifs sont trouvés,
    /// ou None si aucune fusion n'est possible.
    pub fn fuse(&self, graph: &mut OpGraph) -> Option<Vec<FusedKernel>> {
        graph.compute_topo_order();

        let mut fused_kernels = Vec::new();
        let mut visited = vec![false; graph.len()];

        // Parcourir le graphe topologiquement
        for &root in &graph.topo_order {
            if visited[root] {
                continue;
            }

            // Essayer de fusionner à partir de ce nœud
            if let Some(kernel) = self.try_fuse_from(graph, &mut visited, root) {
                fused_kernels.push(kernel);
            }
        }

        if fused_kernels.is_empty() {
            None
        } else {
            Some(fused_kernels)
        }
    }

    /// Essaie de fusionner un sous-graphe à partir d'un nœud racine.
    fn try_fuse_from(
        &self,
        graph: &OpGraph,
        visited: &mut [bool],
        root: usize,
    ) -> Option<FusedKernel> {
        // Collecter le sous-graphe nécessaire
        let mut group = Vec::new();
        let mut stack = vec![root];

        while let Some(node) = stack.pop() {
            if visited[node] {
                continue;
            }

            // Essayer de prolonger la fusion avec les voisins
            let op = graph.op(node);

            // Chercher un successeur qui peut être fusionné
            let fused = self.try_extend(&group, graph, node, visited);

            if fused {
                group.push(node);
                visited[node] = true;
                continue;
            }

            // Sinon, ajouter ce nœud comme kernel standalone
            if !group.is_empty() && group.len() >= 2 {
                let kernel = self.build_kernel(graph, &group);
                return Some(kernel);
            }

            // Check if this node can be the start of a fusion group
            // by looking at its successors
            for next in self.find_fusable_successors(graph, node) {
                if !visited[next] {
                    stack.push(next);
                }
            }
        }

        // Si le groupe a au moins 2 nœuds, construire le kernel
        if group.len() >= 2 {
            Some(self.build_kernel(graph, &group))
        } else {
            None
        }
    }

    /// Trouve les successeurs fusions d'un nœud.
    fn find_fusable_successors(&self, graph: &OpGraph, node: usize) -> Vec<usize> {
        let op = graph.op(node);

        // Trouver tous les nœuds qui ont `node` comme input
        graph.sorted_ops()
            .iter()
            .enumerate()
            .filter_map(|(idx, other_op)| {
                if other_op.inputs.contains(&node)
                    && op.can_fuse_with(&other_op.kind)
                    && self.patterns.is_pattern(op.kind, other_op.kind)
                {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Essaie d'étendre le groupe de fusion avec un nœud.
    fn try_extend(
        &self,
        group: &[usize],
        graph: &OpGraph,
        node: usize,
        _visited: &mut [bool],
    ) -> bool {
        if group.is_empty() {
            return false;
        }
        let group_op = graph.op(*group.last().unwrap());
        let node_op = graph.op(node);

        group_op.can_fuse_with(&node_op.kind)
            && self.patterns.is_pattern(group_op.kind, node_op.kind)
    }

    /// Construit un kernel fusionné à partir d'un groupe de nœuds.
    fn build_kernel(&self, graph: &OpGraph, group: &[usize]) -> FusedKernel {
        // Déterminer le type de kernel
        let kernel_type = self.determine_kernel_type(group, graph);

        // Collecter les inputs et outputs
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        for &idx in group {
            let op = graph.op(idx);
            for &input in &op.inputs {
                // Only add if not already in group
                if !group.contains(&input) && !inputs.contains(&input) {
                    inputs.push(input);
                }
            }
            outputs.push(idx);
        }

        FusedKernel::new(kernel_type, group.to_vec(), inputs, outputs)
    }

    /// Détermine le type de kernel à générer.
    fn determine_kernel_type(&self, group: &[usize], graph: &OpGraph) -> KernelType {
        let kinds: Vec<OpKind> = group.iter().map(|&i| graph.op(i).kind).collect();

        // In a pattern, `OpKind::MatMul` matches MatMul *or* Linear and
        // `OpKind::LayerNorm` matches LayerNorm *or* LayerNormFused — see
        // `matches_pattern` below.
        if matches_pattern(&kinds, &[OpKind::MatMul, OpKind::SiLU]) {
            return KernelType::MatmulSilu;
        }
        if matches_pattern(&kinds, &[OpKind::MatMul, OpKind::SiLU, OpKind::LayerNorm]) {
            return KernelType::MatmulSiluLayerNorm;
        }
        if matches_pattern(&kinds, &[OpKind::MatMul, OpKind::LayerNorm]) {
            return KernelType::MatmulLayerNorm;
        }
        if matches_pattern(&kinds, &[OpKind::MatMul, OpKind::ReLU]) {
            return KernelType::MatmulRelu;
        }
        if matches_pattern(&kinds, &[OpKind::MatMul, OpKind::MatMul, OpKind::Add]) {
            return KernelType::TwoLayerMlp;
        }
        if matches_pattern(&kinds, &[OpKind::SsmStep, OpKind::SsmStep]) {
            return KernelType::SsmScan;
        }
        // LayerNorm → activation (any of SiLU / Gelu / GELU_Approx / ReLU).
        if kinds.len() == 2
            && matches!(kinds[0], OpKind::LayerNorm | OpKind::LayerNormFused)
            && matches!(
                kinds[1],
                OpKind::SiLU | OpKind::Gelu | OpKind::GELU_Approx | OpKind::ReLU
            )
        {
            return KernelType::LayerNormActivation;
        }
        if matches_pattern(&kinds, &[OpKind::MatMul, OpKind::Scale]) {
            return KernelType::MatmulScale;
        }

        KernelType::Identity
    }
}

/// Vérifie si une liste de types d'op correspond à un motif.
fn matches_pattern(kinds: &[OpKind], pattern: &[OpKind]) -> bool {
    if kinds.len() != pattern.len() {
        return false;
    }

    // Utiliser l'appariement de motifs avec wildcards
    kinds.iter().zip(pattern.iter()).all(|(a, b)| match b {
        OpKind::MatMul | OpKind::Linear => matches!(a, OpKind::MatMul | OpKind::Linear),
        OpKind::SiLU => *a == OpKind::SiLU,
        OpKind::ReLU => *a == OpKind::ReLU,
        OpKind::Gelu | OpKind::GELU_Approx => matches!(a, OpKind::Gelu | OpKind::GELU_Approx),
        OpKind::Sigmoid => *a == OpKind::Sigmoid,
        OpKind::Tanh => *a == OpKind::Tanh,
        OpKind::LayerNorm | OpKind::LayerNormFused => {
            matches!(a, OpKind::LayerNorm | OpKind::LayerNormFused)
        }
        OpKind::RMSNorm => *a == OpKind::RMSNorm,
        OpKind::SsmStep => *a == OpKind::SsmStep,
        OpKind::Add => *a == OpKind::Add,
        OpKind::Scale => *a == OpKind::Scale,
        _ => *a == *b,
    })
}

/// Type de kernel fusionné.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelType {
    // Matrix + activation
    MatmulSilu,
    MatmulRelu,
    MatmulLayerNorm,
    MatmulSiluLayerNorm,
    MatmulScale,
    TwoLayerMlp,

    // Normalization + activation
    LayerNormActivation,

    // SSM
    SsmScan,

    // Pas de fusion possible
    Identity,
}

impl Default for FusionPipeline {
    fn default() -> Self {
        Self::new()
    }
}
