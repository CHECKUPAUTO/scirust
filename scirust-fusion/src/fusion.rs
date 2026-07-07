#![allow(non_camel_case_types)]
//! # FusionPipeline — Pipeline de détection et génération de kernels fusionnés
//!
//! Le pipeline prend un `OpGraph` (graphe de dépendance) et:
//! 1. Détecte les motifs de fusion canoniques
//! 2. Regroupe les nœuds compatibles
//! 3. Génère un `FusedKernel` pour chaque groupe

use crate::graph::{FusionConstant, OpGraph, OpKind};
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
        graph.compute_topo_order().ok()?;

        let mut fused_kernels = Vec::new();
        let mut visited = vec![false; graph.len()];

        // Parcourir le graphe topologiquement
        for &root in &graph.topo_order
        {
            if visited[root]
            {
                continue;
            }

            // Essayer de fusionner à partir de ce nœud
            if let Some(kernel) = self.try_fuse_from(graph, &mut visited, root)
            {
                fused_kernels.push(kernel);
            }
        }

        if fused_kernels.is_empty()
        {
            None
        }
        else
        {
            Some(fused_kernels)
        }
    }

    /// Build the maximal fusable chain starting at `root`: seed the group with
    /// `root`, then greedily extend by a canonical pattern (`MatMul → ReLU`,
    /// `Linear → SiLU`, …) as long as the last node has an unvisited, fusable
    /// successor. Returns a kernel only when at least two ops were fused.
    fn try_fuse_from(
        &self,
        graph: &OpGraph,
        visited: &mut [bool],
        root: usize,
    ) -> Option<FusedKernel> {
        if visited[root]
        {
            return None;
        }
        // Seed the chain with the root, then walk forward along fusable edges.
        let mut group = vec![root];
        visited[root] = true;

        loop
        {
            // `group` was seeded with `root` above and is only ever grown, so
            // it is never empty here; index directly rather than `unwrap()`.
            let last = group[group.len() - 1];
            let next = self
                .find_fusable_successors(graph, last)
                .into_iter()
                .find(|&n| !visited[n]);
            match next
            {
                Some(n) =>
                {
                    group.push(n);
                    visited[n] = true;
                },
                None => break,
            }
        }

        if group.len() >= 2
        {
            Some(self.build_kernel(graph, &group))
        }
        else
        {
            None
        }
    }

    /// Trouve les successeurs fusions d'un nœud.
    fn find_fusable_successors(&self, graph: &OpGraph, node: usize) -> Vec<usize> {
        let op = graph.op(node);

        // Trouver tous les nœuds qui ont `node` comme input
        graph
            .sorted_ops()
            .iter()
            .enumerate()
            .filter_map(|(idx, other_op)| {
                if other_op.inputs.contains(&node)
                    && op.can_fuse_with(&other_op.kind)
                    && self.patterns.is_pattern(op.kind, other_op.kind)
                {
                    Some(idx)
                }
                else
                {
                    None
                }
            })
            .collect()
    }

    /// Construit un kernel fusionné à partir d'un groupe de nœuds.
    fn build_kernel(&self, graph: &OpGraph, group: &[usize]) -> FusedKernel {
        // Déterminer le type de kernel
        let kernel_type = self.determine_kernel_type(group, graph);

        // Collecter les inputs et outputs
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        for &idx in group
        {
            let op = graph.op(idx);
            for &input in &op.inputs
            {
                // Only add if not already in group
                if !group.contains(&input) && !inputs.contains(&input)
                {
                    inputs.push(input);
                }
            }
            outputs.push(idx);
        }

        let mut kernel = FusedKernel::new(kernel_type, group.to_vec(), inputs, outputs);

        // `Scale` carries its factor as a scalar constant on the node, not as an
        // input tensor. Lift it into the kernel params so `execute` has it.
        if kernel_type == KernelType::MatmulScale
        {
            if let Some(&idx) = group.iter().find(|&&i| graph.op(i).kind == OpKind::Scale)
            {
                if let Some(FusionConstant::F32(s)) = &graph.op(idx).constant
                {
                    kernel.params.scale = *s;
                }
                // else: no scalar attached → keep the default (1.0, identity).
            }
        }

        kernel
    }

    /// Détermine le type de kernel à générer.
    fn determine_kernel_type(&self, group: &[usize], graph: &OpGraph) -> KernelType {
        let kinds: Vec<OpKind> = group.iter().map(|&i| graph.op(i).kind).collect();

        // In a pattern, `OpKind::MatMul` matches MatMul *or* Linear and
        // `OpKind::LayerNorm` matches LayerNorm *or* LayerNormFused — see
        // `matches_pattern` below.
        if matches_pattern(&kinds, &[OpKind::MatMul, OpKind::SiLU])
        {
            return KernelType::MatmulSilu;
        }
        if matches_pattern(&kinds, &[OpKind::MatMul, OpKind::SiLU, OpKind::LayerNorm])
        {
            return KernelType::MatmulSiluLayerNorm;
        }
        if matches_pattern(&kinds, &[OpKind::MatMul, OpKind::LayerNorm])
        {
            return KernelType::MatmulLayerNorm;
        }
        if matches_pattern(&kinds, &[OpKind::MatMul, OpKind::ReLU])
        {
            return KernelType::MatmulRelu;
        }
        if matches_pattern(&kinds, &[OpKind::MatMul, OpKind::MatMul, OpKind::Add])
        {
            return KernelType::TwoLayerMlp;
        }
        if matches_pattern(&kinds, &[OpKind::SsmStep, OpKind::SsmStep])
        {
            return KernelType::SsmScan;
        }
        // LayerNorm → activation (any of SiLU / Gelu / GeluApprox / ReLU).
        if kinds.len() == 2
            && matches!(kinds[0], OpKind::LayerNorm | OpKind::LayerNormFused)
            && matches!(
                kinds[1],
                OpKind::SiLU | OpKind::Gelu | OpKind::GeluApprox | OpKind::ReLU
            )
        {
            return KernelType::LayerNormActivation;
        }
        if matches_pattern(&kinds, &[OpKind::MatMul, OpKind::Scale])
        {
            return KernelType::MatmulScale;
        }

        KernelType::Identity
    }
}

/// Vérifie si une liste de types d'op correspond à un motif.
fn matches_pattern(kinds: &[OpKind], pattern: &[OpKind]) -> bool {
    if kinds.len() != pattern.len()
    {
        return false;
    }

    // Utiliser l'appariement de motifs avec wildcards
    kinds.iter().zip(pattern.iter()).all(|(a, b)| match b
    {
        OpKind::MatMul | OpKind::Linear => matches!(a, OpKind::MatMul | OpKind::Linear),
        OpKind::SiLU => *a == OpKind::SiLU,
        OpKind::ReLU => *a == OpKind::ReLU,
        OpKind::Gelu | OpKind::GeluApprox => matches!(a, OpKind::Gelu | OpKind::GeluApprox),
        OpKind::Sigmoid => *a == OpKind::Sigmoid,
        OpKind::Tanh => *a == OpKind::Tanh,
        OpKind::LayerNorm | OpKind::LayerNormFused =>
        {
            matches!(a, OpKind::LayerNorm | OpKind::LayerNormFused)
        },
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::OpGraph;
    use crate::kernel::FusedKernel;

    /// `y = act(x @ w)` — a two-node fusable group on top of two inputs.
    fn matmul_act_graph(act: OpKind) -> OpGraph {
        let mut g = OpGraph::new();
        let x = g.add_input(OpKind::Input, None);
        let w = g.add_input(OpKind::Input, None);
        let mm = g.add_binary(OpKind::MatMul, x, w, None);
        let _y = g.add_unary(act, mm, None);
        g
    }

    #[test]
    fn fuses_matmul_relu_into_one_kernel() {
        let pipeline = FusionPipeline::new();
        let mut g = matmul_act_graph(OpKind::ReLU);
        let kernels = pipeline.fuse(&mut g).expect("MatMul→ReLU should fuse");
        assert!(
            kernels
                .iter()
                .any(|k| k.kernel_type == KernelType::MatmulRelu),
            "no MatmulRelu kernel produced: {:?}",
            kernels.iter().map(|k| k.kernel_type).collect::<Vec<_>>()
        );
    }

    #[test]
    fn fuses_matmul_silu_into_one_kernel() {
        let pipeline = FusionPipeline::new();
        let mut g = matmul_act_graph(OpKind::SiLU);
        let kernels = pipeline.fuse(&mut g).expect("MatMul→SiLU should fuse");
        assert!(
            kernels
                .iter()
                .any(|k| k.kernel_type == KernelType::MatmulSilu)
        );
    }

    #[test]
    fn matmul_scale_fuses_and_executes_with_two_inputs() {
        use crate::graph::FusionConstant;

        // Build y = (x @ W) * 2.5, where 2.5 is a scalar constant on the Scale node.
        let mut g = OpGraph::new();
        let x = g.add_input(OpKind::Input, None);
        let w = g.add_input(OpKind::Input, None);
        let mm = g.add_binary(OpKind::MatMul, x, w, None);
        let _s = g.add_unary(OpKind::Scale, mm, Some(FusionConstant::F32(2.5)));

        let pipeline = FusionPipeline::new();
        let kernels = pipeline.fuse(&mut g).expect("MatMul→Scale should fuse");
        let k = kernels
            .iter()
            .find(|k| k.kernel_type() == KernelType::MatmulScale)
            .expect("a MatmulScale kernel");

        // The scale is a constant, so only x and W are runtime inputs.
        assert_eq!(k.inputs.len(), 2, "fused inputs must be exactly [x, W]");
        assert_eq!(k.params().scale, 2.5, "scale constant must be lifted");

        // Execute with two inputs — this used to panic on inputs[2]. The
        // pipeline leaves matmul dims unset (a separate demo limitation), so we
        // parameterize a kernel of the same type with the lifted scale.
        let mut k = FusedKernel::new(
            k.kernel_type(),
            k.group.clone(),
            k.inputs.clone(),
            k.outputs.clone(),
        );
        k.params = crate::kernel::KernelParams {
            in_features: 2,
            out_features: 2,
            scale: 2.5,
            ..Default::default()
        };
        let xd = [1.0f32, 2.0];
        let wd = [1.0f32, 0.0, 0.0, 1.0]; // identity
        let mut out = [0.0f32; 2];
        k.execute(&[&xd, &wd], &mut out);
        assert_eq!(out, [2.5, 5.0]);
    }

    #[test]
    fn nothing_fusable_returns_none() {
        let pipeline = FusionPipeline::new();
        // A lone activation on an input matches no canonical pattern.
        let mut g = OpGraph::new();
        let x = g.add_input(OpKind::Input, None);
        let _y = g.add_unary(OpKind::Tanh, x, None);
        assert!(pipeline.fuse(&mut g).is_none());
    }
}
