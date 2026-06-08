//! # SciRust Fusion — Moteur de fusion d'opérateurs
//!
//! Ce module implémente la fusion d'opérateurs au niveau du graphe de calcul
//! pour éliminer les writes/reads intermédiaires en RAM.
//!
//! ## Pipeline
//!
//! 1. **Graphe de dépendance** — reconstruction du graphe forward depuis la tape
//! 2. **Détection de motifs** — recherche des motifs de fusion canoniques
//! 3. **Génération de noyau** — compilation du graphe fusionné en un seul kernel
//!
//! ## Exemple
//!
//! ```
//! use scirust_fusion::FusionPipeline;
//!
//! let pipeline = FusionPipeline::new();
//!
//! // Définir un motif à fusionner
//! let mut graph = pipeline.build_graph();
//! graph.add_op(Op::MatMul, 0, 1);   // y = x @ W
//! graph.add_op(Op::SiLU, 2);         // y = silu(y)
//! graph.add_op(Op::LayerNorm, 3, 4); // y = layernorm(y, gamma, beta)
//!
//! // Fusionner
//! let fused = pipeline.fuse(&graph).unwrap();
//!
//! // Exécuter le noyau fusionné
//! fused.execute(&inputs, &mut output);
//! ```

mod fusion;
mod graph;
mod kernel;
mod patterns;

pub use fusion::FusionPipeline;
pub use graph::{FusedOp, OpGraph};
pub use kernel::FusedKernel;
pub use patterns::FusionPatterns;
