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
//! use scirust_fusion::{FusionPipeline, OpGraph, OpKind};
//!
//! // Build a small forward graph: y = ReLU(x @ W)
//! let mut graph = OpGraph::new();
//! let x = graph.add_input(OpKind::Input, None);
//! let w = graph.add_input(OpKind::Input, None);
//! let mm = graph.add_binary(OpKind::MatMul, x, w, None);
//! let _y = graph.add_unary(OpKind::ReLU, mm, None);
//!
//! // Detect and fuse canonical patterns (e.g. MatMul → ReLU).
//! let pipeline = FusionPipeline::new();
//! let _fused = pipeline.fuse(&mut graph);
//! ```

mod fusion;
mod graph;
mod kernel;
mod patterns;

pub use fusion::{FusionPipeline, KernelType};
pub use graph::{FusedOp, OpGraph, OpKind};
pub use kernel::{FusedKernel, KernelParams};
pub use patterns::FusionPatterns;
