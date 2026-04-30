// scirust-core/src/nn/mod.rs
//
// Module NN minimal pour v11.1-critical.
//
// SCOPE :
//   - module     : trait Module
//   - rng        : générateur pseudo-aléatoire PCG
//   - init       : initialiseurs de poids (Kaiming, Xavier, Zeros, SmallNormal)
//   - linear     : couche Linear (matmul + bias)
//   - activation : ReLU, Sigmoid (wrappers Module)
//   - sequential : composeur de modules
//
// Les anciens modules (transformer, conv2d, batch_norm, layer_norm, pool,
// parallel, loss/, conv_utils) restent sur le disque mais ne sont PAS
// exposés ici. Ils utilisent des méthodes qui n'existent pas dans le Bloc 1
// reverse.rs et sont gardés pour référence/réintégration future après
// le quickstart fonctionnel (Bloc 5).

pub mod module;
pub mod rng;
pub mod init;
pub mod linear;
pub mod activation;
pub mod sequential;

// Re-exports pour confort
pub use module::Module;
pub use rng::PcgEngine;
pub use init::{Initializer, KaimingNormal, XavierUniform, Zeros, SmallNormal};
pub use linear::Linear;
pub use activation::{ReLU, Sigmoid};
pub use sequential::Sequential;
