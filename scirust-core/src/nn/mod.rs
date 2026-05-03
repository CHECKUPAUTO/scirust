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
//   - loss       : MseLoss, CrossEntropyLoss (avec max-trick stable)
//
// Les anciens modules (transformer, conv2d, batch_norm, layer_norm, pool,
// parallel) sont dans nn/.legacy/ et non exposés.

pub mod module;
pub mod rng;
pub mod init;
pub mod linear;
pub mod activation;
pub mod sequential;
pub mod loss;
pub mod dropout;
pub mod conv_utils;
pub mod conv2d;
pub mod pool;
pub mod batch_norm;
pub mod layer_norm;

// Re-exports pour confort
pub use module::Module;
pub use rng::PcgEngine;
pub use init::{Initializer, KaimingNormal, XavierUniform, Zeros, SmallNormal};
pub use linear::Linear;
pub use activation::{ReLU, Sigmoid};
pub use sequential::Sequential;
pub use loss::{Loss, MseLoss, CrossEntropyLoss, NllLoss};
pub use dropout::Dropout;
pub use conv_utils::{Padding, ConvConfig};
pub use conv2d::Conv2d;
pub use pool::MaxPool2d;
pub use batch_norm::BatchNorm1d;
pub use layer_norm::LayerNorm;
