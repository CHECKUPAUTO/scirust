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

pub mod activation;
pub mod batch_norm;
pub mod batch_norm_2d;
pub mod conv2d;
pub mod conv2d_transpose;
pub mod conv_utils;
pub mod dropout;
pub mod embedding;
pub mod init;
pub mod layer_norm;
pub mod linear;
pub mod lstm;
pub mod loss;
pub mod module;
pub mod pool;
pub mod positional_encoding;
pub mod residual;
pub mod rng;
pub mod rope;
pub mod sequential;
pub mod transformer;
pub mod tt_linear;

// Re-exports pour confort
pub use activation::{ReLU, Sigmoid};
pub use batch_norm::BatchNorm1d;
pub use batch_norm_2d::BatchNorm2d;
pub use conv_utils::{ConvConfig, Padding};
pub use conv2d::Conv2d;
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use init::{Initializer, KaimingNormal, SmallNormal, XavierUniform, Zeros};
pub use layer_norm::LayerNorm;
pub use linear::Linear;
pub use loss::{CrossEntropyLoss, Loss, MseLoss, NllLoss};
pub use module::Module;
pub use pool::MaxPool2d;
pub use positional_encoding::PositionalEncoding;
pub use residual::ResidualBlock;
pub use rng::PcgEngine;
pub use sequential::Sequential;
pub use transformer::{MultiHeadAttention, TransformerBlock, TransformerEncoder};
pub use tt_linear::{tt_decompose, tt_decompose_auto, TTLinear};
