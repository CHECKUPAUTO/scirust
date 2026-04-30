// scirust-core/src/nn/mod.rs
//
// Module nn — façade qui réexporte les sous-modules.

pub mod rng;
pub mod init;
pub mod loss;
pub mod module;
pub mod layer_norm;
pub mod transformer;

// Réexports les plus utilisés pour ergonomie
pub use rng::PcgEngine;
pub use init::{
    Initializer,
    Constant, Zeros, Ones,
    Uniform, Normal,
    XavierUniform, XavierNormal,
    KaimingNormal, KaimingUniform,
};
pub use loss::{Loss, MseLoss, BceLossApprox, HuberPseudoLoss};
pub use module::{
    Module,
    Linear, ReLU, Sigmoid, Dropout, Sequential,
};

pub mod conv2d;
pub mod batch_norm;
pub mod pool;
pub mod conv_utils;
pub mod parallel;
pub mod linear;
pub mod activation;
pub mod sequential;
