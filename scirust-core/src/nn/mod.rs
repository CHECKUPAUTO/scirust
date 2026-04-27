// scirust-core/src/nn/mod.rs
//
// Module nn — façade qui réexporte les sous-modules.

pub mod rng;
pub mod init;
pub mod loss;
pub mod module;

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

pub mod parallel;
