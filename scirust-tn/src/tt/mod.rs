//! Tensor-Train decomposition and TT-based neural network layers.

pub mod decompose;

#[cfg(feature = "core")]
pub mod linear;

#[cfg(feature = "core")]
pub use linear::{tt_decompose, tt_decompose_auto, TTLinear};

pub use decompose::{tt_decompose_matrix, tt_decompose_tensor, TTCores};
