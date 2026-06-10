//! Tensor-Train decomposition and TT-based neural network layers.
//!
//! Decomposition primitives (TTCores, tt_decompose_matrix, etc.) are re-exported
//! from `scirust_core::tn` — see that module for the TT-SVD implementation.

pub mod linear;

pub use linear::{tt_decompose, tt_decompose_auto, TTLinear};
