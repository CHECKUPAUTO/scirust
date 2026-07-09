//! Prelude for `scirust-core` — one import for the symbols the common paths
//! (the README quickstart, a training loop, a numeric kernel) actually need.
//!
//! ```
//! use scirust_core::prelude::*;
//!
//! // Build a tensor and an autodiff tape, no other imports required.
//! let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
//! assert_eq!(x.shape(), (2, 2));
//! ```

// --- Error handling: the Result alias every fallible public API returns. ---
pub use crate::error::{Result, SciRustError};

// --- Tensors & reverse-mode autodiff (the deep-learning core). ---
pub use crate::autodiff::optim::{Adam, Optimizer as NnOptimizer};
pub use crate::autodiff::reverse::{Tape, Tensor, Var};

// --- Neural-network building blocks used by the quickstart / training loops. ---
pub use crate::nn::init::{KaimingNormal, XavierUniform, Zeros};
pub use crate::nn::rng::PcgEngine;
pub use crate::nn::{CrossEntropyLoss, Linear, Loss, Module, MseLoss, ReLU, Sequential, Sigmoid};

// --- Numeric & symbolic conveniences (kept from the original prelude). ---
pub use crate::Dual;
pub use crate::dispatch::gpu_or_cpu;
pub use crate::ops::{add_f32, add_f64, mul_f32, mul_f64};
pub use crate::symbolic::{
    Expr, NaturalCommand, Optimizer, PatternMemory, Pipeline, PipelineOutput, apply_trig_identity,
    derivative_1d, diff, discover_patterns, eval, gradient_2d, gradient_3d, linear_regression,
    parse, parse_natural, polynomial_fit, prove_equal, simplify, solve_linear, solve_quadratic,
    to_rust_code,
};
