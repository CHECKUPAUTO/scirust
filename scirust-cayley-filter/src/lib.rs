//! Experimental Cayley–Dickson filtering for SciRust.
//!
//! The scalar `f64` implementation is the mathematical reference.
//! Optimized kernels must demonstrate numerical parity with this oracle.

#![forbid(unsafe_code)]

pub mod analysis;
pub mod baseline;
pub mod filter;
pub mod operator;
pub mod scalar;

pub use analysis::{AnalysisError, MatrixAnalysis, analyze_matrix, kernel_residual_norm};
pub use baseline::{IdentityFilter, NoiseDirectionProjector, ProjectionError};
pub use filter::{CayleyFilter, FilterEvaluation, FilterMetrics};
pub use operator::{
    LeftMultiplicationOperator, Matrix16, left_multiplication_matrix, matrix_vector_mul,
};
pub use scalar::{
    SEDENION_DIMENSION, Sedenion, basis_vector, conjugate, sedenion_mul, squared_norm,
};
