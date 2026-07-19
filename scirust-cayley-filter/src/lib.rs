//! Experimental Cayley–Dickson filtering for SciRust.
//!
//! The scalar `f64` implementation is the mathematical reference.
//! Optimized kernels must demonstrate numerical parity with this oracle.

#![forbid(unsafe_code)]

pub mod scalar;

pub use scalar::{
    SEDENION_DIMENSION, Sedenion, basis_vector, conjugate, sedenion_mul, squared_norm,
};
