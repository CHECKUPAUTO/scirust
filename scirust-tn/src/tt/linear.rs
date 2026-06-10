//! TT-Linear layer — re-exported from scirust-core.
//!
//! The canonical implementation lives in `scirust_core::nn::tt_linear`.
//! This module exists for backward compatibility and re-exports the
//! Phase 2 implementation (on-tape contraction with gradient flow).

pub use scirust_core::nn::tt_linear::{TTLinear, tt_decompose, tt_decompose_auto};
