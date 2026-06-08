//! # scirust-tn — Tensor Networks for SciRust
//!
//! Tensor-Train (TT) decomposition and Matrix Product State (MPS) primitives
//! that integrate with `scirust-core`'s tape-based autograd.
//!
//! ## Crate layers
//!
//! - [`tensor::TensorND`] — N-dimensional tensor (offline, no autograd).
//!   Used internally for the TT-SVD algorithm where successive unfoldings
//!   of different ranks are needed.
//! - [`ops::svd::truncated_svd`] — Truncated SVD via nalgebra (Phase 1 CPU).
//! - [`tt::decompose`] — Oseledets TT-SVD turning a dense matrix or tensor
//!   into a chain of low-rank cores.
//! - [`tt::TTLinear`] (feature `core`) — `Linear`-shaped layer whose weight
//!   matrix is stored as a TT-chain. Implements `scirust_core::nn::Module`.
//! - [`factorize::auto_factorize`] — Helper to split an integer `n` into
//!   `d` balanced factors.
//!
//! ## Phase 1 forward (this version)
//!
//! `TTLinear::forward` **reconstructs the dense weight matrix** from the cores
//! at each call, then performs the standard `x @ W + b` matmul through
//! `Var::matmul`. This preserves the **memory** savings of TT (the parameter
//! state is just the cores) but not the **compute** savings.
//!
//! Phase 2 will implement the native left-to-right TT contraction once a
//! tensor permutation op is available on `Var`.
//!
//! ## Quick start
//!
//! ```ignore
//! use scirust_tn::factorize::auto_factorize;
//! use scirust_tn::tt::{tt_decompose_auto, TTLinear};
//! use scirust_core::nn::Linear;
//!
//! let linear = Linear::new(768, 3072); // a transformer FFN projection
//! let tt = tt_decompose_auto(&linear, /*n_factors=*/ 3, /*max_rank=*/ 32, /*tol=*/ 1e-4);
//! println!("compression ratio: {:.2}x", tt.compression_ratio());
//! ```

pub mod factorize;
pub mod ops;
pub mod tensor;
pub mod tt;
pub mod discovered;
pub mod discovered_gemm;

// Re-export the most commonly used items at the crate root.
pub use factorize::auto_factorize;
pub use tensor::TensorND;
pub use tt::decompose::{tt_decompose_matrix, TTCores};

#[cfg(feature = "core")]
pub use tt::linear::{tt_decompose, tt_decompose_auto, TTLinear};
