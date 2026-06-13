//! # scirust-tn — Tensor Networks for SciRust
//!
//! Tensor-Train (TT) decomposition and Matrix Product State (MPS) primitives
//! that integrate with `scirust-core`'s tape-based autograd.
//!
//! ## Crate layers
//!
//! - [`tt::TTLinear`] — `Linear`-shaped layer whose weight
//!   matrix is stored as a TT-chain. Implements `scirust_core::nn::Module`.
//!
//! The TT decomposition primitives (truncated SVD, TT-SVD, auto-factorization)
//! live in `scirust_core::tn` and are re-exported here for convenience.
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
//! use scirust_tn::TTLinear;
//! use scirust_core::nn::init::Zeros;
//! use scirust_core::nn::Linear;
//! let mut rng = scirust_core::nn::init::PcgEngine::new(1);
//! let tt = TTLinear::new(768, 3072, &[4, 4], &[4, 4], 32, &Zeros, &mut rng);
//! println!("compression: {:.2}x", tt.compression_ratio());
//! ```

pub mod discovered;
pub mod discovered_gemm;
pub mod tt;

pub use scirust_core::tn::factorize::auto_factorize;
pub use scirust_core::tn::tt_decompose::{TTCores, reconstruct_matrix, tt_decompose_matrix};

pub use tt::{TTLinear, tt_decompose, tt_decompose_auto};
