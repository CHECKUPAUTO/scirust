#![cfg_attr(feature = "portable-simd", feature(portable_simd))]
//! # scirust-core
//!
//! Pure-Rust scientific-computing / deep-learning core: tensors, reverse-mode
//! autograd, a neural-network layer zoo, quantization, and assorted research
//! modules. Part of the larger `scirust-*` workspace (`publish = false`).
//!
//! ## Two tensor/autograd stacks (know which you're on)
//!
//! Historically the crate grew **two** parallel compute stacks that do **not**
//! interoperate — pick one per model:
//!
//! * **2-D tape** — [`autodiff::reverse`] (`Tensor`, `Tape`, `Var`): a matrix
//!   (`rows × cols`) reverse-mode engine. Most `nn::` layers build on it.
//! * **N-D tape** — [`autodiff::nd`] (`NdTape`, `NdVar`) over
//!   [`tensor::TensorND`]: the n-dimensional engine used by [`nn::nd_layers`],
//!   [`nn::nd_optim`], and the newer transformer/SSM layers.
//!
//! ## Backends & reproducibility
//!
//! [`matrix::backend::best_backend`] selects scalar / `portable-simd` / `blas`
//! at compile time. **Note:** enabling the `blas` feature changes the
//! floating-point accumulation order, so results are no longer bit-identical to
//! the scalar/SIMD paths — do not combine `blas` with the bit-exact
//! reproducibility guarantees of [`reproducible`] / [`portable_f32`].
//!
//! ## Security-sensitive modules — read their headers
//!
//! [`homomorphic`], [`dp`], [`nn::lipschitz`], [`nn::smoothing`],
//! [`nn::crown_ibp`], and [`nn::certified`] make cryptographic / privacy /
//! certified-robustness claims. Each module documents exactly what it does and
//! does **not** guarantee; none is production-hardened cryptography.

pub mod io;
pub mod nn;
// Local cache-aware SIMD tiling kernels. This module lives at
// `scirust-core/src/simd/` and is referenced as `crate::simd::tiling::matmul_tiled_f32`
// by `tensor/tiling.rs`; it must be declared here or the crate fails to build.
pub mod simd;
/// Autodiff **scalaire** (f64) de `scirust-autodiff` : nombres duaux
/// ([`scalar_ad::Dual`]) et tape scalaire de démonstration. Namespacé pour ne
/// plus entrer en collision avec la tape **tenseur** ([`autodiff::reverse`]) :
/// l'ancien glob `pub use scirust_autodiff::*;` publiait un `Tape`/`Var`
/// scalaires à la racine du crate, si bien que `scirust_core::Tape` et
/// `scirust_core::prelude::Tape` étaient **deux types différents**.
pub mod scalar_ad {
    pub use scirust_autodiff::*;
}
pub use scirust_macros::autodiff;
pub use scirust_simd::*;

pub mod matrix {
    pub mod backend;
    pub mod csr;
    pub mod soft;
    pub mod view;
}

pub mod autodiff;
pub mod optim;

pub mod data;
pub mod embed;
pub mod tensor;
pub mod tn;

#[cfg(test)]
mod tests;

pub mod error;

// Symbolic math facade (added for soullink-node integration)
pub mod prelude;
pub mod symbolic;

pub use symbolic::{
    Expr, NaturalCommand, Optimizer, PatternMemory, Pipeline, PipelineOutput, apply_trig_identity,
    derivative_1d, diff, discover_patterns, eval, gradient_2d, gradient_3d, linear_regression, ops,
    parse, parse_natural, polynomial_fit, prove_equal, simd_add_one, simplify, solve_linear,
    solve_quadratic, to_rust_code,
};

pub mod aot;
pub mod checkpoint;
pub mod compute_backend;
pub mod homomorphic;
pub mod lazy;
pub mod quantization;
pub mod quantum;
pub mod xai;

pub mod amp;
pub mod distributed;
pub mod dp;
pub mod exact_acc;
pub mod formal_proof;
pub mod logging;
pub mod lowprec;
pub mod philox;
pub mod portable_f32;
pub mod pruning;
pub mod reproducible;
pub mod tree_allreduce;
