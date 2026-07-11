#![cfg_attr(feature = "portable-simd", feature(portable_simd))]

pub mod io;
pub mod nn;
// Local cache-aware SIMD tiling kernels. This module lives at
// `scirust-core/src/simd/` and is referenced as `crate::simd::tiling::matmul_tiled_f32`
// by `tensor/tiling.rs`; it must be declared here or the crate fails to build.
pub mod simd;
pub use scirust_autodiff::*;
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

pub mod dispatch {
    /// GPU or CPU fallback — dispatches work sequentially (rayon optional).
    /// When rayon is available, use par_chunks_mut for parallel execution.
    pub fn gpu_or_cpu<F>(data: &mut [f32], kernel: F)
    where
        F: Fn(&mut [f32]),
    {
        kernel(data);
    }
}
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
