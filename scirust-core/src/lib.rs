pub mod io;
pub mod nn;
pub use scirust_autodiff::*;
pub use scirust_macros::autodiff;
pub use scirust_simd::*;

pub mod matrix {
    pub mod backend;
    pub mod view;
}

pub mod autodiff;
// pub mod optim;
// pub mod scheduler;
// pub mod reverse;

// pub mod lazy;

pub mod data;
pub mod embed;
pub mod tensor;

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
pub mod compute_backend;
pub mod quantization;
pub mod homomorphic;
