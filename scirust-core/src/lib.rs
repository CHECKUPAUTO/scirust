// pub mod nn;
// // pub mod io;
pub use scirust_autodiff::*;
pub use scirust_macros::autodiff;
pub use scirust_simd::*;
pub use scirust_gpu::dispatch;

pub mod matrix {
    pub mod view;
    pub mod backend;
}

pub mod autodiff;
// pub mod optim;
// pub mod scheduler;
// pub mod reverse;

// pub mod lazy;

// pub mod data;
pub mod tensor;

pub mod error;

// Symbolic math facade (added for soullink-node integration)
pub mod symbolic;
pub mod prelude;

pub use symbolic::{
    Expr, parse, simplify, diff, eval, to_rust_code,
    solve_linear, solve_quadratic, Optimizer, apply_trig_identity, prove_equal,
    PatternMemory, polynomial_fit, linear_regression, discover_patterns,
    Pipeline, PipelineOutput, parse_natural, NaturalCommand,
    derivative_1d, gradient_2d, gradient_3d,
    ops, simd_add_one,
};
