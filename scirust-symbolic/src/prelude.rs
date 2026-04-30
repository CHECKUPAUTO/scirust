// Prelude for scirust-core — re-exports the most commonly used symbols.

pub use crate::symbolic::{
    Expr, parse, simplify, diff, eval, to_rust_code,
    solve_linear, solve_quadratic, Optimizer, apply_trig_identity, prove_equal,
    PatternMemory, polynomial_fit, linear_regression, discover_patterns,
    Pipeline, PipelineOutput, parse_natural, NaturalCommand,
    derivative_1d, gradient_2d, gradient_3d,
};
pub use crate::Dual;
pub use crate::ops::{add_f32, mul_f32, add_f64, mul_f64};
pub use crate::dispatch::gpu_or_cpu;
