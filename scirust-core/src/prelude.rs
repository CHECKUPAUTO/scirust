// Prelude for scirust-core — re-exports the most commonly used symbols.

pub use crate::Dual;
pub use crate::dispatch::gpu_or_cpu;
pub use crate::ops::{add_f32, add_f64, mul_f32, mul_f64};
pub use crate::symbolic::{
    Expr, NaturalCommand, Optimizer, PatternMemory, Pipeline, PipelineOutput, apply_trig_identity,
    derivative_1d, diff, discover_patterns, eval, gradient_2d, gradient_3d, linear_regression,
    parse, parse_natural, polynomial_fit, prove_equal, simplify, solve_linear, solve_quadratic,
    to_rust_code,
};
