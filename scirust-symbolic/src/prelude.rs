//! Prelude — re-exports the most commonly used symbols of the symbolic
//! engine so call sites can `use scirust_symbolic::prelude::*;`.

pub use crate::{
    Dual, Expr, NaturalCommand, Optimizer, PatternMemory, Pipeline, PipelineOutput,
    apply_trig_identity, derivative_1d, diff, discover_patterns, eval, gradient_2d, gradient_3d,
    linear_regression, parse, parse_natural, polynomial_fit, prove_equal, simplify, solve_linear,
    solve_quadratic, to_rust_code,
};
