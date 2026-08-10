#![feature(autodiff)]

use std::autodiff::autodiff_forward;

/// Rosenbrock function used as a small cross-oracle probe.
#[autodiff_forward(rosenbrock_forward_x, Dual, Const, Dual)]
pub fn rosenbrock(x: f64, y: f64) -> f64 {
    (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
}

/// Derivative with respect to `x` produced by rustc AutoDiff/Enzyme.
pub fn enzyme_dx(x: f64, y: f64) -> f64 {
    let (_value, derivative) = rosenbrock_forward_x(x, 1.0, y);
    derivative
}

/// The same partial derivative through SciRust's native forward-mode `Dual`.
pub fn scirust_dx(x: f64, y: f64) -> f64 {
    scirust_autodiff::gradient_2d(
        |x_dual, y_dual| {
            (scirust_autodiff::Dual::primal(1.0) - x_dual).powi(2)
                + scirust_autodiff::Dual::primal(100.0)
                    * (y_dual - x_dual * x_dual).powi(2)
        },
        x,
        y,
    )
    .0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enzyme_matches_scirust_on_rosenbrock_x_gradient() {
        for &(x, y) in &[(1.0, 1.0), (3.0, 1.0), (-1.25, 0.75), (0.5, -0.25)]
        {
            let enzyme = enzyme_dx(x, y);
            let native = scirust_dx(x, y);
            let tolerance = 1e-10 * (1.0 + native.abs());
            assert!(
                (enzyme - native).abs() <= tolerance,
                "x={x}, y={y}: Enzyme={enzyme}, SciRust={native}"
            );
        }
    }

    #[test]
    fn known_rosenbrock_point_matches_analytic_derivative() {
        // df/dx = -2(1-x) - 400x(y-x²). At (3,1), this is 9604.
        assert!((enzyme_dx(3.0, 1.0) - 9604.0).abs() < 1e-10);
        assert!((scirust_dx(3.0, 1.0) - 9604.0).abs() < 1e-10);
    }
}
