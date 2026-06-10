//! Méthode de la sécante : équivalent de Newton mais sans dérivée — utilise
//! deux points pour approximer la pente. Ordre de convergence ≈ 1.618.
//!
//! ## Sécurité numérique
//! - Division par zéro interceptée via `|denom| < 1e-30`
//! - Détection de stagnation : `|x2 - x1| < 1e-16` → StepUnderflow
//! - check_finite sur f(x0), f(x1), x2

use crate::{Solution, SolverError, SolverResult, Tolerance};
use tracing::warn;

fn check_finite(v: f64, _label: &str) -> Result<(), SolverError> {
    if !v.is_finite()
    {
        return Err(SolverError::NanDetected { iter: 0, value: v });
    }
    Ok(())
}

/// Cherche une racine en partant de deux estimations `x0, x1`.
pub fn secant<F: Fn(f64) -> f64>(
    f: F,
    mut x0: f64,
    mut x1: f64,
    tol: Tolerance,
) -> SolverResult<Solution<f64>> {
    let mut f0 = f(x0);
    let mut f1 = f(x1);
    check_finite(f0, "f(x0)")?;
    check_finite(f1, "f(x1)")?;

    for k in 0..tol.max_iter
    {
        if f1.abs() < tol.abs
        {
            return Ok(Solution::new(x1, k, f1.abs()));
        }

        let denom = f1 - f0;
        if denom.abs() < 1e-30
        {
            warn!(
                target: "solver",
                "Secant: denominator {:.3e} near-zero at iteration {} — aborting",
                denom, k
            );
            return Err(SolverError::StepUnderflow { step: denom });
        }

        let x2 = x1 - f1 * (x1 - x0) / denom;
        check_finite(x2, "x2")?;

        let step = (x2 - x1).abs();

        if step < 1e-16
        {
            warn!(
                target: "solver",
                "Secant: step underflow {:.3e} at iteration {}",
                step, k
            );
            return Err(SolverError::StepUnderflow { step });
        }

        if step < tol.abs + tol.rel * x2.abs()
        {
            let f2 = f(x2);
            return Ok(Solution::new(x2, k + 1, f2.abs()));
        }

        x0 = x1;
        f0 = f1;
        x1 = x2;
        f1 = f(x1);
        check_finite(f1, "f(x1)")?;
    }

    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: f1.abs(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn secant_cubic() {
        let s = secant(
            |x| x.powi(3) - 2.0 * x - 5.0,
            2.0,
            3.0,
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(s.value, 2.094_551_481_542_326_6, epsilon = 1e-10);
    }

    #[test]
    fn secant_transcendental() {
        let s = secant(|x| x.exp() - 3.0 * x, 0.0, 1.0, Tolerance::default()).unwrap();
        assert!((s.value - 0.6190612867).abs() < 1e-6);
    }
}
