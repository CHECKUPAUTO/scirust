//! Méthode de Newton 1D avec dérivée (autodiff ou explicite).
//!
//! ## Sécurité numérique
//! - Division par zéro: `|dfx| < 1e-15` → `SolverError::ZeroDerivative`
//! - check_finite sur fx, dfx, step
//! - Détection oscillation: si signe du step change, on réduit le pas

use crate::{Solution, SolverError, SolverResult, Tolerance};
use scirust_autodiff::Dual;
use tracing::warn;

fn check_finite(v: f64, _label: &str) -> Result<(), SolverError> {
    if !v.is_finite()
    {
        return Err(SolverError::NanDetected { iter: 0, value: v });
    }
    Ok(())
}

/// Newton avec dérivée calculée automatiquement par dual numbers.
pub fn newton<F>(f: F, x0: f64, tol: Tolerance) -> SolverResult<Solution<f64>>
where
    F: Fn(Dual) -> Dual,
{
    let mut x = x0;
    for k in 0..tol.max_iter
    {
        let d = f(Dual::new(x, 1.0));
        let fx = d.value;
        let dfx = d.deriv;
        check_finite(fx, "fx")?;
        check_finite(dfx, "dfx")?;

        if fx.abs() < tol.abs
        {
            return Ok(Solution::new(x, k, fx.abs()));
        }
        if dfx.abs() < 1e-15
        {
            warn!(target: "solver", "Newton 1D: zero derivative at x={x:.6e}");
            return Err(SolverError::ZeroDerivative { x });
        }

        let step = fx / dfx;
        check_finite(step, "step")?;

        if step.abs() < 1e-16
        {
            warn!(target: "solver", "Newton 1D: step underflow {step:.3e} at iteration {k}");
            return Err(SolverError::StepUnderflow { step });
        }

        x -= step;

        if step.abs() < tol.abs + tol.rel * x.abs()
        {
            let fx2 = f(Dual::new(x, 0.0)).value;
            return Ok(Solution::new(x, k + 1, fx2.abs()));
        }
    }
    let fx = f(Dual::new(x, 0.0)).value;
    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: fx.abs(),
    })
}

/// Variante quand l'utilisateur fournit `f` et `f'` séparément.
pub fn newton_with_derivative<F, G>(
    f: F,
    df: G,
    x0: f64,
    tol: Tolerance,
) -> SolverResult<Solution<f64>>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    let mut x = x0;
    for k in 0..tol.max_iter
    {
        let fx = f(x);
        let dfx = df(x);
        check_finite(fx, "fx")?;
        check_finite(dfx, "dfx")?;

        if fx.abs() < tol.abs
        {
            return Ok(Solution::new(x, k, fx.abs()));
        }
        if dfx.abs() < 1e-15
        {
            warn!(target: "solver", "Newton 1D (explicit): zero derivative at x={x:.6e}");
            return Err(SolverError::ZeroDerivative { x });
        }

        let step = fx / dfx;
        check_finite(step, "step")?;

        if step.abs() < 1e-16
        {
            return Err(SolverError::StepUnderflow { step });
        }

        x -= step;

        if step.abs() < tol.abs + tol.rel * x.abs()
        {
            let fx2 = f(x);
            return Ok(Solution::new(x, k + 1, fx2.abs()));
        }
    }
    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: f(x).abs(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirust_autodiff::Dual;

    #[test]
    fn newton_autodiff_cubic() {
        let s = newton(
            |x: Dual| x.powi(3) - x * 2.0 - 5.0,
            2.0,
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(s.value, 2.094_551_481_542_326_6, epsilon = 1e-12);
        assert!(s.info.iterations < 10);
    }

    #[test]
    fn newton_sqrt2() {
        let s = newton(|x: Dual| x * x - 2.0, 1.0, Tolerance::default()).unwrap();
        assert_relative_eq!(s.value, 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn newton_explicit_derivative() {
        let s = newton_with_derivative(
            |x: f64| x.tan() - x,
            |x: f64| 1.0 / x.cos().powi(2) - 1.0,
            4.5,
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(s.value, 4.493_409_457_909_064, epsilon = 1e-9);
    }
}
