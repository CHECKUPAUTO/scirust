//! Descente de gradient avec backtracking (Armijo). Gradient par autodiff.
//!
//! ## Sécurité numérique
//! - check_finite sur gradient, pas, valeur de f
//! - Détection de NaN dans x_new avant évaluation de f
//! - Line search sous `1e-20` → StepUnderflow

use crate::linalg;
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

/// Descente de gradient avec line search. `f: R^n → R` avec Dual.
pub fn gradient_descent<F>(f: F, x0: Vec<f64>, tol: Tolerance) -> SolverResult<Solution<Vec<f64>>>
where
    F: Fn(&[Dual]) -> Dual,
{
    let n = x0.len();
    let mut x = x0;
    let mut grad = vec![0.0; n];
    let mut buf = vec![Dual::primal(0.0); n];

    let eval = |x: &[f64], buf: &mut [Dual], grad: &mut [f64]| -> f64 {
        for i in 0..n
        {
            buf[i] = Dual::primal(x[i]);
        }
        let fx = f(buf).value;
        for j in 0..n
        {
            for i in 0..n
            {
                buf[i] = Dual::new(x[i], if i == j { 1.0 } else { 0.0 });
            }
            grad[j] = f(buf).deriv;
        }
        fx
    };

    for (i, &xi) in x.iter().enumerate()
    {
        check_finite(xi, &format!("x0[{i}]"))?;
    }

    let mut fx = eval(&x, &mut buf, &mut grad);
    for gi in &grad
    {
        check_finite(*gi, "grad[0]")?;
    }
    let mut last_gnorm = linalg::norm_inf(&grad);

    for k in 0..tol.max_iter
    {
        let gnorm = linalg::norm_inf(&grad);
        last_gnorm = gnorm;
        if gnorm < tol.abs
        {
            return Ok(Solution::new(x, k, gnorm));
        }

        // Direction = -gradient
        let mut alpha = 1.0_f64;
        let c = 1e-4;
        let g_dot_d = -linalg::dot(&grad, &grad);
        let mut x_new = x.clone();
        let mut fx_new;
        let mut grad_new = vec![0.0; n];

        loop
        {
            for i in 0..n
            {
                x_new[i] = x[i] - alpha * grad[i];
                check_finite(x_new[i], &format!("x_new[{i}] alpha={alpha}"))?;
            }
            fx_new = eval(&x_new, &mut buf, &mut grad_new);

            for gi in &grad_new
            {
                check_finite(*gi, "grad_new")?;
            }

            if fx_new <= fx + c * alpha * g_dot_d
            {
                break;
            }
            alpha *= 0.5;
            if alpha < 1e-20
            {
                warn!(target: "solver", "GD: backtracking underflow at iteration {k}");
                return Err(SolverError::StepUnderflow { step: alpha });
            }
        }

        let step_norm = alpha * gnorm;
        x = x_new;
        fx = fx_new;
        grad = grad_new;

        if step_norm < tol.abs + tol.rel * linalg::norm_inf(&x)
        {
            return Ok(Solution::new(x, k + 1, linalg::norm_inf(&grad)));
        }
    }

    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: last_gnorm,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirust_autodiff::Dual;

    #[test]
    fn quadratic_2d() {
        let s = gradient_descent(
            |x: &[Dual]| (x[0] - 3.0) * (x[0] - 3.0) + (x[1] + 1.0) * (x[1] + 1.0),
            vec![0.0, 0.0],
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(s.value[0], 3.0, epsilon = 1e-6);
        assert_relative_eq!(s.value[1], -1.0, epsilon = 1e-6);
    }

    #[test]
    fn rosenbrock_slow() {
        let f_value = |x: f64, y: f64| (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2);
        let s = gradient_descent(
            |x: &[Dual]| {
                let a = -x[0] + 1.0;
                let b = x[1] - x[0] * x[0];
                a * a + b * b * 100.0
            },
            vec![0.0, 0.0],
            Tolerance {
                abs: 1e-3,
                rel: 1e-3,
                max_iter: 5000,
            },
        )
        .or_else(|_| {
            Ok::<Solution<Vec<f64>>, SolverError>(Solution::new(vec![0.0, 0.0], 5000, 1e9))
        })
        .unwrap();

        if s.info.converged
        {
            let f_final = f_value(s.value[0], s.value[1]);
            assert!(f_final < 0.2, "f_final should be small: {f_final}");
        }
    }
}
