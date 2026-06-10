//! Méthode de bissection : robuste, garantit la convergence dès qu'il y a un
//! changement de signe sur l'intervalle. Convergence linéaire (1 bit / itér).

use crate::{ConvergenceInfo, Solution, SolverError, SolverResult, Tolerance};

/// Trouve une racine de `f` dans `[a, b]` sachant que `f(a)·f(b) < 0`.
pub fn bisection<F: Fn(f64) -> f64>(
    f: F,
    mut a: f64,
    mut b: f64,
    tol: Tolerance,
) -> SolverResult<Solution<f64>> {
    if a > b
    {
        std::mem::swap(&mut a, &mut b);
    }
    let mut fa = f(a);
    let fb = f(b);
    if fa.signum() == fb.signum() && fa != 0.0 && fb != 0.0
    {
        return Err(SolverError::NoSignChange { a, b, fa, fb });
    }

    if fa == 0.0
    {
        return Ok(Solution::new(a, 0, 0.0));
    }
    if fb == 0.0
    {
        return Ok(Solution::new(b, 0, 0.0));
    }

    for k in 0..tol.max_iter
    {
        let mid = 0.5 * (a + b);
        let fm = f(mid);
        if fm.abs() < tol.abs || (b - a).abs() < tol.abs + tol.rel * mid.abs()
        {
            return Ok(Solution {
                value: mid,
                info: ConvergenceInfo {
                    iterations: k + 1,
                    residual: fm.abs(),
                    converged: true,
                },
            });
        }
        if fa.signum() == fm.signum()
        {
            a = mid;
            fa = fm;
        }
        else
        {
            b = mid;
            // pas besoin de tracker fb : on connaît son signe (opposé de fa)
        }
    }
    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: (b - a).abs(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn root_of_cos() {
        // racine de cos dans [0, pi] : x = pi/2
        let s = bisection(|x| x.cos(), 0.0, PI, Tolerance::default()).unwrap();
        assert_relative_eq!(s.value, PI / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn cubic_root() {
        // x^3 - 2x - 5 = 0 dans [2, 3] ; racine ~ 2.0945514815...
        let s = bisection(
            |x| x.powi(3) - 2.0 * x - 5.0,
            2.0,
            3.0,
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(s.value, 2.094_551_481_542_326_6, epsilon = 1e-9);
    }

    #[test]
    fn no_sign_change_detected() {
        // x^2 + 1 n'a pas de racine réelle
        assert!(bisection(|x| x * x + 1.0, -1.0, 1.0, Tolerance::default()).is_err());
    }
}
