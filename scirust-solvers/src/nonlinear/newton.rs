//! Newton-Raphson pour systèmes `F: R^n → R^n`.
//!
//! ## Sécurité numérique
//! - `check_finite` sur F et J à chaque itération
//! - Détection de stagnation : pas < 1e-16 → StepUnderflow
//! - Pas de `.unwrap()` — propagation d'erreur via `?`
//! - Newton avec backtracking si le résidu augmente (linesearch simple)
//!
//! Deux entrées :
//!   - `newton_system` : autodiff via Dual numbers
//!   - `newton_system_jac` : F et J fournies séparément

use crate::linalg::{self, Matrix};
use crate::{Solution, SolverError, SolverResult, Tolerance};
use scirust_autodiff::Dual;
use tracing::warn;

fn check_finite(value: f64, _location: &str) -> Result<(), SolverError> {
    if !value.is_finite()
    {
        return Err(SolverError::NanDetected { iter: 0, value });
    }
    Ok(())
}

/// Newton multivarié avec jacobienne automatique via dual numbers.
pub fn newton_system<F>(f: F, x0: Vec<f64>, tol: Tolerance) -> SolverResult<Solution<Vec<f64>>>
where
    F: Fn(&[Dual], &mut [Dual]),
{
    let n = x0.len();
    let mut x = x0;
    let mut buf_in = vec![Dual::primal(0.0); n];
    let mut buf_out = vec![Dual::primal(0.0); n];
    let mut fx = vec![0.0; n];
    let mut jac = Matrix::zeros(n, n);
    let mut last_res = f64::INFINITY;

    for k in 0..tol.max_iter
    {
        // Évalue F et J colonne par colonne via Dual
        for j in 0..n
        {
            for i in 0..n
            {
                buf_in[i] = Dual::new(x[i], if i == j { 1.0 } else { 0.0 });
            }
            f(&buf_in, &mut buf_out);
            for i in 0..n
            {
                let deriv = buf_out[i].deriv;
                check_finite(deriv, &format!("J[{i},{j}] Newton k={k}"))?;
                jac[(i, j)] = deriv;
                if j == 0
                {
                    let val = buf_out[i].value;
                    check_finite(val, &format!("fx[{i}] Newton k={k}"))?;
                    fx[i] = val;
                }
            }
        }

        let res = linalg::norm_inf(&fx);
        last_res = res;
        if res < tol.abs
        {
            return Ok(Solution::new(x, k, res));
        }

        // Résous J · δ = -F
        let rhs: Vec<f64> = fx.iter().map(|v| -v).collect();
        let delta = linalg::solve(jac.clone(), &rhs).map_err(|e| {
            warn!(target: "solver", "Newton: jacobian singular at iteration {k}: {e:?}");
            e
        })?;

        for (i, &d) in delta.iter().enumerate()
        {
            check_finite(d, &format!("delta[{i}] Newton k={k}"))?;
        }

        let step_norm = linalg::norm_inf(&delta);
        if step_norm < 1e-16
        {
            warn!(target: "solver", "Newton: step underflow {step_norm:.3e} at iteration {k}");
            return Err(SolverError::StepUnderflow { step: step_norm });
        }

        // Mise à jour avec backtracking linesearch simple
        // On essaie d'abord le pas complet, on réduit de moitié si le résidu augmente
        let mut lambda = 1.0;
        let mut x_candidate = vec![0.0; n];
        let mut fx_candidate = vec![0.0; n];
        let mut best_fx_norm = f64::INFINITY;
        let mut best_x = x.clone();
        let mut backtrack_success = false;

        for _ in 0..8
        {
            for i in 0..n
            {
                x_candidate[i] = x[i] + lambda * delta[i];
                check_finite(x_candidate[i], &format!("x_cand[{i}] lambda={lambda}"))?;
            }
            // Évaluer F au candidat via Dual
            for i in 0..n
            {
                buf_in[i] = Dual::primal(x_candidate[i]);
            }
            f(&buf_in, &mut buf_out);
            for i in 0..n
            {
                fx_candidate[i] = buf_out[i].value;
            }
            let cand_res = linalg::norm_inf(&fx_candidate);

            if cand_res < best_fx_norm
            {
                best_fx_norm = cand_res;
                best_x.copy_from_slice(&x_candidate);
                backtrack_success = true;
            }

            if cand_res < res
            {
                // Pas acceptable
                x.copy_from_slice(&x_candidate);
                fx.copy_from_slice(&fx_candidate);
                backtrack_success = true;
                break;
            }
            lambda *= 0.5;
        }

        if !backtrack_success
        {
            warn!(target: "solver", "Newton: backtracking failed at iteration {k}, using best candidate");
            x.copy_from_slice(&best_x);
            // Évaluer fx au best_x
            for i in 0..n
            {
                buf_in[i] = Dual::primal(best_x[i]);
            }
            f(&buf_in, &mut buf_out);
            for i in 0..n
            {
                fx[i] = buf_out[i].value;
            }
        }

        let final_norm = linalg::norm_inf(&fx);
        let final_step = linalg::norm_inf(&delta);

        if final_norm < tol.abs || final_step < tol.abs + tol.rel * linalg::norm_inf(&x)
        {
            // Dernière évaluation propre
            for i in 0..n
            {
                buf_in[i] = Dual::primal(x[i]);
            }
            f(&buf_in, &mut buf_out);
            let res_final = buf_out.iter().fold(0.0f64, |a, d| a.max(d.value.abs()));
            return Ok(Solution::new(x, k + 1, res_final));
        }
    }

    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: last_res,
    })
}

/// Variante où l'utilisateur fournit F et J séparément (pas d'autodiff).
pub fn newton_system_jac<F, J>(
    f: F,
    jac: J,
    x0: Vec<f64>,
    tol: Tolerance,
) -> SolverResult<Solution<Vec<f64>>>
where
    F: Fn(&[f64], &mut [f64]),
    J: Fn(&[f64], &mut Matrix),
{
    let n = x0.len();
    let mut x = x0;
    let mut fx = vec![0.0; n];
    let mut j_mat = Matrix::zeros(n, n);

    for k in 0..tol.max_iter
    {
        f(&x, &mut fx);
        for fi in &fx
        {
            check_finite(*fi, &format!("fx J Newton k={k}"))?;
        }

        let res = linalg::norm_inf(&fx);
        if res < tol.abs
        {
            return Ok(Solution::new(x, k, res));
        }

        jac(&x, &mut j_mat);
        // Vérifier que J est finie
        for i in 0..n
        {
            for j in 0..n
            {
                check_finite(j_mat[(i, j)], &format!("J[{i},{j}] Newton k={k}"))?;
            }
        }

        let rhs: Vec<f64> = fx.iter().map(|v| -v).collect();
        let delta = linalg::solve(j_mat.clone(), &rhs).map_err(|e| {
            warn!(target: "solver", "Newton(J): solve failed at iteration {k}: {e:?}");
            e
        })?;

        for (i, &d) in delta.iter().enumerate()
        {
            check_finite(d, &format!("delta[{i}] Newton(J) k={k}"))?;
        }

        let step_norm = linalg::norm_inf(&delta);
        if step_norm < 1e-16
        {
            warn!(target: "solver", "Newton(J): step underflow {step_norm:.3e} at iteration {k}");
            return Err(SolverError::StepUnderflow { step: step_norm });
        }

        for i in 0..n
        {
            x[i] += delta[i];
            check_finite(x[i], &format!("x[{i}] Newton(J) k={k}"))?;
        }

        if step_norm < tol.abs + tol.rel * linalg::norm_inf(&x)
        {
            f(&x, &mut fx);
            return Ok(Solution::new(x, k + 1, linalg::norm_inf(&fx)));
        }
    }

    f(&x, &mut fx);
    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: linalg::norm_inf(&fx),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn newton_2d_circle_diagonal() {
        let s = newton_system(
            |x, out| {
                out[0] = x[0] * x[0] + x[1] * x[1] - 1.0;
                out[1] = x[0] - x[1];
            },
            vec![1.0, 0.5],
            Tolerance::default(),
        )
        .unwrap();
        let expected = (0.5_f64).sqrt();
        assert_relative_eq!(s.value[0], expected, epsilon = 1e-10);
        assert_relative_eq!(s.value[1], expected, epsilon = 1e-10);
    }

    #[test]
    fn rosenbrock_root() {
        let s = newton_system(
            |x, out| {
                out[0] = (x[1] - x[0] * x[0]) * 10.0;
                out[1] = -x[0] + 1.0;
            },
            vec![-1.2, 1.0],
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(s.value[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(s.value[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn powell_badly_scaled() {
        let s = newton_system(
            |x, out| {
                out[0] = x[0] * x[1] * 1e4 - 1.0;
                out[1] = (-x[0]).exp() + (-x[1]).exp() - 1.0001;
            },
            vec![0.0, 1.0],
            Tolerance {
                abs: 1e-8,
                rel: 1e-6,
                max_iter: 200,
            },
        )
        .unwrap();
        let f0 = s.value[0] * s.value[1] * 1e4 - 1.0;
        let f1 = (-s.value[0]).exp() + (-s.value[1]).exp() - 1.0001;
        assert!(f0.abs() < 1e-6);
        assert!(f1.abs() < 1e-6);
    }
}
