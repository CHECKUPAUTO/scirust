//! Méthode de Broyden ("bad" / Broyden 1, formule par-bon-pas).
//!
//! ## Sécurité numérique
//! - `check_finite` après chaque évaluation de F
//! - Vérification NaN sur le pas de correction `delta`
//! - Stagnation détectée : si `step_norm` < 1e-16, on stoppe
//! - `.unwrap()` sur `matvec` et `solve` remplacé par propagation `?`
//! - Réinitialisation de la jacobienne si singulière (DF)

use crate::linalg::{self, Matrix};
use crate::{ConvergenceInfo, Solution, SolverError, SolverResult, Tolerance};
use tracing::warn;

const JACOBIAN_H: f64 = 1e-7;

fn check_finite(value: f64, label: &str) -> Result<(), SolverError> {
    if !value.is_finite()
    {
        return Err(SolverError::NanDetected { iter: 0, value });
    }
    Ok(())
}

/// Calcule la jacobienne par différences finies au point `x`.
fn finite_diff_jacobian<F>(f: &F, x: &[f64], fx: &[f64]) -> Matrix
where
    F: Fn(&[f64], &mut [f64]),
{
    let n = x.len();
    let mut b = Matrix::zeros(n, n);
    let mut x_pert = x.to_vec();
    let mut fx_pert = vec![0.0; n];
    for j in 0..n
    {
        let xj_orig = x[j];
        let hj = JACOBIAN_H * xj_orig.abs().max(1.0);
        x_pert[j] = xj_orig + hj;
        f(&x_pert, &mut fx_pert);
        x_pert[j] = xj_orig;
        for i in 0..n
        {
            b[(i, j)] = (fx_pert[i] - fx[i]) / hj;
        }
    }
    b
}

pub fn broyden<F>(f: F, x0: Vec<f64>, tol: Tolerance) -> SolverResult<Solution<Vec<f64>>>
where
    F: Fn(&[f64], &mut [f64]),
{
    let n = x0.len();
    let mut x = x0;
    let mut fx = vec![0.0; n];
    f(&x, &mut fx);

    let res0 = linalg::norm_inf(&fx);
    if res0 < tol.abs
    {
        return Ok(Solution::new(x, 0, res0));
    }

    for fi in &fx
    {
        check_finite(*fi, "fx[0]")?;
    }

    // Jacobienne initiale par différences finies
    let mut b = finite_diff_jacobian(&f, &x, &fx);

    let mut last_res = res0;
    for k in 0..tol.max_iter
    {
        // Résous B · δ = -F
        let rhs: Vec<f64> = fx.iter().map(|v| -v).collect();
        let delta = match linalg::solve(b.clone(), &rhs)
        {
            Ok(d) => d,
            Err(_) =>
            {
                warn!(target: "solver", "Broyden: jacobian singular at iteration {k} — re-initializing via FD");
                b = finite_diff_jacobian(&f, &x, &fx);
                continue;
            },
        };

        // Vérifier que delta est fini
        for (i, &d) in delta.iter().enumerate()
        {
            check_finite(d, &format!("delta[{i}] Broyden k={k}"))?;
        }

        let step_norm = linalg::norm_inf(&delta);
        if step_norm < 1e-16
        {
            warn!(target: "solver", "Broyden: step underflow {step_norm:.3e} at iteration {k}");
            return Err(SolverError::StepUnderflow { step: step_norm });
        }

        // x_{k+1} = x_k + delta
        let mut x_new = x.clone();
        for i in 0..n
        {
            x_new[i] += delta[i];
            check_finite(x_new[i], &format!("x_new[{i}] Broyden k={k}"))?;
        }

        let mut fx_new = vec![0.0; n];
        f(&x_new, &mut fx_new);
        for fi in &fx_new
        {
            check_finite(*fi, &format!("fx_new Broyden k={k}"))?;
        }

        let res = linalg::norm_inf(&fx_new);
        last_res = res;

        if res < tol.abs || step_norm < tol.abs + tol.rel * linalg::norm_inf(&x_new)
        {
            return Ok(Solution::new(x_new, k + 1, res));
        }

        // Mise à jour de B par rang-1 (Broyden "good")
        let mut df = vec![0.0; n];
        for i in 0..n
        {
            df[i] = fx_new[i] - fx[i];
        }

        // B·δ avec propagation d'erreur (plus de .unwrap())
        let bdelta = match b.matvec(&delta)
        {
            Ok(v) => v,
            Err(e) =>
            {
                warn!(target: "solver", "Broyden: matvec failed at iteration {k}: {e} — re-initializing");
                b = finite_diff_jacobian(&f, &x, &fx);
                x = x_new;
                fx = fx_new;
                continue;
            },
        };

        let denom = linalg::dot(&delta, &delta);
        if denom > 1e-30
        {
            for i in 0..n
            {
                let coef = (df[i] - bdelta[i]) / denom;
                check_finite(coef, &format!("Broyden rank-1 coef[{i}] k={k}"))?;
                for j in 0..n
                {
                    b[(i, j)] += coef * delta[j];
                }
            }
        }

        x = x_new;
        fx = fx_new;
    }

    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: last_res,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn broyden_circle_diagonal() {
        let s = broyden(
            |x, out| {
                out[0] = x[0] * x[0] + x[1] * x[1] - 1.0;
                out[1] = x[0] - x[1];
            },
            vec![1.0, 0.5],
            Tolerance::default(),
        )
        .unwrap();
        let expected = (0.5_f64).sqrt();
        assert_relative_eq!(s.value[0], expected, epsilon = 1e-8);
        assert_relative_eq!(s.value[1], expected, epsilon = 1e-8);
    }

    #[test]
    fn broyden_brown_func() {
        let n = 3;
        let s = broyden(
            move |x, out| {
                let sum: f64 = x.iter().sum();
                for i in 0..(n - 1)
                {
                    out[i] = x[i] + sum - (n as f64 + 1.0);
                }
                out[n - 1] = x.iter().product::<f64>() - 1.0;
            },
            vec![0.5, 0.5, 0.5],
            Tolerance::default(),
        )
        .unwrap();
        for v in &s.value
        {
            assert_relative_eq!(*v, 1.0, epsilon = 1e-6);
        }
    }
}
