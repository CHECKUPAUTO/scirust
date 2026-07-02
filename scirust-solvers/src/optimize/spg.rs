//! Gradient projeté spectral (Spectral Projected Gradient, SPG) pour
//! l'optimisation sous contraintes de boîte `l ≤ x ≤ u`.
//!
//! Combine le pas de Barzilai-Borwein (approximation de second ordre sans
//! Hessienne) avec une recherche linéaire d'Armijo **non monotone** (fenêtre
//! de mémoire fixe sur les dernières valeurs de `f`) et une projection sur
//! la boîte. Plus léger qu'un L-BFGS-B complet mais couvre le même besoin
//! (calibration de modèle bornée, QP de boîte pour MPC) avec une preuve de
//! convergence globale sur ensembles convexes.
//!
//! Référence : E.G. Birgin, J.M. Martínez, M. Raydan, « Nonmonotone
//! Spectral Projected Gradient Methods on Convex Sets », SIAM J. Optim.
//! 10(4), 2000. Voir aussi Nocedal & Wright, *Numerical Optimization*,
//! 2e éd., chap. 17 pour le cadre gradient projeté.
//!
//! ## Déterminisme
//! Fenêtre de mémoire non monotone (`MEMORY`), facteur de rétrécissement du
//! backtracking (`0.5`) et nombre max de retours en arrière (`MAX_BACKTRACK`)
//! tous fixes — aucun critère temporel.

use crate::linalg::{dot, norm2};
use crate::{ConvergenceInfo, Solution, SolverError, SolverResult, Tolerance};

const MEMORY: usize = 10;
const GAMMA: f64 = 1e-4;
const MAX_BACKTRACK: usize = 30;
const ALPHA_MIN: f64 = 1e-10;
const ALPHA_MAX: f64 = 1e10;

fn project_box(x: &[f64], lower: &[f64], upper: &[f64]) -> Vec<f64> {
    x.iter()
        .zip(lower)
        .zip(upper)
        .map(|((&xi, &lo), &hi)| xi.clamp(lo, hi))
        .collect()
}

/// Minimise `f` sous `lower ≤ x ≤ upper` par gradient projeté spectral.
///
/// `grad` doit renvoyer le gradient exact ou une approximation cohérente de
/// `f`. `x0` est projeté sur la boîte avant la première itération.
pub fn spg<F, G>(
    f: F,
    grad: G,
    x0: Vec<f64>,
    lower: &[f64],
    upper: &[f64],
    tol: Tolerance,
) -> SolverResult<Solution<Vec<f64>>>
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if lower.len() != n || upper.len() != n
    {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: lower.len().min(upper.len()),
        });
    }
    for i in 0..n
    {
        if lower[i] > upper[i]
        {
            return Err(SolverError::InvalidInput(format!(
                "spg: lower[{i}] ({}) > upper[{i}] ({})",
                lower[i], upper[i]
            )));
        }
    }

    let mut x = project_box(&x0, lower, upper);
    let mut fx = f(&x);
    if !fx.is_finite()
    {
        return Err(SolverError::NanDetected { iter: 0, value: fx });
    }
    let mut g = grad(&x);
    let mut alpha = {
        let gn = norm2(&g);
        if gn > 1e-300
        {
            (1.0 / gn).clamp(ALPHA_MIN, ALPHA_MAX)
        }
        else
        {
            1.0
        }
    };
    let mut history = vec![fx];

    for k in 0..tol.max_iter
    {
        let mut trial = vec![0.0; n];
        for i in 0..n
        {
            trial[i] = x[i] - alpha * g[i];
        }
        let x_trial = project_box(&trial, lower, upper);
        let mut d = vec![0.0; n];
        for i in 0..n
        {
            d[i] = x_trial[i] - x[i];
        }
        let dnorm = norm2(&d);

        if dnorm <= tol.abs + tol.rel * norm2(&x).max(1.0)
        {
            return Ok(Solution {
                value: x,
                info: ConvergenceInfo {
                    iterations: k,
                    residual: dnorm,
                    converged: true,
                },
            });
        }

        let gd = dot(&g, &d);
        let f_max = history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut lambda = 1.0f64;
        let mut x_new = vec![0.0; n];
        let mut f_new = fx;
        let mut accepted = false;
        for _ in 0..MAX_BACKTRACK
        {
            for i in 0..n
            {
                x_new[i] = x[i] + lambda * d[i];
            }
            f_new = f(&x_new);
            if !f_new.is_finite()
            {
                return Err(SolverError::NanDetected {
                    iter: k,
                    value: f_new,
                });
            }
            if f_new <= f_max + GAMMA * lambda * gd
            {
                accepted = true;
                break;
            }
            lambda *= 0.5;
        }
        if !accepted
        {
            return Err(SolverError::StepUnderflow { step: lambda });
        }

        let g_new = grad(&x_new);
        let mut s = vec![0.0; n];
        let mut y = vec![0.0; n];
        for i in 0..n
        {
            s[i] = x_new[i] - x[i];
            y[i] = g_new[i] - g[i];
        }
        let sy = dot(&s, &y);
        alpha = if sy > 1e-300
        {
            (dot(&s, &s) / sy).clamp(ALPHA_MIN, ALPHA_MAX)
        }
        else
        {
            ALPHA_MAX
        };

        x = x_new;
        g = g_new;
        fx = f_new;
        history.push(fx);
        if history.len() > MEMORY
        {
            history.remove(0);
        }
    }

    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: fx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn spg_finds_unconstrained_minimum_inside_box() {
        // min (x-1)^2 + (y-2)^2 sur [-5,5]x[-5,5] : optimum non contraint (1,2).
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let grad = |x: &[f64]| vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)];
        let sol = spg(
            f,
            grad,
            vec![0.0, 0.0],
            &[-5.0, -5.0],
            &[5.0, 5.0],
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(sol.value[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(sol.value[1], 2.0, epsilon = 1e-5);
    }

    #[test]
    fn spg_clamps_to_active_bound() {
        // min (x-3)^2 + (y+2)^2 sur [0,1]x[-1,1] : optimum au coin (1,-1).
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2);
        let grad = |x: &[f64]| vec![2.0 * (x[0] - 3.0), 2.0 * (x[1] + 2.0)];
        let sol = spg(
            f,
            grad,
            vec![0.5, 0.0],
            &[0.0, -1.0],
            &[1.0, 1.0],
            Tolerance::default(),
        )
        .unwrap();
        assert_relative_eq!(sol.value[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(sol.value[1], -1.0, epsilon = 1e-5);
    }

    #[test]
    fn spg_projects_initial_point_into_box() {
        let f = |x: &[f64]| x[0] * x[0];
        let grad = |x: &[f64]| vec![2.0 * x[0]];
        // x0=10 est hors boîte [-1,1] ; doit converger vers 0 (minimum global, dans la boîte).
        let sol = spg(f, grad, vec![10.0], &[-1.0], &[1.0], Tolerance::default()).unwrap();
        assert_relative_eq!(sol.value[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn spg_rejects_inverted_bounds() {
        let f = |x: &[f64]| x[0] * x[0];
        let grad = |x: &[f64]| vec![2.0 * x[0]];
        let res = spg(f, grad, vec![0.0], &[1.0], &[-1.0], Tolerance::default());
        assert!(res.is_err());
    }
}
