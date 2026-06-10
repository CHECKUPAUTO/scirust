//! Méthode de Nelder-Mead (simplex downhill).
//!
//! ## Sécurité numérique
//! - Vérification que le simplex n'est pas aplati : si `diam < 1e-14`, on stoppe
//! - check_finite sur les points du simplex et les valeurs de f
//! - Détection de colinéarité : si centroid == worst, on réinitialise le simplex
//! - Plus de `.unwrap()` sur `partial_cmp`

use crate::{Solution, SolverError, SolverResult, Tolerance};
use tracing::warn;

fn check_finite(v: f64, _label: &str) -> Result<(), SolverError> {
    if !v.is_finite()
    {
        return Err(SolverError::NanDetected { iter: 0, value: v });
    }
    Ok(())
}

/// Calcule le centroïde (moyenne) d'un ensemble de points.
fn centroid(pts: &[Vec<f64>]) -> Vec<f64> {
    let n = pts[0].len();
    let m = pts.len() as f64;
    let mut c = vec![0.0; n];
    for p in pts
    {
        for d in 0..n
        {
            c[d] += p[d];
        }
    }
    for d in 0..n
    {
        c[d] /= m;
    }
    c
}

/// Diamètre maximal du simplex (plus grande distance entre deux points).
fn simplex_diameter(simplex: &[Vec<f64>]) -> f64 {
    let mut diam = 0.0;
    for i in 0..simplex.len()
    {
        for j in (i + 1)..simplex.len()
        {
            let dist: f64 = simplex[i]
                .iter()
                .zip(&simplex[j])
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            diam = f64::max(diam, dist);
        }
    }
    diam
}

/// Nelder-Mead. `f: R^n → R` sans dérivée.
pub fn nelder_mead<F>(
    f: F,
    x0: Vec<f64>,
    step: f64,
    tol: Tolerance,
) -> SolverResult<Solution<Vec<f64>>>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0
    {
        return Err(SolverError::InvalidInput("empty x0".into()));
    }

    // Vérifier x0
    for (i, &xi) in x0.iter().enumerate()
    {
        check_finite(xi, &format!("x0[{i}]"))?;
    }

    // Construit le simplex initial
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.clone());
    for j in 0..n
    {
        let mut v = x0.clone();
        let delta = if x0[j].abs() < 1e-12
        {
            step
        }
        else
        {
            step * x0[j].abs()
        };
        v[j] += delta;
        simplex.push(v);
    }

    let mut fvals: Vec<f64> = simplex
        .iter()
        .map(|v| {
            let val = f(v);
            let _ = check_finite(val, "f(simplex)");
            val
        })
        .collect();

    let alpha = 1.0;
    let gamma = 2.0;
    let rho = 0.5;
    let sigma = 0.5;

    let mut last_spread = f64::INFINITY;

    for k in 0..tol.max_iter
    {
        // Tri par valeur croissante
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| {
            fvals[a]
                .partial_cmp(&fvals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best = order[0];
        let worst = order[n];
        let second_worst = order[n - 1];

        let spread = fvals[worst] - fvals[best];
        last_spread = spread;
        let diam = simplex_diameter(&simplex);

        if spread < tol.abs && diam < tol.abs + tol.rel
        {
            return Ok(Solution::new(simplex.swap_remove(best), k, spread));
        }

        if diam < 1e-14
        {
            warn!(target: "solver", "Nelder-Mead: simplex collapsed (diam={diam:.3e}) at iteration {k}");
            return Ok(Solution::new(simplex[best].clone(), k, spread));
        }

        // Centroïde
        let mut centroid = vec![0.0; n];
        for &idx in &order[..n]
        {
            for d in 0..n
            {
                centroid[d] += simplex[idx][d];
            }
        }
        for d in 0..n
        {
            centroid[d] /= n as f64;
        }

        // Vérifier que centroid ≠ worst (simplex non dégénéré)
        let is_degenerate = centroid
            .iter()
            .zip(&simplex[worst])
            .all(|(c, w)| (c - w).abs() < 1e-14);
        if is_degenerate
        {
            warn!(target: "solver", "Nelder-Mead: degenerate simplex at iteration {k} — reinitializing");
            // Réinitialiser le simplex autour du meilleur point
            let best_pt = simplex[best].clone();
            for j in 0..n
            {
                let mut v = best_pt.clone();
                let delta = if best_pt[j].abs() < 1e-12
                {
                    step
                }
                else
                {
                    step * best_pt[j].abs()
                };
                v[j] += delta;
                simplex[j + 1] = v;
            }
            for i in 0..=n
            {
                fvals[i] = f(&simplex[i]);
            }
            continue;
        }

        // Réflexion
        let xr: Vec<f64> = (0..n)
            .map(|d| centroid[d] + alpha * (centroid[d] - simplex[worst][d]))
            .collect();
        for &xrd in &xr
        {
            check_finite(xrd, "xr")?;
        }
        let fr = f(&xr);
        check_finite(fr, "fr")?;

        if fr < fvals[best]
        {
            // Expansion
            let xe: Vec<f64> = (0..n)
                .map(|d| centroid[d] + gamma * (xr[d] - centroid[d]))
                .collect();
            for &xed in &xe
            {
                check_finite(xed, "xe")?;
            }
            let fe = f(&xe);
            check_finite(fe, "fe")?;
            if fe < fr
            {
                simplex[worst] = xe;
                fvals[worst] = fe;
            }
            else
            {
                simplex[worst] = xr;
                fvals[worst] = fr;
            }
        }
        else if fr < fvals[second_worst]
        {
            simplex[worst] = xr;
            fvals[worst] = fr;
        }
        else
        {
            // Contraction
            let xc: Vec<f64> = if fr < fvals[worst]
            {
                (0..n)
                    .map(|d| centroid[d] + rho * (xr[d] - centroid[d]))
                    .collect()
            }
            else
            {
                (0..n)
                    .map(|d| centroid[d] + rho * (simplex[worst][d] - centroid[d]))
                    .collect()
            };
            for &xcd in &xc
            {
                check_finite(xcd, "xc")?;
            }
            let fc = f(&xc);
            check_finite(fc, "fc")?;
            if fc < fvals[worst].min(fr)
            {
                simplex[worst] = xc;
                fvals[worst] = fc;
            }
            else
            {
                // Shrink
                let best_pt = simplex[best].clone();
                for i in 0..=n
                {
                    if i == best
                    {
                        continue;
                    }
                    for d in 0..n
                    {
                        simplex[i][d] = best_pt[d] + sigma * (simplex[i][d] - best_pt[d]);
                        check_finite(simplex[i][d], &format!("shrink[{i},{d}]"))?;
                    }
                    fvals[i] = f(&simplex[i]);
                }
            }
        }
    }

    // Meilleur point quand même via NoConvergence
    let mut order: Vec<usize> = (0..=n).collect();
    order.sort_by(|&a, &b| {
        fvals[a]
            .partial_cmp(&fvals[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: last_spread,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn nelder_quadratic() {
        let s = nelder_mead(
            |x| (x[0] - 3.0).powi(2) + (x[1] + 1.0).powi(2),
            vec![0.0, 0.0],
            1.0,
            Tolerance {
                abs: 1e-8,
                rel: 1e-8,
                max_iter: 500,
            },
        )
        .unwrap();
        assert_relative_eq!(s.value[0], 3.0, epsilon = 1e-3);
        assert_relative_eq!(s.value[1], -1.0, epsilon = 1e-3);
    }

    #[test]
    fn nelder_rosenbrock() {
        let s = nelder_mead(
            |x| {
                let a = 1.0 - x[0];
                let b = x[1] - x[0] * x[0];
                a * a + 100.0 * b * b
            },
            vec![-1.2, 1.0],
            0.5,
            Tolerance {
                abs: 1e-8,
                rel: 1e-8,
                max_iter: 5000,
            },
        )
        .unwrap();
        assert!((s.value[0] - 1.0).abs() < 1e-3);
        assert!((s.value[1] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn nelder_non_differentiable() {
        let s = nelder_mead(
            |x| (x[0] - 2.0).abs() + (x[0] + 3.0).abs(),
            vec![10.0],
            1.0,
            Tolerance {
                abs: 1e-6,
                rel: 1e-6,
                max_iter: 500,
            },
        )
        .unwrap();
        let fv = (s.value[0] - 2.0).abs() + (s.value[0] + 3.0).abs();
        assert!((fv - 5.0).abs() < 1e-3, "fv={fv}");
    }
}
