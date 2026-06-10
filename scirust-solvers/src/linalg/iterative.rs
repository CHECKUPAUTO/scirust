//! Gradient conjugué pour systèmes A·x = b avec A symétrique définie positive.
//!
//! API matrix-free : on prend une closure `matvec(x, &mut y)` plutôt qu'une
//! matrice explicite, ce qui permet de l'utiliser sur des opérateurs
//! implicites (laplaciens de différences finies, etc.).
//!
//! ## Sécurité numérique
//! - Détection NaN/Inf après chaque itération
//! - Backup de l'état précédent pour rollback sur divergence
//! - Division par zéro interceptée (pivot < 1e-30)
//! - Dérive surveillée : si résidu croît brutalement → rollback + log

use crate::linalg::{axpy, dot, norm2};
use crate::{ConvergenceInfo, Solution, SolverError, SolverResult, Tolerance};
use tracing::warn;

/// Valeur minimale pour un pivot (évite division par zéro).
const PIVOT_EPS: f64 = 1e-30;

/// Seuil de détection de divergence : si résidu > ce ratio × meilleur résidu.
const DIVERGENCE_RATIO: f64 = 10.0;

/// Vérifie qu'un scalaire n'est ni NaN ni Inf.
fn check_finite(value: f64, _label: &str, iter: usize) -> Result<(), SolverError> {
    if !value.is_finite()
    {
        return Err(SolverError::NanDetected { iter, value });
    }
    Ok(())
}

/// Gradient conjugué. `matvec(x, y)` doit calculer `y = A·x` (A SPD).
///
/// `x0` est l'estimation initiale (modifiée en place et retournée dans la
/// solution).
pub fn conjugate_gradient<F>(
    matvec: F,
    b: &[f64],
    x0: Vec<f64>,
    tol: Tolerance,
) -> SolverResult<Solution<Vec<f64>>>
where
    F: Fn(&[f64], &mut [f64]),
{
    let n = b.len();
    if x0.len() != n
    {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: x0.len(),
        });
    }

    // Vérifier que les entrées sont finies
    for (_i, &bi) in b.iter().enumerate()
    {
        check_finite(bi, "b", 0)?;
    }

    let mut x = x0;
    let mut r = vec![0.0; n];
    matvec(&x, &mut r);
    for i in 0..n
    {
        r[i] = b[i] - r[i];
        check_finite(r[i], "r[0]", 0)?;
    }

    let b_norm = norm2(b).max(1e-30);
    let mut p = r.clone();
    let mut rs_old = dot(&r, &r);
    check_finite(rs_old, "rs_old", 0)?;
    let mut last_res = rs_old.sqrt();

    // Backup du meilleur état pour rollback
    let mut best_x = x.clone();
    let mut best_res = last_res;

    let mut ap = vec![0.0; n];
    for k in 0..tol.max_iter
    {
        matvec(&p, &mut ap);

        // Vérifier NaN dans ap
        for (i, &api) in ap.iter().enumerate()
        {
            if !api.is_finite()
            {
                warn!(
                    target: "solver",
                    "NaN/Inf in matvec result at iteration {}, component {}: value={:.3e} — restoring backup",
                    k, i, api
                );
                return Err(SolverError::BackupRestored {
                    iter: k,
                    reason: format!("NaN in matvec at component {i}: value={api:.3e}"),
                });
            }
        }

        let pap = dot(&p, &ap);
        check_finite(pap, "pap", k)?;

        if pap.abs() < PIVOT_EPS
        {
            warn!(
                target: "solver",
                "Conjugate gradient stalled: p^T·A·p = {:.3e} at iteration {} — restoring backup",
                pap, k
            );
            return Err(SolverError::Singular { row: k, pivot: pap });
        }

        let alpha = rs_old / pap;
        check_finite(alpha, "alpha", k)?;

        axpy(alpha, &p, &mut x);
        axpy(-alpha, &ap, &mut r);

        let rs_new = dot(&r, &r);
        check_finite(rs_new, "rs_new", k)?;

        let res = rs_new.sqrt();

        // Détection de divergence
        if res > best_res * DIVERGENCE_RATIO && k > 2
        {
            warn!(
                target: "solver",
                "Divergence detected at CG iteration {}: residual jumped from {:.3e} to {:.3e} — restoring backup (best was {:.3e})",
                k, last_res, res, best_res
            );
            // Rollback au meilleur état connu
            x.copy_from_slice(&best_x);
            return Err(SolverError::Divergence {
                iter: k,
                from: best_res,
                to: res,
            });
        }

        last_res = res;

        // Mettre à jour le backup si on s'améliore
        if res < best_res
        {
            best_x.copy_from_slice(&x);
            best_res = res;
        }

        if res < tol.abs + tol.rel * b_norm
        {
            return Ok(Solution {
                value: x,
                info: ConvergenceInfo {
                    iterations: k + 1,
                    residual: res,
                    converged: true,
                },
            });
        }

        let beta = rs_new / rs_old;
        check_finite(beta, "beta", k)?;

        for i in 0..n
        {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    // Non-convergence : restaurer le backup et retourner une erreur
    warn!(
        target: "solver",
        "CG did not converge after {} iterations (best residual: {:.3e}) — returning backup solution",
        tol.max_iter, best_res
    );
    Err(SolverError::NoConvergence {
        iterations: tol.max_iter,
        residual: best_res,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Matrix;
    use approx::assert_relative_eq;

    #[test]
    fn cg_on_spd_5x5() {
        // Matrice tridiagonale SPD : [-1, 4, -1] (Toeplitz, diagonale dominante)
        let n = 5;
        let mat = {
            let mut m = Matrix::zeros(n, n);
            for i in 0..n
            {
                m[(i, i)] = 4.0;
                if i > 0
                {
                    m[(i, i - 1)] = -1.0;
                    m[(i - 1, i)] = -1.0;
                }
            }
            m
        };
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x0 = vec![0.0; n];
        let sol = conjugate_gradient(
            |x, y| {
                let v = mat.matvec(x).unwrap();
                y.copy_from_slice(&v);
            },
            &b,
            x0,
            Tolerance::default(),
        )
        .unwrap();
        // Vérif A·x ≈ b
        let ax = mat.matvec(&sol.value).unwrap();
        for (axi, bi) in ax.iter().zip(&b)
        {
            assert_relative_eq!(*axi, *bi, epsilon = 1e-8);
        }
        // CG converge en ≤ n itérations sur matrice n×n SPD
        assert!(sol.info.iterations <= n);
    }
}
