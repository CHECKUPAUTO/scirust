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

/// Seuil de « pivot nul » (courbure `pᵀAp` stagnante) exprimé relativement à
/// `‖p‖²` — dimensionnellement une courbure par unité de norme au carré,
/// donc invariant d'échelle : un système `A·x=b` mis à l'échelle par un
/// facteur physique quelconque (unités SI micro, etc.) n'est plus déclaré
/// singulier à tort (cf. Golub & Van Loan §3.4.6, seuil relatif à la norme
/// plutôt qu'absolu).
const PIVOT_EPS_REL: f64 = 1e-14;

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
    for &bi in b.iter()
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

    // Convergence initiale (b = 0, ou x0 déjà solution) : GMRES et le CG creux
    // font déjà ce test avant la boucle de Krylov ; son absence ici faisait
    // tomber directement sur le test de pivot (p = r = 0 ⇒ pᵀAp = 0) et
    // retourner à tort SolverError::Singular sur un système parfaitement
    // régulier.
    if last_res <= tol.abs + tol.rel * b_norm
    {
        return Ok(Solution {
            value: x,
            info: ConvergenceInfo {
                iterations: 0,
                residual: last_res,
                converged: true,
            },
        });
    }

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

        // p ≠ 0 here (the initial-convergence gate above returns before p can
        // become the zero vector), so ‖p‖² is a safe, strictly positive scale
        // reference for the curvature test.
        let p_norm_sq = dot(&p, &p).max(1e-300);
        if (pap / p_norm_sq).abs() < PIVOT_EPS_REL
        {
            warn!(
                target: "solver",
                "Conjugate gradient stalled: p^T·A·p / ‖p‖² = {:.3e} at iteration {} — restoring backup",
                pap / p_norm_sq, k
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

    #[test]
    fn cg_converges_immediately_on_homogeneous_system() {
        // Regression test for a P1 audit finding: with b = 0 and x0 = 0,
        // r = p = 0 so pᵀAp = 0, which the old absolute PIVOT_EPS test
        // reported as SolverError::Singular on a perfectly regular matrix
        // (the identity). GMRES and the sparse CG both test convergence
        // before entering the Krylov loop; dense CG did not.
        let n = 4;
        let identity = Matrix::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.0 });
        let b = vec![0.0; n];
        let sol = conjugate_gradient(
            |x, y| y.copy_from_slice(&identity.matvec(x).unwrap()),
            &b,
            vec![0.0; n],
            Tolerance::default(),
        )
        .expect("a homogeneous system with a regular matrix must converge, not error");
        assert_eq!(sol.info.iterations, 0);
        for &xi in &sol.value
        {
            assert_eq!(xi, 0.0);
        }
    }

    #[test]
    fn cg_converges_immediately_when_x0_is_already_the_solution() {
        let n = 3;
        let mat = Matrix::from_fn(n, n, |i, j| if i == j { 2.0 } else { 0.0 });
        let x_true = vec![1.0, 2.0, 3.0];
        let b = mat.matvec(&x_true).unwrap();
        let sol = conjugate_gradient(
            |x, y| y.copy_from_slice(&mat.matvec(x).unwrap()),
            &b,
            x_true.clone(),
            Tolerance::default(),
        )
        .expect("starting exactly at the solution must converge, not error");
        assert_eq!(sol.info.iterations, 0);
        assert_relative_eq!(sol.value.as_slice(), x_true.as_slice(), epsilon = 1e-12);
    }

    #[test]
    fn cg_solves_a_tiny_scale_system_without_false_singularity() {
        // Regression test for a P1 audit finding: PIVOT_EPS was an absolute
        // 1e-30 compared directly against pᵀAp, a quantity that scales with
        // the physical scale of A and b — a regular system at a tiny scale
        // (e.g. capacitance in Farads, ~1e-12) was declared singular.
        let n = 4;
        let scale = 1e-9;
        let mat = Matrix::from_fn(n, n, |i, j| {
            if i == j
            {
                4.0 * scale
            }
            else if (i as isize - j as isize).abs() == 1
            {
                -scale
            }
            else
            {
                0.0
            }
        });
        let x_true = vec![1.0 * scale, 2.0 * scale, 3.0 * scale, 4.0 * scale];
        let b = mat.matvec(&x_true).unwrap();
        let sol = conjugate_gradient(
            |x, y| y.copy_from_slice(&mat.matvec(x).unwrap()),
            &b,
            vec![0.0; n],
            Tolerance::default(),
        )
        .expect("a regular system at a tiny physical scale must not be reported singular");
        assert_relative_eq!(
            sol.value.as_slice(),
            x_true.as_slice(),
            epsilon = 1e-6,
            max_relative = 1e-6
        );
    }
}

/// LAPACK-style property test: CG's residual, not just its reported
/// convergence flag, must be small on any SPD system — checked over many
/// randomly generated matrices rather than fixed point values.
#[cfg(test)]
mod proptests {
    use super::*;
    use crate::linalg::Matrix;
    use proptest::prelude::*;

    /// A = MᵀM + n·I is SPD for any M (see cholesky.rs's proptests for the
    /// same construction).
    fn spd_from(n: usize, raw: &[f64]) -> Matrix {
        let m = Matrix::from_row_major(n, n, raw.to_vec());
        let mut a = m.transpose().matmul(&m).unwrap();
        for i in 0..n
        {
            a[(i, i)] += n as f64;
        }
        a
    }

    proptest! {
        #[test]
        fn residual_is_small_on_random_spd_systems(
            raw in prop::collection::vec(-10.0f64..10.0, 16),
            b in prop::collection::vec(-10.0f64..10.0, 4),
        ) {
            let n = 4;
            let a = spd_from(n, &raw);
            let sol = conjugate_gradient(
                |x, y| y.copy_from_slice(&a.matvec(x).unwrap()),
                &b,
                vec![0.0; n],
                Tolerance::default(),
            )
            .expect("CG must converge on a well-conditioned SPD system");
            let ax = a.matvec(&sol.value).unwrap();
            let b_norm = norm2(&b).max(1e-300);
            let res = ax.iter().zip(&b).map(|(axi, bi)| (axi - bi).powi(2)).sum::<f64>().sqrt();
            prop_assert!(res / b_norm < 1e-6, "relative residual {} too large", res / b_norm);
        }
    }
}
