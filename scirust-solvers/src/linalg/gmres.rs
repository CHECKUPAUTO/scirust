//! GMRES(m) redémarré pour systèmes `A·x = b` non symétriques.
//!
//! Implémentation "matrix-free" (closure `matvec`), orthogonalisation de
//! Arnoldi par Gram-Schmidt modifié — séquentielle, donc déterministe même
//! en présence d'annulation numérique : pas de réduction parallèle ni
//! d'ordre dépendant du nombre de threads — et résolution du système des
//! moindres carrés de Hessenberg par rotations de Givens successives (norme
//! de résidu obtenue sans reformer `x` à chaque itération interne).
//!
//! Référence : Y. Saad & M.H. Schultz, « GMRES: A Generalized Minimal
//! Residual Algorithm for Solving Nonsymmetric Linear Systems », SIAM J.
//! Sci. Stat. Comput. 7(3), 1986 ; Saad, *Iterative Methods for Sparse
//! Linear Systems*, 2e éd. (SIAM, 2003), chap. 6.
//!
//! ## Déterminisme
//! Nombre max d'itérations et tolérance fixes (pas de critère d'arrêt basé
//! sur le temps écoulé) ; orthogonalisation séquentielle à ordre de colonnes
//! fixe ; rupture heureuse (« happy breakdown ») détectée par seuil fixe
//! plutôt que laissée diverger.

use crate::linalg::precond::identity_precond;
use crate::linalg::{axpy, dot, norm2};
use crate::{ConvergenceInfo, Solution, SolverError, SolverResult, Tolerance};
use tracing::warn;

const HAPPY_BREAKDOWN_EPS: f64 = 1e-13;

fn check_finite(value: f64, iter: usize) -> SolverResult<()> {
    if !value.is_finite()
    {
        return Err(SolverError::NanDetected { iter, value });
    }
    Ok(())
}

fn check_finite_slice(v: &[f64], iter: usize) -> SolverResult<()> {
    for &x in v
    {
        check_finite(x, iter)?;
    }
    Ok(())
}

/// GMRES(m) non préconditionné. `matvec(x, y)` calcule `y = A·x`.
///
/// `restart` est la taille du sous-espace de Krylov avant redémarrage (le
/// « m » de GMRES(m)) ; `tol.max_iter` borne le nombre total de produits
/// matrice-vecteur, tous cycles de redémarrage confondus.
pub fn gmres<F>(
    matvec: F,
    b: &[f64],
    x0: Vec<f64>,
    restart: usize,
    tol: Tolerance,
) -> SolverResult<Solution<Vec<f64>>>
where
    F: Fn(&[f64], &mut [f64]),
{
    gmres_preconditioned(matvec, identity_precond, b, x0, restart, tol)
}

/// GMRES(m) préconditionné à gauche : résout `M⁻¹A·x = M⁻¹b`.
///
/// `precond(r, z)` doit calculer `z = M⁻¹·r` (voir
/// [`crate::linalg::precond::JacobiPreconditioner`]). Passer
/// [`identity_precond`] pour le cas non préconditionné — c'est ce que fait
/// [`gmres`].
pub fn gmres_preconditioned<F, M>(
    matvec: F,
    precond: M,
    b: &[f64],
    mut x0: Vec<f64>,
    restart: usize,
    tol: Tolerance,
) -> SolverResult<Solution<Vec<f64>>>
where
    F: Fn(&[f64], &mut [f64]),
    M: Fn(&[f64], &mut [f64]),
{
    let n = b.len();
    if x0.len() != n
    {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: x0.len(),
        });
    }
    if restart == 0
    {
        return Err(SolverError::InvalidInput(
            "gmres: restart must be >= 1".to_string(),
        ));
    }
    check_finite_slice(b, 0)?;

    let mut b_prec = vec![0.0; n];
    precond(b, &mut b_prec);
    let b_prec_norm = norm2(&b_prec).max(1e-30);

    let mut total_iters = 0usize;
    let mut best_x = x0.clone();
    let mut best_res = f64::INFINITY;

    loop
    {
        let mut ax0 = vec![0.0; n];
        matvec(&x0, &mut ax0);
        let mut raw_r = vec![0.0; n];
        for i in 0..n
        {
            raw_r[i] = b[i] - ax0[i];
        }
        let mut r0 = vec![0.0; n];
        precond(&raw_r, &mut r0);
        let beta = norm2(&r0);
        check_finite(beta, total_iters)?;

        if beta < best_res
        {
            best_res = beta;
            best_x.copy_from_slice(&x0);
        }
        if beta <= tol.abs + tol.rel * b_prec_norm
        {
            return Ok(Solution {
                value: x0,
                info: ConvergenceInfo {
                    iterations: total_iters,
                    residual: beta,
                    converged: true,
                },
            });
        }
        if total_iters >= tol.max_iter
        {
            warn!(
                target: "solver",
                "GMRES did not converge after {} iterations (best residual: {:.3e})",
                tol.max_iter, best_res
            );
            x0.copy_from_slice(&best_x);
            return Err(SolverError::NoConvergence {
                iterations: tol.max_iter,
                residual: best_res,
            });
        }

        let m = restart.min(tol.max_iter - total_iters);
        let mut v: Vec<Vec<f64>> = Vec::with_capacity(m + 1);
        v.push(r0.iter().map(|x| x / beta).collect());
        let mut h: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut cs = vec![0.0; m];
        let mut sn = vec![0.0; m];
        let mut g = vec![0.0; m + 1];
        g[0] = beta;

        let mut k_done = 0;
        for j in 0..m
        {
            total_iters += 1;
            let mut av = vec![0.0; n];
            matvec(&v[j], &mut av);
            let mut w = vec![0.0; n];
            precond(&av, &mut w);
            check_finite_slice(&w, total_iters)?;

            // Arnoldi : Gram-Schmidt modifié, strictement séquentiel.
            let mut hj = vec![0.0; j + 2];
            for i in 0..=j
            {
                let hij = dot(&v[i], &w);
                check_finite(hij, total_iters)?;
                hj[i] = hij;
                axpy(-hij, &v[i], &mut w);
            }
            let hnorm = norm2(&w);
            check_finite(hnorm, total_iters)?;
            hj[j + 1] = hnorm;

            let breakdown = hnorm <= HAPPY_BREAKDOWN_EPS;
            if breakdown
            {
                v.push(vec![0.0; n]);
            }
            else
            {
                v.push(w.iter().map(|x| x / hnorm).collect());
            }

            // Applique les rotations de Givens précédentes à la nouvelle colonne.
            for i in 0..j
            {
                let temp = cs[i] * hj[i] + sn[i] * hj[i + 1];
                hj[i + 1] = -sn[i] * hj[i] + cs[i] * hj[i + 1];
                hj[i] = temp;
            }
            let denom = (hj[j] * hj[j] + hj[j + 1] * hj[j + 1]).sqrt();
            let (c, s) = if denom < HAPPY_BREAKDOWN_EPS
            {
                (1.0, 0.0)
            }
            else
            {
                (hj[j] / denom, hj[j + 1] / denom)
            };
            cs[j] = c;
            sn[j] = s;
            hj[j] = c * hj[j] + s * hj[j + 1];
            hj[j + 1] = 0.0;
            h.push(hj);

            g[j + 1] = -sn[j] * g[j];
            g[j] *= cs[j];

            k_done = j + 1;
            let res = g[j + 1].abs();
            if res <= tol.abs + tol.rel * b_prec_norm || breakdown
            {
                break;
            }
        }

        // Résout le système triangulaire supérieur k_done×k_done : R·y = g.
        let mut y = vec![0.0; k_done];
        for i in (0..k_done).rev()
        {
            let mut s = g[i];
            for jj in (i + 1)..k_done
            {
                s -= h[jj][i] * y[jj];
            }
            let diag = h[i][i];
            if diag.abs() < HAPPY_BREAKDOWN_EPS
            {
                return Err(SolverError::Singular {
                    row: i,
                    pivot: diag,
                });
            }
            y[i] = s / diag;
        }

        // x = x0 + V_k·y (V est déjà dans l'espace préconditionné à gauche —
        // correct puisqu'on résout M⁻¹A·x = M⁻¹b).
        for i in 0..k_done
        {
            axpy(y[i], &v[i], &mut x0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Matrix;
    use crate::linalg::precond::JacobiPreconditioner;
    use approx::assert_relative_eq;

    #[test]
    fn gmres_on_spd_matches_direct_solve() {
        let n = 6;
        let mat = Matrix::from_fn(n, n, |i, j| {
            if i == j
            {
                4.0
            }
            else if (i as isize - j as isize).abs() == 1
            {
                -1.0
            }
            else
            {
                0.0
            }
        });
        let b: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let sol = gmres(
            |x, y| y.copy_from_slice(&mat.matvec(x).unwrap()),
            &b,
            vec![0.0; n],
            n,
            Tolerance::default(),
        )
        .unwrap();
        let ax = mat.matvec(&sol.value).unwrap();
        for (axi, bi) in ax.iter().zip(&b)
        {
            assert_relative_eq!(*axi, *bi, epsilon = 1e-8);
        }
    }

    #[test]
    fn gmres_on_nonsymmetric_system() {
        // A non symétrique, bien conditionnée.
        let a = Matrix::from_row_major(3, 3, vec![4.0, 1.0, 0.0, 2.0, 5.0, 1.0, 0.0, 3.0, 6.0]);
        let x_true = vec![1.0, -2.0, 3.0];
        let b = a.matvec(&x_true).unwrap();
        let sol = gmres(
            |x, y| y.copy_from_slice(&a.matvec(x).unwrap()),
            &b,
            vec![0.0; 3],
            3,
            Tolerance::default(),
        )
        .unwrap();
        for (xi, ti) in sol.value.iter().zip(&x_true)
        {
            assert_relative_eq!(*xi, *ti, epsilon = 1e-7);
        }
    }

    #[test]
    fn gmres_preconditioned_converges_faster_or_equal() {
        let n = 8;
        let diag: Vec<f64> = (1..=n).map(|i| 10.0 * i as f64).collect();
        let mat = Matrix::from_fn(n, n, |i, j| {
            if i == j
            {
                diag[i]
            }
            else if (i as isize - j as isize).abs() == 1
            {
                1.0
            }
            else
            {
                0.0
            }
        });
        let x_true: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let b = mat.matvec(&x_true).unwrap();
        let jac = JacobiPreconditioner::new(&diag).unwrap();

        let unprec = gmres(
            |x, y| y.copy_from_slice(&mat.matvec(x).unwrap()),
            &b,
            vec![0.0; n],
            n,
            Tolerance::default(),
        )
        .unwrap();
        let prec = gmres_preconditioned(
            |x, y| y.copy_from_slice(&mat.matvec(x).unwrap()),
            jac.as_fn(),
            &b,
            vec![0.0; n],
            n,
            Tolerance::default(),
        )
        .unwrap();

        for (xi, ti) in prec.value.iter().zip(&x_true)
        {
            assert_relative_eq!(*xi, *ti, epsilon = 1e-6);
        }
        assert!(prec.info.iterations <= unprec.info.iterations);
    }

    #[test]
    fn gmres_rejects_dimension_mismatch() {
        let res = gmres(
            |_x, y| y.fill(0.0),
            &[1.0, 2.0],
            vec![0.0; 3],
            2,
            Tolerance::default(),
        );
        assert!(res.is_err());
    }
}
