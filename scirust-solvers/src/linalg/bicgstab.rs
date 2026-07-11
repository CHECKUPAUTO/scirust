//! BiCGSTAB (bi-conjugate gradient stabilisé) pour systèmes `A·x = b`
//! non symétriques.
//!
//! Coût par itération plus faible que GMRES(m) (deux produits
//! matrice-vecteur, pas de stockage de base de Krylov croissante), au prix
//! d'une convergence parfois irrégulière. Bon complément de
//! [`crate::linalg::gmres`] pour les grands systèmes creux où stocker la
//! base d'Arnoldi est coûteux.
//!
//! Référence : H.A. van der Vorst, « Bi-CGSTAB: A Fast and Smoothly
//! Converging Variant of Bi-CG for the Solution of Nonsymmetric Linear
//! Systems », SIAM J. Sci. Stat. Comput. 13(2), 1992.
//!
//! ## Déterminisme
//! Le résidu de référence (« shadow residual ») `r̂₀` est choisi égal au
//! résidu initial `r₀` — un choix fixe et reproductible, plutôt qu'un
//! vecteur aléatoire comme le font certaines implémentations pour éviter
//! une coïncidence de rho nul. Nombre max d'itérations et tolérance fixes.

use crate::linalg::precond::identity_precond;
use crate::linalg::{dot, norm2};
use crate::{ConvergenceInfo, Solution, SolverError, SolverResult, Tolerance};
use tracing::warn;

// Dimensionless breakdown threshold. `rho` and `r_hat_v` below are raw dot
// products of vectors that scale with the physical magnitude of `b` (e.g.
// ~1e-14 for a system in micro-units), so comparing them directly against a
// fixed absolute epsilon declared any small-scale-but-regular system
// "singular". Normalizing each by the product of the vector norms turns the
// test into a cosine-like, scale-invariant measure (Cauchy–Schwarz bounds it
// to [-1, 1]) — this is the normalization Barrett et al., "Templates for the
// Solution of Linear Systems" (1994), use for breakdown detection.
const BREAKDOWN_EPS: f64 = 1e-13;

fn check_finite(value: f64, iter: usize) -> SolverResult<()> {
    if !value.is_finite()
    {
        return Err(SolverError::NanDetected { iter, value });
    }
    Ok(())
}

/// BiCGSTAB non préconditionné. `matvec(x, y)` calcule `y = A·x`.
pub fn bicgstab<F>(
    matvec: F,
    b: &[f64],
    x0: Vec<f64>,
    tol: Tolerance,
) -> SolverResult<Solution<Vec<f64>>>
where
    F: Fn(&[f64], &mut [f64]),
{
    bicgstab_preconditioned(matvec, identity_precond, b, x0, tol)
}

/// BiCGSTAB préconditionné à gauche : résout `M⁻¹A·x = M⁻¹b`.
///
/// `precond(r, z)` doit calculer `z = M⁻¹·r`. Passer
/// [`identity_precond`] pour le cas non préconditionné — c'est ce que fait
/// [`bicgstab`].
pub fn bicgstab_preconditioned<F, M>(
    matvec: F,
    precond: M,
    b: &[f64],
    x0: Vec<f64>,
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
    for &bi in b
    {
        check_finite(bi, 0)?;
    }

    let mut x = x0;
    let mut ax = vec![0.0; n];
    matvec(&x, &mut ax);
    let mut r = vec![0.0; n];
    for i in 0..n
    {
        r[i] = b[i] - ax[i];
    }
    // Résidu de référence fixe (pas de tirage aléatoire — voir doc de module).
    let r_hat = r.clone();
    let r_hat_norm = norm2(&r_hat).max(1e-300);

    let b_norm = norm2(b).max(1e-30);
    let mut best_x = x.clone();
    let mut best_res = norm2(&r);

    // Convergence initiale (b = 0, ou x0 déjà solution) : GMRES fait déjà ce
    // test avant la boucle de Krylov. Sans lui, r_hat = r = 0 rend
    // rho_new = ⟨r_hat, r⟩ = 0 et le test de breakdown ci-dessous déclarait à
    // tort un système régulier « singulier ».
    if best_res <= tol.abs + tol.rel * b_norm
    {
        return Ok(Solution {
            value: x,
            info: ConvergenceInfo {
                iterations: 0,
                residual: best_res,
                converged: true,
            },
        });
    }

    let mut rho_old = 1.0f64;
    let mut alpha = 1.0f64;
    let mut omega = 1.0f64;
    let mut v = vec![0.0; n];
    let mut p = vec![0.0; n];

    for k in 0..tol.max_iter
    {
        let rho_new = dot(&r_hat, &r);
        check_finite(rho_new, k)?;
        let r_norm = norm2(&r).max(1e-300);
        if (rho_new / (r_hat_norm * r_norm)).abs() < BREAKDOWN_EPS
        {
            warn!(target: "solver", "BiCGSTAB breakdown: rho ≈ 0 at iteration {k}");
            x.copy_from_slice(&best_x);
            return Err(SolverError::Singular {
                row: k,
                pivot: rho_new,
            });
        }

        if k == 0
        {
            p.copy_from_slice(&r);
        }
        else
        {
            if omega.abs() < BREAKDOWN_EPS
            {
                warn!(target: "solver", "BiCGSTAB breakdown: omega ≈ 0 at iteration {k}");
                x.copy_from_slice(&best_x);
                return Err(SolverError::Singular {
                    row: k,
                    pivot: omega,
                });
            }
            let beta = (rho_new / rho_old) * (alpha / omega);
            check_finite(beta, k)?;
            for i in 0..n
            {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
        }

        let mut p_hat = vec![0.0; n];
        precond(&p, &mut p_hat);
        matvec(&p_hat, &mut v);

        let r_hat_v = dot(&r_hat, &v);
        check_finite(r_hat_v, k)?;
        let v_norm = norm2(&v).max(1e-300);
        if (r_hat_v / (r_hat_norm * v_norm)).abs() < BREAKDOWN_EPS
        {
            warn!(target: "solver", "BiCGSTAB breakdown: r_hat·v ≈ 0 at iteration {k}");
            x.copy_from_slice(&best_x);
            return Err(SolverError::Singular {
                row: k,
                pivot: r_hat_v,
            });
        }
        alpha = rho_new / r_hat_v;
        check_finite(alpha, k)?;

        let mut s = vec![0.0; n];
        for i in 0..n
        {
            s[i] = r[i] - alpha * v[i];
        }
        let s_norm = norm2(&s);
        if s_norm < best_res
        {
            best_res = s_norm;
            for i in 0..n
            {
                best_x[i] = x[i] + alpha * p_hat[i];
            }
        }
        if s_norm <= tol.abs + tol.rel * b_norm
        {
            for i in 0..n
            {
                x[i] += alpha * p_hat[i];
            }
            return Ok(Solution {
                value: x,
                info: ConvergenceInfo {
                    iterations: k + 1,
                    residual: s_norm,
                    converged: true,
                },
            });
        }

        let mut s_hat = vec![0.0; n];
        precond(&s, &mut s_hat);
        let mut t = vec![0.0; n];
        matvec(&s_hat, &mut t);

        let tt = dot(&t, &t);
        omega = if tt > 1e-300 { dot(&t, &s) / tt } else { 0.0 };
        check_finite(omega, k)?;

        for i in 0..n
        {
            x[i] += alpha * p_hat[i] + omega * s_hat[i];
            r[i] = s[i] - omega * t[i];
        }
        let res = norm2(&r);
        if res < best_res
        {
            best_res = res;
            best_x.copy_from_slice(&x);
        }
        if res <= tol.abs + tol.rel * b_norm
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
        rho_old = rho_new;
    }

    warn!(
        target: "solver",
        "BiCGSTAB did not converge after {} iterations (best residual: {:.3e})",
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
    use crate::linalg::precond::JacobiPreconditioner;
    use approx::assert_relative_eq;

    #[test]
    fn bicgstab_on_spd_matches_direct_solve() {
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
        let sol = bicgstab(
            |x, y| y.copy_from_slice(&mat.matvec(x).unwrap()),
            &b,
            vec![0.0; n],
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
    fn bicgstab_on_nonsymmetric_system() {
        let a = Matrix::from_row_major(
            4,
            4,
            vec![
                10.0, 1.0, 0.0, 0.0, 2.0, 8.0, 1.0, 0.0, 0.0, 3.0, 9.0, 1.0, 0.0, 0.0, 2.0, 7.0,
            ],
        );
        let x_true = vec![1.0, 2.0, -1.0, 3.0];
        let b = a.matvec(&x_true).unwrap();
        let sol = bicgstab(
            |x, y| y.copy_from_slice(&a.matvec(x).unwrap()),
            &b,
            vec![0.0; 4],
            Tolerance::default(),
        )
        .unwrap();
        for (xi, ti) in sol.value.iter().zip(&x_true)
        {
            assert_relative_eq!(*xi, *ti, epsilon = 1e-7);
        }
    }

    #[test]
    fn bicgstab_preconditioned_matches_true_solution() {
        let n = 5;
        let diag: Vec<f64> = (1..=n).map(|i| 20.0 * i as f64).collect();
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
        let x_true: Vec<f64> = (0..n).map(|i| (i as f64) - 2.0).collect();
        let b = mat.matvec(&x_true).unwrap();
        let jac = JacobiPreconditioner::new(&diag).unwrap();
        let sol = bicgstab_preconditioned(
            |x, y| y.copy_from_slice(&mat.matvec(x).unwrap()),
            jac.as_fn(),
            &b,
            vec![0.0; n],
            Tolerance::default(),
        )
        .unwrap();
        for (xi, ti) in sol.value.iter().zip(&x_true)
        {
            assert_relative_eq!(*xi, *ti, epsilon = 1e-6);
        }
    }

    #[test]
    fn bicgstab_rejects_dimension_mismatch() {
        let res = bicgstab(
            |_x, y| y.fill(0.0),
            &[1.0, 2.0],
            vec![0.0; 3],
            Tolerance::default(),
        );
        assert!(res.is_err());
    }

    #[test]
    fn bicgstab_converges_immediately_on_homogeneous_system() {
        // Regression test for a P1 audit finding: with b = 0, r = r_hat = 0
        // so rho_new = ⟨r_hat, r⟩ = 0, which the old absolute BREAKDOWN_EPS
        // test reported as SolverError::Singular on a perfectly regular
        // matrix (the identity).
        let n = 4;
        let identity = Matrix::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.0 });
        let sol = bicgstab(
            |x, y| y.copy_from_slice(&identity.matvec(x).unwrap()),
            &vec![0.0; n],
            vec![0.0; n],
            Tolerance::default(),
        )
        .expect("a homogeneous system with a regular matrix must converge, not error");
        assert_eq!(sol.info.iterations, 0);
    }

    #[test]
    fn bicgstab_solves_a_tiny_scale_system_without_false_breakdown() {
        // Regression test for a P1 audit finding: BREAKDOWN_EPS was compared
        // directly against rho = ⟨r_hat, r⟩, a quantity that scales
        // quadratically with ‖b‖ — a regular system with a small ‖b‖ (e.g.
        // ~1e-7, plausible in SI micro-units) was declared singular even
        // though it is perfectly well-conditioned (here, the identity).
        let n = 3;
        let identity = Matrix::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.0 });
        let b = vec![1e-7, 1e-7, 1e-7];
        let sol = bicgstab(
            |x, y| y.copy_from_slice(&identity.matvec(x).unwrap()),
            &b,
            vec![0.0; n],
            Tolerance::default(),
        )
        .expect("a regular system with a tiny ‖b‖ must not be reported singular");
        assert_relative_eq!(sol.value.as_slice(), b.as_slice(), epsilon = 1e-9);
    }
}

/// LAPACK-style property test: BiCGSTAB's residual must be small on a
/// well-conditioned, possibly non-symmetric system — the regime it targets
/// that plain CG cannot handle — checked over many random matrices.
#[cfg(test)]
mod proptests {
    use super::*;
    use crate::linalg::Matrix;
    use proptest::prelude::*;

    /// Force strict row diagonal dominance (Gershgorin ⇒ nonsingular and
    /// well-conditioned), without symmetry — the target regime for
    /// BiCGSTAB, vs. the SPD-only regime CG handles.
    fn diagonally_dominant(n: usize, raw: &[f64]) -> Matrix {
        let mut m = Matrix::from_row_major(n, n, raw.to_vec());
        for i in 0..n
        {
            let off_sum: f64 = (0..n).filter(|&j| j != i).map(|j| m[(i, j)].abs()).sum();
            let sign = if m[(i, i)] < 0.0 { -1.0 } else { 1.0 };
            m[(i, i)] = sign * (m[(i, i)].abs() + off_sum) + sign;
        }
        m
    }

    proptest! {
        #[test]
        fn residual_is_small_on_diagonally_dominant_nonsymmetric_systems(
            raw in prop::collection::vec(-10.0f64..10.0, 16),
            b in prop::collection::vec(-10.0f64..10.0, 4),
        ) {
            let n = 4;
            let a = diagonally_dominant(n, &raw);
            let tol = Tolerance { max_iter: 1000, ..Tolerance::default() };
            let sol = bicgstab(
                |x, y| y.copy_from_slice(&a.matvec(x).unwrap()),
                &b,
                vec![0.0; n],
                tol,
            )
            .expect("BiCGSTAB must converge on a well-conditioned system");
            let ax = a.matvec(&sol.value).unwrap();
            let b_norm = norm2(&b).max(1e-300);
            let res = ax.iter().zip(&b).map(|(axi, bi)| (axi - bi).powi(2)).sum::<f64>().sqrt();
            prop_assert!(res / b_norm < 1e-6, "relative residual {} too large", res / b_norm);
        }
    }
}
