//! Décomposition de Cholesky pour matrices symétriques définies positives.
//!
//! A = L · L^T où L est triangulaire inférieure.

use super::Matrix;
use crate::{SolverError, SolverResult};
use tracing::warn;

/// Given the largest-magnitude entry seen so far and the matrix size,
/// returns the pivot-rejection threshold `n · eps · max|·|` (Golub & Van
/// Loan, *Matrix Computations*, §3.4.6) — relative to scale rather than a
/// fixed absolute constant, so a regular matrix at a small physical scale
/// isn't declared singular/non-SPD.
fn pivot_tol(n: usize, max_abs: f64) -> f64 {
    (n as f64) * f64::EPSILON * max_abs.max(1e-300)
}

fn check_finite(value: f64, _location: &str) -> Result<(), SolverError> {
    if !value.is_finite()
    {
        return Err(SolverError::NanDetected { iter: 0, value });
    }
    Ok(())
}

/// Cholesky en place : remplit la partie triangulaire inférieure de A avec L.
/// L'entrée doit être symétrique définie positive ; sinon `NotSpd`.
/// Renvoie L (matrice triangulaire inf avec zéros au-dessus).
pub fn cholesky_decompose(a: Matrix) -> SolverResult<Matrix> {
    let n = a.ensure_square()?;
    let max_abs = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .fold(0.0f64, |acc, (i, j)| acc.max(a[(i, j)].abs()));
    let piv_tol = pivot_tol(n, max_abs);
    let mut l = Matrix::zeros(n, n);

    for i in 0..n
    {
        for j in 0..=i
        {
            let mut s = 0.0;
            for k in 0..j
            {
                let lik = l[(i, k)];
                let ljk = l[(j, k)];
                check_finite(lik, &format!("l[{i},{k}] Cholesky"))?;
                check_finite(ljk, &format!("l[{j},{k}] Cholesky"))?;
                s += lik * ljk;
            }
            let aij = a[(i, j)];
            check_finite(aij, &format!("a[{i},{j}] Cholesky"))?;

            if i == j
            {
                let val = aij - s;
                if val <= 0.0
                {
                    warn!(
                        target: "solver",
                        "Cholesky failed: a[{i},{i}] - s = {:.3e} <= 0 (not SPD)",
                        val
                    );
                    return Err(SolverError::NotSpd);
                }
                let root = val.sqrt();
                check_finite(root, &format!("sqrt Cholesky [{i},{i}]"))?;
                l[(i, j)] = root;
            }
            else
            {
                let ljj = l[(j, j)];
                if ljj.abs() < piv_tol
                {
                    warn!(
                        target: "solver",
                        "Cholesky: l[{j},{j}] = {:.3e} near-zero at row {} — not SPD",
                        ljj, j
                    );
                    return Err(SolverError::NotSpd);
                }
                let entry = (aij - s) / ljj;
                check_finite(entry, &format!("l[{i},{j}] Cholesky"))?;
                l[(i, j)] = entry;
            }
        }
    }
    Ok(l)
}

/// Résout A · x = b sachant A = L·L^T, en deux passes triangulaires.
pub fn solve_cholesky(l: &Matrix, b: &[f64]) -> SolverResult<Vec<f64>> {
    let n = l.rows();
    if b.len() != n
    {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }
    // Vérifier que b est fini
    for (i, &bi) in b.iter().enumerate()
    {
        check_finite(bi, &format!("b[{i}]"))?;
    }

    // L n'est pas reçue avec la matrice A d'origine ; sa propre diagonale est
    // la référence d'échelle disponible ici (cf. cholesky_decompose, qui
    // utilise max|a_ij|).
    let max_abs = (0..n).fold(0.0f64, |acc, i| acc.max(l[(i, i)].abs()));
    let piv_tol = pivot_tol(n, max_abs);

    // L · y = b (substitution avant)
    let mut y = vec![0.0; n];
    for i in 0..n
    {
        let mut s = b[i];
        for j in 0..i
        {
            s -= l[(i, j)] * y[j];
        }
        let diag = l[(i, i)];
        if diag.abs() < piv_tol
        {
            return Err(SolverError::Singular {
                row: i,
                pivot: diag,
            });
        }
        y[i] = s / diag;
        check_finite(y[i], &format!("y[{i}]"))?;
    }
    // L^T · x = y (substitution arrière)
    let mut x = vec![0.0; n];
    for i in (0..n).rev()
    {
        let mut s = y[i];
        for j in (i + 1)..n
        {
            s -= l[(j, i)] * x[j];
        }
        let diag = l[(i, i)];
        if diag.abs() < piv_tol
        {
            return Err(SolverError::Singular {
                row: i,
                pivot: diag,
            });
        }
        x[i] = s / diag;
        check_finite(x[i], &format!("x[{i}] Cholesky"))?;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cholesky_3x3() -> SolverResult<()> {
        let a = Matrix::from_row_major(
            3,
            3,
            vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
        );
        let l = cholesky_decompose(a.clone())?;
        assert_relative_eq!(l[(0, 0)], 2.0, epsilon = 1e-10);
        assert_relative_eq!(l[(1, 1)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(l[(2, 2)], 3.0, epsilon = 1e-10);
        let lt = l.transpose();
        let prod = l.matmul(&lt)?;
        for i in 0..3
        {
            for j in 0..3
            {
                assert_relative_eq!(prod[(i, j)], a[(i, j)], epsilon = 1e-10);
            }
        }
        Ok(())
    }

    #[test]
    fn cholesky_solve() -> SolverResult<()> {
        let a = Matrix::from_row_major(
            3,
            3,
            vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
        );
        let b = vec![1.0, 2.0, 3.0];
        let l = cholesky_decompose(a.clone())?;
        let x = solve_cholesky(&l, &b)?;
        let ax = a.matvec(&x)?;
        for (axi, bi) in ax.iter().zip(&b)
        {
            assert_relative_eq!(*axi, *bi, epsilon = 1e-9);
        }
        Ok(())
    }

    #[test]
    fn cholesky_solve_at_a_tiny_physical_scale() -> SolverResult<()> {
        // Regression test for a P1 audit finding: PIVOT_EPS was a fixed
        // absolute 1e-15 compared directly against L's diagonal — the same
        // regular SPD system as `cholesky_solve` above, scaled down so that L's
        // diagonal (which scales as √scale) falls well below the old cutoff,
        // was declared not-SPD even though it is perfectly well-conditioned.
        let scale = 1e-34;
        let a = Matrix::from_row_major(
            3,
            3,
            vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0]
                .into_iter()
                .map(|v| v * scale)
                .collect(),
        );
        let b = vec![1.0 * scale, 2.0 * scale, 3.0 * scale];
        let l = cholesky_decompose(a.clone())?;
        let x = solve_cholesky(&l, &b)?;
        let ax = a.matvec(&x)?;
        for (axi, bi) in ax.iter().zip(&b)
        {
            assert_relative_eq!(*axi, *bi, epsilon = 1e-9, max_relative = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn rejects_non_spd() {
        // Pas SPD (négative)
        let a = Matrix::from_row_major(2, 2, vec![-1.0, 0.0, 0.0, -1.0]);
        assert!(matches!(cholesky_decompose(a), Err(SolverError::NotSpd)));
    }
}
