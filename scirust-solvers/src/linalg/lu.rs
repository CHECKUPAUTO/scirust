//! Décomposition LU avec pivot partiel.
//!
//! `lu_decompose(A)` factorise A en P·A = L·U où L est triangulaire inf
//! avec 1 sur la diagonale, U est triangulaire sup, et P permutation.
//! Renvoie une structure `Lu` qui combine L et U dans une seule matrice
//! (parties triangulaires) plus le vecteur de permutation et le compteur
//! de swaps (pour le calcul du déterminant).

use super::Matrix;
use crate::{SolverError, SolverResult};
use tracing::warn;

/// Given the largest-magnitude entry of the matrix and its size, returns the
/// pivot-rejection threshold `n · eps · max|a_ij|` (Golub & Van Loan, *Matrix
/// Computations*, §3.4.6) — relative to the matrix's own scale rather than a
/// fixed absolute constant, so a regular matrix at a small physical scale
/// isn't declared singular.
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

#[derive(Debug, Clone)]
pub struct Lu {
    /// Matrice combinée : partie strictement inf = L (sans la diag = 1),
    /// partie sup + diag = U.
    pub lu: Matrix,
    /// Permutation : ligne i de A originale → ligne `piv[i]` après pivot.
    pub piv: Vec<usize>,
    /// Nombre de swaps effectués (pour le signe du déterminant).
    pub swap_count: usize,
    /// Seuil de pivot (relatif à l'échelle de la matrice d'origine), reporté
    /// depuis `lu_decompose` pour que `solve_lu` applique le même critère.
    piv_tol: f64,
}

/// Factorisation LU avec pivot partiel par ligne (Doolittle).
/// Mute `a` (copie locale) ; renvoie l'objet `Lu`.
pub fn lu_decompose(mut a: Matrix) -> SolverResult<Lu> {
    let n = a.ensure_square()?;
    let max_abs = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .fold(0.0f64, |acc, (i, j)| acc.max(a[(i, j)].abs()));
    let piv_tol = pivot_tol(n, max_abs);
    let mut piv = (0..n).collect::<Vec<_>>();
    let mut swap_count = 0;

    for k in 0..n
    {
        // Pivot partiel
        let mut max_idx = k;
        let mut max_val = a[(k, k)].abs();
        for i in (k + 1)..n
        {
            let v = a[(i, k)].abs();
            check_finite(v, &format!("LU pivot a[{i},{k}]"))?;
            if v > max_val
            {
                max_val = v;
                max_idx = i;
            }
        }
        if max_val < piv_tol
        {
            warn!(
                target: "solver",
                "LU: singular matrix at column {k}, max pivot candidate = {max_val:.3e}",
            );
            return Err(SolverError::Singular {
                row: k,
                pivot: a[(k, k)],
            });
        }
        if max_idx != k
        {
            a.swap_rows(k, max_idx);
            piv.swap(k, max_idx);
            swap_count += 1;
        }

        // Élimination
        let pivot = a[(k, k)];
        for i in (k + 1)..n
        {
            let factor = a[(i, k)] / pivot;
            check_finite(factor, &format!("LU factor L[{i},{k}]"))?;
            a[(i, k)] = factor;
            for j in (k + 1)..n
            {
                let aij = a[(i, j)] - factor * a[(k, j)];
                check_finite(aij, &format!("LU update a[{i},{j}]"))?;
                a[(i, j)] = aij;
            }
        }
    }

    Ok(Lu {
        lu: a,
        piv,
        swap_count,
        piv_tol,
    })
}

/// Résout L·U·x = P·b avec une factorisation déjà calculée.
pub fn solve_lu(lu: &Lu, b: &[f64]) -> SolverResult<Vec<f64>> {
    let n = lu.lu.rows();
    if b.len() != n
    {
        return Err(SolverError::DimensionMismatch {
            expected: n,
            got: b.len(),
        });
    }

    for (i, &bi) in b.iter().enumerate()
    {
        check_finite(bi, &format!("b[{i}] LU"))?;
    }

    // Applique la permutation : b' = P·b
    let mut x = vec![0.0; n];
    for i in 0..n
    {
        x[i] = b[lu.piv[i]];
    }

    // Substitution avant : L·y = b'
    for i in 0..n
    {
        let mut s = x[i];
        for j in 0..i
        {
            s -= lu.lu[(i, j)] * x[j];
        }
        x[i] = s;
    }

    // Substitution arrière : U·x = y
    for i in (0..n).rev()
    {
        let mut s = x[i];
        for j in (i + 1)..n
        {
            s -= lu.lu[(i, j)] * x[j];
        }
        let pivot = lu.lu[(i, i)];
        if pivot.abs() < lu.piv_tol
        {
            warn!(
                target: "solver",
                "LU back-substitution: near-singular pivot {pivot:.3e} at row {i}",
            );
            return Err(SolverError::Singular { row: i, pivot });
        }
        x[i] = s / pivot;
        check_finite(x[i], &format!("x[{i}] LU solve"))?;
    }

    Ok(x)
}

/// Helper : résout A·x = b en une seule étape (factorise + résout).
pub fn solve(a: Matrix, b: &[f64]) -> SolverResult<Vec<f64>> {
    let lu = lu_decompose(a)?;
    solve_lu(&lu, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn solve_2x2() -> SolverResult<()> {
        let a = Matrix::from_row_major(2, 2, vec![2.0, 1.0, 1.0, 3.0]);
        let b = vec![4.0, 5.0];
        let x = solve(a, &b)?;
        assert_relative_eq!(x[0], 1.4, epsilon = 1e-12);
        assert_relative_eq!(x[1], 1.2, epsilon = 1e-12);
        Ok(())
    }

    #[test]
    fn solve_2x2_at_a_tiny_physical_scale() -> SolverResult<()> {
        // Regression test for a P1 audit finding: PIVOT_EPS was a fixed
        // absolute 1e-14 compared directly against the pivot magnitude — the
        // same regular 2x2 system as `solve_2x2` above, scaled down to a tiny
        // physical magnitude, was declared singular even though it is
        // perfectly well-conditioned (scaling doesn't change the condition
        // number).
        let scale = 1e-16;
        let a = Matrix::from_row_major(
            2,
            2,
            vec![2.0 * scale, 1.0 * scale, 1.0 * scale, 3.0 * scale],
        );
        let b = vec![4.0 * scale, 5.0 * scale];
        let x = solve(a, &b)?;
        assert_relative_eq!(x[0], 1.4, epsilon = 1e-9);
        assert_relative_eq!(x[1], 1.2, epsilon = 1e-9);
        Ok(())
    }

    #[test]
    fn solve_4x4_with_pivot() -> SolverResult<()> {
        let a = Matrix::from_row_major(
            4,
            4,
            vec![
                0.0, 2.0, 0.0, 1.0, 2.0, 2.0, 3.0, 2.0, 4.0, -3.0, 0.0, 1.0, 6.0, 1.0, -6.0, -5.0,
            ],
        );
        let b = vec![0.0, -2.0, -7.0, 6.0];
        let x = solve(a.clone(), &b)?;
        let ax = a.matvec(&x)?;
        for (axi, bi) in ax.iter().zip(&b)
        {
            assert_relative_eq!(*axi, *bi, epsilon = 1e-10);
        }
        Ok(())
    }

    #[test]
    fn determinant_3x3() -> SolverResult<()> {
        let a = Matrix::from_row_major(3, 3, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 1.0, 0.0, 6.0]);
        assert_relative_eq!(a.determinant()?, 22.0, epsilon = 1e-12);
        Ok(())
    }

    #[test]
    fn inverse_3x3() -> SolverResult<()> {
        let a = Matrix::from_row_major(3, 3, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 1.0, 0.0, 6.0]);
        let inv = a.inverse()?;
        let prod = a.matmul(&inv)?;
        let id = Matrix::identity(3);
        for i in 0..3
        {
            for j in 0..3
            {
                assert_relative_eq!(prod[(i, j)], id[(i, j)], epsilon = 1e-10);
            }
        }
        Ok(())
    }
}
