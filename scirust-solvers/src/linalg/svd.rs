//! SVD dense générale par la méthode de Jacobi à un côté (« one-sided
//! Jacobi », Hestenes 1958).
//!
//! Contrairement à la troncature de `scirust-core::tn::ops::svd` (qui
//! s'appuie sur `nalgebra`, pensée pour les réseaux de tenseurs), cette
//! implémentation est écrite from scratch et s'applique à n'importe quelle
//! matrice dense `(m, n)` — pseudo-inverse, moindres carrés de rang
//! déficient, PCA par SVD plutôt que par la matrice de covariance.
//!
//! Référence : M.R. Hestenes, « Inversion of Matrices by Biorthogonalization
//! and Related Results », J. SIAM 6(1), 1958 ; Golub & Van Loan, *Matrix
//! Computations*, 4e éd., §8.6.3 ; analyse de précision moderne dans
//! Drmač & Veselić, « New Fast and Accurate Jacobi SVD Algorithm », SIAM J.
//! Matrix Anal. Appl. 29(4), 2008.
//!
//! ## Déterminisme
//! Balayage des paires de colonnes `(p, q)` dans un ordre lexicographique
//! fixe, nombre max de balayages fixe (`MAX_SWEEPS`), seuil de convergence
//! fixe — aucune dépendance à l'ordre d'exécution ou au nombre de threads.

use crate::linalg::{Matrix, dot};
use crate::{SolverError, SolverResult};

const MAX_SWEEPS: usize = 60;
const CONV_EPS: f64 = 1e-14;

/// SVD fine (« thin ») : `A ≈ U · diag(s) · Vᵀ`.
///
/// `u` a la forme `(m, k)`, `v` la forme `(n, k)` avec `k = min(m, n)`, et
/// `s` (longueur `k`) est trié par ordre décroissant.
///
/// Pour une valeur singulière nulle (matrice de rang déficient), la colonne
/// de `u` correspondante n'est pas définie de façon unique ; elle est
/// laissée à zéro, ce qui reste correct pour la reconstruction (`0·x = 0`)
/// mais n'est alors pas orthonormée — documenté ici plutôt que caché.
#[derive(Debug, Clone)]
pub struct Svd {
    pub u: Matrix,
    pub s: Vec<f64>,
    pub v: Matrix,
}

/// Calcule la SVD fine de `a`, quelle que soit sa forme `(m, n)`.
pub fn svd(a: &Matrix) -> SolverResult<Svd> {
    let (m, n) = a.shape();
    if m == 0 || n == 0
    {
        return Err(SolverError::InvalidInput("svd: empty matrix".to_string()));
    }
    for &x in a.data()
    {
        if !x.is_finite()
        {
            return Err(SolverError::NanDetected { iter: 0, value: x });
        }
    }

    if m >= n
    {
        svd_tall(a)
    }
    else
    {
        // A (m<n) : calcule la SVD de Aᵀ (tall) puis échange U et V —
        // A = U·S·Vᵀ  ⟺  Aᵀ = V·S·Uᵀ.
        let at = a.transpose();
        let tall = svd_tall(&at)?;
        Ok(Svd {
            u: tall.v,
            s: tall.s,
            v: tall.u,
        })
    }
}

fn svd_tall(a: &Matrix) -> SolverResult<Svd> {
    let (m, n) = a.shape();
    let mut cols: Vec<Vec<f64>> = (0..n)
        .map(|j| (0..m).map(|i| a[(i, j)]).collect())
        .collect();
    let mut v = Matrix::identity(n);

    for _sweep in 0..MAX_SWEEPS
    {
        let mut max_off = 0.0f64;
        for p in 0..n
        {
            for q in (p + 1)..n
            {
                let alpha = dot(&cols[p], &cols[p]);
                let beta = dot(&cols[q], &cols[q]);
                let gamma = dot(&cols[p], &cols[q]);
                let denom = (alpha * beta).sqrt();
                if denom > 1e-300
                {
                    max_off = max_off.max(gamma.abs() / denom);
                }
                if denom < 1e-300 || gamma.abs() <= CONV_EPS * denom
                {
                    continue;
                }

                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta >= 0.0
                {
                    1.0 / (zeta + (1.0 + zeta * zeta).sqrt())
                }
                else
                {
                    -1.0 / (-zeta + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                for i in 0..m
                {
                    let cp = cols[p][i];
                    let cq = cols[q][i];
                    cols[p][i] = c * cp - s * cq;
                    cols[q][i] = s * cp + c * cq;
                }
                for i in 0..n
                {
                    let vp = v[(i, p)];
                    let vq = v[(i, q)];
                    v[(i, p)] = c * vp - s * vq;
                    v[(i, q)] = s * vp + c * vq;
                }
            }
        }
        if max_off <= CONV_EPS
        {
            break;
        }
    }

    let sigma: Vec<f64> = cols.iter().map(|c| dot(c, c).sqrt()).collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        sigma[j]
            .partial_cmp(&sigma[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut u = Matrix::zeros(m, n);
    let mut v_sorted = Matrix::zeros(n, n);
    let mut s_sorted = vec![0.0; n];
    for (new_j, &old_j) in order.iter().enumerate()
    {
        s_sorted[new_j] = sigma[old_j];
        if sigma[old_j] > 1e-300
        {
            for i in 0..m
            {
                u[(i, new_j)] = cols[old_j][i] / sigma[old_j];
            }
        }
        for i in 0..n
        {
            v_sorted[(i, new_j)] = v[(i, old_j)];
        }
    }

    Ok(Svd {
        u,
        s: s_sorted,
        v: v_sorted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn reconstruct(svd: &Svd, m: usize, n: usize) -> Matrix {
        let k = svd.s.len();
        let mut out = Matrix::zeros(m, n);
        for i in 0..m
        {
            for j in 0..n
            {
                let mut acc = 0.0;
                for r in 0..k
                {
                    acc += svd.u[(i, r)] * svd.s[r] * svd.v[(j, r)];
                }
                out[(i, j)] = acc;
            }
        }
        out
    }

    #[test]
    fn svd_of_diagonal_matrix_is_its_diagonal() {
        let a = Matrix::from_row_major(2, 2, vec![3.0, 0.0, 0.0, 1.0]);
        let s = svd(&a).unwrap();
        assert_relative_eq!(s.s[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(s.s[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn reconstruction_matches_original_square() {
        let a = Matrix::from_row_major(3, 3, vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0]);
        let s = svd(&a).unwrap();
        let rebuilt = reconstruct(&s, 3, 3);
        for i in 0..3
        {
            for j in 0..3
            {
                assert_relative_eq!(rebuilt[(i, j)], a[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn reconstruction_matches_original_tall() {
        let a = Matrix::from_row_major(
            5,
            3,
            vec![
                1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.0, 2.0, 1.0,
            ],
        );
        let s = svd(&a).unwrap();
        let rebuilt = reconstruct(&s, 5, 3);
        for i in 0..5
        {
            for j in 0..3
            {
                assert_relative_eq!(rebuilt[(i, j)], a[(i, j)], epsilon = 1e-8);
            }
        }
        // Valeurs singulières décroissantes.
        assert!(s.s[0] >= s.s[1]);
        assert!(s.s[1] >= s.s[2]);
    }

    #[test]
    fn reconstruction_matches_original_wide() {
        let a = Matrix::from_row_major(2, 4, vec![1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 3.0, 1.0]);
        let s = svd(&a).unwrap();
        let rebuilt = reconstruct(&s, 2, 4);
        for i in 0..2
        {
            for j in 0..4
            {
                assert_relative_eq!(rebuilt[(i, j)], a[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn v_is_orthonormal() {
        let a = Matrix::from_row_major(
            4,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 1.0, 0.0, 1.0],
        );
        let s = svd(&a).unwrap();
        let n = 3;
        for i in 0..n
        {
            for j in 0..n
            {
                let mut acc = 0.0;
                for k in 0..n
                {
                    acc += s.v[(k, i)] * s.v[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(acc, expected, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn rank_deficient_matrix_has_zero_singular_value() {
        // Colonne 2 = 2 × colonne 1 ⇒ rang 1, la 2e valeur singulière est nulle.
        let a = Matrix::from_row_major(2, 2, vec![1.0, 2.0, 3.0, 6.0]);
        let s = svd(&a).unwrap();
        assert!(s.s[0] > 1e-6);
        assert!(s.s[1] < 1e-8);
    }

    #[test]
    fn rejects_empty_matrix() {
        assert!(svd(&Matrix::zeros(0, 0)).is_err());
    }
}
