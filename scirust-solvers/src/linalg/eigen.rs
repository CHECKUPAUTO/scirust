//! Décomposition en valeurs propres d'une matrice dense **symétrique**.
//!
//! Algorithme classique en deux étapes (Golub & Van Loan, *Matrix
//! Computations*, 4e éd., §8.3–8.4 ; portage fidèle de la paire
//! `tred2`/`tql2` d'EISPACK, du domaine public, telle que popularisée par
//! Numerical Recipes et JAMA) :
//!
//! 1. **Tridiagonalisation de Householder** (`tred2`) : `A = Q·T·Qᵀ` avec
//!    `T` tridiagonale, en accumulant `Q` en place.
//! 2. **Algorithme QL implicite à décalage de Wilkinson** (`tql2`) sur `T`,
//!    en faisant tourner l'accumulateur `Q` par les mêmes rotations de
//!    Givens que celles qui diagonalisent `T` — donne directement les
//!    vecteurs propres, pas seulement les valeurs propres.
//!
//! Ceci remplace, comme primitive partagée et publique, les implémentations
//! privées et non réutilisables de Jacobi cyclique qui existaient déjà
//! (`scirust-multivariate` pour la PCA) : utile pour la stabilité de
//! commande (pôles, Lyapunov), l'analyse modale/vibratoire, la PCA par
//! valeurs propres de la covariance, etc.
//!
//! ## Déterminisme
//! Boucle de décalage QL bornée par un budget total d'itérations fixe
//! (`MAX_TOTAL_SWEEPS`) plutôt qu'un critère temporel ; tolérance de
//! déflation `eps * (|d_l| + |e_l|)` avec `eps` la précision machine fixe —
//! aucune dépendance au nombre de threads (l'algorithme est intrinsèquement
//! séquentiel).

use crate::linalg::Matrix;
use crate::{SolverError, SolverResult};

const MACHINE_EPS: f64 = 2.220446049250313e-16; // 2^-52

/// Résultat d'une décomposition en valeurs propres symétrique.
///
/// `eigenvalues` est trié par ordre croissant ; `eigenvectors[(i, j)]` est
/// la `i`-ème composante du vecteur propre associé à `eigenvalues[j]`
/// (vecteurs en colonnes, orthonormés).
#[derive(Debug, Clone)]
pub struct EigenSymmetric {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Matrix,
}

/// Décompose une matrice symétrique dense `A = V·diag(λ)·Vᵀ`.
///
/// Retourne une erreur si `A` n'est pas (approximativement) symétrique ou
/// contient des valeurs non finies. La tolérance de symétrie est relative à
/// la norme de Frobenius de `A`.
pub fn eigen_symmetric(a: &Matrix) -> SolverResult<EigenSymmetric> {
    let n = a.ensure_square()?;
    if n == 0
    {
        return Err(SolverError::InvalidInput(
            "eigen_symmetric: empty matrix".to_string(),
        ));
    }
    for &x in a.data()
    {
        if !x.is_finite()
        {
            return Err(SolverError::NanDetected { iter: 0, value: x });
        }
    }
    let fro = a.frobenius_norm().max(1.0);
    for i in 0..n
    {
        for j in (i + 1)..n
        {
            if (a[(i, j)] - a[(j, i)]).abs() > 1e-8 * fro
            {
                return Err(SolverError::InvalidInput(format!(
                    "eigen_symmetric: matrix is not symmetric at ({i},{j}): {} vs {}",
                    a[(i, j)],
                    a[(j, i)]
                )));
            }
        }
    }

    let mut v = a.clone();
    // Symétrise exactement (moyenne) pour absorber le bruit d'arrondi
    // toléré ci-dessus avant de lancer les algorithmes qui supposent une
    // symétrie exacte.
    for i in 0..n
    {
        for j in (i + 1)..n
        {
            let avg = 0.5 * (v[(i, j)] + v[(j, i)]);
            v[(i, j)] = avg;
            v[(j, i)] = avg;
        }
    }

    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    tred2(&mut v, &mut d, &mut e, n);
    tql2(&mut v, &mut d, &mut e, n)?;

    Ok(EigenSymmetric {
        eigenvalues: d,
        eigenvectors: v,
    })
}

/// Tridiagonalisation de Householder. `v` contient `A` en entrée et
/// l'accumulateur de transformation en sortie ; `d`/`e` reçoivent la
/// diagonale et la sous-diagonale de `T`.
fn tred2(v: &mut Matrix, d: &mut [f64], e: &mut [f64], n: usize) {
    for j in 0..n
    {
        d[j] = v[(n - 1, j)];
    }

    for i in (1..n).rev()
    {
        let mut scale = 0.0;
        let mut h = 0.0;
        for k in 0..i
        {
            scale += d[k].abs();
        }
        if scale == 0.0
        {
            e[i] = d[i - 1];
            for j in 0..i
            {
                d[j] = v[(i - 1, j)];
                v[(i, j)] = 0.0;
                v[(j, i)] = 0.0;
            }
        }
        else
        {
            for k in 0..i
            {
                d[k] /= scale;
                h += d[k] * d[k];
            }
            let mut f = d[i - 1];
            let mut g = h.sqrt();
            if f > 0.0
            {
                g = -g;
            }
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            for j in 0..i
            {
                e[j] = 0.0;
            }
            for j in 0..i
            {
                f = d[j];
                v[(j, i)] = f;
                g = e[j] + v[(j, j)] * f;
                for k in (j + 1)..i
                {
                    g += v[(k, j)] * d[k];
                    e[k] += v[(k, j)] * f;
                }
                e[j] = g;
            }
            f = 0.0;
            for j in 0..i
            {
                e[j] /= h;
                f += e[j] * d[j];
            }
            let hh = f / (h + h);
            for j in 0..i
            {
                e[j] -= hh * d[j];
            }
            for j in 0..i
            {
                f = d[j];
                g = e[j];
                for k in j..i
                {
                    v[(k, j)] -= f * e[k] + g * d[k];
                }
                d[j] = v[(i - 1, j)];
                v[(i, j)] = 0.0;
            }
        }
        d[i] = h;
    }

    for i in 0..(n - 1)
    {
        v[(n - 1, i)] = v[(i, i)];
        v[(i, i)] = 1.0;
        let h = d[i + 1];
        if h != 0.0
        {
            for k in 0..=i
            {
                d[k] = v[(k, i + 1)] / h;
            }
            for j in 0..=i
            {
                let mut g = 0.0;
                for k in 0..=i
                {
                    g += v[(k, i + 1)] * v[(k, j)];
                }
                for k in 0..=i
                {
                    v[(k, j)] -= g * d[k];
                }
            }
        }
        for k in 0..=i
        {
            v[(k, i + 1)] = 0.0;
        }
    }
    for j in 0..n
    {
        d[j] = v[(n - 1, j)];
        v[(n - 1, j)] = 0.0;
    }
    v[(n - 1, n - 1)] = 1.0;
    e[0] = 0.0;
}

const MAX_TOTAL_SWEEPS: usize = 128;

/// Algorithme QL implicite à décalage de Wilkinson, appliqué à la
/// tridiagonale `(d, e)` produite par `tred2`, en faisant tourner `v` par
/// les mêmes rotations pour obtenir les vecteurs propres.
fn tql2(v: &mut Matrix, d: &mut [f64], e: &mut [f64], n: usize) -> SolverResult<()> {
    for i in 1..n
    {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;

    let mut f: f64 = 0.0;
    let mut tst1: f64 = 0.0;
    let mut budget = n.saturating_mul(MAX_TOTAL_SWEEPS);

    for l in 0..n
    {
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n
        {
            if e[m].abs() <= MACHINE_EPS * tst1
            {
                break;
            }
            m += 1;
        }

        if m > l
        {
            loop
            {
                if budget == 0
                {
                    return Err(SolverError::NoConvergence {
                        iterations: n * MAX_TOTAL_SWEEPS,
                        residual: e[l].abs(),
                    });
                }
                budget -= 1;

                let mut g = d[l];
                let p0 = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p0.hypot(1.0);
                if p0 < 0.0
                {
                    r = -r;
                }
                d[l] = e[l] / (p0 + r);
                d[l + 1] = e[l] * (p0 + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for i in (l + 2)..n
                {
                    d[i] -= h;
                }
                f += h;

                let mut p = d[m];
                let mut c = 1.0;
                let mut c2 = c;
                let mut c3 = c;
                let el1 = e[l + 1];
                let mut s = 0.0;
                let mut s2 = 0.0;

                for i in (l..=(m - 1)).rev()
                {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = p.hypot(e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);

                    for k in 0..n
                    {
                        h = v[(k, i + 1)];
                        v[(k, i + 1)] = s * v[(k, i)] + c * h;
                        v[(k, i)] = c * v[(k, i)] - s * h;
                    }
                }

                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;

                if e[l].abs() <= MACHINE_EPS * tst1
                {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }

    // Tri croissant des valeurs propres (et permutation assortie des
    // vecteurs) — ordre de sortie stable et indépendant de l'implémentation.
    for i in 0..(n - 1)
    {
        let mut k = i;
        let mut p = d[i];
        for (j, &dj) in d.iter().enumerate().skip(i + 1)
        {
            if dj < p
            {
                k = j;
                p = dj;
            }
        }
        if k != i
        {
            d[k] = d[i];
            d[i] = p;
            for j in 0..n
            {
                let tmp = v[(j, i)];
                v[(j, i)] = v[(j, k)];
                v[(j, k)] = tmp;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn reconstruct(eig: &EigenSymmetric, n: usize) -> Matrix {
        let mut out = Matrix::zeros(n, n);
        for i in 0..n
        {
            for j in 0..n
            {
                let mut s = 0.0;
                for k in 0..n
                {
                    s += eig.eigenvectors[(i, k)] * eig.eigenvalues[k] * eig.eigenvectors[(j, k)];
                }
                out[(i, j)] = s;
            }
        }
        out
    }

    #[test]
    fn identity_has_unit_eigenvalues() {
        let eig = eigen_symmetric(&Matrix::identity(4)).unwrap();
        for &l in &eig.eigenvalues
        {
            assert_relative_eq!(l, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn diagonal_matrix_returns_sorted_diagonal() {
        let a = Matrix::from_row_major(3, 3, vec![5.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0]);
        let eig = eigen_symmetric(&a).unwrap();
        assert_relative_eq!(eig.eigenvalues[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(eig.eigenvalues[1], 3.0, epsilon = 1e-10);
        assert_relative_eq!(eig.eigenvalues[2], 5.0, epsilon = 1e-10);
    }

    #[test]
    fn known_2x2_eigenvalues() {
        // [[2,1],[1,2]] a pour valeurs propres 1 et 3.
        let a = Matrix::from_row_major(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let eig = eigen_symmetric(&a).unwrap();
        assert_relative_eq!(eig.eigenvalues[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(eig.eigenvalues[1], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn reconstruction_matches_original_5x5() {
        // Tridiagonale symétrique bien conditionnée (pas un Hilbert — trop
        // mal conditionné pour une comparaison directe à 1e-8).
        let n = 5;
        let a = Matrix::from_fn(n, n, |i, j| {
            if i == j
            {
                4.0 + i as f64
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
        let eig = eigen_symmetric(&a).unwrap();
        let rebuilt = reconstruct(&eig, n);
        for i in 0..n
        {
            for j in 0..n
            {
                assert_relative_eq!(rebuilt[(i, j)], a[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn eigenvectors_are_orthonormal() {
        let n = 4;
        let a = Matrix::from_row_major(
            4,
            4,
            vec![
                4.0, 1.0, 0.0, 0.5, 1.0, 3.0, 0.2, 0.0, 0.0, 0.2, 2.0, 0.1, 0.5, 0.0, 0.1, 1.0,
            ],
        );
        let eig = eigen_symmetric(&a).unwrap();
        for i in 0..n
        {
            for j in 0..n
            {
                let mut dot = 0.0;
                for k in 0..n
                {
                    dot += eig.eigenvectors[(k, i)] * eig.eigenvectors[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(dot, expected, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn rejects_non_symmetric_matrix() {
        let a = Matrix::from_row_major(2, 2, vec![1.0, 2.0, 0.0, 1.0]);
        assert!(eigen_symmetric(&a).is_err());
    }

    #[test]
    fn rejects_non_square_matrix() {
        let a = Matrix::from_row_major(2, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert!(eigen_symmetric(&a).is_err());
    }
}
