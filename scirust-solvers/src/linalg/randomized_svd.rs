//! SVD aléatoire (randomized range finder + SVD, Halko, Martinsson & Tropp
//! 2011) : approxime la SVD tronquée de rang `k` d'une matrice dense `(m,
//! n)` en projetant sur un sous-espace aléatoire de dimension `k +
//! oversampling` plutôt qu'en décomposant la matrice complète — utile quand
//! `min(m, n)` est grand mais que seules les premières valeurs/vecteurs
//! singuliers comptent (PCA sur beaucoup de variables, réduction de modèle).
//!
//! Algorithme (Halko, Martinsson, Tropp, « Finding Structure with
//! Randomness: Probabilistic Algorithms for Constructing Approximate Matrix
//! Decompositions », SIAM Review 53(2), 2011, Algorithme 4.4 / 5.1) :
//! 1. Tire une matrice test gaussienne `Ω` (n × l), `l = k + oversampling`.
//! 2. `Y = A·Ω` (m × l) ; `q` itérations de puissance optionnelles
//!    `Y = A·(Aᵀ·Y)` avec ré-orthonormalisation QR entre chaque étape
//!    (stabilité numérique — évite que le sous-espace ne s'aplatisse).
//! 3. `Q` = base orthonormée de `Y` (QR de Householder déjà présent dans ce
//!    module).
//! 4. `B = Qᵀ·A` (l × n, petite) ; SVD exacte de `B` via
//!    [`crate::linalg::svd`] (Jacobi à un côté, déjà déterministe).
//! 5. `U = Q·U_B`, tronqué aux `k` premières colonnes (déjà triées par
//!    valeur singulière décroissante).
//!
//! ## Déterminisme
//! Le tirage aléatoire utilise [`super::rng::SplitMix64`], germé
//! explicitement par l'appelant — même graine ⇒ sortie bit-identique. Le
//! nombre d'itérations de puissance est fixe (paramètre, pas adaptatif).

use crate::linalg::rng::SplitMix64;
use crate::linalg::svd::{Svd, svd};
use crate::linalg::{Matrix, qr_decompose};
use crate::{SolverError, SolverResult};

/// Calcule une SVD fine approchée de rang `k` par projection aléatoire.
///
/// `oversampling` (typiquement 5 à 10) élargit le sous-espace de travail
/// au-delà de `k` pour améliorer la précision de l'approximation ;
/// `power_iterations` (typiquement 0 à 2) affine encore le sous-espace pour
/// les matrices à spectre de valeurs singulières à décroissance lente.
pub fn randomized_svd(
    a: &Matrix,
    k: usize,
    oversampling: usize,
    power_iterations: usize,
    seed: u64,
) -> SolverResult<Svd> {
    let (m, n) = a.shape();
    if m == 0 || n == 0
    {
        return Err(SolverError::InvalidInput(
            "randomized_svd: empty matrix".to_string(),
        ));
    }
    if k == 0
    {
        return Err(SolverError::InvalidInput(
            "randomized_svd: k must be >= 1".to_string(),
        ));
    }
    for &x in a.data()
    {
        if !x.is_finite()
        {
            return Err(SolverError::NanDetected { iter: 0, value: x });
        }
    }

    let l = (k + oversampling).min(m).min(n);
    if l == 0
    {
        return Err(SolverError::InvalidInput(
            "randomized_svd: k exceeds matrix dimensions".to_string(),
        ));
    }

    let mut rng = SplitMix64::new(seed);
    let mut omega = Matrix::zeros(n, l);
    for i in 0..n
    {
        for j in 0..l
        {
            omega[(i, j)] = rng.next_gaussian();
        }
    }

    let mut y = a.matmul(&omega)?;
    let at = a.transpose();
    for _ in 0..power_iterations
    {
        // Ré-orthonormalise entre chaque application pour éviter que le
        // sous-espace ne perde en rang numérique sur un spectre étalé.
        y = orthonormal_basis(&y)?;
        let z = at.matmul(&y)?;
        y = a.matmul(&z)?;
    }
    let q = orthonormal_basis(&y)?;

    let b = q.transpose().matmul(a)?;
    let svd_b = svd(&b)?;

    let u_full = q.matmul(&svd_b.u)?;
    let rank = k.min(svd_b.s.len());
    let (u_rows, _) = u_full.shape();
    let mut u = Matrix::zeros(u_rows, rank);
    let mut s = vec![0.0; rank];
    let (v_rows, _) = svd_b.v.shape();
    let mut v = Matrix::zeros(v_rows, rank);
    for j in 0..rank
    {
        s[j] = svd_b.s[j];
        for i in 0..u_rows
        {
            u[(i, j)] = u_full[(i, j)];
        }
        for i in 0..v_rows
        {
            v[(i, j)] = svd_b.v[(i, j)];
        }
    }

    Ok(Svd { u, s, v })
}

/// Base orthonormée des colonnes de `y` (m × l, l <= m) via QR de
/// Householder ; renvoie les `l` premières colonnes de `Q`.
fn orthonormal_basis(y: &Matrix) -> SolverResult<Matrix> {
    let (m, l) = y.shape();
    let qr = qr_decompose(y.clone())?;
    let q_full = qr.q();
    let mut q = Matrix::zeros(m, l);
    for i in 0..m
    {
        for j in 0..l
        {
            q[(i, j)] = q_full[(i, j)];
        }
    }
    Ok(q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn recovers_singular_values_of_a_low_rank_matrix() {
        // Construit une matrice de rang exactement 2 : A = u1 s1 v1^T + u2 s2 v2^T
        // avec des vecteurs orthonormés simples, valeurs singulières 5 et 2.
        let n = 6;
        let a = Matrix::from_fn(n, n, |i, j| {
            let u1 = if i % 2 == 0 { 1.0 } else { -1.0 } / (n as f64).sqrt();
            let v1 = if j % 2 == 0 { 1.0 } else { -1.0 } / (n as f64).sqrt();
            let u2 = 1.0 / (n as f64).sqrt();
            let v2 = 1.0 / (n as f64).sqrt();
            5.0 * u1 * v1 + 2.0 * u2 * v2
        });
        let approx = randomized_svd(&a, 2, 4, 2, 42).unwrap();
        let mut sorted = approx.s.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert_relative_eq!(sorted[0], 5.0, epsilon = 1e-8);
        assert_relative_eq!(sorted[1], 2.0, epsilon = 1e-8);
    }

    #[test]
    fn reconstruction_approximates_original_within_truncation_error() {
        let a = Matrix::from_row_major(
            4,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 1.0, 1.0],
        );
        let full = svd(&a).unwrap();
        let approx = randomized_svd(&a, 3, 5, 2, 7).unwrap();
        // Rang plein demandé (k = min(m,n)) : doit reconstruire quasi exactement.
        let mut rebuilt = Matrix::zeros(4, 3);
        for i in 0..4
        {
            for j in 0..3
            {
                let mut acc = 0.0;
                for r in 0..approx.s.len()
                {
                    acc += approx.u[(i, r)] * approx.s[r] * approx.v[(j, r)];
                }
                rebuilt[(i, j)] = acc;
            }
        }
        for i in 0..4
        {
            for j in 0..3
            {
                assert_relative_eq!(rebuilt[(i, j)], a[(i, j)], epsilon = 1e-6);
            }
        }
        assert_eq!(full.s.len(), approx.s.len().max(full.s.len()).min(3));
    }

    #[test]
    fn same_seed_is_bit_identical() {
        let a = Matrix::from_fn(8, 5, |i, j| {
            (i as f64 + 1.0) * (j as f64 + 2.0) + (i * j) as f64
        });
        let s1 = randomized_svd(&a, 2, 3, 1, 123).unwrap();
        let s2 = randomized_svd(&a, 2, 3, 1, 123).unwrap();
        assert_eq!(s1.s, s2.s);
    }

    #[test]
    fn different_seeds_still_agree_on_singular_values() {
        // Le sous-espace aléatoire diffère mais l'approximation de rang
        // plein doit converger vers les mêmes valeurs singulières.
        let a = Matrix::from_fn(6, 6, |i, j| if i == j { (i + 1) as f64 } else { 0.1 });
        let s1 = randomized_svd(&a, 6, 4, 3, 1).unwrap();
        let s2 = randomized_svd(&a, 6, 4, 3, 999).unwrap();
        for (a, b) in s1.s.iter().zip(&s2.s)
        {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[test]
    fn rejects_zero_rank() {
        let a = Matrix::identity(3);
        assert!(randomized_svd(&a, 0, 2, 0, 1).is_err());
    }

    #[test]
    fn rejects_empty_matrix() {
        assert!(randomized_svd(&Matrix::zeros(0, 0), 1, 2, 0, 1).is_err());
    }
}
