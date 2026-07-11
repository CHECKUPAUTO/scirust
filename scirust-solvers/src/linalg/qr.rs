//! Décomposition QR par réflexions de Householder.
//!
//! Pour une matrice A (m × n) avec m ≥ n :
//!   A = Q · R    où Q est orthogonale m×m et R triangulaire sup m×n.
//!
//! Stockage compact : on garde R dans la partie supérieure de la matrice
//! d'entrée et les vecteurs de Householder dans la partie inférieure
//! (avec leur tau dans un vecteur séparé).

use super::Matrix;
use crate::{SolverError, SolverResult};
use tracing::warn;

/// Seuil de NaN/Inf.
const FINITE_EPS: f64 = 1e-15;

/// Given the largest-magnitude entry seen so far and the matrix size,
/// returns the pivot-rejection threshold `n · eps · max|·|` (Golub & Van
/// Loan, *Matrix Computations*, §3.4.6) — relative to scale rather than a
/// fixed absolute constant, so a regular system at a small physical scale
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
pub struct Qr {
    /// Stockage compact (m × n) : R en sup, vecteurs Householder en inf.
    pub data: Matrix,
    /// Coefficients de Householder (longueur min(m,n)).
    pub tau: Vec<f64>,
    pub m: usize,
    pub n: usize,
}

impl Qr {
    /// Extrait R (n × n pour le cas overdéterminé, ou min(m,n) × n).
    pub fn r(&self) -> Matrix {
        let p = self.m.min(self.n);
        let mut r = Matrix::zeros(p, self.n);
        for i in 0..p
        {
            for j in i..self.n
            {
                r[(i, j)] = self.data[(i, j)];
            }
        }
        r
    }

    /// Reconstruit Q explicitement (m × m). Coûteux ; à éviter en hot path.
    pub fn q(&self) -> Matrix {
        let mut q = Matrix::identity(self.m);
        // Q = H_1 · H_2 · ... · H_p
        // On applique chaque H_k à droite de Q via "appliquer à chaque colonne".
        let p = self.tau.len();
        for k in (0..p).rev()
        {
            let tau_k = self.tau[k];
            if tau_k == 0.0
            {
                continue;
            }
            // v = (1, data[k+1..m, k]) — vecteur de Householder
            let mut v = vec![0.0; self.m - k];
            v[0] = 1.0;
            for i in (k + 1)..self.m
            {
                v[i - k] = self.data[(i, k)];
            }
            // Pour chaque colonne j de Q (lignes k..m), Q' = Q - tau * v * v^T · Q
            for j in 0..self.m
            {
                let mut dot = 0.0;
                for i in k..self.m
                {
                    dot += v[i - k] * q[(i, j)];
                }
                let s = tau_k * dot;
                for i in k..self.m
                {
                    q[(i, j)] -= s * v[i - k];
                }
            }
        }
        q
    }
}

/// Calcule la décomposition QR via Householder.
pub fn qr_decompose(mut a: Matrix) -> SolverResult<Qr> {
    let m = a.rows();
    let n = a.cols();
    if m < n
    {
        return Err(SolverError::InvalidInput(format!(
            "QR requires m >= n, got {}x{}",
            m, n
        )));
    }
    let p = m.min(n);
    let mut tau = vec![0.0; p];

    for k in 0..p
    {
        // Calcule la norme de a[k..m, k]
        let mut sigma_sq = 0.0;
        for i in k..m
        {
            let aik = a[(i, k)];
            check_finite(aik, &format!("a[{i},{k}] in QR sigma"))?;
            sigma_sq += aik * aik;
        }
        check_finite(sigma_sq, &format!("sigma_sq QR k={k}"))?;

        let sigma = sigma_sq.sqrt();
        if sigma < FINITE_EPS
        {
            // Colonne déjà à zéro → R[k,k] = 0, on saute le pivot
            tau[k] = 0.0;
            continue;
        }
        let akk = a[(k, k)];
        let alpha = if akk >= 0.0 { -sigma } else { sigma };
        let diff = akk - alpha;
        if diff.abs() < FINITE_EPS
        {
            tau[k] = 0.0;
            a[(k, k)] = alpha;
            continue;
        }
        // tau = 2 / (v^T · v) = (akk - alpha) / (-alpha)
        let tau_k = diff / (-alpha);
        check_finite(tau_k, &format!("tau_k QR k={k}"))?;
        tau[k] = tau_k;

        // Normalise v[1..] dans la colonne k
        for i in (k + 1)..m
        {
            let normalized = a[(i, k)] / diff;
            check_finite(normalized, &format!("v_norm QR k={k},i={i}"))?;
            a[(i, k)] = normalized;
        }
        a[(k, k)] = alpha;

        // Applique H_k = I - tau · v · v^T aux colonnes k+1..n
        for j in (k + 1)..n
        {
            let mut dot = a[(k, j)];
            for i in (k + 1)..m
            {
                dot += a[(i, k)] * a[(i, j)];
            }
            let s = tau_k * dot;
            check_finite(s, &format!("s QR k={k},j={j}"))?;
            a[(k, j)] -= s;
            for i in (k + 1)..m
            {
                let updated = a[(i, j)] - s * a[(i, k)];
                check_finite(updated, &format!("a_upd QR k={k},i={i},j={j}"))?;
                a[(i, j)] = updated;
            }
        }
    }

    Ok(Qr { data: a, tau, m, n })
}

/// Résout le système des moindres carrés : minimise ||A·x - b||₂.
///
/// - Si m = n et A inversible : solution exacte du système carré
/// - Si m > n : solution au sens des moindres carrés (régression linéaire)
pub fn solve_qr_least_squares(qr: &Qr, b: &[f64]) -> SolverResult<Vec<f64>> {
    if b.len() != qr.m
    {
        return Err(SolverError::DimensionMismatch {
            expected: qr.m,
            got: b.len(),
        });
    }

    // Vérifier que b est fini
    for (i, &bi) in b.iter().enumerate()
    {
        check_finite(bi, &format!("b[{i}]"))?;
    }

    // R n'est pas reçue avec la matrice A d'origine ; sa propre diagonale
    // (la partie triangulaire sup. de `qr.data`) est la référence d'échelle
    // disponible ici pour le test de pivot ci-dessous.
    let max_abs = (0..qr.n.min(qr.m)).fold(0.0f64, |acc, i| acc.max(qr.data[(i, i)].abs()));
    let piv_tol = pivot_tol(qr.n, max_abs);

    // y = Q^T · b (en appliquant les H_k dans l'ordre)
    let mut y = b.to_vec();
    let p = qr.tau.len();
    for k in 0..p
    {
        let tau_k = qr.tau[k];
        if tau_k == 0.0
        {
            continue;
        }
        let mut dot = y[k];
        for i in (k + 1)..qr.m
        {
            dot += qr.data[(i, k)] * y[i];
        }
        let s = tau_k * dot;
        check_finite(s, &format!("s Q^T·b k={k}"))?;
        y[k] -= s;
        for i in (k + 1)..qr.m
        {
            y[i] -= s * qr.data[(i, k)];
        }
    }

    // Résout R · x = y[0..n] par substitution arrière
    let mut x = vec![0.0; qr.n];
    for i in (0..qr.n).rev()
    {
        let mut s = y[i];
        for j in (i + 1)..qr.n
        {
            s -= qr.data[(i, j)] * x[j];
        }
        let pivot = qr.data[(i, i)];
        if pivot.abs() < piv_tol
        {
            warn!(
                target: "solver",
                "QR back-substitution: near-singular pivot {:.3e} at row {} — restoring backup",
                pivot, i
            );
            return Err(SolverError::Singular { row: i, pivot });
        }
        x[i] = s / pivot;
        check_finite(x[i], &format!("x[{i}] QR solve"))?;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn qr_solve_square() -> SolverResult<()> {
        let a = Matrix::from_row_major(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]);
        let b = vec![6.0, 15.0, 25.0];
        let qr = qr_decompose(a.clone())?;
        let x = solve_qr_least_squares(&qr, &b)?;
        let ax = a.matvec(&x)?;
        for (axi, bi) in ax.iter().zip(&b)
        {
            assert_relative_eq!(*axi, *bi, epsilon = 1e-9);
        }
        Ok(())
    }

    #[test]
    fn qr_solve_square_at_a_tiny_physical_scale() -> SolverResult<()> {
        // Regression test for a P1 audit finding: FINITE_EPS was a fixed
        // absolute 1e-15 compared directly against R's diagonal (which scales
        // linearly with A) — the same regular system as `qr_solve_square`
        // above, scaled down, was declared singular even though it is
        // perfectly well-conditioned.
        let scale = 1e-17;
        let a = Matrix::from_row_major(
            3,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
                .into_iter()
                .map(|v| v * scale)
                .collect(),
        );
        let b = vec![6.0 * scale, 15.0 * scale, 25.0 * scale];
        let qr = qr_decompose(a.clone())?;
        let x = solve_qr_least_squares(&qr, &b)?;
        let ax = a.matvec(&x)?;
        for (axi, bi) in ax.iter().zip(&b)
        {
            assert_relative_eq!(*axi, *bi, epsilon = 1e-9, max_relative = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn least_squares_linear_regression() -> SolverResult<()> {
        let xs = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let ys = [1.05_f64, 2.97, 5.02, 6.99, 9.01];
        let mut data = Vec::with_capacity(10);
        for &x in &xs
        {
            data.push(x);
            data.push(1.0);
        }
        let a = Matrix::from_row_major(5, 2, data);
        let qr = qr_decompose(a)?;
        let beta = solve_qr_least_squares(&qr, &ys)?;
        assert_relative_eq!(beta[0], 2.0, epsilon = 0.05);
        assert_relative_eq!(beta[1], 1.0, epsilon = 0.1);
        Ok(())
    }
}

/// LAPACK-style property tests over random matrices, checking the
/// structural invariants of Householder QR (orthogonality, reconstruction)
/// and the least-squares solve's residual, rather than fixed point values.
#[cfg(test)]
mod proptests {
    use super::*;
    use crate::linalg::{Matrix, norm2};
    use proptest::prelude::*;

    proptest! {
        /// Q, reconstructed from the Householder reflections, must be
        /// orthogonal: QᵀQ = I (Golub & Van Loan, §5.2) — true for *any*
        /// matrix (even rank-deficient ones), so no conditioning is forced
        /// here.
        #[test]
        fn q_is_orthogonal(raw in prop::collection::vec(-10.0f64..10.0, 16)) {
            let n = 4;
            let a = Matrix::from_row_major(n, n, raw);
            let qr = qr_decompose(a).unwrap();
            let q = qr.q();
            let qtq = q.transpose().matmul(&q).unwrap();
            let id = Matrix::identity(n);
            for i in 0..n
            {
                for j in 0..n
                {
                    prop_assert!(
                        (qtq[(i, j)] - id[(i, j)]).abs() < 1e-8,
                        "QᵀQ != I at ({i},{j}): {}", qtq[(i, j)]
                    );
                }
            }
        }

        /// Reconstruction: Q·R must equal the original A.
        #[test]
        fn reconstructs_a_as_q_r(raw in prop::collection::vec(-10.0f64..10.0, 16)) {
            let n = 4;
            let a = Matrix::from_row_major(n, n, raw);
            let qr = qr_decompose(a.clone()).unwrap();
            let qr_prod = qr.q().matmul(&qr.r()).unwrap();
            for i in 0..n
            {
                for j in 0..n
                {
                    let tol = 1e-8 * (1.0 + a[(i, j)].abs());
                    prop_assert!(
                        (qr_prod[(i, j)] - a[(i, j)]).abs() < tol,
                        "Q·R != A at ({i},{j}): {} vs {}", qr_prod[(i, j)], a[(i, j)]
                    );
                }
            }
        }

        /// LAPACK-style residual check for the square least-squares solve.
        #[test]
        fn least_squares_residual_is_small_on_square_systems(
            raw in prop::collection::vec(-10.0f64..10.0, 16),
            b in prop::collection::vec(-10.0f64..10.0, 4),
        ) {
            let n = 4;
            // Force diagonal dominance so the system is well-conditioned —
            // otherwise a generic random 4x4 matrix is singular/near-
            // singular often enough to make this property flaky.
            let mut a = Matrix::from_row_major(n, n, raw);
            for i in 0..n
            {
                let off_sum: f64 = (0..n).filter(|&j| j != i).map(|j| a[(i, j)].abs()).sum();
                let sign = if a[(i, i)] < 0.0 { -1.0 } else { 1.0 };
                a[(i, i)] = sign * (a[(i, i)].abs() + off_sum) + sign;
            }
            let qr = qr_decompose(a.clone()).unwrap();
            let x = solve_qr_least_squares(&qr, &b).expect("well-conditioned square solve must succeed");
            let ax = a.matvec(&x).unwrap();
            let b_norm = norm2(&b).max(1e-300);
            let res = ax.iter().zip(&b).map(|(axi, bi)| (axi - bi).powi(2)).sum::<f64>().sqrt();
            prop_assert!(res / b_norm < 1e-7, "relative residual {} too large", res / b_norm);
        }
    }
}
