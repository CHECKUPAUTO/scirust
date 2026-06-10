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

fn check_finite(value: f64, location: &str) -> Result<(), SolverError> {
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
        if pivot.abs() < FINITE_EPS
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
