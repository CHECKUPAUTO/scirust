//! Algèbre linéaire dense f64. Tout est row-major, owned.
//!
//! Pour les opérations de bas-niveau on s'appuie sur le backend matriciel
//! de `scirust-core` (gemv/gemm/cholesky), mais on garde ici un type
//! `Matrix` simple et autonome avec son propre code LU/QR pour ne pas
//! introduire de dépendance circulaire avec `scirust-core`.

use crate::{SolverError, SolverResult};
use std::fmt;

pub mod bicgstab;
pub mod cholesky;
pub mod eigen;
pub mod gmres;
pub mod iterative;
pub mod lu;
pub mod precond;
pub mod qr;
pub mod svd;

pub use bicgstab::{bicgstab, bicgstab_preconditioned};
pub use cholesky::{cholesky_decompose, solve_cholesky};
pub use eigen::{EigenSymmetric, eigen_symmetric};
pub use gmres::{gmres, gmres_preconditioned};
pub use iterative::conjugate_gradient;
pub use lu::{Lu, lu_decompose, solve, solve_lu};
pub use precond::{JacobiPreconditioner, identity_precond};
pub use qr::{Qr, qr_decompose, solve_qr_least_squares};
pub use svd::{Svd, svd};

// ─── Matrix dense row-major ─────────────────────────────────────────────────

/// Matrice dense f64 en stockage row-major contigu.
///
/// Pour les opérations modifiant la structure (transpose, swap_rows, etc.)
/// on travaille en place autant que possible.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// Crée une matrice depuis un Vec ligne-par-ligne (row-major).
    /// Panique si `data.len() != rows*cols`.
    pub fn from_row_major(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "data size mismatch");
        Self { rows, cols, data }
    }

    /// Crée une matrice depuis une lambda `(i,j) -> f64`.
    pub fn from_fn<F: FnMut(usize, usize) -> f64>(rows: usize, cols: usize, mut f: F) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows
        {
            for j in 0..cols
            {
                data.push(f(i, j));
            }
        }
        Self { rows, cols, data }
    }

    /// Matrice nulle.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Matrice identité n×n.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n
        {
            m[(i, i)] = 1.0;
        }
        m
    }

    /// Crée un vecteur colonne (n × 1).
    pub fn from_col_vec(v: &[f64]) -> Self {
        Self {
            rows: v.len(),
            cols: 1,
            data: v.to_vec(),
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }
    pub fn data(&self) -> &[f64] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Accès direct ligne sous forme slice.
    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [f64] {
        let cols = self.cols;
        &mut self.data[i * cols..(i + 1) * cols]
    }

    pub fn swap_rows(&mut self, i: usize, j: usize) {
        if i == j
        {
            return;
        }
        let cols = self.cols;
        for k in 0..cols
        {
            self.data.swap(i * cols + k, j * cols + k);
        }
    }

    /// Transposée (alloue une nouvelle matrice).
    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows
        {
            for j in 0..self.cols
            {
                out[(j, i)] = self[(i, j)];
            }
        }
        out
    }

    /// Produit matriciel naïf O(n³).
    pub fn matmul(&self, other: &Self) -> SolverResult<Self> {
        if self.cols != other.rows
        {
            return Err(SolverError::DimensionMismatch {
                expected: self.cols,
                got: other.rows,
            });
        }
        let mut out = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows
        {
            for k in 0..self.cols
            {
                let aik = self[(i, k)];
                if aik == 0.0
                {
                    continue;
                }
                for j in 0..other.cols
                {
                    out[(i, j)] += aik * other[(k, j)];
                }
            }
        }
        Ok(out)
    }

    /// `y = A * x` pour x un slice ; vérifie les dimensions.
    pub fn matvec(&self, x: &[f64]) -> SolverResult<Vec<f64>> {
        if self.cols != x.len()
        {
            return Err(SolverError::DimensionMismatch {
                expected: self.cols,
                got: x.len(),
            });
        }
        let mut y = vec![0.0; self.rows];
        for i in 0..self.rows
        {
            let mut acc = 0.0;
            let row = self.row(i);
            for j in 0..self.cols
            {
                acc += row[j] * x[j];
            }
            y[i] = acc;
        }
        Ok(y)
    }

    /// Norme de Frobenius.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Vérifie qu'une matrice est carrée, sinon erreur.
    pub fn ensure_square(&self) -> SolverResult<usize> {
        if self.rows != self.cols
        {
            return Err(SolverError::NotSquare {
                rows: self.rows,
                cols: self.cols,
            });
        }
        Ok(self.rows)
    }

    /// Déterminant via LU (signe inclus). Renvoie 0.0 si singulière.
    pub fn determinant(&self) -> SolverResult<f64> {
        let n = self.ensure_square()?;
        match lu_decompose(self.clone())
        {
            Ok(lu) =>
            {
                let mut det = if lu.swap_count % 2 == 0 { 1.0 } else { -1.0 };
                for i in 0..n
                {
                    det *= lu.lu[(i, i)];
                }
                Ok(det)
            },
            Err(SolverError::Singular { .. }) => Ok(0.0),
            Err(e) => Err(e),
        }
    }

    /// Inverse par LU + résolution colonne par colonne (n résolutions).
    pub fn inverse(&self) -> SolverResult<Self> {
        let n = self.ensure_square()?;
        let lu = lu_decompose(self.clone())?;
        let mut inv = Self::zeros(n, n);
        let mut e = vec![0.0; n];
        for j in 0..n
        {
            e.fill(0.0);
            e[j] = 1.0;
            let x = solve_lu(&lu, &e)?;
            for i in 0..n
            {
                inv[(i, j)] = x[i];
            }
        }
        Ok(inv)
    }
}

// ─── Indexation ─────────────────────────────────────────────────────────────

impl std::ops::Index<(usize, usize)> for Matrix {
    type Output = f64;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i * self.cols + j]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Matrix {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i * self.cols + j]
    }
}

// ─── Display ────────────────────────────────────────────────────────────────

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows
        {
            write!(f, "[")?;
            for j in 0..self.cols
            {
                if j > 0
                {
                    write!(f, ", ")?;
                }
                write!(f, "{:>10.4}", self[(i, j)])?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

// ─── Helpers vecteurs libres ────────────────────────────────────────────────

/// Norme L2 d'un vecteur.
#[inline]
pub fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Norme L∞ (max absolu) d'un vecteur.
#[inline]
pub fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |a, &x| a.max(x.abs()))
}

/// Produit scalaire de deux vecteurs (même taille).
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// `y += alpha * x`.
#[inline]
pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x)
    {
        *yi += alpha * xi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_fn_identity() {
        let m = Matrix::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        assert_eq!(m.shape(), (3, 3));
        for i in 0..3
        {
            for j in 0..3
            {
                assert_eq!(m[(i, j)], if i == j { 1.0 } else { 0.0 });
            }
        }
    }

    #[test]
    fn test_from_fn_hilbert() {
        let n = 4;
        let h = Matrix::from_fn(n, n, |i, j| 1.0 / (i + j + 1) as f64);
        // H[0,0] = 1/(0+0+1) = 1.0
        assert!((h[(0, 0)] - 1.0).abs() < 1e-15);
        // H[3,3] = 1/(3+3+1) = 1/7
        assert!((h[(3, 3)] - 1.0 / 7.0).abs() < 1e-15);
        // H[0,3] = 1/(0+3+1) = 1/4
        assert!((h[(0, 3)] - 0.25).abs() < 1e-15);
    }
}
