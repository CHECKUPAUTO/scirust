//! Minimal dense `f64` linear algebra (row-major), deterministic and pure Rust.
//!
//! Just enough for small Kalman/EKF state spaces: multiply, transpose, add/sub,
//! matrix-vector, and a Gauss–Jordan inverse with partial pivoting. Operations
//! accumulate in a fixed order, so results are bit-reproducible run to run.

use serde::{Deserialize, Serialize};

/// A dense row-major `rows × cols` matrix of `f64`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    /// New matrix from row-major data; panics if `data.len() != rows*cols`.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "Mat::new size mismatch");
        Self { rows, cols, data }
    }

    /// `rows × cols` zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// `n × n` identity.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n
        {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    /// Diagonal matrix from `d`.
    pub fn diag(d: &[f64]) -> Self {
        let n = d.len();
        let mut m = Self::zeros(n, n);
        for (i, &v) in d.iter().enumerate()
        {
            m.data[i * n + i] = v;
        }
        m
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    /// Transpose.
    pub fn t(&self) -> Mat {
        let mut out = Mat::zeros(self.cols, self.rows);
        for i in 0..self.rows
        {
            for j in 0..self.cols
            {
                out.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        out
    }

    /// Matrix product `self · other`.
    pub fn matmul(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.rows, "matmul dim mismatch");
        let mut out = Mat::zeros(self.rows, other.cols);
        for i in 0..self.rows
        {
            for k in 0..self.cols
            {
                let a = self.data[i * self.cols + k];
                for j in 0..other.cols
                {
                    out.data[i * other.cols + j] += a * other.data[k * other.cols + j];
                }
            }
        }
        out
    }

    /// Element-wise `self + other`.
    pub fn add(&self, other: &Mat) -> Mat {
        assert!(
            self.rows == other.rows && self.cols == other.cols,
            "add dim"
        );
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Mat::new(self.rows, self.cols, data)
    }

    /// Element-wise `self - other`.
    pub fn sub(&self, other: &Mat) -> Mat {
        assert!(
            self.rows == other.rows && self.cols == other.cols,
            "sub dim"
        );
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a - b)
            .collect();
        Mat::new(self.rows, self.cols, data)
    }

    /// Matrix times column vector.
    pub fn matvec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols, v.len(), "matvec dim mismatch");
        (0..self.rows)
            .map(|i| {
                let row = &self.data[i * self.cols..(i + 1) * self.cols];
                row.iter().zip(v).map(|(a, b)| a * b).sum()
            })
            .collect()
    }

    /// Inverse via Gauss–Jordan elimination with partial pivoting; `None` if
    /// singular (pivot below `1e-12`).
    pub fn inverse(&self) -> Option<Mat> {
        assert_eq!(self.rows, self.cols, "inverse needs square matrix");
        let n = self.rows;
        // Augment [A | I].
        let mut a = vec![0.0; n * 2 * n];
        for i in 0..n
        {
            for j in 0..n
            {
                a[i * 2 * n + j] = self.data[i * n + j];
            }
            a[i * 2 * n + n + i] = 1.0;
        }
        for col in 0..n
        {
            // Partial pivot: largest |value| in this column at/under the diagonal.
            let mut piv = col;
            let mut best = a[col * 2 * n + col].abs();
            for r in (col + 1)..n
            {
                let v = a[r * 2 * n + col].abs();
                if v > best
                {
                    best = v;
                    piv = r;
                }
            }
            if best < 1e-12
            {
                return None;
            }
            if piv != col
            {
                for k in 0..2 * n
                {
                    a.swap(col * 2 * n + k, piv * 2 * n + k);
                }
            }
            let pivot = a[col * 2 * n + col];
            for k in 0..2 * n
            {
                a[col * 2 * n + k] /= pivot;
            }
            for r in 0..n
            {
                if r != col
                {
                    let factor = a[r * 2 * n + col];
                    if factor != 0.0
                    {
                        for k in 0..2 * n
                        {
                            a[r * 2 * n + k] -= factor * a[col * 2 * n + k];
                        }
                    }
                }
            }
        }
        let mut inv = Mat::zeros(n, n);
        for i in 0..n
        {
            for j in 0..n
            {
                inv.data[i * n + j] = a[i * 2 * n + n + j];
            }
        }
        Some(inv)
    }

    /// Lower-triangular Cholesky factor `L` with `L·Lᵀ = self` for a symmetric
    /// positive-definite matrix; `None` if not positive-definite.
    pub fn cholesky(&self) -> Option<Mat> {
        assert_eq!(self.rows, self.cols, "cholesky needs square matrix");
        let n = self.rows;
        let mut l = Mat::zeros(n, n);
        for i in 0..n
        {
            for j in 0..=i
            {
                let mut sum = self.data[i * n + j];
                for k in 0..j
                {
                    sum -= l.data[i * n + k] * l.data[j * n + k];
                }
                if i == j
                {
                    if sum <= 0.0
                    {
                        return None;
                    }
                    l.data[i * n + j] = sum.sqrt();
                }
                else
                {
                    l.data[i * n + j] = sum / l.data[j * n + j];
                }
            }
        }
        Some(l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_and_transpose() {
        let a = Mat::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Mat::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
        assert_eq!(a.t().data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn inverse_round_trips_to_identity() {
        let a = Mat::new(3, 3, vec![2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 0.0]);
        let inv = a.inverse().expect("nonsingular");
        let prod = a.matmul(&inv);
        let id = Mat::identity(3);
        for (p, q) in prod.data.iter().zip(&id.data)
        {
            assert!((p - q).abs() < 1e-10, "{p} != {q}");
        }
    }

    #[test]
    fn singular_matrix_has_no_inverse() {
        let a = Mat::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
        assert!(a.inverse().is_none());
    }
}
