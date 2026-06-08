//! N-dimensional tensor for offline use (no autograd integration).
//!
//! `TensorND` is used inside the TT-SVD algorithm where we need to express
//! tensors of arbitrary rank as their unfoldings. Internally the data is a
//! flat `Vec<f32>` in row-major (C) order.
//!
//! Row-major linearization: for shape `[s_0, s_1, ..., s_{d-1}]`, the element
//! at multi-index `(i_0, i_1, ..., i_{d-1})` is at flat position
//! `i_0 * (s_1 * s_2 * ... * s_{d-1}) + i_1 * (s_2 * ... * s_{d-1}) + ... + i_{d-1}`.

use std::fmt;

#[derive(Clone)]
pub struct TensorND {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl TensorND {
    /// Construct from raw shape and row-major data. Panics if `data.len()`
    /// does not match the product of the shape.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(
            n,
            data.len(),
            "TensorND::new: shape product {n} != data.len() {}",
            data.len()
        );
        Self { shape, data }
    }

    /// Zero tensor of given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self { shape, data: vec![0.0; n] }
    }

    /// Construct from a 2D matrix in row-major order: `data[i * cols + j]` = element `(i, j)`.
    pub fn from_matrix(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        Self::new(vec![rows, cols], data)
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Mode-`k` unfolding: rows = product of `shape[..k]`, cols = product of `shape[k..]`.
    /// Zero-copy because row-major data preserves the unfolding layout.
    ///
    /// Example: shape `[2, 3, 4]`, unfold at `k=1` gives a `(2, 12)` matrix.
    pub fn unfold(&self, k: usize) -> (usize, usize, Vec<f32>) {
        assert!(k <= self.shape.len(), "unfold index {k} out of bounds");
        let rows: usize = self.shape[..k].iter().product::<usize>().max(1);
        let cols: usize = self.shape[k..].iter().product::<usize>().max(1);
        debug_assert_eq!(rows * cols, self.data.len());
        (rows, cols, self.data.clone())
    }

    /// Reshape to a new shape. Total element count must match. Zero-cost,
    /// the underlying data is preserved unchanged.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let n: usize = new_shape.iter().product();
        assert_eq!(
            n,
            self.data.len(),
            "reshape: new shape product {n} != current numel {}",
            self.data.len()
        );
        Self { shape: new_shape, data: self.data.clone() }
    }

    /// Maximum absolute element value (used for tolerance checks).
    pub fn abs_max(&self) -> f32 {
        self.data.iter().fold(0.0_f32, |acc, &x| acc.max(x.abs()))
    }

    /// Frobenius norm: sqrt(sum of squares).
    pub fn frob_norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

impl fmt::Debug for TensorND {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TensorND {{ shape: {:?}, numel: {}, frob: {:.4e} }}",
            self.shape,
            self.numel(),
            self.frob_norm()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_shape() {
        let t = TensorND::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn test_reshape() {
        let t = TensorND::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = t.reshape(vec![6]);
        assert_eq!(r.shape, vec![6]);
        assert_eq!(r.data, t.data);
    }

    #[test]
    fn test_unfold_3d() {
        // shape [2, 3, 4] = 24 elements, unfold at k=1 gives (2, 12)
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = TensorND::new(vec![2, 3, 4], data.clone());
        let (rows, cols, flat) = t.unfold(1);
        assert_eq!((rows, cols), (2, 12));
        assert_eq!(flat, data); // row-major preserves layout
    }

    #[test]
    fn test_frob_norm() {
        let t = TensorND::new(vec![2, 2], vec![3.0, 4.0, 0.0, 0.0]);
        assert!((t.frob_norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "shape product")]
    fn test_new_size_mismatch() {
        TensorND::new(vec![2, 3], vec![1.0; 5]); // 5 != 6
    }
}
