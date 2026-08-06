//! A minimal, dependency-free 3-D scalar field.
//!
//! Mirrors [`crate::field::Field2`] one dimension up: dense row-major storage,
//! validated construction, and the axis convention of the reference NumPy code
//! — index `k` runs along `z` (axis 0), `i` along `y` (axis 1) and `j` along
//! `x` (axis 2), so the flat offset is `(k * ny + i) * nx + j`. Getting this
//! order wrong is the exact "axis permutation" failure mode the region oracle
//! fixtures exist to catch.

use crate::error::{ItdError, Result};

/// A dense row-major 3-D field of `f64` values with `nz` planes, `ny` rows and
/// `nx` columns.
#[derive(Debug, Clone, PartialEq)]
pub struct Field3 {
    nz: usize,
    ny: usize,
    nx: usize,
    data: Vec<f64>,
}

impl Field3 {
    /// Builds a field from row-major data. Fails if `data.len() != nz * ny * nx`
    /// or any dimension is zero.
    ///
    /// # Errors
    ///
    /// [`ItdError::ShapeMismatch`] on a zero dimension or a length that does
    /// not match the declared shape.
    pub fn from_vec(nz: usize, ny: usize, nx: usize, data: Vec<f64>) -> Result<Self> {
        if nz == 0 || ny == 0 || nx == 0
        {
            return Err(ItdError::ShapeMismatch(
                "field dimensions must be non-zero".into(),
            ));
        }
        let expected = nz * ny * nx;
        if data.len() != expected
        {
            return Err(ItdError::ShapeMismatch(format!(
                "expected {expected} values for {nz}x{ny}x{nx}, got {}",
                data.len()
            )));
        }
        Ok(Self { nz, ny, nx, data })
    }

    /// Builds a field by evaluating `f(k, i, j)` at every cell, in row-major
    /// order (`z` outermost, `x` innermost).
    pub fn from_fn<F: FnMut(usize, usize, usize) -> f64>(
        nz: usize,
        ny: usize,
        nx: usize,
        mut f: F,
    ) -> Self {
        let mut data = Vec::with_capacity(nz * ny * nx);
        for k in 0..nz
        {
            for i in 0..ny
            {
                for j in 0..nx
                {
                    data.push(f(k, i, j));
                }
            }
        }
        Self { nz, ny, nx, data }
    }

    /// A field of zeros.
    pub fn zeros(nz: usize, ny: usize, nx: usize) -> Self {
        Self {
            nz,
            ny,
            nx,
            data: vec![0.0; nz * ny * nx],
        }
    }

    /// Number of planes (the `z`-axis / axis-0 length).
    #[inline]
    pub fn nz(&self) -> usize {
        self.nz
    }

    /// Number of rows (the `y`-axis / axis-1 length).
    #[inline]
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Number of columns (the `x`-axis / axis-2 length).
    #[inline]
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// The `(nz, ny, nx)` shape.
    #[inline]
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.nz, self.ny, self.nx)
    }

    /// Value at plane `k`, row `i`, column `j`.
    #[inline]
    pub fn get(&self, k: usize, i: usize, j: usize) -> f64 {
        self.data[(k * self.ny + i) * self.nx + j]
    }

    /// Mutable reference to the value at plane `k`, row `i`, column `j`.
    #[inline]
    pub fn get_mut(&mut self, k: usize, i: usize, j: usize) -> &mut f64 {
        &mut self.data[(k * self.ny + i) * self.nx + j]
    }

    /// The underlying row-major data.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// `true` when every value is finite.
    pub fn all_finite(&self) -> bool {
        self.data.iter().all(|v| v.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec_rejects_zero_dimensions_and_wrong_lengths() {
        assert!(matches!(
            Field3::from_vec(0, 2, 2, vec![]),
            Err(ItdError::ShapeMismatch(_))
        ));
        assert!(matches!(
            Field3::from_vec(2, 2, 2, vec![0.0; 7]),
            Err(ItdError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn layout_is_z_outermost_x_innermost() {
        let field = Field3::from_fn(2, 3, 4, |k, i, j| (k * 100 + i * 10 + j) as f64);
        assert_eq!(field.get(0, 0, 0), 0.0);
        assert_eq!(field.get(0, 0, 3), 3.0);
        assert_eq!(field.get(0, 2, 1), 21.0);
        assert_eq!(field.get(1, 2, 3), 123.0);
        // Flat offset of (k=1, i=0, j=0) is ny * nx = 12.
        assert_eq!(field.as_slice()[12], 100.0);
    }

    #[test]
    fn all_finite_flags_nan_and_infinity() {
        let mut field = Field3::zeros(2, 2, 2);
        assert!(field.all_finite());
        *field.get_mut(1, 0, 1) = f64::NAN;
        assert!(!field.all_finite());
    }
}
