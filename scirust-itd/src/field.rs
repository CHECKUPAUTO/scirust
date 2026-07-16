//! A minimal, dependency-free 2-D scalar field.

use crate::error::{ItdError, Result};

/// A dense row-major 2-D field of `f64` values with `ny` rows and `nx`
/// columns. Row index `i` runs along the `y` axis (axis 0 in the reference
/// NumPy code); column index `j` runs along the `x` axis (axis 1).
#[derive(Debug, Clone, PartialEq)]
pub struct Field2 {
    ny: usize,
    nx: usize,
    data: Vec<f64>,
}

impl Field2 {
    /// Builds a field from row-major data. Fails if `data.len() != ny * nx` or
    /// either dimension is zero.
    pub fn from_vec(ny: usize, nx: usize, data: Vec<f64>) -> Result<Self> {
        if ny == 0 || nx == 0
        {
            return Err(ItdError::ShapeMismatch(
                "field dimensions must be non-zero".into(),
            ));
        }
        if data.len() != ny * nx
        {
            return Err(ItdError::ShapeMismatch(format!(
                "expected {} values for {ny}x{nx}, got {}",
                ny * nx,
                data.len()
            )));
        }
        Ok(Self { ny, nx, data })
    }

    /// Builds a field by evaluating `f(i, j)` at every cell.
    pub fn from_fn<F: FnMut(usize, usize) -> f64>(ny: usize, nx: usize, mut f: F) -> Self {
        let mut data = Vec::with_capacity(ny * nx);
        for i in 0..ny
        {
            for j in 0..nx
            {
                data.push(f(i, j));
            }
        }
        Self { ny, nx, data }
    }

    /// A field of zeros.
    pub fn zeros(ny: usize, nx: usize) -> Self {
        Self {
            ny,
            nx,
            data: vec![0.0; ny * nx],
        }
    }

    /// Number of rows (the `y`-axis / axis-0 length).
    #[inline]
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Number of columns (the `x`-axis / axis-1 length).
    #[inline]
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// The `(ny, nx)` shape.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.ny, self.nx)
    }

    /// Value at row `i`, column `j`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.nx + j]
    }

    /// Mutable value at row `i`, column `j`.
    #[inline]
    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut f64 {
        &mut self.data[i * self.nx + j]
    }

    /// The underlying row-major slice.
    #[inline]
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// True if every value is finite.
    pub fn all_finite(&self) -> bool {
        self.data.iter().all(|v| v.is_finite())
    }

    /// A new field with `f` applied to every value.
    pub fn map<F: FnMut(f64) -> f64>(&self, mut f: F) -> Field2 {
        Field2 {
            ny: self.ny,
            nx: self.nx,
            data: self.data.iter().map(|&v| f(v)).collect(),
        }
    }

    /// A new field combining `self` and `other` element-wise with `f`.
    /// Fails if the shapes differ.
    pub fn zip_map<F: FnMut(f64, f64) -> f64>(&self, other: &Field2, mut f: F) -> Result<Field2> {
        if self.shape() != other.shape()
        {
            return Err(ItdError::ShapeMismatch(format!(
                "{:?} vs {:?}",
                self.shape(),
                other.shape()
            )));
        }
        Ok(Field2 {
            ny: self.ny,
            nx: self.nx,
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| f(a, b))
                .collect(),
        })
    }
}
