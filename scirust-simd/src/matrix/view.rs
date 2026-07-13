//! Checked row-major matrix views used by the SIMD kernels.

/// Error returned when a matrix shape cannot describe the supplied storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixViewError {
    /// `rows * cols` cannot be represented by `usize`.
    ShapeOverflow { rows: usize, cols: usize },
    /// The backing slice does not contain exactly `rows * cols` elements.
    LengthMismatch { expected: usize, actual: usize },
}

impl core::fmt::Display for MatrixViewError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self
        {
            Self::ShapeOverflow { rows, cols } =>
            {
                write!(f, "matrix shape {rows}x{cols} overflows usize")
            },
            Self::LengthMismatch { expected, actual } => write!(
                f,
                "matrix storage has {actual} elements, expected exactly {expected}"
            ),
        }
    }
}

impl std::error::Error for MatrixViewError {}

fn checked_len(rows: usize, cols: usize) -> Result<usize, MatrixViewError> {
    rows.checked_mul(cols)
        .ok_or(MatrixViewError::ShapeOverflow { rows, cols })
}

#[derive(Debug, Clone)]
pub struct MatrixView<'a, T> {
    data: &'a [T],
    rows: usize,
    cols: usize,
}

impl<'a, T> MatrixView<'a, T> {
    /// Construct a checked row-major view.
    ///
    /// # Panics
    ///
    /// Panics when the shape overflows or does not match `data.len()`. Use
    /// [`MatrixView::try_new`] when dimensions come from untrusted input.
    pub fn new(data: &'a [T], rows: usize, cols: usize) -> Self {
        Self::try_new(data, rows, cols).expect("invalid matrix view")
    }

    /// Try to construct a row-major view without panicking.
    pub fn try_new(data: &'a [T], rows: usize, cols: usize) -> Result<Self, MatrixViewError> {
        let expected = checked_len(rows, cols)?;
        if data.len() != expected
        {
            return Err(MatrixViewError::LengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { data, rows, cols })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn row_slice(&self, i: usize) -> Option<&[T]> {
        if i >= self.rows
        {
            return None;
        }
        let start = i.checked_mul(self.cols)?;
        let end = start.checked_add(self.cols)?;
        self.data.get(start..end)
    }

    /// Return the element at `(row, col)`, or `None` when either index is out
    /// of bounds.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.rows || col >= self.cols
        {
            return None;
        }
        let index = row.checked_mul(self.cols)?.checked_add(col)?;
        self.data.get(index)
    }
}

#[derive(Debug)]
pub struct MatrixViewMut<'a, T> {
    data: &'a mut [T],
    rows: usize,
    cols: usize,
}

impl<'a, T> MatrixViewMut<'a, T> {
    /// Construct a checked mutable row-major view.
    ///
    /// # Panics
    ///
    /// Panics when the shape overflows or does not match `data.len()`. Use
    /// [`MatrixViewMut::try_new`] when dimensions come from untrusted input.
    pub fn new(data: &'a mut [T], rows: usize, cols: usize) -> Self {
        Self::try_new(data, rows, cols).expect("invalid mutable matrix view")
    }

    /// Try to construct a mutable row-major view without panicking.
    pub fn try_new(data: &'a mut [T], rows: usize, cols: usize) -> Result<Self, MatrixViewError> {
        let expected = checked_len(rows, cols)?;
        if data.len() != expected
        {
            return Err(MatrixViewError::LengthMismatch {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { data, rows, cols })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Row-major read access to row `i` (contiguous `cols` elements).
    pub fn row_slice(&self, i: usize) -> Option<&[T]> {
        if i >= self.rows
        {
            return None;
        }
        let start = i.checked_mul(self.cols)?;
        let end = start.checked_add(self.cols)?;
        self.data.get(start..end)
    }

    /// Row-major mutable access to row `i` (contiguous `cols` elements).
    pub fn row_slice_mut(&mut self, i: usize) -> Option<&mut [T]> {
        if i >= self.rows
        {
            return None;
        }
        let start = i.checked_mul(self.cols)?;
        let end = start.checked_add(self.cols)?;
        self.data.get_mut(start..end)
    }

    /// Return mutable access to `(row, col)`, or `None` when either index is
    /// out of bounds.
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row >= self.rows || col >= self.cols
        {
            return None;
        }
        let index = row.checked_mul(self.cols)?.checked_add(col)?;
        self.data.get_mut(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_storage_and_overflowing_shapes() {
        assert_eq!(
            MatrixView::try_new(&[1, 2, 3], 2, 2).unwrap_err(),
            MatrixViewError::LengthMismatch {
                expected: 4,
                actual: 3,
            }
        );
        assert!(matches!(
            MatrixView::<u8>::try_new(&[], usize::MAX, 2),
            Err(MatrixViewError::ShapeOverflow { .. })
        ));
    }

    #[test]
    fn checked_access_never_aliases_out_of_bounds_indices() {
        let view = MatrixView::new(&[1, 2, 3, 4], 2, 2);
        assert_eq!(view.row_slice(1), Some(&[3, 4][..]));
        assert_eq!(view.row_slice(2), None);
        assert_eq!(view.get(0, 2), None);

        let mut data = [1, 2, 3, 4];
        let mut view = MatrixViewMut::new(&mut data, 2, 2);
        *view.get_mut(1, 0).unwrap() = 9;
        assert_eq!(view.row_slice(1), Some(&[9, 4][..]));
        assert_eq!(view.row_slice_mut(2), None);
    }
}
