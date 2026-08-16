extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::{ComputeError, ComputeResult, Shape};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Strides(Vec<usize>);

impl Strides {
    /// Construct an explicit element-stride vector.
    ///
    /// This is intentionally a metadata constructor: layout validity depends on
    /// the tensor shape/storage pairing and is checked by the tensor layer. Zero
    /// strides are permitted so broadcast views can be represented without a
    /// copy.
    pub fn new(values: impl Into<Vec<usize>>) -> Self {
        Self(values.into())
    }

    pub fn contiguous(shape: &Shape) -> ComputeResult<Self> {
        let rank = shape.rank();
        let mut values = vec![1usize; rank];

        if rank > 1
        {
            for axis in (0..rank - 1).rev()
            {
                values[axis] = values[axis + 1]
                    .checked_mul(shape.dims()[axis + 1])
                    .ok_or(ComputeError::ShapeOverflow)?;
            }
        }

        Ok(Self(values))
    }

    pub fn values(&self) -> &[usize] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_strides_are_row_major() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(Strides::contiguous(&shape).unwrap().values(), &[12, 4, 1]);
    }

    #[test]
    fn scalar_strides_are_empty() {
        assert!(
            Strides::contiguous(&Shape::scalar())
                .unwrap()
                .values()
                .is_empty()
        );
    }

    #[test]
    fn stride_overflow_is_rejected() {
        let shape = Shape::new(vec![2, usize::MAX, 2]);
        assert_eq!(
            Strides::contiguous(&shape),
            Err(ComputeError::ShapeOverflow)
        );
    }

    #[test]
    fn explicit_strides_preserve_zero_for_broadcast_views() {
        let strides = Strides::new(vec![0, 4, 1]);
        assert_eq!(strides.values(), &[0, 4, 1]);
    }
}
