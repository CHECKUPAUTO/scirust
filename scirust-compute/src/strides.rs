extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::{ComputeError, ComputeResult, Shape};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Strides(Vec<usize>);

impl Strides {
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
}
