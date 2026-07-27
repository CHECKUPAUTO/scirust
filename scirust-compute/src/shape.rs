extern crate alloc;

use alloc::vec::Vec;

use crate::{ComputeError, ComputeResult};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self(dims.into())
    }

    pub fn scalar() -> Self {
        Self(Vec::new())
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn checked_num_elements(&self) -> ComputeResult<usize> {
        self.0.iter().try_fold(1usize, |total, &dim| {
            total.checked_mul(dim).ok_or(ComputeError::ShapeOverflow)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_contains_one_element() {
        assert_eq!(Shape::scalar().checked_num_elements(), Ok(1));
    }

    #[test]
    fn shape_overflow_is_rejected() {
        assert_eq!(
            Shape::new(vec![usize::MAX, 2]).checked_num_elements(),
            Err(ComputeError::ShapeOverflow)
        );
    }
}
