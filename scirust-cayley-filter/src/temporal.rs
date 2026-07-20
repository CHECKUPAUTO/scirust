//! Deterministic temporal application of 16D Cayley filters.

use crate::operator::{Matrix16, matrix_vector_mul};
use crate::projector::CayleyProjector;
use crate::scalar::{SEDENION_DIMENSION, Sedenion};
use crate::soft::SoftCayleyFilter;

/// Number of scalar samples represented by one sedenion.
pub const TEMPORAL_BLOCK_SIZE: usize = SEDENION_DIMENSION;

/// Length-preserving non-overlapping temporal block filter.
///
/// Each complete group of 16 consecutive samples is interpreted as one
/// sedenion. A final incomplete block is copied unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalBlockFilter {
    transform: Matrix16,
}

impl TemporalBlockFilter {
    /// Constructs a temporal filter from an arbitrary real 16 × 16 transform.
    #[must_use]
    pub const fn new(transform: Matrix16) -> Self {
        Self { transform }
    }

    /// Constructs a temporal filter from a hard Cayley projector.
    #[must_use]
    pub const fn from_hard(projector: &CayleyProjector) -> Self {
        Self::new(*projector.projection())
    }

    /// Constructs a temporal filter from a soft Cayley spectral transform.
    #[must_use]
    pub const fn from_soft(filter: &SoftCayleyFilter) -> Self {
        Self::new(*filter.transform())
    }

    /// Applies the transform independently to every complete 16-sample block.
    ///
    /// The output always has exactly the same length as the input.
    #[must_use]
    pub fn apply(&self, input: &[f64]) -> Vec<f64> {
        let complete_length = input.len() / TEMPORAL_BLOCK_SIZE * TEMPORAL_BLOCK_SIZE;

        let mut output = input.to_vec();

        for offset in (0..complete_length).step_by(TEMPORAL_BLOCK_SIZE)
        {
            let block: Sedenion = core::array::from_fn(|index| input[offset + index]);

            let filtered = matrix_vector_mul(&self.transform, &block);

            output[offset..offset + TEMPORAL_BLOCK_SIZE].copy_from_slice(&filtered);
        }

        output
    }

    /// Returns the underlying row-major transform.
    #[must_use]
    pub const fn transform(&self) -> &Matrix16 {
        &self.transform
    }
}

#[cfg(test)]
mod tests {
    use super::{TEMPORAL_BLOCK_SIZE, TemporalBlockFilter};
    use crate::{CayleyProjector, basis_vector, left_multiplication_matrix};

    #[test]
    fn identity_transform_preserves_every_sample() {
        let identity = left_multiplication_matrix(basis_vector(0).expect("e0 exists"));

        let filter = TemporalBlockFilter::new(identity);

        let input: Vec<f64> = (0..37).map(|index| index as f64 - 18.0).collect();

        assert_eq!(filter.apply(&input), input);
    }

    #[test]
    fn temporal_block_matches_direct_hard_projection() {
        let mut multiplier = [0.0; TEMPORAL_BLOCK_SIZE];
        multiplier[1] = 1.0;
        multiplier[10] = 1.0;

        let projector = CayleyProjector::new(multiplier, 1.0e-12).expect("valid projector");

        let temporal = TemporalBlockFilter::from_hard(&projector);

        let input: Vec<f64> = (0..TEMPORAL_BLOCK_SIZE)
            .map(|index| index as f64 - 7.0)
            .collect();

        let block = core::array::from_fn(|index| input[index]);
        let expected = projector.apply(&block);

        assert_eq!(temporal.apply(&input), expected.to_vec());
    }

    #[test]
    fn incomplete_trailing_block_is_unchanged() {
        let mut multiplier = [0.0; TEMPORAL_BLOCK_SIZE];
        multiplier[1] = 1.0;
        multiplier[10] = 1.0;

        let projector = CayleyProjector::new(multiplier, 1.0e-12).expect("valid projector");

        let temporal = TemporalBlockFilter::from_hard(&projector);

        let input: Vec<f64> = (0..TEMPORAL_BLOCK_SIZE + 5)
            .map(|index| index as f64)
            .collect();

        let output = temporal.apply(&input);

        assert_eq!(
            &output[TEMPORAL_BLOCK_SIZE..],
            &input[TEMPORAL_BLOCK_SIZE..],
        );
    }

    #[test]
    fn empty_input_is_supported() {
        let identity = left_multiplication_matrix(basis_vector(0).expect("e0 exists"));

        let filter = TemporalBlockFilter::new(identity);

        assert!(filter.apply(&[]).is_empty());
    }
}
