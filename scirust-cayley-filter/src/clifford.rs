//! Associative split-Clifford projectors on R^16.

use crate::filter::FilterEvaluation;
use crate::operator::{Matrix16, matrix_vector_mul};
use crate::scalar::{SEDENION_DIMENSION, Sedenion};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CliffordProjectorError {
    InvalidRejectedDimension,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SplitCliffordProjector {
    involution: Matrix16,
    projection: Matrix16,
    rejected_dimension: usize,
}

impl SplitCliffordProjector {
    pub fn canonical(rejected_dimension: usize) -> Result<Self, CliffordProjectorError> {
        if rejected_dimension > SEDENION_DIMENSION
        {
            return Err(CliffordProjectorError::InvalidRejectedDimension);
        }

        let involution = core::array::from_fn(|row| {
            core::array::from_fn(|column| {
                if row != column
                {
                    0.0
                }
                else if row < rejected_dimension
                {
                    -1.0
                }
                else
                {
                    1.0
                }
            })
        });

        let projection = core::array::from_fn(|row| {
            core::array::from_fn(|column| {
                let identity = if row == column { 1.0 } else { 0.0 };
                0.5 * (identity + involution[row][column])
            })
        });

        Ok(Self {
            involution,
            projection,
            rejected_dimension,
        })
    }

    #[must_use]
    pub fn apply(&self, input: &Sedenion) -> Sedenion {
        matrix_vector_mul(&self.projection, input)
    }

    #[must_use]
    pub fn evaluate(&self, signal: &Sedenion, noise: &Sedenion) -> FilterEvaluation {
        FilterEvaluation::from_linear_outputs(signal, noise, self.apply(signal), self.apply(noise))
    }

    #[must_use]
    pub const fn involution(&self) -> &Matrix16 {
        &self.involution
    }

    #[must_use]
    pub const fn projection(&self) -> &Matrix16 {
        &self.projection
    }

    #[must_use]
    pub const fn rejected_dimension(&self) -> usize {
        self.rejected_dimension
    }
}

#[cfg(test)]
mod tests {
    use super::{CliffordProjectorError, SplitCliffordProjector};
    use crate::{SEDENION_DIMENSION, basis_vector, matrix_vector_mul};

    #[test]
    fn canonical_projector_rejects_expected_subspace() {
        let projector = SplitCliffordProjector::canonical(4).unwrap();

        for index in 0..SEDENION_DIMENSION
        {
            let input = basis_vector(index).unwrap();
            let output = projector.apply(&input);

            if index < 4
            {
                assert_eq!(output, [0.0; SEDENION_DIMENSION]);
            }
            else
            {
                assert_eq!(output, input);
            }
        }
    }

    #[test]
    fn projection_is_idempotent_and_involution_squares_to_identity() {
        let projector = SplitCliffordProjector::canonical(8).unwrap();

        for index in 0..SEDENION_DIMENSION
        {
            let input = basis_vector(index).unwrap();
            let once = projector.apply(&input);
            assert_eq!(projector.apply(&once), once);

            let transformed = matrix_vector_mul(projector.involution(), &input);
            assert_eq!(
                matrix_vector_mul(projector.involution(), &transformed),
                input
            );
        }
    }

    #[test]
    fn invalid_dimension_is_rejected() {
        assert_eq!(
            SplitCliffordProjector::canonical(17),
            Err(CliffordProjectorError::InvalidRejectedDimension)
        );
    }
}
