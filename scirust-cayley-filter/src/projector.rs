//! Orthogonal projector derived from a Cayley left-multiplication operator.

use core::fmt;

use scirust_solvers::Matrix;
use scirust_solvers::linalg::svd;

use crate::filter::FilterEvaluation;
use crate::operator::{Matrix16, left_multiplication_matrix, matrix_vector_mul};
use crate::scalar::{SEDENION_DIMENSION, Sedenion};

/// Failure while constructing a Cayley projector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProjectorError {
    /// The relative singular-value threshold is invalid.
    InvalidRelativeThreshold,

    /// SciRust's deterministic SVD failed.
    Decomposition(String),
}

impl fmt::Display for ProjectorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidRelativeThreshold =>
            {
                formatter.write_str("relative threshold must be finite and non-negative")
            },
            Self::Decomposition(message) =>
            {
                write!(formatter, "SVD decomposition failed: {message}")
            },
        }
    }
}

impl std::error::Error for ProjectorError {}

/// Orthogonal projector whose rejected subspace is the numerical right
/// quasi-kernel of a sedenion left-multiplication matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct CayleyProjector {
    multiplier: Sedenion,
    projection: Matrix16,
    singular_values: Vec<f64>,
    rejected_dimension: usize,
    relative_threshold: f64,
}

impl CayleyProjector {
    /// Builds the projector from the SVD of `L_a`.
    ///
    /// A right-singular direction is rejected when:
    ///
    /// `sigma <= relative_threshold * sigma_max`.
    pub fn new(multiplier: Sedenion, relative_threshold: f64) -> Result<Self, ProjectorError> {
        if !relative_threshold.is_finite() || relative_threshold < 0.0
        {
            return Err(ProjectorError::InvalidRelativeThreshold);
        }

        let operator = left_multiplication_matrix(multiplier);
        let data = operator
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

        let matrix = Matrix::from_row_major(SEDENION_DIMENSION, SEDENION_DIMENSION, data);

        let decomposition =
            svd(&matrix).map_err(|error| ProjectorError::Decomposition(error.to_string()))?;

        let sigma_max = decomposition.s.first().copied().unwrap_or(0.0);
        let cutoff = relative_threshold * sigma_max;

        let rejected: Vec<usize> = decomposition
            .s
            .iter()
            .enumerate()
            .filter_map(|(index, &sigma)| (sigma_max == 0.0 || sigma <= cutoff).then_some(index))
            .collect();

        let projection = core::array::from_fn(|row| {
            core::array::from_fn(|column| {
                let identity = if row == column { 1.0 } else { 0.0 };

                rejected.iter().fold(identity, |value, &direction| {
                    value - decomposition.v[(row, direction)] * decomposition.v[(column, direction)]
                })
            })
        });

        Ok(Self {
            multiplier,
            projection,
            singular_values: decomposition.s,
            rejected_dimension: rejected.len(),
            relative_threshold,
        })
    }

    /// Applies the orthogonal projection.
    #[must_use]
    pub fn apply(&self, input: &Sedenion) -> Sedenion {
        matrix_vector_mul(&self.projection, input)
    }

    /// Evaluates signal preservation and noise rejection.
    #[must_use]
    pub fn evaluate(&self, signal: &Sedenion, noise: &Sedenion) -> FilterEvaluation {
        FilterEvaluation::from_linear_outputs(signal, noise, self.apply(signal), self.apply(noise))
    }

    #[must_use]
    pub const fn projection(&self) -> &Matrix16 {
        &self.projection
    }

    #[must_use]
    pub fn singular_values(&self) -> &[f64] {
        &self.singular_values
    }

    #[must_use]
    pub const fn rejected_dimension(&self) -> usize {
        self.rejected_dimension
    }

    #[must_use]
    pub const fn multiplier(&self) -> &Sedenion {
        &self.multiplier
    }

    #[must_use]
    pub const fn relative_threshold(&self) -> f64 {
        self.relative_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::{CayleyProjector, ProjectorError};
    use crate::scalar::{SEDENION_DIMENSION, Sedenion, basis_vector, squared_norm};

    const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];
    const THRESHOLD: f64 = 1.0e-12;

    fn squared_distance(left: &Sedenion, right: &Sedenion) -> f64 {
        left.iter().zip(right).fold(0.0, |sum, (a, b)| {
            let difference = a - b;
            sum + difference * difference
        })
    }

    #[test]
    fn full_rank_multiplier_produces_identity_projection() {
        let projector = CayleyProjector::new(basis_vector(0).expect("e0 exists"), THRESHOLD)
            .expect("SVD succeeds");

        assert_eq!(projector.rejected_dimension(), 0);

        let input = basis_vector(7).expect("e7 exists");
        assert_eq!(projector.apply(&input), input);
    }

    #[test]
    fn known_kernel_noise_is_removed_without_signal_distortion() {
        let mut multiplier = ZERO;
        multiplier[1] = 1.0;
        multiplier[10] = 1.0;

        let mut noise = ZERO;
        noise[4] = 1.0;
        noise[15] = -1.0;

        let signal = basis_vector(0).expect("e0 exists");

        let projector = CayleyProjector::new(multiplier, THRESHOLD).expect("SVD succeeds");

        let evaluation = projector.evaluate(&signal, &noise);

        assert!(projector.rejected_dimension() > 0);
        assert!(squared_norm(evaluation.filtered_noise()) < 1.0e-20);
        assert!(squared_distance(evaluation.filtered_signal(), &signal) < 1.0e-20);
    }

    #[test]
    fn projection_is_idempotent() {
        let mut multiplier = ZERO;
        multiplier[1] = 1.0;
        multiplier[10] = 1.0;

        let projector = CayleyProjector::new(multiplier, THRESHOLD).expect("SVD succeeds");

        let input = [
            1.0, -2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -3.0, 1.5, 0.0, -0.5, 2.5, -4.0, 1.0, 3.0, -2.0,
        ];

        let once = projector.apply(&input);
        let twice = projector.apply(&once);

        assert!(squared_distance(&once, &twice) < 1.0e-20);
    }

    #[test]
    fn invalid_threshold_is_rejected() {
        let multiplier = basis_vector(0).expect("e0 exists");

        for threshold in [-1.0, f64::INFINITY, f64::NAN]
        {
            assert_eq!(
                CayleyProjector::new(multiplier, threshold),
                Err(ProjectorError::InvalidRelativeThreshold)
            );
        }
    }
}
