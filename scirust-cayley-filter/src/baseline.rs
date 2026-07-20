//! Ordinary real-linear reference filters.
//!
//! These baselines prevent attributing to sedenions a result already obtained
//! by a simpler linear operator.

use core::fmt;

use crate::filter::FilterEvaluation;
use crate::scalar::{Sedenion, squared_norm};

/// Identity baseline: the input is returned unchanged.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct IdentityFilter;

impl IdentityFilter {
    /// Returns the input unchanged.
    #[must_use]
    pub const fn apply(&self, input: &Sedenion) -> Sedenion {
        *input
    }

    /// Evaluates the identity baseline.
    #[must_use]
    pub fn evaluate(&self, signal: &Sedenion, noise: &Sedenion) -> FilterEvaluation {
        FilterEvaluation::from_linear_outputs(signal, noise, self.apply(signal), self.apply(noise))
    }
}

/// Error returned when constructing a noise-direction projector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectionError {
    /// The supplied direction is exactly zero.
    ZeroDirection,

    /// At least one coordinate, or the resulting norm, is non-finite.
    NonFiniteDirection,
}

impl fmt::Display for ProjectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::ZeroDirection => formatter.write_str("noise direction must be non-zero"),
            Self::NonFiniteDirection => formatter.write_str("noise direction must be finite"),
        }
    }
}

impl std::error::Error for ProjectionError {}

/// Orthogonal projector removing one known noise direction.
///
/// For a direction `d`, the operator is:
///
/// `P(x) = x - d × ((dᵀx) / (dᵀd))`.
///
/// This baseline annihilates `d` using ordinary real linear algebra. It also
/// exposes the unavoidable loss of any useful signal parallel to `d`.
#[derive(Clone, Debug, PartialEq)]
pub struct NoiseDirectionProjector {
    direction: Sedenion,
    inverse_squared_norm: f64,
}

impl NoiseDirectionProjector {
    /// Constructs a projector from a finite, non-zero direction.
    pub fn new(direction: Sedenion) -> Result<Self, ProjectionError> {
        if direction.iter().any(|value| !value.is_finite())
        {
            return Err(ProjectionError::NonFiniteDirection);
        }

        let norm_squared = squared_norm(&direction);

        if !norm_squared.is_finite()
        {
            return Err(ProjectionError::NonFiniteDirection);
        }

        if norm_squared == 0.0
        {
            return Err(ProjectionError::ZeroDirection);
        }

        Ok(Self {
            direction,
            inverse_squared_norm: 1.0 / norm_squared,
        })
    }

    /// Removes the input component parallel to the configured direction.
    #[must_use]
    pub fn apply(&self, input: &Sedenion) -> Sedenion {
        let dot_product = self
            .direction
            .iter()
            .zip(input.iter())
            .fold(0.0, |sum, (direction, value)| sum + direction * value);

        let scale = dot_product * self.inverse_squared_norm;

        core::array::from_fn(|index| input[index] - scale * self.direction[index])
    }

    /// Evaluates signal preservation and noise rejection separately.
    #[must_use]
    pub fn evaluate(&self, signal: &Sedenion, noise: &Sedenion) -> FilterEvaluation {
        FilterEvaluation::from_linear_outputs(signal, noise, self.apply(signal), self.apply(noise))
    }

    /// Returns the configured direction.
    #[must_use]
    pub const fn direction(&self) -> &Sedenion {
        &self.direction
    }
}

#[cfg(test)]
mod tests {
    use super::{IdentityFilter, NoiseDirectionProjector, ProjectionError};
    use crate::scalar::{SEDENION_DIMENSION, Sedenion, basis_vector, squared_norm};

    const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];

    #[test]
    fn identity_changes_nothing() {
        let signal = basis_vector(2).expect("e2 exists");
        let noise = basis_vector(9).expect("e9 exists");

        let evaluation = IdentityFilter.evaluate(&signal, &noise);

        assert_eq!(evaluation.filtered_signal(), &signal);
        assert_eq!(evaluation.filtered_noise(), &noise);
        assert_eq!(evaluation.metrics().signal_distortion_energy, 0.0);
        assert_eq!(evaluation.metrics().noise_attenuation_db, Some(0.0));
    }

    #[test]
    fn projector_rejects_invalid_directions() {
        assert_eq!(
            NoiseDirectionProjector::new(ZERO),
            Err(ProjectionError::ZeroDirection)
        );

        let mut non_finite = ZERO;
        non_finite[3] = f64::INFINITY;

        assert_eq!(
            NoiseDirectionProjector::new(non_finite),
            Err(ProjectionError::NonFiniteDirection)
        );
    }

    #[test]
    fn configured_noise_direction_is_annihilated() {
        let mut noise = ZERO;
        noise[4] = 1.0;
        noise[15] = -1.0;

        let projector = NoiseDirectionProjector::new(noise).expect("valid direction");

        assert_eq!(projector.apply(&noise), ZERO);
    }

    #[test]
    fn orthogonal_signal_is_preserved() {
        let mut noise = ZERO;
        noise[4] = 1.0;
        noise[15] = -1.0;

        let signal = basis_vector(0).expect("e0 exists");
        let projector = NoiseDirectionProjector::new(noise).expect("valid direction");

        assert_eq!(projector.apply(&signal), signal);
    }

    #[test]
    fn parallel_signal_is_destroyed() {
        let mut direction = ZERO;
        direction[4] = 1.0;
        direction[15] = -1.0;

        let projector = NoiseDirectionProjector::new(direction).expect("valid direction");

        assert_eq!(projector.apply(&direction), ZERO);
    }

    #[test]
    fn projection_improves_snr_for_known_direction() {
        let mut noise = ZERO;
        noise[4] = 1.0;
        noise[15] = -1.0;

        let signal = basis_vector(0).expect("e0 exists");
        let projector = NoiseDirectionProjector::new(noise).expect("valid direction");
        let evaluation = projector.evaluate(&signal, &noise);

        assert_eq!(squared_norm(evaluation.filtered_noise()), 0.0);
        assert_eq!(evaluation.metrics().signal_distortion_energy, 0.0);
        assert_eq!(evaluation.metrics().output_snr_db, Some(f64::INFINITY));
    }
}
