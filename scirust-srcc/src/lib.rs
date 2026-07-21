//! SciRust Resonant Consensus Closure.
//!
//! SRCC builds structured rejection subspaces from deterministic
//! agreement between independent linear transport paths.

#![forbid(unsafe_code)]

mod closure;
mod evaluation;
mod projector;
mod scoring;

pub use closure::{SrccAdmissionCertificate, SrccClosure, SrccClosureError};
pub use evaluation::SrccEvaluation;
pub use projector::SrccProjector;
pub use scoring::{SrccCase, SrccScore, SrccScoringError, score_projector};

use core::fmt;

pub const SRCC_DIMENSION: usize = 16;

pub type Vector16 = [f64; SRCC_DIMENSION];
pub type LinearMap16 = [[f64; SRCC_DIMENSION]; SRCC_DIMENSION];

#[must_use]
pub fn basis_vector(index: usize) -> Option<Vector16> {
    if index >= SRCC_DIMENSION
    {
        return None;
    }

    let mut vector = [0.0; SRCC_DIMENSION];
    vector[index] = 1.0;
    Some(vector)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SrccConfig {
    pub novelty_threshold: f64,
    pub resonance_threshold: f64,
    pub minimum_support: usize,
    pub maximum_dimension: usize,
    pub maximum_rounds: usize,
    pub energy_floor: f64,
}

impl Default for SrccConfig {
    fn default() -> Self {
        Self {
            novelty_threshold: 1.0e-10,
            resonance_threshold: 0.999,
            minimum_support: 2,
            maximum_dimension: SRCC_DIMENSION,
            maximum_rounds: SRCC_DIMENSION,
            energy_floor: 1.0e-30,
        }
    }
}

impl SrccConfig {
    pub fn validate(self) -> Result<Self, SrccError> {
        if !self.novelty_threshold.is_finite() || self.novelty_threshold <= 0.0
        {
            return Err(SrccError::InvalidNoveltyThreshold);
        }

        if !self.resonance_threshold.is_finite()
            || self.resonance_threshold <= 0.0
            || self.resonance_threshold > 1.0
        {
            return Err(SrccError::InvalidResonanceThreshold);
        }

        if self.minimum_support == 0
        {
            return Err(SrccError::InvalidMinimumSupport);
        }

        if self.maximum_dimension == 0 || self.maximum_dimension > SRCC_DIMENSION
        {
            return Err(SrccError::InvalidMaximumDimension);
        }

        if self.maximum_rounds == 0
        {
            return Err(SrccError::InvalidMaximumRounds);
        }

        if !self.energy_floor.is_finite() || self.energy_floor <= 0.0
        {
            return Err(SrccError::InvalidEnergyFloor);
        }

        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SrccError {
    InvalidNoveltyThreshold,
    InvalidResonanceThreshold,
    InvalidMinimumSupport,
    InvalidMaximumDimension,
    InvalidMaximumRounds,
    InvalidEnergyFloor,
}

impl fmt::Display for SrccError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self
        {
            Self::InvalidNoveltyThreshold => "novelty threshold must be finite and positive",
            Self::InvalidResonanceThreshold => "resonance threshold must belong to (0, 1]",
            Self::InvalidMinimumSupport => "minimum support must be positive",
            Self::InvalidMaximumDimension => "maximum dimension must belong to 1..=16",
            Self::InvalidMaximumRounds => "maximum rounds must be positive",
            Self::InvalidEnergyFloor => "energy floor must be finite and positive",
        };

        formatter.write_str(message)
    }
}

impl std::error::Error for SrccError {}

#[must_use]
pub fn apply_linear_map(map: &LinearMap16, input: &Vector16) -> Vector16 {
    core::array::from_fn(|row| {
        map[row]
            .iter()
            .zip(input)
            .fold(0.0, |sum, (coefficient, value)| sum + coefficient * value)
    })
}

#[must_use]
pub fn dot(left: &Vector16, right: &Vector16) -> f64 {
    left.iter().zip(right).fold(0.0, |sum, (a, b)| sum + a * b)
}

#[must_use]
pub fn squared_norm(vector: &Vector16) -> f64 {
    dot(vector, vector)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_configuration_is_valid() {
        assert_eq!(SrccConfig::default().validate(), Ok(SrccConfig::default()),);
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        let invalid = SrccConfig {
            minimum_support: 0,
            ..SrccConfig::default()
        };

        assert_eq!(invalid.validate(), Err(SrccError::InvalidMinimumSupport),);
    }

    #[test]
    fn linear_map_uses_fixed_accumulation_order() {
        let map: LinearMap16 = core::array::from_fn(|row| {
            core::array::from_fn(|column| if row == column { 2.0 } else { 0.0 })
        });

        let input: Vector16 = core::array::from_fn(|index| index as f64);

        let output = apply_linear_map(&map, &input);

        assert_eq!(output, core::array::from_fn(|index| { 2.0 * index as f64 }),);
    }
}
