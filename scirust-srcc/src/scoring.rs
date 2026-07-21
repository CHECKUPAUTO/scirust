//! Deterministic aggregate scoring for SRCC projectors.

use core::fmt;

use crate::{SrccEvaluation, SrccProjector, Vector16};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SrccCase {
    pub signal: Vector16,
    pub noise: Vector16,
}

impl SrccCase {
    #[must_use]
    pub const fn new(signal: Vector16, noise: Vector16) -> Self {
        Self { signal, noise }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SrccScore {
    pub mean_residual_noise_ratio: f64,
    pub mean_signal_distortion_ratio: f64,
    pub loss: f64,
    pub case_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SrccScoringError {
    EmptyCases,
    InvalidDistortionWeight,
}

impl fmt::Display for SrccScoringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyCases => formatter.write_str("at least one scoring case is required"),
            Self::InvalidDistortionWeight =>
            {
                formatter.write_str("distortion weight must be finite and non-negative")
            },
        }
    }
}

impl std::error::Error for SrccScoringError {}

pub fn score_projector(
    cases: &[SrccCase],
    projector: &SrccProjector,
    distortion_weight: f64,
) -> Result<SrccScore, SrccScoringError> {
    if cases.is_empty()
    {
        return Err(SrccScoringError::EmptyCases);
    }

    if !distortion_weight.is_finite() || distortion_weight < 0.0
    {
        return Err(SrccScoringError::InvalidDistortionWeight);
    }

    let mut residual_noise = 0.0;
    let mut signal_distortion = 0.0;

    for case in cases
    {
        let evaluation = SrccEvaluation::evaluate(projector, &case.signal, &case.noise);

        residual_noise += evaluation.residual_noise_ratio;

        signal_distortion += evaluation.signal_distortion_ratio;
    }

    let inverse_count = 1.0 / cases.len() as f64;

    let mean_residual_noise_ratio = residual_noise * inverse_count;

    let mean_signal_distortion_ratio = signal_distortion * inverse_count;

    let loss = mean_residual_noise_ratio + distortion_weight * mean_signal_distortion_ratio;

    Ok(SrccScore {
        mean_residual_noise_ratio,
        mean_signal_distortion_ratio,
        loss,
        case_count: cases.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LinearMap16, SRCC_DIMENSION, SrccConfig, basis_vector};

    fn transport(source: usize, target: usize, coefficient: f64) -> LinearMap16 {
        let mut map = [[0.0; SRCC_DIMENSION]; SRCC_DIMENSION];

        map[target][source] = coefficient;
        map
    }

    fn projector() -> SrccProjector {
        SrccProjector::build(
            &[basis_vector(1).unwrap()],
            &[transport(1, 2, 1.0), transport(1, 2, -1.0)],
            SrccConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn exact_cases_receive_near_zero_loss() {
        let cases = [
            SrccCase::new(basis_vector(8).unwrap(), basis_vector(1).unwrap()),
            SrccCase::new(basis_vector(9).unwrap(), basis_vector(2).unwrap()),
        ];

        let score = score_projector(&cases, &projector(), 10.0).unwrap();

        assert_eq!(score.case_count, 2);
        assert!(score.loss < 1.0e-24);
    }

    #[test]
    fn distortion_is_aggregated() {
        let cases = [
            SrccCase::new(basis_vector(1).unwrap(), basis_vector(8).unwrap()),
            SrccCase::new(basis_vector(8).unwrap(), basis_vector(2).unwrap()),
        ];

        let score = score_projector(&cases, &projector(), 10.0).unwrap();

        assert!((score.mean_signal_distortion_ratio - 0.5).abs() < 1.0e-15);

        assert!((score.mean_residual_noise_ratio - 0.5).abs() < 1.0e-15);

        assert!((score.loss - 5.5).abs() < 1.0e-15);
    }

    #[test]
    fn invalid_scoring_inputs_are_rejected() {
        assert_eq!(
            score_projector(&[], &projector(), 1.0),
            Err(SrccScoringError::EmptyCases),
        );

        let cases = [SrccCase::new(
            basis_vector(8).unwrap(),
            basis_vector(2).unwrap(),
        )];

        assert_eq!(
            score_projector(&cases, &projector(), f64::NAN,),
            Err(SrccScoringError::InvalidDistortionWeight,),
        );
    }
}
