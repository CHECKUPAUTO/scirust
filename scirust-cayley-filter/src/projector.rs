//! Orthogonal projector derived from a Cayley left-multiplication operator.

use core::fmt;

use scirust_solvers::Matrix;
use scirust_solvers::linalg::svd;

use crate::filter::FilterEvaluation;
use crate::operator::{Matrix16, left_multiplication_matrix, matrix_vector_mul};
use crate::optimizer::{MultiplierCase, MultiplierScore};
use crate::scalar::{SEDENION_DIMENSION, Sedenion, squared_norm};
use crate::search::zero_divisor_two_term_directions;

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

#[derive(Clone, Debug, PartialEq)]
pub struct HardCayleyProjectorCandidate {
    pub multiplier: Sedenion,
    pub first_index: usize,
    pub second_index: usize,
    pub second_sign: i8,
    pub projector: CayleyProjector,
    pub score: MultiplierScore,
}

pub fn rank_hard_zero_divisor_projectors(
    cases: &[MultiplierCase],
    distortion_weight: f64,
    analysis_tolerance: f64,
    relative_threshold: f64,
) -> Result<Vec<HardCayleyProjectorCandidate>, String> {
    let directions = zero_divisor_two_term_directions(analysis_tolerance)?;
    let mut candidates = Vec::with_capacity(directions.len());

    for direction in directions
    {
        let projector = CayleyProjector::new(direction.multiplier, relative_threshold)
            .map_err(|error| error.to_string())?;

        let score = score_cayley_projector(cases, &projector, distortion_weight)?;

        candidates.push(HardCayleyProjectorCandidate {
            multiplier: direction.multiplier,
            first_index: direction.first_index,
            second_index: direction.second_index,
            second_sign: direction.second_sign,
            projector,
            score,
        });
    }

    candidates.sort_by(|a, b| {
        a.score
            .loss
            .total_cmp(&b.score.loss)
            .then_with(|| a.first_index.cmp(&b.first_index))
            .then_with(|| a.second_index.cmp(&b.second_index))
            .then_with(|| a.second_sign.cmp(&b.second_sign))
    });

    Ok(candidates)
}

const ENERGY_FLOOR: f64 = 1.0e-30;

pub fn score_cayley_projector(
    cases: &[MultiplierCase],
    projector: &CayleyProjector,
    distortion_weight: f64,
) -> Result<MultiplierScore, String> {
    if cases.is_empty()
    {
        return Err("at least one case is required".into());
    }
    if !distortion_weight.is_finite() || distortion_weight < 0.0
    {
        return Err("distortion weight must be finite and non-negative".into());
    }

    let mut noise = 0.0;
    let mut distortion = 0.0;

    for case in cases
    {
        let filtered_signal = projector.apply(&case.signal);
        let filtered_noise = projector.apply(&case.noise);

        noise += squared_norm(&filtered_noise) / squared_norm(&case.noise).max(ENERGY_FLOOR);

        distortion += case
            .signal
            .iter()
            .zip(filtered_signal)
            .fold(0.0, |sum, (input, output)| sum + (input - output).powi(2))
            / squared_norm(&case.signal).max(ENERGY_FLOOR);
    }

    let count = cases.len() as f64;
    let mean_noise_ratio = noise / count;
    let mean_distortion_ratio = distortion / count;

    Ok(MultiplierScore {
        loss: mean_noise_ratio + distortion_weight * mean_distortion_ratio,
        mean_noise_ratio,
        mean_distortion_ratio,
        rejected_dimension: projector.rejected_dimension(),
    })
}

#[cfg(test)]
mod tests {
    use super::{CayleyProjector, ProjectorError, score_cayley_projector};
    use crate::optimizer::MultiplierCase;
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
    fn hard_zero_divisor_ranking_contains_84_candidates() {
        let case = MultiplierCase::new(basis_vector(7).unwrap(), basis_vector(2).unwrap());

        let ranked =
            super::rank_hard_zero_divisor_projectors(&[case], 10.0, 1.0e-12, 1.0e-12).unwrap();

        assert_eq!(ranked.len(), 84);
        assert!(ranked.iter().all(|c| c.projector.rejected_dimension() == 4));
    }

    #[test]
    fn exact_kernel_receives_zero_projector_loss() {
        let mut multiplier = ZERO;
        multiplier[1] = 1.0;
        multiplier[10] = 1.0;

        let mut noise = ZERO;
        noise[4] = 1.0;
        noise[15] = -1.0;

        let case = MultiplierCase::new(basis_vector(0).unwrap(), noise);
        let projector = CayleyProjector::new(multiplier, THRESHOLD).unwrap();
        let score = score_cayley_projector(&[case], &projector, 10.0).unwrap();

        assert!(score.loss < 1.0e-20);
        assert_eq!(score.rejected_dimension, 4);
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
