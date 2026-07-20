//! Associative split-Clifford projectors on R^16.

use crate::filter::FilterEvaluation;
use crate::operator::{Matrix16, matrix_vector_mul};
use crate::optimizer::{MultiplierCase, MultiplierScore};
use crate::scalar::{SEDENION_DIMENSION, Sedenion, squared_norm};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CliffordProjectorError {
    InvalidRejectedDimension,
    InvalidDirection,
    InvalidOrthonormalBasis,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SplitCliffordProjector {
    involution: Matrix16,
    projection: Matrix16,
    rejected_dimension: usize,
}

impl SplitCliffordProjector {
    pub fn from_direction(direction: Sedenion) -> Result<Self, CliffordProjectorError> {
        let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !norm.is_finite() || norm == 0.0
        {
            return Err(CliffordProjectorError::InvalidDirection);
        }
        let unit = direction.map(|x| x / norm);
        let involution = core::array::from_fn(|r| {
            core::array::from_fn(|c| {
                let identity = if r == c { 1.0 } else { 0.0 };
                identity - 2.0 * unit[r] * unit[c]
            })
        });

        let projection = core::array::from_fn(|r| {
            core::array::from_fn(|c| {
                let identity = if r == c { 1.0 } else { 0.0 };
                identity - unit[r] * unit[c]
            })
        });
        Ok(Self {
            involution,
            projection,
            rejected_dimension: 1,
        })
    }

    pub fn from_orthonormal_basis(basis: &[Sedenion]) -> Result<Self, CliffordProjectorError> {
        if basis.is_empty() || basis.len() > SEDENION_DIMENSION
        {
            return Err(CliffordProjectorError::InvalidRejectedDimension);
        }

        for (i, u) in basis.iter().enumerate()
        {
            let norm = u.iter().map(|x| x * x).sum::<f64>();
            if !norm.is_finite() || (norm - 1.0).abs() > 1.0e-12
            {
                return Err(CliffordProjectorError::InvalidOrthonormalBasis);
            }
            for v in &basis[..i]
            {
                let dot = u.iter().zip(v).map(|(a, b)| a * b).sum::<f64>();
                if dot.abs() > 1.0e-12
                {
                    return Err(CliffordProjectorError::InvalidOrthonormalBasis);
                }
            }
        }

        let projection = core::array::from_fn(|r| {
            core::array::from_fn(|c| {
                let identity = if r == c { 1.0 } else { 0.0 };
                basis.iter().fold(identity, |value, u| value - u[r] * u[c])
            })
        });

        let involution = core::array::from_fn(|r| {
            core::array::from_fn(|c| 2.0 * projection[r][c] - if r == c { 1.0 } else { 0.0 })
        });

        Ok(Self {
            involution,
            projection,
            rejected_dimension: basis.len(),
        })
    }

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

#[derive(Clone, Debug, PartialEq)]
pub struct CliffordProjectorCandidate {
    pub direction: Sedenion,
    pub first_index: usize,
    pub second_index: usize,
    pub second_sign: i8,
    pub projector: SplitCliffordProjector,
    pub score: MultiplierScore,
}

pub fn rank_two_term_clifford_projectors(
    cases: &[MultiplierCase],
    distortion_weight: f64,
) -> Result<Vec<CliffordProjectorCandidate>, String> {
    let mut candidates = Vec::with_capacity(240);

    for first_index in 0..SEDENION_DIMENSION
    {
        for second_index in (first_index + 1)..SEDENION_DIMENSION
        {
            for second_sign in [-1_i8, 1_i8]
            {
                let mut direction = [0.0; SEDENION_DIMENSION];
                direction[first_index] = 1.0;
                direction[second_index] = f64::from(second_sign);

                let projector = SplitCliffordProjector::from_direction(direction)
                    .map_err(|_| "invalid Clifford direction".to_string())?;
                let score = score_clifford_projector(cases, &projector, distortion_weight)?;

                candidates.push(CliffordProjectorCandidate {
                    direction,
                    first_index,
                    second_index,
                    second_sign,
                    projector,
                    score,
                });
            }
        }
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

pub fn score_clifford_projector(
    cases: &[MultiplierCase],
    projector: &SplitCliffordProjector,
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
        distortion += squared_distance(&case.signal, &filtered_signal)
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

fn squared_distance(left: &Sedenion, right: &Sedenion) -> f64 {
    left.iter().zip(right).map(|(a, b)| (a - b) * (a - b)).sum()
}

#[cfg(test)]
mod tests {
    use super::rank_two_term_clifford_projectors;
    use super::{CliffordProjectorError, SplitCliffordProjector, score_clifford_projector};
    use crate::MultiplierCase;
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
    fn oriented_subspace_is_removed_and_complement_preserved() {
        let basis = [
            basis_vector(2).unwrap(),
            basis_vector(5).unwrap(),
            basis_vector(9).unwrap(),
            basis_vector(14).unwrap(),
        ];
        let projector = SplitCliffordProjector::from_orthonormal_basis(&basis).unwrap();

        assert_eq!(projector.rejected_dimension(), 4);

        for direction in basis
        {
            assert_eq!(projector.apply(&direction), [0.0; SEDENION_DIMENSION]);
        }

        let signal = basis_vector(7).unwrap();
        assert_eq!(projector.apply(&signal), signal);
        assert_eq!(projector.apply(&projector.apply(&signal)), signal);
    }

    #[test]
    fn oriented_projector_removes_its_direction() {
        let direction = [1.0; SEDENION_DIMENSION];
        let projector = SplitCliffordProjector::from_direction(direction).unwrap();
        assert_eq!(projector.rejected_dimension(), 1);
        assert!(
            projector
                .apply(&direction)
                .iter()
                .all(|x| x.abs() < 1.0e-12)
        );
    }

    #[test]
    fn non_orthonormal_basis_is_rejected() {
        let u = basis_vector(2).unwrap();
        assert_eq!(
            SplitCliffordProjector::from_orthonormal_basis(&[u, u]),
            Err(CliffordProjectorError::InvalidOrthonormalBasis)
        );
    }

    #[test]
    fn invalid_direction_is_rejected() {
        assert_eq!(
            SplitCliffordProjector::from_direction([0.0; SEDENION_DIMENSION]),
            Err(CliffordProjectorError::InvalidDirection)
        );
    }

    #[test]
    fn sparse_search_contains_240_candidates() {
        let case = MultiplierCase::new(basis_vector(7).unwrap(), basis_vector(2).unwrap());
        let candidates = rank_two_term_clifford_projectors(&[case], 1.0).unwrap();
        assert_eq!(candidates.len(), 240);
    }

    #[test]
    fn score_rewards_exact_noise_rejection() {
        let p = SplitCliffordProjector::canonical(4).unwrap();
        let case = MultiplierCase::new(basis_vector(7).unwrap(), basis_vector(2).unwrap());
        let score = score_clifford_projector(&[case], &p, 1.0).unwrap();
        assert_eq!(score.loss, 0.0);
        assert_eq!(score.rejected_dimension, 4);
    }

    #[test]
    fn invalid_dimension_is_rejected() {
        assert_eq!(
            SplitCliffordProjector::canonical(17),
            Err(CliffordProjectorError::InvalidRejectedDimension)
        );
    }
}
