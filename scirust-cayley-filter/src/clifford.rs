//! Associative split-Clifford projectors on R^16.

use scirust_solvers::{Matrix, linalg::svd};

use crate::filter::FilterEvaluation;
use crate::operator::{Matrix16, matrix_vector_mul};
use crate::optimizer::{MultiplierCase, MultiplierScore};
use crate::scalar::{SEDENION_DIMENSION, Sedenion, squared_norm};
use crate::search::zero_divisor_two_term_directions;
use crate::selection::{DevelopmentGateDecision, development_gate};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CliffordProjectorError {
    InvalidRejectedDimension,
    InvalidDirection,
    InvalidOrthonormalBasis,
    InvalidRotation,
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

    pub fn rotated_canonical(
        rejected_dimension: usize,
        rejected_index: usize,
        kept_index: usize,
        slope: i8,
    ) -> Result<Self, CliffordProjectorError> {
        if rejected_dimension == 0
            || rejected_dimension >= SEDENION_DIMENSION
            || rejected_index >= rejected_dimension
            || kept_index < rejected_dimension
            || kept_index >= SEDENION_DIMENSION
            || slope == 0
        {
            return Err(CliffordProjectorError::InvalidRotation);
        }

        let mut basis = Vec::with_capacity(rejected_dimension);
        for index in 0..rejected_dimension
        {
            let mut vector = [0.0; SEDENION_DIMENSION];
            vector[index] = 1.0;
            basis.push(vector);
        }

        let slope = f64::from(slope);
        let norm = (1.0 + slope * slope).sqrt();
        basis[rejected_index][rejected_index] = 1.0 / norm;
        basis[rejected_index][kept_index] = slope / norm;

        Self::from_orthonormal_basis(&basis)
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

pub fn rank_two_term_nullity_four_clifford_projectors(
    cases: &[MultiplierCase],
    distortion_weight: f64,
) -> Result<Vec<CliffordProjectorCandidate>, String> {
    const NULLITY: usize = 4;
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

                let mut basis = Vec::with_capacity(NULLITY);
                basis.push(direction.map(|x| x / 2.0_f64.sqrt()));

                for index in 1..SEDENION_DIMENSION
                {
                    if index != first_index && index != second_index
                    {
                        let mut axis = [0.0; SEDENION_DIMENSION];
                        axis[index] = 1.0;
                        basis.push(axis);
                        if basis.len() == NULLITY
                        {
                            break;
                        }
                    }
                }

                let projector = SplitCliffordProjector::from_orthonormal_basis(&basis)
                    .map_err(|_| "invalid nullity-four Clifford basis".to_string())?;
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

pub fn rank_zero_divisor_matched_nullity_four_clifford_projectors(
    cases: &[MultiplierCase],
    distortion_weight: f64,
    analysis_tolerance: f64,
) -> Result<Vec<CliffordProjectorCandidate>, String> {
    let directions = zero_divisor_two_term_directions(analysis_tolerance)?;
    let mut candidates = rank_two_term_nullity_four_clifford_projectors(cases, distortion_weight)?;

    candidates.retain(|candidate| {
        directions.iter().any(|direction| {
            direction.first_index == candidate.first_index
                && direction.second_index == candidate.second_index
                && direction.second_sign == candidate.second_sign
        })
    });

    Ok(candidates)
}

pub fn fit_clifford_noise_subspace(
    cases: &[MultiplierCase],
    max_dimension: usize,
    relative_tolerance: f64,
) -> Result<SplitCliffordProjector, String> {
    if cases.is_empty()
    {
        return Err("at least one training case is required".into());
    }
    if max_dimension == 0 || max_dimension > SEDENION_DIMENSION
    {
        return Err("max dimension must be between 1 and 16".into());
    }
    if !relative_tolerance.is_finite() || relative_tolerance <= 0.0
    {
        return Err("relative tolerance must be finite and positive".into());
    }

    let mut covariance = [[0.0; SEDENION_DIMENSION]; SEDENION_DIMENSION];

    for case in cases
    {
        if case.noise.iter().any(|value| !value.is_finite())
        {
            return Err("training noise must be finite".into());
        }

        for (row, covariance_row) in covariance.iter_mut().enumerate()
        {
            let noise_row = case.noise[row];

            for (column, value) in covariance_row.iter_mut().enumerate()
            {
                *value += noise_row * case.noise[column];
            }
        }
    }

    let data = covariance
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();

    let matrix = Matrix::from_row_major(SEDENION_DIMENSION, SEDENION_DIMENSION, data);
    let decomposition = svd(&matrix).map_err(|error| error.to_string())?;

    let sigma_max = decomposition.s.first().copied().unwrap_or(0.0);
    if sigma_max == 0.0
    {
        return Err("training noise must contain non-zero energy".into());
    }

    let cutoff = relative_tolerance * sigma_max;
    let dimension = decomposition
        .s
        .iter()
        .take(max_dimension)
        .take_while(|&&sigma| sigma > cutoff)
        .count();

    if dimension == 0
    {
        return Err("no stable noise direction was identified".into());
    }

    let basis: Vec<Sedenion> = (0..dimension)
        .map(|direction| core::array::from_fn(|row| decomposition.v[(row, direction)]))
        .collect();

    SplitCliffordProjector::from_orthonormal_basis(&basis)
        .map_err(|_| "invalid learned Clifford basis".to_string())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CliffordGateDecision {
    Identity,
    Clifford,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectedCliffordCandidate {
    pub first_index: usize,
    pub second_index: usize,
    pub second_sign: i8,
    pub direction: Sedenion,
    pub projector: SplitCliffordProjector,
    pub train_score: MultiplierScore,
    pub dev_score: MultiplierScore,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CliffordSelectionResult {
    pub selected: SelectedCliffordCandidate,
    pub candidates: Vec<SelectedCliffordCandidate>,
    pub decision: CliffordGateDecision,
}

pub fn select_clifford_train_dev(
    train: &[MultiplierCase],
    dev: &[MultiplierCase],
    top_k: usize,
    distortion_weight: f64,
) -> Result<CliffordSelectionResult, String> {
    if top_k == 0
    {
        return Err("top_k must be strictly positive".into());
    }

    let ranked = rank_two_term_clifford_projectors(train, distortion_weight)?;
    let mut candidates = Vec::with_capacity(top_k.min(ranked.len()));

    for candidate in ranked.into_iter().take(top_k)
    {
        let dev_score = score_clifford_projector(dev, &candidate.projector, distortion_weight)?;

        candidates.push(SelectedCliffordCandidate {
            first_index: candidate.first_index,
            second_index: candidate.second_index,
            second_sign: candidate.second_sign,
            direction: candidate.direction,
            projector: candidate.projector,
            train_score: candidate.score,
            dev_score,
        });
    }

    candidates.sort_by(|a, b| {
        a.dev_score
            .loss
            .total_cmp(&b.dev_score.loss)
            .then_with(|| a.train_score.loss.total_cmp(&b.train_score.loss))
            .then_with(|| a.first_index.cmp(&b.first_index))
            .then_with(|| a.second_index.cmp(&b.second_index))
            .then_with(|| a.second_sign.cmp(&b.second_sign))
    });

    let selected = candidates
        .first()
        .cloned()
        .ok_or_else(|| "no Clifford candidate was produced".to_string())?;
    let decision = match development_gate(&selected.dev_score)?
    {
        DevelopmentGateDecision::Identity => CliffordGateDecision::Identity,
        DevelopmentGateDecision::Cayley => CliffordGateDecision::Clifford,
    };

    Ok(CliffordSelectionResult {
        selected,
        candidates,
        decision,
    })
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
    use crate::{
        CayleyProjector, NoiseSubspaceProjector, SEDENION_DIMENSION, analyze_matrix, basis_vector,
        left_multiplication_matrix, matrix_vector_mul,
    };

    #[test]
    fn learned_clifford_recovers_full_cayley_kernel() {
        let mut multiplier = [0.0; SEDENION_DIMENSION];
        multiplier[1] = 1.0;
        multiplier[10] = 1.0;

        let matrix = left_multiplication_matrix(multiplier);
        let analysis = analyze_matrix(&matrix, 1.0e-12).unwrap();

        let signal = basis_vector(0).unwrap();
        let cases: Vec<_> = analysis
            .kernel_basis()
            .iter()
            .copied()
            .map(|noise| MultiplierCase::new(signal, noise))
            .collect();

        let projector = super::fit_clifford_noise_subspace(&cases, 4, 1.0e-12).unwrap();

        assert_eq!(projector.rejected_dimension(), 4);

        for noise in analysis.kernel_basis()
        {
            let residual = projector.apply(noise);
            let energy = residual.iter().map(|x| x * x).sum::<f64>();
            assert!(energy < 1.0e-20);
        }

        assert_eq!(projector.apply(&signal), signal);
    }

    #[test]
    fn cayley_kernel_and_clifford_projectors_are_equivalent() {
        let mut multiplier = [0.0; SEDENION_DIMENSION];
        multiplier[1] = 1.0;
        multiplier[10] = 1.0;

        let matrix = left_multiplication_matrix(multiplier);
        let analysis = analyze_matrix(&matrix, 1.0e-12).unwrap();
        let subspace = NoiseSubspaceProjector::new(analysis.kernel_basis(), 1.0e-12).unwrap();

        let clifford =
            SplitCliffordProjector::from_orthonormal_basis(subspace.orthonormal_basis()).unwrap();

        let cayley = CayleyProjector::new(multiplier, 1.0e-12).unwrap();

        assert_eq!(clifford.rejected_dimension(), 4);
        assert_eq!(cayley.rejected_dimension(), 4);

        for index in 0..SEDENION_DIMENSION
        {
            let input = basis_vector(index).unwrap();
            let left = clifford.apply(&input);
            let right = cayley.apply(&input);

            let error = left
                .iter()
                .zip(right)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>();

            assert!(error < 1.0e-20);
        }
    }

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
    fn rotated_nullity_four_removes_target_direction() {
        let p = SplitCliffordProjector::rotated_canonical(4, 0, 4, 1).unwrap();
        let mut noise = [0.0; SEDENION_DIMENSION];
        noise[0] = 1.0;
        noise[4] = 1.0;
        assert_eq!(p.rejected_dimension(), 4);
        assert!(p.apply(&noise).iter().all(|x| x.abs() < 1.0e-12));
        let signal = basis_vector(7).unwrap();
        assert_eq!(p.apply(&signal), signal);
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
    fn train_dev_selection_activates_clifford() {
        let mut noise = [0.0; SEDENION_DIMENSION];
        noise[2] = 1.0;
        noise[3] = 1.0;

        let case = MultiplierCase::new(basis_vector(7).unwrap(), noise);
        let result = super::select_clifford_train_dev(
            std::slice::from_ref(&case),
            std::slice::from_ref(&case),
            8,
            10.0,
        )
        .unwrap();

        assert_eq!(result.decision, super::CliffordGateDecision::Clifford);
        assert!(result.selected.dev_score.loss < 1.0e-20);
    }

    #[test]
    fn train_dev_shift_selects_identity() {
        let mut train_noise = [0.0; SEDENION_DIMENSION];
        train_noise[2] = 1.0;
        train_noise[3] = 1.0;

        let mut dev_noise = [0.0; SEDENION_DIMENSION];
        dev_noise[4] = 1.0;
        dev_noise[5] = 1.0;

        let train = MultiplierCase::new(basis_vector(7).unwrap(), train_noise);
        let dev = MultiplierCase::new(basis_vector(7).unwrap(), dev_noise);

        let result = super::select_clifford_train_dev(&[train], &[dev], 1, 10.0).unwrap();

        assert_eq!(result.decision, super::CliffordGateDecision::Identity);
        assert_eq!(result.selected.dev_score.loss, 1.0);
    }

    #[test]
    fn matched_nullity_four_candidates_preserve_scalar_axis() {
        let case = MultiplierCase::new(basis_vector(0).unwrap(), basis_vector(4).unwrap());

        let ranked = super::rank_zero_divisor_matched_nullity_four_clifford_projectors(
            &[case],
            10.0,
            1.0e-12,
        )
        .unwrap();

        let scalar = basis_vector(0).unwrap();
        assert!(ranked.iter().all(|c| c.projector.apply(&scalar) == scalar));
    }

    #[test]
    fn matched_nullity_four_search_contains_84_candidates() {
        let case = MultiplierCase::new(basis_vector(7).unwrap(), basis_vector(2).unwrap());

        let ranked = super::rank_zero_divisor_matched_nullity_four_clifford_projectors(
            &[case],
            10.0,
            1.0e-12,
        )
        .unwrap();

        assert_eq!(ranked.len(), 84);
        assert!(ranked.iter().all(|c| c.projector.rejected_dimension() == 4));
    }

    #[test]
    fn nullity_four_search_contains_240_candidates() {
        let case = MultiplierCase::new(basis_vector(7).unwrap(), basis_vector(2).unwrap());
        let ranked = super::rank_two_term_nullity_four_clifford_projectors(&[case], 10.0).unwrap();

        assert_eq!(ranked.len(), 240);
        assert!(ranked.iter().all(|c| c.projector.rejected_dimension() == 4));
    }

    #[test]
    fn sparse_search_contains_240_candidates() {
        let case = MultiplierCase::new(basis_vector(7).unwrap(), basis_vector(2).unwrap());
        let candidates =
            rank_two_term_clifford_projectors(std::slice::from_ref(&case), 1.0).unwrap();
        assert_eq!(candidates.len(), 240);
    }

    #[test]
    fn score_rewards_exact_noise_rejection() {
        let p = SplitCliffordProjector::canonical(4).unwrap();
        let case = MultiplierCase::new(basis_vector(7).unwrap(), basis_vector(2).unwrap());
        let score = score_clifford_projector(std::slice::from_ref(&case), &p, 1.0).unwrap();
        assert_eq!(score.loss, 0.0);
        assert_eq!(score.rejected_dimension, 4);
    }

    #[test]
    fn invalid_rotation_is_rejected() {
        let error = SplitCliffordProjector::rotated_canonical(4, 4, 5, 1).unwrap_err();
        assert_eq!(error, CliffordProjectorError::InvalidRotation);
    }

    #[test]
    fn invalid_dimension_is_rejected() {
        assert_eq!(
            SplitCliffordProjector::canonical(17),
            Err(CliffordProjectorError::InvalidRejectedDimension)
        );
    }
}
