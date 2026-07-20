//! Deterministic coarse search over sparse Cayley multipliers.

use crate::analysis::analyze_matrix;
use crate::operator::left_multiplication_matrix;
use crate::optimizer::{MultiplierCase, MultiplierScore, score_multiplier};
use crate::scalar::{SEDENION_DIMENSION, Sedenion};

/// One neutral two-term zero-divisor direction `e_i ± e_j`.
///
/// This structure carries only algebraic information. It does not
/// assume a signal-plus-noise model and can therefore be reused by
/// diagnostic, classification, or preservation protocols.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseMultiplierDirection {
    pub multiplier: Sedenion,
    pub first_index: usize,
    pub second_index: usize,
    pub second_sign: i8,
    pub kernel_dimension: usize,
}

/// One ranked two-term multiplier `e_i ± e_j`.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseMultiplierCandidate {
    pub multiplier: Sedenion,
    pub first_index: usize,
    pub second_index: usize,
    pub second_sign: i8,
    pub score: MultiplierScore,
}

/// Exhaustively ranks every scale-independent two-term direction
/// `e_i ± e_j`, including directions containing the real unit `e0`.
///
/// The result contains exactly:
///
/// `2 × C(16, 2) = 240` candidates.
pub fn rank_two_term_multipliers(
    cases: &[MultiplierCase],
    relative_scale: f64,
    distortion_weight: f64,
) -> Result<Vec<SparseMultiplierCandidate>, String> {
    rank_two_term_multipliers_from(cases, relative_scale, distortion_weight, 0)
}

/// Exhaustively ranks purely imaginary two-term directions
/// `e_i ± e_j`, with `1 ≤ i < j ≤ 15`.
///
/// The result contains exactly:
///
/// `2 × C(15, 2) = 210` candidates.
pub fn rank_imaginary_two_term_multipliers(
    cases: &[MultiplierCase],
    relative_scale: f64,
    distortion_weight: f64,
) -> Result<Vec<SparseMultiplierCandidate>, String> {
    rank_two_term_multipliers_from(cases, relative_scale, distortion_weight, 1)
}

/// Enumerates every purely imaginary sparse two-term multiplier whose
/// left-multiplication operator has a non-empty deterministic kernel.
///
/// The output order is algebraic and deterministic:
///
/// 1. ascending first index;
/// 2. ascending second index;
/// 3. negative sign before positive sign.
///
/// No signal, noise, label, or empirical score is used.
pub fn zero_divisor_two_term_directions(
    analysis_tolerance: f64,
) -> Result<Vec<SparseMultiplierDirection>, String> {
    let mut directions = Vec::new();

    for first_index in 1..SEDENION_DIMENSION
    {
        for second_index in (first_index + 1)..SEDENION_DIMENSION
        {
            for second_sign in [-1_i8, 1_i8]
            {
                let mut multiplier = [0.0; SEDENION_DIMENSION];

                multiplier[first_index] = 1.0;
                multiplier[second_index] = f64::from(second_sign);

                let matrix = left_multiplication_matrix(multiplier);

                let analysis = analyze_matrix(&matrix, analysis_tolerance)
                    .map_err(|error| error.to_string())?;

                let kernel_dimension = analysis.kernel_basis().len();

                if kernel_dimension == 0
                {
                    continue;
                }

                directions.push(SparseMultiplierDirection {
                    multiplier,
                    first_index,
                    second_index,
                    second_sign,
                    kernel_dimension,
                });
            }
        }
    }

    Ok(directions)
}

/// Ranks only purely imaginary two-term multipliers whose left
/// multiplication operator has a non-empty deterministic kernel.
///
/// These are genuine sparse zero-divisor directions. Full-rank
/// multipliers that merely produce near-isotropic attenuation are
/// excluded before scoring.
pub fn rank_zero_divisor_two_term_multipliers(
    cases: &[MultiplierCase],
    relative_scale: f64,
    distortion_weight: f64,
    analysis_tolerance: f64,
) -> Result<Vec<SparseMultiplierCandidate>, String> {
    let directions = zero_divisor_two_term_directions(analysis_tolerance)?;

    let mut candidates = Vec::with_capacity(directions.len());

    for direction in directions
    {
        let score = score_multiplier(
            cases,
            &direction.multiplier,
            relative_scale,
            distortion_weight,
        )?;

        candidates.push(SparseMultiplierCandidate {
            multiplier: direction.multiplier,
            first_index: direction.first_index,
            second_index: direction.second_index,
            second_sign: direction.second_sign,
            score,
        });
    }

    sort_candidates(&mut candidates);

    Ok(candidates)
}

fn rank_two_term_multipliers_from(
    cases: &[MultiplierCase],
    relative_scale: f64,
    distortion_weight: f64,
    first_allowed_index: usize,
) -> Result<Vec<SparseMultiplierCandidate>, String> {
    let direction_count = SEDENION_DIMENSION - first_allowed_index;

    let mut candidates = Vec::with_capacity(direction_count * direction_count.saturating_sub(1));

    for first_index in first_allowed_index..SEDENION_DIMENSION
    {
        for second_index in (first_index + 1)..SEDENION_DIMENSION
        {
            for second_sign in [-1_i8, 1_i8]
            {
                let mut multiplier = [0.0; SEDENION_DIMENSION];

                multiplier[first_index] = 1.0;
                multiplier[second_index] = f64::from(second_sign);

                let score =
                    score_multiplier(cases, &multiplier, relative_scale, distortion_weight)?;

                candidates.push(SparseMultiplierCandidate {
                    multiplier,
                    first_index,
                    second_index,
                    second_sign,
                    score,
                });
            }
        }
    }

    sort_candidates(&mut candidates);

    Ok(candidates)
}

fn sort_candidates(candidates: &mut [SparseMultiplierCandidate]) {
    candidates.sort_by(|left, right| {
        left.score
            .loss
            .total_cmp(&right.score.loss)
            .then_with(|| left.first_index.cmp(&right.first_index))
            .then_with(|| left.second_index.cmp(&right.second_index))
            .then_with(|| left.second_sign.cmp(&right.second_sign))
    });
}

#[cfg(test)]
mod tests {
    use super::{
        rank_imaginary_two_term_multipliers, rank_two_term_multipliers,
        rank_zero_divisor_two_term_multipliers, zero_divisor_two_term_directions,
    };
    use crate::analysis::analyze_matrix;
    use crate::operator::left_multiplication_matrix;
    use crate::optimizer::MultiplierCase;
    use crate::scalar::{SEDENION_DIMENSION, Sedenion, basis_vector};

    const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];
    const ANALYSIS_TOLERANCE: f64 = 1.0e-12;

    fn known_case() -> MultiplierCase {
        let signal = basis_vector(0).expect("e0 exists");

        let mut noise = ZERO;
        noise[4] = 1.0;
        noise[15] = -1.0;

        MultiplierCase::new(signal, noise)
    }

    #[test]
    fn exhaustive_search_contains_240_candidates() {
        let case = known_case();

        let candidates = rank_two_term_multipliers(std::slice::from_ref(&case), 1.0e-6, 10.0)
            .expect("search succeeds");

        assert_eq!(candidates.len(), 240);
    }

    #[test]
    fn imaginary_search_contains_210_candidates() {
        let case = known_case();

        let candidates =
            rank_imaginary_two_term_multipliers(std::slice::from_ref(&case), 1.0e-6, 10.0)
                .expect("search succeeds");

        assert_eq!(candidates.len(), 210);

        assert!(candidates.iter().all(|candidate| {
            candidate.first_index >= 1
                && candidate.second_index >= 1
                && candidate.multiplier[0] == 0.0
        }));
    }

    #[test]
    fn zero_divisor_search_is_non_empty_and_strict() {
        let case = known_case();

        let candidates = rank_zero_divisor_two_term_multipliers(
            std::slice::from_ref(&case),
            1.0e-6,
            10.0,
            ANALYSIS_TOLERANCE,
        )
        .expect("search succeeds");

        assert!(!candidates.is_empty());
        assert!(candidates.len() < 210);

        for candidate in candidates
        {
            assert_eq!(candidate.multiplier[0], 0.0);

            let matrix = left_multiplication_matrix(candidate.multiplier);

            let analysis = analyze_matrix(&matrix, ANALYSIS_TOLERANCE).expect("analysis succeeds");

            assert!(!analysis.kernel_basis().is_empty());
        }
    }

    #[test]
    fn search_is_deterministic() {
        let case = known_case();

        let first = rank_two_term_multipliers(std::slice::from_ref(&case), 1.0e-6, 10.0)
            .expect("first search succeeds");

        let second = rank_two_term_multipliers(std::slice::from_ref(&case), 1.0e-6, 10.0)
            .expect("second search succeeds");

        assert_eq!(first, second);
    }

    #[test]
    fn zero_divisor_search_is_deterministic() {
        let case = known_case();

        let first = rank_zero_divisor_two_term_multipliers(
            std::slice::from_ref(&case),
            1.0e-6,
            10.0,
            ANALYSIS_TOLERANCE,
        )
        .expect("first search succeeds");

        let second = rank_zero_divisor_two_term_multipliers(
            std::slice::from_ref(&case),
            1.0e-6,
            10.0,
            ANALYSIS_TOLERANCE,
        )
        .expect("second search succeeds");

        assert_eq!(first, second);
    }

    #[test]
    fn global_search_finds_an_exact_noise_annihilator() {
        let case = known_case();

        let candidates = rank_two_term_multipliers(std::slice::from_ref(&case), 1.0e-6, 10.0)
            .expect("search succeeds");

        let best = candidates.first().expect("candidate exists");

        assert!(best.score.loss < 1.0e-10);
        assert!(best.score.mean_noise_ratio < 1.0e-16);
        assert!(best.score.mean_distortion_ratio < 1.0e-10);
    }

    #[test]
    fn zero_divisor_search_finds_exact_annihilator() {
        let case = known_case();

        let candidates = rank_zero_divisor_two_term_multipliers(
            std::slice::from_ref(&case),
            1.0e-6,
            10.0,
            ANALYSIS_TOLERANCE,
        )
        .expect("search succeeds");

        let best = candidates.first().expect("candidate exists");

        assert!(best.first_index >= 1);
        assert_eq!(best.multiplier[0], 0.0);
        assert!(best.score.loss < 1.0e-10);
        assert!(best.score.mean_noise_ratio < 1.0e-16);
        assert!(best.score.mean_distortion_ratio < 1.0e-10);
        assert_eq!(best.score.rejected_dimension, 4);
    }

    #[test]
    fn zero_divisor_directions_are_neutral_and_deterministic() {
        let first = zero_divisor_two_term_directions(ANALYSIS_TOLERANCE)
            .expect("first enumeration succeeds");

        let second = zero_divisor_two_term_directions(ANALYSIS_TOLERANCE)
            .expect("second enumeration succeeds");

        assert_eq!(first, second);
        assert!(!first.is_empty());
        assert!(first.len() < 210);

        assert!(first.iter().all(|direction| {
            direction.first_index >= 1
                && direction.second_index > direction.first_index
                && direction.multiplier[0] == 0.0
                && direction.kernel_dimension > 0
        }));

        let known = first
            .iter()
            .find(|direction| {
                direction.first_index == 1
                    && direction.second_index == 10
                    && direction.second_sign == 1
            })
            .expect("known e1+e10 zero divisor exists");

        assert_eq!(known.kernel_dimension, 4);
    }

    #[test]
    fn zero_divisor_direction_search_rejects_invalid_tolerance() {
        for tolerance in [-1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]
        {
            assert!(zero_divisor_two_term_directions(tolerance,).is_err());
        }
    }
}
