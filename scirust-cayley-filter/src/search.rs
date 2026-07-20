//! Deterministic coarse search over sparse Cayley multipliers.

use crate::optimizer::{MultiplierCase, MultiplierScore, score_multiplier};
use crate::scalar::{SEDENION_DIMENSION, Sedenion};

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
/// `e_i ± e_j`.
///
/// The result contains exactly:
///
/// `2 × C(16, 2) = 240` candidates.
///
/// Ordering is deterministic:
///
/// 1. ascending loss;
/// 2. ascending first index;
/// 3. ascending second index;
/// 4. negative sign before positive sign.
pub fn rank_two_term_multipliers(
    cases: &[MultiplierCase],
    relative_scale: f64,
    distortion_weight: f64,
) -> Result<Vec<SparseMultiplierCandidate>, String> {
    let mut candidates = Vec::with_capacity(SEDENION_DIMENSION * (SEDENION_DIMENSION - 1));

    for first_index in 0..SEDENION_DIMENSION
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

    candidates.sort_by(|left, right| {
        left.score
            .loss
            .total_cmp(&right.score.loss)
            .then_with(|| left.first_index.cmp(&right.first_index))
            .then_with(|| left.second_index.cmp(&right.second_index))
            .then_with(|| left.second_sign.cmp(&right.second_sign))
    });

    Ok(candidates)
}

#[cfg(test)]
mod tests {
    use super::rank_two_term_multipliers;
    use crate::optimizer::MultiplierCase;
    use crate::scalar::{SEDENION_DIMENSION, Sedenion, basis_vector};

    const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];

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
    fn search_is_deterministic() {
        let case = known_case();

        let first = rank_two_term_multipliers(std::slice::from_ref(&case), 1.0e-6, 10.0)
            .expect("first search succeeds");

        let second = rank_two_term_multipliers(std::slice::from_ref(&case), 1.0e-6, 10.0)
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
}
