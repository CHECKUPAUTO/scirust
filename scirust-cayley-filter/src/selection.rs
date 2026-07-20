//! Deterministic train/development selection of Cayley multipliers.

use scirust_solvers::Tolerance;

use crate::optimizer::{MultiplierCase, MultiplierScore, optimize_multiplier, score_multiplier};
use crate::scalar::Sedenion;
use crate::search::rank_two_term_multipliers;

/// One globally seeded candidate evaluated on separate development data.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectedMultiplierCandidate {
    pub seed_rank: usize,
    pub seed_first_index: usize,
    pub seed_second_index: usize,
    pub seed_second_sign: i8,
    pub seed_multiplier: Sedenion,
    pub multiplier: Sedenion,
    pub train_score: MultiplierScore,
    pub dev_score: MultiplierScore,
    pub refinement_used: bool,
    pub refinement_error: Option<String>,
    pub iterations: Option<usize>,
    pub residual: Option<f64>,
}

/// Complete deterministic train/development selection report.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiplierSelectionResult {
    pub selected: SelectedMultiplierCandidate,
    pub candidates: Vec<SelectedMultiplierCandidate>,
}

/// Ranks sparse multipliers on `train`, refines the best `top_k` on `train`,
/// then selects exactly once using `dev`.
///
/// A refined multiplier replaces its sparse seed only when its development
/// loss is strictly lower. Failed or harmful refinements remain visible in
/// the report and do not invalidate the complete search.
pub fn select_multiplier_train_dev(
    train: &[MultiplierCase],
    dev: &[MultiplierCase],
    top_k: usize,
    relative_scale: f64,
    distortion_weight: f64,
    initial_step: f64,
    tolerance: Tolerance,
) -> Result<MultiplierSelectionResult, String> {
    if top_k == 0
    {
        return Err("top_k must be strictly positive".into());
    }

    let ranked = rank_two_term_multipliers(train, relative_scale, distortion_weight)?;

    let mut candidates = Vec::with_capacity(top_k.min(ranked.len()));

    for (seed_rank, seed) in ranked.into_iter().take(top_k).enumerate()
    {
        let seed_multiplier = seed.multiplier;

        let seed_dev_score =
            score_multiplier(dev, &seed_multiplier, relative_scale, distortion_weight)?;

        let mut candidate = SelectedMultiplierCandidate {
            seed_rank,
            seed_first_index: seed.first_index,
            seed_second_index: seed.second_index,
            seed_second_sign: seed.second_sign,
            seed_multiplier,
            multiplier: seed_multiplier,
            train_score: seed.score,
            dev_score: seed_dev_score,
            refinement_used: false,
            refinement_error: None,
            iterations: None,
            residual: None,
        };

        match optimize_multiplier(
            train,
            seed_multiplier,
            relative_scale,
            distortion_weight,
            initial_step,
            tolerance,
        )
        {
            Ok(refined) =>
            {
                let refined_dev_score =
                    score_multiplier(dev, &refined.multiplier, relative_scale, distortion_weight)?;

                candidate.iterations = Some(refined.iterations);
                candidate.residual = Some(refined.residual);

                if refined_dev_score.loss < candidate.dev_score.loss
                {
                    candidate.multiplier = refined.multiplier;
                    candidate.train_score = refined.score;
                    candidate.dev_score = refined_dev_score;
                    candidate.refinement_used = true;
                }
            },
            Err(error) =>
            {
                candidate.refinement_error = Some(error);
            },
        }

        candidates.push(candidate);
    }

    candidates.sort_by(|left, right| {
        left.dev_score
            .loss
            .total_cmp(&right.dev_score.loss)
            .then_with(|| left.train_score.loss.total_cmp(&right.train_score.loss))
            .then_with(|| left.seed_rank.cmp(&right.seed_rank))
    });

    let selected = candidates
        .first()
        .cloned()
        .ok_or_else(|| "no multiplier candidate was produced".to_string())?;

    Ok(MultiplierSelectionResult {
        selected,
        candidates,
    })
}

#[cfg(test)]
mod tests {
    use super::select_multiplier_train_dev;
    use crate::optimizer::MultiplierCase;
    use crate::scalar::{SEDENION_DIMENSION, Sedenion, basis_vector};
    use scirust_solvers::Tolerance;

    const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];

    fn known_case() -> MultiplierCase {
        let signal = basis_vector(0).expect("e0 exists");

        let mut noise = ZERO;
        noise[4] = 1.0;
        noise[15] = -1.0;

        MultiplierCase::new(signal, noise)
    }

    #[test]
    fn zero_top_k_is_rejected() {
        let case = known_case();

        assert!(
            select_multiplier_train_dev(
                std::slice::from_ref(&case),
                std::slice::from_ref(&case),
                0,
                1.0e-6,
                10.0,
                0.05,
                Tolerance::default(),
            )
            .is_err()
        );
    }

    #[test]
    fn selection_is_deterministic_and_finds_an_annihilator() {
        let case = known_case();
        let tolerance = Tolerance::new(1.0e-5, 1.0e-3, 1);

        let first = select_multiplier_train_dev(
            std::slice::from_ref(&case),
            std::slice::from_ref(&case),
            1,
            1.0e-6,
            10.0,
            0.05,
            tolerance,
        )
        .expect("first selection succeeds");

        let second = select_multiplier_train_dev(
            std::slice::from_ref(&case),
            std::slice::from_ref(&case),
            1,
            1.0e-6,
            10.0,
            0.05,
            tolerance,
        )
        .expect("second selection succeeds");

        assert_eq!(first, second);
        assert_eq!(first.candidates.len(), 1);
        assert!(first.selected.dev_score.loss < 1.0e-10);
        assert!(first.selected.dev_score.mean_noise_ratio < 1.0e-16);
    }
}
