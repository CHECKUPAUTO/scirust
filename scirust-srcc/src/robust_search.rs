//! Opt-in deterministic structural search using robust SRCC fitting.
//!
//! The historical mean-based search APIs remain unchanged. This module
//! provides parallel APIs that use target-medoid fitting and robust
//! leave-one-out stability.

use core::fmt;

use crate::{
    SRCC_DIMENSION, SrccCase, SrccConfig, SrccGateDecision, SrccRobustFitError,
    SrccRobustStabilityError, SrccSearchCandidate, SrccSearchResult, SrccSelectionError,
    SrccStableSearchCandidate, SrccStableSearchConfig, SrccStableSearchResult, SrccTransportSample,
    Vector16, evaluate_robust_leave_one_out_stability, fit_robust_srcc_projector_from_views,
    select_srcc_train_dev,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SrccRobustSearchError {
    EmptyResonanceThresholds,
    Fit(SrccRobustFitError),
    Selection(SrccSelectionError),
}

impl fmt::Display for SrccRobustSearchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyResonanceThresholds =>
            {
                formatter.write_str("resonance-threshold grid must not be empty")
            },
            Self::Fit(error) => error.fmt(formatter),
            Self::Selection(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SrccRobustSearchError {}

impl From<SrccRobustFitError> for SrccRobustSearchError {
    fn from(error: SrccRobustFitError) -> Self {
        Self::Fit(error)
    }
}

impl From<SrccSelectionError> for SrccRobustSearchError {
    fn from(error: SrccSelectionError) -> Self {
        Self::Selection(error)
    }
}

/// Searches resonance thresholds using robust explicit-view fitting.
///
/// This function is opt-in. It does not alter
/// [`crate::search_srcc_structures_from_views`].
pub fn search_robust_srcc_structures_from_views(
    seeds: &[Vector16],
    views: &[&[SrccTransportSample]],
    resonance_thresholds: &[f64],
    base_config: SrccConfig,
    train: &[SrccCase],
    dev: &[SrccCase],
    distortion_weight: f64,
) -> Result<SrccSearchResult, SrccRobustSearchError> {
    if resonance_thresholds.is_empty()
    {
        return Err(SrccRobustSearchError::EmptyResonanceThresholds);
    }

    let mut ordered_thresholds = resonance_thresholds.to_vec();

    ordered_thresholds.sort_by(f64::total_cmp);

    ordered_thresholds.dedup_by(|left, right| left.total_cmp(right).is_eq());

    let mut projectors = Vec::with_capacity(ordered_thresholds.len());

    for &resonance_threshold in &ordered_thresholds
    {
        let config = SrccConfig {
            resonance_threshold,
            ..base_config
        };

        let fit = fit_robust_srcc_projector_from_views(seeds, views, config)?;

        projectors.push(fit.projector);
    }

    let selection = select_srcc_train_dev(&projectors, train, dev, distortion_weight)?;

    let candidates: Vec<_> = selection
        .candidates
        .iter()
        .map(|candidate| {
            let resonance_threshold = ordered_thresholds[candidate.candidate_index];

            SrccSearchCandidate {
                view_count: views.len(),
                resonance_threshold,
                projector: candidate.projector.clone(),
                train_score: candidate.train_score,
                dev_score: candidate.dev_score,
            }
        })
        .collect();

    let selected = candidates[0].clone();

    Ok(SrccSearchResult {
        selected,
        candidates,
        identity_dev_score: selection.identity_dev_score,
        decision: selection.decision,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SrccRobustStableSearchError {
    InvalidMaximumFrobeniusDistance,
    InvalidMinimumDimensionStabilityRatio,
    InvalidMinimumRejectedDimension,
    Search(SrccRobustSearchError),
    Stability(SrccRobustStabilityError),
}

impl fmt::Display for SrccRobustStableSearchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidMaximumFrobeniusDistance =>
            {
                formatter.write_str("maximum Frobenius distance must be finite and non-negative")
            },
            Self::InvalidMinimumDimensionStabilityRatio =>
            {
                formatter.write_str("minimum dimension stability ratio must belong to [0, 1]")
            },
            Self::InvalidMinimumRejectedDimension =>
            {
                formatter.write_str("minimum rejected dimension must not exceed the SRCC dimension")
            },
            Self::Search(error) => error.fmt(formatter),
            Self::Stability(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SrccRobustStableSearchError {}

impl From<SrccRobustSearchError> for SrccRobustStableSearchError {
    fn from(error: SrccRobustSearchError) -> Self {
        Self::Search(error)
    }
}

impl From<SrccRobustStabilityError> for SrccRobustStableSearchError {
    fn from(error: SrccRobustStabilityError) -> Self {
        Self::Stability(error)
    }
}

/// Performs stable structural search with robust fitting and robust LOO.
///
/// A reduced-view target tie is returned as a typed stability error.
/// It is never silently converted into a stable candidate.
pub fn search_stable_robust_srcc_structures_from_views(
    seeds: &[Vector16],
    views: &[&[SrccTransportSample]],
    resonance_thresholds: &[f64],
    config: SrccStableSearchConfig,
    train: &[SrccCase],
    dev: &[SrccCase],
) -> Result<SrccStableSearchResult, SrccRobustStableSearchError> {
    if !config.maximum_frobenius_distance.is_finite() || config.maximum_frobenius_distance < 0.0
    {
        return Err(SrccRobustStableSearchError::InvalidMaximumFrobeniusDistance);
    }

    if !config.minimum_dimension_stability_ratio.is_finite()
        || !(0.0..=1.0).contains(&config.minimum_dimension_stability_ratio)
    {
        return Err(SrccRobustStableSearchError::InvalidMinimumDimensionStabilityRatio);
    }

    if config.minimum_rejected_dimension > SRCC_DIMENSION
    {
        return Err(SrccRobustStableSearchError::InvalidMinimumRejectedDimension);
    }

    let search = search_robust_srcc_structures_from_views(
        seeds,
        views,
        resonance_thresholds,
        config.base_config,
        train,
        dev,
        config.distortion_weight,
    )?;

    let mut candidates = Vec::with_capacity(search.candidates.len());

    for candidate in search.candidates
    {
        let threshold_config = SrccConfig {
            resonance_threshold: candidate.resonance_threshold,
            ..config.base_config
        };

        let stability = evaluate_robust_leave_one_out_stability(seeds, views, threshold_config)?;

        let passes_stability_gate = stability.maximum_frobenius_distance
            <= config.maximum_frobenius_distance
            && stability.dimension_stability_ratio() >= config.minimum_dimension_stability_ratio
            && candidate.projector.rejected_dimension() >= config.minimum_rejected_dimension;

        candidates.push(SrccStableSearchCandidate {
            candidate,
            stability,
            passes_stability_gate,
        });
    }

    let selected = candidates
        .iter()
        .find(|candidate| candidate.passes_stability_gate)
        .cloned();

    let decision = match &selected
    {
        Some(candidate) if candidate.candidate.dev_score.loss < search.identity_dev_score.loss =>
        {
            SrccGateDecision::Srcc
        },
        _ => SrccGateDecision::Identity,
    };

    Ok(SrccStableSearchResult {
        selected,
        candidates,
        identity_dev_score: search.identity_dev_score,
        decision,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{SrccRobustFitError, SrccRobustStabilityError, basis_vector, squared_norm};

    fn normalize(vector: Vector16) -> Vector16 {
        let norm = squared_norm(&vector).sqrt();

        vector.map(|value| value / norm)
    }

    fn base_config() -> SrccConfig {
        SrccConfig {
            novelty_threshold: 1.0e-10,
            resonance_threshold: 0.999,
            minimum_support: 2,
            maximum_dimension: 2,
            maximum_rounds: 2,
            energy_floor: 1.0e-30,
        }
    }

    fn stable_config() -> SrccStableSearchConfig {
        SrccStableSearchConfig {
            base_config: base_config(),
            distortion_weight: 10.0,
            maximum_frobenius_distance: 0.0,
            minimum_dimension_stability_ratio: 1.0,
            minimum_rejected_dimension: 2,
        }
    }

    fn cases() -> [SrccCase; 1] {
        [SrccCase::new(
            basis_vector(8).unwrap(),
            basis_vector(2).unwrap(),
        )]
    }

    fn contaminated_views(
        clean_repetitions: usize,
    ) -> (Vector16, Vec<SrccTransportSample>, Vec<SrccTransportSample>) {
        let source = basis_vector(1).unwrap();
        let target = basis_vector(2).unwrap();
        let contaminant = basis_vector(8).unwrap();
        let epsilon = 1.0e-3;

        let negative = target.map(|value| -value);

        let contaminated_positive = normalize(core::array::from_fn(|index| {
            target[index] + epsilon * contaminant[index]
        }));

        let contaminated_negative = normalize(core::array::from_fn(|index| {
            negative[index] + epsilon * contaminant[index]
        }));

        let mut positive = Vec::with_capacity(clean_repetitions + 1);

        let mut negative_view = Vec::with_capacity(clean_repetitions + 1);

        for _ in 0..clean_repetitions
        {
            positive.push(SrccTransportSample::new(source, target));

            negative_view.push(SrccTransportSample::new(source, negative));
        }

        positive.push(SrccTransportSample::new(source, contaminated_positive));

        negative_view.push(SrccTransportSample::new(source, contaminated_negative));

        (source, positive, negative_view)
    }

    #[test]
    fn robust_search_selects_contamination_free_structure() {
        let (source, positive, negative) = contaminated_views(2);

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = cases();

        let result = search_robust_srcc_structures_from_views(
            &[source],
            &views,
            &[1.0, 0.999],
            base_config(),
            &cases,
            &cases,
            10.0,
        )
        .unwrap();

        assert_eq!(result.candidates.len(), 2);

        assert_eq!(result.selected.resonance_threshold, 0.999,);

        assert_eq!(result.selected.projector.rejected_dimension(), 2,);

        assert_eq!(result.decision, SrccGateDecision::Srcc,);

        assert!(result.selected.dev_score.loss < 1.0e-24);
    }

    #[test]
    fn robust_search_ignores_threshold_order() {
        let (source, positive, negative) = contaminated_views(2);

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = cases();

        let first = search_robust_srcc_structures_from_views(
            &[source],
            &views,
            &[1.0, 0.999],
            base_config(),
            &cases,
            &cases,
            10.0,
        )
        .unwrap();

        let second = search_robust_srcc_structures_from_views(
            &[source],
            &views,
            &[0.999, 1.0],
            base_config(),
            &cases,
            &cases,
            10.0,
        )
        .unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn three_clean_samples_pass_robust_stable_search() {
        let (source, positive, negative) = contaminated_views(3);

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = cases();

        let result = search_stable_robust_srcc_structures_from_views(
            &[source],
            &views,
            &[1.0, 0.999],
            stable_config(),
            &cases,
            &cases,
        )
        .unwrap();

        assert_eq!(result.decision, SrccGateDecision::Srcc,);

        let selected = result.selected.as_ref().unwrap();

        assert!(selected.passes_stability_gate);

        assert_eq!(selected.candidate.projector.rejected_dimension(), 2,);

        assert_eq!(selected.stability.maximum_frobenius_distance, 0.0,);

        assert_eq!(selected.stability.dimension_stability_ratio(), 1.0,);

        assert!(selected.candidate.dev_score.loss < 1.0e-24);
    }

    #[test]
    fn two_clean_samples_return_typed_loo_ambiguity() {
        let (source, positive, negative) = contaminated_views(2);

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = cases();

        assert_eq!(
            search_stable_robust_srcc_structures_from_views(
                &[source],
                &views,
                &[0.999],
                stable_config(),
                &cases,
                &cases,
            ),
            Err(SrccRobustStableSearchError::Stability(
                SrccRobustStabilityError::Fit(SrccRobustFitError::AmbiguousTargetConsensus {
                    view_index: 0,
                    source_group_index: 0,
                },),
            ),),
        );
    }

    #[test]
    fn empty_robust_threshold_grid_is_rejected() {
        let (source, positive, negative) = contaminated_views(3);

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = cases();

        assert_eq!(
            search_robust_srcc_structures_from_views(
                &[source],
                &views,
                &[],
                base_config(),
                &cases,
                &cases,
                10.0,
            ),
            Err(SrccRobustSearchError::EmptyResonanceThresholds,),
        );
    }

    #[test]
    fn invalid_robust_stability_gate_is_rejected() {
        let (source, positive, negative) = contaminated_views(3);

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = cases();

        let mut config = stable_config();
        config.maximum_frobenius_distance = -1.0;

        assert_eq!(
            search_stable_robust_srcc_structures_from_views(
                &[source],
                &views,
                &[0.999],
                config,
                &cases,
                &cases,
            ),
            Err(SrccRobustStableSearchError::InvalidMaximumFrobeniusDistance,),
        );
    }
}
