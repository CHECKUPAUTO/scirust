//! Opt-in deterministic structural search using robust source clustering.
//!
//! This module composes:
//!
//! - deterministic complete-link clustering of approximately equal sources;
//! - observed-source medoid canonicalization;
//! - observed-target medoid robust fitting;
//! - train/development selection;
//! - leave-one-out stability with source clusters recomputed after removal.
//!
//! Historical mean-based, exact-source robust and source-clustering fitting
//! APIs remain unchanged.

use crate::{
    SRCC_DIMENSION, SrccCase, SrccConfig, SrccGateDecision, SrccRobustSearchError,
    SrccRobustSourceClusteringConfig, SrccRobustStableSearchError, SrccSearchCandidate,
    SrccSearchResult, SrccStableSearchCandidate, SrccStableSearchConfig, SrccStableSearchResult,
    SrccTransportSample, Vector16, evaluate_source_clustered_robust_leave_one_out_stability,
    fit_source_clustered_robust_srcc_projector_from_views, select_srcc_train_dev,
};

/// Configuration for robust source-clustered structural search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SrccSourceClusteredSearchConfig {
    /// Policy controlling approximate source grouping.
    pub source_clustering: SrccRobustSourceClusteringConfig,
    /// Base SRCC discovery and closure configuration.
    pub base_config: SrccConfig,
    /// Weight assigned to retained-signal distortion during selection.
    pub distortion_weight: f64,
}

/// Searches resonance thresholds using robust source-clustered fitting.
///
/// This function is opt-in. It does not alter exact-source robust search or
/// any historical mean-based search API.
pub fn search_source_clustered_robust_srcc_structures_from_views(
    seeds: &[Vector16],
    views: &[&[SrccTransportSample]],
    resonance_thresholds: &[f64],
    config: SrccSourceClusteredSearchConfig,
    train: &[SrccCase],
    dev: &[SrccCase],
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
        let threshold_config = SrccConfig {
            resonance_threshold,
            ..config.base_config
        };

        let fit = fit_source_clustered_robust_srcc_projector_from_views(
            seeds,
            views,
            config.source_clustering,
            threshold_config,
        )?;

        projectors.push(fit.projector);
    }

    let selection = select_srcc_train_dev(&projectors, train, dev, config.distortion_weight)?;

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

/// Performs stable structural search using robust source-clustered fitting.
///
/// Source clusters and target medoids are recomputed for every leave-one-out
/// variant. An ambiguous source assignment or target consensus remains a typed
/// error and is never silently converted into a stable candidate.
pub fn search_stable_source_clustered_robust_srcc_structures_from_views(
    seeds: &[Vector16],
    views: &[&[SrccTransportSample]],
    source_config: SrccRobustSourceClusteringConfig,
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

    let search = search_source_clustered_robust_srcc_structures_from_views(
        seeds,
        views,
        resonance_thresholds,
        SrccSourceClusteredSearchConfig {
            source_clustering: source_config,
            base_config: config.base_config,
            distortion_weight: config.distortion_weight,
        },
        train,
        dev,
    )?;

    let mut candidates = Vec::with_capacity(search.candidates.len());

    for candidate in search.candidates
    {
        let threshold_config = SrccConfig {
            resonance_threshold: candidate.resonance_threshold,
            ..config.base_config
        };

        let stability = evaluate_source_clustered_robust_leave_one_out_stability(
            seeds,
            views,
            source_config,
            threshold_config,
        )?;

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

    use crate::{
        SrccRobustFitError, SrccRobustStabilityError, basis_vector,
        search_robust_srcc_structures_from_views, search_stable_robust_srcc_structures_from_views,
        squared_norm,
    };

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

    fn source_config(maximum_source_distance: f64) -> SrccRobustSourceClusteringConfig {
        SrccRobustSourceClusteringConfig {
            maximum_source_distance,
        }
    }

    fn search_config(maximum_source_distance: f64) -> SrccSourceClusteredSearchConfig {
        SrccSourceClusteredSearchConfig {
            source_clustering: source_config(maximum_source_distance),
            base_config: base_config(),
            distortion_weight: 10.0,
        }
    }

    fn cases() -> [SrccCase; 1] {
        [SrccCase::new(
            basis_vector(8).unwrap(),
            basis_vector(2).unwrap(),
        )]
    }

    fn jittered_views(
        clean_repetitions: usize,
        jitter: f64,
    ) -> (Vector16, Vec<SrccTransportSample>, Vec<SrccTransportSample>) {
        let source = basis_vector(1).unwrap();
        let perturbation = basis_vector(8).unwrap();
        let target = basis_vector(2).unwrap();
        let target_outlier = basis_vector(9).unwrap();

        let nearby_source = normalize(core::array::from_fn(|index| {
            source[index] + jitter * perturbation[index]
        }));

        let negative_target = target.map(|value| -value);

        let mut positive = Vec::with_capacity(clean_repetitions + 1);

        let mut negative = Vec::with_capacity(clean_repetitions + 1);

        for _ in 0..clean_repetitions
        {
            positive.push(SrccTransportSample::new(source, target));

            negative.push(SrccTransportSample::new(source, negative_target));
        }

        positive.push(SrccTransportSample::new(nearby_source, target_outlier));

        negative.push(SrccTransportSample::new(nearby_source, target_outlier));

        (source, positive, negative)
    }

    #[test]
    fn source_clustered_search_recovers_jittered_structure() {
        let (source, positive, negative) = jittered_views(2, 1.0e-3);

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = cases();

        let result = search_source_clustered_robust_srcc_structures_from_views(
            &[source],
            &views,
            &[1.0, 0.999],
            search_config(1.0e-2),
            &cases,
            &cases,
        )
        .unwrap();

        assert_eq!(result.candidates.len(), 2);

        assert_eq!(result.selected.resonance_threshold, 0.999,);

        assert_eq!(result.selected.projector.rejected_dimension(), 2,);

        assert_eq!(result.decision, SrccGateDecision::Srcc);
        assert!(result.selected.dev_score.loss < 1.0e-24);
    }

    #[test]
    fn source_clustered_search_ignores_threshold_order() {
        let (source, positive, negative) = jittered_views(2, 1.0e-3);

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = cases();

        let first = search_source_clustered_robust_srcc_structures_from_views(
            &[source],
            &views,
            &[1.0, 0.999],
            search_config(1.0e-2),
            &cases,
            &cases,
        )
        .unwrap();

        let second = search_source_clustered_robust_srcc_structures_from_views(
            &[source],
            &views,
            &[0.999, 1.0],
            search_config(1.0e-2),
            &cases,
            &cases,
        )
        .unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn zero_source_radius_matches_exact_robust_search() {
        let (source, positive, negative) = jittered_views(3, 0.0);

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = cases();

        let exact = search_robust_srcc_structures_from_views(
            &[source],
            &views,
            &[1.0, 0.999],
            base_config(),
            &cases,
            &cases,
            10.0,
        )
        .unwrap();

        let clustered = search_source_clustered_robust_srcc_structures_from_views(
            &[source],
            &views,
            &[1.0, 0.999],
            search_config(0.0),
            &cases,
            &cases,
        )
        .unwrap();

        assert_eq!(exact, clustered);
    }

    #[test]
    fn zero_source_radius_matches_exact_robust_stable_search() {
        let (source, positive, negative) = jittered_views(3, 0.0);

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = cases();

        let exact = search_stable_robust_srcc_structures_from_views(
            &[source],
            &views,
            &[1.0, 0.999],
            stable_config(),
            &cases,
            &cases,
        )
        .unwrap();

        let clustered = search_stable_source_clustered_robust_srcc_structures_from_views(
            &[source],
            &views,
            SrccRobustSourceClusteringConfig::default(),
            &[1.0, 0.999],
            stable_config(),
            &cases,
            &cases,
        )
        .unwrap();

        assert_eq!(exact, clustered);
    }

    #[test]
    fn three_clean_samples_pass_source_clustered_stable_search() {
        let (source, positive, negative) = jittered_views(3, 1.0e-3);

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = cases();

        let result = search_stable_source_clustered_robust_srcc_structures_from_views(
            &[source],
            &views,
            source_config(1.0e-2),
            &[1.0, 0.999],
            stable_config(),
            &cases,
            &cases,
        )
        .unwrap();

        assert_eq!(result.decision, SrccGateDecision::Srcc);

        let selected = result.selected.as_ref().unwrap();

        assert!(selected.passes_stability_gate);

        assert_eq!(selected.candidate.projector.rejected_dimension(), 2,);

        assert_eq!(selected.stability.maximum_frobenius_distance, 0.0,);

        assert_eq!(selected.stability.dimension_stability_ratio(), 1.0,);

        assert!(selected.candidate.dev_score.loss < 1.0e-24);
    }

    #[test]
    fn two_clean_samples_return_clustered_loo_ambiguity() {
        let (source, positive, negative) = jittered_views(2, 1.0e-3);

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = cases();

        assert_eq!(
            search_stable_source_clustered_robust_srcc_structures_from_views(
                &[source],
                &views,
                source_config(1.0e-2),
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
    fn invalid_source_distance_is_propagated_by_search() {
        let (source, positive, negative) = jittered_views(3, 1.0e-3);

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = cases();

        assert_eq!(
            search_source_clustered_robust_srcc_structures_from_views(
                &[source],
                &views,
                &[0.999],
                search_config(-1.0),
                &cases,
                &cases,
            ),
            Err(SrccRobustSearchError::Fit(
                SrccRobustFitError::InvalidMaximumSourceDistance,
            ),),
        );
    }

    #[test]
    fn empty_source_clustered_threshold_grid_is_rejected() {
        let (source, positive, negative) = jittered_views(3, 1.0e-3);

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = cases();

        assert_eq!(
            search_source_clustered_robust_srcc_structures_from_views(
                &[source],
                &views,
                &[],
                search_config(1.0e-2),
                &cases,
                &cases,
            ),
            Err(SrccRobustSearchError::EmptyResonanceThresholds,),
        );
    }

    #[test]
    fn invalid_source_clustered_stability_gate_is_rejected() {
        let (source, positive, negative) = jittered_views(3, 1.0e-3);

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = cases();
        let mut config = stable_config();

        config.maximum_frobenius_distance = -1.0;

        assert_eq!(
            search_stable_source_clustered_robust_srcc_structures_from_views(
                &[source],
                &views,
                source_config(1.0e-2),
                &[0.999],
                config,
                &cases,
                &cases,
            ),
            Err(SrccRobustStableSearchError::InvalidMaximumFrobeniusDistance,),
        );
    }
}
