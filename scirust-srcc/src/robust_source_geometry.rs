//! Opt-in scale-aware source geometry for robust SRCC source clustering.
//!
//! The historical source-clustering pipeline measures source proximity with a
//! fixed-order raw Euclidean norm, so its clustering radius silently depends on
//! the units and magnitudes of each source coordinate. This module adds an
//! explicit, opt-in geometry layer:
//!
//! - [`SrccSourceGeometrySpec::RawEuclidean`] reproduces the historical
//!   behaviour **bit for bit** (it delegates to the exact same frozen distance
//!   body on identical inputs in identical order);
//! - [`SrccSourceGeometrySpec::RobustDiagonal`] fits a per-coordinate robust
//!   scaler (from `scirust-multivariate`, phase 722) on the observed sources
//!   and measures distances in fitted-scale units. Clustering decisions are
//!   then invariant — after refitting — to positive per-coordinate rescaling of
//!   the sources **provided the scaler configuration is itself
//!   scale-covariant**: `minimum_scale = 0` with the `Error` or
//!   `DropDimension` policy. `UnitScale` keeps a degenerate coordinate in raw
//!   units, and a positive `minimum_scale` is an absolute threshold in raw
//!   units; both deliberately re-introduce a scale dependence and void the
//!   invariance.
//!
//! # No leakage
//!
//! The geometry model is fitted **inside** every fit call, from the observed
//! sources of exactly the views being fitted. Leave-one-out stability therefore
//! refits the scaler after every removal: the removed sample never influences
//! the reduced geometry, and the certificate covers the complete pipeline, not
//! a frozen preprocessing result.
//!
//! # What is and is not invariant
//!
//! `RobustDiagonal` clustering decisions are invariant (within floating-point
//! tolerance) to positive per-coordinate rescaling of the *sources* when the
//! scaler is refit on the rescaled data and its configuration is
//! scale-covariant (`minimum_scale = 0`, policy `Error` or `DropDimension` —
//! see above). Nothing here makes the learned
//! transports or the projector invariant to source rescaling — transport
//! learning still sees the raw coordinates — and no affine invariance is
//! claimed. Scale estimation inherits the breakdown limits of the underlying
//! robust statistics: a coordinate whose observations are split half and half
//! between two states can see its scale inflated by the separation itself.
//!
//! Historical entry points are untouched; everything in this module is new and
//! opt-in.

use scirust_multivariate::{Matrix, RobustGeometryError, RobustScaler, RobustScalerConfig};

use crate::robust_source::{
    SourceMetric, canonicalize_source_clusters, compare_vectors, projector_frobenius_distance,
    validate_source_config,
};
use crate::{
    SRCC_DIMENSION, SrccCase, SrccConfig, SrccFitResult, SrccGateDecision, SrccRobustFitError,
    SrccRobustSearchError, SrccRobustSourceClusteringConfig, SrccRobustStabilityError,
    SrccRobustStableSearchError, SrccSearchCandidate, SrccSearchResult, SrccStabilityReport,
    SrccStabilityVariant, SrccStableSearchCandidate, SrccStableSearchConfig,
    SrccStableSearchResult, SrccTransportSample, Vector16, fit_robust_srcc_projector_from_views,
    learn_transport_views, select_srcc_train_dev,
};

/// How source proximity is measured during source clustering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SrccSourceGeometrySpec {
    /// The historical fixed-order raw Euclidean source metric. Selecting this
    /// reproduces the existing source-clustered pipeline bit for bit at every
    /// radius.
    RawEuclidean,
    /// Per-coordinate robust diagonal scaling: a [`RobustScaler`] is refit on
    /// the observed sources of every fit call (including each leave-one-out
    /// reduction) and coordinate differences are divided by the fitted scales.
    /// `maximum_source_distance` is then expressed in fitted-scale units.
    ///
    /// `scaler_config.center` is irrelevant to distances (differences cancel
    /// the location) and is ignored by the metric; the scaler's zero-scale
    /// policy applies: `Error` fails with
    /// [`SrccRobustFitError::DegenerateSourceScale`], `UnitScale` keeps the
    /// coordinate **in raw units** (voiding rescaling invariance for that
    /// coordinate), and `DropDimension` removes it from the metric
    /// (all-dropped fails with
    /// [`SrccRobustFitError::NoActiveSourceDimensions`]). A scale estimate
    /// that overflows, or whose reciprocal overflows, is a typed
    /// [`SrccRobustFitError::NonFiniteSourceScale`].
    RobustDiagonal {
        /// Configuration for the per-coordinate robust scaler.
        scaler_config: RobustScalerConfig,
    },
}

/// Scale-aware source clustering policy: the geometry and the clustering
/// radius travel together because the meaning of `maximum_source_distance`
/// depends on the geometry (raw coordinate units for
/// [`SrccSourceGeometrySpec::RawEuclidean`], fitted-scale units for
/// [`SrccSourceGeometrySpec::RobustDiagonal`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SrccScaleAwareSourceClusteringConfig {
    /// How source proximity is measured.
    pub geometry: SrccSourceGeometrySpec,
    /// Policy controlling approximate source grouping, interpreted in the
    /// geometry's units.
    pub clustering: SrccRobustSourceClusteringConfig,
}

/// Configuration for scale-aware robust source-clustered structural search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SrccScaleAwareSourceClusteredSearchConfig {
    /// Scale-aware source clustering policy.
    pub source: SrccScaleAwareSourceClusteringConfig,
    /// Base SRCC discovery and closure configuration.
    pub base_config: SrccConfig,
    /// Weight assigned to retained-signal distortion during selection.
    pub distortion_weight: f64,
}

/// Fits a robust SRCC projector after deterministic scale-aware source
/// clustering.
///
/// With [`SrccSourceGeometrySpec::RawEuclidean`] this is bit-identical to
/// [`crate::fit_source_clustered_robust_srcc_projector_from_views`]. With
/// [`SrccSourceGeometrySpec::RobustDiagonal`] the per-coordinate scaler is
/// fitted from the observed sources of `views` before clustering, and
/// `maximum_source_distance` is interpreted in fitted-scale units.
pub fn fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
    seeds: &[Vector16],
    views: &[&[SrccTransportSample]],
    source: SrccScaleAwareSourceClusteringConfig,
    config: SrccConfig,
) -> Result<SrccFitResult, SrccRobustFitError> {
    validate_source_config(source.clustering)?;

    // Validate every original observation before any canonicalization,
    // preserving the historical error precedence.
    let _validated_transports = learn_transport_views(views, config.energy_floor)?;

    let metric = fit_source_metric(views, source.geometry)?;

    let clustered_storage = canonicalize_source_clusters(views, source.clustering, &metric)?;

    let clustered_views: Vec<&[SrccTransportSample]> =
        clustered_storage.iter().map(Vec::as_slice).collect();

    fit_robust_srcc_projector_from_views(seeds, &clustered_views, config)
}

/// Evaluates leave-one-out stability while refitting the source geometry and
/// recomputing source clusters after every removal.
///
/// Refitting the geometry inside every reduced fit is essential: the removed
/// sample must not influence the reduced scaler, so the report certifies the
/// complete scale-aware pipeline, not a frozen preprocessing result.
pub fn evaluate_scale_aware_source_clustered_robust_leave_one_out_stability(
    seeds: &[Vector16],
    views: &[&[SrccTransportSample]],
    source: SrccScaleAwareSourceClusteringConfig,
    config: SrccConfig,
) -> Result<SrccStabilityReport, SrccRobustStabilityError> {
    let full = fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
        seeds, views, source, config,
    )?;

    let full_dimension = full.projector.rejected_dimension();
    let mut variants = Vec::new();

    for (view_index, view) in views.iter().enumerate()
    {
        if view.len() <= 1
        {
            continue;
        }

        for sample_index in 0..view.len()
        {
            let mut reduced_views: Vec<Vec<SrccTransportSample>> =
                views.iter().map(|samples| samples.to_vec()).collect();

            reduced_views[view_index].remove(sample_index);

            let reduced_references: Vec<&[SrccTransportSample]> =
                reduced_views.iter().map(Vec::as_slice).collect();

            let reduced = fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
                seeds,
                &reduced_references,
                source,
                config,
            )?;

            variants.push(SrccStabilityVariant {
                removed_view_index: view_index,
                removed_sample_index: sample_index,
                rejected_dimension: reduced.projector.rejected_dimension(),
                frobenius_distance: projector_frobenius_distance(
                    &full.projector,
                    &reduced.projector,
                ),
            });
        }
    }

    if variants.is_empty()
    {
        return Err(SrccRobustStabilityError::NoRemovableSamples);
    }

    let distance_sum = variants
        .iter()
        .map(|variant| variant.frobenius_distance)
        .sum::<f64>();

    let maximum_frobenius_distance = variants
        .iter()
        .map(|variant| variant.frobenius_distance)
        .fold(0.0, f64::max);

    let stable_dimension_count = variants
        .iter()
        .filter(|variant| variant.rejected_dimension == full_dimension)
        .count();

    Ok(SrccStabilityReport {
        full_projector: full.projector,
        mean_frobenius_distance: distance_sum / variants.len() as f64,
        maximum_frobenius_distance,
        stable_dimension_count,
        variants,
    })
}

/// Searches resonance thresholds using scale-aware robust source-clustered
/// fitting.
///
/// This function is opt-in. It does not alter exact-source robust search or
/// any historical source-clustered search API.
pub fn search_scale_aware_source_clustered_robust_srcc_structures_from_views(
    seeds: &[Vector16],
    views: &[&[SrccTransportSample]],
    resonance_thresholds: &[f64],
    config: SrccScaleAwareSourceClusteredSearchConfig,
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

        let fit = fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
            seeds,
            views,
            config.source,
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

/// Performs stable structural search using scale-aware robust source-clustered
/// fitting.
///
/// The source geometry, source clusters and target medoids are recomputed for
/// every leave-one-out variant. An ambiguous source assignment, an ambiguous
/// target consensus, or a degenerate source scale remains a typed error and is
/// never silently converted into a stable candidate.
pub fn search_stable_scale_aware_source_clustered_robust_srcc_structures_from_views(
    seeds: &[Vector16],
    views: &[&[SrccTransportSample]],
    source: SrccScaleAwareSourceClusteringConfig,
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

    let search = search_scale_aware_source_clustered_robust_srcc_structures_from_views(
        seeds,
        views,
        resonance_thresholds,
        SrccScaleAwareSourceClusteredSearchConfig {
            source,
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

        let stability = evaluate_scale_aware_source_clustered_robust_leave_one_out_stability(
            seeds,
            views,
            source,
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

/// Fits the source metric declared by `geometry` from the observed sources of
/// `views`.
///
/// For [`SrccSourceGeometrySpec::RobustDiagonal`], every source across every
/// view is collected and sorted into the crate's canonical vector order before
/// fitting, so the fitted geometry is invariant to both sample order and view
/// order (this matters for the row-order-sensitive standard-deviation method;
/// the order-statistic methods are insensitive anyway). Duplicated sources are
/// deliberately kept: repetition is evidence of an operating state and must
/// weigh on the scale estimate.
fn fit_source_metric(
    views: &[&[SrccTransportSample]],
    geometry: SrccSourceGeometrySpec,
) -> Result<SourceMetric, SrccRobustFitError> {
    match geometry
    {
        SrccSourceGeometrySpec::RawEuclidean => Ok(SourceMetric::Raw),
        SrccSourceGeometrySpec::RobustDiagonal { scaler_config } =>
        {
            let mut sources: Vec<Vector16> = views
                .iter()
                .flat_map(|view| view.iter().map(|sample| sample.source))
                .collect();

            sources.sort_by(compare_vectors);

            let data = Matrix {
                rows: sources.len(),
                cols: SRCC_DIMENSION,
                data: sources.iter().map(|source| source.to_vec()).collect(),
            };

            let scaler = RobustScaler::fit(&data, scaler_config).map_err(map_scaler_error)?;

            let mut inverse_scales = [0.0; SRCC_DIMENSION];

            for (index, inverse_scale) in inverse_scales.iter_mut().enumerate()
            {
                if scaler.active_dimensions[index]
                {
                    let inverse = 1.0 / scaler.scale[index];

                    // A subnormal fitted scale makes the reciprocal overflow to
                    // infinity, which would poison every distance with a
                    // misattributed non-finite-distance error; surface the
                    // overflow where it happens instead.
                    if !(inverse.is_finite() && inverse > 0.0)
                    {
                        return Err(SrccRobustFitError::NonFiniteSourceScale { dimension: index });
                    }

                    *inverse_scale = inverse;
                }
            }

            Ok(SourceMetric::Diagonal { inverse_scales })
        },
    }
}

/// Maps a scaler-fitting failure onto the crate's typed fit errors.
///
/// The mapping is deliberately coarse where the upstream payload cannot be
/// carried (`SrccRobustFitError` derives `Eq`; the upstream error does not):
/// degenerate-scale and no-active-dimension failures keep their structure,
/// every other configuration or validation failure becomes
/// [`SrccRobustFitError::InvalidSourceGeometry`].
fn map_scaler_error(error: RobustGeometryError) -> SrccRobustFitError {
    match error
    {
        RobustGeometryError::DegenerateDimension { dimension, .. } =>
        {
            SrccRobustFitError::DegenerateSourceScale { dimension }
        },
        RobustGeometryError::NonFiniteScale { dimension, .. } =>
        {
            SrccRobustFitError::NonFiniteSourceScale { dimension }
        },
        RobustGeometryError::NoActiveDimensions => SrccRobustFitError::NoActiveSourceDimensions,
        _ => SrccRobustFitError::InvalidSourceGeometry,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use scirust_multivariate::{RobustScaleMethod, ZeroScalePolicy};

    use crate::{
        SrccSourceClusteredSearchConfig, basis_vector,
        evaluate_source_clustered_robust_leave_one_out_stability,
        fit_source_clustered_robust_srcc_projector_from_views,
        search_source_clustered_robust_srcc_structures_from_views,
        search_stable_source_clustered_robust_srcc_structures_from_views, squared_norm,
    };

    fn test_config() -> SrccConfig {
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
            base_config: test_config(),
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

    fn scale_aware(
        geometry: SrccSourceGeometrySpec,
        maximum_source_distance: f64,
    ) -> SrccScaleAwareSourceClusteringConfig {
        SrccScaleAwareSourceClusteringConfig {
            geometry,
            clustering: source_config(maximum_source_distance),
        }
    }

    fn diagonal_geometry() -> SrccSourceGeometrySpec {
        SrccSourceGeometrySpec::RobustDiagonal {
            scaler_config: RobustScalerConfig {
                center: true,
                scale_method: RobustScaleMethod::MedianAbsoluteDeviation,
                zero_scale_policy: ZeroScalePolicy::DropDimension,
                minimum_scale: 0.0,
            },
        }
    }

    fn normalize(vector: Vector16) -> Vector16 {
        let norm = squared_norm(&vector).sqrt();

        vector.map(|value| value / norm)
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

    /// Two operating states whose meaningful separation lives on a coordinate
    /// three orders of magnitude smaller than a pure-noise coordinate.
    ///
    /// Coordinate 0 carries pure jitter of magnitude ~`0.02` around `1.0`;
    /// coordinate 8 carries the state signal: `0.001` (state A, majority of
    /// six) vs `0.002` (state B, minority of three) with jitter `1e-6`. In raw
    /// Euclidean coordinates the smallest inter-state distance (the raw
    /// coordinate-8 separation, `0.001`, reached at identical coordinate-0
    /// jitter) is far below the largest intra-state distance (`0.04` of pure
    /// coordinate-0 jitter), so **no raw radius groups the intra-state jitter
    /// while keeping the states apart**. Concretely the raw pipeline has three
    /// regimes: below the coordinate-8 separation it fragments every sample
    /// into singletons (no grouping — the fit still succeeds via exact
    /// repetition across views); in a middle bridging band it mixes the states
    /// inside one cluster and fails with a typed consensus ambiguity; above
    /// the intra-state spread it merges everything and the six-to-three
    /// majority silently outvotes state B (an `Ok` with state B's targets
    /// discarded). In fitted MAD units the majority state keeps the
    /// coordinate-8 scale at the jitter level, so the states are hundreds of
    /// scale units apart — and the six-to-three majority keeps that true for
    /// every single-sample removal (a five-to-three reduction still leaves the
    /// small deviations in the majority), which the leave-one-out tests rely
    /// on.
    fn anisotropic_state_views() -> (Vector16, Vec<SrccTransportSample>, Vec<SrccTransportSample>) {
        let target_a = basis_vector(2).unwrap();
        let target_b = basis_vector(3).unwrap();

        let source_a = |noise: f64, epsilon: f64| -> Vector16 {
            let mut source = [0.0; SRCC_DIMENSION];
            source[0] = 1.0 + noise;
            source[8] = 0.001 + epsilon;
            source
        };

        let source_b = |noise: f64, epsilon: f64| -> Vector16 {
            let mut source = [0.0; SRCC_DIMENSION];
            source[0] = 1.0 + noise;
            source[8] = 0.002 + epsilon;
            source
        };

        // Majority state A (six repetitions) with deterministic jitter,
        // minority state B (three repetitions) sharing three of A's
        // coordinate-0 jitter values, so the smallest raw inter-state distance
        // is exactly the coordinate-8 separation (0.001) while the largest raw
        // intra-state distance is the coordinate-0 jitter spread (0.04).
        let a_noise = [-0.02, -0.01, 0.0, 0.005, 0.01, 0.02];
        let a_epsilon = [0.0, 1.0e-6, -1.0e-6, 5.0e-7, -5.0e-7, 2.0e-7];
        let b_noise = [-0.02, 0.0, 0.02];
        let b_epsilon = [0.0, -1.0e-6, 1.0e-6];

        let mut view = Vec::with_capacity(a_noise.len() + b_noise.len());

        for (noise, epsilon) in a_noise.iter().zip(a_epsilon.iter())
        {
            view.push(SrccTransportSample::new(
                source_a(*noise, *epsilon),
                target_a,
            ));
        }

        for (noise, epsilon) in b_noise.iter().zip(b_epsilon.iter())
        {
            view.push(SrccTransportSample::new(
                source_b(*noise, *epsilon),
                target_b,
            ));
        }

        (source_a(0.0, 0.0), view.clone(), view)
    }

    #[test]
    fn raw_euclidean_fit_matches_source_clustered_fit() {
        for (clean_repetitions, jitter, radius) in
            [(3, 0.0, 0.0), (3, 1.0e-3, 1.0e-2), (2, 1.0e-3, 1.0e-2)]
        {
            let (source, positive, negative) = jittered_views(clean_repetitions, jitter);

            let views = [positive.as_slice(), negative.as_slice()];

            let historical = fit_source_clustered_robust_srcc_projector_from_views(
                &[source],
                &views,
                source_config(radius),
                test_config(),
            );

            let scale_aware_result =
                fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
                    &[source],
                    &views,
                    scale_aware(SrccSourceGeometrySpec::RawEuclidean, radius),
                    test_config(),
                );

            assert_eq!(historical, scale_aware_result);
        }
    }

    #[test]
    fn raw_euclidean_loo_matches_source_clustered_loo() {
        let (source, positive, negative) = jittered_views(3, 1.0e-3);

        let views = [positive.as_slice(), negative.as_slice()];

        let historical = evaluate_source_clustered_robust_leave_one_out_stability(
            &[source],
            &views,
            source_config(1.0e-2),
            test_config(),
        );

        let scale_aware_result =
            evaluate_scale_aware_source_clustered_robust_leave_one_out_stability(
                &[source],
                &views,
                scale_aware(SrccSourceGeometrySpec::RawEuclidean, 1.0e-2),
                test_config(),
            );

        assert_eq!(historical, scale_aware_result);
    }

    #[test]
    fn raw_euclidean_search_matches_source_clustered_search() {
        let (source, positive, negative) = jittered_views(2, 1.0e-3);

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = cases();

        let historical = search_source_clustered_robust_srcc_structures_from_views(
            &[source],
            &views,
            &[1.0, 0.999],
            SrccSourceClusteredSearchConfig {
                source_clustering: source_config(1.0e-2),
                base_config: test_config(),
                distortion_weight: 10.0,
            },
            &cases,
            &cases,
        );

        let scale_aware_result =
            search_scale_aware_source_clustered_robust_srcc_structures_from_views(
                &[source],
                &views,
                &[1.0, 0.999],
                SrccScaleAwareSourceClusteredSearchConfig {
                    source: scale_aware(SrccSourceGeometrySpec::RawEuclidean, 1.0e-2),
                    base_config: test_config(),
                    distortion_weight: 10.0,
                },
                &cases,
                &cases,
            );

        assert_eq!(historical, scale_aware_result);
    }

    #[test]
    fn raw_euclidean_stable_search_matches_source_clustered_stable_search() {
        let (source, positive, negative) = jittered_views(3, 1.0e-3);

        let views = [positive.as_slice(), negative.as_slice()];

        let cases = cases();

        let historical = search_stable_source_clustered_robust_srcc_structures_from_views(
            &[source],
            &views,
            source_config(1.0e-2),
            &[1.0, 0.999],
            stable_config(),
            &cases,
            &cases,
        );

        let scale_aware_result =
            search_stable_scale_aware_source_clustered_robust_srcc_structures_from_views(
                &[source],
                &views,
                scale_aware(SrccSourceGeometrySpec::RawEuclidean, 1.0e-2),
                &[1.0, 0.999],
                stable_config(),
                &cases,
                &cases,
            );

        assert_eq!(historical, scale_aware_result);
    }

    #[test]
    fn raw_euclidean_zero_radius_matches_exact_robust_fit() {
        let source = basis_vector(1).unwrap();
        let target = basis_vector(2).unwrap();

        let positive = [
            SrccTransportSample::new(source, target),
            SrccTransportSample::new(source, target),
        ];

        let negative_target = target.map(|value| -value);

        let negative = [
            SrccTransportSample::new(source, negative_target),
            SrccTransportSample::new(source, negative_target),
        ];

        let views = [positive.as_slice(), negative.as_slice()];

        let exact = fit_robust_srcc_projector_from_views(&[source], &views, test_config()).unwrap();

        let scale_aware_result = fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
            &[source],
            &views,
            SrccScaleAwareSourceClusteringConfig {
                geometry: SrccSourceGeometrySpec::RawEuclidean,
                clustering: SrccRobustSourceClusteringConfig::default(),
            },
            test_config(),
        )
        .unwrap();

        assert_eq!(exact, scale_aware_result);
    }

    #[test]
    fn anisotropic_states_expose_raw_euclidean_three_regime_failure() {
        // No raw radius groups the intra-state jitter while separating the
        // states. The three regimes are pinned explicitly:
        //
        // - a radius below the coordinate-8 separation fragments every jittered
        //   sample into a singleton (no grouping happens; the fit succeeds only
        //   because the fixture repeats sources across views);
        // - a bridging radius mixes the states inside one cluster and fails
        //   with the typed consensus ambiguity;
        // - a radius above the intra-state spread merges everything and the
        //   six-to-three majority silently outvotes state B (an Ok whose
        //   clustering has destroyed the two-state structure).
        //
        // Only the robust-diagonal geometry achieves grouped-and-separated.
        let (seed, view_a, view_b) = anisotropic_state_views();

        let views = [view_a.as_slice(), view_b.as_slice()];

        let raw_at = |radius: f64| {
            fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
                &[seed],
                &views,
                scale_aware(SrccSourceGeometrySpec::RawEuclidean, radius),
                test_config(),
            )
        };

        // Fragmenting regime: succeeds, but no approximate grouping occurred.
        assert!(raw_at(5.0e-4).is_ok());

        // Bridging regime: typed ambiguity.
        assert!(matches!(
            raw_at(2.0e-2),
            Err(SrccRobustFitError::AmbiguousTargetConsensus { .. })
        ));

        // Merging regime: Ok, with the minority state silently outvoted.
        assert!(raw_at(1.0e-1).is_ok());
    }

    #[test]
    fn robust_diagonal_recovers_anisotropic_states() {
        // In fitted MAD units the states are far apart on coordinate 8 (the
        // majority state keeps that scale at jitter level) while intra-state
        // spread stays small, so a moderate scaled radius clusters cleanly.
        let (seed, view_a, view_b) = anisotropic_state_views();

        let views = [view_a.as_slice(), view_b.as_slice()];

        let result = fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
            &[seed],
            &views,
            scale_aware(diagonal_geometry(), 10.0),
            test_config(),
        )
        .unwrap();

        // Both operating states survive as separate source groups, giving the
        // closure the two orthogonal target directions it needs to reject the
        // configured maximum dimension.
        assert_eq!(result.projector.rejected_dimension(), 2);
    }

    #[test]
    fn robust_diagonal_is_invariant_to_coordinate_rescaling() {
        // Rescale every source coordinate by an independent positive factor
        // and refit: the scale-aware clustering must reach the same structure,
        // demonstrated end to end by identical rejected dimension, transport
        // count, and success, at the SAME scaled radius.
        let (seed, view_a, _) = anisotropic_state_views();

        let factors: Vector16 = core::array::from_fn(|index| match index
        {
            0 => 1.0e-6,
            8 => 1.0e3,
            _ => 1.0,
        });

        let rescale = |sample: &SrccTransportSample| -> SrccTransportSample {
            let mut source = sample.source;

            for (value, factor) in source.iter_mut().zip(factors.iter())
            {
                *value *= factor;
            }

            SrccTransportSample::new(source, sample.target)
        };

        let rescaled_view: Vec<SrccTransportSample> = view_a.iter().map(rescale).collect();

        let mut rescaled_seed = seed;

        for (value, factor) in rescaled_seed.iter_mut().zip(factors.iter())
        {
            *value *= factor;
        }

        let original_views = [view_a.as_slice(), view_a.as_slice()];
        let rescaled_views = [rescaled_view.as_slice(), rescaled_view.as_slice()];

        let original = fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
            &[seed],
            &original_views,
            scale_aware(diagonal_geometry(), 10.0),
            test_config(),
        )
        .unwrap();

        let rescaled = fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
            &[rescaled_seed],
            &rescaled_views,
            scale_aware(diagonal_geometry(), 10.0),
            test_config(),
        )
        .unwrap();

        // The clustering structure is preserved: the fit succeeds at the same
        // scaled radius and rejects the same dimension. The projector itself
        // is not compared: transport learning sees raw coordinates, so it is
        // not (and is not claimed to be) rescaling invariant.
        assert_eq!(
            original.projector.rejected_dimension(),
            rescaled.projector.rejected_dimension(),
        );

        // Exact structural check, not a proxy: canonicalizing the rescaled
        // views must yield exactly the canonicalized original views with every
        // rewritten source multiplied coordinate-wise by the factors (the
        // medoid is an observed source, so the correspondence is bit-exact:
        // both sides multiply the same f64 values by the same factors).
        let original_metric = fit_source_metric(&original_views, diagonal_geometry()).unwrap();
        let rescaled_metric = fit_source_metric(&rescaled_views, diagonal_geometry()).unwrap();

        let original_canonical =
            canonicalize_source_clusters(&original_views, source_config(10.0), &original_metric)
                .unwrap();

        let rescaled_canonical =
            canonicalize_source_clusters(&rescaled_views, source_config(10.0), &rescaled_metric)
                .unwrap();

        let expected_canonical: Vec<Vec<SrccTransportSample>> = original_canonical
            .iter()
            .map(|view| view.iter().map(rescale).collect())
            .collect();

        assert_eq!(rescaled_canonical, expected_canonical);
    }

    #[test]
    fn robust_diagonal_is_invariant_to_sample_and_view_order() {
        let (seed, view_a, view_b) = anisotropic_state_views();

        let views = [view_a.as_slice(), view_b.as_slice()];

        let first = fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
            &[seed],
            &views,
            scale_aware(diagonal_geometry(), 10.0),
            test_config(),
        )
        .unwrap();

        let mut reversed_a = view_a.clone();
        let mut reversed_b = view_b.clone();

        reversed_a.reverse();
        reversed_b.reverse();

        let reversed_views = [reversed_a.as_slice(), reversed_b.as_slice()];

        let second = fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
            &[seed],
            &reversed_views,
            scale_aware(diagonal_geometry(), 10.0),
            test_config(),
        )
        .unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn robust_diagonal_fit_is_deterministic() {
        let (seed, view_a, view_b) = anisotropic_state_views();

        let views = [view_a.as_slice(), view_b.as_slice()];

        let first = fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
            &[seed],
            &views,
            scale_aware(diagonal_geometry(), 10.0),
            test_config(),
        );

        let second = fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
            &[seed],
            &views,
            scale_aware(diagonal_geometry(), 10.0),
            test_config(),
        );

        assert_eq!(first, second);
    }

    #[test]
    fn error_zero_scale_policy_reports_degenerate_coordinate() {
        // Coordinate 1 is identically zero across every observed source, so
        // its MAD is zero; the Error policy must surface the first degenerate
        // coordinate as a typed failure.
        let (seed, view_a, view_b) = anisotropic_state_views();

        let views = [view_a.as_slice(), view_b.as_slice()];

        let strict = SrccSourceGeometrySpec::RobustDiagonal {
            scaler_config: RobustScalerConfig {
                center: true,
                scale_method: RobustScaleMethod::MedianAbsoluteDeviation,
                zero_scale_policy: ZeroScalePolicy::Error,
                minimum_scale: 0.0,
            },
        };

        assert_eq!(
            fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
                &[seed],
                &views,
                scale_aware(strict, 10.0),
                test_config(),
            ),
            Err(SrccRobustFitError::DegenerateSourceScale { dimension: 1 }),
        );
    }

    #[test]
    fn invalid_scaler_config_is_a_typed_geometry_error() {
        let (seed, view_a, view_b) = anisotropic_state_views();

        let views = [view_a.as_slice(), view_b.as_slice()];

        let invalid = SrccSourceGeometrySpec::RobustDiagonal {
            scaler_config: RobustScalerConfig {
                center: true,
                scale_method: RobustScaleMethod::MedianAbsoluteDeviation,
                zero_scale_policy: ZeroScalePolicy::DropDimension,
                minimum_scale: -1.0,
            },
        };

        assert_eq!(
            fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
                &[seed],
                &views,
                scale_aware(invalid, 10.0),
                test_config(),
            ),
            Err(SrccRobustFitError::InvalidSourceGeometry),
        );
    }

    #[test]
    fn invalid_radius_precedes_geometry_errors() {
        // Historical error precedence: the clustering-radius validation fires
        // before any geometry fitting, even when the geometry is also invalid.
        let (seed, view_a, view_b) = anisotropic_state_views();

        let views = [view_a.as_slice(), view_b.as_slice()];

        let invalid = SrccSourceGeometrySpec::RobustDiagonal {
            scaler_config: RobustScalerConfig {
                center: true,
                scale_method: RobustScaleMethod::MedianAbsoluteDeviation,
                zero_scale_policy: ZeroScalePolicy::DropDimension,
                minimum_scale: -1.0,
            },
        };

        assert_eq!(
            fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
                &[seed],
                &views,
                scale_aware(invalid, -1.0),
                test_config(),
            ),
            Err(SrccRobustFitError::InvalidMaximumSourceDistance),
        );
    }

    #[test]
    fn fitted_metric_is_invariant_to_view_and_sample_order_even_for_std_dev() {
        // The standard-deviation scale is summation-order sensitive, so this
        // guards the canonical global sort in fit_source_metric: deleting the
        // sort makes the pooled accumulation order depend on view/sample
        // order and this test fails.
        let (_, view_a, _) = anisotropic_state_views();

        let mut view_b: Vec<SrccTransportSample> = view_a.clone();
        view_b.truncate(6);

        let mut shuffled_a = view_a.clone();
        shuffled_a.reverse();
        shuffled_a.swap(0, 3);

        let geometry = SrccSourceGeometrySpec::RobustDiagonal {
            scaler_config: RobustScalerConfig {
                center: true,
                scale_method: RobustScaleMethod::StandardDeviation,
                zero_scale_policy: ZeroScalePolicy::DropDimension,
                minimum_scale: 0.0,
            },
        };

        let forward: [&[SrccTransportSample]; 2] = [view_a.as_slice(), view_b.as_slice()];
        let swapped: [&[SrccTransportSample]; 2] = [view_b.as_slice(), view_a.as_slice()];
        let shuffled: [&[SrccTransportSample]; 2] = [shuffled_a.as_slice(), view_b.as_slice()];

        let reference = fit_source_metric(&forward, geometry).unwrap();

        assert_eq!(reference, fit_source_metric(&swapped, geometry).unwrap());
        assert_eq!(reference, fit_source_metric(&shuffled, geometry).unwrap());
    }

    #[test]
    fn unit_scale_policy_keeps_degenerate_coordinates_in_raw_units() {
        // UnitScale marks every degenerate coordinate active with scale 1.0.
        // On this fixture the degenerate coordinates are identically zero in
        // every source, so they contribute |0 - 0| = 0 and the clustering
        // matches the DropDimension outcome; the policy is exercised
        // end to end and documented as voiding rescaling invariance.
        let (seed, view_a, view_b) = anisotropic_state_views();

        let views = [view_a.as_slice(), view_b.as_slice()];

        let unit_scale = SrccSourceGeometrySpec::RobustDiagonal {
            scaler_config: RobustScalerConfig {
                center: true,
                scale_method: RobustScaleMethod::MedianAbsoluteDeviation,
                zero_scale_policy: ZeroScalePolicy::UnitScale,
                minimum_scale: 0.0,
            },
        };

        let result = fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
            &[seed],
            &views,
            scale_aware(unit_scale, 10.0),
            test_config(),
        )
        .unwrap();

        assert_eq!(result.projector.rejected_dimension(), 2);
    }

    #[test]
    fn identical_sources_drop_every_dimension_with_typed_error() {
        // Every observed source identical: all 16 coordinate MADs are zero,
        // DropDimension deactivates everything, and the typed
        // NoActiveSourceDimensions error fires at geometry fitting.
        let source = basis_vector(1).unwrap();
        let target = basis_vector(2).unwrap();

        let samples = [
            SrccTransportSample::new(source, target),
            SrccTransportSample::new(source, target),
        ];

        let views = [samples.as_slice(), samples.as_slice()];

        assert_eq!(
            fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
                &[source],
                &views,
                scale_aware(diagonal_geometry(), 10.0),
                test_config(),
            ),
            Err(SrccRobustFitError::NoActiveSourceDimensions),
        );
    }

    #[test]
    fn discovery_errors_precede_geometry_fitting() {
        // Historical error precedence: transport validation runs on the
        // original views before any geometry fitting, so a degenerate sample
        // surfaces as a discovery error even when the geometry is also
        // invalid.
        let zero_source = [0.0; SRCC_DIMENSION];
        let target = basis_vector(2).unwrap();

        let samples = [
            SrccTransportSample::new(zero_source, target),
            SrccTransportSample::new(zero_source, target),
        ];

        let views = [samples.as_slice(), samples.as_slice()];

        let invalid = SrccSourceGeometrySpec::RobustDiagonal {
            scaler_config: RobustScalerConfig {
                center: true,
                scale_method: RobustScaleMethod::MedianAbsoluteDeviation,
                zero_scale_policy: ZeroScalePolicy::DropDimension,
                minimum_scale: -1.0,
            },
        };

        assert!(matches!(
            fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
                &[basis_vector(1).unwrap()],
                &views,
                scale_aware(invalid, 10.0),
                test_config(),
            ),
            Err(SrccRobustFitError::Discovery(_)),
        ));
    }

    #[test]
    fn balanced_removal_breaks_robust_scale_with_typed_error() {
        // The geometry is fitted on the sources pooled across views. With an
        // unbalanced view pair (four-to-three plus three-to-three) the pooled
        // majority is only 7 vs 6: removing one majority sample balances the
        // pooled states 6-6, the coordinate-8 MAD inflates to the separation
        // itself, the reduced clustering merges the states, and the tied
        // target consensus surfaces as a typed error. This is the documented
        // breakdown limit of the robust scale - never a silent certificate.
        let (seed, view_a, _) = anisotropic_state_views();

        let mut three_a: Vec<SrccTransportSample> = Vec::new();
        let mut four_a: Vec<SrccTransportSample> = Vec::new();

        // view_a holds six A samples then three B samples; keep the first
        // four (resp. three) A samples plus every B sample.
        for (index, sample) in view_a.iter().enumerate()
        {
            if !(4..6).contains(&index)
            {
                four_a.push(*sample);
            }

            if !(3..6).contains(&index)
            {
                three_a.push(*sample);
            }
        }

        let views = [four_a.as_slice(), three_a.as_slice()];

        let full = fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
            &[seed],
            &views,
            scale_aware(diagonal_geometry(), 10.0),
            test_config(),
        );

        assert!(full.is_ok());

        assert!(matches!(
            evaluate_scale_aware_source_clustered_robust_leave_one_out_stability(
                &[seed],
                &views,
                scale_aware(diagonal_geometry(), 10.0),
                test_config(),
            ),
            Err(SrccRobustStabilityError::Fit(
                SrccRobustFitError::AmbiguousTargetConsensus { .. },
            )),
        ));
    }

    #[test]
    fn robust_diagonal_stable_search_certifies_anisotropic_states() {
        let (seed, view_a, view_b) = anisotropic_state_views();

        let views = [view_a.as_slice(), view_b.as_slice()];

        let cases = cases();

        // The jittered sources mean a removal can change the observed source
        // medoid, so the reduced projector legitimately moves by a small
        // Frobenius distance; the gate bounds it instead of demanding an exact
        // zero (which only exactly-repeated sources can achieve). Dimension
        // stability must still be perfect: the six-to-three majority keeps the
        // two-state structure identifiable under every single removal.
        let config = SrccStableSearchConfig {
            maximum_frobenius_distance: 1.0,
            ..stable_config()
        };

        let result = search_stable_scale_aware_source_clustered_robust_srcc_structures_from_views(
            &[seed],
            &views,
            scale_aware(diagonal_geometry(), 10.0),
            &[0.999],
            config,
            &cases,
            &cases,
        )
        .unwrap();

        let selected = result.selected.as_ref().unwrap();

        assert!(selected.passes_stability_gate);
        assert!(selected.stability.maximum_frobenius_distance <= 1.0);
        assert_eq!(selected.stability.dimension_stability_ratio(), 1.0);
    }
}
