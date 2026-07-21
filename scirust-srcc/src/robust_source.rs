//! Opt-in deterministic clustering of approximately equal SRCC sources.
//!
//! The historical robust fitter groups only exactly equal source vectors.
//! This module adds a preprocessing layer that:
//!
//! 1. validates the original observations;
//! 2. builds deterministic complete-link source clusters;
//! 3. selects an observed source medoid for each cluster;
//! 4. rewrites cluster members to that observed source;
//! 5. delegates target aggregation and fitting to the historical robust fitter.
//!
//! No synthetic source or target vector is created.

use core::cmp::Ordering;

use crate::{
    SRCC_DIMENSION, SrccConfig, SrccFitResult, SrccProjector, SrccRobustFitError,
    SrccRobustStabilityError, SrccStabilityReport, SrccStabilityVariant, SrccTransportSample,
    Vector16, fit_robust_srcc_projector_from_views, learn_transport_views,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SrccRobustSourceClusteringConfig {
    /// Maximum Euclidean distance allowed between every pair of sources
    /// belonging to the same complete-link cluster.
    pub maximum_source_distance: f64,
}

impl Default for SrccRobustSourceClusteringConfig {
    fn default() -> Self {
        Self {
            maximum_source_distance: 0.0,
        }
    }
}

/// Source metric used by the clustering helpers.
///
/// `Raw` delegates to the historical [`source_distance`] body, so every
/// existing entry point keeps its bit-identical behaviour. `Diagonal` is the
/// opt-in scale-aware variant used by
/// [`crate::robust_source_geometry`]: coordinate differences are multiplied by
/// fitted inverse scales before entering the exact same fixed-order scaled
/// accumulation. An inverse scale of `0.0` marks a dropped coordinate, which is
/// skipped structurally before any arithmetic — so even an overflowing raw
/// difference on a dropped coordinate contributes nothing (`inf * 0.0` would
/// otherwise be `NaN`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum SourceMetric {
    Raw,
    Diagonal { inverse_scales: Vector16 },
}

impl SourceMetric {
    pub(crate) fn distance(&self, left: &Vector16, right: &Vector16) -> f64 {
        match self
        {
            Self::Raw => source_distance(left, right),
            Self::Diagonal { inverse_scales } =>
            {
                scaled_source_distance(left, right, inverse_scales)
            },
        }
    }
}

/// Fits a robust SRCC projector after deterministic source clustering.
///
/// A sample joins a source cluster only when its source is within
/// `maximum_source_distance` of every existing cluster member.
///
/// If a source is compatible with multiple already distinct complete-link
/// clusters, fitting fails explicitly rather than choosing one arbitrarily.
///
/// Each cluster is represented by an observed source medoid. The canonical
/// lexicographically smallest observed source is retained when several source
/// medoids have the same score.
///
/// Target aggregation remains the observed-target medoid procedure implemented
/// by [`crate::fit_robust_srcc_projector_from_views`].
pub fn fit_source_clustered_robust_srcc_projector_from_views(
    seeds: &[Vector16],
    views: &[&[SrccTransportSample]],
    source_config: SrccRobustSourceClusteringConfig,
    config: SrccConfig,
) -> Result<SrccFitResult, SrccRobustFitError> {
    validate_source_config(source_config)?;

    // Validate every original observation before any canonicalization.
    let _validated_transports = learn_transport_views(views, config.energy_floor)?;

    let clustered_storage = canonicalize_source_clusters(views, source_config, &SourceMetric::Raw)?;

    let clustered_views: Vec<&[SrccTransportSample]> =
        clustered_storage.iter().map(Vec::as_slice).collect();

    fit_robust_srcc_projector_from_views(seeds, &clustered_views, config)
}

/// Evaluates leave-one-out stability while recomputing source clusters after
/// every removal.
///
/// Recomputing the clusters is essential: the report certifies the complete
/// source-clustering and target-medoid pipeline, not a frozen preprocessing
/// result.
pub fn evaluate_source_clustered_robust_leave_one_out_stability(
    seeds: &[Vector16],
    views: &[&[SrccTransportSample]],
    source_config: SrccRobustSourceClusteringConfig,
    config: SrccConfig,
) -> Result<SrccStabilityReport, SrccRobustStabilityError> {
    let full =
        fit_source_clustered_robust_srcc_projector_from_views(seeds, views, source_config, config)?;

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

            let reduced = fit_source_clustered_robust_srcc_projector_from_views(
                seeds,
                &reduced_references,
                source_config,
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

pub(crate) fn validate_source_config(
    config: SrccRobustSourceClusteringConfig,
) -> Result<(), SrccRobustFitError> {
    if !config.maximum_source_distance.is_finite() || config.maximum_source_distance < 0.0
    {
        return Err(SrccRobustFitError::InvalidMaximumSourceDistance);
    }

    Ok(())
}

pub(crate) fn canonicalize_source_clusters(
    views: &[&[SrccTransportSample]],
    config: SrccRobustSourceClusteringConfig,
    metric: &SourceMetric,
) -> Result<Vec<Vec<SrccTransportSample>>, SrccRobustFitError> {
    let mut result = Vec::with_capacity(views.len());

    for (view_index, view) in views.iter().enumerate()
    {
        let mut ordered_samples = view.to_vec();
        ordered_samples.sort_by(compare_samples);

        let clusters = build_complete_link_clusters(
            ordered_samples,
            view_index,
            config.maximum_source_distance,
            metric,
        )?;

        let mut canonical_view = Vec::with_capacity(view.len());

        for (source_cluster_index, cluster) in clusters.iter().enumerate()
        {
            let representative_source =
                source_medoid(cluster, view_index, source_cluster_index, metric)?;

            canonical_view.extend(
                cluster
                    .iter()
                    .map(|sample| SrccTransportSample::new(representative_source, sample.target)),
            );
        }

        result.push(canonical_view);
    }

    Ok(result)
}

pub(crate) fn build_complete_link_clusters(
    ordered_samples: Vec<SrccTransportSample>,
    view_index: usize,
    maximum_source_distance: f64,
    metric: &SourceMetric,
) -> Result<Vec<Vec<SrccTransportSample>>, SrccRobustFitError> {
    let mut clusters: Vec<Vec<SrccTransportSample>> = Vec::new();

    for (sample_index, sample) in ordered_samples.into_iter().enumerate()
    {
        let mut compatible_cluster = None;

        for (source_cluster_index, cluster) in clusters.iter().enumerate()
        {
            let mut compatible = true;

            for member in cluster
            {
                let distance = metric.distance(&sample.source, &member.source);

                if !distance.is_finite()
                {
                    return Err(SrccRobustFitError::NonFiniteSourceDistance {
                        view_index,
                        source_cluster_index,
                    });
                }

                if distance > maximum_source_distance
                {
                    compatible = false;
                    break;
                }
            }

            if compatible
            {
                if compatible_cluster.is_some()
                {
                    return Err(SrccRobustFitError::AmbiguousSourceClusterAssignment {
                        view_index,
                        sample_index,
                    });
                }

                compatible_cluster = Some(source_cluster_index);
            }
        }

        match compatible_cluster
        {
            Some(source_cluster_index) =>
            {
                clusters[source_cluster_index].push(sample);
            },
            None => clusters.push(vec![sample]),
        }
    }

    Ok(clusters)
}

pub(crate) fn source_medoid(
    cluster: &[SrccTransportSample],
    view_index: usize,
    source_cluster_index: usize,
    metric: &SourceMetric,
) -> Result<Vector16, SrccRobustFitError> {
    let mut best_source = cluster[0].source;

    let mut best_score = source_score(
        &best_source,
        cluster,
        view_index,
        source_cluster_index,
        metric,
    )?;

    for sample in cluster.iter().skip(1)
    {
        let score = source_score(
            &sample.source,
            cluster,
            view_index,
            source_cluster_index,
            metric,
        )?;

        if score.total_cmp(&best_score).is_lt()
        {
            best_source = sample.source;
            best_score = score;
        }
    }

    Ok(best_source)
}

pub(crate) fn source_score(
    candidate: &Vector16,
    cluster: &[SrccTransportSample],
    view_index: usize,
    source_cluster_index: usize,
    metric: &SourceMetric,
) -> Result<f64, SrccRobustFitError> {
    let mut score = 0.0;

    for sample in cluster
    {
        let distance = metric.distance(candidate, &sample.source);

        if !distance.is_finite()
        {
            return Err(SrccRobustFitError::NonFiniteSourceDistance {
                view_index,
                source_cluster_index,
            });
        }

        score += distance;

        if !score.is_finite()
        {
            return Err(SrccRobustFitError::NonFiniteSourceDistance {
                view_index,
                source_cluster_index,
            });
        }
    }

    Ok(score)
}

/// Fixed-order scaled Euclidean norm.
///
/// The scaled accumulation prevents intermediate overflow for ordinary finite
/// differences while preserving a deterministic coordinate order.
fn source_distance(left: &Vector16, right: &Vector16) -> f64 {
    let mut scale = 0.0;
    let mut scaled_sum = 1.0;

    for (left_value, right_value) in left.iter().zip(right)
    {
        let difference = (left_value - right_value).abs();

        if !difference.is_finite()
        {
            return f64::INFINITY;
        }

        if difference == 0.0
        {
            continue;
        }

        if scale < difference
        {
            let ratio = scale / difference;
            scaled_sum = 1.0 + scaled_sum * ratio * ratio;
            scale = difference;
        }
        else
        {
            let ratio = difference / scale;
            scaled_sum += ratio * ratio;
        }
    }

    if scale == 0.0
    {
        0.0
    }
    else
    {
        scale * scaled_sum.sqrt()
    }
}

/// Scale-aware variant of [`source_distance`].
///
/// This mirrors the frozen [`source_distance`] body line for line, with two
/// changes: a coordinate whose inverse scale is exactly `0.0` (a dropped
/// dimension) is skipped **before any arithmetic** — otherwise an overflowing
/// raw difference on a dropped coordinate would produce `inf * 0.0 = NaN` and
/// abort the fit even though the policy removed that coordinate — and each
/// remaining coordinate difference is multiplied by its fitted inverse scale
/// before entering the accumulation.
fn scaled_source_distance(left: &Vector16, right: &Vector16, inverse_scales: &Vector16) -> f64 {
    let mut scale = 0.0;
    let mut scaled_sum = 1.0;

    for ((left_value, right_value), inverse_scale_value) in
        left.iter().zip(right).zip(inverse_scales)
    {
        if *inverse_scale_value == 0.0
        {
            continue;
        }

        let difference = ((left_value - right_value) * inverse_scale_value).abs();

        if !difference.is_finite()
        {
            return f64::INFINITY;
        }

        if difference == 0.0
        {
            continue;
        }

        if scale < difference
        {
            let ratio = scale / difference;
            scaled_sum = 1.0 + scaled_sum * ratio * ratio;
            scale = difference;
        }
        else
        {
            let ratio = difference / scale;
            scaled_sum += ratio * ratio;
        }
    }

    if scale == 0.0
    {
        0.0
    }
    else
    {
        scale * scaled_sum.sqrt()
    }
}

pub(crate) fn projector_frobenius_distance(left: &SrccProjector, right: &SrccProjector) -> f64 {
    let squared_distance = left
        .transform()
        .iter()
        .flatten()
        .zip(right.transform().iter().flatten())
        .fold(0.0, |sum, (left_value, right_value)| {
            let difference = left_value - right_value;
            sum + difference * difference
        });

    squared_distance.sqrt() / (SRCC_DIMENSION as f64).sqrt()
}

pub(crate) fn compare_samples(left: &SrccTransportSample, right: &SrccTransportSample) -> Ordering {
    compare_vectors(&left.source, &right.source)
        .then_with(|| compare_vectors(&left.target, &right.target))
}

pub(crate) fn compare_vectors(left: &Vector16, right: &Vector16) -> Ordering {
    left.iter()
        .zip(right)
        .find_map(|(left_value, right_value)| {
            let ordering = left_value.total_cmp(right_value);
            ordering.is_ne().then_some(ordering)
        })
        .unwrap_or(Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        apply_linear_map, basis_vector, evaluate_robust_leave_one_out_stability, squared_norm,
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

    fn source_config() -> SrccRobustSourceClusteringConfig {
        SrccRobustSourceClusteringConfig {
            maximum_source_distance: 1.0e-2,
        }
    }

    fn normalize(vector: Vector16) -> Vector16 {
        let norm = squared_norm(&vector).sqrt();
        vector.map(|value| value / norm)
    }

    fn perturbed_source(source: Vector16, epsilon: f64) -> Vector16 {
        let perturbation = basis_vector(8).unwrap();

        normalize(core::array::from_fn(|index| {
            source[index] + epsilon * perturbation[index]
        }))
    }

    fn contaminated_views(
        clean_repetitions: usize,
    ) -> (
        Vector16,
        Vector16,
        Vec<SrccTransportSample>,
        Vec<SrccTransportSample>,
    ) {
        let source = basis_vector(1).unwrap();
        let nearby_source = perturbed_source(source, 1.0e-3);
        let target = basis_vector(2).unwrap();
        let contaminant = basis_vector(9).unwrap();
        let negative = target.map(|value| -value);

        let mut positive = Vec::with_capacity(clean_repetitions + 1);
        let mut negative_view = Vec::with_capacity(clean_repetitions + 1);

        for _ in 0..clean_repetitions
        {
            positive.push(SrccTransportSample::new(source, target));

            negative_view.push(SrccTransportSample::new(source, negative));
        }

        positive.push(SrccTransportSample::new(nearby_source, contaminant));

        negative_view.push(SrccTransportSample::new(nearby_source, contaminant));

        (source, target, positive, negative_view)
    }

    #[test]
    fn zero_source_distance_matches_exact_robust_fitting() {
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

        let clustered = fit_source_clustered_robust_srcc_projector_from_views(
            &[source],
            &views,
            SrccRobustSourceClusteringConfig::default(),
            test_config(),
        )
        .unwrap();

        assert_eq!(exact, clustered);
    }

    #[test]
    fn nearby_sources_share_target_consensus() {
        let (source, target, positive, negative) = contaminated_views(2);

        let views = [positive.as_slice(), negative.as_slice()];

        let result = fit_source_clustered_robust_srcc_projector_from_views(
            &[source],
            &views,
            source_config(),
            test_config(),
        )
        .unwrap();

        assert_eq!(result.projector.rejected_dimension(), 2);
        assert_eq!(result.projector.closure().rounds(), 1);
        assert_eq!(result.projector.closure().certificates().len(), 1);

        assert_eq!(apply_linear_map(&result.transports[0], &source), target,);

        assert_eq!(
            apply_linear_map(&result.transports[1], &source),
            target.map(|value| -value),
        );

        assert!(squared_norm(&result.projector.apply(&target)) < 1.0e-24);
    }

    #[test]
    fn source_clustering_is_invariant_to_sample_order() {
        let (source, _, positive, negative) = contaminated_views(2);

        let first_views = [positive.as_slice(), negative.as_slice()];

        let first = fit_source_clustered_robust_srcc_projector_from_views(
            &[source],
            &first_views,
            source_config(),
            test_config(),
        )
        .unwrap();

        let mut reversed_positive = positive;
        let mut reversed_negative = negative;

        reversed_positive.reverse();
        reversed_negative.reverse();

        let reversed_views = [reversed_positive.as_slice(), reversed_negative.as_slice()];

        let reversed = fit_source_clustered_robust_srcc_projector_from_views(
            &[source],
            &reversed_views,
            source_config(),
            test_config(),
        )
        .unwrap();

        assert_eq!(first, reversed);
    }

    #[test]
    fn bridge_source_assignment_is_rejected() {
        let mut first = [0.0; SRCC_DIMENSION];
        first[0] = 1.0;

        let mut second = [0.0; SRCC_DIMENSION];
        second[0] = 1.0;
        second[1] = 2.0;

        let mut bridge = [0.0; SRCC_DIMENSION];
        bridge[0] = 1.1;
        bridge[1] = 1.0;

        let target = basis_vector(3).unwrap();

        let samples = [
            SrccTransportSample::new(first, target),
            SrccTransportSample::new(second, target),
            SrccTransportSample::new(bridge, target),
        ];

        let views = [samples.as_slice()];

        assert_eq!(
            fit_source_clustered_robust_srcc_projector_from_views(
                &[first],
                &views,
                SrccRobustSourceClusteringConfig {
                    maximum_source_distance: 1.1,
                },
                test_config(),
            ),
            Err(SrccRobustFitError::AmbiguousSourceClusterAssignment {
                view_index: 0,
                sample_index: 2,
            },),
        );
    }

    #[test]
    fn invalid_source_distance_is_rejected() {
        let source = basis_vector(1).unwrap();
        let target = basis_vector(2).unwrap();

        let samples = [SrccTransportSample::new(source, target)];
        let views = [samples.as_slice()];

        for maximum_source_distance in [-1.0, f64::NAN, f64::INFINITY]
        {
            assert_eq!(
                fit_source_clustered_robust_srcc_projector_from_views(
                    &[source],
                    &views,
                    SrccRobustSourceClusteringConfig {
                        maximum_source_distance,
                    },
                    test_config(),
                ),
                Err(SrccRobustFitError::InvalidMaximumSourceDistance,),
            );
        }
    }

    #[test]
    fn two_clean_targets_remain_loo_ambiguous() {
        let (source, _, positive, negative) = contaminated_views(2);

        let views = [positive.as_slice(), negative.as_slice()];

        assert_eq!(
            evaluate_source_clustered_robust_leave_one_out_stability(
                &[source],
                &views,
                source_config(),
                test_config(),
            ),
            Err(SrccRobustStabilityError::Fit(
                SrccRobustFitError::AmbiguousTargetConsensus {
                    view_index: 0,
                    source_group_index: 0,
                },
            ),),
        );
    }

    #[test]
    fn three_clean_targets_certify_clustered_loo_stability() {
        let (source, _, positive, negative) = contaminated_views(3);

        let views = [positive.as_slice(), negative.as_slice()];

        let report = evaluate_source_clustered_robust_leave_one_out_stability(
            &[source],
            &views,
            source_config(),
            test_config(),
        )
        .unwrap();

        assert_eq!(report.removal_count(), 8);

        assert_eq!(report.stable_dimension_count, report.removal_count(),);

        assert_eq!(report.dimension_stability_ratio(), 1.0);
        assert_eq!(report.maximum_frobenius_distance, 0.0);
        assert_eq!(report.mean_frobenius_distance, 0.0);
    }

    #[test]
    fn zero_distance_clustered_loo_matches_exact_loo() {
        let source = basis_vector(1).unwrap();
        let target = basis_vector(2).unwrap();

        let positive = [
            SrccTransportSample::new(source, target),
            SrccTransportSample::new(source, target),
            SrccTransportSample::new(source, target),
        ];

        let negative_target = target.map(|value| -value);

        let negative = [
            SrccTransportSample::new(source, negative_target),
            SrccTransportSample::new(source, negative_target),
            SrccTransportSample::new(source, negative_target),
        ];

        let views = [positive.as_slice(), negative.as_slice()];

        let exact =
            evaluate_robust_leave_one_out_stability(&[source], &views, test_config()).unwrap();

        let clustered = evaluate_source_clustered_robust_leave_one_out_stability(
            &[source],
            &views,
            SrccRobustSourceClusteringConfig::default(),
            test_config(),
        )
        .unwrap();

        assert_eq!(exact, clustered);
    }
}
