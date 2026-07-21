//! Deterministic robust fitting for repeated-source transport observations.
//!
//! The standard SRCC learner averages every normalized rank-one observation.
//! That behaviour remains unchanged.
//!
//! This module provides an opt-in estimator for explicit views containing
//! repeated observations of exactly the same source. For each exact-source
//! group, it selects an observed target medoid before learning the transport.
//!
//! This protects against a minority of target outliers without inventing a
//! synthetic target. A tied consensus between distinct targets is rejected
//! rather than resolved arbitrarily.

use core::{cmp::Ordering, fmt};

use crate::{
    SRCC_DIMENSION, SrccClosureError, SrccConfig, SrccDiscoveryError, SrccFitResult, SrccProjector,
    SrccStabilityReport, SrccStabilityVariant, SrccTransportSample, Vector16,
    learn_transport_views,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SrccRobustFitError {
    Discovery(SrccDiscoveryError),
    Closure(SrccClosureError),
    InvalidMaximumSourceDistance,
    NonFiniteSourceDistance {
        view_index: usize,
        source_cluster_index: usize,
    },
    AmbiguousSourceClusterAssignment {
        view_index: usize,
        sample_index: usize,
    },
    AmbiguousTargetConsensus {
        view_index: usize,
        source_group_index: usize,
    },
    NonFiniteTargetDistance {
        view_index: usize,
        source_group_index: usize,
    },
    InvalidSourceGeometry,
    DegenerateSourceScale {
        dimension: usize,
    },
    NoActiveSourceDimensions,
}

impl fmt::Display for SrccRobustFitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Discovery(error) => error.fmt(formatter),
            Self::Closure(error) => error.fmt(formatter),
            Self::InvalidMaximumSourceDistance =>
            {
                formatter.write_str("maximum source distance must be finite and non-negative")
            },
            Self::NonFiniteSourceDistance {
                view_index,
                source_cluster_index,
            } =>
            {
                write!(
                    formatter,
                    "transport view {view_index}, source cluster {source_cluster_index} has a non-finite source distance",
                )
            },
            Self::AmbiguousSourceClusterAssignment {
                view_index,
                sample_index,
            } =>
            {
                write!(
                    formatter,
                    "transport view {view_index}, canonically ordered sample {sample_index} is compatible with multiple source clusters",
                )
            },
            Self::AmbiguousTargetConsensus {
                view_index,
                source_group_index,
            } =>
            {
                write!(
                    formatter,
                    "transport view {view_index}, source group {source_group_index} \
has an ambiguous target consensus",
                )
            },
            Self::NonFiniteTargetDistance {
                view_index,
                source_group_index,
            } =>
            {
                write!(
                    formatter,
                    "transport view {view_index}, source group {source_group_index} \
has a non-finite target distance",
                )
            },
            Self::InvalidSourceGeometry =>
            {
                formatter.write_str("source geometry specification is invalid")
            },
            Self::DegenerateSourceScale { dimension } =>
            {
                write!(
                    formatter,
                    "source coordinate {dimension} has a degenerate robust scale \
under the configured zero-scale policy",
                )
            },
            Self::NoActiveSourceDimensions =>
            {
                formatter.write_str("every source coordinate was dropped; no geometry remains")
            },
        }
    }
}

impl std::error::Error for SrccRobustFitError {}

impl From<SrccDiscoveryError> for SrccRobustFitError {
    fn from(error: SrccDiscoveryError) -> Self {
        Self::Discovery(error)
    }
}

impl From<SrccClosureError> for SrccRobustFitError {
    fn from(error: SrccClosureError) -> Self {
        Self::Closure(error)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SrccRobustStabilityError {
    NoRemovableSamples,
    Fit(SrccRobustFitError),
}

impl fmt::Display for SrccRobustStabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::NoRemovableSamples =>
            {
                formatter.write_str("at least one explicit view must contain two samples")
            },
            Self::Fit(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SrccRobustStabilityError {}

impl From<SrccRobustFitError> for SrccRobustStabilityError {
    fn from(error: SrccRobustFitError) -> Self {
        Self::Fit(error)
    }
}

/// Fits an SRCC projector after deterministic target-medoid aggregation.
///
/// Samples are grouped only when their source vectors are exactly equal.
/// Every distinct source contributes one representative rank-one action.
///
/// Within an exact-source group, the selected target is the observed target
/// minimizing the fixed-order sum of squared distances to all group targets.
///
/// If distinct targets tie for the minimum score, fitting fails explicitly.
pub fn fit_robust_srcc_projector_from_views(
    seeds: &[Vector16],
    views: &[&[SrccTransportSample]],
    config: SrccConfig,
) -> Result<SrccFitResult, SrccRobustFitError> {
    let _validated_transports = learn_transport_views(views, config.energy_floor)?;

    let representative_storage = representative_views(views)?;

    let representative_views: Vec<&[SrccTransportSample]> =
        representative_storage.iter().map(Vec::as_slice).collect();

    let transports = learn_transport_views(&representative_views, config.energy_floor)?;

    let projector = SrccProjector::build(seeds, &transports, config)?;

    Ok(SrccFitResult {
        transports,
        projector,
    })
}

/// Evaluates leave-one-out stability using robust target-medoid fitting.
///
/// An ambiguous reduced target consensus is returned as an explicit error.
/// This prevents an uncertified deterministic tie-break from being mistaken
/// for robust evidence.
pub fn evaluate_robust_leave_one_out_stability(
    seeds: &[Vector16],
    views: &[&[SrccTransportSample]],
    config: SrccConfig,
) -> Result<SrccStabilityReport, SrccRobustStabilityError> {
    let full = fit_robust_srcc_projector_from_views(seeds, views, config)?;

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

            let reduced = fit_robust_srcc_projector_from_views(seeds, &reduced_references, config)?;

            variants.push(SrccStabilityVariant {
                removed_view_index: view_index,
                removed_sample_index: sample_index,
                rejected_dimension: reduced.projector.rejected_dimension(),
                frobenius_distance: robust_projector_frobenius_distance(
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

fn robust_projector_frobenius_distance(left: &SrccProjector, right: &SrccProjector) -> f64 {
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

fn representative_views(
    views: &[&[SrccTransportSample]],
) -> Result<Vec<Vec<SrccTransportSample>>, SrccRobustFitError> {
    let mut result = Vec::with_capacity(views.len());

    for (view_index, view) in views.iter().enumerate()
    {
        let mut ordered_samples = view.to_vec();
        ordered_samples.sort_by(compare_samples);

        let mut source_groups: Vec<Vec<SrccTransportSample>> = Vec::new();

        for sample in ordered_samples
        {
            match source_groups.last_mut()
            {
                Some(group) if group[0].source == sample.source => group.push(sample),
                _ => source_groups.push(vec![sample]),
            }
        }

        let mut representatives = Vec::with_capacity(source_groups.len());

        for (source_group_index, group) in source_groups.iter().enumerate()
        {
            let target = target_medoid(group, view_index, source_group_index)?;

            representatives.push(SrccTransportSample::new(group[0].source, target));
        }

        result.push(representatives);
    }

    Ok(result)
}

fn target_medoid(
    group: &[SrccTransportSample],
    view_index: usize,
    source_group_index: usize,
) -> Result<Vector16, SrccRobustFitError> {
    let mut best_target = group[0].target;
    let mut best_score = target_score(&best_target, group);

    if !best_score.is_finite()
    {
        return Err(SrccRobustFitError::NonFiniteTargetDistance {
            view_index,
            source_group_index,
        });
    }

    let mut ambiguous = false;

    for sample in group.iter().skip(1)
    {
        let score = target_score(&sample.target, group);

        if !score.is_finite()
        {
            return Err(SrccRobustFitError::NonFiniteTargetDistance {
                view_index,
                source_group_index,
            });
        }

        match score.total_cmp(&best_score)
        {
            Ordering::Less =>
            {
                best_target = sample.target;
                best_score = score;
                ambiguous = false;
            },
            Ordering::Equal if sample.target != best_target =>
            {
                ambiguous = true;
            },
            Ordering::Equal | Ordering::Greater =>
            {},
        }
    }

    if ambiguous
    {
        return Err(SrccRobustFitError::AmbiguousTargetConsensus {
            view_index,
            source_group_index,
        });
    }

    Ok(best_target)
}

fn target_score(candidate: &Vector16, group: &[SrccTransportSample]) -> f64 {
    group.iter().fold(0.0, |sum, sample| {
        sum + squared_distance(candidate, &sample.target)
    })
}

fn squared_distance(left: &Vector16, right: &Vector16) -> f64 {
    left.iter().zip(right).fold(0.0, |sum, (left, right)| {
        let difference = left - right;
        sum + difference * difference
    })
}

fn compare_samples(left: &SrccTransportSample, right: &SrccTransportSample) -> Ordering {
    compare_vectors(&left.source, &right.source)
        .then_with(|| compare_vectors(&left.target, &right.target))
}

fn compare_vectors(left: &Vector16, right: &Vector16) -> Ordering {
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
    use crate::{apply_linear_map, basis_vector, fit_srcc_projector_from_views, squared_norm};

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

    fn normalize(vector: Vector16) -> Vector16 {
        let norm = squared_norm(&vector).sqrt();
        vector.map(|value| value / norm)
    }

    fn contaminated_views(
        epsilon: f64,
    ) -> (
        Vector16,
        Vector16,
        Vec<SrccTransportSample>,
        Vec<SrccTransportSample>,
    ) {
        let source = basis_vector(1).unwrap();
        let target = basis_vector(2).unwrap();
        let contaminant = basis_vector(8).unwrap();

        let negative = target.map(|value| -value);

        let contaminated_positive = normalize(core::array::from_fn(|index| {
            target[index] + epsilon * contaminant[index]
        }));

        let contaminated_negative = normalize(core::array::from_fn(|index| {
            negative[index] + epsilon * contaminant[index]
        }));

        let positive = vec![
            SrccTransportSample::new(source, target),
            SrccTransportSample::new(source, target),
            SrccTransportSample::new(source, contaminated_positive),
        ];

        let negative_view = vec![
            SrccTransportSample::new(source, negative),
            SrccTransportSample::new(source, negative),
            SrccTransportSample::new(source, contaminated_negative),
        ];

        (source, target, positive, negative_view)
    }

    #[test]
    fn minority_target_contamination_is_removed() {
        let (source, target, positive, negative) = contaminated_views(1.0e-6);

        let views = [positive.as_slice(), negative.as_slice()];

        let result =
            fit_robust_srcc_projector_from_views(&[source], &views, test_config()).unwrap();

        assert_eq!(result.projector.rejected_dimension(), 2);
        assert_eq!(result.projector.closure().rounds(), 1);
        assert_eq!(result.projector.closure().certificates().len(), 1);

        assert_eq!(apply_linear_map(&result.transports[0], &source), target);

        assert_eq!(
            apply_linear_map(&result.transports[1], &source),
            target.map(|value| -value),
        );

        assert!(squared_norm(&result.projector.apply(&target)) < 1.0e-24);
    }

    #[test]
    fn robust_fitting_is_invariant_to_sample_order() {
        let (source, _, positive, negative) = contaminated_views(1.0e-3);

        let first_views = [positive.as_slice(), negative.as_slice()];

        let first =
            fit_robust_srcc_projector_from_views(&[source], &first_views, test_config()).unwrap();

        let mut reversed_positive = positive;
        let mut reversed_negative = negative;

        reversed_positive.reverse();
        reversed_negative.reverse();

        let reversed_views = [reversed_positive.as_slice(), reversed_negative.as_slice()];

        let reversed =
            fit_robust_srcc_projector_from_views(&[source], &reversed_views, test_config())
                .unwrap();

        assert_eq!(first, reversed);
    }

    #[test]
    fn robust_view_order_does_not_change_projector() {
        let (source, _, positive, negative) = contaminated_views(1.0e-3);

        let first_views = [positive.as_slice(), negative.as_slice()];

        let reversed_views = [negative.as_slice(), positive.as_slice()];

        let first =
            fit_robust_srcc_projector_from_views(&[source], &first_views, test_config()).unwrap();

        let reversed =
            fit_robust_srcc_projector_from_views(&[source], &reversed_views, test_config())
                .unwrap();

        assert_eq!(first.projector.transform(), reversed.projector.transform(),);

        assert_eq!(
            first.projector.rejected_dimension(),
            reversed.projector.rejected_dimension(),
        );

        assert_eq!(first.transports[0], reversed.transports[1],);

        assert_eq!(first.transports[1], reversed.transports[0],);
    }

    #[test]
    fn singleton_groups_match_standard_fitting() {
        let source = basis_vector(1).unwrap();
        let target = basis_vector(2).unwrap();

        let positive = [SrccTransportSample::new(source, target)];

        let negative = [SrccTransportSample::new(source, target.map(|value| -value))];

        let views = [positive.as_slice(), negative.as_slice()];

        let standard = fit_srcc_projector_from_views(&[source], &views, test_config()).unwrap();

        let robust =
            fit_robust_srcc_projector_from_views(&[source], &views, test_config()).unwrap();

        assert_eq!(standard, robust);
    }

    #[test]
    fn two_clean_samples_do_not_certify_leave_one_out_majority() {
        let (source, _, positive, negative) = contaminated_views(1.0e-3);

        let views = [positive.as_slice(), negative.as_slice()];

        assert_eq!(
            evaluate_robust_leave_one_out_stability(&[source], &views, test_config(),),
            Err(SrccRobustStabilityError::Fit(
                SrccRobustFitError::AmbiguousTargetConsensus {
                    view_index: 0,
                    source_group_index: 0,
                },
            )),
        );
    }

    #[test]
    fn three_clean_samples_certify_leave_one_out_stability() {
        let (source, target, mut positive, mut negative) = contaminated_views(1.0e-3);

        positive.insert(0, SrccTransportSample::new(source, target));

        negative.insert(
            0,
            SrccTransportSample::new(source, target.map(|value| -value)),
        );

        let views = [positive.as_slice(), negative.as_slice()];

        let report =
            evaluate_robust_leave_one_out_stability(&[source], &views, test_config()).unwrap();

        assert_eq!(report.removal_count(), 8);

        assert_eq!(report.stable_dimension_count, report.removal_count(),);

        assert_eq!(report.dimension_stability_ratio(), 1.0,);

        assert_eq!(report.maximum_frobenius_distance, 0.0,);

        assert_eq!(report.mean_frobenius_distance, 0.0,);
    }

    #[test]
    fn tied_distinct_targets_are_rejected() {
        let source = basis_vector(1).unwrap();

        let samples = [
            SrccTransportSample::new(source, basis_vector(2).unwrap()),
            SrccTransportSample::new(source, basis_vector(3).unwrap()),
        ];

        let views = [samples.as_slice()];

        assert_eq!(
            fit_robust_srcc_projector_from_views(&[source], &views, test_config(),),
            Err(SrccRobustFitError::AmbiguousTargetConsensus {
                view_index: 0,
                source_group_index: 0,
            }),
        );
    }
}
