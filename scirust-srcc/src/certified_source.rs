//! Opt-in certified source clustering for SRCC (phase 726).
//!
//! The historical (and scale-aware) source clustering is a deterministic
//! greedy complete-link pass: fast, canonical, but with no optimality claim,
//! and it rejects bridge-ambiguous inputs rather than resolving them. This
//! module lets the caller opt into the certified diameter-constrained medoid
//! solver from `scirust-solvers`:
//!
//! - [`SrccSourceClusteringSolver::GreedyCompleteLink`] delegates to the
//!   scale-aware greedy pipeline unchanged (bit-identical results, no
//!   certificates) — historical defaults are untouched;
//! - [`SrccSourceClusteringSolver::CertifiedMedoid`] partitions each view's
//!   sources with the exact/hybrid branch-and-bound solver and returns one
//!   [`ClusteringCertificate`] per view. `proven_optimal` is `true` only when
//!   the search completed; a budget-limited run carries an explicit gap.
//!
//! Semantic differences, stated honestly: the certified solver *optimizes*
//! the lexicographic (cluster count, observed-medoid cost, canonical
//! assignment) objective, so a bridge sample that the greedy pass rejects as
//! ambiguous is *assigned* here — optimally, deterministically, and with a
//! certificate — rather than refused. Both behaviours are legitimate; the
//! caller chooses the semantics by choosing the solver. Sources rewritten to
//! cluster representatives remain **observed** source medoids (smallest index
//! on ties), and target aggregation stays the historical observed-target
//! medoid procedure.

use scirust_solvers::combinatorial::{
    CertifiedClusteringMode, CertifiedMedoidClusteringConfig, CertifiedMedoidClusteringResult,
    ClusteringCertificate, DistanceMatrix, certified_medoid_clustering,
};

use crate::robust_source::compare_samples;
use crate::robust_source_geometry::fit_source_metric;
use crate::{
    SrccConfig, SrccFitResult, SrccRobustFitError, SrccScaleAwareSourceClusteringConfig,
    SrccTransportSample, Vector16, fit_robust_srcc_projector_from_views,
    fit_scale_aware_source_clustered_robust_srcc_projector_from_views, learn_transport_views,
};

/// Which partitioner groups approximately equal sources.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SrccSourceClusteringSolver {
    /// The historical deterministic greedy complete-link pass (the default
    /// everywhere else; no optimality claim, bridge ambiguity is a typed
    /// error).
    GreedyCompleteLink,
    /// The certified diameter-constrained medoid solver: proven optimality or
    /// an explicit gap, one certificate per view.
    CertifiedMedoid {
        /// Exact or budget-limited hybrid search.
        mode: CertifiedClusteringMode,
    },
}

/// Result of a certified source-clustered fit.
#[derive(Clone, Debug, PartialEq)]
pub struct SrccCertifiedFitResult {
    /// The fitted transports and projector.
    pub fit: SrccFitResult,
    /// One clustering certificate per view (empty for
    /// [`SrccSourceClusteringSolver::GreedyCompleteLink`], which certifies
    /// nothing).
    pub view_certificates: Vec<ClusteringCertificate>,
}

/// Fits a robust SRCC projector with an explicitly chosen source partitioner.
///
/// `GreedyCompleteLink` reproduces
/// [`fit_scale_aware_source_clustered_robust_srcc_projector_from_views`]
/// exactly. `CertifiedMedoid` builds each view's pairwise source-distance
/// matrix under the configured geometry (fitted per call — no leakage),
/// solves the certified partitioning problem, rewrites every cluster member
/// to the observed source medoid, and delegates target aggregation to the
/// historical robust fitter.
pub fn fit_certified_source_clustered_robust_srcc_projector_from_views(
    seeds: &[Vector16],
    views: &[&[SrccTransportSample]],
    source: SrccScaleAwareSourceClusteringConfig,
    solver: SrccSourceClusteringSolver,
    config: SrccConfig,
) -> Result<SrccCertifiedFitResult, SrccRobustFitError> {
    match solver
    {
        SrccSourceClusteringSolver::GreedyCompleteLink =>
        {
            let fit = fit_scale_aware_source_clustered_robust_srcc_projector_from_views(
                seeds, views, source, config,
            )?;

            Ok(SrccCertifiedFitResult {
                fit,
                view_certificates: Vec::new(),
            })
        },
        SrccSourceClusteringSolver::CertifiedMedoid { mode } =>
        {
            fit_with_certified_medoid(seeds, views, source, mode, config)
        },
    }
}

fn fit_with_certified_medoid(
    seeds: &[Vector16],
    views: &[&[SrccTransportSample]],
    source: SrccScaleAwareSourceClusteringConfig,
    mode: CertifiedClusteringMode,
    config: SrccConfig,
) -> Result<SrccCertifiedFitResult, SrccRobustFitError> {
    if !source.clustering.maximum_source_distance.is_finite()
        || source.clustering.maximum_source_distance < 0.0
    {
        return Err(SrccRobustFitError::InvalidMaximumSourceDistance);
    }

    // Historical precedence: validate every original observation first.
    let _validated = learn_transport_views(views, config.energy_floor)?;

    // Fitted geometry, derived from exactly these views (no leakage).
    let metric = fit_source_metric(views, source.geometry)?;

    let mut representative_storage: Vec<Vec<SrccTransportSample>> = Vec::with_capacity(views.len());
    let mut view_certificates: Vec<ClusteringCertificate> = Vec::with_capacity(views.len());

    for (view_index, view) in views.iter().enumerate()
    {
        // Canonical sample order, as everywhere else in the crate.
        let mut ordered = view.to_vec();
        ordered.sort_by(compare_samples);

        let n = ordered.len();
        let mut values = vec![0.0; n * n];

        for i in 0..n
        {
            for j in (i + 1)..n
            {
                let distance = metric.distance(&ordered[i].source, &ordered[j].source);

                if !distance.is_finite()
                {
                    return Err(SrccRobustFitError::NonFiniteSourceDistance {
                        view_index,
                        source_cluster_index: 0,
                    });
                }

                values[i * n + j] = distance;
                values[j * n + i] = distance;
            }
        }

        let matrix = DistanceMatrix::new(n, values).map_err(|error| {
            SrccRobustFitError::CertifiedClusteringFailed {
                view_index,
                detail: error.to_string(),
            }
        })?;

        let CertifiedMedoidClusteringResult {
            assignments,
            medoid_indices,
            certificate,
        } = certified_medoid_clustering(
            &matrix,
            CertifiedMedoidClusteringConfig {
                maximum_cluster_diameter: source.clustering.maximum_source_distance,
                mode,
            },
        )
        .map_err(|error| SrccRobustFitError::CertifiedClusteringFailed {
            view_index,
            detail: error.to_string(),
        })?;

        // Rewrite every member to its cluster's observed source medoid, in
        // cluster-label order then canonical member order (mirroring the
        // greedy canonicalization's output layout).
        let label_count = medoid_indices.len();
        let mut canonical_view = Vec::with_capacity(n);

        for label in 0..label_count
        {
            let representative_source = ordered[medoid_indices[label]].source;

            for (position, sample) in ordered.iter().enumerate()
            {
                if assignments[position] == label
                {
                    canonical_view.push(SrccTransportSample::new(
                        representative_source,
                        sample.target,
                    ));
                }
            }
        }

        representative_storage.push(canonical_view);
        view_certificates.push(certificate);
    }

    let representative_views: Vec<&[SrccTransportSample]> =
        representative_storage.iter().map(Vec::as_slice).collect();

    let fit = fit_robust_srcc_projector_from_views(seeds, &representative_views, config)?;

    Ok(SrccCertifiedFitResult {
        fit,
        view_certificates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        SRCC_DIMENSION, SrccRobustSourceClusteringConfig, SrccSourceGeometrySpec, basis_vector,
        fit_source_clustered_robust_srcc_projector_from_views, squared_norm,
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

    fn raw(maximum_source_distance: f64) -> SrccScaleAwareSourceClusteringConfig {
        SrccScaleAwareSourceClusteringConfig {
            geometry: SrccSourceGeometrySpec::RawEuclidean,
            clustering: SrccRobustSourceClusteringConfig {
                maximum_source_distance,
            },
        }
    }

    fn normalize(vector: Vector16) -> Vector16 {
        let norm = squared_norm(&vector).sqrt();

        vector.map(|value| value / norm)
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
    fn greedy_solver_delegates_bit_exactly() {
        let (source, positive, negative) = jittered_views(3, 1.0e-3);

        let views = [positive.as_slice(), negative.as_slice()];

        let historical = fit_source_clustered_robust_srcc_projector_from_views(
            &[source],
            &views,
            SrccRobustSourceClusteringConfig {
                maximum_source_distance: 1.0e-2,
            },
            test_config(),
        )
        .unwrap();

        let certified = fit_certified_source_clustered_robust_srcc_projector_from_views(
            &[source],
            &views,
            raw(1.0e-2),
            SrccSourceClusteringSolver::GreedyCompleteLink,
            test_config(),
        )
        .unwrap();

        assert_eq!(historical, certified.fit);
        assert!(certified.view_certificates.is_empty());
    }

    #[test]
    fn certified_solver_matches_greedy_on_unambiguous_inputs() {
        let (source, mut positive, mut negative) = jittered_views(3, 1.0e-3);

        // A second, well-separated source group (distance √2 from the first,
        // far above the 1e-2 radius) so the proven-optimal partition per view
        // is the non-trivial two-cluster one, not a single merged ball.
        let distant_source = basis_vector(4).unwrap();
        let distant_target = basis_vector(5).unwrap();

        for _ in 0..2
        {
            positive.push(SrccTransportSample::new(distant_source, distant_target));

            negative.push(SrccTransportSample::new(distant_source, distant_target));
        }

        let views = [positive.as_slice(), negative.as_slice()];

        let greedy = fit_certified_source_clustered_robust_srcc_projector_from_views(
            &[source],
            &views,
            raw(1.0e-2),
            SrccSourceClusteringSolver::GreedyCompleteLink,
            test_config(),
        )
        .unwrap();

        let certified = fit_certified_source_clustered_robust_srcc_projector_from_views(
            &[source],
            &views,
            raw(1.0e-2),
            SrccSourceClusteringSolver::CertifiedMedoid {
                mode: CertifiedClusteringMode::Exact,
            },
            test_config(),
        )
        .unwrap();

        assert_eq!(greedy.fit, certified.fit);
        assert_eq!(certified.view_certificates.len(), 2);

        for certificate in &certified.view_certificates
        {
            assert!(certificate.proven_optimal);
            assert_eq!(certificate.optimality_gap, 0.0);
            assert_eq!(certificate.objective_cluster_count, 2);
        }
    }

    #[test]
    fn certified_solver_resolves_the_bridge_the_greedy_pass_rejects() {
        // The PR #719 bridge fixture: greedy complete-link rejects the bridge
        // sample as ambiguous; the certified solver assigns it optimally with
        // a proven two-cluster certificate.
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

        let views = [samples.as_slice(), samples.as_slice()];

        let greedy = fit_certified_source_clustered_robust_srcc_projector_from_views(
            &[first],
            &views,
            raw(1.1),
            SrccSourceClusteringSolver::GreedyCompleteLink,
            test_config(),
        );

        assert_eq!(
            greedy,
            Err(SrccRobustFitError::AmbiguousSourceClusterAssignment {
                view_index: 0,
                sample_index: 2,
            }),
        );

        let certified = fit_certified_source_clustered_robust_srcc_projector_from_views(
            &[first],
            &views,
            raw(1.1),
            SrccSourceClusteringSolver::CertifiedMedoid {
                mode: CertifiedClusteringMode::Exact,
            },
            test_config(),
        )
        .unwrap();

        for certificate in &certified.view_certificates
        {
            assert!(certificate.proven_optimal);
            assert_eq!(certificate.objective_cluster_count, 2);
        }
    }

    #[test]
    fn certified_fit_is_deterministic() {
        let (source, positive, negative) = jittered_views(3, 1.0e-3);

        let views = [positive.as_slice(), negative.as_slice()];

        let run = || {
            fit_certified_source_clustered_robust_srcc_projector_from_views(
                &[source],
                &views,
                raw(1.0e-2),
                SrccSourceClusteringSolver::CertifiedMedoid {
                    mode: CertifiedClusteringMode::Exact,
                },
                test_config(),
            )
        };

        assert_eq!(run().unwrap(), run().unwrap());
    }

    #[test]
    fn hybrid_budget_exhaustion_carries_the_gap_into_view_certificates() {
        let (source, positive, negative) = jittered_views(3, 1.0e-3);

        let views = [positive.as_slice(), negative.as_slice()];

        let certified = fit_certified_source_clustered_robust_srcc_projector_from_views(
            &[source],
            &views,
            raw(1.0e-2),
            SrccSourceClusteringSolver::CertifiedMedoid {
                mode: CertifiedClusteringMode::Hybrid {
                    maximum_nodes: 1,
                    maximum_iterations: 1,
                },
            },
            test_config(),
        )
        .unwrap();

        assert!(
            certified
                .view_certificates
                .iter()
                .any(|certificate| !certificate.proven_optimal && certificate.optimality_gap > 0.0)
        );
    }

    #[test]
    fn invalid_radius_is_rejected_before_solving() {
        let (source, positive, negative) = jittered_views(2, 0.0);

        let views = [positive.as_slice(), negative.as_slice()];

        assert_eq!(
            fit_certified_source_clustered_robust_srcc_projector_from_views(
                &[source],
                &views,
                raw(-1.0),
                SrccSourceClusteringSolver::CertifiedMedoid {
                    mode: CertifiedClusteringMode::Exact,
                },
                test_config(),
            ),
            Err(SrccRobustFitError::InvalidMaximumSourceDistance),
        );
    }
}
