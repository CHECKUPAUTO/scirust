//! Deterministic mechanism benchmark for robust SRCC source clustering.
//!
//! Two explicit transport views should extend the rejected structure from
//! `e1` to `span(e1, e2)`.
//!
//! Each view contains repeated clean observations from `e1` and one target
//! outlier whose source is a perturbed version of `e1`.
//!
//! The benchmark compares:
//!
//! - exact-source robust fitting;
//! - complete-link source-clustered robust fitting.
//!
//! The source-clustered estimator is expected to recover the true structure
//! only while the perturbed source remains inside the configured complete-link
//! radius.

use scirust_srcc::{
    SrccConfig, SrccRobustFitError, SrccRobustSourceClusteringConfig, SrccRobustStabilityError,
    SrccTransportSample, Vector16, basis_vector,
    evaluate_source_clustered_robust_leave_one_out_stability, fit_robust_srcc_projector_from_views,
    fit_source_clustered_robust_srcc_projector_from_views, squared_norm,
};

const MAXIMUM_SOURCE_DISTANCE: f64 = 1.0001e-2;
const EXPECTED_DIMENSION: usize = 2;

type SourceClusteringViews = (
    Vector16,
    Vector16,
    Vec<SrccTransportSample>,
    Vec<SrccTransportSample>,
    f64,
);

fn config() -> SrccConfig {
    SrccConfig {
        novelty_threshold: 1.0e-10,
        resonance_threshold: 0.999,
        minimum_support: 2,
        maximum_dimension: EXPECTED_DIMENSION,
        maximum_rounds: EXPECTED_DIMENSION,
        energy_floor: 1.0e-30,
    }
}

fn source_config() -> SrccRobustSourceClusteringConfig {
    SrccRobustSourceClusteringConfig {
        maximum_source_distance: MAXIMUM_SOURCE_DISTANCE,
    }
}

fn normalize(vector: Vector16) -> Result<Vector16, String> {
    let norm = squared_norm(&vector).sqrt();

    if !norm.is_finite() || norm <= 1.0e-15
    {
        return Err("cannot normalize a zero or non-finite vector".into());
    }

    Ok(vector.map(|value| value / norm))
}

fn perturbed_source(source: Vector16, jitter: f64) -> Result<Vector16, String> {
    let perturbation = basis_vector(8).ok_or_else(|| "missing perturbation axis".to_owned())?;

    normalize(core::array::from_fn(|index| {
        source[index] + jitter * perturbation[index]
    }))
}

fn euclidean_distance(left: &Vector16, right: &Vector16) -> f64 {
    left.iter()
        .zip(right)
        .fold(0.0, |sum, (left_value, right_value)| {
            let difference = left_value - right_value;
            sum + difference * difference
        })
        .sqrt()
}

fn build_views(clean_repetitions: usize, jitter: f64) -> Result<SourceClusteringViews, String> {
    let source = basis_vector(1).ok_or_else(|| "missing source axis".to_owned())?;

    let target = basis_vector(2).ok_or_else(|| "missing target axis".to_owned())?;

    let target_outlier = basis_vector(9).ok_or_else(|| "missing target-outlier axis".to_owned())?;

    let nearby_source = perturbed_source(source, jitter)?;

    let source_distance = euclidean_distance(&source, &nearby_source);

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

    Ok((source, target, positive, negative, source_distance))
}

fn main() -> Result<(), String> {
    println!(
        "jitter,source_distance,clean_repetitions,\
total_samples,within_radius,exact_dimension,\
clustered_dimension,clustered_target_residual,\
clustered_loo_status,clustered_max_loo_distance,\
clustered_dimension_stability"
    );

    for clean_repetitions in [2_usize, 3_usize]
    {
        for jitter in [
            0.0, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-3, 5.0e-3, 1.0e-2, 2.0e-2, 5.0e-2, 1.0e-1,
        ]
        {
            let (source, target, positive, negative, source_distance) =
                build_views(clean_repetitions, jitter)?;

            let views = [positive.as_slice(), negative.as_slice()];

            let total_samples = positive.len() + negative.len();

            let within_radius = source_distance <= MAXIMUM_SOURCE_DISTANCE;

            let exact = fit_robust_srcc_projector_from_views(&[source], &views, config())
                .map_err(|error| error.to_string())?;

            let exact_dimension = exact.projector.rejected_dimension();

            let clustered = fit_source_clustered_robust_srcc_projector_from_views(
                &[source],
                &views,
                source_config(),
                config(),
            )
            .map_err(|error| error.to_string())?;

            let clustered_dimension = clustered.projector.rejected_dimension();

            let clustered_target_residual = squared_norm(&clustered.projector.apply(&target));

            if jitter == 0.0
            {
                if exact_dimension != EXPECTED_DIMENSION
                {
                    return Err(format!(
                        "exact fitting failed for the unperturbed \
source with {clean_repetitions} clean observations",
                    ));
                }
            }
            else if exact_dimension != 1
            {
                return Err(format!(
                    "exact-source fitting unexpectedly recovered \
dimension {exact_dimension} for jitter={jitter}",
                ));
            }

            if within_radius
            {
                if clustered_dimension != EXPECTED_DIMENSION || clustered_target_residual > 1.0e-24
                {
                    return Err(format!(
                        "source clustering failed inside its radius: \
jitter={jitter}, distance={source_distance}, \
dimension={clustered_dimension}, \
residual={clustered_target_residual}",
                    ));
                }

                match evaluate_source_clustered_robust_leave_one_out_stability(
                    &[source],
                    &views,
                    source_config(),
                    config(),
                )
                {
                    Ok(report) if clean_repetitions == 3 =>
                    {
                        if report.maximum_frobenius_distance != 0.0
                            || report.dimension_stability_ratio() != 1.0
                        {
                            return Err(format!(
                                "3+1 clustered stability failed for \
jitter={jitter}",
                            ));
                        }

                        println!(
                            "{jitter:.17e},{source_distance:.17e},\
{clean_repetitions},{total_samples},true,\
{exact_dimension},{clustered_dimension},\
{clustered_target_residual:.17e},stable,\
{:.17e},{:.17e}",
                            report.maximum_frobenius_distance,
                            report.dimension_stability_ratio(),
                        );
                    },
                    Err(SrccRobustStabilityError::Fit(
                        SrccRobustFitError::AmbiguousTargetConsensus { .. },
                    )) if clean_repetitions == 2 =>
                    {
                        println!(
                            "{jitter:.17e},{source_distance:.17e},\
{clean_repetitions},{total_samples},true,\
{exact_dimension},{clustered_dimension},\
{clustered_target_residual:.17e},ambiguous,\
nan,nan",
                        );
                    },
                    Ok(_) =>
                    {
                        return Err(format!(
                            "2+1 observations were incorrectly \
certified for jitter={jitter}",
                        ));
                    },
                    Err(error) =>
                    {
                        return Err(format!(
                            "unexpected clustered LOO result for \
jitter={jitter}, clean_repetitions=\
{clean_repetitions}: {error}",
                        ));
                    },
                }
            }
            else
            {
                if clustered_dimension != 1 || clustered_target_residual < 0.99
                {
                    return Err(format!(
                        "out-of-radius sources were incorrectly merged: \
jitter={jitter}, distance={source_distance}, \
dimension={clustered_dimension}, \
residual={clustered_target_residual}",
                    ));
                }

                println!(
                    "{jitter:.17e},{source_distance:.17e},\
{clean_repetitions},{total_samples},false,\
{exact_dimension},{clustered_dimension},\
{clustered_target_residual:.17e},\
out_of_radius,nan,nan",
                );
            }
        }
    }

    Ok(())
}
