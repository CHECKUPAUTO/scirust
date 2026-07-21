//! Deterministic mechanism benchmark for source-clustered stable SRCC search.
//!
//! The benchmark compares exact-source robust stable search against the opt-in
//! source-clustered variant.
//!
//! It verifies that:
//!
//! - non-zero source jitter prevents exact-source recovery;
//! - complete-link clustering recovers bounded jitter;
//! - `2+1` target redundancy remains leave-one-out ambiguous;
//! - `3+1` target redundancy receives a perfect stability certificate;
//! - zero source radius and out-of-radius inputs reproduce exact-source search.

use scirust_srcc::{
    SrccCase, SrccConfig, SrccGateDecision, SrccRobustFitError, SrccRobustSourceClusteringConfig,
    SrccRobustStabilityError, SrccRobustStableSearchError, SrccStableSearchConfig,
    SrccStableSearchResult, SrccTransportSample, Vector16, basis_vector,
    search_stable_robust_srcc_structures_from_views,
    search_stable_source_clustered_robust_srcc_structures_from_views, squared_norm,
};

const MAXIMUM_SOURCE_DISTANCE: f64 = 1.0001e-2;
const EXPECTED_DIMENSION: usize = 2;

type StableSearchResult = Result<SrccStableSearchResult, SrccRobustStableSearchError>;

#[derive(Clone, Copy, Debug, PartialEq)]
struct StableOutcome {
    name: &'static str,
    selected_dimension: usize,
    maximum_loo_distance: f64,
    dimension_stability_ratio: f64,
}

fn base_config() -> SrccConfig {
    SrccConfig {
        novelty_threshold: 1.0e-10,
        resonance_threshold: 0.999,
        minimum_support: 2,
        maximum_dimension: EXPECTED_DIMENSION,
        maximum_rounds: EXPECTED_DIMENSION,
        energy_floor: 1.0e-30,
    }
}

fn stable_config() -> SrccStableSearchConfig {
    SrccStableSearchConfig {
        base_config: base_config(),
        distortion_weight: 10.0,
        maximum_frobenius_distance: 0.0,
        minimum_dimension_stability_ratio: 1.0,
        minimum_rejected_dimension: EXPECTED_DIMENSION,
    }
}

fn source_config() -> SrccRobustSourceClusteringConfig {
    SrccRobustSourceClusteringConfig {
        maximum_source_distance: MAXIMUM_SOURCE_DISTANCE,
    }
}

fn cases() -> [SrccCase; 1] {
    [SrccCase::new(
        basis_vector(8).expect("benchmark input axis"),
        basis_vector(2).expect("benchmark rejected axis"),
    )]
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

fn build_views(
    clean_repetitions: usize,
    jitter: f64,
) -> Result<
    (
        Vector16,
        Vec<SrccTransportSample>,
        Vec<SrccTransportSample>,
        f64,
    ),
    String,
> {
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

    Ok((source, positive, negative, source_distance))
}

fn summarize(result: &StableSearchResult) -> Result<StableOutcome, String> {
    match result
    {
        Ok(search) => match &search.decision
        {
            SrccGateDecision::Srcc =>
            {
                let selected = search
                    .selected
                    .as_ref()
                    .ok_or_else(|| "SRCC decision has no selected candidate".to_owned())?;

                Ok(StableOutcome {
                    name: "srcc",
                    selected_dimension: selected.candidate.projector.rejected_dimension(),
                    maximum_loo_distance: selected.stability.maximum_frobenius_distance,
                    dimension_stability_ratio: selected.stability.dimension_stability_ratio(),
                })
            },
            SrccGateDecision::Identity =>
            {
                if search.selected.is_some()
                {
                    return Err("identity decision unexpectedly retained \
a stable selected candidate"
                        .into());
                }

                Ok(StableOutcome {
                    name: "identity",
                    selected_dimension: 0,
                    maximum_loo_distance: f64::NAN,
                    dimension_stability_ratio: f64::NAN,
                })
            },
        },
        Err(SrccRobustStableSearchError::Stability(SrccRobustStabilityError::Fit(
            SrccRobustFitError::AmbiguousTargetConsensus { .. },
        ))) => Ok(StableOutcome {
            name: "ambiguous",
            selected_dimension: 0,
            maximum_loo_distance: f64::NAN,
            dimension_stability_ratio: f64::NAN,
        }),
        Err(error) => Err(format!("unexpected stable-search error: {error}",)),
    }
}

fn validate_expected_outcome(
    label: &str,
    actual: StableOutcome,
    expected: &'static str,
) -> Result<(), String> {
    if actual.name != expected
    {
        return Err(format!(
            "{label}: expected outcome {expected}, \
found {}",
            actual.name,
        ));
    }

    match expected
    {
        "srcc" =>
        {
            if actual.selected_dimension != EXPECTED_DIMENSION
                || actual.maximum_loo_distance != 0.0
                || actual.dimension_stability_ratio != 1.0
            {
                return Err(format!(
                    "{label}: invalid stability certificate: \
dimension={}, maximum_distance={}, ratio={}",
                    actual.selected_dimension,
                    actual.maximum_loo_distance,
                    actual.dimension_stability_ratio,
                ));
            }
        },
        "identity" | "ambiguous" =>
        {
            if actual.selected_dimension != 0
            {
                return Err(format!("{label}: no selected SRCC dimension expected",));
            }
        },
        _ =>
        {
            return Err(format!("{label}: unknown expected outcome {expected}",));
        },
    }

    Ok(())
}

fn main() -> Result<(), String> {
    println!(
        "jitter,source_distance,clean_repetitions,\
within_radius,exact_outcome,exact_selected_dimension,\
clustered_outcome,clustered_selected_dimension,\
clustered_max_loo_distance,\
clustered_dimension_stability,equivalent_pipeline"
    );

    let evaluation_cases = cases();

    for clean_repetitions in [2_usize, 3_usize]
    {
        for jitter in [
            0.0, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-3, 5.0e-3, 1.0e-2, 2.0e-2, 5.0e-2, 1.0e-1,
        ]
        {
            let (source, positive, negative, source_distance) =
                build_views(clean_repetitions, jitter)?;

            let views = [positive.as_slice(), negative.as_slice()];

            let within_radius = source_distance <= MAXIMUM_SOURCE_DISTANCE;

            let exact = search_stable_robust_srcc_structures_from_views(
                &[source],
                &views,
                &[1.0, 0.999],
                stable_config(),
                &evaluation_cases,
                &evaluation_cases,
            );

            let clustered = search_stable_source_clustered_robust_srcc_structures_from_views(
                &[source],
                &views,
                source_config(),
                &[1.0, 0.999],
                stable_config(),
                &evaluation_cases,
                &evaluation_cases,
            );

            let exact_outcome = summarize(&exact)?;
            let clustered_outcome = summarize(&clustered)?;

            let expected_exact = if jitter == 0.0
            {
                if clean_repetitions == 2
                {
                    "ambiguous"
                }
                else
                {
                    "srcc"
                }
            }
            else
            {
                "identity"
            };

            let expected_clustered = if within_radius
            {
                if clean_repetitions == 2
                {
                    "ambiguous"
                }
                else
                {
                    "srcc"
                }
            }
            else
            {
                "identity"
            };

            validate_expected_outcome("exact search", exact_outcome, expected_exact)?;

            validate_expected_outcome(
                "source-clustered search",
                clustered_outcome,
                expected_clustered,
            )?;

            let equivalent_pipeline = jitter == 0.0 || !within_radius;

            if equivalent_pipeline && exact != clustered
            {
                return Err(format!(
                    "pipelines differ despite equivalent source \
partition: jitter={jitter}, \
clean_repetitions={clean_repetitions}",
                ));
            }

            if !equivalent_pipeline && exact == clustered
            {
                return Err(format!(
                    "source clustering produced no observable change \
inside its positive-radius regime: jitter={jitter}, \
clean_repetitions={clean_repetitions}",
                ));
            }

            println!(
                "{jitter:.17e},{source_distance:.17e},\
{clean_repetitions},{within_radius},{},\
{}, {},{}, {:.17e},{:.17e},{}",
                exact_outcome.name,
                exact_outcome.selected_dimension,
                clustered_outcome.name,
                clustered_outcome.selected_dimension,
                clustered_outcome.maximum_loo_distance,
                clustered_outcome.dimension_stability_ratio,
                equivalent_pipeline,
            );
        }
    }

    Ok(())
}
