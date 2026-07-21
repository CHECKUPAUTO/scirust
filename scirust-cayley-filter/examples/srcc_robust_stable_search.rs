//! Deterministic deployment benchmark for robust SRCC stable search.
//!
//! This example compares:
//!
//! - the historical mean-based stable search;
//! - the opt-in robust target-medoid stable search.
//!
//! The true rejected structure is a four-dimensional sedenion kernel.
//! Each explicit view contains repeated clean targets and one contaminated
//! target.
//!
//! Redundancy contracts:
//!
//! - two clean observations plus one contaminant recover the full robust fit,
//!   but do not certify leave-one-out stability;
//! - three clean observations plus one contaminant retain a clean majority
//!   after every leave-one-out removal.

use scirust_cayley_filter::{
    NoiseSubspaceProjector, SEDENION_DIMENSION, Sedenion, analyze_matrix,
    left_multiplication_matrix, squared_norm,
};
use scirust_srcc::{
    SrccCase, SrccConfig, SrccGateDecision, SrccRobustFitError, SrccRobustStabilityError,
    SrccRobustStableSearchError, SrccStableSearchConfig, SrccTransportSample,
    search_stable_robust_srcc_structures_from_views, search_stable_srcc_structures_from_views,
};

const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];
const TOLERANCE: f64 = 1.0e-12;
const ENERGY_FLOOR: f64 = 1.0e-30;
const REJECTED_DIMENSION: usize = 4;
const DISTORTION_WEIGHT: f64 = 10.0;
const THRESHOLDS: [f64; 4] = [0.90, 0.95, 0.99, 0.999];

fn base_config() -> SrccConfig {
    SrccConfig {
        novelty_threshold: 1.0e-10,
        resonance_threshold: 0.999,
        minimum_support: 2,
        maximum_dimension: REJECTED_DIMENSION,
        maximum_rounds: REJECTED_DIMENSION,
        energy_floor: ENERGY_FLOOR,
    }
}

fn stable_config() -> SrccStableSearchConfig {
    SrccStableSearchConfig {
        base_config: base_config(),
        distortion_weight: DISTORTION_WEIGHT,
        maximum_frobenius_distance: 1.0e-12,
        minimum_dimension_stability_ratio: 1.0,
        minimum_rejected_dimension: REJECTED_DIMENSION,
    }
}

fn normalize(vector: Sedenion) -> Result<Sedenion, String> {
    let norm = squared_norm(&vector).sqrt();

    if !norm.is_finite() || norm <= ENERGY_FLOOR.sqrt()
    {
        return Err("cannot normalize a zero or non-finite vector".into());
    }

    Ok(vector.map(|value| value / norm))
}

fn add_scaled(left: Sedenion, right: Sedenion, scale: f64) -> Sedenion {
    core::array::from_fn(|index| left[index] + scale * right[index])
}

fn deterministic_probe(split: usize, sample: usize) -> Sedenion {
    core::array::from_fn(|coordinate| {
        let integer = ((split + 5) * 41
            + (sample + 7) * 37
            + (coordinate + 11) * 23
            + (split + 3) * (sample + 5) * (coordinate + 7))
            % 67;

        (integer as f64 - 33.0) / 10.0
    })
}

fn make_cases(oracle: &NoiseSubspaceProjector, split: usize, count: usize) -> Vec<SrccCase> {
    let basis = oracle.orthonormal_basis();

    (0..count)
        .map(|sample| {
            let raw = deterministic_probe(split, sample);

            let signal = oracle.apply(&raw);
            let mut noise = ZERO;

            for (direction_index, direction) in basis.iter().enumerate()
            {
                let integer = ((split + 3) * 19
                    + (sample + 5) * 17
                    + (direction_index + 7) * 13
                    + (split + 1) * (sample + 2) * (direction_index + 3))
                    % 17;

                let centered = integer as f64 - 8.0;

                let coefficient = if centered == 0.0
                {
                    0.25
                }
                else
                {
                    centered / 5.0
                };

                for coordinate in 0..SEDENION_DIMENSION
                {
                    noise[coordinate] += coefficient * direction[coordinate];
                }
            }

            SrccCase::new(signal, noise)
        })
        .collect()
}

fn orthogonal_contaminant(oracle: &NoiseSubspaceProjector) -> Result<Sedenion, String> {
    for coordinate in 0..SEDENION_DIMENSION
    {
        let mut axis = ZERO;
        axis[coordinate] = 1.0;

        let projected = oracle.apply(&axis);

        if squared_norm(&projected) > 1.0e-12
        {
            return normalize(projected);
        }
    }

    Err("no off-kernel contaminant was found".into())
}

fn build_views(
    directions: &[Sedenion],
    contaminant: Sedenion,
    epsilon: f64,
    clean_repetitions: usize,
) -> Result<Vec<Vec<SrccTransportSample>>, String> {
    if directions.len() != REJECTED_DIMENSION
    {
        return Err("expected exactly four kernel directions".into());
    }

    if clean_repetitions < 2
    {
        return Err("at least two clean repetitions are required".into());
    }

    let mut views = Vec::with_capacity(2 * (directions.len() - 1));

    for pair in directions.windows(2)
    {
        let source = pair[0];
        let target = pair[1];
        let negative = target.map(|value| -value);

        let contaminated_positive = normalize(add_scaled(target, contaminant, epsilon))?;

        let contaminated_negative = normalize(add_scaled(negative, contaminant, epsilon))?;

        let mut positive = Vec::with_capacity(clean_repetitions + 1);

        let mut negative_view = Vec::with_capacity(clean_repetitions + 1);

        for _ in 0..clean_repetitions
        {
            positive.push(SrccTransportSample::new(source, target));

            negative_view.push(SrccTransportSample::new(source, negative));
        }

        positive.push(SrccTransportSample::new(source, contaminated_positive));

        negative_view.push(SrccTransportSample::new(source, contaminated_negative));

        views.push(positive);
        views.push(negative_view);
    }

    Ok(views)
}

fn decision_name(decision: SrccGateDecision) -> &'static str {
    match decision
    {
        SrccGateDecision::Srcc => "Srcc",
        SrccGateDecision::Identity => "Identity",
    }
}

fn main() -> Result<(), String> {
    let mut multiplier = ZERO;
    multiplier[1] = 1.0;
    multiplier[10] = 1.0;

    let matrix = left_multiplication_matrix(multiplier);

    let analysis = analyze_matrix(&matrix, TOLERANCE).map_err(|error| error.to_string())?;

    let oracle = NoiseSubspaceProjector::new(analysis.kernel_basis(), TOLERANCE)
        .map_err(|error| error.to_string())?;

    if oracle.dimension() != REJECTED_DIMENSION
    {
        return Err(format!(
            "expected nullity {}, found {}",
            REJECTED_DIMENSION,
            oracle.dimension(),
        ));
    }

    let directions: Vec<_> = oracle
        .orthonormal_basis()
        .iter()
        .copied()
        .map(normalize)
        .collect::<Result<_, _>>()?;

    let contaminant = orthogonal_contaminant(&oracle)?;

    let train = make_cases(&oracle, 1, 24);
    let dev = make_cases(&oracle, 2, 24);

    println!(
        "epsilon,clean_repetitions,total_samples,\
standard_decision,standard_selected_dimension,\
standard_candidate_dev_loss,\
robust_status,robust_decision,\
robust_selected_dimension,robust_dev_loss,\
robust_max_loo_distance,\
robust_dimension_stability"
    );

    for clean_repetitions in [2_usize, 3_usize]
    {
        for epsilon in [
            0.0, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1, 2.5e-1, 5.0e-1, 1.0,
        ]
        {
            let storage = build_views(&directions, contaminant, epsilon, clean_repetitions)?;

            let views: Vec<&[SrccTransportSample]> = storage.iter().map(Vec::as_slice).collect();

            let total_samples = storage.iter().map(Vec::len).sum::<usize>();

            let standard = search_stable_srcc_structures_from_views(
                &[directions[0]],
                &views,
                &THRESHOLDS,
                stable_config(),
                &train,
                &dev,
            )
            .map_err(|error| error.to_string())?;

            let (standard_dimension, standard_dev_loss) = match standard.selected.as_ref()
            {
                Some(selected) => (
                    selected.candidate.projector.rejected_dimension(),
                    selected.candidate.dev_score.loss,
                ),
                None => (0, f64::NAN),
            };

            if epsilon == 0.0
            {
                if standard.decision != SrccGateDecision::Srcc
                {
                    return Err(format!(
                        "clean historical search failed for \
clean_repetitions={clean_repetitions}",
                    ));
                }
            }
            else if standard.decision != SrccGateDecision::Identity
            {
                return Err(format!(
                    "historical search failed to fall back for \
epsilon={epsilon}, \
clean_repetitions={clean_repetitions}",
                ));
            }

            match search_stable_robust_srcc_structures_from_views(
                &[directions[0]],
                &views,
                &THRESHOLDS,
                stable_config(),
                &train,
                &dev,
            )
            {
                Ok(robust) =>
                {
                    let selected = robust.selected.as_ref().ok_or_else(|| {
                        format!(
                            "robust search selected no candidate \
for epsilon={epsilon}, \
clean_repetitions={clean_repetitions}",
                        )
                    })?;

                    let robust_dimension = selected.candidate.projector.rejected_dimension();

                    let robust_dev_loss = selected.candidate.dev_score.loss;

                    let maximum_distance = selected.stability.maximum_frobenius_distance;

                    let dimension_stability = selected.stability.dimension_stability_ratio();

                    if robust.decision != SrccGateDecision::Srcc
                        || robust_dimension != REJECTED_DIMENSION
                        || robust_dev_loss > 1.0e-20
                        || maximum_distance != 0.0
                        || dimension_stability != 1.0
                    {
                        return Err(format!(
                            "robust stable search failed for \
epsilon={epsilon}, \
clean_repetitions={clean_repetitions}",
                        ));
                    }

                    println!(
                        "{epsilon},{clean_repetitions},\
{total_samples},{},{},{:.17e},\
stable,{},{},{:.17e},{:.17e},{:.17e}",
                        decision_name(standard.decision),
                        standard_dimension,
                        standard_dev_loss,
                        decision_name(robust.decision),
                        robust_dimension,
                        robust_dev_loss,
                        maximum_distance,
                        dimension_stability,
                    );
                },
                Err(SrccRobustStableSearchError::Stability(SrccRobustStabilityError::Fit(
                    SrccRobustFitError::AmbiguousTargetConsensus { .. },
                ))) if clean_repetitions == 2 && epsilon > 0.0 =>
                {
                    println!(
                        "{epsilon},{clean_repetitions},\
{total_samples},{},{},{:.17e},\
ambiguous,NA,0,nan,nan,nan",
                        decision_name(standard.decision),
                        standard_dimension,
                        standard_dev_loss,
                    );
                },
                Err(error) =>
                {
                    return Err(format!(
                        "unexpected robust search result for \
epsilon={epsilon}, \
clean_repetitions={clean_repetitions}: {error}",
                    ));
                },
            }
        }
    }

    Ok(())
}
