//! Deterministic diagnostic for SRCC transport-view contamination.
//!
//! The true rejected subspace is a four-dimensional sedenion kernel.
//! SRCC receives one seed and must recover three further directions.
//!
//! Every transport view contains two clean observations and one contaminated
//! observation. The contamination amplitude is swept deterministically.
//! Development scoring and leave-one-out stability are both reported.
//!
//! This is a native SRCC supervision diagnostic. Cayley and Clifford are
//! evaluated only as fixed references on the same signal/noise cases because
//! they do not consume transport correspondences.

use scirust_cayley_filter::{
    MultiplierCase, NoiseSubspaceProjector, SEDENION_DIMENSION, Sedenion, analyze_matrix,
    fit_clifford_noise_subspace, left_multiplication_matrix, score_clifford_projector,
    select_hard_cayley_train_dev, squared_norm,
};
use scirust_srcc::{
    SrccCase, SrccConfig, SrccGateDecision, SrccStableSearchConfig, SrccTransportSample,
    search_stable_srcc_structures_from_views,
};

const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];
const TOLERANCE: f64 = 1.0e-12;
const WEIGHT: f64 = 10.0;
const ENERGY_FLOOR: f64 = 1.0e-30;
const REJECTED_DIMENSION: usize = 4;
const CAYLEY_TOP_K: usize = 84;

#[derive(Clone, Copy)]
struct Case {
    signal: Sedenion,
    noise: Sedenion,
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

fn deterministic_probe(sample: usize) -> Sedenion {
    core::array::from_fn(|coordinate| {
        let integer =
            ((sample + 5) * 37 + (coordinate + 11) * 23 + (sample + 3) * (coordinate + 7) * 5) % 61;

        (integer as f64 - 30.0) / 9.0
    })
}

fn make_cases(oracle: &NoiseSubspaceProjector, split: usize, count: usize) -> Vec<Case> {
    let basis = oracle.orthonormal_basis();

    (0..count)
        .map(|sample| {
            let raw = deterministic_probe(split * 100 + sample);
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

            Case { signal, noise }
        })
        .collect()
}

fn as_multiplier_cases(cases: &[Case]) -> Vec<MultiplierCase> {
    cases
        .iter()
        .map(|case| MultiplierCase::new(case.signal, case.noise))
        .collect()
}

fn as_srcc_cases(cases: &[Case]) -> Vec<SrccCase> {
    cases
        .iter()
        .map(|case| SrccCase::new(case.signal, case.noise))
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
) -> Result<Vec<Vec<SrccTransportSample>>, String> {
    if directions.len() != REJECTED_DIMENSION
    {
        return Err("expected exactly four training directions".into());
    }

    let mut views = Vec::with_capacity(2 * (directions.len() - 1));

    for pair in directions.windows(2)
    {
        let source = pair[0];
        let target = pair[1];

        let contaminated_positive = normalize(add_scaled(target, contaminant, epsilon))?;

        let contaminated_negative =
            normalize(add_scaled(target.map(|value| -value), contaminant, epsilon))?;

        views.push(vec![
            SrccTransportSample::new(source, target),
            SrccTransportSample::new(source, target),
            SrccTransportSample::new(source, contaminated_positive),
        ]);

        views.push(vec![
            SrccTransportSample::new(source, target.map(|value| -value)),
            SrccTransportSample::new(source, target.map(|value| -value)),
            SrccTransportSample::new(source, contaminated_negative),
        ]);
    }

    Ok(views)
}

fn projector_loss(cases: &[Case], mut apply: impl FnMut(&Sedenion) -> Sedenion) -> f64 {
    let mut noise = 0.0;
    let mut distortion = 0.0;

    for case in cases
    {
        let filtered_noise = apply(&case.noise);
        let filtered_signal = apply(&case.signal);

        noise += squared_norm(&filtered_noise) / squared_norm(&case.noise).max(ENERGY_FLOOR);

        distortion += case
            .signal
            .iter()
            .zip(filtered_signal)
            .map(|(input, output)| {
                let difference = input - output;
                difference * difference
            })
            .sum::<f64>()
            / squared_norm(&case.signal).max(ENERGY_FLOOR);
    }

    let count = cases.len() as f64;
    noise / count + WEIGHT * distortion / count
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

    let train = make_cases(&oracle, 0, 12);
    let dev = make_cases(&oracle, 1, 12);
    let test = make_cases(&oracle, 2, 24);

    let train_multiplier = as_multiplier_cases(&train);
    let dev_multiplier = as_multiplier_cases(&dev);

    let cayley = select_hard_cayley_train_dev(
        &train_multiplier,
        &dev_multiplier,
        CAYLEY_TOP_K,
        WEIGHT,
        TOLERANCE,
        TOLERANCE,
    )?;

    let clifford = fit_clifford_noise_subspace(&train_multiplier, REJECTED_DIMENSION, TOLERANCE)?;

    let cayley_test = projector_loss(&test, |input| cayley.selected.projector.apply(input));

    let clifford_test =
        score_clifford_projector(&as_multiplier_cases(&test), &clifford, WEIGHT)?.loss;

    println!(
        "epsilon,decision,stable_candidate,dimension,rounds,certificates,\
max_loo_distance,dimension_stability,dev_loss,test_loss,\
cayley_test,clifford_test"
    );

    for epsilon in [
        0.0, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1, 2.5e-1, 5.0e-1, 1.0,
    ]
    {
        let storage = build_views(&directions, contaminant, epsilon)?;

        let views: Vec<&[SrccTransportSample]> = storage.iter().map(Vec::as_slice).collect();

        let result = search_stable_srcc_structures_from_views(
            &[directions[0]],
            &views,
            &[0.90, 0.95, 0.99, 0.999],
            SrccStableSearchConfig {
                base_config: SrccConfig {
                    novelty_threshold: 1.0e-10,
                    resonance_threshold: 0.999,
                    minimum_support: 2,
                    maximum_dimension: REJECTED_DIMENSION,
                    maximum_rounds: REJECTED_DIMENSION,
                    energy_floor: ENERGY_FLOOR,
                },
                distortion_weight: WEIGHT,
                maximum_frobenius_distance: 5.0e-2,
                minimum_dimension_stability_ratio: 1.0,
                minimum_rejected_dimension: REJECTED_DIMENSION,
            },
            &as_srcc_cases(&train),
            &as_srcc_cases(&dev),
        )
        .map_err(|error| error.to_string())?;

        match &result.selected
        {
            Some(selected) =>
            {
                let projector = &selected.candidate.projector;

                let test_loss = projector_loss(&test, |input| projector.apply(input));

                println!(
                    "{epsilon},{:?},true,{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
                    result.decision,
                    projector.rejected_dimension(),
                    projector.closure().rounds(),
                    projector.closure().certificates().len(),
                    selected.stability.maximum_frobenius_distance,
                    selected.stability.dimension_stability_ratio(),
                    selected.candidate.dev_score.loss,
                    test_loss,
                    cayley_test,
                    clifford_test,
                );
            },
            None =>
            {
                println!(
                    "{epsilon},{:?},false,0,0,0,nan,nan,{:.17e},{:.17e},{:.17e},{:.17e}",
                    result.decision,
                    result.identity_dev_score.loss,
                    1.0,
                    cayley_test,
                    clifford_test,
                );
            },
        }

        if epsilon == 0.0
        {
            let selected = result
                .selected
                .as_ref()
                .ok_or_else(|| "clean SRCC structure was rejected".to_string())?;

            if result.decision != SrccGateDecision::Srcc
                || selected.candidate.projector.rejected_dimension() != REJECTED_DIMENSION
                || selected.candidate.projector.closure().rounds() != 3
                || selected.candidate.projector.closure().certificates().len() != 3
            {
                return Err("clean SRCC closure did not satisfy the oracle".into());
            }
        }
    }

    Ok(())
}
