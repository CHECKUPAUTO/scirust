//! Deterministic comparison of mean and robust SRCC transport fitting.
//!
//! The true rejected subspace is a four-dimensional sedenion kernel.
//! Every explicit transport view contains repeated clean observations and
//! one contaminated target.
//!
//! Two redundancy contracts are reported:
//!
//! - two clean samples plus one contaminant: robust full fitting succeeds,
//!   but leave-one-out removal of a clean sample creates an ambiguous 1-1 tie;
//! - three clean samples plus one contaminant: robust fitting remains exact
//!   after every leave-one-out removal.

use scirust_cayley_filter::{
    NoiseSubspaceProjector, SEDENION_DIMENSION, Sedenion, analyze_matrix,
    left_multiplication_matrix, squared_norm,
};
use scirust_srcc::{
    SrccConfig, SrccRobustFitError, SrccRobustStabilityError, SrccTransportSample,
    evaluate_robust_leave_one_out_stability, fit_robust_srcc_projector_from_views,
    fit_srcc_projector_from_views,
};

const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];
const TOLERANCE: f64 = 1.0e-12;
const ENERGY_FLOOR: f64 = 1.0e-30;
const REJECTED_DIMENSION: usize = 4;
const WEIGHT: f64 = 10.0;

#[derive(Clone, Copy)]
struct Case {
    signal: Sedenion,
    noise: Sedenion,
}

fn config() -> SrccConfig {
    SrccConfig {
        novelty_threshold: 1.0e-10,
        resonance_threshold: 0.999,
        minimum_support: 2,
        maximum_dimension: REJECTED_DIMENSION,
        maximum_rounds: REJECTED_DIMENSION,
        energy_floor: ENERGY_FLOOR,
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

        let mut positive_view = Vec::with_capacity(clean_repetitions + 1);

        let mut negative_view = Vec::with_capacity(clean_repetitions + 1);

        for _ in 0..clean_repetitions
        {
            positive_view.push(SrccTransportSample::new(source, target));

            negative_view.push(SrccTransportSample::new(source, negative));
        }

        positive_view.push(SrccTransportSample::new(source, contaminated_positive));

        negative_view.push(SrccTransportSample::new(source, contaminated_negative));

        views.push(positive_view);
        views.push(negative_view);
    }

    Ok(views)
}

fn squared_distance(left: &Sedenion, right: &Sedenion) -> f64 {
    left.iter().zip(right).fold(0.0, |sum, (left, right)| {
        let difference = left - right;
        sum + difference * difference
    })
}

fn projector_loss(cases: &[Case], mut apply: impl FnMut(&Sedenion) -> Sedenion) -> f64 {
    let mut noise = 0.0;
    let mut distortion = 0.0;

    for case in cases
    {
        let filtered_noise = apply(&case.noise);
        let filtered_signal = apply(&case.signal);

        noise += squared_norm(&filtered_noise) / squared_norm(&case.noise).max(ENERGY_FLOOR);

        distortion += squared_distance(&case.signal, &filtered_signal)
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
    let test = make_cases(&oracle, 7, 32);

    println!(
        "epsilon,clean_repetitions,total_samples,\
mean_dimension,mean_rounds,mean_test_loss,\
robust_dimension,robust_rounds,robust_certificates,\
robust_test_loss,robust_loo_status,\
robust_max_loo_distance,robust_dimension_stability"
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

            let mean = fit_srcc_projector_from_views(&[directions[0]], &views, config())
                .map_err(|error| error.to_string())?;

            let robust = fit_robust_srcc_projector_from_views(&[directions[0]], &views, config())
                .map_err(|error| error.to_string())?;

            let mean_loss = projector_loss(&test, |input| mean.projector.apply(input));

            let robust_loss = projector_loss(&test, |input| robust.projector.apply(input));

            if robust.projector.rejected_dimension() != REJECTED_DIMENSION
                || robust.projector.closure().rounds() != 3
                || robust.projector.closure().accepted_per_round() != [1, 1, 1]
                || robust.projector.closure().certificates().len() != 3
                || robust_loss > 1.0e-20
            {
                return Err(format!(
                    "robust fitting failed for epsilon={epsilon}, \
clean_repetitions={clean_repetitions}",
                ));
            }

            match evaluate_robust_leave_one_out_stability(&[directions[0]], &views, config())
            {
                Ok(report) =>
                {
                    if report.maximum_frobenius_distance != 0.0
                        || report.dimension_stability_ratio() != 1.0
                    {
                        return Err(format!(
                            "unexpected robust instability for \
epsilon={epsilon}, \
clean_repetitions={clean_repetitions}",
                        ));
                    }

                    println!(
                        "{epsilon},{clean_repetitions},\
{total_samples},{},{},{:.17e},\
{},{},{},{:.17e},stable,{:.17e},{:.17e}",
                        mean.projector.rejected_dimension(),
                        mean.projector.closure().rounds(),
                        mean_loss,
                        robust.projector.rejected_dimension(),
                        robust.projector.closure().rounds(),
                        robust.projector.closure().certificates().len(),
                        robust_loss,
                        report.maximum_frobenius_distance,
                        report.dimension_stability_ratio(),
                    );
                },
                Err(SrccRobustStabilityError::Fit(
                    SrccRobustFitError::AmbiguousTargetConsensus { .. },
                )) if clean_repetitions == 2 && epsilon > 0.0 =>
                {
                    println!(
                        "{epsilon},{clean_repetitions},\
{total_samples},{},{},{:.17e},\
{},{},{},{:.17e},ambiguous,nan,nan",
                        mean.projector.rejected_dimension(),
                        mean.projector.closure().rounds(),
                        mean_loss,
                        robust.projector.rejected_dimension(),
                        robust.projector.closure().rounds(),
                        robust.projector.closure().certificates().len(),
                        robust_loss,
                    );
                },
                Err(error) =>
                {
                    return Err(format!(
                        "unexpected robust stability result for \
epsilon={epsilon}, \
clean_repetitions={clean_repetitions}: {error}",
                    ));
                },
            }
        }
    }

    Ok(())
}
