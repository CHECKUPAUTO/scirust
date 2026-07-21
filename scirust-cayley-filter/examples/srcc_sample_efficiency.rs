//! Deterministic sample-efficiency protocols for Cayley, Clifford, and SRCC.
//!
//! Two supervision contracts are reported separately.
//!
//! `raw_noise`:
//! All methods receive exactly the same observed noise directions.
//! SRCC is restricted to the span of those observations.
//!
//! `relational_srcc`:
//! SRCC receives one seed plus explicit transport correspondences.
//! This is native relational supervision and must not be interpreted as
//! identical input information to the raw-noise protocol.

use scirust_cayley_filter::{
    MultiplierCase, NoiseSubspaceProjector, SEDENION_DIMENSION, Sedenion, analyze_matrix,
    basis_vector, fit_clifford_noise_subspace, left_multiplication_matrix,
    rank_hard_zero_divisor_projectors, squared_norm,
};
use scirust_srcc::{
    LinearMap16, SrccConfig, SrccProjector, SrccTransportSample, evaluate_leave_one_out_stability,
    fit_srcc_projector_from_views,
};

const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];
const TOLERANCE: f64 = 1.0e-12;
const WEIGHT: f64 = 10.0;
const ENERGY_FLOOR: f64 = 1.0e-30;
const REJECTED_DIMENSION: usize = 4;

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

fn cases(basis: &[Sedenion]) -> Vec<Case> {
    let signal = basis_vector(0).expect("the scalar basis direction must exist");

    basis
        .iter()
        .copied()
        .map(|noise| Case { signal, noise })
        .collect()
}

fn multiplier_cases(cases: &[Case]) -> Vec<MultiplierCase> {
    cases
        .iter()
        .map(|case| MultiplierCase::new(case.signal, case.noise))
        .collect()
}

fn squared_distance(left: &Sedenion, right: &Sedenion) -> f64 {
    left.iter().zip(right).fold(0.0, |sum, (a, b)| {
        let difference = a - b;
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

fn signed_identity(sign: f64) -> LinearMap16 {
    core::array::from_fn(|row| {
        core::array::from_fn(|column| if row == column { sign } else { 0.0 })
    })
}

fn srcc_config() -> SrccConfig {
    SrccConfig {
        novelty_threshold: 1.0e-10,
        resonance_threshold: 0.999,
        minimum_support: 2,
        maximum_dimension: REJECTED_DIMENSION,
        maximum_rounds: REJECTED_DIMENSION,
        energy_floor: ENERGY_FLOOR,
    }
}

fn fit_seed_span(observed: &[Sedenion]) -> Result<SrccProjector, String> {
    let transports = [signed_identity(1.0), signed_identity(-1.0)];

    SrccProjector::build(observed, &transports, srcc_config()).map_err(|error| error.to_string())
}

fn relational_views(
    directions: &[Sedenion],
    repetitions: usize,
) -> Result<Vec<Vec<SrccTransportSample>>, String> {
    if directions.len() != REJECTED_DIMENSION
    {
        return Err("expected exactly four kernel directions".into());
    }

    if repetitions == 0
    {
        return Err("repetitions must be strictly positive".into());
    }

    let mut views = Vec::with_capacity(2 * (directions.len() - 1));

    for pair in directions.windows(2)
    {
        let source = pair[0];
        let positive_target = pair[1];
        let negative_target = positive_target.map(|value| -value);

        views.push(
            (0..repetitions)
                .map(|_| SrccTransportSample::new(source, positive_target))
                .collect(),
        );

        views.push(
            (0..repetitions)
                .map(|_| SrccTransportSample::new(source, negative_target))
                .collect(),
        );
    }

    Ok(views)
}

fn run_raw_noise_protocol(kernel: &[Sedenion], test: &[Case]) -> Result<(), String> {
    println!("protocol,observations,method,rejected_dimension,test_loss");

    for observed in 1..=REJECTED_DIMENSION
    {
        let train = cases(&kernel[..observed]);
        let train_multiplier = multiplier_cases(&train);

        let cayley =
            rank_hard_zero_divisor_projectors(&train_multiplier, WEIGHT, TOLERANCE, TOLERANCE)?;

        let selected_cayley = &cayley[0].projector;

        let clifford =
            fit_clifford_noise_subspace(&train_multiplier, REJECTED_DIMENSION, TOLERANCE)?;

        let srcc = fit_seed_span(&kernel[..observed])?;

        let cayley_loss = projector_loss(test, |input| selected_cayley.apply(input));

        let clifford_loss = projector_loss(test, |input| clifford.apply(input));

        let srcc_loss = projector_loss(test, |input| srcc.apply(input));

        println!(
            "raw_noise,{observed},cayley,{},{:.17e}",
            selected_cayley.rejected_dimension(),
            cayley_loss,
        );

        println!(
            "raw_noise,{observed},clifford,{},{:.17e}",
            clifford.rejected_dimension(),
            clifford_loss,
        );

        println!(
            "raw_noise,{observed},srcc_seed_span,{},{:.17e}",
            srcc.rejected_dimension(),
            srcc_loss,
        );

        let expected_span_loss = (REJECTED_DIMENSION - observed) as f64 / REJECTED_DIMENSION as f64;

        if selected_cayley.rejected_dimension() != REJECTED_DIMENSION || cayley_loss > 1.0e-20
        {
            return Err(format!("Cayley failed with {observed} observations",));
        }

        if clifford.rejected_dimension() != observed
            || (clifford_loss - expected_span_loss).abs() > 1.0e-12
        {
            return Err(format!(
                "Clifford span recovery failed with {observed} observations",
            ));
        }

        if srcc.rejected_dimension() != observed
            || srcc.closure().rounds() != 0
            || !srcc.closure().certificates().is_empty()
            || (srcc_loss - expected_span_loss).abs() > 1.0e-12
        {
            return Err(format!(
                "SRCC seed-span recovery failed with {observed} observations",
            ));
        }
    }

    Ok(())
}

fn run_relational_protocol(kernel: &[Sedenion], test: &[Case]) -> Result<(), String> {
    println!(
        "protocol,repetitions,total_transport_samples,\
rejected_dimension,rounds,certificates,\
loo_available,max_loo_distance,dimension_stability,test_loss"
    );

    for repetitions in 1..=4
    {
        let storage = relational_views(kernel, repetitions)?;

        let views: Vec<&[SrccTransportSample]> = storage.iter().map(Vec::as_slice).collect();

        let fit = fit_srcc_projector_from_views(&[kernel[0]], &views, srcc_config())
            .map_err(|error| error.to_string())?;

        let projector = &fit.projector;

        let test_loss = projector_loss(test, |input| projector.apply(input));

        let total_transport_samples = storage.iter().map(Vec::len).sum::<usize>();

        if repetitions == 1
        {
            println!(
                "relational_srcc,{repetitions},{total_transport_samples},\
{},{},{},false,nan,nan,{:.17e}",
                projector.rejected_dimension(),
                projector.closure().rounds(),
                projector.closure().certificates().len(),
                test_loss,
            );
        }
        else
        {
            let stability = evaluate_leave_one_out_stability(&[kernel[0]], &views, srcc_config())
                .map_err(|error| error.to_string())?;

            println!(
                "relational_srcc,{repetitions},{total_transport_samples},\
{},{},{},true,{:.17e},{:.17e},{:.17e}",
                projector.rejected_dimension(),
                projector.closure().rounds(),
                projector.closure().certificates().len(),
                stability.maximum_frobenius_distance,
                stability.dimension_stability_ratio(),
                test_loss,
            );

            if stability.maximum_frobenius_distance > 1.0e-12
                || stability.dimension_stability_ratio() != 1.0
            {
                return Err(format!(
                    "relational SRCC is unstable with {repetitions} repetitions",
                ));
            }
        }

        if projector.rejected_dimension() != REJECTED_DIMENSION
            || projector.closure().rounds() != 3
            || projector.closure().accepted_per_round() != [1, 1, 1]
            || projector.closure().certificates().len() != 3
            || test_loss > 1.0e-20
        {
            return Err(format!(
                "relational SRCC failed with {repetitions} repetitions",
            ));
        }
    }

    Ok(())
}

fn main() -> Result<(), String> {
    let mut multiplier = ZERO;
    multiplier[1] = 1.0;
    multiplier[10] = 1.0;

    let matrix = left_multiplication_matrix(multiplier);

    let analysis = analyze_matrix(&matrix, TOLERANCE).map_err(|error| error.to_string())?;

    if analysis.nullity() != REJECTED_DIMENSION
    {
        return Err(format!(
            "expected nullity {}, found {}",
            REJECTED_DIMENSION,
            analysis.nullity(),
        ));
    }

    let oracle = NoiseSubspaceProjector::new(analysis.kernel_basis(), TOLERANCE)
        .map_err(|error| error.to_string())?;

    let kernel: Vec<_> = oracle
        .orthonormal_basis()
        .iter()
        .copied()
        .map(normalize)
        .collect::<Result<_, _>>()?;

    let test = cases(&kernel);

    run_raw_noise_protocol(&kernel, &test)?;

    println!();

    run_relational_protocol(&kernel, &test)?;

    Ok(())
}
