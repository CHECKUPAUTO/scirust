//! Train/dev/test comparison exercising genuine SRCC resonant closure.
//!
//! All evaluated projectors use the same signal/noise splits and the same
//! rejected dimension.
//!
//! SRCC receives one seed extracted from the training split. Its transport
//! views are derived only from the first four training-noise observations.
//! It must recover the remaining three directions through three successive
//! consensus-closure rounds. Neither development nor test data participates
//! in fitting.

use scirust_cayley_filter::{
    DevelopmentGateDecision, MultiplierCase, NoiseSubspaceProjector, SEDENION_DIMENSION, Sedenion,
    SplitCliffordProjector, analyze_matrix, fit_clifford_noise_subspace,
    left_multiplication_matrix, select_hard_cayley_train_dev, squared_norm,
};
use scirust_srcc::{
    SrccCase, SrccConfig, SrccGateDecision, SrccProjector, SrccTransportSample,
    search_srcc_structures_from_views,
};

const ZERO: Sedenion = [0.0; SEDENION_DIMENSION];
const ANALYSIS_TOLERANCE: f64 = 1.0e-12;
const RELATIVE_THRESHOLD: f64 = 1.0e-12;
const DISTORTION_WEIGHT: f64 = 10.0;
const ENERGY_FLOOR: f64 = 1.0e-30;
const REJECTED_DIMENSION: usize = 4;
const FAMILY_COUNT: usize = 3;
const CAYLEY_TOP_K: usize = 84;
const TRAIN_CASES: usize = 8;
const DEV_CASES: usize = 8;
const TEST_CASES: usize = 16;

#[derive(Clone, Copy, Debug)]
struct Case {
    signal: Sedenion,
    noise: Sedenion,
}

#[derive(Clone, Copy, Debug)]
struct Score {
    loss: f64,
    residual_noise_ratio: f64,
    signal_distortion_ratio: f64,
}

fn family_multiplier(family: usize) -> Result<Sedenion, String> {
    let second_index = match family
    {
        0 => 10,
        1 => 11,
        2 => 12,
        _ => return Err(format!("invalid family index {family}")),
    };

    let mut multiplier = ZERO;
    multiplier[1] = 1.0;
    multiplier[second_index] = -1.0;
    Ok(multiplier)
}

fn deterministic_vector(family: usize, split: usize, sample: usize) -> Sedenion {
    core::array::from_fn(|coordinate| {
        let integer = ((family + 3) * 47
            + (split + 5) * 43
            + (sample + 7) * 37
            + (coordinate + 11) * 19
            + (family + 1) * (split + 2) * (sample + 3) * (coordinate + 5) * 7)
            % 67;

        (integer as f64 - 33.0) / 10.0
    })
}

fn mixed_noise(basis: &[Sedenion], family: usize, split: usize, sample: usize) -> Sedenion {
    if split == 0 && sample < basis.len()
    {
        let amplitude = 0.75 + 0.2 * sample as f64;
        return basis[sample].map(|value| amplitude * value);
    }

    let mut noise = ZERO;

    for (direction_index, direction) in basis.iter().enumerate()
    {
        let integer = ((family + 2) * 23
            + (split + 3) * 19
            + (sample + 5) * 17
            + (direction_index + 7) * 13
            + (family + 1) * (split + 2) * (sample + 3) * (direction_index + 5))
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

    noise
}

fn make_split(
    oracle: &NoiseSubspaceProjector,
    family: usize,
    split: usize,
    count: usize,
) -> Vec<Case> {
    (0..count)
        .map(|sample| {
            let raw = deterministic_vector(family, split, sample);

            Case {
                signal: oracle.apply(&raw),
                noise: mixed_noise(oracle.orthonormal_basis(), family, split, sample),
            }
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

fn normalize(vector: &Sedenion) -> Result<Sedenion, String> {
    let norm = squared_norm(vector).sqrt();

    if !norm.is_finite() || norm <= ENERGY_FLOOR.sqrt()
    {
        return Err("training direction must have finite non-zero energy".into());
    }

    Ok(vector.map(|value| value / norm))
}

fn training_directions(train: &[Case]) -> Result<Vec<Sedenion>, String> {
    if train.len() < REJECTED_DIMENSION
    {
        return Err(format!(
            "at least {REJECTED_DIMENSION} training cases are required",
        ));
    }

    train[..REJECTED_DIMENSION]
        .iter()
        .map(|case| normalize(&case.noise))
        .collect()
}

fn transport_views(directions: &[Sedenion]) -> Result<Vec<Vec<SrccTransportSample>>, String> {
    if directions.len() != REJECTED_DIMENSION
    {
        return Err(format!("expected {REJECTED_DIMENSION} training directions",));
    }

    let mut views = Vec::with_capacity(2 * (directions.len() - 1));

    for pair in directions.windows(2)
    {
        let source = pair[0];
        let target = pair[1];

        views.push(vec![SrccTransportSample::new(source, target)]);

        views.push(vec![SrccTransportSample::new(
            source,
            target.map(|value| -value),
        )]);
    }

    Ok(views)
}

fn fit_resonant_srcc(
    train: &[Case],
    dev: &[Case],
) -> Result<(SrccProjector, SrccGateDecision, f64, usize), String> {
    let directions = training_directions(train)?;
    let storage = transport_views(&directions)?;

    let views: Vec<&[SrccTransportSample]> = storage.iter().map(Vec::as_slice).collect();

    let train_cases = as_srcc_cases(train);
    let dev_cases = as_srcc_cases(dev);

    let base_config = SrccConfig {
        novelty_threshold: 1.0e-10,
        resonance_threshold: 0.999,
        minimum_support: 2,
        maximum_dimension: REJECTED_DIMENSION,
        maximum_rounds: REJECTED_DIMENSION,
        energy_floor: ENERGY_FLOOR,
    };

    let result = search_srcc_structures_from_views(
        &[directions[0]],
        &views,
        &[0.999, 1.0],
        base_config,
        &train_cases,
        &dev_cases,
        DISTORTION_WEIGHT,
    )
    .map_err(|error| error.to_string())?;

    Ok((
        result.selected.projector,
        result.decision,
        result.selected.resonance_threshold,
        result.candidates.len(),
    ))
}

fn squared_distance(left: &Sedenion, right: &Sedenion) -> f64 {
    left.iter().zip(right).fold(0.0, |sum, (a, b)| {
        let difference = a - b;
        sum + difference * difference
    })
}

fn evaluate(cases: &[Case], mut apply: impl FnMut(&Sedenion) -> Sedenion) -> Score {
    let mut residual_noise_ratio = 0.0;
    let mut signal_distortion_ratio = 0.0;

    for case in cases
    {
        let filtered_signal = apply(&case.signal);
        let filtered_noise = apply(&case.noise);

        residual_noise_ratio +=
            squared_norm(&filtered_noise) / squared_norm(&case.noise).max(ENERGY_FLOOR);

        signal_distortion_ratio += squared_distance(&case.signal, &filtered_signal)
            / squared_norm(&case.signal).max(ENERGY_FLOOR);
    }

    let inverse_count = 1.0 / cases.len() as f64;
    let residual_noise_ratio = residual_noise_ratio * inverse_count;
    let signal_distortion_ratio = signal_distortion_ratio * inverse_count;

    Score {
        loss: residual_noise_ratio + DISTORTION_WEIGHT * signal_distortion_ratio,
        residual_noise_ratio,
        signal_distortion_ratio,
    }
}

fn maximum_gap(
    family: usize,
    mut left: impl FnMut(&Sedenion) -> Sedenion,
    mut right: impl FnMut(&Sedenion) -> Sedenion,
) -> f64 {
    let mut maximum = 0.0_f64;

    for index in 0..SEDENION_DIMENSION
    {
        let mut basis = ZERO;
        basis[index] = 1.0;

        maximum = maximum.max(squared_distance(&left(&basis), &right(&basis)));
    }

    for sample in 0..32
    {
        let probe = deterministic_vector(family, 9, sample);

        maximum = maximum.max(squared_distance(&left(&probe), &right(&probe)));
    }

    maximum.sqrt()
}

fn print_result(
    family: usize,
    method: &str,
    rejected_dimension: usize,
    train_score: Score,
    dev_score: Score,
    test_score: Score,
    dev_beats_identity: bool,
) {
    println!(
        "{family},{method},{rejected_dimension},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{dev_beats_identity}",
        train_score.loss,
        dev_score.loss,
        test_score.loss,
        test_score.residual_noise_ratio,
        test_score.signal_distortion_ratio,
    );
}

fn main() -> Result<(), String> {
    println!(
        "family,method,rejected_dimension,train_loss,dev_loss,test_loss,test_noise_ratio,test_distortion_ratio,dev_beats_identity"
    );

    for family in 0..FAMILY_COUNT
    {
        let multiplier = family_multiplier(family)?;
        let matrix = left_multiplication_matrix(multiplier);

        let analysis =
            analyze_matrix(&matrix, ANALYSIS_TOLERANCE).map_err(|error| error.to_string())?;

        let oracle = NoiseSubspaceProjector::new(analysis.kernel_basis(), ANALYSIS_TOLERANCE)
            .map_err(|error| error.to_string())?;

        if oracle.dimension() != REJECTED_DIMENSION
        {
            return Err(format!(
                "family {family}: expected oracle dimension {}, found {}",
                REJECTED_DIMENSION,
                oracle.dimension(),
            ));
        }

        let train = make_split(&oracle, family, 0, TRAIN_CASES);
        let dev = make_split(&oracle, family, 1, DEV_CASES);
        let test = make_split(&oracle, family, 2, TEST_CASES);

        let train_multiplier = as_multiplier_cases(&train);
        let dev_multiplier = as_multiplier_cases(&dev);

        let cayley_selection = select_hard_cayley_train_dev(
            &train_multiplier,
            &dev_multiplier,
            CAYLEY_TOP_K,
            DISTORTION_WEIGHT,
            ANALYSIS_TOLERANCE,
            RELATIVE_THRESHOLD,
        )?;

        let cayley = &cayley_selection.selected.projector;

        let clifford: SplitCliffordProjector =
            fit_clifford_noise_subspace(&train_multiplier, REJECTED_DIMENSION, ANALYSIS_TOLERANCE)?;

        let (srcc, srcc_decision, srcc_threshold, srcc_candidates) =
            fit_resonant_srcc(&train, &dev)?;

        if cayley.rejected_dimension() != REJECTED_DIMENSION
            || clifford.rejected_dimension() != REJECTED_DIMENSION
            || srcc.rejected_dimension() != REJECTED_DIMENSION
        {
            return Err(format!(
                "family {family}: learned dimensions are not matched",
            ));
        }

        let identity_train = evaluate(&train, |input| *input);
        let identity_dev = evaluate(&dev, |input| *input);
        let identity_test = evaluate(&test, |input| *input);

        let oracle_train = evaluate(&train, |input| oracle.apply(input));
        let oracle_dev = evaluate(&dev, |input| oracle.apply(input));
        let oracle_test = evaluate(&test, |input| oracle.apply(input));

        let cayley_train = evaluate(&train, |input| cayley.apply(input));
        let cayley_dev = evaluate(&dev, |input| cayley.apply(input));
        let cayley_test = evaluate(&test, |input| cayley.apply(input));

        let clifford_train = evaluate(&train, |input| clifford.apply(input));
        let clifford_dev = evaluate(&dev, |input| clifford.apply(input));
        let clifford_test = evaluate(&test, |input| clifford.apply(input));

        let srcc_train = evaluate(&train, |input| srcc.apply(input));
        let srcc_dev = evaluate(&dev, |input| srcc.apply(input));
        let srcc_test = evaluate(&test, |input| srcc.apply(input));

        print_result(
            family,
            "identity",
            0,
            identity_train,
            identity_dev,
            identity_test,
            false,
        );

        print_result(
            family,
            "real_subspace_oracle",
            oracle.dimension(),
            oracle_train,
            oracle_dev,
            oracle_test,
            oracle_dev.loss < identity_dev.loss,
        );

        print_result(
            family,
            "learned_cayley",
            cayley.rejected_dimension(),
            cayley_train,
            cayley_dev,
            cayley_test,
            cayley_dev.loss < identity_dev.loss,
        );

        print_result(
            family,
            "learned_clifford",
            clifford.rejected_dimension(),
            clifford_train,
            clifford_dev,
            clifford_test,
            clifford_dev.loss < identity_dev.loss,
        );

        print_result(
            family,
            "learned_srcc_resonant",
            srcc.rejected_dimension(),
            srcc_train,
            srcc_dev,
            srcc_test,
            srcc_dev.loss < identity_dev.loss,
        );

        let cayley_gap = maximum_gap(
            family,
            |input| oracle.apply(input),
            |input| cayley.apply(input),
        );

        let clifford_gap = maximum_gap(
            family,
            |input| oracle.apply(input),
            |input| clifford.apply(input),
        );

        let srcc_gap = maximum_gap(
            family,
            |input| oracle.apply(input),
            |input| srcc.apply(input),
        );

        println!(
            "# family={family} cayley_decision={:?} cayley_candidates={} srcc_decision={:?} srcc_threshold={:.17e} srcc_candidates={} srcc_rounds={} srcc_accepted={:?} srcc_certificates={} cayley_gap={:.17e} clifford_gap={:.17e} srcc_gap={:.17e}",
            cayley_selection.decision,
            cayley_selection.candidates.len(),
            srcc_decision,
            srcc_threshold,
            srcc_candidates,
            srcc.closure().rounds(),
            srcc.closure().accepted_per_round(),
            srcc.closure().certificates().len(),
            cayley_gap,
            clifford_gap,
            srcc_gap,
        );

        if cayley_selection.decision != DevelopmentGateDecision::Cayley
        {
            return Err(format!(
                "family {family}: Cayley failed its development gate",
            ));
        }

        if srcc_decision != SrccGateDecision::Srcc
        {
            return Err(format!("family {family}: SRCC failed its development gate",));
        }

        if srcc.closure().rounds() != 3
            || srcc.closure().accepted_per_round() != [1, 1, 1]
            || srcc.closure().certificates().len() != 3
        {
            return Err(format!(
                "family {family}: SRCC did not produce the expected three-round closure",
            ));
        }

        if cayley_dev.loss >= identity_dev.loss
            || clifford_dev.loss >= identity_dev.loss
            || srcc_dev.loss >= identity_dev.loss
        {
            return Err(format!(
                "family {family}: a learned projector failed its development gate",
            ));
        }

        if oracle_test.loss > 1.0e-20
            || cayley_test.loss > 1.0e-20
            || clifford_test.loss > 1.0e-20
            || srcc_test.loss > 1.0e-20
        {
            return Err(format!(
                "family {family}: a learned projector failed the held-out test oracle",
            ));
        }

        if cayley_gap > 1.0e-10 || clifford_gap > 1.0e-10 || srcc_gap > 1.0e-10
        {
            return Err(format!(
                "family {family}: learned projector equivalence tolerance exceeded",
            ));
        }
    }

    Ok(())
}
