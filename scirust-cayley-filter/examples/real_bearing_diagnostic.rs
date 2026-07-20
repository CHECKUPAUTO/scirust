use scirust_cayley_filter::{
    CayleyProjector, Sedenion, SoftCayleyFilter, squared_norm, zero_divisor_two_term_directions,
};

const BLOCK_SIZE: usize = 16;
const FEATURE_COUNT: usize = 6;

const TRAIN_SAMPLES: usize = 1_536;
const DEV_SAMPLES: usize = 1_024;

const TOP_K: usize = 24;
const RELATIVE_SCALE: f64 = 5.0e-2;
const ANALYSIS_TOLERANCE: f64 = 1.0e-12;

const MIN_FAULT_ENERGY_RETENTION: f64 = 0.75;
const MIN_FAULT_PEAK_RETENTION: f64 = 0.75;
const NUMERICAL_FLOOR: f64 = 1.0e-30;

type BearingFixture = (Vec<f64>, Vec<f64>);
type FeatureVector = [f64; FEATURE_COUNT];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FilterKind {
    Soft,
    Hard,
}

impl FilterKind {
    const fn rank(self) -> u8 {
        match self
        {
            Self::Soft => 0,
            Self::Hard => 1,
        }
    }

    const fn name(self) -> &'static str {
        match self
        {
            Self::Soft => "soft",
            Self::Hard => "hard",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct DiagnosticEvaluation {
    separation: f64,
    healthy_energy_retention: f64,
    fault_energy_retention: f64,
    fault_peak_retention: f64,
}

#[derive(Clone, Debug)]
struct Candidate {
    multiplier: Sedenion,
    first_index: usize,
    second_index: usize,
    second_sign: i8,
    kernel_dimension: usize,
    kind: FilterKind,
    train: DiagnosticEvaluation,
    train_separation_ratio: f64,
    dev: Option<DiagnosticEvaluation>,
    dev_separation_ratio: Option<f64>,
}

enum Transform {
    Identity,
    Soft(SoftCayleyFilter),
    Hard(CayleyProjector),
}

impl Transform {
    fn apply(&self, input: &Sedenion) -> Sedenion {
        match self
        {
            Self::Identity => *input,
            Self::Soft(filter) => filter.apply(input),
            Self::Hard(filter) => filter.apply(input),
        }
    }
}

fn load_fixture() -> Result<BearingFixture, String> {
    let raw = include_str!("../../scirust-signal/tests/data/cwru_bearing.csv");

    let mut healthy = Vec::new();
    let mut fault = Vec::new();

    for (line_index, line) in raw.lines().enumerate()
    {
        if line.starts_with('#') || line.trim().is_empty()
        {
            continue;
        }

        let (healthy_text, fault_text) = line
            .split_once(',')
            .ok_or_else(|| format!("invalid CSV line {}", line_index + 1,))?;

        healthy.push(healthy_text.trim().parse::<f64>().map_err(|error| {
            format!("invalid healthy sample at line {}: {error}", line_index + 1,)
        })?);

        fault.push(fault_text.trim().parse::<f64>().map_err(|error| {
            format!("invalid fault sample at line {}: {error}", line_index + 1,)
        })?);
    }

    if healthy.len() != 4_096 || fault.len() != healthy.len()
    {
        return Err(format!(
            "unexpected fixture dimensions: healthy={}, fault={}",
            healthy.len(),
            fault.len(),
        ));
    }

    Ok((healthy, fault))
}

fn make_blocks(samples: &[f64]) -> Result<Vec<Sedenion>, String> {
    if samples.is_empty() || !samples.len().is_multiple_of(BLOCK_SIZE)
    {
        return Err("split must be non-empty and block aligned".into());
    }

    let (blocks, remainder) = samples.as_chunks::<BLOCK_SIZE>();

    if !remainder.is_empty()
    {
        return Err("unexpected incomplete block".into());
    }

    Ok(blocks.to_vec())
}

fn block_peak(block: &Sedenion) -> f64 {
    block.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

fn block_features(block: &Sedenion) -> FeatureVector {
    let count = BLOCK_SIZE as f64;

    let mean = block.iter().sum::<f64>() / count;

    let mean_abs = block.iter().map(|value| value.abs()).sum::<f64>() / count;

    let mean_square = squared_norm(block) / count;

    let rms = mean_square.sqrt();
    let peak = block_peak(block);

    let line_length = block
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .sum::<f64>()
        / (BLOCK_SIZE - 1) as f64;

    let variance = block
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / count;

    let fourth_moment = block
        .iter()
        .map(|value| {
            let centered = value - mean;
            let squared = centered * centered;
            squared * squared
        })
        .sum::<f64>()
        / count;

    let crest = peak / rms.max(NUMERICAL_FLOOR);

    let kurtosis = fourth_moment / (variance * variance + NUMERICAL_FLOOR);

    [
        (rms + NUMERICAL_FLOOR).ln(),
        (mean_abs + NUMERICAL_FLOOR).ln(),
        (peak + NUMERICAL_FLOOR).ln(),
        (line_length + NUMERICAL_FLOOR).ln(),
        (crest + NUMERICAL_FLOOR).ln(),
        (kurtosis + NUMERICAL_FLOOR).ln(),
    ]
}

fn feature_moments(features: &[FeatureVector]) -> Result<(FeatureVector, FeatureVector), String> {
    if features.is_empty()
    {
        return Err("feature collection is empty".into());
    }

    let count = features.len() as f64;

    let mean = core::array::from_fn(|coordinate| {
        features
            .iter()
            .map(|feature| feature[coordinate])
            .sum::<f64>()
            / count
    });

    let variance = core::array::from_fn(|coordinate| {
        features
            .iter()
            .map(|feature| {
                let delta = feature[coordinate] - mean[coordinate];

                delta * delta
            })
            .sum::<f64>()
            / count
    });

    Ok((mean, variance))
}

fn diagonal_fisher_separation(
    healthy: &[FeatureVector],
    fault: &[FeatureVector],
) -> Result<f64, String> {
    let (healthy_mean, healthy_variance) = feature_moments(healthy)?;

    let (fault_mean, fault_variance) = feature_moments(fault)?;

    Ok((0..FEATURE_COUNT)
        .map(|coordinate| {
            let delta = fault_mean[coordinate] - healthy_mean[coordinate];

            let pooled_variance =
                healthy_variance[coordinate] + fault_variance[coordinate] + NUMERICAL_FLOOR;

            delta * delta / pooled_variance
        })
        .sum())
}

fn evaluate(
    transform: &Transform,
    healthy: &[Sedenion],
    fault: &[Sedenion],
) -> Result<DiagnosticEvaluation, String> {
    if healthy.is_empty() || fault.is_empty()
    {
        return Err("diagnostic split is empty".into());
    }

    let mut healthy_features = Vec::with_capacity(healthy.len());

    let mut fault_features = Vec::with_capacity(fault.len());

    let mut healthy_input_energy = 0.0;
    let mut healthy_output_energy = 0.0;
    let mut fault_input_energy = 0.0;
    let mut fault_output_energy = 0.0;
    let mut fault_input_peak = 0.0;
    let mut fault_output_peak = 0.0;

    for block in healthy
    {
        let output = transform.apply(block);

        healthy_input_energy += squared_norm(block);
        healthy_output_energy += squared_norm(&output);
        healthy_features.push(block_features(&output));
    }

    for block in fault
    {
        let output = transform.apply(block);

        fault_input_energy += squared_norm(block);
        fault_output_energy += squared_norm(&output);
        fault_input_peak += block_peak(block);
        fault_output_peak += block_peak(&output);
        fault_features.push(block_features(&output));
    }

    Ok(DiagnosticEvaluation {
        separation: diagonal_fisher_separation(&healthy_features, &fault_features)?,
        healthy_energy_retention: healthy_output_energy / healthy_input_energy.max(NUMERICAL_FLOOR),
        fault_energy_retention: fault_output_energy / fault_input_energy.max(NUMERICAL_FLOOR),
        fault_peak_retention: fault_output_peak / fault_input_peak.max(NUMERICAL_FLOOR),
    })
}

fn make_transform(kind: FilterKind, multiplier: Sedenion) -> Result<Transform, String> {
    match kind
    {
        FilterKind::Soft => Ok(Transform::Soft(
            SoftCayleyFilter::new(multiplier, RELATIVE_SCALE).map_err(|error| error.to_string())?,
        )),
        FilterKind::Hard => Ok(Transform::Hard(
            CayleyProjector::new(multiplier, RELATIVE_SCALE).map_err(|error| error.to_string())?,
        )),
    }
}

fn gate(evaluation: DiagnosticEvaluation, separation_ratio: f64) -> bool {
    separation_ratio > 1.0
        && evaluation.fault_energy_retention >= MIN_FAULT_ENERGY_RETENTION
        && evaluation.fault_peak_retention >= MIN_FAULT_PEAK_RETENTION
}

fn main() -> Result<(), String> {
    let (healthy, fault) = load_fixture()?;
    let dev_end = TRAIN_SAMPLES + DEV_SAMPLES;

    let healthy_train = make_blocks(&healthy[..TRAIN_SAMPLES])?;

    let fault_train = make_blocks(&fault[..TRAIN_SAMPLES])?;

    let healthy_dev = make_blocks(&healthy[TRAIN_SAMPLES..dev_end])?;

    let fault_dev = make_blocks(&fault[TRAIN_SAMPLES..dev_end])?;

    let healthy_test = make_blocks(&healthy[dev_end..])?;

    let fault_test = make_blocks(&fault[dev_end..])?;

    let identity = Transform::Identity;

    let train_identity = evaluate(&identity, &healthy_train, &fault_train)?;

    let dev_identity = evaluate(&identity, &healthy_dev, &fault_dev)?;

    let test_identity = evaluate(&identity, &healthy_test, &fault_test)?;

    let directions = zero_divisor_two_term_directions(ANALYSIS_TOLERANCE)?;

    let mut candidates = Vec::new();

    for direction in directions
    {
        for kind in [FilterKind::Soft, FilterKind::Hard]
        {
            let transform = make_transform(kind, direction.multiplier)?;

            let train = evaluate(&transform, &healthy_train, &fault_train)?;

            candidates.push(Candidate {
                multiplier: direction.multiplier,
                first_index: direction.first_index,
                second_index: direction.second_index,
                second_sign: direction.second_sign,
                kernel_dimension: direction.kernel_dimension,
                kind,
                train,
                train_separation_ratio: train.separation
                    / train_identity.separation.max(NUMERICAL_FLOOR),
                dev: None,
                dev_separation_ratio: None,
            });
        }
    }

    candidates.sort_by(|left, right| {
        right
            .train_separation_ratio
            .total_cmp(&left.train_separation_ratio)
            .then_with(|| {
                right
                    .train
                    .fault_energy_retention
                    .total_cmp(&left.train.fault_energy_retention)
            })
            .then_with(|| left.first_index.cmp(&right.first_index))
            .then_with(|| left.second_index.cmp(&right.second_index))
            .then_with(|| left.second_sign.cmp(&right.second_sign))
            .then_with(|| left.kind.rank().cmp(&right.kind.rank()))
    });

    candidates.truncate(TOP_K.min(candidates.len()));

    for candidate in &mut candidates
    {
        let transform = make_transform(candidate.kind, candidate.multiplier)?;

        let dev = evaluate(&transform, &healthy_dev, &fault_dev)?;

        candidate.dev_separation_ratio =
            Some(dev.separation / dev_identity.separation.max(NUMERICAL_FLOOR));

        candidate.dev = Some(dev);
    }

    candidates.sort_by(|left, right| {
        let left_ratio = left.dev_separation_ratio.unwrap_or(0.0);

        let right_ratio = right.dev_separation_ratio.unwrap_or(0.0);

        right_ratio
            .total_cmp(&left_ratio)
            .then_with(|| {
                right
                    .train_separation_ratio
                    .total_cmp(&left.train_separation_ratio)
            })
            .then_with(|| left.first_index.cmp(&right.first_index))
            .then_with(|| left.second_index.cmp(&right.second_index))
            .then_with(|| left.second_sign.cmp(&right.second_sign))
            .then_with(|| left.kind.rank().cmp(&right.kind.rank()))
    });

    let selected = candidates
        .first()
        .ok_or_else(|| "no diagnostic candidate was produced".to_string())?;

    let selected_dev = selected
        .dev
        .ok_or_else(|| "selected candidate has no development score".to_string())?;

    let selected_dev_ratio = selected
        .dev_separation_ratio
        .ok_or_else(|| "selected candidate has no development ratio".to_string())?;

    let uses_cayley = gate(selected_dev, selected_dev_ratio);

    let candidate_transform = make_transform(selected.kind, selected.multiplier)?;

    let candidate_test = evaluate(&candidate_transform, &healthy_test, &fault_test)?;

    let candidate_test_ratio =
        candidate_test.separation / test_identity.separation.max(NUMERICAL_FLOOR);

    let safe_test = if uses_cayley
    {
        candidate_test
    }
    else
    {
        test_identity
    };

    let safe_test_ratio = safe_test.separation / test_identity.separation.max(NUMERICAL_FLOOR);

    println!(
        "fixture_samples={},train_samples={},dev_samples={},test_samples={},block_size={}",
        healthy.len(),
        TRAIN_SAMPLES,
        DEV_SAMPLES,
        healthy.len() - dev_end,
        BLOCK_SIZE,
    );

    println!(
        "identity_train_separation={},identity_dev_separation={},identity_test_separation={}",
        train_identity.separation, dev_identity.separation, test_identity.separation,
    );

    println!(
        "selected=e{}{}e{},kind={},kernel_dimension={},train_separation_ratio={},dev_separation_ratio={},dev_healthy_energy_retention={},dev_fault_energy_retention={},dev_fault_peak_retention={},decision={}",
        selected.first_index,
        if selected.second_sign > 0 { "+" } else { "-" },
        selected.second_index,
        selected.kind.name(),
        selected.kernel_dimension,
        selected.train_separation_ratio,
        selected_dev_ratio,
        selected_dev.healthy_energy_retention,
        selected_dev.fault_energy_retention,
        selected_dev.fault_peak_retention,
        if uses_cayley { "Cayley" } else { "Identity" },
    );

    println!(
        "result,separation,separation_ratio,healthy_energy_retention,fault_energy_retention,fault_peak_retention"
    );

    println!(
        "identity_test,{},{},{},{},{}",
        test_identity.separation,
        1.0,
        test_identity.healthy_energy_retention,
        test_identity.fault_energy_retention,
        test_identity.fault_peak_retention,
    );

    println!(
        "candidate_test,{},{},{},{},{}",
        candidate_test.separation,
        candidate_test_ratio,
        candidate_test.healthy_energy_retention,
        candidate_test.fault_energy_retention,
        candidate_test.fault_peak_retention,
    );

    println!(
        "safe_output_test,{},{},{},{},{}",
        safe_test.separation,
        safe_test_ratio,
        safe_test.healthy_energy_retention,
        safe_test.fault_energy_retention,
        safe_test.fault_peak_retention,
    );

    Ok(())
}
