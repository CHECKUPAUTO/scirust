use scirust_tdi::{
    Action, ExactRatio, State, TableSystem, analyze_branching_recovery, explore,
    uniform_branching_path_entropy_bits,
};

const TRAIN_WIDTH_3: u8 = 3;
const TRAIN_WIDTH_4: u8 = 4;
const OOD_WIDTH: u8 = 5;

const TRAIN_SYSTEMS_PER_WIDTH: usize = 10_000;
const HOLDOUT_SYSTEMS_PER_WIDTH: usize = 5_000;
const OOD_SYSTEMS: usize = 5_000;

const TRAIN_WIDTH_3_SEED_OFFSET: u64 = 30_000_000;
const HOLDOUT_WIDTH_3_SEED_OFFSET: u64 = 31_000_000;
const TRAIN_WIDTH_4_SEED_OFFSET: u64 = 40_000_000;
const HOLDOUT_WIDTH_4_SEED_OFFSET: u64 = 41_000_000;
const OOD_WIDTH_5_SEED_OFFSET: u64 = 50_000_000;

const OBSERVATION_HORIZON: usize = 2;
const OUTCOME_HORIZON: usize = 6;

const BASELINE_FEATURE_COUNT: usize = 13;
const TDI_FEATURE_COUNT: usize = 3;

const RIDGE_LAMBDA: f64 = 1.0;
const BOOTSTRAP_REPLICATES: usize = 2_000;
const BOOTSTRAP_SEED: u64 = 0x5444_4934_4745_4F4D;

#[derive(Clone, Debug)]
struct Record {
    baseline: [f64; BASELINE_FEATURE_COUNT],
    tdi: [f64; TDI_FEATURE_COUNT],
    overlap: f64,
    recovered: bool,
    conditional_u: Option<f64>,
}

#[derive(Clone, Debug)]
struct RidgeModel {
    means: Vec<f64>,
    scales: Vec<f64>,
    coefficients: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Metrics {
    mse: f64,
    mae: f64,
    r_squared: f64,
    spearman: f64,
    bias: f64,
    observed_mean: f64,
    predicted_mean: f64,
    calibration_intercept: f64,
    calibration_slope: f64,
    zero_fraction: f64,
    one_fraction: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ConfidenceInterval {
    lower: f64,
    median: f64,
    upper: f64,
}

#[derive(Clone, Copy, Debug)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        splitmix64(self.state)
    }

    fn index(&mut self, upper: usize) -> usize {
        (self.next_u64() % upper as u64) as usize
    }
}

impl RidgeModel {
    fn predict_linear(&self, features: &[f64]) -> f64 {
        assert_eq!(features.len(), self.means.len());
        assert_eq!(features.len(), self.scales.len());
        assert_eq!(self.coefficients.len(), features.len() + 1);

        features
            .iter()
            .zip(&self.means)
            .zip(&self.scales)
            .zip(self.coefficients.iter().skip(1))
            .fold(
                self.coefficients[0],
                |accumulator, (((value, mean), scale), coefficient)| {
                    accumulator + coefficient * ((value - mean) / scale)
                },
            )
    }

    fn predict_probability(&self, features: &[f64]) -> f64 {
        self.predict_linear(features).clamp(0.0, 1.0)
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);

    let mut mixed = value;
    mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

    mixed ^ (mixed >> 31)
}

fn state_count(width: u8) -> Result<usize, String> {
    1_usize
        .checked_shl(u32::from(width))
        .ok_or_else(|| format!("state-count overflow for width {width}"))
}

fn nonempty_successor_set_count(width: u8) -> Result<u64, String> {
    let count = state_count(width)?;

    1_u64
        .checked_shl(count as u32)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| format!("successor-mask space cannot be represented for width {width}"))
}

fn generate_successor_masks(width: u8, seed: u64) -> Result<Vec<u64>, String> {
    let states = state_count(width)?;
    let mask_count = nonempty_successor_set_count(width)?;

    let mut masks = vec![0_u64; states];
    let mut generator = seed;

    for mask in &mut masks
    {
        generator = splitmix64(generator);
        *mask = generator % mask_count + 1;
    }

    Ok(masks)
}

fn build_system(width: u8, masks: &[u64]) -> Result<TableSystem, String> {
    let states = state_count(width)?;

    if masks.len() != states
    {
        return Err(format!(
            "expected {states} successor masks, received {}",
            masks.len()
        ));
    }

    let mut system = TableSystem::new(width)
        .map_err(|error| format!("cannot create branching system: {error:?}"))?;

    for (source_bits, &mask) in masks.iter().enumerate()
    {
        let source = State::new(source_bits as u64, width).map_err(|error| error.to_string())?;

        let successors = (0..states)
            .filter(|target| mask & (1_u64 << target) != 0)
            .map(|target| State::new(target as u64, width).map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()?;

        system
            .insert(source, Action::Noop, successors)
            .map_err(|error| {
                format!(
                    "cannot insert branching transition for state \
                     {source_bits}: {error:?}"
                )
            })?;
    }

    Ok(system)
}

fn entropy_profile(
    system: &TableSystem,
    initial: State,
) -> Result<[f64; OBSERVATION_HORIZON], String> {
    let mut profile = [0.0_f64; OBSERVATION_HORIZON];

    for depth in 1..=OBSERVATION_HORIZON
    {
        profile[depth - 1] =
            uniform_branching_path_entropy_bits(system, initial, Action::Noop, depth)
                .map_err(|error| format!("branching entropy failed at depth {depth}: {error:?}"))?;
    }

    Ok(profile)
}

fn topology_profile(
    system: &TableSystem,
    initial: State,
) -> Result<([f64; OBSERVATION_HORIZON], [f64; OBSERVATION_HORIZON]), String> {
    let actions = [Action::Noop; OBSERVATION_HORIZON];

    let report = explore(system, initial, &actions)
        .map_err(|error| format!("branching exploration failed: {error:?}"))?;

    let mut reachable = [0.0_f64; OBSERVATION_HORIZON];
    let mut paths = [0.0_f64; OBSERVATION_HORIZON];

    for depth in 1..=OBSERVATION_HORIZON
    {
        reachable[depth - 1] = report
            .reachable_count(depth)
            .ok_or_else(|| format!("missing reachable layer {depth}"))?
            as f64;

        paths[depth - 1] = report
            .path_count(depth)
            .ok_or_else(|| format!("missing path-count layer {depth}"))?
            as f64;
    }

    Ok((reachable, paths))
}

fn ratio_value(ratio: &ExactRatio) -> f64 {
    ratio.as_f64()
}

fn biguint_log2_from_u64_digits(digits: &[u64]) -> Result<f64, String> {
    let top = digits
        .last()
        .copied()
        .ok_or_else(|| "cannot calculate log2 of zero".to_owned())?;

    if top == 0
    {
        return Err("invalid leading zero BigUint limb".to_owned());
    }

    let top_bits = 64_usize - top.leading_zeros() as usize;
    let bit_length = (digits.len() - 1) * 64 + top_bits;

    let combined = if digits.len() >= 2
    {
        (u128::from(top) << 64) | u128::from(digits[digits.len() - 2])
    }
    else
    {
        u128::from(top)
    };

    let combined_bits = if digits.len() >= 2
    {
        top_bits + 64
    }
    else
    {
        top_bits
    };

    let shift = combined_bits.saturating_sub(53);
    let significant = (combined >> shift) as u64;
    let significant_bits = combined_bits - shift;

    let mantissa = significant as f64 / 2.0_f64.powi((significant_bits - 1) as i32);

    if !mantissa.is_finite() || !(1.0..2.0).contains(&mantissa)
    {
        return Err("invalid normalized BigUint mantissa".to_owned());
    }

    let logarithm = (bit_length - 1) as f64 + mantissa.log2();

    if !logarithm.is_finite()
    {
        return Err("non-finite BigUint logarithm".to_owned());
    }

    Ok(logarithm)
}

fn exact_overlap_deficit_u(ratio: &ExactRatio) -> Result<f64, String> {
    if ratio.numerator() >= ratio.denominator()
    {
        return Err("conditional overlap must be strictly below one".to_owned());
    }

    let deficit_numerator = ratio.denominator() - ratio.numerator();

    let numerator_log2 = biguint_log2_from_u64_digits(&deficit_numerator.to_u64_digits())?;

    let denominator_log2 = biguint_log2_from_u64_digits(&ratio.denominator().to_u64_digits())?;

    let transformed = denominator_log2 - numerator_log2;

    if !transformed.is_finite() || transformed < 0.0
    {
        return Err(format!(
            "invalid conditional target geometry: {transformed}"
        ));
    }

    Ok(transformed)
}

fn normalized_entropy(entropy_bits: f64, width: u8) -> Result<f64, String> {
    let states = state_count(width)? as f64;
    let denominator = states.ln();

    if !denominator.is_finite() || denominator <= 0.0
    {
        return Err(format!("invalid entropy normalizer for width {width}"));
    }

    Ok(entropy_bits * std::f64::consts::LN_2 / denominator)
}

fn normalized_reachable(reachable: f64, width: u8) -> Result<f64, String> {
    let states = state_count(width)? as f64;
    let normalized = reachable / states;

    if !normalized.is_finite()
    {
        return Err(format!("non-finite reachable fraction for width {width}"));
    }

    Ok(normalized)
}

fn transformed_path_count(path_count: f64) -> Result<f64, String> {
    let transformed = path_count.ln_1p();

    if !transformed.is_finite()
    {
        return Err("non-finite transformed path count".to_owned());
    }

    Ok(transformed)
}

fn analyze_seed(width: u8, seed: u64) -> Result<Option<Record>, String> {
    let masks = generate_successor_masks(width, seed)?;
    let system = build_system(width, &masks)?;

    let reference = State::new(0, width).map_err(|error| error.to_string())?;

    let perturbation = Action::Flip { node: width - 1 };

    let perturbed = perturbation
        .apply(reference)
        .map_err(|error| error.to_string())?;

    let reference_entropy = entropy_profile(&system, reference)?;
    let perturbed_entropy = entropy_profile(&system, perturbed)?;

    let (reference_reachable, reference_paths) = topology_profile(&system, reference)?;

    let (perturbed_reachable, perturbed_paths) = topology_profile(&system, perturbed)?;

    let observation = analyze_branching_recovery(
        &system,
        reference,
        perturbation,
        Action::Noop,
        OBSERVATION_HORIZON,
    )
    .map_err(|error| {
        format!(
            "observation recovery analysis failed for width {width}, \
             seed {seed}: {error:?}"
        )
    })?;

    if observation.fully_recovered()
    {
        return Ok(None);
    }

    let outcome = analyze_branching_recovery(
        &system,
        reference,
        perturbation,
        Action::Noop,
        OUTCOME_HORIZON,
    )
    .map_err(|error| {
        format!(
            "outcome recovery analysis failed for width {width}, \
             seed {seed}: {error:?}"
        )
    })?;

    let overlap_profile = observation.overlap_profile();

    if overlap_profile.len() != OBSERVATION_HORIZON
    {
        return Err(format!(
            "expected {OBSERVATION_HORIZON} overlap values, received {}",
            overlap_profile.len()
        ));
    }

    let first_overlap = ratio_value(&overlap_profile[0]);
    let second_overlap = ratio_value(&overlap_profile[1]);

    let final_overlap_ratio = outcome
        .final_overlap()
        .ok_or_else(|| "outcome horizon produced no overlap".to_owned())?;

    let recovered = outcome.fully_recovered();
    let final_overlap = final_overlap_ratio.as_f64();

    if !final_overlap.is_finite() || !(0.0..=1.0).contains(&final_overlap)
    {
        return Err(format!(
            "invalid final overlap for width {width}, seed {seed}: {final_overlap}"
        ));
    }

    let conditional_u = if recovered
    {
        None
    }
    else
    {
        Some(
            exact_overlap_deficit_u(&final_overlap_ratio).map_err(|error| {
                format!(
                    "cannot calculate conditional target for width \
                         {width}, seed {seed}: {error}"
                )
            })?,
        )
    };

    let baseline = [
        normalized_entropy(reference_entropy[0], width)?,
        normalized_entropy(reference_entropy[1], width)?,
        normalized_entropy(perturbed_entropy[0], width)?,
        normalized_entropy(perturbed_entropy[1], width)?,
        normalized_reachable(reference_reachable[0], width)?,
        normalized_reachable(reference_reachable[1], width)?,
        transformed_path_count(reference_paths[0])?,
        transformed_path_count(reference_paths[1])?,
        normalized_reachable(perturbed_reachable[0], width)?,
        normalized_reachable(perturbed_reachable[1], width)?,
        transformed_path_count(perturbed_paths[0])?,
        transformed_path_count(perturbed_paths[1])?,
        f64::from(width),
    ];

    let tdi = [
        first_overlap,
        second_overlap,
        second_overlap - first_overlap,
    ];

    Ok(Some(Record {
        baseline,
        tdi,
        overlap: final_overlap,
        recovered,
        conditional_u,
    }))
}

fn generate_records(
    width: u8,
    start_seed: u64,
    count: usize,
) -> Result<(Vec<Record>, u64, usize), String> {
    let mut records = Vec::with_capacity(count);
    let mut seed = start_seed;
    let mut excluded = 0_usize;

    while records.len() < count
    {
        match analyze_seed(width, seed)?
        {
            Some(record) => records.push(record),
            None => excluded += 1,
        }

        seed = seed
            .checked_add(1)
            .ok_or_else(|| "seed range overflow".to_owned())?;
    }

    Ok((records, seed, excluded))
}

fn baseline_features(record: &Record) -> Vec<f64> {
    record.baseline.to_vec()
}

fn challenger_features(record: &Record) -> Vec<f64> {
    record.baseline.iter().chain(&record.tdi).copied().collect()
}

fn feature_matrix<F>(records: &[Record], feature_fn: F) -> Vec<Vec<f64>>
where
    F: Fn(&Record) -> Vec<f64>,
{
    records.iter().map(feature_fn).collect()
}

fn fit_ridge(features: &[Vec<f64>], targets: &[f64]) -> Result<RidgeModel, String> {
    if features.is_empty()
    {
        return Err("cannot fit ridge regression on an empty dataset".to_owned());
    }

    if features.len() != targets.len()
    {
        return Err(format!(
            "feature/target length mismatch: {} versus {}",
            features.len(),
            targets.len()
        ));
    }

    let feature_count = features[0].len();

    if feature_count == 0
    {
        return Err("ridge regression requires at least one feature".to_owned());
    }

    if features.iter().any(|row| row.len() != feature_count)
    {
        return Err("inconsistent feature-vector lengths".to_owned());
    }

    let sample_count = features.len() as f64;
    let mut means = vec![0.0_f64; feature_count];

    for row in features
    {
        for (mean, value) in means.iter_mut().zip(row)
        {
            *mean += value;
        }
    }

    for mean in &mut means
    {
        *mean /= sample_count;
    }

    let mut scales = vec![0.0_f64; feature_count];

    for row in features
    {
        for ((scale, value), mean) in scales.iter_mut().zip(row).zip(&means)
        {
            let difference = value - mean;
            *scale += difference * difference;
        }
    }

    for scale in &mut scales
    {
        *scale = (*scale / sample_count).sqrt();

        if !scale.is_finite() || *scale <= 1.0e-12
        {
            *scale = 1.0;
        }
    }

    let dimension = feature_count + 1;
    let mut normal = vec![vec![0.0_f64; dimension]; dimension];
    let mut right_hand_side = vec![0.0_f64; dimension];

    for (row, &target) in features.iter().zip(targets)
    {
        let mut standardized = Vec::with_capacity(dimension);
        standardized.push(1.0);

        standardized.extend(
            row.iter()
                .zip(&means)
                .zip(&scales)
                .map(|((value, mean), scale)| (value - mean) / scale),
        );

        for (left_index, &left_value) in standardized.iter().enumerate()
        {
            right_hand_side[left_index] += left_value * target;

            for (right_index, &right_value) in standardized.iter().enumerate()
            {
                normal[left_index][right_index] += left_value * right_value;
            }
        }
    }

    for (index, row) in normal.iter_mut().enumerate().skip(1)
    {
        row[index] += RIDGE_LAMBDA;
    }

    let coefficients = solve_linear_system(normal, right_hand_side)?;

    Ok(RidgeModel {
        means,
        scales,
        coefficients,
    })
}

fn solve_linear_system(
    mut matrix: Vec<Vec<f64>>,
    mut right_hand_side: Vec<f64>,
) -> Result<Vec<f64>, String> {
    let dimension = matrix.len();

    if dimension == 0 || right_hand_side.len() != dimension
    {
        return Err("invalid linear-system dimensions".to_owned());
    }

    if matrix.iter().any(|row| row.len() != dimension)
    {
        return Err("linear-system matrix is not square".to_owned());
    }

    for column in 0..dimension
    {
        let pivot_row = (column..dimension)
            .max_by(|&left, &right| {
                matrix[left][column]
                    .abs()
                    .total_cmp(&matrix[right][column].abs())
            })
            .ok_or_else(|| "missing pivot row".to_owned())?;

        let pivot_value = matrix[pivot_row][column];

        if !pivot_value.is_finite() || pivot_value.abs() <= 1.0e-12
        {
            return Err(format!(
                "singular or ill-conditioned normal matrix at column {column}"
            ));
        }

        if pivot_row != column
        {
            matrix.swap(pivot_row, column);
            right_hand_side.swap(pivot_row, column);
        }

        let pivot_values = matrix[column].clone();
        let pivot_denominator = pivot_values[column];
        let pivot_right_hand_side = right_hand_side[column];

        for (row_index, row_values) in matrix.iter_mut().enumerate().skip(column + 1)
        {
            let factor = row_values[column] / pivot_denominator;

            row_values[column] = 0.0;

            for (value, pivot_value) in row_values.iter_mut().zip(&pivot_values).skip(column + 1)
            {
                *value -= factor * pivot_value;
            }

            right_hand_side[row_index] -= factor * pivot_right_hand_side;
        }
    }

    let mut solution = vec![0.0_f64; dimension];

    for row in (0..dimension).rev()
    {
        let trailing_sum = matrix[row]
            .iter()
            .enumerate()
            .skip(row + 1)
            .map(|(column, coefficient)| coefficient * solution[column])
            .sum::<f64>();

        solution[row] = (right_hand_side[row] - trailing_sum) / matrix[row][row];

        if !solution[row].is_finite()
        {
            return Err(format!("non-finite linear-system solution at row {row}"));
        }
    }

    Ok(solution)
}

fn predictions(model: &RidgeModel, features: &[Vec<f64>]) -> Vec<f64> {
    features
        .iter()
        .map(|row| model.predict_probability(row))
        .collect()
}

fn calculate_metrics(targets: &[f64], predicted: &[f64]) -> Metrics {
    assert_eq!(targets.len(), predicted.len());
    assert!(!targets.is_empty());

    let sample_count = targets.len() as f64;
    let observed_mean = targets.iter().sum::<f64>() / sample_count;
    let predicted_mean = predicted.iter().sum::<f64>() / sample_count;

    let mut squared_error = 0.0_f64;
    let mut absolute_error = 0.0_f64;
    let mut total_variance = 0.0_f64;
    let mut calibration_covariance = 0.0_f64;
    let mut prediction_variance = 0.0_f64;
    let mut zero_count = 0_usize;
    let mut one_count = 0_usize;

    for (&target, &prediction) in targets.iter().zip(predicted)
    {
        let residual = target - prediction;
        squared_error += residual * residual;
        absolute_error += residual.abs();

        let centered_target = target - observed_mean;
        let centered_prediction = prediction - predicted_mean;

        total_variance += centered_target * centered_target;
        calibration_covariance += centered_prediction * centered_target;
        prediction_variance += centered_prediction * centered_prediction;

        if prediction == 0.0
        {
            zero_count += 1;
        }

        if prediction == 1.0
        {
            one_count += 1;
        }
    }

    let r_squared = if total_variance <= 1.0e-15
    {
        0.0
    }
    else
    {
        1.0 - squared_error / total_variance
    };

    let calibration_slope = if prediction_variance <= 1.0e-15
    {
        0.0
    }
    else
    {
        calibration_covariance / prediction_variance
    };

    let calibration_intercept = observed_mean - calibration_slope * predicted_mean;

    Metrics {
        mse: squared_error / sample_count,
        mae: absolute_error / sample_count,
        r_squared,
        spearman: spearman_correlation(targets, predicted),
        bias: predicted_mean - observed_mean,
        observed_mean,
        predicted_mean,
        calibration_intercept,
        calibration_slope,
        zero_fraction: zero_count as f64 / sample_count,
        one_fraction: one_count as f64 / sample_count,
    }
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut indices = (0..values.len()).collect::<Vec<_>>();

    indices.sort_by(|&left, &right| {
        values[left]
            .total_cmp(&values[right])
            .then_with(|| left.cmp(&right))
    });

    let mut ranks = vec![0.0_f64; values.len()];
    let mut start = 0_usize;

    while start < indices.len()
    {
        let mut end = start + 1;

        while end < indices.len()
            && values[indices[start]].total_cmp(&values[indices[end]]) == std::cmp::Ordering::Equal
        {
            end += 1;
        }

        let average_rank = (start + 1 + end) as f64 / 2.0;

        for &index in &indices[start..end]
        {
            ranks[index] = average_rank;
        }

        start = end;
    }

    ranks
}

fn pearson_correlation(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());

    let count = left.len() as f64;
    let left_mean = left.iter().sum::<f64>() / count;
    let right_mean = right.iter().sum::<f64>() / count;

    let mut covariance = 0.0_f64;
    let mut left_variance = 0.0_f64;
    let mut right_variance = 0.0_f64;

    for (&left_value, &right_value) in left.iter().zip(right)
    {
        let centered_left = left_value - left_mean;
        let centered_right = right_value - right_mean;

        covariance += centered_left * centered_right;
        left_variance += centered_left * centered_left;
        right_variance += centered_right * centered_right;
    }

    let denominator = (left_variance * right_variance).sqrt();

    if denominator <= 1.0e-15
    {
        0.0
    }
    else
    {
        covariance / denominator
    }
}

fn spearman_correlation(left: &[f64], right: &[f64]) -> f64 {
    let left_ranks = average_ranks(left);
    let right_ranks = average_ranks(right);

    pearson_correlation(&left_ranks, &right_ranks)
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let position = quantile * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;

    if lower == upper
    {
        sorted[lower]
    }
    else
    {
        let weight = position - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

fn confidence_interval(mut values: Vec<f64>) -> ConfidenceInterval {
    values.sort_by(f64::total_cmp);

    ConfidenceInterval {
        lower: percentile(&values, 0.025),
        median: percentile(&values, 0.500),
        upper: percentile(&values, 0.975),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TargetScaler {
    mean: f64,
    scale: f64,
}

#[derive(Clone, Debug)]
struct TwoPartModel {
    binary: RidgeModel,
    conditional: RidgeModel,
    scaler: TargetScaler,
}

#[derive(Clone, Copy, Debug)]
struct TwoPartPrediction {
    probability: f64,
    conditional_standardized: f64,
    reconstructed_overlap: f64,
}

#[derive(Clone, Copy, Debug)]
struct BinaryMetrics {
    brier: f64,
    observed_rate: f64,
    predicted_mean: f64,
    calibration_intercept: f64,
    calibration_slope: f64,
    zero_fraction: f64,
    one_fraction: f64,
}

#[derive(Clone, Copy, Debug)]
struct CompositeMetrics {
    binary: BinaryMetrics,
    conditional: Metrics,
    reconstruction: Metrics,
    loss: f64,
}

#[derive(Clone, Debug)]
struct DatasetEvaluation {
    baseline: CompositeMetrics,
    challenger: CompositeMetrics,
    baseline_predictions: Vec<TwoPartPrediction>,
    challenger_predictions: Vec<TwoPartPrediction>,
}

#[derive(Clone, Copy, Debug)]
struct BootstrapIntervals {
    composite: ConfidenceInterval,
    brier: ConfidenceInterval,
    conditional_mse: ConfidenceInterval,
    reconstruction_mse: ConfidenceInterval,
    reconstruction_mae: ConfidenceInterval,
}

impl TargetScaler {
    fn fit(records: &[Record]) -> Result<Self, String> {
        let values = records
            .iter()
            .filter_map(|record| record.conditional_u)
            .collect::<Vec<_>>();

        if values.is_empty()
        {
            return Err("training population contains no non-recovered systems".to_owned());
        }

        let count = values.len() as f64;
        let mean = values.iter().sum::<f64>() / count;

        let variance = values
            .iter()
            .map(|value| {
                let difference = value - mean;
                difference * difference
            })
            .sum::<f64>()
            / count;

        let scale = variance.sqrt();

        if !mean.is_finite() || !scale.is_finite() || scale <= 1.0e-12
        {
            return Err("conditional target has invalid training geometry".to_owned());
        }

        Ok(Self { mean, scale })
    }

    fn standardize(self, value: f64) -> f64 {
        (value - self.mean) / self.scale
    }

    fn unstandardize(self, value: f64) -> f64 {
        self.mean + self.scale * value
    }
}

fn overlap_targets(records: &[Record]) -> Vec<f64> {
    records.iter().map(|record| record.overlap).collect()
}

fn binary_targets(records: &[Record]) -> Vec<f64> {
    records
        .iter()
        .map(|record| if record.recovered { 1.0 } else { 0.0 })
        .collect()
}

fn fit_two_part_model<F>(
    records: &[Record],
    feature_fn: F,
    scaler: TargetScaler,
) -> Result<TwoPartModel, String>
where
    F: Fn(&Record) -> Vec<f64> + Copy,
{
    let all_features = feature_matrix(records, feature_fn);
    let binary = fit_ridge(&all_features, &binary_targets(records))?;

    let mut conditional_features = Vec::new();
    let mut conditional_targets = Vec::new();

    for record in records
    {
        if let Some(value) = record.conditional_u
        {
            conditional_features.push(feature_fn(record));
            conditional_targets.push(scaler.standardize(value));
        }
    }

    if conditional_features.is_empty()
    {
        return Err("cannot fit conditional head without non-recovered systems".to_owned());
    }

    let conditional = fit_ridge(&conditional_features, &conditional_targets)?;

    Ok(TwoPartModel {
        binary,
        conditional,
        scaler,
    })
}

fn reconstruct_overlap(probability: f64, conditional_u: f64) -> f64 {
    if probability >= 1.0
    {
        return 1.0;
    }

    let deficit = 2.0_f64.powf(-conditional_u);

    if !deficit.is_finite()
    {
        return 0.0;
    }

    (1.0 - (1.0 - probability) * deficit).clamp(0.0, 1.0)
}

fn predict_two_part(model: &TwoPartModel, features: &[f64]) -> Result<TwoPartPrediction, String> {
    let probability = model.binary.predict_probability(features);
    let conditional_standardized = model.conditional.predict_linear(features);

    if !probability.is_finite() || !conditional_standardized.is_finite()
    {
        return Err("non-finite two-part prediction".to_owned());
    }

    let conditional_u = model.scaler.unstandardize(conditional_standardized);

    if !conditional_u.is_finite()
    {
        return Err("non-finite unstandardized conditional prediction".to_owned());
    }

    let reconstructed_overlap = reconstruct_overlap(probability, conditional_u);

    if !reconstructed_overlap.is_finite()
    {
        return Err("non-finite reconstructed overlap".to_owned());
    }

    Ok(TwoPartPrediction {
        probability,
        conditional_standardized,
        reconstructed_overlap,
    })
}

fn predict_dataset<F>(
    records: &[Record],
    feature_fn: F,
    model: &TwoPartModel,
) -> Result<Vec<TwoPartPrediction>, String>
where
    F: Fn(&Record) -> Vec<f64>,
{
    records
        .iter()
        .map(|record| predict_two_part(model, &feature_fn(record)))
        .collect()
}

fn binary_metrics(records: &[Record], predictions: &[TwoPartPrediction]) -> BinaryMetrics {
    let targets = binary_targets(records);
    let predicted = predictions
        .iter()
        .map(|prediction| prediction.probability)
        .collect::<Vec<_>>();

    let metrics = calculate_metrics(&targets, &predicted);

    BinaryMetrics {
        brier: metrics.mse,
        observed_rate: metrics.observed_mean,
        predicted_mean: metrics.predicted_mean,
        calibration_intercept: metrics.calibration_intercept,
        calibration_slope: metrics.calibration_slope,
        zero_fraction: metrics.zero_fraction,
        one_fraction: metrics.one_fraction,
    }
}

fn conditional_metrics(
    records: &[Record],
    predictions: &[TwoPartPrediction],
    scaler: TargetScaler,
) -> Result<Metrics, String> {
    let mut targets = Vec::new();
    let mut predicted = Vec::new();

    for (record, prediction) in records.iter().zip(predictions)
    {
        if let Some(value) = record.conditional_u
        {
            targets.push(scaler.standardize(value));
            predicted.push(prediction.conditional_standardized);
        }
    }

    if targets.is_empty()
    {
        return Err("evaluation population contains no non-recovered systems".to_owned());
    }

    Ok(calculate_metrics(&targets, &predicted))
}

fn reconstruction_metrics(records: &[Record], predictions: &[TwoPartPrediction]) -> Metrics {
    let targets = overlap_targets(records);
    let predicted = predictions
        .iter()
        .map(|prediction| prediction.reconstructed_overlap)
        .collect::<Vec<_>>();

    calculate_metrics(&targets, &predicted)
}

fn composite_metrics(
    records: &[Record],
    predictions: &[TwoPartPrediction],
    scaler: TargetScaler,
) -> Result<CompositeMetrics, String> {
    let binary = binary_metrics(records, predictions);
    let conditional = conditional_metrics(records, predictions, scaler)?;
    let reconstruction = reconstruction_metrics(records, predictions);

    let loss = 0.5 * binary.brier + 0.5 * conditional.mse;

    Ok(CompositeMetrics {
        binary,
        conditional,
        reconstruction,
        loss,
    })
}

fn evaluate_dataset(
    label: &str,
    records: &[Record],
    baseline_model: &TwoPartModel,
    challenger_model: &TwoPartModel,
) -> Result<DatasetEvaluation, String> {
    if baseline_model.scaler != challenger_model.scaler
    {
        return Err("baseline and challenger target scalers differ".to_owned());
    }

    let baseline_predictions = predict_dataset(records, baseline_features, baseline_model)?;

    let challenger_predictions = predict_dataset(records, challenger_features, challenger_model)?;

    let baseline = composite_metrics(records, &baseline_predictions, baseline_model.scaler)?;

    let challenger = composite_metrics(records, &challenger_predictions, challenger_model.scaler)?;

    println!();
    println!("{label}");
    print_composite_metrics("BASELINE APPARIÉE", baseline);
    println!();
    print_composite_metrics("BASELINE + TDI-4", challenger);

    let improvement = baseline.loss - challenger.loss;
    let relative = relative_reduction(baseline.loss, challenger.loss);

    println!();
    println!("amélioration perte composite : {improvement:.9}");
    println!("réduction relative composite : {:.6} %", relative * 100.0);
    println!(
        "amélioration Brier            : {:.9}",
        baseline.binary.brier - challenger.binary.brier
    );
    println!(
        "amélioration MSE conditionnelle: {:.9}",
        baseline.conditional.mse - challenger.conditional.mse
    );
    println!(
        "amélioration MSE reconstruite : {:.9}",
        baseline.reconstruction.mse - challenger.reconstruction.mse
    );
    println!(
        "amélioration MAE reconstruite : {:.9}",
        baseline.reconstruction.mae - challenger.reconstruction.mae
    );

    Ok(DatasetEvaluation {
        baseline,
        challenger,
        baseline_predictions,
        challenger_predictions,
    })
}

fn relative_reduction(baseline: f64, challenger: f64) -> f64 {
    if baseline <= 0.0
    {
        0.0
    }
    else
    {
        (baseline - challenger) / baseline
    }
}

fn print_binary_metrics(label: &str, metrics: BinaryMetrics) {
    println!("{label}");
    println!("  Brier                 : {:.9}", metrics.brier);
    println!("  récupération observée : {:.9}", metrics.observed_rate);
    println!("  probabilité moyenne   : {:.9}", metrics.predicted_mean);
    println!(
        "  intercept calibration : {:.9}",
        metrics.calibration_intercept
    );
    println!("  pente calibration     : {:.9}", metrics.calibration_slope);
    println!(
        "  prédictions à 0       : {:.6} %",
        metrics.zero_fraction * 100.0
    );
    println!(
        "  prédictions à 1       : {:.6} %",
        metrics.one_fraction * 100.0
    );
}

fn print_metrics(label: &str, metrics: Metrics) {
    println!("{label}");
    println!("  MSE                   : {:.9}", metrics.mse);
    println!("  MAE                   : {:.9}", metrics.mae);
    println!("  R²                    : {:.9}", metrics.r_squared);
    println!("  Spearman              : {:.9}", metrics.spearman);
    println!("  biais moyen           : {:.9}", metrics.bias);
    println!("  moyenne observée      : {:.9}", metrics.observed_mean);
    println!("  moyenne prédite       : {:.9}", metrics.predicted_mean);
    println!(
        "  intercept calibration : {:.9}",
        metrics.calibration_intercept
    );
    println!("  pente calibration     : {:.9}", metrics.calibration_slope);
    println!(
        "  prédictions à 0       : {:.6} %",
        metrics.zero_fraction * 100.0
    );
    println!(
        "  prédictions à 1       : {:.6} %",
        metrics.one_fraction * 100.0
    );
}

fn print_composite_metrics(label: &str, metrics: CompositeMetrics) {
    println!("{label}");
    println!("  perte composite       : {:.9}", metrics.loss);
    println!();
    print_binary_metrics("  TÊTE BINAIRE", metrics.binary);
    println!();
    print_metrics("  TÊTE CONDITIONNELLE STANDARDISÉE", metrics.conditional);
    println!();
    print_metrics("  RECONSTRUCTION O₆", metrics.reconstruction);
}

fn print_model(label: &str, model: &RidgeModel) {
    println!();
    println!("{label}");
    println!("  intercept : {:.12}", model.coefficients[0]);

    for index in 0..model.means.len()
    {
        println!(
            "  feature {index:02} | moyenne={:.12} | \
             échelle={:.12} | coefficient={:.12}",
            model.means[index],
            model.scales[index],
            model.coefficients[index + 1],
        );
    }
}

fn print_interval(label: &str, interval: ConfidenceInterval) {
    println!(
        "{label}: [{:.9}, {:.9}] (médiane {:.9})",
        interval.lower, interval.upper, interval.median
    );
}

fn bootstrap_two_part(
    records: &[Record],
    baseline: &[TwoPartPrediction],
    challenger: &[TwoPartPrediction],
    scaler: TargetScaler,
) -> Result<BootstrapIntervals, String> {
    if records.len() != baseline.len() || records.len() != challenger.len() || records.is_empty()
    {
        return Err("invalid bootstrap dimensions".to_owned());
    }

    let mut rng = DeterministicRng::new(BOOTSTRAP_SEED);

    let mut composite = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    let mut brier = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    let mut conditional_mse = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    let mut reconstruction_mse = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    let mut reconstruction_mae = Vec::with_capacity(BOOTSTRAP_REPLICATES);

    for _ in 0..BOOTSTRAP_REPLICATES
    {
        let mut baseline_brier = 0.0;
        let mut challenger_brier = 0.0;
        let mut baseline_conditional = 0.0;
        let mut challenger_conditional = 0.0;
        let mut baseline_reconstruction_squared = 0.0;
        let mut challenger_reconstruction_squared = 0.0;
        let mut baseline_reconstruction_absolute = 0.0;
        let mut challenger_reconstruction_absolute = 0.0;
        let mut conditional_count = 0_usize;

        for _ in 0..records.len()
        {
            let index = rng.index(records.len());
            let record = &records[index];
            let baseline_prediction = baseline[index];
            let challenger_prediction = challenger[index];

            let binary_target = if record.recovered { 1.0 } else { 0.0 };

            let baseline_binary_residual = binary_target - baseline_prediction.probability;

            let challenger_binary_residual = binary_target - challenger_prediction.probability;

            baseline_brier += baseline_binary_residual * baseline_binary_residual;

            challenger_brier += challenger_binary_residual * challenger_binary_residual;

            let baseline_reconstruction_residual =
                record.overlap - baseline_prediction.reconstructed_overlap;

            let challenger_reconstruction_residual =
                record.overlap - challenger_prediction.reconstructed_overlap;

            baseline_reconstruction_squared +=
                baseline_reconstruction_residual * baseline_reconstruction_residual;

            challenger_reconstruction_squared +=
                challenger_reconstruction_residual * challenger_reconstruction_residual;

            baseline_reconstruction_absolute += baseline_reconstruction_residual.abs();

            challenger_reconstruction_absolute += challenger_reconstruction_residual.abs();

            if let Some(value) = record.conditional_u
            {
                let target = scaler.standardize(value);

                let baseline_residual = target - baseline_prediction.conditional_standardized;

                let challenger_residual = target - challenger_prediction.conditional_standardized;

                baseline_conditional += baseline_residual * baseline_residual;

                challenger_conditional += challenger_residual * challenger_residual;

                conditional_count += 1;
            }
        }

        if conditional_count == 0
        {
            return Err("bootstrap replicate contains no conditional observation".to_owned());
        }

        let count = records.len() as f64;
        let conditional_count = conditional_count as f64;

        let baseline_brier = baseline_brier / count;
        let challenger_brier = challenger_brier / count;

        let baseline_conditional = baseline_conditional / conditional_count;

        let challenger_conditional = challenger_conditional / conditional_count;

        let baseline_loss = 0.5 * baseline_brier + 0.5 * baseline_conditional;

        let challenger_loss = 0.5 * challenger_brier + 0.5 * challenger_conditional;

        composite.push(baseline_loss - challenger_loss);
        brier.push(baseline_brier - challenger_brier);

        conditional_mse.push(baseline_conditional - challenger_conditional);

        reconstruction_mse.push(
            baseline_reconstruction_squared / count - challenger_reconstruction_squared / count,
        );

        reconstruction_mae.push(
            baseline_reconstruction_absolute / count - challenger_reconstruction_absolute / count,
        );
    }

    Ok(BootstrapIntervals {
        composite: confidence_interval(composite),
        brier: confidence_interval(brier),
        conditional_mse: confidence_interval(conditional_mse),
        reconstruction_mse: confidence_interval(reconstruction_mse),
        reconstruction_mae: confidence_interval(reconstruction_mae),
    })
}

fn bootstrap_report(
    label: &str,
    records: &[Record],
    evaluation: &DatasetEvaluation,
    scaler: TargetScaler,
) -> Result<BootstrapIntervals, String> {
    let intervals = bootstrap_two_part(
        records,
        &evaluation.baseline_predictions,
        &evaluation.challenger_predictions,
        scaler,
    )?;

    println!();
    println!("{label}");

    print_interval(
        "IC 95 % amélioration perte composite TDI-4",
        intervals.composite,
    );

    print_interval("IC 95 % amélioration Brier TDI-4", intervals.brier);

    print_interval(
        "IC 95 % amélioration MSE conditionnelle TDI-4",
        intervals.conditional_mse,
    );

    print_interval(
        "IC 95 % amélioration MSE reconstruite TDI-4",
        intervals.reconstruction_mse,
    );

    print_interval(
        "IC 95 % amélioration MAE reconstruite TDI-4",
        intervals.reconstruction_mae,
    );

    Ok(intervals)
}

fn ensure_seed_ranges(ranges: &[(u64, u64, &str)]) -> Result<(), String> {
    for pair in ranges.windows(2)
    {
        let (_, previous_end, previous_label) = pair[0];
        let (next_start, _, next_label) = pair[1];

        if previous_end > next_start
        {
            return Err(format!(
                "seed ranges overlap: {previous_label} ends at \
                 {previous_end}, {next_label} starts at {next_start}"
            ));
        }
    }

    Ok(())
}

fn print_target_geometry(label: &str, records: &[Record]) {
    let recovered = records.iter().filter(|record| record.recovered).count();

    let conditional = records
        .iter()
        .filter_map(|record| record.conditional_u)
        .collect::<Vec<_>>();

    println!();
    println!("{label}");
    println!("  systèmes                 : {}", records.len());
    println!("  récupérations exactes    : {recovered}");
    println!(
        "  taux récupération exacte : {:.9}",
        recovered as f64 / records.len() as f64
    );
    println!("  systèmes conditionnels   : {}", conditional.len());

    if !conditional.is_empty()
    {
        let count = conditional.len() as f64;
        let mean = conditional.iter().sum::<f64>() / count;

        let variance = conditional
            .iter()
            .map(|value| {
                let difference = value - mean;
                difference * difference
            })
            .sum::<f64>()
            / count;

        let minimum = conditional
            .iter()
            .copied()
            .min_by(f64::total_cmp)
            .expect("non-empty conditional values");

        let maximum = conditional
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .expect("non-empty conditional values");

        println!("  U moyen                  : {mean:.9}");
        println!("  U écart-type             : {:.9}", variance.sqrt());
        println!("  U minimum                : {minimum:.9}");
        println!("  U maximum                : {maximum:.9}");
    }
}

fn print_deficit_deciles(label: &str, records: &[Record], evaluation: &DatasetEvaluation) {
    let mut entries = records
        .iter()
        .zip(&evaluation.baseline_predictions)
        .zip(&evaluation.challenger_predictions)
        .filter_map(|((record, baseline), challenger)| {
            record.conditional_u.map(|value| {
                (
                    2.0_f64.powf(-value),
                    record.overlap,
                    baseline.reconstructed_overlap,
                    challenger.reconstructed_overlap,
                )
            })
        })
        .collect::<Vec<_>>();

    entries.sort_by(|left, right| left.0.total_cmp(&right.0));

    println!();
    println!("{label}");

    if entries.is_empty()
    {
        println!("  aucune observation conditionnelle");
        return;
    }

    let bucket_count = 10_usize.min(entries.len());

    for bucket in 0..bucket_count
    {
        let start = bucket * entries.len() / bucket_count;
        let end = (bucket + 1) * entries.len() / bucket_count;
        let slice = &entries[start..end];

        let count = slice.len() as f64;

        let baseline_mse = slice
            .iter()
            .map(|(_, target, baseline, _)| {
                let residual = target - baseline;
                residual * residual
            })
            .sum::<f64>()
            / count;

        let challenger_mse = slice
            .iter()
            .map(|(_, target, _, challenger)| {
                let residual = target - challenger;
                residual * residual
            })
            .sum::<f64>()
            / count;

        println!(
            "  décile {:02} | n={} | déficit=[{:.12}, {:.12}] | \
             MSE baseline={:.12} | MSE TDI-4={:.12}",
            bucket + 1,
            slice.len(),
            slice.first().expect("non-empty bucket").0,
            slice.last().expect("non-empty bucket").0,
            baseline_mse,
            challenger_mse,
        );
    }
}

fn evaluate_direct_dataset(
    label: &str,
    records: &[Record],
    baseline_model: &RidgeModel,
    challenger_model: &RidgeModel,
) {
    let baseline_matrix = feature_matrix(records, baseline_features);

    let challenger_matrix = feature_matrix(records, challenger_features);

    let targets = overlap_targets(records);

    let baseline_predictions = predictions(baseline_model, &baseline_matrix);

    let challenger_predictions = predictions(challenger_model, &challenger_matrix);

    let baseline = calculate_metrics(&targets, &baseline_predictions);

    let challenger = calculate_metrics(&targets, &challenger_predictions);

    println!();
    println!("{label}");
    print_metrics("RÉGRESSION DIRECTE BASELINE", baseline);
    println!();
    print_metrics("RÉGRESSION DIRECTE BASELINE + TDI", challenger);
    println!(
        "  réduction relative MSE directe : {:.6} %",
        relative_reduction(baseline.mse, challenger.mse) * 100.0
    );
}

fn main() -> Result<(), String> {
    println!("Generating preregistered TDI-4 width-3 training systems...");

    let (training_width_3, training_width_3_next_seed, training_width_3_excluded) =
        generate_records(
            TRAIN_WIDTH_3,
            TRAIN_WIDTH_3_SEED_OFFSET,
            TRAIN_SYSTEMS_PER_WIDTH,
        )?;

    println!("Generating untouched TDI-4 width-3 holdout systems...");

    let (holdout_width_3, holdout_width_3_next_seed, holdout_width_3_excluded) = generate_records(
        TRAIN_WIDTH_3,
        HOLDOUT_WIDTH_3_SEED_OFFSET,
        HOLDOUT_SYSTEMS_PER_WIDTH,
    )?;

    println!("Generating preregistered TDI-4 width-4 training systems...");

    let (training_width_4, training_width_4_next_seed, training_width_4_excluded) =
        generate_records(
            TRAIN_WIDTH_4,
            TRAIN_WIDTH_4_SEED_OFFSET,
            TRAIN_SYSTEMS_PER_WIDTH,
        )?;

    println!("Generating untouched TDI-4 width-4 holdout systems...");

    let (holdout_width_4, holdout_width_4_next_seed, holdout_width_4_excluded) = generate_records(
        TRAIN_WIDTH_4,
        HOLDOUT_WIDTH_4_SEED_OFFSET,
        HOLDOUT_SYSTEMS_PER_WIDTH,
    )?;

    println!("Generating untouched TDI-4 width-5 OOD systems...");

    let (holdout_width_5, holdout_width_5_next_seed, holdout_width_5_excluded) =
        generate_records(OOD_WIDTH, OOD_WIDTH_5_SEED_OFFSET, OOD_SYSTEMS)?;

    ensure_seed_ranges(&[
        (
            TRAIN_WIDTH_3_SEED_OFFSET,
            training_width_3_next_seed,
            "training width 3",
        ),
        (
            HOLDOUT_WIDTH_3_SEED_OFFSET,
            holdout_width_3_next_seed,
            "holdout width 3",
        ),
        (
            TRAIN_WIDTH_4_SEED_OFFSET,
            training_width_4_next_seed,
            "training width 4",
        ),
        (
            HOLDOUT_WIDTH_4_SEED_OFFSET,
            holdout_width_4_next_seed,
            "holdout width 4",
        ),
        (
            OOD_WIDTH_5_SEED_OFFSET,
            holdout_width_5_next_seed,
            "holdout width 5",
        ),
    ])?;

    let training_width_3_count = training_width_3.len();
    let training_width_4_count = training_width_4.len();

    let mut training = training_width_3;
    training.extend(training_width_4);

    let mut combined_holdout = holdout_width_3.clone();
    combined_holdout.extend(holdout_width_4.iter().cloned());

    let scaler = TargetScaler::fit(&training)?;

    let baseline_model = fit_two_part_model(&training, baseline_features, scaler)?;

    let challenger_model = fit_two_part_model(&training, challenger_features, scaler)?;

    let training_baseline = feature_matrix(&training, baseline_features);

    let training_challenger = feature_matrix(&training, challenger_features);

    let direct_targets = overlap_targets(&training);

    let direct_baseline_model = fit_ridge(&training_baseline, &direct_targets)?;

    let direct_challenger_model = fit_ridge(&training_challenger, &direct_targets)?;

    println!();
    println!("TDI-4 preregistered target-geometry evaluation");
    println!("training widths             : 3 and 4");
    println!("OOD width                   : 5");
    println!("training width 3 systems    : {training_width_3_count}");
    println!("training width 4 systems    : {training_width_4_count}");
    println!("combined training systems   : {}", training.len());
    println!("holdout width 3 systems     : {}", holdout_width_3.len());
    println!("holdout width 4 systems     : {}", holdout_width_4.len());
    println!("combined holdout systems    : {}", combined_holdout.len());
    println!("holdout width 5 systems     : {}", holdout_width_5.len());
    println!("observation horizon         : {OBSERVATION_HORIZON}");
    println!("outcome horizon             : {OUTCOME_HORIZON}");
    println!("ridge lambda                : {RIDGE_LAMBDA}");
    println!("bootstrap replicates        : {BOOTSTRAP_REPLICATES}");
    println!("conditional U mean          : {:.12}", scaler.mean);
    println!("conditional U scale         : {:.12}", scaler.scale);
    println!("training width 3 excluded   : {training_width_3_excluded}");
    println!("holdout width 3 excluded    : {holdout_width_3_excluded}");
    println!("training width 4 excluded   : {training_width_4_excluded}");
    println!("holdout width 4 excluded    : {holdout_width_4_excluded}");
    println!("holdout width 5 excluded    : {holdout_width_5_excluded}");
    println!("training width 3 next seed  : {training_width_3_next_seed}");
    println!("holdout width 3 next seed   : {holdout_width_3_next_seed}");
    println!("training width 4 next seed  : {training_width_4_next_seed}");
    println!("holdout width 4 next seed   : {holdout_width_4_next_seed}");
    println!("holdout width 5 next seed   : {holdout_width_5_next_seed}");

    print_target_geometry("TARGET GEOMETRY — TRAINING", &training);
    print_target_geometry("TARGET GEOMETRY — WIDTH 3", &holdout_width_3);
    print_target_geometry("TARGET GEOMETRY — WIDTH 4", &holdout_width_4);
    print_target_geometry("TARGET GEOMETRY — WIDTH 5", &holdout_width_5);

    print_model("BASELINE — TÊTE BINAIRE", &baseline_model.binary);
    print_model(
        "BASELINE — TÊTE CONDITIONNELLE",
        &baseline_model.conditional,
    );
    print_model("BASELINE + TDI-4 — TÊTE BINAIRE", &challenger_model.binary);
    print_model(
        "BASELINE + TDI-4 — TÊTE CONDITIONNELLE",
        &challenger_model.conditional,
    );
    print_model("COMPARATEUR DIRECT — BASELINE", &direct_baseline_model);
    print_model(
        "COMPARATEUR DIRECT — BASELINE + TDI",
        &direct_challenger_model,
    );

    let width_3 = evaluate_dataset(
        "HOLDOUT WIDTH 3",
        &holdout_width_3,
        &baseline_model,
        &challenger_model,
    )?;

    let width_4 = evaluate_dataset(
        "HOLDOUT WIDTH 4",
        &holdout_width_4,
        &baseline_model,
        &challenger_model,
    )?;

    let combined = evaluate_dataset(
        "HOLDOUT COMBINÉ WIDTHS 3 ET 4",
        &combined_holdout,
        &baseline_model,
        &challenger_model,
    )?;

    let width_5 = evaluate_dataset(
        "HOLDOUT HORS DISTRIBUTION — WIDTH 5",
        &holdout_width_5,
        &baseline_model,
        &challenger_model,
    )?;

    evaluate_direct_dataset(
        "COMPARATEUR SECONDAIRE DIRECT — WIDTH 3",
        &holdout_width_3,
        &direct_baseline_model,
        &direct_challenger_model,
    );

    evaluate_direct_dataset(
        "COMPARATEUR SECONDAIRE DIRECT — WIDTH 4",
        &holdout_width_4,
        &direct_baseline_model,
        &direct_challenger_model,
    );

    evaluate_direct_dataset(
        "COMPARATEUR SECONDAIRE DIRECT — WIDTH 5",
        &holdout_width_5,
        &direct_baseline_model,
        &direct_challenger_model,
    );

    print_deficit_deciles("DÉCILES DU DÉFICIT — WIDTH 3", &holdout_width_3, &width_3);

    print_deficit_deciles("DÉCILES DU DÉFICIT — WIDTH 4", &holdout_width_4, &width_4);

    print_deficit_deciles("DÉCILES DU DÉFICIT — WIDTH 5", &holdout_width_5, &width_5);

    println!();
    println!(
        "Bootstrap apparié déterministe : \
         {BOOTSTRAP_REPLICATES} réplications"
    );

    let width_3_intervals =
        bootstrap_report("BOOTSTRAP WIDTH 3", &holdout_width_3, &width_3, scaler)?;

    let width_4_intervals =
        bootstrap_report("BOOTSTRAP WIDTH 4", &holdout_width_4, &width_4, scaler)?;

    let combined_intervals = bootstrap_report(
        "BOOTSTRAP HOLDOUT COMBINÉ WIDTHS 3 ET 4",
        &combined_holdout,
        &combined,
        scaler,
    )?;

    let width_5_intervals =
        bootstrap_report("BOOTSTRAP WIDTH 5", &holdout_width_5, &width_5, scaler)?;

    let tdi_4a_success = relative_reduction(combined.baseline.loss, combined.challenger.loss)
        >= 0.05
        && combined_intervals.composite.lower > 0.0
        && combined_intervals.brier.lower > 0.0
        && width_3_intervals.composite.lower > 0.0
        && width_4_intervals.composite.lower > 0.0
        && combined.baseline.reconstruction.mse - combined.challenger.reconstruction.mse > 0.0
        && combined.baseline.reconstruction.mae - combined.challenger.reconstruction.mae > 0.0
        && width_3.challenger.conditional.spearman > 0.0
        && width_4.challenger.conditional.spearman > 0.0;

    let tdi_4b_success = width_5_intervals.composite.lower > 0.0
        && relative_reduction(
            width_5.baseline.reconstruction.mse,
            width_5.challenger.reconstruction.mse,
        ) >= 0.05
        && width_5_intervals.reconstruction_mse.lower > 0.0
        && width_5_intervals.brier.lower > 0.0
        && width_5.challenger.conditional.spearman > 0.0
        && width_5.challenger.conditional.spearman >= width_5.baseline.conditional.spearman
        && width_5.challenger.reconstruction.bias.abs()
            < width_5.baseline.reconstruction.bias.abs();

    println!();
    println!(
        "CRITÈRE PRINCIPAL TDI-4A : {}",
        if tdi_4a_success
        {
            "RÉUSSI"
        }
        else
        {
            "ÉCHOUÉ"
        }
    );

    println!(
        "CRITÈRE TRANSFERT TDI-4B : {}",
        if tdi_4b_success
        {
            "RÉUSSI"
        }
        else
        {
            "ÉCHOUÉ"
        }
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        BASELINE_FEATURE_COUNT, BOOTSTRAP_SEED, ConfidenceInterval, DeterministicRng, Metrics,
        RIDGE_LAMBDA, Record, TDI_FEATURE_COUNT, TargetScaler, average_ranks, calculate_metrics,
        confidence_interval, reconstruct_overlap, splitmix64,
    };

    #[test]
    fn deterministic_rng_is_reproducible() {
        let mut first = DeterministicRng::new(BOOTSTRAP_SEED);
        let mut second = DeterministicRng::new(BOOTSTRAP_SEED);

        for _ in 0..100
        {
            assert_eq!(first.next_u64(), second.next_u64());
        }
    }

    #[test]
    fn splitmix_is_deterministic() {
        assert_eq!(splitmix64(42), splitmix64(42));
        assert_ne!(splitmix64(42), splitmix64(43));
    }

    #[test]
    fn exact_deficit_geometry_is_correct() {
        let ratio = scirust_tdi::ExactRatio::new(7, 8).expect("valid ratio");

        let transformed =
            super::exact_overlap_deficit_u(&ratio).expect("valid conditional geometry");

        assert!((transformed - 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn biguint_logarithm_supports_more_than_128_bits() {
        let digits = [0_u64, 0_u64, 1_u64];

        let logarithm =
            super::biguint_log2_from_u64_digits(&digits).expect("large integer logarithm");

        assert!((logarithm - 128.0).abs() < 1.0e-12);
    }

    #[test]
    fn target_scaler_round_trips() {
        let records = [
            Record {
                baseline: [0.0; BASELINE_FEATURE_COUNT],
                tdi: [0.0; TDI_FEATURE_COUNT],
                overlap: 0.5,
                recovered: false,
                conditional_u: Some(1.0),
            },
            Record {
                baseline: [0.0; BASELINE_FEATURE_COUNT],
                tdi: [0.0; TDI_FEATURE_COUNT],
                overlap: 0.75,
                recovered: false,
                conditional_u: Some(2.0),
            },
        ];

        let scaler = TargetScaler::fit(&records).expect("valid scaler");
        let value = 1.75;

        assert!((scaler.unstandardize(scaler.standardize(value)) - value).abs() < 1.0e-12);
    }

    #[test]
    fn reconstruction_respects_unit_interval() {
        assert_eq!(reconstruct_overlap(1.0, -1000.0), 1.0);
        assert_eq!(reconstruct_overlap(0.0, 0.0), 0.0);

        let reconstructed = reconstruct_overlap(0.5, 3.0);

        assert!((0.0..=1.0).contains(&reconstructed));
        assert!((reconstructed - 0.9375).abs() < 1.0e-12);
    }

    #[test]
    fn identity_metrics_are_exact() {
        let values = [0.1, 0.3, 0.6, 0.9];

        assert_eq!(
            calculate_metrics(&values, &values),
            Metrics {
                mse: 0.0,
                mae: 0.0,
                r_squared: 1.0,
                spearman: 1.0,
                bias: 0.0,
                observed_mean: 0.475,
                predicted_mean: 0.475,
                calibration_intercept: 0.0,
                calibration_slope: 1.0,
                zero_fraction: 0.0,
                one_fraction: 0.0,
            }
        );
    }

    #[test]
    fn ranks_handle_ties() {
        assert_eq!(
            average_ranks(&[3.0, 1.0, 1.0, 2.0]),
            vec![4.0, 1.5, 1.5, 3.0]
        );
    }

    #[test]
    fn confidence_interval_is_ordered() {
        let interval = confidence_interval(vec![3.0, 1.0, 4.0, 2.0]);

        assert!(interval.lower <= interval.median);
        assert!(interval.median <= interval.upper);

        let _ = ConfidenceInterval {
            lower: interval.lower,
            median: interval.median,
            upper: interval.upper,
        };
    }

    #[test]
    fn preregistered_layout_is_fixed() {
        assert_eq!(BASELINE_FEATURE_COUNT, 13);
        assert_eq!(TDI_FEATURE_COUNT, 3);
        assert_eq!(RIDGE_LAMBDA, 1.0);
        assert_eq!(BOOTSTRAP_SEED, 0x5444_4934_4745_4F4D);
    }
}
