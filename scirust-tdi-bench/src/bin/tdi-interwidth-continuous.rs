use scirust_tdi::{
    Action, ExactRatio, State, TableSystem, analyze_branching_recovery, explore,
    uniform_branching_path_entropy_bits,
};

const TRAIN_WIDTH_3: u8 = 3;
const TRAIN_WIDTH_4: u8 = 4;
const OOD_WIDTH: u8 = 5;

const TRAIN_SYSTEMS_PER_WIDTH: usize = 8_000;
const HOLDOUT_SYSTEMS_PER_WIDTH: usize = 4_000;
const OOD_SYSTEMS: usize = 4_000;

const TRAIN_WIDTH_3_SEED_OFFSET: u64 = 0;
const HOLDOUT_WIDTH_3_SEED_OFFSET: u64 = 1_000_000;
const TRAIN_WIDTH_4_SEED_OFFSET: u64 = 10_000_000;
const HOLDOUT_WIDTH_4_SEED_OFFSET: u64 = 11_000_000;
const OOD_WIDTH_5_SEED_OFFSET: u64 = 20_000_000;

const OBSERVATION_HORIZON: usize = 2;
const OUTCOME_HORIZON: usize = 6;

const BASELINE_FEATURE_COUNT: usize = 13;
const TDI_FEATURE_COUNT: usize = 3;

const RIDGE_LAMBDA: f64 = 1.0;
const BOOTSTRAP_REPLICATES: usize = 2_000;
const BOOTSTRAP_SEED: u64 = 0x5444_4933_494E_5445;

#[derive(Clone, Debug)]
struct Record {
    baseline: [f64; BASELINE_FEATURE_COUNT],
    tdi: [f64; TDI_FEATURE_COUNT],
    target: f64,
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
    fn predict(&self, features: &[f64]) -> f64 {
        assert_eq!(features.len(), self.means.len());
        assert_eq!(features.len(), self.scales.len());
        assert_eq!(self.coefficients.len(), features.len() + 1);

        let linear = features
            .iter()
            .zip(&self.means)
            .zip(&self.scales)
            .zip(self.coefficients.iter().skip(1))
            .fold(
                self.coefficients[0],
                |accumulator, (((value, mean), scale), coefficient)| {
                    accumulator + coefficient * ((value - mean) / scale)
                },
            );

        linear.clamp(0.0, 1.0)
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

    let final_overlap = outcome
        .final_overlap()
        .ok_or_else(|| "outcome horizon produced no overlap".to_owned())?
        .as_f64();

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
        target: final_overlap,
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

fn target_vector(records: &[Record]) -> Vec<f64> {
    records.iter().map(|record| record.target).collect()
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
    features.iter().map(|row| model.predict(row)).collect()
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

fn paired_bootstrap(
    targets: &[f64],
    baseline: &[f64],
    challenger: &[f64],
) -> (ConfidenceInterval, ConfidenceInterval) {
    assert_eq!(targets.len(), baseline.len());
    assert_eq!(targets.len(), challenger.len());

    let mut rng = DeterministicRng::new(BOOTSTRAP_SEED);
    let mut mse_improvements = Vec::with_capacity(BOOTSTRAP_REPLICATES);
    let mut mae_improvements = Vec::with_capacity(BOOTSTRAP_REPLICATES);

    for _ in 0..BOOTSTRAP_REPLICATES
    {
        let mut baseline_squared_error = 0.0_f64;
        let mut challenger_squared_error = 0.0_f64;
        let mut baseline_absolute_error = 0.0_f64;
        let mut challenger_absolute_error = 0.0_f64;

        for _ in 0..targets.len()
        {
            let index = rng.index(targets.len());
            let target = targets[index];

            let baseline_residual = target - baseline[index];
            let challenger_residual = target - challenger[index];

            baseline_squared_error += baseline_residual * baseline_residual;

            challenger_squared_error += challenger_residual * challenger_residual;

            baseline_absolute_error += baseline_residual.abs();
            challenger_absolute_error += challenger_residual.abs();
        }

        let count = targets.len() as f64;

        mse_improvements.push(baseline_squared_error / count - challenger_squared_error / count);

        mae_improvements.push(baseline_absolute_error / count - challenger_absolute_error / count);
    }

    (
        confidence_interval(mse_improvements),
        confidence_interval(mae_improvements),
    )
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

fn print_interval(label: &str, interval: ConfidenceInterval) {
    println!(
        "{label}: [{:.9}, {:.9}] (médiane {:.9})",
        interval.lower, interval.upper, interval.median
    );
}

fn evaluate_dataset(
    label: &str,
    records: &[Record],
    baseline_model: &RidgeModel,
    challenger_model: &RidgeModel,
) -> (Metrics, Metrics, Vec<f64>, Vec<f64>, Vec<f64>) {
    let baseline_matrix = feature_matrix(records, baseline_features);
    let challenger_matrix = feature_matrix(records, challenger_features);
    let targets = target_vector(records);

    let baseline_predictions = predictions(baseline_model, &baseline_matrix);

    let challenger_predictions = predictions(challenger_model, &challenger_matrix);

    let baseline_metrics = calculate_metrics(&targets, &baseline_predictions);

    let challenger_metrics = calculate_metrics(&targets, &challenger_predictions);

    println!();
    println!("{label}");

    println!();
    print_metrics("BASELINE APPARIÉE", baseline_metrics);

    println!();
    print_metrics("BASELINE + TDI-3", challenger_metrics);

    let mse_improvement = baseline_metrics.mse - challenger_metrics.mse;

    let relative_mse_reduction = if baseline_metrics.mse <= 0.0
    {
        0.0
    }
    else
    {
        mse_improvement / baseline_metrics.mse
    };

    println!();
    println!("amélioration MSE          : {mse_improvement:.9}");
    println!(
        "réduction relative MSE    : {:.6} %",
        relative_mse_reduction * 100.0
    );
    println!(
        "amélioration MAE          : {:.9}",
        baseline_metrics.mae - challenger_metrics.mae
    );

    (
        baseline_metrics,
        challenger_metrics,
        targets,
        baseline_predictions,
        challenger_predictions,
    )
}

fn print_model(label: &str, model: &RidgeModel) {
    println!();
    println!("{label}");
    println!("  intercept : {:.12}", model.coefficients[0]);

    for index in 0..model.means.len()
    {
        println!(
            "  feature {index:02} | moyenne={:.12} | échelle={:.12} | coefficient={:.12}",
            model.means[index],
            model.scales[index],
            model.coefficients[index + 1],
        );
    }
}

fn ensure_seed_ranges(ranges: &[(u64, u64, &str)]) -> Result<(), String> {
    for pair in ranges.windows(2)
    {
        let (_, previous_end, previous_label) = pair[0];
        let (next_start, _, next_label) = pair[1];

        if previous_end > next_start
        {
            return Err(format!(
                "seed ranges overlap: {previous_label} ends at {previous_end},                  {next_label} starts at {next_start}"
            ));
        }
    }

    Ok(())
}

fn bootstrap_report(
    label: &str,
    targets: &[f64],
    baseline_predictions: &[f64],
    challenger_predictions: &[f64],
) -> (ConfidenceInterval, ConfidenceInterval) {
    let (mse_interval, mae_interval) =
        paired_bootstrap(targets, baseline_predictions, challenger_predictions);

    println!();
    println!("{label}");
    print_interval("IC 95 % amélioration MSE TDI-3", mse_interval);
    print_interval("IC 95 % amélioration MAE TDI-3", mae_interval);

    (mse_interval, mae_interval)
}

fn main() -> Result<(), String> {
    println!("Generating preregistered TDI-3 width-3 training systems...");

    let (training_width_3, training_width_3_next_seed, training_width_3_excluded) =
        generate_records(
            TRAIN_WIDTH_3,
            TRAIN_WIDTH_3_SEED_OFFSET,
            TRAIN_SYSTEMS_PER_WIDTH,
        )?;

    println!("Generating untouched TDI-3 width-3 holdout systems...");

    let (holdout_width_3, holdout_width_3_next_seed, holdout_width_3_excluded) = generate_records(
        TRAIN_WIDTH_3,
        HOLDOUT_WIDTH_3_SEED_OFFSET,
        HOLDOUT_SYSTEMS_PER_WIDTH,
    )?;

    println!("Generating preregistered TDI-3 width-4 training systems...");

    let (training_width_4, training_width_4_next_seed, training_width_4_excluded) =
        generate_records(
            TRAIN_WIDTH_4,
            TRAIN_WIDTH_4_SEED_OFFSET,
            TRAIN_SYSTEMS_PER_WIDTH,
        )?;

    println!("Generating untouched TDI-3 width-4 holdout systems...");

    let (holdout_width_4, holdout_width_4_next_seed, holdout_width_4_excluded) = generate_records(
        TRAIN_WIDTH_4,
        HOLDOUT_WIDTH_4_SEED_OFFSET,
        HOLDOUT_SYSTEMS_PER_WIDTH,
    )?;

    println!("Generating untouched TDI-3 width-5 OOD systems...");

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

    let training_baseline = feature_matrix(&training, baseline_features);
    let training_challenger = feature_matrix(&training, challenger_features);
    let training_targets = target_vector(&training);

    let baseline_model = fit_ridge(&training_baseline, &training_targets)?;
    let challenger_model = fit_ridge(&training_challenger, &training_targets)?;

    println!();
    println!("TDI-3 preregistered inter-width continuous evaluation");
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

    print_model("MODÈLE BASELINE", &baseline_model);
    print_model("MODÈLE BASELINE + TDI-3", &challenger_model);

    let (
        width_3_baseline,
        width_3_challenger,
        width_3_targets,
        width_3_baseline_predictions,
        width_3_challenger_predictions,
    ) = evaluate_dataset(
        "HOLDOUT WIDTH 3",
        &holdout_width_3,
        &baseline_model,
        &challenger_model,
    );

    let (
        width_4_baseline,
        width_4_challenger,
        width_4_targets,
        width_4_baseline_predictions,
        width_4_challenger_predictions,
    ) = evaluate_dataset(
        "HOLDOUT WIDTH 4",
        &holdout_width_4,
        &baseline_model,
        &challenger_model,
    );

    let (
        combined_baseline,
        combined_challenger,
        combined_targets,
        combined_baseline_predictions,
        combined_challenger_predictions,
    ) = evaluate_dataset(
        "HOLDOUT COMBINÉ WIDTHS 3 ET 4",
        &combined_holdout,
        &baseline_model,
        &challenger_model,
    );

    let (
        width_5_baseline,
        width_5_challenger,
        width_5_targets,
        width_5_baseline_predictions,
        width_5_challenger_predictions,
    ) = evaluate_dataset(
        "HOLDOUT HORS DISTRIBUTION — WIDTH 5",
        &holdout_width_5,
        &baseline_model,
        &challenger_model,
    );

    println!();
    println!("Bootstrap apparié déterministe :          {BOOTSTRAP_REPLICATES} réplications");

    let _ = bootstrap_report(
        "BOOTSTRAP WIDTH 3",
        &width_3_targets,
        &width_3_baseline_predictions,
        &width_3_challenger_predictions,
    );

    let _ = bootstrap_report(
        "BOOTSTRAP WIDTH 4",
        &width_4_targets,
        &width_4_baseline_predictions,
        &width_4_challenger_predictions,
    );

    let (combined_mse_interval, combined_mae_interval) = bootstrap_report(
        "BOOTSTRAP HOLDOUT COMBINÉ WIDTHS 3 ET 4",
        &combined_targets,
        &combined_baseline_predictions,
        &combined_challenger_predictions,
    );

    let (width_5_mse_interval, width_5_mae_interval) = bootstrap_report(
        "BOOTSTRAP WIDTH 5",
        &width_5_targets,
        &width_5_baseline_predictions,
        &width_5_challenger_predictions,
    );

    let combined_mse_improvement = combined_baseline.mse - combined_challenger.mse;

    let combined_relative_mse_reduction = if combined_baseline.mse <= 0.0
    {
        0.0
    }
    else
    {
        combined_mse_improvement / combined_baseline.mse
    };

    let tdi_3a_success = combined_relative_mse_reduction >= 0.05
        && combined_mse_interval.lower > 0.0
        && combined_mae_interval.lower > 0.0
        && width_3_baseline.mse - width_3_challenger.mse > 0.0
        && width_4_baseline.mse - width_4_challenger.mse > 0.0
        && width_3_challenger.spearman > 0.0
        && width_4_challenger.spearman > 0.0
        && width_3_challenger.r_squared > 0.0
        && width_4_challenger.r_squared > 0.0;

    let tdi_3b_success = width_5_baseline.mse - width_5_challenger.mse > 0.0
        && width_5_mse_interval.lower > 0.0
        && width_5_mae_interval.lower > 0.0
        && width_5_challenger.r_squared > 0.0
        && width_5_challenger.spearman > 0.0
        && width_5_challenger.bias.abs() < width_5_baseline.bias.abs();

    println!();
    println!(
        "CRITÈRE PRINCIPAL TDI-3A : {}",
        if tdi_3a_success
        {
            "RÉUSSI"
        }
        else
        {
            "ÉCHOUÉ"
        }
    );

    println!(
        "CRITÈRE TRANSFERT TDI-3B : {}",
        if tdi_3b_success
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
        BASELINE_FEATURE_COUNT, BOOTSTRAP_SEED, ConfidenceInterval, HOLDOUT_WIDTH_3_SEED_OFFSET,
        HOLDOUT_WIDTH_4_SEED_OFFSET, Metrics, OOD_WIDTH_5_SEED_OFFSET, Record, RidgeModel,
        TDI_FEATURE_COUNT, TRAIN_WIDTH_3_SEED_OFFSET, TRAIN_WIDTH_4_SEED_OFFSET, analyze_seed,
        average_ranks, calculate_metrics, fit_ridge, generate_records, generate_successor_masks,
        paired_bootstrap, predictions,
    };

    #[test]
    fn successor_generation_is_deterministic() {
        let first = generate_successor_masks(3, 42).expect("generation succeeds");

        let second = generate_successor_masks(3, 42).expect("generation succeeds");

        assert_eq!(first, second);
        assert!(first.iter().all(|mask| *mask != 0));
    }

    #[test]
    fn average_ranks_handle_ties() {
        assert_eq!(
            average_ranks(&[3.0, 1.0, 1.0, 2.0]),
            vec![4.0, 1.5, 1.5, 3.0]
        );
    }

    #[test]
    fn ridge_model_learns_a_continuous_signal() {
        let features = (0..100)
            .map(|index| vec![index as f64 / 100.0])
            .collect::<Vec<_>>();

        let targets = features
            .iter()
            .map(|row| 0.1 + 0.7 * row[0])
            .collect::<Vec<_>>();

        let model = fit_ridge(&features, &targets).expect("ridge fit succeeds");

        let predicted = predictions(&model, &features);
        let metrics = calculate_metrics(&targets, &predicted);

        assert!(metrics.mse < 0.001);
        assert!(metrics.spearman > 0.99);
        assert!(metrics.r_squared > 0.99);
    }

    #[test]
    fn identity_predictions_have_exact_calibration() {
        let values = [0.1, 0.3, 0.6, 0.9];
        let metrics = calculate_metrics(&values, &values);

        assert_eq!(
            metrics,
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
    fn ridge_predictions_are_bounded() {
        let model = RidgeModel {
            means: vec![0.0],
            scales: vec![1.0],
            coefficients: vec![0.5, 10.0],
        };

        assert_eq!(model.predict(&[-1.0]), 0.0);
        assert_eq!(model.predict(&[1.0]), 1.0);
    }

    #[test]
    fn feature_layout_has_preregistered_lengths() {
        let record = Record {
            baseline: [0.0; BASELINE_FEATURE_COUNT],
            tdi: [0.0; TDI_FEATURE_COUNT],
            target: 0.5,
        };

        assert_eq!(record.baseline.len(), 13);
        assert_eq!(record.tdi.len(), 3);
    }

    #[test]
    fn normalized_features_are_finite() {
        let (records, _, _) = generate_records(3, 0, 8).expect("records generated");

        for record in records
        {
            assert!(record.baseline.iter().all(|value| value.is_finite()));
            assert!(record.tdi.iter().all(|value| value.is_finite()));
            assert!((0.0..=1.0).contains(&record.target));
            assert_eq!(record.baseline[12], 3.0);
        }
    }

    #[test]
    fn preregistered_seed_offsets_are_disjoint() {
        let offsets = [
            TRAIN_WIDTH_3_SEED_OFFSET,
            HOLDOUT_WIDTH_3_SEED_OFFSET,
            TRAIN_WIDTH_4_SEED_OFFSET,
            HOLDOUT_WIDTH_4_SEED_OFFSET,
            OOD_WIDTH_5_SEED_OFFSET,
        ];

        assert!(
            offsets.windows(2).all(|pair| pair[0] < pair[1]),
            "preregistered seed offsets must be strictly increasing"
        );
    }

    #[test]
    fn width_five_regression_seed_does_not_overflow() {
        assert!(
            analyze_seed(5, OOD_WIDTH_5_SEED_OFFSET).is_ok(),
            "width-5 seed 20_000_000 must not overflow exact arithmetic"
        );
    }

    #[test]
    fn bootstrap_is_deterministic() {
        let targets = [0.1, 0.4, 0.7, 0.9];
        let baseline = [0.2, 0.5, 0.6, 0.8];
        let challenger = [0.1, 0.4, 0.7, 0.9];

        let first = paired_bootstrap(&targets, &baseline, &challenger);

        let second = paired_bootstrap(&targets, &baseline, &challenger);

        assert_eq!(first, second);
        assert_eq!(BOOTSTRAP_SEED, 0x5444_4933_494E_5445);

        let positive = ConfidenceInterval {
            lower: first.0.lower,
            median: first.0.median,
            upper: first.0.upper,
        };

        assert!(positive.lower > 0.0);
    }
}
