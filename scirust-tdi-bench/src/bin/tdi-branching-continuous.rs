use scirust_tdi::{
    Action, ExactRatio, State, TableSystem, analyze_branching_recovery, explore,
    uniform_branching_path_entropy_bits,
};

const TRAIN_WIDTH: u8 = 3;
const OOD_WIDTH: u8 = 4;

const TRAIN_SYSTEMS: usize = 12_000;
const TEST_SYSTEMS: usize = 4_000;
const OOD_SYSTEMS: usize = 4_000;

const TEST_SEED_OFFSET: u64 = 1_000_000;
const OOD_SEED_OFFSET: u64 = 2_000_000;

const OBSERVATION_HORIZON: usize = 2;
const OUTCOME_HORIZON: usize = 6;

const BASELINE_FEATURE_COUNT: usize = 12;
const TDI_FEATURE_COUNT: usize = 3;

const RIDGE_LAMBDA: f64 = 1.0;
const BOOTSTRAP_REPLICATES: usize = 2_000;
const BOOTSTRAP_SEED: u64 = 0x5444_4932_434F_4E54;

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

#[derive(Clone, Copy, Debug)]
struct Metrics {
    mse: f64,
    mae: f64,
    r_squared: f64,
    spearman: f64,
}

#[derive(Clone, Copy, Debug)]
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
        reference_entropy[0],
        reference_entropy[1],
        perturbed_entropy[0],
        perturbed_entropy[1],
        reference_reachable[0],
        reference_reachable[1],
        reference_paths[0],
        reference_paths[1],
        perturbed_reachable[0],
        perturbed_reachable[1],
        perturbed_paths[0],
        perturbed_paths[1],
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

    let sample_count = targets.len() as f64;
    let target_mean = targets.iter().sum::<f64>() / sample_count;

    let mut squared_error = 0.0_f64;
    let mut absolute_error = 0.0_f64;
    let mut total_variance = 0.0_f64;

    for (&target, &prediction) in targets.iter().zip(predicted)
    {
        let residual = target - prediction;

        squared_error += residual * residual;
        absolute_error += residual.abs();

        let centered = target - target_mean;
        total_variance += centered * centered;
    }

    let r_squared = if total_variance <= 1.0e-15
    {
        0.0
    }
    else
    {
        1.0 - squared_error / total_variance
    };

    Metrics {
        mse: squared_error / sample_count,
        mae: absolute_error / sample_count,
        r_squared,
        spearman: spearman_correlation(targets, predicted),
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
    println!("  MSE      : {:.9}", metrics.mse);
    println!("  MAE      : {:.9}", metrics.mae);
    println!("  R²       : {:.9}", metrics.r_squared);
    println!("  Spearman : {:.9}", metrics.spearman);
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
    print_metrics("BASELINE + TDI-2", challenger_metrics);

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

fn main() -> Result<(), String> {
    println!("Generating preregistered TDI-2 training systems...");

    let (training, training_next_seed, training_excluded) =
        generate_records(TRAIN_WIDTH, 0, TRAIN_SYSTEMS)?;

    println!("Generating untouched in-distribution holdout...");

    let (test, test_next_seed, test_excluded) =
        generate_records(TRAIN_WIDTH, TEST_SEED_OFFSET, TEST_SYSTEMS)?;

    println!("Generating untouched out-of-distribution holdout...");

    let (ood, ood_next_seed, ood_excluded) =
        generate_records(OOD_WIDTH, OOD_SEED_OFFSET, OOD_SYSTEMS)?;

    let training_baseline = feature_matrix(&training, baseline_features);

    let training_challenger = feature_matrix(&training, challenger_features);

    let training_targets = target_vector(&training);

    let baseline_model = fit_ridge(&training_baseline, &training_targets)?;

    let challenger_model = fit_ridge(&training_challenger, &training_targets)?;

    println!();
    println!("TDI-2 preregistered continuous evaluation");
    println!("training width          : {TRAIN_WIDTH}");
    println!("OOD width               : {OOD_WIDTH}");
    println!("training systems        : {}", training.len());
    println!("holdout systems         : {}", test.len());
    println!("OOD holdout systems     : {}", ood.len());
    println!("observation horizon     : {OBSERVATION_HORIZON}");
    println!("outcome horizon         : {OUTCOME_HORIZON}");
    println!("ridge lambda            : {RIDGE_LAMBDA}");
    println!("training excluded       : {training_excluded}");
    println!("holdout excluded        : {test_excluded}");
    println!("OOD excluded            : {ood_excluded}");
    println!("training next seed      : {training_next_seed}");
    println!("holdout next seed       : {test_next_seed}");
    println!("OOD next seed           : {ood_next_seed}");

    let (
        test_baseline,
        test_challenger,
        test_targets,
        test_baseline_predictions,
        test_challenger_predictions,
    ) = evaluate_dataset(
        "HOLDOUT PRINCIPAL — WIDTH 3",
        &test,
        &baseline_model,
        &challenger_model,
    );

    let (ood_baseline, ood_challenger, _, _, _) = evaluate_dataset(
        "HOLDOUT HORS DISTRIBUTION — WIDTH 4",
        &ood,
        &baseline_model,
        &challenger_model,
    );

    println!();
    println!(
        "Bootstrap apparié déterministe : \
         {BOOTSTRAP_REPLICATES} réplications"
    );

    let (mse_interval, mae_interval) = paired_bootstrap(
        &test_targets,
        &test_baseline_predictions,
        &test_challenger_predictions,
    );

    print_interval("IC 95 % amélioration MSE TDI-2", mse_interval);

    print_interval("IC 95 % amélioration MAE TDI-2", mae_interval);

    let observed_mse_improvement = test_baseline.mse - test_challenger.mse;

    let relative_mse_reduction = if test_baseline.mse <= 0.0
    {
        0.0
    }
    else
    {
        observed_mse_improvement / test_baseline.mse
    };

    let primary_success =
        relative_mse_reduction >= 0.05 && mse_interval.lower > 0.0 && mae_interval.lower > 0.0;

    let ood_improvement = ood_baseline.mse - ood_challenger.mse;

    let ood_confirmation = ood_improvement > 0.0;

    println!();
    println!(
        "CRITÈRE PRINCIPAL TDI-2 CONTINU : {}",
        if primary_success
        {
            "RÉUSSI"
        }
        else
        {
            "ÉCHOUÉ"
        }
    );

    println!(
        "CONFIRMATION HORS DISTRIBUTION  : {}",
        if ood_confirmation
        {
            "RÉUSSIE"
        }
        else
        {
            "ÉCHOUÉE"
        }
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Record, average_ranks, calculate_metrics, fit_ridge, generate_successor_masks, predictions,
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
    }

    #[test]
    fn feature_layout_has_preregistered_lengths() {
        let record = Record {
            baseline: [0.0; 12],
            tdi: [0.0; 3],
            target: 0.5,
        };

        assert_eq!(record.baseline.len(), 12);
        assert_eq!(record.tdi.len(), 3);
    }
}
