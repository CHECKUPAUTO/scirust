//! Third program, direction D — the contamination-frontier law.
//!
//! The program has repeatedly found that robustness helps *sometimes*: it wins on
//! heavy-tailed channels and ties (or loses a little) on light-tailed ones. Direction
//! D turns that pattern into a **falsifiable law** and tests it:
//!
//! > A robust fit (Huber IRLS) beats OLS on the **bulk** — the *median* absolute
//! > error, the central data rather than the tails — **if and only if** the OLS
//! > residuals' **excess kurtosis** exceeds a threshold `τ*`.
//!
//! The mechanism is precise: Huber down-weights outliers, so it *always* helps the
//! tail (mean absolute error) a little; but it only helps the **bulk** once the
//! outliers are numerous/extreme enough to have distorted the central OLS fit — and
//! excess kurtosis is exactly the fourth-moment measure of that heavy-tailedness. The
//! law says a *single, pre-measurable scalar* (kurtosis, computable from OLS alone,
//! before ever fitting a robust model) predicts the sign of the robust bulk gain.
//!
//! Tested across the **12 real OBD2 sensor channels**, each used in turn as a
//! regression target on the other 11 — twelve genuine industrial regression tasks
//! spanning a wide kurtosis range. For each: fit OLS + Huber on a train split, take
//! the OLS-residual excess kurtosis as the diagnostic, and the median-absolute-error
//! reduction on held-out segments as the bulk payoff. The law is then checked for a
//! separating threshold and reported as-is — a clean split confirms it, a violation
//! falsifies it. Deterministic; run twice, byte-identical.

use std::fs;
use std::path::{Path, PathBuf};

use scirust_bench_schema::{BenchRecord, to_jsonl};
use scirust_learning::{
    LinearRegressionModel, RegressionDataset, RobustLoss, RobustRegressionConfig,
    RobustRegressionMethod, fit_robust_regression,
};
use scirust_solvers::linalg::Matrix as SolversMatrix;
use scirust_srcc_bench::{mean_absolute_error, median_absolute_error, parse_obd2, sha256_hex};

const OBD2_SHA: &str = "229ef4488a89b62be530acce36ec7522421e7b82b1b5279022ffac72f0bb0751";

/// Every OBD2 sensor channel, each used in turn as the regression target on the
/// other eleven. Order fixed for determinism.
const OBD2_CHANNELS: [&str; 12] = [
    "RPM",
    "SPEED",
    "THROTTLE_POS",
    "MAF",
    "COOLANT_TEMP",
    "INTAKE_TEMP",
    "O2_B1S1",
    "ENGINE_LOAD",
    "INTAKE_PRESSURE",
    "O2_B1S2",
    "SHORT_FUEL_TRIM_1",
    "LONG_FUEL_TRIM_1",
];

/// Standard Huber threshold (≈95 % Gaussian efficiency), in robust-scale units.
const HUBER_DELTA: f64 = 1.345;
/// Floor for a standard deviation before dividing.
const SD_FLOOR: f64 = 1.0e-12;
/// Number of trailing segments (by id) held out as the test split.
const TEST_SEGMENTS: usize = 2;
/// Default proper-training decimation: `1` means full training (the committed
/// run). `--train-stride N` keeps every Nth training row for a faster pilot; the
/// verdict is stride-stable (the law is falsified at full training and at 4×).
const DEFAULT_TRAIN_STRIDE: usize = 1;

fn read_verified(path: &Path, expected_sha: &str) -> String {
    let bytes = fs::read(path).unwrap_or_else(|error| {
        panic!(
            "cannot read {}: {error}. The OBD2 telemetry ships in-repo under \
examples/obd2_diagnostic/data/.",
            path.display()
        )
    });

    let actual = sha256_hex(&bytes);

    assert!(
        actual == expected_sha,
        "checksum mismatch for {}: expected {expected_sha}, found {actual}",
        path.display()
    );

    String::from_utf8(bytes).expect("dataset file is valid UTF-8")
}

/// Train-fitted per-column mean and (floored) standard deviation.
struct Standardizer {
    means: Vec<f64>,
    sds: Vec<f64>,
}

impl Standardizer {
    fn fit(features: &[Vec<f64>]) -> Self {
        let rows = features.len();
        let cols = features.first().map_or(0, Vec::len);
        let mut means = vec![0.0; cols];

        for row in features
        {
            for (accumulator, value) in means.iter_mut().zip(row)
            {
                *accumulator += value;
            }
        }

        for mean in &mut means
        {
            *mean /= rows as f64;
        }

        let mut variances = vec![0.0; cols];

        for row in features
        {
            for (index, value) in row.iter().enumerate()
            {
                let delta = value - means[index];
                variances[index] += delta * delta;
            }
        }

        let sds: Vec<f64> = variances
            .iter()
            .map(|variance| (variance / rows as f64).sqrt().max(SD_FLOOR))
            .collect();

        Self { means, sds }
    }

    fn apply(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        features
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(index, value)| (value - self.means[index]) / self.sds[index])
                    .collect()
            })
            .collect()
    }
}

fn solvers_matrix(features: &[Vec<f64>]) -> SolversMatrix {
    let rows = features.len();
    let cols = features.first().map_or(0, Vec::len);
    let mut data = Vec::with_capacity(rows * cols);

    for row in features
    {
        data.extend_from_slice(row);
    }

    SolversMatrix::from_row_major(rows, cols, data)
}

fn dataset(features: &[Vec<f64>], targets: &[f64]) -> RegressionDataset {
    RegressionDataset {
        features: solvers_matrix(features),
        targets: SolversMatrix::from_row_major(targets.len(), 1, targets.to_vec()),
        sample_weights: None,
    }
}

/// Column of single-output predictions for `queries` under `model`.
fn predict_all(model: &LinearRegressionModel, queries: &SolversMatrix, rows: usize) -> Vec<f64> {
    let predictions = model.predict(queries).expect("prediction shapes match");
    (0..rows).map(|row| predictions[(row, 0)]).collect()
}

/// Fits OLS on `(features, targets)`.
fn fit_ols(features: &[Vec<f64>], targets: &[f64]) -> LinearRegressionModel {
    let config = RobustRegressionConfig {
        method: RobustRegressionMethod::OrdinaryLeastSquares,
        ..RobustRegressionConfig::default()
    };
    fit_robust_regression(&dataset(features, targets), config)
        .expect("OLS fits the standardized OBD2 design")
        .model
}

/// Fits Huber IRLS on `(features, targets)`.
fn fit_huber(features: &[Vec<f64>], targets: &[f64]) -> LinearRegressionModel {
    let config = RobustRegressionConfig {
        method: RobustRegressionMethod::IterativelyReweightedLeastSquares,
        loss: RobustLoss::Huber { delta: HUBER_DELTA },
        ..RobustRegressionConfig::default()
    };
    fit_robust_regression(&dataset(features, targets), config)
        .expect("Huber IRLS fits the standardized OBD2 design")
        .model
}

fn distinct_sorted(groups: &[u64]) -> Vec<u64> {
    let mut distinct: Vec<u64> = Vec::new();

    for &group in groups
    {
        if !distinct.contains(&group)
        {
            distinct.push(group);
        }
    }

    distinct.sort_unstable();
    distinct
}

/// Keeps every `stride`-th element of `rows`. `stride == 1` is the identity.
fn decimate(rows: &[usize], stride: usize) -> Vec<usize> {
    rows.iter()
        .enumerate()
        .filter(|(position, _)| position % stride == 0)
        .map(|(_, &row)| row)
        .collect()
}

fn rows_features(data_features: &[Vec<f64>], rows: &[usize]) -> Vec<Vec<f64>> {
    rows.iter().map(|&row| data_features[row].clone()).collect()
}

/// Population excess kurtosis `m₄ / m₂² − 3` (0 for a Gaussian). Empty or
/// zero-variance input yields `0.0`.
fn excess_kurtosis(values: &[f64]) -> f64 {
    let count = values.len();
    if count == 0
    {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / count as f64;
    let second = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / count as f64;
    let fourth = values
        .iter()
        .map(|value| (value - mean).powi(4))
        .sum::<f64>()
        / count as f64;

    if second <= 0.0
    {
        return 0.0;
    }

    fourth / (second * second) - 3.0
}

/// Average ranks (1-based, ties averaged), ordered by `f64::total_cmp`.
fn ranks(values: &[f64]) -> Vec<f64> {
    let count = values.len();
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));

    let mut result = vec![0.0; count];
    let mut start = 0;

    while start < count
    {
        let mut end = start;
        while end + 1 < count
            && values[order[end + 1]].total_cmp(&values[order[start]]) == std::cmp::Ordering::Equal
        {
            end += 1;
        }

        let average_rank = (start + end) as f64 / 2.0 + 1.0;
        for position in start..=end
        {
            result[order[position]] = average_rank;
        }

        start = end + 1;
    }

    result
}

/// Pearson correlation of two equal-length vectors (`0.0` if either is constant).
fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let count = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / count;
    let mean_y = ys.iter().sum::<f64>() / count;

    let mut covariance = 0.0;
    let mut variance_x = 0.0;
    let mut variance_y = 0.0;

    for (x, y) in xs.iter().zip(ys)
    {
        covariance += (x - mean_x) * (y - mean_y);
        variance_x += (x - mean_x).powi(2);
        variance_y += (y - mean_y).powi(2);
    }

    if variance_x <= 0.0 || variance_y <= 0.0
    {
        return 0.0;
    }

    covariance / (variance_x.sqrt() * variance_y.sqrt())
}

/// Spearman rank correlation.
fn spearman(xs: &[f64], ys: &[f64]) -> f64 {
    pearson(&ranks(xs), &ranks(ys))
}

/// One channel's measured frontier point.
struct FrontierPoint {
    channel: &'static str,
    excess_kurtosis: f64,
    /// Absolute median-AE reduction (target units) — sign is comparable across
    /// channels, magnitude is not.
    bulk_improvement: f64,
    /// Scale-free median-AE reduction as a fraction of OLS's median AE — the
    /// cross-channel-comparable magnitude the frontier correlation uses.
    relative_bulk_improvement: f64,
    tail_improvement: f64,
    median_ae_ols: f64,
    median_ae_huber: f64,
}

/// The best sign-separating kurtosis threshold and how well it separates.
struct Verdict {
    threshold: Option<f64>,
    separable: bool,
    violations: usize,
    spearman: f64,
}

/// Finds the kurtosis threshold that best separates positive-bulk-gain channels
/// from the rest, and counts the channels it misclassifies.
fn assess_law(points: &[FrontierPoint]) -> Verdict {
    let kurtoses: Vec<f64> = points.iter().map(|point| point.excess_kurtosis).collect();
    // Correlate against the scale-free relative gain (absolute gains are in
    // per-channel target units and not comparable across channels).
    let improvements: Vec<f64> = points
        .iter()
        .map(|point| point.relative_bulk_improvement)
        .collect();
    let spearman_value = spearman(&kurtoses, &improvements);

    // Candidate thresholds: each observed kurtosis, plus ±∞ sentinels. A channel is
    // classified "robust wins" when its kurtosis exceeds the threshold; count how
    // many channels that mislabels against the observed sign of the bulk gain.
    let mut candidates: Vec<f64> = kurtoses.clone();
    candidates.push(f64::NEG_INFINITY);
    candidates.sort_by(f64::total_cmp);

    let mut best_threshold = candidates[0];
    let mut best_violations = usize::MAX;

    for &threshold in &candidates
    {
        let violations = points
            .iter()
            .filter(|point| {
                let predicted_win = point.excess_kurtosis > threshold;
                let actual_win = point.bulk_improvement > 0.0;
                predicted_win != actual_win
            })
            .count();

        if violations < best_violations
        {
            best_violations = violations;
            best_threshold = threshold;
        }
    }

    // Report the threshold as the midpoint of the separating gap when clean.
    let separable = best_violations == 0;
    let threshold = if separable
    {
        let highest_negative = points
            .iter()
            .filter(|point| point.bulk_improvement <= 0.0)
            .map(|point| point.excess_kurtosis)
            .fold(f64::NEG_INFINITY, f64::max);
        let lowest_positive = points
            .iter()
            .filter(|point| point.bulk_improvement > 0.0)
            .map(|point| point.excess_kurtosis)
            .fold(f64::INFINITY, f64::min);

        match (highest_negative.is_finite(), lowest_positive.is_finite())
        {
            (true, true) => Some((highest_negative + lowest_positive) / 2.0),
            _ => Some(best_threshold),
        }
    }
    else
    {
        None
    };

    Verdict {
        threshold,
        separable,
        violations: best_violations,
        spearman: spearman_value,
    }
}

fn main() {
    let mut data_path = PathBuf::from("examples/obd2_diagnostic/data/opel_corsa_telemetry.csv");
    let mut out_dir = PathBuf::from("results");
    let mut train_stride = DEFAULT_TRAIN_STRIDE;

    let mut arguments = std::env::args().skip(1);

    while let Some(argument) = arguments.next()
    {
        let mut value = |name: &str| {
            arguments
                .next()
                .unwrap_or_else(|| panic!("{name} requires a value"))
        };

        match argument.as_str()
        {
            "--data" => data_path = PathBuf::from(value("--data")),
            "--out" => out_dir = PathBuf::from(value("--out")),
            "--train-stride" =>
            {
                train_stride = value("--train-stride")
                    .parse()
                    .expect("--train-stride is a positive integer");
                assert!(train_stride >= 1, "--train-stride must be at least 1");
            },
            other => panic!("unknown argument: {other}"),
        }
    }

    let text = read_verified(&data_path, OBD2_SHA);
    let mut records: Vec<BenchRecord> = Vec::new();
    let mut points: Vec<FrontierPoint> = Vec::new();

    println!("# obd2_contamination_frontier — Direction D: the contamination-frontier law");
    println!(
        "# claim: Huber beats OLS on the bulk (median abs. error) iff OLS-residual excess \
kurtosis > τ*"
    );
    let training_note = if train_stride == 1
    {
        "full proper-training".to_string()
    }
    else
    {
        format!("proper-training decimated {train_stride}x")
    };
    println!(
        "# 12 OBD2 channels as targets; Huber δ={HUBER_DELTA}; last {TEST_SEGMENTS} segments \
test; {training_note}"
    );

    for channel in OBD2_CHANNELS
    {
        let data = parse_obd2(&text, channel).expect("OBD2 parses for the target channel");
        let groups = data.groups.as_ref().expect("OBD2 carries segment groups");
        let segments = distinct_sorted(groups);
        let rows = data.targets.len();

        let test_ids: Vec<u64> = segments.iter().rev().take(TEST_SEGMENTS).copied().collect();
        let is_test = |row: usize| test_ids.contains(&groups[row]);

        let train_all: Vec<usize> = (0..rows).filter(|&row| !is_test(row)).collect();
        let test_rows: Vec<usize> = (0..rows).filter(|&row| is_test(row)).collect();
        let proper_rows = decimate(&train_all, train_stride);

        let proper_features_raw = rows_features(&data.features, &proper_rows);
        let proper_targets: Vec<f64> = proper_rows.iter().map(|&row| data.targets[row]).collect();
        let standardizer = Standardizer::fit(&proper_features_raw);
        let proper_features = standardizer.apply(&proper_features_raw);
        let proper_matrix = solvers_matrix(&proper_features);

        let test_matrix =
            solvers_matrix(&standardizer.apply(&rows_features(&data.features, &test_rows)));
        let test_targets: Vec<f64> = test_rows.iter().map(|&row| data.targets[row]).collect();

        let ols = fit_ols(&proper_features, &proper_targets);
        let huber = fit_huber(&proper_features, &proper_targets);

        // Diagnostic: excess kurtosis of the OLS residuals on the training split.
        let ols_train = predict_all(&ols, &proper_matrix, proper_targets.len());
        let residuals: Vec<f64> = proper_targets
            .iter()
            .zip(&ols_train)
            .map(|(target, prediction)| target - prediction)
            .collect();
        let kurtosis = excess_kurtosis(&residuals);

        // Payoff: bulk (median) and tail (mean) absolute-error reduction on test.
        let ols_test = predict_all(&ols, &test_matrix, test_targets.len());
        let huber_test = predict_all(&huber, &test_matrix, test_targets.len());
        let median_ae_ols =
            median_absolute_error(&ols_test, &test_targets).expect("aligned test vectors");
        let median_ae_huber =
            median_absolute_error(&huber_test, &test_targets).expect("aligned test vectors");
        let mean_ae_ols =
            mean_absolute_error(&ols_test, &test_targets).expect("aligned test vectors");
        let mean_ae_huber =
            mean_absolute_error(&huber_test, &test_targets).expect("aligned test vectors");

        let bulk_improvement = median_ae_ols - median_ae_huber;
        let relative_bulk_improvement = if median_ae_ols > 0.0
        {
            bulk_improvement / median_ae_ols
        }
        else
        {
            0.0
        };

        let point = FrontierPoint {
            channel,
            excess_kurtosis: kurtosis,
            bulk_improvement,
            relative_bulk_improvement,
            tail_improvement: mean_ae_ols - mean_ae_huber,
            median_ae_ols,
            median_ae_huber,
        };

        for (metric, value) in [
            ("ols_residual_excess_kurtosis", point.excess_kurtosis),
            ("bulk_median_ae_improvement", point.bulk_improvement),
            ("relative_bulk_improvement", point.relative_bulk_improvement),
            ("tail_mean_ae_improvement", point.tail_improvement),
            ("median_ae_ols", point.median_ae_ols),
            ("median_ae_huber", point.median_ae_huber),
        ]
        {
            records.push(BenchRecord::new(
                "obd2_contamination_frontier/law",
                format!("obd2/{channel}"),
                "huber_vs_ols",
                0,
                metric,
                value,
            ));
        }

        points.push(point);
    }

    // Report each channel sorted by kurtosis — the frontier, ascending.
    points.sort_by(|left, right| left.excess_kurtosis.total_cmp(&right.excess_kurtosis));

    println!(
        "# channel                excess_kurt   bulk_Δ(median)   rel_bulk   tail_Δ(mean)   robust_bulk_win"
    );
    for point in &points
    {
        println!(
            "#   {:<20} {:>10.3}   {:>12.4}   {:>7.1}%   {:>12.4}   {}",
            point.channel,
            point.excess_kurtosis,
            point.bulk_improvement,
            point.relative_bulk_improvement * 100.0,
            point.tail_improvement,
            if point.bulk_improvement > 0.0
            {
                "yes"
            }
            else
            {
                "no"
            },
        );
    }

    let verdict = assess_law(&points);

    println!("# --- frontier law ---");
    println!(
        "#   Spearman(kurtosis, relative bulk gain) = {:.3}; sign-separable = {}; violations = {}/{}",
        verdict.spearman,
        verdict.separable,
        verdict.violations,
        points.len()
    );
    match verdict.threshold
    {
        Some(threshold) if verdict.separable =>
        {
            println!(
                "#   LAW HOLDS on this corpus: robust wins the bulk iff excess kurtosis > τ* ≈ {threshold:.3}"
            );
        },
        _ =>
        {
            println!(
                "#   LAW FALSIFIED on this corpus: no kurtosis threshold cleanly separates the \
bulk-gain sign ({} misclassified)",
                verdict.violations
            );
        },
    }

    for (metric, value) in [
        ("frontier_spearman", verdict.spearman),
        ("frontier_threshold", verdict.threshold.unwrap_or(f64::NAN)),
        ("frontier_violations", verdict.violations as f64),
        ("frontier_channels", points.len() as f64),
    ]
    {
        records.push(BenchRecord::new(
            "obd2_contamination_frontier/law",
            "obd2/_corpus",
            "huber_vs_ols",
            0,
            metric,
            value,
        ));
    }

    fs::create_dir_all(&out_dir).expect("results directory is writable");
    fs::write(
        out_dir.join("obd2_contamination_frontier.jsonl"),
        to_jsonl(&records),
    )
    .expect("results file is writable");

    println!("# records={}", records.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn excess_kurtosis_is_zero_for_a_flat_spread_and_high_for_a_spike() {
        // A symmetric two-value set has light tails (negative excess kurtosis).
        let flat = vec![-1.0, -1.0, 1.0, 1.0];
        assert!(excess_kurtosis(&flat) < 0.0);
        // A spike with rare large outliers is strongly leptokurtic.
        let mut spiky = vec![0.0; 100];
        spiky[0] = 30.0;
        spiky[1] = -30.0;
        assert!(excess_kurtosis(&spiky) > 5.0);
    }

    #[test]
    fn spearman_is_one_for_a_monotone_relationship() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = vec![10.0, 20.0, 25.0, 40.0, 100.0];
        assert!((spearman(&xs, &ys) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn assess_law_finds_a_clean_threshold_when_separable() {
        let make = |kurtosis: f64, gain: f64| FrontierPoint {
            channel: "x",
            excess_kurtosis: kurtosis,
            bulk_improvement: gain,
            relative_bulk_improvement: gain,
            tail_improvement: gain,
            median_ae_ols: 0.0,
            median_ae_huber: 0.0,
        };
        // Negative gains below kurtosis 5, positive gains above — cleanly separable.
        let points = vec![
            make(1.0, -0.1),
            make(3.0, -0.05),
            make(8.0, 0.2),
            make(12.0, 0.5),
        ];
        let verdict = assess_law(&points);
        assert!(verdict.separable);
        assert_eq!(verdict.violations, 0);
        let threshold = verdict
            .threshold
            .expect("a separable corpus has a threshold");
        assert!(threshold > 3.0 && threshold < 8.0);
    }

    #[test]
    fn assess_law_reports_violations_when_not_separable() {
        let make = |kurtosis: f64, gain: f64| FrontierPoint {
            channel: "x",
            excess_kurtosis: kurtosis,
            bulk_improvement: gain,
            relative_bulk_improvement: gain,
            tail_improvement: gain,
            median_ae_ols: 0.0,
            median_ae_huber: 0.0,
        };
        // A high-kurtosis loss and a low-kurtosis win break any single threshold.
        let points = vec![
            make(1.0, 0.2),
            make(3.0, -0.1),
            make(8.0, -0.2),
            make(12.0, 0.3),
        ];
        let verdict = assess_law(&points);
        assert!(!verdict.separable);
        assert!(verdict.violations >= 1);
        assert!(verdict.threshold.is_none());
    }
}
