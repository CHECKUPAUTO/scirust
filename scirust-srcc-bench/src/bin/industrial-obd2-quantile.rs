//! Third program, direction C (sub-PR 2) — native quantile-regression intervals.
//!
//! Sub-PR 1 wrapped a *point* predictor in a split-conformal band of **constant**
//! width. Quantile regression instead predicts the conditional quantiles directly
//! (`scirust_learning::fit_quantile_regression`, pinball loss): fitting `τ = 0.05`
//! and `τ = 0.95` gives a **native 90 % interval** `[q₀.₀₅(x), q₀.₉₅(x)]` whose
//! width **adapts** to the local noise. This binary asks, on the real OBD2
//! workload, whether that adaptive native band is competitive with the C.1
//! conformal band in coverage and average width.
//!
//! Same leave-one-segment-out protocol and three heavy-tailed targets. Two interval
//! methods, each used naturally: **quantile-native** fits `q₀.₀₅` and `q₀.₉₅` on the
//! training segments and reads off `[q₀.₀₅(x), q₀.₉₅(x)]`; **OLS-conformal** fits
//! OLS on a proper-training subset, calibrates the half-width on the held-out
//! calibration rows (every 5th training row), and reads off `ŷ ± q`. Both target
//! nominal 0.9 on the held-out segment.
//!
//! The honest contrast: conformal *guarantees* coverage but pays a constant width;
//! quantile regression *adapts* the width but its coverage is only as good as the
//! estimated quantiles (no finite-sample guarantee — that is what sub-PR 3's
//! conformalized quantile regression will restore). Both coverage and width are
//! reported as-is; quantile crossings (`q₀.₀₅ > q₀.₉₅`) are counted, not hidden.
//! Deterministic; run twice, byte-identical.

use std::fs;
use std::path::{Path, PathBuf};

use scirust_bench_schema::{BenchRecord, to_jsonl};
use scirust_learning::{
    QuantileRegressionConfig, RegressionDataset, RobustRegressionConfig, RobustRegressionMethod,
    fit_quantile_regression, fit_robust_regression,
};
use scirust_solvers::linalg::Matrix as SolversMatrix;
use scirust_srcc_bench::{SplitConformal, parse_obd2, sha256_hex};

const OBD2_SHA: &str = "229ef4488a89b62be530acce36ec7522421e7b82b1b5279022ffac72f0bb0751";

/// Real engine channels used as regression targets (heaviest native tails first).
const TARGETS: [&str; 3] = ["ENGINE_LOAD", "THROTTLE_POS", "MAF"];

/// Nominal interval coverage and its two-sided quantiles.
const LEVEL: f64 = 0.9;
const TAU_LOW: f64 = 0.05;
const TAU_HIGH: f64 = 0.95;
/// Every `CALIBRATION_STRIDE`-th training row calibrates the OLS-conformal band.
const CALIBRATION_STRIDE: usize = 5;
/// Floor for a standard deviation before dividing.
const SD_FLOOR: f64 = 1.0e-12;

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

/// OLS predictions for `queries` from a model fit on `(features, targets)`.
fn ols_predict(
    features: &[Vec<f64>],
    targets: &[f64],
    queries: &SolversMatrix,
    query_rows: usize,
) -> Vec<f64> {
    let config = RobustRegressionConfig {
        method: RobustRegressionMethod::OrdinaryLeastSquares,
        ..RobustRegressionConfig::default()
    };
    let report = fit_robust_regression(&dataset(features, targets), config)
        .expect("OLS fits the standardized OBD2 design");
    let predictions = report
        .model
        .predict(queries)
        .expect("prediction shapes match");
    (0..query_rows).map(|row| predictions[(row, 0)]).collect()
}

/// Quantile-`tau` predictions for `queries` from a model fit on `(features, targets)`.
fn quantile_predict(
    tau: f64,
    features: &[Vec<f64>],
    targets: &[f64],
    queries: &SolversMatrix,
    query_rows: usize,
) -> Vec<f64> {
    let report = fit_quantile_regression(
        &dataset(features, targets),
        QuantileRegressionConfig::new(tau),
    )
    .expect("quantile regression fits the standardized OBD2 design");
    let predictions = report
        .model
        .predict(queries)
        .expect("prediction shapes match");
    (0..query_rows).map(|row| predictions[(row, 0)]).collect()
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

/// Splits `train_rows` into (calibration, proper-training): every `stride`-th row
/// calibrates; the rest train the point predictor.
fn split_calibration(train_rows: &[usize], stride: usize) -> (Vec<usize>, Vec<usize>) {
    let mut calibration = Vec::new();
    let mut proper = Vec::new();

    for (position, &row) in train_rows.iter().enumerate()
    {
        if position % stride == 0
        {
            calibration.push(row);
        }
        else
        {
            proper.push(row);
        }
    }

    (calibration, proper)
}

/// Pooled coverage/width accumulator for one interval method across folds.
#[derive(Default)]
struct IntervalAccumulator {
    covered: usize,
    total: usize,
    width_sum: f64,
    crossings: usize,
}

impl IntervalAccumulator {
    fn add(&mut self, lower: f64, upper: f64, actual: f64) {
        if lower > upper
        {
            self.crossings += 1;
        }

        if lower <= actual && actual <= upper
        {
            self.covered += 1;
        }

        self.total += 1;
        self.width_sum += upper - lower;
    }

    fn coverage(&self) -> f64 {
        self.covered as f64 / self.total as f64
    }

    fn mean_width(&self) -> f64 {
        self.width_sum / self.total as f64
    }
}

fn rows_features(data_features: &[Vec<f64>], rows: &[usize]) -> Vec<Vec<f64>> {
    rows.iter().map(|&row| data_features[row].clone()).collect()
}

fn main() {
    let mut data_path = PathBuf::from("examples/obd2_diagnostic/data/opel_corsa_telemetry.csv");
    let mut out_dir = PathBuf::from("results");

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
            other => panic!("unknown argument: {other}"),
        }
    }

    let text = read_verified(&data_path, OBD2_SHA);
    let mut records: Vec<BenchRecord> = Vec::new();

    println!(
        "# obd2_quantile — Direction C.2: native quantile intervals vs conformal (level {LEVEL})"
    );
    println!(
        "# real OBD2 leave-one-segment-out; quantile-native [q{TAU_LOW},q{TAU_HIGH}] on full train \
vs OLS split-conformal (calibration every {CALIBRATION_STRIDE}th train row)"
    );

    for target in TARGETS
    {
        let data = parse_obd2(&text, target).expect("OBD2 parses for the target channel");
        let groups = data.groups.as_ref().expect("OBD2 carries segment groups");
        let segments = distinct_sorted(groups);
        let rows = data.targets.len();

        let mut quantile = IntervalAccumulator::default();
        let mut conformal = IntervalAccumulator::default();

        for &held in &segments
        {
            let train_rows: Vec<usize> = (0..rows).filter(|&row| groups[row] != held).collect();
            let test_rows: Vec<usize> = (0..rows).filter(|&row| groups[row] == held).collect();

            let test_targets: Vec<f64> = test_rows.iter().map(|&row| data.targets[row]).collect();

            // --- quantile-native: fit q05, q95 on the full training segments ---
            let train_features_raw = rows_features(&data.features, &train_rows);
            let train_targets: Vec<f64> = train_rows.iter().map(|&row| data.targets[row]).collect();
            let quantile_standardizer = Standardizer::fit(&train_features_raw);
            let quantile_train = quantile_standardizer.apply(&train_features_raw);
            let quantile_test =
                quantile_standardizer.apply(&rows_features(&data.features, &test_rows));
            let quantile_test_matrix = solvers_matrix(&quantile_test);

            let low = quantile_predict(
                TAU_LOW,
                &quantile_train,
                &train_targets,
                &quantile_test_matrix,
                test_targets.len(),
            );
            let high = quantile_predict(
                TAU_HIGH,
                &quantile_train,
                &train_targets,
                &quantile_test_matrix,
                test_targets.len(),
            );

            for ((lower, upper), actual) in low.iter().zip(&high).zip(&test_targets)
            {
                quantile.add(*lower, *upper, *actual);
            }

            // --- OLS split-conformal: fit on proper-train, calibrate on the rest ---
            let (calibration_rows, proper_rows) =
                split_calibration(&train_rows, CALIBRATION_STRIDE);
            let proper_features_raw = rows_features(&data.features, &proper_rows);
            let proper_targets: Vec<f64> =
                proper_rows.iter().map(|&row| data.targets[row]).collect();
            let conformal_standardizer = Standardizer::fit(&proper_features_raw);
            let proper_features = conformal_standardizer.apply(&proper_features_raw);

            let calibration_features =
                conformal_standardizer.apply(&rows_features(&data.features, &calibration_rows));
            let calibration_targets: Vec<f64> = calibration_rows
                .iter()
                .map(|&row| data.targets[row])
                .collect();
            let calibration_predictions = ols_predict(
                &proper_features,
                &proper_targets,
                &solvers_matrix(&calibration_features),
                calibration_targets.len(),
            );
            let calibration_residuals: Vec<f64> = calibration_predictions
                .iter()
                .zip(&calibration_targets)
                .map(|(prediction, actual)| actual - prediction)
                .collect();
            let band = SplitConformal::fit(&calibration_residuals, LEVEL)
                .expect("calibration set is large enough at level 0.9");

            let conformal_test =
                conformal_standardizer.apply(&rows_features(&data.features, &test_rows));
            let conformal_predictions = ols_predict(
                &proper_features,
                &proper_targets,
                &solvers_matrix(&conformal_test),
                test_targets.len(),
            );

            for (prediction, actual) in conformal_predictions.iter().zip(&test_targets)
            {
                let (lower, upper) = band.interval(*prediction);
                conformal.add(lower, upper, *actual);
            }
        }

        for (method, accumulator) in [
            ("quantile_native", &quantile),
            ("ols_conformal", &conformal),
        ]
        {
            for (metric, value) in [
                ("empirical_coverage", accumulator.coverage()),
                ("mean_interval_width", accumulator.mean_width()),
                ("quantile_crossings", accumulator.crossings as f64),
            ]
            {
                records.push(BenchRecord::new(
                    "obd2_quantile/interval",
                    format!("obd2/{target}"),
                    method,
                    0,
                    metric,
                    value,
                ));
            }
        }

        println!("# {target} (nominal coverage {LEVEL}):");
        println!(
            "#   quantile-native coverage={:.4} mean_width={:.4} crossings={}",
            quantile.coverage(),
            quantile.mean_width(),
            quantile.crossings
        );
        println!(
            "#   OLS-conformal   coverage={:.4} mean_width={:.4}",
            conformal.coverage(),
            conformal.mean_width()
        );
    }

    fs::create_dir_all(&out_dir).expect("results directory is writable");
    fs::write(out_dir.join("obd2_quantile.jsonl"), to_jsonl(&records))
        .expect("results file is writable");

    println!("# records={}", records.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_accumulator_counts_coverage_width_and_crossings() {
        let mut accumulator = IntervalAccumulator::default();
        accumulator.add(0.0, 2.0, 1.0); // covered, width 2
        accumulator.add(0.0, 2.0, 3.0); // not covered
        accumulator.add(2.0, 1.0, 5.0); // crossed (lower > upper), not covered, width -1
        assert_eq!(accumulator.total, 3);
        assert_eq!(accumulator.covered, 1);
        assert_eq!(accumulator.crossings, 1);
        assert!((accumulator.coverage() - 1.0 / 3.0).abs() < 1e-12);
        assert!((accumulator.mean_width() - 1.0).abs() < 1e-12); // (2 + 2 - 1) / 3 = 1.0
    }

    #[test]
    fn split_calibration_places_every_row_once() {
        let train_rows: Vec<usize> = (0..10).collect();
        let (calibration, proper) = split_calibration(&train_rows, 5);
        assert_eq!(calibration, vec![0, 5]);
        assert_eq!(calibration.len() + proper.len(), train_rows.len());
    }
}
