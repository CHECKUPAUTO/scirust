//! Third program, direction C (sub-PR 3) — conformalized quantile regression.
//!
//! Sub-PR 1 gave a conformal band with **guaranteed** coverage but **constant**
//! width; sub-PR 2 gave a native quantile band with **adaptive** width but **no**
//! coverage guarantee (and quantile crossings). Conformalized quantile regression
//! (CQR; Romano, Patterson & Candès, 2019) unites them: fit `q₀.₀₅` and `q₀.₉₅`,
//! then a single conformal offset `Q` from a calibration set adjusts the interval
//! to `[q₀.₀₅(x) − Q, q₀.₉₅(x) + Q]`, restoring the finite-sample coverage guarantee
//! **while keeping the adaptive shape**. `Q` can be negative — a too-wide native
//! band is tightened.
//!
//! On the real OBD2 workload (leave-one-segment-out, three heavy-tailed targets),
//! this compares three intervals at level 0.9 on the **same** proper-training /
//! calibration split: **quantile-native** `[q₀.₀₅, q₀.₉₅]`, **CQR** (native shifted
//! by `Q`), and **OLS-conformal** (the constant-width C.1 band). The expectation
//! CQR is built to meet: match conformal's guaranteed coverage at (or below) the
//! native adaptive width — dominating both halves. Whether it does on this data is
//! reported as-is. Deterministic; run twice, byte-identical.

use std::fs;
use std::path::{Path, PathBuf};

use scirust_bench_schema::{BenchRecord, to_jsonl};
use scirust_learning::{
    LinearRegressionModel, QuantileRegressionConfig, RegressionDataset, RobustRegressionConfig,
    RobustRegressionMethod, fit_quantile_regression, fit_robust_regression,
};
use scirust_solvers::linalg::Matrix as SolversMatrix;
use scirust_srcc_bench::{ConformalizedQuantile, SplitConformal, parse_obd2, sha256_hex};

const OBD2_SHA: &str = "229ef4488a89b62be530acce36ec7522421e7b82b1b5279022ffac72f0bb0751";

/// Real engine channels used as regression targets (heaviest native tails first).
const TARGETS: [&str; 3] = ["ENGINE_LOAD", "THROTTLE_POS", "MAF"];

/// Nominal interval coverage and its two-sided quantiles.
const LEVEL: f64 = 0.9;
const TAU_LOW: f64 = 0.05;
const TAU_HIGH: f64 = 0.95;
/// Every `CALIBRATION_STRIDE`-th training row is held out to calibrate.
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

/// Column of single-output predictions for `queries` under `model`.
fn predict_all(model: &LinearRegressionModel, queries: &SolversMatrix, rows: usize) -> Vec<f64> {
    let predictions = model.predict(queries).expect("prediction shapes match");
    (0..rows).map(|row| predictions[(row, 0)]).collect()
}

/// Fits a quantile-`tau` model on `(features, targets)`.
fn fit_quantile(tau: f64, features: &[Vec<f64>], targets: &[f64]) -> LinearRegressionModel {
    fit_quantile_regression(
        &dataset(features, targets),
        QuantileRegressionConfig::new(tau),
    )
    .expect("quantile regression fits the standardized OBD2 design")
    .model
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
/// calibrates; the rest train the predictors.
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
}

impl IntervalAccumulator {
    fn add(&mut self, lower: f64, upper: f64, actual: f64) {
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

    println!("# obd2_cqr — Direction C.3: conformalized quantile regression (level {LEVEL})");
    println!(
        "# real OBD2 leave-one-segment-out; quantile-native vs CQR vs OLS-conformal on the same \
proper/calibration split (every {CALIBRATION_STRIDE}th train row calibrates)"
    );

    for target in TARGETS
    {
        let data = parse_obd2(&text, target).expect("OBD2 parses for the target channel");
        let groups = data.groups.as_ref().expect("OBD2 carries segment groups");
        let segments = distinct_sorted(groups);
        let rows = data.targets.len();

        let mut native = IntervalAccumulator::default();
        let mut cqr = IntervalAccumulator::default();
        let mut conformal = IntervalAccumulator::default();

        for &held in &segments
        {
            let train_rows: Vec<usize> = (0..rows).filter(|&row| groups[row] != held).collect();
            let test_rows: Vec<usize> = (0..rows).filter(|&row| groups[row] == held).collect();
            let (calibration_rows, proper_rows) =
                split_calibration(&train_rows, CALIBRATION_STRIDE);

            // One standardizer per fold, fit on proper-training rows.
            let proper_features_raw = rows_features(&data.features, &proper_rows);
            let proper_targets: Vec<f64> =
                proper_rows.iter().map(|&row| data.targets[row]).collect();
            let standardizer = Standardizer::fit(&proper_features_raw);
            let proper_features = standardizer.apply(&proper_features_raw);

            let calibration_matrix = solvers_matrix(
                &standardizer.apply(&rows_features(&data.features, &calibration_rows)),
            );
            let calibration_targets: Vec<f64> = calibration_rows
                .iter()
                .map(|&row| data.targets[row])
                .collect();
            let test_matrix =
                solvers_matrix(&standardizer.apply(&rows_features(&data.features, &test_rows)));
            let test_targets: Vec<f64> = test_rows.iter().map(|&row| data.targets[row]).collect();

            // Quantile predictors fit once on proper-train, used on both splits.
            let low_model = fit_quantile(TAU_LOW, &proper_features, &proper_targets);
            let high_model = fit_quantile(TAU_HIGH, &proper_features, &proper_targets);

            let low_cal = predict_all(&low_model, &calibration_matrix, calibration_targets.len());
            let high_cal = predict_all(&high_model, &calibration_matrix, calibration_targets.len());
            let low_test = predict_all(&low_model, &test_matrix, test_targets.len());
            let high_test = predict_all(&high_model, &test_matrix, test_targets.len());

            let band = ConformalizedQuantile::fit(&low_cal, &high_cal, &calibration_targets, LEVEL)
                .expect("calibration set is large enough at level 0.9");

            // OLS split-conformal on the same proper/calibration split.
            let ols_model = fit_ols(&proper_features, &proper_targets);
            let ols_cal = predict_all(&ols_model, &calibration_matrix, calibration_targets.len());
            let ols_residuals: Vec<f64> = ols_cal
                .iter()
                .zip(&calibration_targets)
                .map(|(prediction, actual)| actual - prediction)
                .collect();
            let ols_band = SplitConformal::fit(&ols_residuals, LEVEL)
                .expect("calibration set is large enough at level 0.9");
            let ols_test = predict_all(&ols_model, &test_matrix, test_targets.len());

            for index in 0..test_targets.len()
            {
                let actual = test_targets[index];
                native.add(low_test[index], high_test[index], actual);
                let (cqr_low, cqr_high) = band.interval(low_test[index], high_test[index]);
                cqr.add(cqr_low, cqr_high, actual);
                let (ols_low, ols_high) = ols_band.interval(ols_test[index]);
                conformal.add(ols_low, ols_high, actual);
            }
        }

        for (method, accumulator) in [
            ("quantile_native", &native),
            ("cqr", &cqr),
            ("ols_conformal", &conformal),
        ]
        {
            for (metric, value) in [
                ("empirical_coverage", accumulator.coverage()),
                ("mean_interval_width", accumulator.mean_width()),
            ]
            {
                records.push(BenchRecord::new(
                    "obd2_cqr/interval",
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
            "#   quantile-native coverage={:.4} mean_width={:.4}",
            native.coverage(),
            native.mean_width()
        );
        println!(
            "#   CQR             coverage={:.4} mean_width={:.4}",
            cqr.coverage(),
            cqr.mean_width()
        );
        println!(
            "#   OLS-conformal   coverage={:.4} mean_width={:.4}",
            conformal.coverage(),
            conformal.mean_width()
        );
    }

    fs::create_dir_all(&out_dir).expect("results directory is writable");
    fs::write(out_dir.join("obd2_cqr.jsonl"), to_jsonl(&records))
        .expect("results file is writable");

    println!("# records={}", records.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_accumulator_tracks_coverage_and_width() {
        let mut accumulator = IntervalAccumulator::default();
        accumulator.add(0.0, 2.0, 1.0); // covered, width 2
        accumulator.add(0.0, 2.0, 5.0); // not covered, width 2
        assert_eq!(accumulator.total, 2);
        assert_eq!(accumulator.covered, 1);
        assert!((accumulator.coverage() - 0.5).abs() < 1e-12);
        assert!((accumulator.mean_width() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn split_calibration_places_every_row_once() {
        let train_rows: Vec<usize> = (0..10).collect();
        let (calibration, proper) = split_calibration(&train_rows, 5);
        assert_eq!(calibration, vec![0, 5]);
        assert_eq!(calibration.len() + proper.len(), train_rows.len());
    }
}
