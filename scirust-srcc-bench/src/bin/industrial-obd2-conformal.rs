//! Third program, direction C (sub-PR 1) — split-conformal intervals on real data.
//!
//! Axis 3 showed a *robust* point predictor (Huber) tracks the bulk of real
//! heavy-tailed OBD2 residuals better than OLS. This binary asks the uncertainty
//! question that follows: turned into **prediction intervals** with a
//! distribution-free split-conformal band ([`scirust_srcc_bench::SplitConformal`]),
//! does the robust predictor give **tighter valid** intervals than OLS?
//!
//! Protocol mirrors axis 3 — leave-one-segment-out over the seven OBD2 driving
//! segments, three heavy-tailed targets, features standardized on the fit rows. In
//! each fold the six training segments are split deterministically into a
//! **proper-training** set (the point predictor is fit here) and a **calibration**
//! set (every `CALIBRATION_STRIDE`-th training row); the conformal half-width is the
//! `⌈(n+1)·0.9⌉`-th smallest absolute calibration residual, and the held-out
//! segment is scored for empirical coverage and mean interval width. OLS and Huber
//! use the *same* split, so the head-to-head is exact.
//!
//! Note this is conformal under a mild **distribution shift** (calibrate on some
//! segments, deploy on a held-out one) — the realistic deployment condition, not
//! the i.i.d. ideal — so empirical coverage may sit a little under the nominal 0.9;
//! what the comparison isolates is whether robustness buys a tighter band at
//! matched coverage. Reported as-is. Deterministic; run twice, byte-identical.

use std::fs;
use std::path::{Path, PathBuf};

use scirust_bench_schema::{BenchRecord, to_jsonl};
use scirust_learning::{
    RegressionDataset, RobustLoss, RobustRegressionConfig, RobustRegressionMethod,
    fit_robust_regression,
};
use scirust_solvers::linalg::Matrix as SolversMatrix;
use scirust_srcc_bench::{SplitConformal, parse_obd2, sha256_hex};

const OBD2_SHA: &str = "229ef4488a89b62be530acce36ec7522421e7b82b1b5279022ffac72f0bb0751";

/// Real engine channels used as regression targets (heaviest native tails first).
const TARGETS: [&str; 3] = ["ENGINE_LOAD", "THROTTLE_POS", "MAF"];

/// Nominal conformal coverage level.
const LEVEL: f64 = 0.9;
/// Every `CALIBRATION_STRIDE`-th training row becomes a calibration point.
const CALIBRATION_STRIDE: usize = 5;
/// Huber transition point in MAD-scale units.
const HUBER_DELTA: f64 = 1.345;
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

fn method_config(huber: bool) -> RobustRegressionConfig {
    let base = RobustRegressionConfig::default();

    if huber
    {
        RobustRegressionConfig {
            method: RobustRegressionMethod::IterativelyReweightedLeastSquares,
            loss: RobustLoss::Huber { delta: HUBER_DELTA },
            ..base
        }
    }
    else
    {
        RobustRegressionConfig {
            method: RobustRegressionMethod::OrdinaryLeastSquares,
            ..base
        }
    }
}

/// Fits `huber`/OLS on `(train_features, train_targets)` and predicts `queries`.
fn fit_predict(
    huber: bool,
    train_features: &[Vec<f64>],
    train_targets: &[f64],
    queries: &SolversMatrix,
    query_rows: usize,
) -> Vec<f64> {
    let dataset = RegressionDataset {
        features: solvers_matrix(train_features),
        targets: SolversMatrix::from_row_major(train_features.len(), 1, train_targets.to_vec()),
        sample_weights: None,
    };

    let report = fit_robust_regression(&dataset, method_config(huber))
        .expect("regression fits the standardized OBD2 design");
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

/// Splits `train_rows` into (calibration, proper-training) by taking every
/// `stride`-th row as calibration; the rest train the point predictor.
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

/// Pooled coverage/width accumulator for one method across folds.
#[derive(Default)]
struct CoverageAccumulator {
    covered: usize,
    total: usize,
    width_sum: f64,
}

impl CoverageAccumulator {
    fn add_fold(&mut self, band: &SplitConformal, predictions: &[f64], actuals: &[f64]) {
        for (prediction, actual) in predictions.iter().zip(actuals)
        {
            if band.covers(*prediction, *actual)
            {
                self.covered += 1;
            }

            self.total += 1;
            self.width_sum += band.width();
        }
    }

    fn coverage(&self) -> f64 {
        self.covered as f64 / self.total as f64
    }

    fn mean_width(&self) -> f64 {
        self.width_sum / self.total as f64
    }
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
        "# obd2_conformal — Direction C: does a robust point predictor give tighter valid intervals?"
    );
    println!(
        "# split-conformal at level {LEVEL} on real OBD2, leave-one-segment-out; calibration = every \
{CALIBRATION_STRIDE}th train row; OLS vs Huber share the split"
    );

    for target in TARGETS
    {
        let data = parse_obd2(&text, target).expect("OBD2 parses for the target channel");
        let groups = data.groups.as_ref().expect("OBD2 carries segment groups");
        let segments = distinct_sorted(groups);
        let rows = data.targets.len();

        let mut ols = CoverageAccumulator::default();
        let mut huber = CoverageAccumulator::default();

        for &held in &segments
        {
            let train_rows: Vec<usize> = (0..rows).filter(|&row| groups[row] != held).collect();
            let test_rows: Vec<usize> = (0..rows).filter(|&row| groups[row] == held).collect();
            let (calibration_rows, proper_rows) =
                split_calibration(&train_rows, CALIBRATION_STRIDE);

            let proper_features_raw: Vec<Vec<f64>> = proper_rows
                .iter()
                .map(|&row| data.features[row].clone())
                .collect();
            let proper_targets: Vec<f64> =
                proper_rows.iter().map(|&row| data.targets[row]).collect();

            let standardizer = Standardizer::fit(&proper_features_raw);
            let proper_features = standardizer.apply(&proper_features_raw);

            let calibration_features = standardizer.apply(
                &calibration_rows
                    .iter()
                    .map(|&row| data.features[row].clone())
                    .collect::<Vec<_>>(),
            );
            let calibration_targets: Vec<f64> = calibration_rows
                .iter()
                .map(|&row| data.targets[row])
                .collect();
            let calibration_matrix = solvers_matrix(&calibration_features);

            let test_features = standardizer.apply(
                &test_rows
                    .iter()
                    .map(|&row| data.features[row].clone())
                    .collect::<Vec<_>>(),
            );
            let test_targets: Vec<f64> = test_rows.iter().map(|&row| data.targets[row]).collect();
            let test_matrix = solvers_matrix(&test_features);

            for (huber_flag, accumulator) in [(false, &mut ols), (true, &mut huber)]
            {
                let calibration_predictions = fit_predict(
                    huber_flag,
                    &proper_features,
                    &proper_targets,
                    &calibration_matrix,
                    calibration_targets.len(),
                );
                let calibration_residuals: Vec<f64> = calibration_predictions
                    .iter()
                    .zip(&calibration_targets)
                    .map(|(prediction, actual)| actual - prediction)
                    .collect();

                let band = SplitConformal::fit(&calibration_residuals, LEVEL)
                    .expect("calibration set is large enough at level 0.9");

                let test_predictions = fit_predict(
                    huber_flag,
                    &proper_features,
                    &proper_targets,
                    &test_matrix,
                    test_targets.len(),
                );

                accumulator.add_fold(&band, &test_predictions, &test_targets);
            }
        }

        for (method, accumulator) in [("ols", &ols), ("huber_irls", &huber)]
        {
            for (metric, value) in [
                ("empirical_coverage", accumulator.coverage()),
                ("mean_interval_width", accumulator.mean_width()),
            ]
            {
                records.push(BenchRecord::new(
                    "obd2_conformal/interval",
                    format!("obd2/{target}"),
                    method,
                    0,
                    metric,
                    value,
                ));
            }
        }

        let width_ratio = huber.mean_width() / ols.mean_width();
        records.push(BenchRecord::new(
            "obd2_conformal/interval",
            format!("obd2/{target}"),
            "huber_over_ols",
            0,
            "width_ratio",
            width_ratio,
        ));

        println!("# {target} (nominal coverage {LEVEL}):");
        println!(
            "#   OLS   coverage={:.4} mean_width={:.4}",
            ols.coverage(),
            ols.mean_width()
        );
        println!(
            "#   Huber coverage={:.4} mean_width={:.4}",
            huber.coverage(),
            huber.mean_width()
        );
        println!(
            "#   Huber/OLS width ratio = {width_ratio:.4} ({})",
            if width_ratio < 1.0
            {
                "Huber tighter"
            }
            else
            {
                "OLS tighter"
            }
        );
    }

    fs::create_dir_all(&out_dir).expect("results directory is writable");
    fs::write(out_dir.join("obd2_conformal.jsonl"), to_jsonl(&records))
        .expect("results file is writable");

    println!("# records={}", records.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_calibration_takes_every_stride_th_row() {
        let train_rows: Vec<usize> = (10..20).collect();
        let (calibration, proper) = split_calibration(&train_rows, 5);
        // Positions 0 and 5 → rows 10 and 15.
        assert_eq!(calibration, vec![10, 15]);
        assert_eq!(proper, vec![11, 12, 13, 14, 16, 17, 18, 19]);
        // Every row is placed exactly once.
        assert_eq!(calibration.len() + proper.len(), train_rows.len());
    }

    #[test]
    fn accumulator_tracks_coverage_and_width() {
        let band =
            SplitConformal::fit(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 0.9).unwrap();
        let mut accumulator = CoverageAccumulator::default();
        // Predictions all 0; actuals within/without the half-width (= 9).
        accumulator.add_fold(&band, &[0.0, 0.0, 0.0], &[1.0, 9.0, 100.0]);
        assert_eq!(accumulator.total, 3);
        assert_eq!(accumulator.covered, 2); // 1 and 9 covered, 100 not
        assert!((accumulator.mean_width() - band.width()).abs() < 1e-12);
    }
}
