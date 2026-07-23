//! Third program, direction C (sub-PR 4) — a promotion gate on interval quality.
//!
//! Axis 4 built promote/hold gates for *point* accuracy; direction C built
//! *intervals* (split-conformal C.1, quantile-native C.2, CQR C.3). This closes
//! the loop: an [`IntervalPromotionGate`] decides whether a candidate interval is
//! worth deploying over an incumbent. Its rule is **coverage-constrained width
//! improvement** — promote only when the candidate is defensibly *tighter* AND
//! still defensibly *covers* an absolute floor. A point-accuracy gate cannot say
//! this: coverage is an absolute constraint (meet a nominal SLA), width is a
//! relative gain, and a tighter band that under-covers is worse, not better.
//!
//! On the real OBD2 workload (leave-one-segment-out, three heavy-tailed targets),
//! the **incumbent** is the C.1 OLS-conformal band at level 0.9. Three candidates
//! face the gate on the same pooled test points:
//!
//! - **cqr** — the C.3 conformalized-quantile band (adaptive width, guaranteed);
//! - **quantile_native** — the C.2 native `[q₀.₀₅, q₀.₉₅]` band (adaptive, no guarantee);
//! - **conformal_shrunk** — the incumbent band scaled to 60 % of its half-width, a
//!   stand-in for "an operator who chased tightness past the coverage floor".
//!
//! The gate should promote the first two (tighter, still covering ≥ 0.85) and hold
//! the third (tightest, but under-covering). Whether each does on this data is
//! reported as-is. Deterministic; run twice, byte-identical.

use std::fs;
use std::path::{Path, PathBuf};

use scirust_bench_schema::{BenchRecord, to_jsonl};
use scirust_learning::{
    LinearRegressionModel, QuantileRegressionConfig, RegressionDataset, RobustRegressionConfig,
    RobustRegressionMethod, fit_quantile_regression, fit_robust_regression,
};
use scirust_solvers::linalg::Matrix as SolversMatrix;
use scirust_srcc_bench::{
    ConformalizedQuantile, Decision, IntervalPromotionGate, IntervalPromotionReport,
    IntervalSample, SplitConformal, parse_obd2, sha256_hex,
};

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

/// Absolute coverage the candidate must defensibly meet — the 0.9 nominal minus a
/// five-point operational slack, a realistic deployment SLA under segment shift.
const COVERAGE_FLOOR: f64 = 0.85;
/// Any defensible tightening qualifies on width.
const MIN_WIDTH_IMPROVEMENT: f64 = 0.0;
/// Bootstrap resamples and confidence level for the gate's paired inference.
const GATE_RESAMPLES: usize = 2000;
const GATE_LEVEL: f64 = 0.95;
/// Recorded gate seed (direction C.4).
const GATE_SEED: u64 = 729_004;
/// The "over-tightened" stand-in keeps this fraction of the conformal half-width.
const SHRINK: f64 = 0.6;
/// Default proper-training decimation (calibration and test stay full): the
/// quantile IRLS fits dominate runtime, so this keeps the demonstration tractable
/// without touching the honesty of the pooled coverage/width the gate sees.
const DEFAULT_TRAIN_STRIDE: usize = 6;

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

/// Keeps every `stride`-th element of `rows` (decimation of the proper-training
/// set only). `stride == 1` is the identity.
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

/// Per-method pooled prediction intervals across folds, aligned point-for-point.
#[derive(Default)]
struct PooledIntervals {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl PooledIntervals {
    fn push(&mut self, lower: f64, upper: f64) {
        self.lower.push(lower);
        self.upper.push(upper);
    }

    fn sample(&self) -> IntervalSample {
        IntervalSample {
            lower: self.lower.clone(),
            upper: self.upper.clone(),
        }
    }
}

/// Empirical coverage of a pooled band against the pooled targets.
fn empirical_coverage(band: &PooledIntervals, actual: &[f64]) -> f64 {
    let covered = band
        .lower
        .iter()
        .zip(&band.upper)
        .zip(actual)
        .filter(|((low, high), target)| low <= target && target <= high)
        .count();
    covered as f64 / actual.len() as f64
}

/// Mean width of a pooled band.
fn mean_width(band: &PooledIntervals) -> f64 {
    let sum: f64 = band
        .lower
        .iter()
        .zip(&band.upper)
        .map(|(low, high)| high - low)
        .sum();
    sum / band.lower.len() as f64
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

    let gate = IntervalPromotionGate {
        coverage_floor: COVERAGE_FLOOR,
        min_width_improvement: MIN_WIDTH_IMPROVEMENT,
        resamples: GATE_RESAMPLES,
        level: GATE_LEVEL,
        seed: GATE_SEED,
    };

    println!("# obd2_interval_gate — Direction C.4: a promotion gate on interval quality");
    println!(
        "# rule: promote iff candidate coverage CI-lower ≥ {COVERAGE_FLOOR} AND width-reduction \
CI-lower > {MIN_WIDTH_IMPROVEMENT}"
    );
    println!(
        "# incumbent = OLS-conformal @ {LEVEL}; candidates = cqr, quantile_native, \
conformal_shrunk_{SHRINK}"
    );
    println!(
        "# real OBD2 leave-one-segment-out; bootstrap level={GATE_LEVEL} resamples={GATE_RESAMPLES} \
seed={GATE_SEED}; proper-training decimated {train_stride}x (calibration + test full)"
    );

    for target in TARGETS
    {
        let data = parse_obd2(&text, target).expect("OBD2 parses for the target channel");
        let groups = data.groups.as_ref().expect("OBD2 carries segment groups");
        let segments = distinct_sorted(groups);
        let rows = data.targets.len();

        let mut conformal = PooledIntervals::default();
        let mut cqr = PooledIntervals::default();
        let mut native = PooledIntervals::default();
        let mut shrunk = PooledIntervals::default();
        let mut pooled_actual: Vec<f64> = Vec::new();

        for &held in &segments
        {
            let train_rows: Vec<usize> = (0..rows).filter(|&row| groups[row] != held).collect();
            let test_rows: Vec<usize> = (0..rows).filter(|&row| groups[row] == held).collect();
            let (calibration_rows, proper_all) = split_calibration(&train_rows, CALIBRATION_STRIDE);
            let proper_rows = decimate(&proper_all, train_stride);

            // One standardizer per fold, fit on the (decimated) proper-training rows.
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
            let half_width = ols_band.half_width();

            for index in 0..test_targets.len()
            {
                let (ols_low, ols_high) = ols_band.interval(ols_test[index]);
                conformal.push(ols_low, ols_high);

                let (cqr_low, cqr_high) = band.interval(low_test[index], high_test[index]);
                cqr.push(cqr_low, cqr_high);

                native.push(low_test[index], high_test[index]);

                // The over-tightened stand-in: the conformal band scaled toward its
                // centre. No new fit — a deliberate chase for width at coverage's cost.
                shrunk.push(
                    ols_test[index] - SHRINK * half_width,
                    ols_test[index] + SHRINK * half_width,
                );

                pooled_actual.push(test_targets[index]);
            }
        }

        let incumbent = conformal.sample();
        let incumbent_width = mean_width(&conformal);

        println!("# {target}:");
        println!(
            "#   incumbent OLS-conformal: coverage={:.4} mean_width={:.4}",
            empirical_coverage(&conformal, &pooled_actual),
            incumbent_width
        );

        let shrunk_label = format!("conformal_shrunk_{SHRINK}");

        for (method, band) in [
            ("cqr", &cqr),
            ("quantile_native", &native),
            (shrunk_label.as_str(), &shrunk),
        ]
        {
            let report: IntervalPromotionReport = gate
                .decide(&incumbent, &band.sample(), &pooled_actual)
                .expect("gate inputs are aligned and finite");

            let decision_value = match report.decision
            {
                Decision::Promote => 1.0,
                Decision::Hold => 0.0,
            };

            for (metric, value) in [
                ("promotion_decision", decision_value),
                ("candidate_coverage", report.coverage.mean),
                ("coverage_ci_lower", report.coverage.confidence_interval.lo),
                ("mean_width_reduction", report.width.mean),
                (
                    "width_reduction_ci_lower",
                    report.width.confidence_interval.lo,
                ),
            ]
            {
                records.push(BenchRecord::new(
                    "obd2_interval_gate/decision",
                    format!("obd2/{target}"),
                    method,
                    GATE_SEED,
                    metric,
                    value,
                ));
            }

            let verdict = match report.decision
            {
                Decision::Promote => "PROMOTE",
                Decision::Hold => "HOLD",
            };
            println!(
                "#   {method:<22} {verdict:<7} coverage={:.4} (ci_lo {:.4}, floor {COVERAGE_FLOOR}) \
Δwidth={:.4} (ci_lo {:.4})",
                report.coverage.mean,
                report.coverage.confidence_interval.lo,
                report.width.mean,
                report.width.confidence_interval.lo,
            );
            if !report.reasons.is_empty()
            {
                println!("#     held: {}", report.reasons.join("; "));
            }
        }
    }

    fs::create_dir_all(&out_dir).expect("results directory is writable");
    fs::write(out_dir.join("obd2_interval_gate.jsonl"), to_jsonl(&records))
        .expect("results file is writable");

    println!("# records={}", records.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimate_keeps_every_strided_row() {
        let rows: Vec<usize> = (0..10).collect();
        assert_eq!(decimate(&rows, 3), vec![0, 3, 6, 9]);
        assert_eq!(decimate(&rows, 1), rows);
    }

    #[test]
    fn pooled_coverage_and_width_are_exact() {
        let mut band = PooledIntervals::default();
        band.push(0.0, 2.0); // covers 1.0, width 2
        band.push(0.0, 2.0); // misses 5.0, width 2
        let actual = vec![1.0, 5.0];
        assert!((empirical_coverage(&band, &actual) - 0.5).abs() < 1e-12);
        assert!((mean_width(&band) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn shrinking_a_conformal_band_narrows_it_and_can_drop_coverage() {
        let prediction = 10.0;
        let half_width = 2.0;
        let parent = (prediction - half_width, prediction + half_width); // [8, 12]
        let shrunk = (
            prediction - SHRINK * half_width,
            prediction + SHRINK * half_width,
        ); // [8.8, 11.2]

        assert!((shrunk.1 - shrunk.0) < (parent.1 - parent.0));
        // A target near the edge is inside the parent but outside the stand-in.
        let target = 8.5;
        assert!(parent.0 <= target && target <= parent.1);
        assert!(!(shrunk.0 <= target && target <= shrunk.1));
    }
}
