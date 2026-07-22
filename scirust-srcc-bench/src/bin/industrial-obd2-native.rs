//! Follow-up 7 / axis 3 — native heavy-tailed contamination on real data.
//!
//! Follow-up 3 established *where* robust regression wins — pervasive
//! heavy-tailed vertical noise — but did it on a **semi-synthetic** target:
//! the real C-MAPSS design matrix carried a planted linear signal plus injected
//! Student-t errors, because C-MAPSS's own residuals are not that heavy. This
//! binary removes the injection entirely and anchors item 3 in **real
//! measurements**: the in-repo Opel-Corsa OBD2 telemetry (43 139 rows, seven
//! driving segments), predicting one real engine channel from the others. No
//! planted signal, no injected noise — whatever heaviness the residuals have is
//! the car's.
//!
//! Three targets are used, each chosen because a plain OLS fit leaves natively
//! heavy-tailed residuals (excess kurtosis 45 / 19 / 10 — the binary re-derives
//! and prints these, so the "native" claim is checked, not asserted):
//! `ENGINE_LOAD`, `THROTTLE_POS`, `MAF`. Evaluation is honest **leave-one-segment-out**
//! cross-validation — each of the seven segments is the held-out test set once,
//! so no within-segment autocorrelation leaks train→test — with the estimator
//! fit on the other six segments (features standardized on the training rows).
//! Fixed a-priori hyperparameters (Huber `δ = 1.345`, trimmed keep `0.9`); no
//! tuning, nothing selected on any outcome.
//!
//! Because real held-out targets are themselves noisy (there is no noiseless
//! truth to recover), robustness is judged on *bulk* prediction: alongside RMSE
//! (tail-sensitive, OLS-favourable) the binary reports MAE, median absolute
//! error and a 10 %-trimmed RMSE, and runs the seeded paired bootstrap on
//! per-row absolute errors (OLS − robust) for a signed verdict. Whether robust
//! regression natively beats OLS on this real workload — and on which metric —
//! is reported as-is. Deterministic; run twice, byte-identical.

use std::fs;
use std::path::{Path, PathBuf};

use scirust_bench_schema::{BenchRecord, to_jsonl};
use scirust_learning::{
    RegressionDataset, RobustLoss, RobustRegressionConfig, RobustRegressionMethod,
    fit_robust_regression,
};
use scirust_solvers::linalg::Matrix as SolversMatrix;
use scirust_srcc_bench::{paired_bootstrap, paired_differences, parse_obd2, sha256_hex};
use serde::Deserialize;

const CONFIG_TEXT: &str = include_str!("../../configs/phase728.json");

/// SHA-256 of the in-repo OBD2 telemetry CSV (checksum-verified for determinism).
const OBD2_SHA: &str = "229ef4488a89b62be530acce36ec7522421e7b82b1b5279022ffac72f0bb0751";

/// Real engine channels used as regression targets. Each leaves natively
/// heavy-tailed OLS residuals (verified in-binary); ordered heaviest first.
const TARGETS: [&str; 3] = ["ENGINE_LOAD", "THROTTLE_POS", "MAF"];

/// Huber transition point in MAD-scale units (standard 95 %-efficiency value).
const HUBER_DELTA: f64 = 1.345;
/// Fraction of rows retained by the trimmed estimator at every iteration.
const TRIMMED_FRACTION: f64 = 0.9;
/// Fraction of (smallest-|residual|) rows kept by the trimmed-RMSE metric.
const TRIMMED_METRIC_FRACTION: f64 = 0.9;
/// Floor for a standard deviation before dividing.
const SD_FLOOR: f64 = 1.0e-12;

#[derive(Deserialize)]
struct Config {
    bootstrap: BootstrapConfig,
}

#[derive(Deserialize)]
struct BootstrapConfig {
    resamples: usize,
    level: f64,
    seed: u64,
}

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

/// The three estimators under test, with fixed a-priori configuration.
fn method_config(method: &str) -> RobustRegressionConfig {
    let base = RobustRegressionConfig::default();

    match method
    {
        "ols" => RobustRegressionConfig {
            method: RobustRegressionMethod::OrdinaryLeastSquares,
            ..base
        },
        "huber_irls" => RobustRegressionConfig {
            method: RobustRegressionMethod::IterativelyReweightedLeastSquares,
            loss: RobustLoss::Huber { delta: HUBER_DELTA },
            ..base
        },
        "trimmed_ls" => RobustRegressionConfig {
            method: RobustRegressionMethod::TrimmedLeastSquares {
                retained_fraction: TRIMMED_FRACTION,
            },
            ..base
        },
        other => panic!("unknown method {other}"),
    }
}

/// Fits `method` on standardized `(features, targets)` and predicts `test`.
fn fit_predict(
    method: &str,
    train_features: &[Vec<f64>],
    train_targets: &[f64],
    test_features: &SolversMatrix,
    test_rows: usize,
) -> Vec<f64> {
    let dataset = RegressionDataset {
        features: solvers_matrix(train_features),
        targets: SolversMatrix::from_row_major(train_features.len(), 1, train_targets.to_vec()),
        sample_weights: None,
    };

    let report = fit_robust_regression(&dataset, method_config(method))
        .expect("regression fits the standardized OBD2 design");
    let predictions = report
        .model
        .predict(test_features)
        .expect("prediction shapes match");

    (0..test_rows).map(|row| predictions[(row, 0)]).collect()
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

fn rmse(residuals: &[f64]) -> f64 {
    (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt()
}

fn mae(residuals: &[f64]) -> f64 {
    residuals.iter().map(|r| r.abs()).sum::<f64>() / residuals.len() as f64
}

fn sorted_absolute(residuals: &[f64]) -> Vec<f64> {
    let mut absolute: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
    absolute.sort_by(f64::total_cmp);
    absolute
}

fn median_absolute(residuals: &[f64]) -> f64 {
    let absolute = sorted_absolute(residuals);
    let n = absolute.len();

    if n == 0
    {
        return 0.0;
    }

    if n % 2 == 1
    {
        absolute[n / 2]
    }
    else
    {
        0.5 * (absolute[n / 2 - 1] + absolute[n / 2])
    }
}

/// RMSE over the `fraction` smallest-magnitude residuals (bulk fit quality).
fn trimmed_rmse(residuals: &[f64], fraction: f64) -> f64 {
    let absolute = sorted_absolute(residuals);
    let keep = ((fraction * absolute.len() as f64).floor() as usize).max(1);
    (absolute[..keep].iter().map(|r| r * r).sum::<f64>() / keep as f64).sqrt()
}

/// Normal-consistent MAD scale of the residuals.
fn mad_scale(residuals: &[f64]) -> f64 {
    let median = {
        let absolute_free = {
            let mut sorted: Vec<f64> = residuals.to_vec();
            sorted.sort_by(f64::total_cmp);
            sorted
        };
        let n = absolute_free.len();
        if n % 2 == 1
        {
            absolute_free[n / 2]
        }
        else
        {
            0.5 * (absolute_free[n / 2 - 1] + absolute_free[n / 2])
        }
    };

    let mut deviations: Vec<f64> = residuals.iter().map(|r| (r - median).abs()).collect();
    deviations.sort_by(f64::total_cmp);
    let n = deviations.len();
    let mad = if n % 2 == 1
    {
        deviations[n / 2]
    }
    else
    {
        0.5 * (deviations[n / 2 - 1] + deviations[n / 2])
    };

    (1.4826 * mad).max(SD_FLOOR)
}

/// Excess kurtosis (fourth standardized moment − 3); `0` under a normal.
fn excess_kurtosis(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;

    if variance <= 0.0
    {
        return 0.0;
    }

    let fourth = values.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / n;
    fourth / (variance * variance) - 3.0
}

/// Fraction of residuals whose magnitude exceeds `multiple × MAD-scale`.
fn tail_fraction(residuals: &[f64], multiple: f64) -> f64 {
    let scale = mad_scale(residuals);
    let threshold = multiple * scale;
    residuals.iter().filter(|r| r.abs() > threshold).count() as f64 / residuals.len() as f64
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

    let config: Config = serde_json::from_str(CONFIG_TEXT).expect("embedded config is valid");
    let bootstrap = &config.bootstrap;

    let text = read_verified(&data_path, OBD2_SHA);

    let methods = ["ols", "huber_irls", "trimmed_ls"];
    let mut records: Vec<BenchRecord> = Vec::new();

    println!("# obd2_native — Axis 3: does robustness win on REAL native heavy-tailed residuals?");
    println!(
        "# real Opel-Corsa OBD2 telemetry, leave-one-segment-out CV; fixed Huber δ={HUBER_DELTA}, \
trimmed keep={TRIMMED_FRACTION}; no injection, no planted signal"
    );

    for target in TARGETS
    {
        let data = parse_obd2(&text, target).expect("OBD2 parses for the target channel");
        let groups = data.groups.as_ref().expect("OBD2 carries segment groups");
        let segments = distinct_sorted(groups);
        let rows = data.targets.len();

        // Pooled held-out residuals per method, in a fixed (segment-sorted, then
        // row-index) order so the paired vectors align and the run is deterministic.
        let mut pooled: Vec<Vec<f64>> = vec![Vec::with_capacity(rows); methods.len()];

        for &held in &segments
        {
            let train_rows: Vec<usize> = (0..rows).filter(|&row| groups[row] != held).collect();
            let test_rows: Vec<usize> = (0..rows).filter(|&row| groups[row] == held).collect();

            let train_features_raw: Vec<Vec<f64>> = train_rows
                .iter()
                .map(|&row| data.features[row].clone())
                .collect();
            let train_targets: Vec<f64> = train_rows.iter().map(|&row| data.targets[row]).collect();
            let test_features_raw: Vec<Vec<f64>> = test_rows
                .iter()
                .map(|&row| data.features[row].clone())
                .collect();
            let test_targets: Vec<f64> = test_rows.iter().map(|&row| data.targets[row]).collect();

            let standardizer = Standardizer::fit(&train_features_raw);
            let train_features = standardizer.apply(&train_features_raw);
            let test_features = standardizer.apply(&test_features_raw);
            let test_matrix = solvers_matrix(&test_features);

            for (method_index, &method) in methods.iter().enumerate()
            {
                let predictions = fit_predict(
                    method,
                    &train_features,
                    &train_targets,
                    &test_matrix,
                    test_targets.len(),
                );

                for (prediction, actual) in predictions.iter().zip(&test_targets)
                {
                    pooled[method_index].push(actual - prediction);
                }
            }
        }

        // Native-tail characterization from the pooled OLS residuals.
        let ols_residuals = &pooled[0];
        let kurtosis = excess_kurtosis(ols_residuals);
        let beyond_3 = tail_fraction(ols_residuals, 3.0);
        let beyond_5 = tail_fraction(ols_residuals, 5.0);

        for (metric, value) in [
            ("excess_kurtosis", kurtosis),
            ("beyond_3mad_fraction", beyond_3),
            ("beyond_5mad_fraction", beyond_5),
        ]
        {
            records.push(BenchRecord::new(
                "obd2_native/tails",
                format!("obd2/{target}"),
                "ols_residual",
                0,
                metric,
                value,
            ));
        }

        println!(
            "# {target}: OLS residual excess kurtosis={kurtosis:.1}, beyond 3·MAD={:.2}% beyond 5·MAD={:.3}% \
(normal: 0, 0.27%, 0.00006%)",
            100.0 * beyond_3,
            100.0 * beyond_5
        );
        println!("#   method       rmse       mae    medAE  trim10RMSE");

        for (method_index, &method) in methods.iter().enumerate()
        {
            let residuals = &pooled[method_index];
            let method_rmse = rmse(residuals);
            let method_mae = mae(residuals);
            let method_median = median_absolute(residuals);
            let method_trimmed = trimmed_rmse(residuals, TRIMMED_METRIC_FRACTION);

            for (metric, value) in [
                ("rmse", method_rmse),
                ("mae", method_mae),
                ("median_abs_error", method_median),
                ("trimmed_rmse", method_trimmed),
            ]
            {
                records.push(BenchRecord::new(
                    "obd2_native/metrics",
                    format!("obd2/{target}"),
                    method.to_string(),
                    0,
                    metric,
                    value,
                ));
            }

            println!(
                "#   {method:<11} {method_rmse:8.4} {method_mae:8.4} {method_median:8.4} {method_trimmed:11.4}"
            );
        }

        // Signed verdict: paired bootstrap of per-row absolute-error reduction.
        for robust in ["huber_irls", "trimmed_ls"]
        {
            let robust_index = methods.iter().position(|&m| m == robust).expect("present");
            let ols_absolute: Vec<f64> = pooled[0].iter().map(|r| r.abs()).collect();
            let robust_absolute: Vec<f64> = pooled[robust_index].iter().map(|r| r.abs()).collect();

            let differences = paired_differences(&ols_absolute, &robust_absolute)
                .expect("aligned finite per-row absolute errors");
            let report = paired_bootstrap(
                &differences,
                bootstrap.resamples,
                bootstrap.level,
                bootstrap.seed,
            )
            .expect("enough held-out rows for the bootstrap");

            let verdict = if report.confidence_interval.lo > 0.0
            {
                "robust_wins"
            }
            else if report.confidence_interval.hi < 0.0
            {
                "ols_wins"
            }
            else
            {
                "straddles_zero"
            };

            records.push(
                BenchRecord::new(
                    "obd2_native/paired",
                    format!("obd2/{target}"),
                    format!("ols_minus_{robust}"),
                    bootstrap.seed,
                    "abs_error_paired_mean_difference",
                    report.mean_difference,
                )
                .with_ci(report.confidence_interval),
            );

            println!(
                "#   {robust} vs OLS (abs-error): Δ={:+.4} CI=[{:+.4},{:+.4}] {verdict}",
                report.mean_difference,
                report.confidence_interval.lo,
                report.confidence_interval.hi,
            );
        }
    }

    fs::create_dir_all(&out_dir).expect("results directory is writable");
    fs::write(out_dir.join("obd2_native.jsonl"), to_jsonl(&records))
        .expect("results file is writable");

    println!("# records={}", records.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmse_mae_are_consistent() {
        let residuals = [3.0, -4.0];
        assert!((rmse(&residuals) - (12.5_f64).sqrt()).abs() < 1e-12);
        assert!((mae(&residuals) - 3.5).abs() < 1e-12);
    }

    #[test]
    fn median_absolute_handles_even_and_odd() {
        assert!((median_absolute(&[1.0, -3.0, 2.0]) - 2.0).abs() < 1e-12);
        assert!((median_absolute(&[1.0, -3.0, 2.0, -4.0]) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn trimmed_rmse_drops_the_worst() {
        // Keeping the smallest 50 % of {1,2,3,100} keeps {1,2} → sqrt(2.5).
        let value = trimmed_rmse(&[1.0, -2.0, 3.0, -100.0], 0.5);
        assert!((value - (2.5_f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn excess_kurtosis_is_zero_for_a_flat_pair() {
        // A symmetric two-point set has kurtosis 1 → excess −2.
        assert!((excess_kurtosis(&[-1.0, 1.0]) + 2.0).abs() < 1e-12);
    }

    #[test]
    fn tail_fraction_counts_beyond_scaled_mad() {
        // One clear outlier among tight values sits beyond 5·MAD.
        let residuals = [0.0, 0.1, -0.1, 0.05, -0.05, 50.0];
        assert!(tail_fraction(&residuals, 5.0) > 0.0);
    }
}
