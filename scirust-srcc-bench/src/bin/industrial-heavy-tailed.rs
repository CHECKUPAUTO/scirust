//! Follow-up 3 — the regime that reopens the question: native heavy-tailed noise.
//!
//! Lever 2 ran the decisive robustness test with **injected high-leverage**
//! contamination on C-MAPSS and found a clean null: far-out training points have
//! bounded influence on in-distribution held-out predictions, so OLS was never
//! beaten. That result is specific to the *leverage* attack. This follow-up tests
//! the other canonical contamination regime — the one robust M-/trimmed
//! estimators are actually designed for: **pervasive heavy-tailed vertical
//! outliers**, i.e. a native error distribution with heavy tails rather than a
//! handful of adversarial leverage points.
//!
//! It is a controlled, honest semi-synthetic test on the **real** C-MAPSS FD001
//! design matrix (imputed, standardized — realistic feature geometry and
//! collinearity), with a fixed planted linear signal and native errors drawn from
//! a **Student-t** whose degrees of freedom `ν` sweep the tail heaviness
//! (`ν = 1` Cauchy … `ν = 30` ≈ Gaussian). The target carries genuine heavy-tailed
//! noise; recovery is measured against the *noiseless* signal on a held-out split,
//! so the metric is "how well did each estimator recover the truth despite
//! heavy-tailed training noise". OLS, Huber-IRLS and trimmed LS are compared with
//! the same seeded paired bootstrap as lever 2.
//!
//! Expectation, stated in advance: as `ν` falls the tails thicken and Huber /
//! trimmed should increasingly beat OLS (a CI strictly above zero), while near
//! Gaussian (`ν = 30`) OLS should tie or win — the mirror image of lever 2's
//! leverage null. If so, this identifies the regime that *does* reward the
//! current robust regressors; the real industrial RUL workload simply does not
//! live in it (its native residuals are not this heavy), which is why lever 2 was
//! null. Whatever the CIs say is reported as-is. Deterministic; run twice,
//! byte-identical.

use std::fs;
use std::path::{Path, PathBuf};

use scirust_bench_schema::{BenchRecord, to_jsonl};
use scirust_learning::{
    RegressionDataset, RobustLoss, RobustRegressionConfig, RobustRegressionMethod,
    fit_robust_regression,
};
use scirust_solvers::linalg::Matrix as SolversMatrix;
use scirust_srcc_bench::{
    FittedImputer, MissingValuePolicy, SplitStrategy, TabularDataset, paired_bootstrap,
    paired_differences, parse_cmapss_training, sha256_hex, split_dataset,
};
use scirust_stats::{Distribution, SplitMix64, StudentT};
use serde::Deserialize;

const CONFIG_TEXT: &str = include_str!("../../configs/phase728.json");

const TRAIN_FD001_SHA: &str = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8";

/// Decimation stride (matches lever 2 / the phase-728 baseline).
const STRIDE: usize = 20;
/// Student-t degrees of freedom swept, heaviest tail first; `ν = 30` is the
/// near-Gaussian control where OLS should be optimal.
const DF_GRID: [f64; 5] = [1.0, 2.0, 3.0, 5.0, 30.0];
/// Noise scale as a multiple of the planted-signal standard deviation (SNR ≈ 1).
const NOISE_SCALE: f64 = 1.0;
/// Base seed for the native error draws (combined with each `ν`).
const NOISE_SEED: u64 = 0x0728_0003;
/// Quantile clamp so a `u ∈ {0, 1}` draw cannot map to a non-finite error.
const QUANTILE_CLAMP: f64 = 1.0e-9;

#[derive(Deserialize)]
struct Config {
    bootstrap: BootstrapConfig,
    cmapss: Section,
}

#[derive(Deserialize)]
struct BootstrapConfig {
    resamples: usize,
    level: f64,
    seed: u64,
}

#[derive(Deserialize)]
struct Section {
    regression: RegressionConfig,
}

#[derive(Deserialize)]
struct RegressionConfig {
    split_seed: u64,
    missing_maximum_fraction: f64,
    train_fraction: f64,
    validation_fraction: f64,
    huber_delta: f64,
    trimmed_fraction: f64,
}

fn read_verified(path: &Path, expected_sha: &str) -> String {
    let bytes = fs::read(path).unwrap_or_else(|error| {
        panic!(
            "cannot read {}: {error}. Run scripts/fetch_industrial_datasets.sh first.",
            path.display()
        )
    });

    let actual = sha256_hex(&bytes);

    assert!(
        actual == expected_sha,
        "checksum mismatch for {}: expected {expected_sha}, found {actual}",
        path.display()
    );

    String::from_utf8(bytes).expect("dataset files are valid UTF-8")
}

fn decimate_by_group(dataset: &TabularDataset, stride: usize) -> TabularDataset {
    let groups = dataset.groups.as_ref().expect("decimation needs groups");
    let time = dataset.time_index.as_ref().expect("decimation needs time");

    let mut distinct: Vec<u64> = Vec::new();

    for &group in groups
    {
        if !distinct.contains(&group)
        {
            distinct.push(group);
        }
    }

    distinct.sort_unstable();

    let mut keep: Vec<usize> = Vec::new();

    for &group in &distinct
    {
        let mut members: Vec<usize> = (0..dataset.sample_count())
            .filter(|&row| groups[row] == group)
            .collect();

        members.sort_by_key(|&row| (time[row], row));

        keep.extend(members.iter().copied().step_by(stride));
    }

    keep.sort_unstable();
    dataset.select_rows(&keep)
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
            .map(|variance| (variance / rows as f64).sqrt().max(1.0e-12))
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

/// The fixed a-priori planted signal coefficients (alternating ±1 across the
/// standardized features), not tuned on any outcome.
fn planted_coefficients(dimension: usize) -> Vec<f64> {
    (0..dimension)
        .map(|j| if j % 2 == 0 { 1.0 } else { -1.0 })
        .collect()
}

/// Noiseless signal `X · β` for every row.
fn signal(features: &[Vec<f64>], beta: &[f64]) -> Vec<f64> {
    features
        .iter()
        .map(|row| row.iter().zip(beta).map(|(x, b)| x * b).sum())
        .collect()
}

fn population_sd(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    variance.sqrt()
}

fn method_config(method: &str, config: &RegressionConfig) -> RobustRegressionConfig {
    let base = RobustRegressionConfig::default();

    match method
    {
        "ols" => RobustRegressionConfig {
            method: RobustRegressionMethod::OrdinaryLeastSquares,
            ..base
        },
        "huber_irls" => RobustRegressionConfig {
            method: RobustRegressionMethod::IterativelyReweightedLeastSquares,
            loss: RobustLoss::Huber {
                delta: config.huber_delta,
            },
            ..base
        },
        "trimmed_ls" => RobustRegressionConfig {
            method: RobustRegressionMethod::TrimmedLeastSquares {
                retained_fraction: config.trimmed_fraction,
            },
            ..base
        },
        other => panic!("unknown method {other}"),
    }
}

/// Fits `method` on `(features, targets)` and predicts the test rows.
fn fit_predict(
    method: &str,
    config: &RegressionConfig,
    features: &[Vec<f64>],
    targets: &[f64],
    test_features: &SolversMatrix,
    test_rows: usize,
) -> Option<Vec<f64>> {
    let dataset = RegressionDataset {
        features: solvers_matrix(features),
        targets: SolversMatrix::from_row_major(features.len(), 1, targets.to_vec()),
        sample_weights: None,
    };

    let report = fit_robust_regression(&dataset, method_config(method, config)).ok()?;
    let predictions = report.model.predict(test_features).ok()?;

    Some((0..test_rows).map(|row| predictions[(row, 0)]).collect())
}

/// Native heavy-tailed errors: `scale · t_ν(uᵢ)` for seeded uniforms `uᵢ`.
fn student_t_errors(count: usize, nu: f64, scale: f64, seed: u64) -> Vec<f64> {
    let distribution = StudentT::new(nu);
    let mut rng = SplitMix64::new(seed);

    (0..count)
        .map(|_| {
            let u = rng.next_f64().clamp(QUANTILE_CLAMP, 1.0 - QUANTILE_CLAMP);
            scale * distribution.quantile(u)
        })
        .collect()
}

/// Squared per-row error of predictions against the noiseless signal.
fn squared_errors(predictions: &[f64], truth: &[f64]) -> Vec<f64> {
    predictions
        .iter()
        .zip(truth)
        .map(|(p, t)| (p - t).powi(2))
        .collect()
}

fn rmse_from_squared(squared: &[f64]) -> f64 {
    (squared.iter().sum::<f64>() / squared.len() as f64).sqrt()
}

fn main() {
    let mut data_dir = PathBuf::from("data/industrial");
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
            "--data-dir" => data_dir = PathBuf::from(value("--data-dir")),
            "--out" => out_dir = PathBuf::from(value("--out")),
            other => panic!("unknown argument: {other}"),
        }
    }

    let config: Config = serde_json::from_str(CONFIG_TEXT).expect("embedded config is valid");
    let regression = &config.cmapss.regression;
    let bootstrap = &config.bootstrap;

    let fd001 = parse_cmapss_training(&read_verified(
        &data_dir.join("cmapss/train_FD001.txt"),
        TRAIN_FD001_SHA,
    ))
    .expect("FD001 parses");

    let decimated = decimate_by_group(&fd001, STRIDE);

    let split = split_dataset(
        &decimated,
        &SplitStrategy::GroupedHoldout {
            train_fraction: regression.train_fraction,
            validation_fraction: regression.validation_fraction,
        },
        regression.split_seed,
        Some("unit"),
    )
    .expect("grouped split is valid");

    let imputer = FittedImputer::fit(
        &decimated.select_rows(&split.train).features,
        MissingValuePolicy {
            maximum_missing_fraction: regression.missing_maximum_fraction,
        },
    )
    .expect("training keeps a varying column");

    let train_imputed = imputer
        .transform(&decimated.select_rows(&split.train))
        .expect("shapes match");
    let test_imputed = imputer
        .transform(&decimated.select_rows(&split.test))
        .expect("shapes match");

    // Real design matrix with realistic geometry, standardized on train.
    let standardizer = Standardizer::fit(&train_imputed.features);
    let train_features = standardizer.apply(&train_imputed.features);
    let test_features = standardizer.apply(&test_imputed.features);

    let dimension = train_features.first().map_or(0, Vec::len);
    let beta = planted_coefficients(dimension);
    let train_signal = signal(&train_features, &beta);
    let test_signal = signal(&test_features, &beta);
    let noise_scale = NOISE_SCALE * population_sd(&train_signal).max(1.0e-12);

    let test_matrix = solvers_matrix(&test_features);
    let test_rows = test_features.len();

    let methods = ["ols", "huber_irls", "trimmed_ls"];

    let mut records: Vec<BenchRecord> = Vec::new();
    let mut summary: Vec<String> = Vec::new();

    println!("# heavy_tailed — Follow-up 3: does robustness win under native heavy-tailed noise?");
    println!(
        "# real C-MAPSS FD001 design (stride {STRIDE}, standardized), planted signal + Student-t \
errors, scale={NOISE_SCALE}×signal_sd"
    );
    println!(
        "# train={} test={} features={dimension}; verdict from paired bootstrap of \
signal-squared-error (OLS − robust)",
        train_features.len(),
        test_rows
    );

    for &nu in &DF_GRID
    {
        let errors = student_t_errors(
            train_features.len(),
            nu,
            noise_scale,
            NOISE_SEED ^ nu.to_bits(),
        );
        let train_targets: Vec<f64> = train_signal
            .iter()
            .zip(&errors)
            .map(|(s, e)| s + e)
            .collect();

        let cell = format!("cmapss/df_{nu}");

        let mut per_method_squared: Vec<(&str, Vec<f64>)> = Vec::new();

        for &method in &methods
        {
            let predictions = fit_predict(
                method,
                regression,
                &train_features,
                &train_targets,
                &test_matrix,
                test_rows,
            )
            .expect("regression fits the standardized design");

            let squared = squared_errors(&predictions, &test_signal);
            let signal_rmse = rmse_from_squared(&squared);

            records.push(BenchRecord::new(
                "heavy_tailed/recovery",
                cell.clone(),
                method.to_string(),
                NOISE_SEED,
                "signal_rmse",
                signal_rmse,
            ));

            per_method_squared.push((method, squared));
        }

        let ols_squared = &per_method_squared[0].1;
        let mut verdicts: Vec<String> = Vec::new();

        for robust in ["huber_irls", "trimmed_ls"]
        {
            let robust_squared = &per_method_squared
                .iter()
                .find(|(name, _)| *name == robust)
                .expect("method present")
                .1;

            let differences = paired_differences(ols_squared, robust_squared)
                .expect("aligned finite per-row squared errors");
            let report = paired_bootstrap(
                &differences,
                bootstrap.resamples,
                bootstrap.level,
                bootstrap.seed,
            )
            .expect("enough test rows for the bootstrap");

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
                    "heavy_tailed/paired",
                    cell.clone(),
                    format!("ols_minus_{robust}"),
                    bootstrap.seed,
                    "signal_sqerr_paired_mean_difference",
                    report.mean_difference,
                )
                .with_ci(report.confidence_interval),
            );

            verdicts.push(format!(
                "{robust}: Δ={:+.3} CI=[{:+.3},{:+.3}] {verdict}",
                report.mean_difference,
                report.confidence_interval.lo,
                report.confidence_interval.hi,
            ));
        }

        let ols_rmse = rmse_from_squared(ols_squared);
        summary.push(format!(
            "# df={nu:<4} OLS signal_rmse={ols_rmse:7.3} | {}",
            verdicts.join(" | ")
        ));
    }

    for line in &summary
    {
        println!("{line}");
    }

    fs::create_dir_all(&out_dir).expect("results directory is writable");
    fs::write(out_dir.join("heavy_tailed.jsonl"), to_jsonl(&records))
        .expect("results file is writable");

    println!("# records={}", records.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn student_t_errors_are_deterministic_and_scaled() {
        let a = student_t_errors(64, 3.0, 2.0, 123);
        let b = student_t_errors(64, 3.0, 2.0, 123);
        assert_eq!(a, b);
        // A different seed gives a different sequence.
        let c = student_t_errors(64, 3.0, 2.0, 124);
        assert_ne!(a, c);
        assert!(a.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn heavier_tails_produce_a_larger_spread() {
        // ν = 1 (Cauchy) must have a wider error spread than ν = 30 (~Gaussian).
        let cauchy = student_t_errors(2000, 1.0, 1.0, 7);
        let gaussian = student_t_errors(2000, 30.0, 1.0, 7);
        let spread = |v: &[f64]| {
            let mut absolute: Vec<f64> = v.iter().map(|x| x.abs()).collect();
            absolute.sort_by(|a, b| a.total_cmp(b));
            absolute[absolute.len() * 99 / 100] // 99th percentile of |error|
        };
        assert!(
            spread(&cauchy) > spread(&gaussian),
            "expected heavier ν=1 tails than ν=30"
        );
    }

    #[test]
    fn signal_is_the_linear_combination() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let beta = vec![1.0, -1.0];
        assert_eq!(signal(&features, &beta), vec![-1.0, -1.0]);
    }

    #[test]
    fn planted_coefficients_alternate_sign() {
        assert_eq!(planted_coefficients(4), vec![1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    fn squared_error_and_rmse_are_consistent() {
        let squared = squared_errors(&[1.0, 4.0], &[0.0, 0.0]);
        assert_eq!(squared, vec![1.0, 16.0]);
        assert!((rmse_from_squared(&squared) - (8.5_f64).sqrt()).abs() < 1e-12);
    }
}
