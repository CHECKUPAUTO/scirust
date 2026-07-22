//! Follow-up 4 / phase 729 — shadow deployment & promotion gates.
//!
//! The whole program produced *evidence*; production needs a reproducible
//! *decision*. This binary drives the [`scirust_srcc_bench::PromotionGate`] on a
//! real shadow comparison that operationalizes follow-up 3's finding: a
//! **candidate** Huber-IRLS regressor is scored alongside the **incumbent** OLS
//! on the same held-out C-MAPSS rows, under native Student-t training noise swept
//! across tail heaviness. The gate — a *preregistered* rule, primary improvement
//! plus a guardrail, both decided on the seeded paired bootstrap — then issues
//! promote/hold.
//!
//! The expected behaviour makes the MLOps point concretely: under heavy tails
//! (`ν = 2`) Huber's improvement clears the bar and the gate **promotes**;
//! toward Gaussian (`ν = 30`) the gain vanishes and the gate **holds** the
//! incumbent — exactly the deployment decision follow-up 3's evidence implies,
//! made mechanically and reproducibly rather than by eyeballing a mean.
//!
//! Same real C-MAPSS FD001 design, planted signal and seeded `t`-errors as
//! `industrial-heavy-tailed`; the primary metric is per-row signal squared error
//! and the guardrail is per-row signal absolute error (both lower-is-better).
//! Deterministic; run twice, byte-identical.

use std::fs;
use std::path::{Path, PathBuf};

use scirust_bench_schema::{BenchRecord, to_jsonl};
use scirust_learning::{
    RegressionDataset, RobustLoss, RobustRegressionConfig, RobustRegressionMethod,
    fit_robust_regression,
};
use scirust_solvers::linalg::Matrix as SolversMatrix;
use scirust_srcc_bench::{
    Decision, FittedImputer, Guardrail, MissingValuePolicy, Orientation, PairedMetric,
    PrimaryCriterion, PromotionGate, SplitStrategy, TabularDataset, parse_cmapss_training,
    sha256_hex, split_dataset,
};
use scirust_stats::{Distribution, SplitMix64, StudentT};
use serde::Deserialize;

const CONFIG_TEXT: &str = include_str!("../../configs/phase728.json");

const TRAIN_FD001_SHA: &str = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8";

const STRIDE: usize = 20;
/// Tail heaviness of the native noise for the shadow scenarios: heavy, moderate,
/// near-Gaussian.
const DF_GRID: [f64; 3] = [2.0, 5.0, 30.0];
const NOISE_SCALE: f64 = 1.0;
const NOISE_SEED: u64 = 0x0729_0003;
const QUANTILE_CLAMP: f64 = 1.0e-9;

/// Preregistered gate: promote on any statistically defensible squared-error
/// improvement, provided the absolute-error guardrail does not regress beyond a
/// tolerance fixed a priori.
const GUARDRAIL_MAX_REGRESSION: f64 = 0.5;

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

/// Train-fitted per-column standardization.
fn standardize(train: &[Vec<f64>], other: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let rows = train.len();
    let cols = train.first().map_or(0, Vec::len);
    let mut means = vec![0.0; cols];

    for row in train
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

    let mut sds = vec![0.0; cols];

    for row in train
    {
        for (index, value) in row.iter().enumerate()
        {
            sds[index] += (value - means[index]).powi(2);
        }
    }

    for sd in &mut sds
    {
        *sd = (*sd / rows as f64).sqrt().max(1.0e-12);
    }

    let apply = |features: &[Vec<f64>]| -> Vec<Vec<f64>> {
        features
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(index, value)| (value - means[index]) / sds[index])
                    .collect()
            })
            .collect()
    };

    (apply(train), apply(other))
}

/// Fixed a-priori planted signal coefficients (alternating ±1).
fn planted_coefficients(dimension: usize) -> Vec<f64> {
    (0..dimension)
        .map(|j| if j % 2 == 0 { 1.0 } else { -1.0 })
        .collect()
}

fn signal(features: &[Vec<f64>], beta: &[f64]) -> Vec<f64> {
    features
        .iter()
        .map(|row| row.iter().zip(beta).map(|(x, b)| x * b).sum())
        .collect()
}

fn population_sd(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt()
}

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

fn regression_config(method: &str, huber_delta: f64) -> RobustRegressionConfig {
    let base = RobustRegressionConfig::default();

    match method
    {
        "ols" => RobustRegressionConfig {
            method: RobustRegressionMethod::OrdinaryLeastSquares,
            ..base
        },
        "huber_irls" => RobustRegressionConfig {
            method: RobustRegressionMethod::IterativelyReweightedLeastSquares,
            loss: RobustLoss::Huber { delta: huber_delta },
            ..base
        },
        other => panic!("unknown method {other}"),
    }
}

/// Fits `method` on `(features, targets)` and predicts the test rows.
fn fit_predict(
    method: &str,
    huber_delta: f64,
    features: &[Vec<f64>],
    targets: &[f64],
    test_features: &SolversMatrix,
    test_rows: usize,
) -> Vec<f64> {
    let dataset = RegressionDataset {
        features: solvers_matrix(features),
        targets: SolversMatrix::from_row_major(features.len(), 1, targets.to_vec()),
        sample_weights: None,
    };

    let report = fit_robust_regression(&dataset, regression_config(method, huber_delta))
        .expect("regression fits the standardized design");
    let predictions = report
        .model
        .predict(test_features)
        .expect("prediction shapes match");

    (0..test_rows).map(|row| predictions[(row, 0)]).collect()
}

fn absolute_errors(predictions: &[f64], truth: &[f64]) -> Vec<f64> {
    predictions
        .iter()
        .zip(truth)
        .map(|(p, t)| (p - t).abs())
        .collect()
}

fn squared_errors(predictions: &[f64], truth: &[f64]) -> Vec<f64> {
    predictions
        .iter()
        .zip(truth)
        .map(|(p, t)| (p - t).powi(2))
        .collect()
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

    let (train_features, test_features) =
        standardize(&train_imputed.features, &test_imputed.features);

    let dimension = train_features.first().map_or(0, Vec::len);
    let beta = planted_coefficients(dimension);
    let train_signal = signal(&train_features, &beta);
    let test_signal = signal(&test_features, &beta);
    let noise_scale = NOISE_SCALE * population_sd(&train_signal).max(1.0e-12);

    let test_matrix = solvers_matrix(&test_features);
    let test_rows = test_features.len();

    // The preregistered gate, identical across scenarios.
    let gate = PromotionGate {
        primary: PrimaryCriterion {
            metric: "signal_squared_error".to_string(),
            min_improvement: 0.0,
        },
        guardrails: vec![Guardrail {
            metric: "signal_absolute_error".to_string(),
            max_regression: GUARDRAIL_MAX_REGRESSION,
        }],
        resamples: bootstrap.resamples,
        level: bootstrap.level,
        seed: bootstrap.seed,
    };

    let mut records: Vec<BenchRecord> = Vec::new();
    let mut summary: Vec<String> = Vec::new();

    println!(
        "# promotion_gate — phase 729: shadow deployment of Huber (candidate) vs OLS (incumbent)"
    );
    println!(
        "# preregistered gate: primary=signal_squared_error (min_improvement 0), \
guardrail=signal_absolute_error (max_regression {GUARDRAIL_MAX_REGRESSION})"
    );
    println!("# under native Student-t training noise; promote only on a defensible improvement");

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

        let ols = fit_predict(
            "ols",
            regression.huber_delta,
            &train_features,
            &train_targets,
            &test_matrix,
            test_rows,
        );
        let huber = fit_predict(
            "huber_irls",
            regression.huber_delta,
            &train_features,
            &train_targets,
            &test_matrix,
            test_rows,
        );

        let shadow = [
            PairedMetric {
                metric: "signal_squared_error".to_string(),
                orientation: Orientation::LowerIsBetter,
                incumbent: squared_errors(&ols, &test_signal),
                candidate: squared_errors(&huber, &test_signal),
            },
            PairedMetric {
                metric: "signal_absolute_error".to_string(),
                orientation: Orientation::LowerIsBetter,
                incumbent: absolute_errors(&ols, &test_signal),
                candidate: absolute_errors(&huber, &test_signal),
            },
        ];

        let report = gate.decide(&shadow).expect("well-formed shadow comparison");
        let promoted = report.decision == Decision::Promote;

        let cell = format!("cmapss/df_{nu}");

        records.push(BenchRecord::new(
            "promotion_gate/decision",
            cell.clone(),
            "huber_vs_ols",
            bootstrap.seed,
            "promote",
            if promoted { 1.0 } else { 0.0 },
        ));
        records.push(
            BenchRecord::new(
                "promotion_gate/primary",
                cell.clone(),
                "signal_squared_error",
                bootstrap.seed,
                "improvement_mean",
                report.primary.mean,
            )
            .with_ci(report.primary.confidence_interval),
        );
        records.push(
            BenchRecord::new(
                "promotion_gate/guardrail",
                cell.clone(),
                "signal_absolute_error",
                bootstrap.seed,
                "regression_mean",
                report.guardrails[0].mean,
            )
            .with_ci(report.guardrails[0].confidence_interval),
        );

        summary.push(format!(
            "# df={nu:<4} decision={:<7} | primary Δ={:+.3} CI=[{:+.3},{:+.3}] pass={} | \
guardrail reg={:+.3} CI_hi={:+.3} pass={}",
            if promoted { "PROMOTE" } else { "HOLD" },
            report.primary.mean,
            report.primary.confidence_interval.lo,
            report.primary.confidence_interval.hi,
            report.primary.passed,
            report.guardrails[0].mean,
            report.guardrails[0].confidence_interval.hi,
            report.guardrails[0].passed,
        ));
    }

    for line in &summary
    {
        println!("{line}");
    }

    fs::create_dir_all(&out_dir).expect("results directory is writable");
    fs::write(out_dir.join("promotion_gate.jsonl"), to_jsonl(&records))
        .expect("results file is writable");

    println!("# records={}", records.len());
}
