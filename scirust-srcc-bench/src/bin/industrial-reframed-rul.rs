//! Lever 1 — re-framed C-MAPSS RUL regression.
//!
//! The phase-728 diagnostic showed the pooled *raw-linear* RUL fit had a low
//! task ceiling (clean-fit RMSE only ~35 % below predict-the-mean) and lost
//! statistical power to stride-20 decimation. This binary isolates the two
//! re-framing levers and measures their effect against the diagnostic
//! baseline, changing nothing else:
//!
//! - **piecewise-linear RUL** capped at the canonical `R_early = 125` knee
//!   (Heimes 2008) — early-life cycles carry no degradation signal, so
//!   capping their target removes an unlearnable regression burden;
//! - **reduced decimation** (stride 20 → 5) — recovering ~4× the training
//!   rows the diagnostic flagged as a power sacrifice.
//!
//! For each subset it reports, over a 2×2 grid of {raw, piecewise} RUL ×
//! {stride 20, stride 5}: the predict-the-mean RMSE (task ceiling), each
//! method's clean-fit test RMSE, and the ceiling ratio. The split seeds,
//! fractions and missing-value policy are read verbatim from the frozen
//! `configs/phase728.json` so the only differences are the two levers.
//!
//! This is exploratory re-framing, not a preregistered test; `R_early = 125`
//! and stride 5 are literature-standard / power-restoring choices fixed a
//! priori, not tuned on outcomes. Deterministic; run twice, byte-identical.

use std::fs;
use std::path::{Path, PathBuf};

use scirust_bench_schema::{BenchRecord, to_jsonl};
use scirust_learning::{
    RegressionDataset, RobustLoss, RobustRegressionConfig, RobustRegressionMethod,
    fit_robust_regression,
};
use scirust_solvers::linalg::Matrix as SolversMatrix;
use scirust_srcc_bench::{
    FittedImputer, MissingValuePolicy, SplitStrategy, TabularDataset, clip_rul_targets,
    parse_cmapss_training, sha256_hex, split_dataset,
};
use serde::Deserialize;

const CONFIG_TEXT: &str = include_str!("../../configs/phase728.json");

const TRAIN_FD001_SHA: &str = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8";
const TRAIN_FD003_SHA: &str = "2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2";

/// The canonical C-MAPSS piecewise-linear RUL knee (Heimes 2008).
const R_EARLY: f64 = 125.0;
/// Decimation strides compared: the phase-728 baseline and the power-restored
/// reframe.
const STRIDES: [usize; 2] = [20, 5];

#[derive(Deserialize)]
struct Config {
    cmapss: Section,
    cmapss_fd003: Section,
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
    if stride <= 1
    {
        return dataset.clone();
    }

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

fn rmse(predictions: &[f64], references: &[f64]) -> f64 {
    let sum: f64 = predictions
        .iter()
        .zip(references)
        .map(|(p, r)| (p - r).powi(2))
        .sum();

    (sum / predictions.len() as f64).sqrt()
}

fn method_configs(config: &RegressionConfig) -> Vec<(&'static str, RobustRegressionConfig)> {
    let base = RobustRegressionConfig::default();

    vec![
        (
            "ols",
            RobustRegressionConfig {
                method: RobustRegressionMethod::OrdinaryLeastSquares,
                ..base
            },
        ),
        (
            "huber_irls",
            RobustRegressionConfig {
                method: RobustRegressionMethod::IterativelyReweightedLeastSquares,
                loss: RobustLoss::Huber {
                    delta: config.huber_delta,
                },
                ..base
            },
        ),
        (
            "trimmed_ls",
            RobustRegressionConfig {
                method: RobustRegressionMethod::TrimmedLeastSquares {
                    retained_fraction: config.trimmed_fraction,
                },
                ..base
            },
        ),
    ]
}

fn clean_fit_rmse(
    configuration: RobustRegressionConfig,
    train: &TabularDataset,
    test_features: &SolversMatrix,
    test: &TabularDataset,
) -> Option<f64> {
    let dataset = RegressionDataset {
        features: solvers_matrix(&train.features),
        targets: SolversMatrix::from_row_major(train.sample_count(), 1, train.targets.clone()),
        sample_weights: None,
    };

    let report = fit_robust_regression(&dataset, configuration).ok()?;
    let predictions = report.model.predict(test_features).ok()?;

    let values: Vec<f64> = (0..test.sample_count())
        .map(|row| predictions[(row, 0)])
        .collect();

    Some(rmse(&values, &test.targets))
}

fn evaluate(
    label: &str,
    raw: &TabularDataset,
    config: &RegressionConfig,
    records: &mut Vec<BenchRecord>,
    summary: &mut Vec<String>,
) {
    for &stride in &STRIDES
    {
        for (rul_name, dataset) in [
            ("raw_linear", raw.clone()),
            ("piecewise_125", clip_rul_targets(raw, R_EARLY)),
        ]
        {
            let decimated = decimate_by_group(&dataset, stride);

            let split = split_dataset(
                &decimated,
                &SplitStrategy::GroupedHoldout {
                    train_fraction: config.train_fraction,
                    validation_fraction: config.validation_fraction,
                },
                config.split_seed,
                Some("unit"),
            )
            .expect("grouped split is valid");

            let train_raw = decimated.select_rows(&split.train);
            let test_raw = decimated.select_rows(&split.test);

            let imputer = FittedImputer::fit(
                &train_raw.features,
                MissingValuePolicy {
                    maximum_missing_fraction: config.missing_maximum_fraction,
                },
            )
            .expect("training keeps a varying column");

            let train = imputer.transform(&train_raw).expect("shapes match");
            let test = imputer.transform(&test_raw).expect("shapes match");
            let test_features = solvers_matrix(&test.features);

            let train_mean = train.targets.iter().sum::<f64>() / train.sample_count() as f64;
            let predict_mean_rmse = rmse(&vec![train_mean; test.sample_count()], &test.targets);

            let cell = format!("{label}/{rul_name}/stride_{stride}");

            records.push(BenchRecord::new(
                "reframed_rul/ceiling",
                cell.clone(),
                "predict_mean",
                0,
                "rmse",
                predict_mean_rmse,
            ));

            let mut best_ratio = f64::INFINITY;

            for (method, configuration) in method_configs(config)
            {
                let Some(clean) = clean_fit_rmse(configuration, &train, &test_features, &test)
                else
                {
                    records.push(BenchRecord::new(
                        "reframed_rul/ceiling",
                        cell.clone(),
                        method.to_string(),
                        0,
                        "fit_ok",
                        0.0,
                    ));
                    continue;
                };

                let ratio = clean / predict_mean_rmse;
                best_ratio = best_ratio.min(ratio);

                records.push(BenchRecord::new(
                    "reframed_rul/ceiling",
                    cell.clone(),
                    method.to_string(),
                    0,
                    "clean_fit_rmse",
                    clean,
                ));
                records.push(BenchRecord::new(
                    "reframed_rul/ceiling",
                    cell.clone(),
                    method.to_string(),
                    0,
                    "task_ceiling_ratio",
                    ratio,
                ));
            }

            summary.push(format!(
                "# {label:12} {rul_name:14} stride={stride:<2} train={:<4} test={:<4} \
predict_mean_rmse={predict_mean_rmse:8.3} best_ceiling_ratio={best_ratio:.4}",
                train.sample_count(),
                test.sample_count(),
            ));
        }
    }
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

    let mut records: Vec<BenchRecord> = Vec::new();
    let mut summary: Vec<String> = Vec::new();

    println!("# reframed_rul — Lever 1: piecewise RUL (R_early=125) + reduced decimation");
    println!("# lower ceiling ratio = the features explain more of the target");

    let fd001 = parse_cmapss_training(&read_verified(
        &data_dir.join("cmapss/train_FD001.txt"),
        TRAIN_FD001_SHA,
    ))
    .expect("FD001 parses");
    evaluate(
        "cmapss",
        &fd001,
        &config.cmapss.regression,
        &mut records,
        &mut summary,
    );

    let fd003 = parse_cmapss_training(&read_verified(
        &data_dir.join("cmapss/train_FD003.txt"),
        TRAIN_FD003_SHA,
    ))
    .expect("FD003 parses");
    evaluate(
        "cmapss_fd003",
        &fd003,
        &config.cmapss_fd003.regression,
        &mut records,
        &mut summary,
    );

    for line in &summary
    {
        println!("{line}");
    }

    fs::create_dir_all(&out_dir).expect("results directory is writable");
    fs::write(out_dir.join("reframed_rul.jsonl"), to_jsonl(&records))
        .expect("results file is writable");

    println!("# records={}", records.len());
}
