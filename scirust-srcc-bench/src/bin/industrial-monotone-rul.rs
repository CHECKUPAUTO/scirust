//! Item 1 — monotone RUL: realizing the lever-1 C-MAPSS ceiling.
//!
//! Lever 1 ([`industrial-reframed-rul`]) lowered the C-MAPSS RUL *task ceiling*
//! by making the target piecewise-linear (capped at the `R_early = 125` knee),
//! but the linear/robust regressors left that ceiling **unrealized** — a
//! piecewise target that a purely affine model cannot bend to. This binary adds
//! the missing piece: a **monotone recalibration** of the linear degradation
//! score with the new [`scirust_learning::isotonic::IsotonicRegression`] (PAVA).
//!
//! For each cell of the same 2×2 grid — {raw-linear, piecewise-125} RUL ×
//! {stride 20, 5} — it fits an ordinary-least-squares regressor to obtain a
//! scalar degradation score, then fits an isotonic map from that **train** score
//! to the **train** RUL (no leakage) and applies it to the test scores. The
//! monotone map is exactly the shape freedom the linear fit lacks: it can flatten
//! the score in the early-life plateau and steepen it near failure.
//!
//! It reports, per cell, the predict-the-mean RMSE (the ceiling), the plain-OLS
//! test RMSE and ceiling ratio (the lever-1 baseline), and the isotonic-recalibrated
//! test RMSE and ceiling ratio — a direct head-to-head of *does a monotone model
//! realize more of the ceiling than the linear fit?* Splits, fractions and the
//! missing-value policy are read verbatim from the frozen `configs/phase728.json`.
//!
//! Honest framing: isotonic recalibration is rank-preserving, so it can only help
//! when the score already orders the test engines well; where it does not lower
//! the ratio, that is reported as-is. Deterministic; run twice, byte-identical.

use std::fs;
use std::path::{Path, PathBuf};

use scirust_bench_schema::{BenchRecord, to_jsonl};
use scirust_learning::{
    IsotonicRegression, MonotoneDirection, RegressionDataset, RobustRegressionConfig,
    RobustRegressionMethod, fit_robust_regression,
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

/// Only the split/imputation fields are read; extra frozen-config keys
/// (`huber_delta`, `trimmed_fraction`, …) are intentionally ignored here — this
/// binary compares OLS against its own isotonic recalibration, nothing else.
#[derive(Deserialize)]
struct RegressionConfig {
    split_seed: u64,
    missing_maximum_fraction: f64,
    train_fraction: f64,
    validation_fraction: f64,
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

/// Fits ordinary least squares on the training features and returns the fitted
/// degradation scores (its own predictions) for the train and test rows.
fn ols_scores(
    train: &TabularDataset,
    test_features: &SolversMatrix,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let dataset = RegressionDataset {
        features: solvers_matrix(&train.features),
        targets: SolversMatrix::from_row_major(train.sample_count(), 1, train.targets.clone()),
        sample_weights: None,
    };

    let configuration = RobustRegressionConfig {
        method: RobustRegressionMethod::OrdinaryLeastSquares,
        ..RobustRegressionConfig::default()
    };

    let report = fit_robust_regression(&dataset, configuration).ok()?;

    let train_features = solvers_matrix(&train.features);
    let train_prediction = report.model.predict(&train_features).ok()?;
    let test_prediction = report.model.predict(test_features).ok()?;

    let train_scores: Vec<f64> = (0..train.sample_count())
        .map(|row| train_prediction[(row, 0)])
        .collect();
    let test_scores: Vec<f64> = (0..test_features.rows())
        .map(|row| test_prediction[(row, 0)])
        .collect();

    Some((train_scores, test_scores))
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
                "monotone_rul/ceiling",
                cell.clone(),
                "predict_mean",
                0,
                "rmse",
                predict_mean_rmse,
            ));

            let Some((train_scores, test_scores)) = ols_scores(&train, &test_features)
            else
            {
                records.push(BenchRecord::new(
                    "monotone_rul/ceiling",
                    cell.clone(),
                    "ols",
                    0,
                    "fit_ok",
                    0.0,
                ));
                summary.push(format!("# {cell:34} OLS FIT FAILED"));
                continue;
            };

            // Baseline: the plain-OLS prediction is the linear degradation score.
            let ols_rmse = rmse(&test_scores, &test.targets);
            let ols_ratio = ols_rmse / predict_mean_rmse;

            // Monotone recalibration: fit an isotonic map from the train score to
            // the train RUL (no leakage), apply to the test scores.
            let isotonic = IsotonicRegression::fit(
                &train_scores,
                &train.targets,
                MonotoneDirection::NonDecreasing,
                None,
            )
            .expect("finite scores and targets");
            let isotonic_prediction = isotonic.predict_slice(&test_scores);
            let isotonic_rmse = rmse(&isotonic_prediction, &test.targets);
            let isotonic_ratio = isotonic_rmse / predict_mean_rmse;

            for (method, fit_rmse, ratio) in [
                ("ols", ols_rmse, ols_ratio),
                ("isotonic_ols", isotonic_rmse, isotonic_ratio),
            ]
            {
                records.push(BenchRecord::new(
                    "monotone_rul/ceiling",
                    cell.clone(),
                    method.to_string(),
                    0,
                    "clean_fit_rmse",
                    fit_rmse,
                ));
                records.push(BenchRecord::new(
                    "monotone_rul/ceiling",
                    cell.clone(),
                    method.to_string(),
                    0,
                    "task_ceiling_ratio",
                    ratio,
                ));
            }

            summary.push(format!(
                "# {cell:34} train={:<4} test={:<4} ceiling_rmse={predict_mean_rmse:8.3} \
ols_ratio={ols_ratio:.4} isotonic_ratio={isotonic_ratio:.4} delta={:+.4}",
                train.sample_count(),
                test.sample_count(),
                ols_ratio - isotonic_ratio,
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

    println!("# monotone_rul — Item 1: isotonic recalibration of the OLS degradation score");
    println!(
        "# lower ceiling ratio = more of the task ceiling realized; delta>0 = isotonic beats OLS"
    );

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
    fs::write(out_dir.join("monotone_rul.jsonl"), to_jsonl(&records))
        .expect("results file is writable");

    println!("# records={}", records.len());
}
