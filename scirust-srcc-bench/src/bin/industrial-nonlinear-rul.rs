//! Follow-up 5 / axis 1 — multivariate nonlinear RUL: closing the ceiling further.
//!
//! Follow-up 1 realized part of the lever-1 C-MAPSS ceiling with a *one-
//! dimensional* monotone recalibration (isotonic on the OLS score) — rank-
//! preserving, so it cannot recover signal the OLS ordering discards. This
//! binary tests the model class beyond it: a genuinely **multivariate
//! nonlinear** regressor, RBF `scirust_learning::KernelRidgeRegression`, and asks
//! whether it closes the task ceiling further.
//!
//! On the same frozen phase-728 grid ({raw-linear, piecewise-125 RUL} × C-MAPSS
//! FD001/FD003), with the train-fitted [`FittedImputer`] and train-fitted
//! standardization (the RBF distance needs comparable feature scales), it reports
//! per cell the predict-the-mean RMSE (ceiling) and the ceiling ratio
//! (`clean_fit_rmse / predict_mean_rmse`, lower = more ceiling realized) for three
//! models on the **same** held-out test rows:
//!
//! - **OLS** — the linear baseline;
//! - **isotonic-OLS** — follow-up 1's 1-D monotone recalibration of the OLS score;
//! - **kernel ridge** — RBF, with `(γ, λ)` selected on the validation split over a
//!   fixed a-priori grid, then frozen on test.
//!
//! Stride is fixed at **40**: kernel ridge is a dense `O(n³)` Cholesky solve, and
//! the shared `scirust-solvers` factorization carries a large per-iteration
//! constant, so the finer strides balloon the grid-search runtime. Stride 40
//! (~320 pooled train rows) keeps the whole `(γ, λ)` sweep tractable while all
//! three models fit and are scored on the **same** rows — a fair head-to-head.
//! The absolute ratios therefore differ from follow-up 1's stride-20 numbers;
//! what this binary tests is the *ordering* (kernel ridge vs the 1-D isotonic
//! recalibration on identical data), reported as-is — it may be marginal.
//! Deterministic; run twice, byte-identical.

use std::fs;
use std::path::{Path, PathBuf};

use scirust_bench_schema::{BenchRecord, to_jsonl};
use scirust_learning::{
    IsotonicRegression, KernelRidgeConfig, KernelRidgeRegression, LinearRegressionModel,
    MonotoneDirection, RegressionDataset, RobustRegressionConfig, RobustRegressionMethod,
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
/// Default decimation stride (~320 pooled train rows). Overridable with
/// `--stride`. The original stride-40 default was forced by a large
/// per-iteration constant in the shared Cholesky (an eager `format!` in its
/// O(n³) hot loop); with that removed the finer stride 20 is tractable, and the
/// re-test at stride 20 is what this binary now also supports.
const STRIDE: usize = 40;
/// A-priori RBF bandwidth grid, selected on validation (not tuned on test).
const GAMMA_GRID: [f64; 3] = [0.01, 0.05, 0.1];
/// A-priori ridge grid.
const LAMBDA_GRID: [f64; 2] = [0.1, 1.0];

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

fn rmse(predictions: &[f64], references: &[f64]) -> f64 {
    let sum: f64 = predictions
        .iter()
        .zip(references)
        .map(|(p, r)| (p - r).powi(2))
        .sum();

    (sum / predictions.len() as f64).sqrt()
}

/// Train-fitted per-column standardization.
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

        let mut sds = vec![0.0; cols];

        for row in features
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

fn fit_ols(train_features: &[Vec<f64>], train_targets: &[f64]) -> Option<LinearRegressionModel> {
    let dataset = RegressionDataset {
        features: solvers_matrix(train_features),
        targets: SolversMatrix::from_row_major(train_features.len(), 1, train_targets.to_vec()),
        sample_weights: None,
    };

    let configuration = RobustRegressionConfig {
        method: RobustRegressionMethod::OrdinaryLeastSquares,
        ..RobustRegressionConfig::default()
    };

    fit_robust_regression(&dataset, configuration)
        .ok()
        .map(|report| report.model)
}

fn ols_predict(model: &LinearRegressionModel, features: &[Vec<f64>]) -> Option<Vec<f64>> {
    let matrix = solvers_matrix(features);
    let predictions = model.predict(&matrix).ok()?;
    Some(
        (0..features.len())
            .map(|row| predictions[(row, 0)])
            .collect(),
    )
}

/// The best kernel-ridge configuration found on the validation split.
struct KernelBest {
    gamma: f64,
    ridge: f64,
    model: KernelRidgeRegression,
}

fn select_kernel_ridge(
    train_features: &[Vec<f64>],
    train_targets: &[f64],
    validation_features: &[Vec<f64>],
    validation_targets: &[f64],
) -> KernelBest {
    let mut best: Option<(f64, KernelBest)> = None;

    for &gamma in &GAMMA_GRID
    {
        for &ridge in &LAMBDA_GRID
        {
            let model = KernelRidgeRegression::fit(
                train_features,
                train_targets,
                KernelRidgeConfig { gamma, ridge },
            )
            .expect("kernel ridge fits the standardized design with a positive ridge");

            let validation_rmse = rmse(
                &model.predict_slice(validation_features),
                validation_targets,
            );

            if best
                .as_ref()
                .is_none_or(|(best_rmse, _)| validation_rmse < *best_rmse)
            {
                best = Some((
                    validation_rmse,
                    KernelBest {
                        gamma,
                        ridge,
                        model,
                    },
                ));
            }
        }
    }

    best.expect("the grid is non-empty").1
}

fn evaluate(
    label: &str,
    raw: &TabularDataset,
    config: &RegressionConfig,
    stride: usize,
    records: &mut Vec<BenchRecord>,
    summary: &mut Vec<String>,
) {
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

        let policy = MissingValuePolicy {
            maximum_missing_fraction: config.missing_maximum_fraction,
        };
        let imputer = FittedImputer::fit(&decimated.select_rows(&split.train).features, policy)
            .expect("training keeps a varying column");

        let train = imputer
            .transform(&decimated.select_rows(&split.train))
            .expect("shapes match");
        let validation = imputer
            .transform(&decimated.select_rows(&split.validation))
            .expect("shapes match");
        let test = imputer
            .transform(&decimated.select_rows(&split.test))
            .expect("shapes match");

        let standardizer = Standardizer::fit(&train.features);
        let train_features = standardizer.apply(&train.features);
        let validation_features = standardizer.apply(&validation.features);
        let test_features = standardizer.apply(&test.features);

        let train_mean = train.targets.iter().sum::<f64>() / train.sample_count() as f64;
        let predict_mean_rmse = rmse(&vec![train_mean; test.sample_count()], &test.targets);

        let cell = format!("{label}/{rul_name}/stride_{stride}");

        records.push(BenchRecord::new(
            "nonlinear_rul/ceiling",
            cell.clone(),
            "predict_mean",
            0,
            "rmse",
            predict_mean_rmse,
        ));

        let ols_model = fit_ols(&train_features, &train.targets).expect("OLS fits");
        let ols_train = ols_predict(&ols_model, &train_features).expect("OLS predicts train");
        let ols_test = ols_predict(&ols_model, &test_features).expect("OLS predicts test");
        let ols_ratio = rmse(&ols_test, &test.targets) / predict_mean_rmse;

        let isotonic = IsotonicRegression::fit(
            &ols_train,
            &train.targets,
            MonotoneDirection::NonDecreasing,
            None,
        )
        .expect("finite OLS scores and targets");
        let isotonic_test = isotonic.predict_slice(&ols_test);
        let isotonic_ratio = rmse(&isotonic_test, &test.targets) / predict_mean_rmse;

        let kernel = select_kernel_ridge(
            &train_features,
            &train.targets,
            &validation_features,
            &validation.targets,
        );
        let kernel_test = kernel.model.predict_slice(&test_features);
        let kernel_ratio = rmse(&kernel_test, &test.targets) / predict_mean_rmse;

        for (method, ratio) in [
            ("ols", ols_ratio),
            ("isotonic_ols", isotonic_ratio),
            ("kernel_ridge", kernel_ratio),
        ]
        {
            records.push(BenchRecord::new(
                "nonlinear_rul/ceiling",
                cell.clone(),
                method.to_string(),
                0,
                "task_ceiling_ratio",
                ratio,
            ));
        }

        records.push(BenchRecord::new(
            "nonlinear_rul/ceiling",
            cell.clone(),
            "kernel_ridge",
            0,
            "selected_gamma",
            kernel.gamma,
        ));
        records.push(BenchRecord::new(
            "nonlinear_rul/ceiling",
            cell.clone(),
            "kernel_ridge",
            0,
            "selected_ridge",
            kernel.ridge,
        ));

        summary.push(format!(
            "# {cell:30} ceiling_rmse={predict_mean_rmse:8.3} ols={ols_ratio:.4} \
isotonic={isotonic_ratio:.4} kernel={kernel_ratio:.4} (g={:.2},l={:.1}) kernel_vs_iso={:+.4}",
            kernel.gamma,
            kernel.ridge,
            isotonic_ratio - kernel_ratio,
        ));
    }
}

fn main() {
    let mut data_dir = PathBuf::from("data/industrial");
    let mut out_dir = PathBuf::from("results");
    let mut stride = STRIDE;

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
            "--stride" =>
            {
                stride = value("--stride")
                    .parse()
                    .expect("--stride is a positive integer")
            },
            other => panic!("unknown argument: {other}"),
        }
    }

    assert!(stride > 0, "--stride must be positive");

    let config: Config = serde_json::from_str(CONFIG_TEXT).expect("embedded config is valid");

    let mut records: Vec<BenchRecord> = Vec::new();
    let mut summary: Vec<String> = Vec::new();

    println!(
        "# nonlinear_rul — Axis 1: does a multivariate nonlinear model close the ceiling further?"
    );
    println!(
        "# lower ceiling ratio = more realized; kernel_vs_iso>0 = kernel ridge beats 1-D isotonic"
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
        stride,
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
        stride,
        &mut records,
        &mut summary,
    );

    for line in &summary
    {
        println!("{line}");
    }

    fs::create_dir_all(&out_dir).expect("results directory is writable");
    let out_name = if stride == STRIDE
    {
        "nonlinear_rul.jsonl".to_string()
    }
    else
    {
        format!("nonlinear_rul_stride{stride}.jsonl")
    };
    fs::write(out_dir.join(out_name), to_jsonl(&records)).expect("results file is writable");

    println!("# records={}", records.len());
}
