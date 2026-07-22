//! Follow-up 6 / axis 2 — drift-corrected supervised SECOM.
//!
//! Item 2 stabilized the supervised SECOM discriminant (standardization +
//! univariate feature selection + within-class LDA) into a *selectable*
//! ~0.567 frozen-test AUROC, but could not raise the ceiling — the residual
//! barrier the diagnostic named is **temporal drift**: the SECOM process moves,
//! so the earliest training wafers are the least like the test period. This
//! binary attacks that directly, staying linear (and fast): it fits the item-2
//! within-class LDA on a **recency window** — the most-recent fraction `w` of the
//! temporally-ordered train split, closest to the held-out test regime — and
//! selects `(w, k, ridge)` on validation, frozen on test.
//!
//! Everything (standardization, univariate top-`k` selection, the within-class
//! discriminant) is fit on the window only; `w = 1.0` recovers item 2's
//! all-train model. A window with too few of either class is skipped (a
//! discriminant needs both). No kernel and no `O(n³)` solve — the LDA covariance
//! is `k × k` — so this runs fast in debug.
//!
//! Honest framing: SECOM is a hard, drifting dataset; whether trimming stale
//! training data lifts the frozen-test AUROC above item 2's 0.567 (and lever 3's
//! 0.581) is reported as-is. Deterministic; run twice, byte-identical.

use std::fs;
use std::path::{Path, PathBuf};

use scirust_bench_schema::{BenchRecord, to_jsonl};
use scirust_multivariate::{FittedDistanceMetric, Matrix as MultivariateMatrix};
use scirust_srcc_bench::{
    FittedImputer, MissingValuePolicy, SplitStrategy, metrics::auroc, parse_secom, sha256_hex,
    split_dataset,
};
use serde::Deserialize;

const CONFIG_TEXT: &str = include_str!("../../configs/phase728.json");

const SECOM_DATA_SHA: &str = "20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c";
const SECOM_LABELS_SHA: &str = "126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03";

/// The phase-728 unsupervised frozen-test baseline (best density detector).
const UNSUPERVISED_TEST_AUROC: f64 = 0.469;
/// The lever-3 supervised frozen-test AUROC (all features, total scatter).
const LEVER3_SUPERVISED_TEST_AUROC: f64 = 0.581;
/// The item-2 stabilized frozen-test AUROC (all train, within-class LDA).
const ITEM2_TEST_AUROC: f64 = 0.567;

/// Recency-window fractions of the train split (most-recent rows); `1.0` = all
/// train = item 2's model. Fixed a priori.
const WINDOW_GRID: [f64; 4] = [0.4, 0.6, 0.8, 1.0];
/// Univariate feature-count grid.
const K_GRID: [usize; 4] = [5, 10, 20, 40];
/// Ridge grid for the within-class covariance.
const RIDGE_GRID: [f64; 3] = [0.1, 1.0, 10.0];

/// Minimum rows of each class a window must hold to fit a discriminant.
const MIN_CLASS: usize = 5;
/// Floor for a standard deviation before dividing.
const SD_FLOOR: f64 = 1.0e-12;

#[derive(Deserialize)]
struct Config {
    secom: SecomConfig,
}

#[derive(Deserialize)]
struct SecomConfig {
    train_fraction: f64,
    validation_fraction: f64,
    missing_maximum_fraction: f64,
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

/// Class mean over the rows whose label equals `target`.
fn class_mean(features: &[Vec<f64>], labels: &[f64], target: f64) -> Vec<f64> {
    let columns = features.first().map_or(0, Vec::len);
    let mut sum = vec![0.0; columns];
    let mut count = 0usize;

    for (row, &label) in features.iter().zip(labels)
    {
        if label == target
        {
            for (accumulator, value) in sum.iter_mut().zip(row)
            {
                *accumulator += value;
            }

            count += 1;
        }
    }

    assert!(count > 0, "class {target} is empty in the window");

    for value in &mut sum
    {
        *value /= count as f64;
    }

    sum
}

/// Per-column within-class variance for one class (population form).
fn class_variance(features: &[Vec<f64>], labels: &[f64], target: f64, mean: &[f64]) -> Vec<f64> {
    let columns = mean.len();
    let mut sum = vec![0.0; columns];
    let mut count = 0usize;

    for (row, &label) in features.iter().zip(labels)
    {
        if label == target
        {
            for (index, value) in row.iter().enumerate()
            {
                let delta = value - mean[index];
                sum[index] += delta * delta;
            }

            count += 1;
        }
    }

    for value in &mut sum
    {
        *value /= count as f64;
    }

    sum
}

/// Absolute pooled-standardized (Cohen's-d) mean difference per feature.
fn univariate_scores(features: &[Vec<f64>], labels: &[f64]) -> Vec<f64> {
    let mu_pass = class_mean(features, labels, 0.0);
    let mu_fail = class_mean(features, labels, 1.0);
    let var_pass = class_variance(features, labels, 0.0, &mu_pass);
    let var_fail = class_variance(features, labels, 1.0, &mu_fail);

    (0..mu_pass.len())
        .map(|index| {
            let pooled_sd = (0.5 * (var_pass[index] + var_fail[index]))
                .sqrt()
                .max(SD_FLOOR);
            (mu_fail[index] - mu_pass[index]).abs() / pooled_sd
        })
        .collect()
}

/// Indices of the top-`k` features by score, ascending index order.
fn select_top_k(scores_by_feature: &[f64], k: usize) -> Vec<usize> {
    let mut ranked: Vec<usize> = (0..scores_by_feature.len()).collect();
    ranked.sort_by(|&a, &b| {
        scores_by_feature[b]
            .total_cmp(&scores_by_feature[a])
            .then(a.cmp(&b))
    });
    ranked.truncate(k.min(scores_by_feature.len()));
    ranked.sort_unstable();
    ranked
}

/// Projects each row onto the selected columns.
fn subset_columns(features: &[Vec<f64>], indices: &[usize]) -> Vec<Vec<f64>> {
    features
        .iter()
        .map(|row| indices.iter().map(|&index| row[index]).collect())
        .collect()
}

/// The regularized within-class LDA direction `w = S_W⁻¹ (μ_fail − μ_pass)`.
fn fit_lda_direction(features: &[Vec<f64>], labels: &[f64], ridge: f64) -> Vec<f64> {
    let mu_pass = class_mean(features, labels, 0.0);
    let mu_fail = class_mean(features, labels, 1.0);

    let within: Vec<Vec<f64>> = features
        .iter()
        .zip(labels)
        .map(|(row, &label)| {
            let mean = if label == 1.0 { &mu_fail } else { &mu_pass };
            row.iter()
                .zip(mean)
                .map(|(value, centre)| value - centre)
                .collect()
        })
        .collect();

    let matrix = MultivariateMatrix {
        rows: within.len(),
        cols: mu_pass.len(),
        data: within,
    };

    let metric = FittedDistanceMetric::fit_regularized_mahalanobis(&matrix, ridge)
        .expect("regularized within-class Mahalanobis fits the window");

    let FittedDistanceMetric::RegularizedMahalanobis {
        inverse_scatter, ..
    } = &metric
    else
    {
        panic!("expected a regularized Mahalanobis metric");
    };

    let dimension = mu_pass.len();
    let delta: Vec<f64> = (0..dimension).map(|i| mu_fail[i] - mu_pass[i]).collect();

    (0..dimension)
        .map(|i| {
            (0..dimension)
                .map(|j| inverse_scatter.data[i][j] * delta[j])
                .sum()
        })
        .collect()
}

/// The discriminant scores `w · x` for every row.
fn scores(direction: &[f64], features: &[Vec<f64>]) -> Vec<f64> {
    features
        .iter()
        .map(|row| direction.iter().zip(row).map(|(w, x)| w * x).sum())
        .collect()
}

/// A fitted drift-corrected model: the window standardizer, selected columns and
/// discriminant direction.
struct WindowModel {
    standardizer: Standardizer,
    indices: Vec<usize>,
    direction: Vec<f64>,
}

impl WindowModel {
    /// Fits the full item-2 pipeline on the window features/targets.
    fn fit(features: &[Vec<f64>], labels: &[f64], k: usize, ridge: f64) -> Self {
        let standardizer = Standardizer::fit(features);
        let standardized = standardizer.apply(features);
        let indices = select_top_k(&univariate_scores(&standardized, labels), k);
        let selected = subset_columns(&standardized, &indices);
        let direction = fit_lda_direction(&selected, labels, ridge);

        Self {
            standardizer,
            indices,
            direction,
        }
    }

    /// Discriminant scores for arbitrary (un-standardized, full-width) rows.
    fn score(&self, features: &[Vec<f64>]) -> Vec<f64> {
        let standardized = self.standardizer.apply(features);
        let selected = subset_columns(&standardized, &self.indices);
        scores(&self.direction, &selected)
    }
}

fn class_counts(labels: &[f64]) -> (usize, usize) {
    let fails = labels.iter().filter(|&&label| label == 1.0).count();
    (labels.len() - fails, fails)
}

/// The best `(w, k, ridge)` found on validation.
struct SweepBest {
    window_fraction: f64,
    k: usize,
    ridge: f64,
    validation_auroc: f64,
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
    let secom = &config.secom;

    let data = parse_secom(
        &read_verified(&data_dir.join("secom/secom.data"), SECOM_DATA_SHA),
        &read_verified(&data_dir.join("secom/secom_labels.data"), SECOM_LABELS_SHA),
    )
    .expect("secom parses");

    let split = split_dataset(
        &data,
        &SplitStrategy::Temporal {
            train_fraction: secom.train_fraction,
            validation_fraction: secom.validation_fraction,
        },
        0,
        None,
    )
    .expect("temporal split is valid");

    let imputer = FittedImputer::fit(
        &data.select_rows(&split.train).features,
        MissingValuePolicy {
            maximum_missing_fraction: secom.missing_maximum_fraction,
        },
    )
    .expect("secom training keeps varying columns");

    let train = imputer
        .transform(&data.select_rows(&split.train))
        .expect("shapes match");
    let validation = imputer
        .transform(&data.select_rows(&split.validation))
        .expect("shapes match");
    let test = imputer
        .transform(&data.select_rows(&split.test))
        .expect("shapes match");

    let train_rows = train.sample_count();

    let mut records: Vec<BenchRecord> = Vec::new();

    println!("# secom_drift — Axis 2: recency-window drift correction of the supervised LDA");
    println!(
        "# baselines — unsupervised {UNSUPERVISED_TEST_AUROC}, lever-3 {LEVER3_SUPERVISED_TEST_AUROC}, item-2 {ITEM2_TEST_AUROC}"
    );
    println!("# train rows={train_rows}; columns: window,k,ridge,validation_auroc");

    let mut best: Option<SweepBest> = None;

    for &window_fraction in &WINDOW_GRID
    {
        let window_rows = ((window_fraction * train_rows as f64).round() as usize).max(1);
        let start = train_rows - window_rows;
        let window_features = &train.features[start..];
        let window_targets = &train.targets[start..];

        let (passes, fails) = class_counts(window_targets);
        if passes < MIN_CLASS || fails < MIN_CLASS
        {
            println!(
                "# window={window_fraction} skipped: passes={passes} fails={fails} (< {MIN_CLASS})"
            );
            continue;
        }

        for &k in &K_GRID
        {
            for &ridge in &RIDGE_GRID
            {
                let model = WindowModel::fit(window_features, window_targets, k, ridge);
                let validation_auroc =
                    auroc(&model.score(&validation.features), &validation.targets)
                        .expect("both classes present in validation");

                println!("{window_fraction},{k},{ridge},{validation_auroc:.17e}");

                records.push(BenchRecord::new(
                    "secom_drift/validation",
                    format!("secom/w_{window_fraction}/k_{k}/ridge_{ridge}"),
                    "recency_window_lda",
                    0,
                    "validation_auroc",
                    validation_auroc,
                ));

                if best
                    .as_ref()
                    .is_none_or(|current| validation_auroc > current.validation_auroc)
                {
                    best = Some(SweepBest {
                        window_fraction,
                        k,
                        ridge,
                        validation_auroc,
                    });
                }
            }
        }
    }

    let best = best.expect("at least one window configuration is evaluable");

    // Refit the winning configuration and score the frozen test.
    let window_rows = ((best.window_fraction * train_rows as f64).round() as usize).max(1);
    let start = train_rows - window_rows;
    let model = WindowModel::fit(
        &train.features[start..],
        &train.targets[start..],
        best.k,
        best.ridge,
    );
    let test_auroc =
        auroc(&model.score(&test.features), &test.targets).expect("both classes in test");

    for (metric, quantity) in [
        ("selected_window_fraction", best.window_fraction),
        ("selected_k", best.k as f64),
        ("selected_ridge", best.ridge),
        ("validation_auroc", best.validation_auroc),
        ("test_auroc", test_auroc),
    ]
    {
        records.push(BenchRecord::new(
            "secom_drift/test",
            "secom/temporal_test",
            "recency_window_lda",
            0,
            metric,
            quantity,
        ));
    }

    for (method, value) in [
        ("item2_baseline", ITEM2_TEST_AUROC),
        ("lever3_supervised_baseline", LEVER3_SUPERVISED_TEST_AUROC),
        ("unsupervised_baseline", UNSUPERVISED_TEST_AUROC),
    ]
    {
        records.push(BenchRecord::new(
            "secom_drift/test",
            "secom/temporal_test",
            method,
            0,
            "test_auroc",
            value,
        ));
    }

    println!(
        "# selected window={} k={} ridge={} (validation AUROC {:.4}); frozen-test AUROC = {test_auroc:.4} \
vs item-2 {ITEM2_TEST_AUROC} vs lever-3 {LEVER3_SUPERVISED_TEST_AUROC} vs unsupervised {UNSUPERVISED_TEST_AUROC}",
        best.window_fraction, best.k, best.ridge, best.validation_auroc,
    );
    println!(
        "# validation→test drift: {:+.4}",
        test_auroc - best.validation_auroc
    );

    fs::create_dir_all(&out_dir).expect("results directory is writable");
    fs::write(out_dir.join("secom_drift.jsonl"), to_jsonl(&records))
        .expect("results file is writable");

    println!("# records={}", records.len());
}
