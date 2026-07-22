//! Follow-up 6 / axis 2 — nonlinear supervised SECOM (degree-2 polynomial LDA).
//!
//! The companion `industrial-secom-drift` binary attacks the ~0.567 frozen-test
//! ceiling from the *temporal* side (recency windowing). This binary attacks it
//! from the *functional* side: it keeps item 2's within-class LDA but lets the
//! discriminant be **nonlinear** in the top-`k` selected features, by expanding
//! them with a degree-2 polynomial map — squares and pairwise products — before
//! fitting the direction. `degree = 1` recovers item 2's linear model exactly, so
//! the sweep contains its own linear baseline; whichever `(degree, k, ridge)`
//! wins on validation is frozen on the test period.
//!
//! The honest constraint is small samples: SECOM fails are rare (~6.6 %), so the
//! training split holds only a few dozen failures. A discriminant with more free
//! parameters than the rarer class is rank-starved, so every configuration whose
//! expanded width exceeds the training failure count is **skipped a priori** — the
//! complexity budget is tied to the data, not chosen to flatter the result. That
//! caps degree-2 at small `k` (its width grows as `k(k+3)/2`), which is exactly
//! the regime where nonlinearity has a chance without overfitting.
//!
//! No kernel and no `O(n³)` solve — the LDA covariance is `d × d` in the expanded
//! width `d` — so this stays fast in debug. Whether low-order nonlinearity lifts
//! the frozen-test AUROC above item 2's 0.567 (and lever 3's 0.581) is reported
//! as-is. Deterministic; run twice, byte-identical.

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

/// Polynomial degrees to sweep; `1` is item 2's linear discriminant.
const DEGREE_GRID: [usize; 2] = [1, 2];
/// Univariate feature-count grid (pre-expansion).
const K_GRID: [usize; 5] = [4, 6, 8, 12, 20];
/// Ridge grid for the within-class covariance.
const RIDGE_GRID: [f64; 3] = [0.1, 1.0, 10.0];

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

    assert!(count > 0, "class {target} is empty");

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

/// Expanded width of a degree-`degree` polynomial map over `k` inputs.
///
/// Degree 1 is the identity (`k`); degree 2 appends the `k` squares and the
/// `k(k-1)/2` pairwise products, i.e. `k + k + k(k-1)/2 = k(k + 3)/2`.
fn expanded_dim(k: usize, degree: usize) -> usize {
    match degree
    {
        1 => k,
        _ => k * (k + 3) / 2,
    }
}

/// Degree-`degree` polynomial feature map of each row (`degree` in `{1, 2}`).
fn expand_polynomial(features: &[Vec<f64>], degree: usize) -> Vec<Vec<f64>> {
    features
        .iter()
        .map(|row| {
            let mut expanded: Vec<f64> = row.clone();

            if degree >= 2
            {
                for (i, &value_i) in row.iter().enumerate()
                {
                    expanded.push(value_i * value_i);

                    for &value_j in &row[i + 1..]
                    {
                        expanded.push(value_i * value_j);
                    }
                }
            }

            expanded
        })
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
        .expect("regularized within-class Mahalanobis fits");

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

/// A fitted nonlinear model: the standardizer, selected columns, polynomial
/// degree and the discriminant direction in the expanded space.
struct NonlinearModel {
    standardizer: Standardizer,
    indices: Vec<usize>,
    degree: usize,
    direction: Vec<f64>,
}

impl NonlinearModel {
    /// Fits standardize → top-`k` select → degree-`degree` expand → within-class
    /// LDA on the training features/labels.
    fn fit(features: &[Vec<f64>], labels: &[f64], k: usize, degree: usize, ridge: f64) -> Self {
        let standardizer = Standardizer::fit(features);
        let standardized = standardizer.apply(features);
        let indices = select_top_k(&univariate_scores(&standardized, labels), k);
        let selected = subset_columns(&standardized, &indices);
        let expanded = expand_polynomial(&selected, degree);
        let direction = fit_lda_direction(&expanded, labels, ridge);

        Self {
            standardizer,
            indices,
            degree,
            direction,
        }
    }

    /// Discriminant scores for arbitrary (un-standardized, full-width) rows.
    fn score(&self, features: &[Vec<f64>]) -> Vec<f64> {
        let standardized = self.standardizer.apply(features);
        let selected = subset_columns(&standardized, &self.indices);
        let expanded = expand_polynomial(&selected, self.degree);
        scores(&self.direction, &expanded)
    }
}

fn class_counts(labels: &[f64]) -> (usize, usize) {
    let fails = labels.iter().filter(|&&label| label == 1.0).count();
    (labels.len() - fails, fails)
}

/// The best `(degree, k, ridge)` found on validation.
struct SweepBest {
    degree: usize,
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

    let (train_passes, train_fails) = class_counts(&train.targets);
    // Cap the expanded discriminant width at the rarer class count: a direction
    // with more free parameters than training failures is rank-starved.
    let complexity_budget = train_fails;

    let mut records: Vec<BenchRecord> = Vec::new();

    println!("# secom_nonlinear — Axis 2: degree-2 polynomial LDA vs the linear ceiling");
    println!(
        "# baselines — unsupervised {UNSUPERVISED_TEST_AUROC}, lever-3 {LEVER3_SUPERVISED_TEST_AUROC}, item-2 {ITEM2_TEST_AUROC}"
    );
    println!(
        "# train passes={train_passes} fails={train_fails}; complexity budget (max expanded width) = {complexity_budget}"
    );
    println!("# columns: degree,k,expanded_dim,ridge,validation_auroc");

    let mut best: Option<SweepBest> = None;

    for &degree in &DEGREE_GRID
    {
        for &k in &K_GRID
        {
            let dimension = expanded_dim(k, degree);
            if dimension > complexity_budget
            {
                println!(
                    "# degree={degree} k={k} skipped: expanded_dim={dimension} > budget {complexity_budget}"
                );
                continue;
            }

            for &ridge in &RIDGE_GRID
            {
                let model = NonlinearModel::fit(&train.features, &train.targets, k, degree, ridge);
                let validation_auroc =
                    auroc(&model.score(&validation.features), &validation.targets)
                        .expect("both classes present in validation");

                println!("{degree},{k},{dimension},{ridge},{validation_auroc:.17e}");

                records.push(BenchRecord::new(
                    "secom_nonlinear/validation",
                    format!("secom/degree_{degree}/k_{k}/ridge_{ridge}"),
                    "polynomial_lda",
                    0,
                    "validation_auroc",
                    validation_auroc,
                ));

                if best
                    .as_ref()
                    .is_none_or(|current| validation_auroc > current.validation_auroc)
                {
                    best = Some(SweepBest {
                        degree,
                        k,
                        ridge,
                        validation_auroc,
                    });
                }
            }
        }
    }

    let best = best.expect("at least one configuration fits the complexity budget");

    // Refit the winning configuration and score the frozen test.
    let model = NonlinearModel::fit(
        &train.features,
        &train.targets,
        best.k,
        best.degree,
        best.ridge,
    );
    let test_auroc =
        auroc(&model.score(&test.features), &test.targets).expect("both classes in test");

    for (metric, quantity) in [
        ("selected_degree", best.degree as f64),
        ("selected_k", best.k as f64),
        (
            "selected_expanded_dim",
            expanded_dim(best.k, best.degree) as f64,
        ),
        ("selected_ridge", best.ridge),
        ("validation_auroc", best.validation_auroc),
        ("test_auroc", test_auroc),
    ]
    {
        records.push(BenchRecord::new(
            "secom_nonlinear/test",
            "secom/temporal_test",
            "polynomial_lda",
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
            "secom_nonlinear/test",
            "secom/temporal_test",
            method,
            0,
            "test_auroc",
            value,
        ));
    }

    println!(
        "# selected degree={} k={} ridge={} (validation AUROC {:.4}); frozen-test AUROC = {test_auroc:.4} \
vs item-2 {ITEM2_TEST_AUROC} vs lever-3 {LEVER3_SUPERVISED_TEST_AUROC} vs unsupervised {UNSUPERVISED_TEST_AUROC}",
        best.degree, best.k, best.ridge, best.validation_auroc,
    );
    println!(
        "# validation→test drift: {:+.4}",
        test_auroc - best.validation_auroc
    );

    fs::create_dir_all(&out_dir).expect("results directory is writable");
    fs::write(out_dir.join("secom_nonlinear.jsonl"), to_jsonl(&records))
        .expect("results file is writable");

    println!("# records={}", records.len());
}
