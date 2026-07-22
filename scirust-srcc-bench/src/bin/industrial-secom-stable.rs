//! Follow-up 2 — stabilizing the supervised SECOM discriminant.
//!
//! Lever 3 ([`industrial-secom-supervised`]) showed the SECOM null was a
//! problem-*formulation* result: a supervised linear discriminant clears chance
//! on the frozen test (AUROC 0.581) where every unsupervised density detector
//! failed (0.469). But it also over-fit hard — all ~416 imputed features, a
//! total-scatter covariance, ridge only in {0.001, 0.1}, in-sample AUROC 0.975
//! collapsing to 0.40–0.43 on validation. This binary builds the model lever 3
//! identified but did not fit: a **regularized LDA** that actually tries to
//! generalize, via three additive stabilizers, each aimed at the p ≫ n overfit:
//!
//! 1. **Standardization** — train-fitted per-feature z-scoring, so the ridge and
//!    the feature scores are scale-fair.
//! 2. **Univariate feature selection** — keep the top-`k` features by the
//!    absolute pooled-standardized (Cohen's-d) mean difference between the
//!    classes on the **train** split only, cutting ~416 → `k` and making the
//!    covariance estimate sound (`k` ≪ the ~90 minority-class rows).
//! 3. **Within-class (Fisher) covariance** — the discriminant uses the pooled
//!    *within-class* scatter `S_W⁻¹`, not lever 3's total scatter, obtained by
//!    centering each row by its own class mean before the regularized-Mahalanobis
//!    fit. `w = S_W⁻¹ (μ_fail − μ_pass)`.
//!
//! `(k, ridge)` is selected on **validation** AUROC over fixed a-priori grids
//! (never on test), then frozen and reported on the same phase-728 temporal
//! test split with the train-fitted [`FittedImputer`] (no leakage).
//!
//! Honesty: SECOM stays hard and drifts; this reports whatever the frozen-test
//! AUROC is, against both the lever-3 supervised number (0.581) and the
//! unsupervised baseline (0.469). Feature selection and standardization curb
//! variance; they cannot manufacture signal, and the validation→test gap is
//! reported as the drift it is. Deterministic; run twice, byte-identical.

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

/// A-priori feature-count grid for univariate selection (fixed, not tuned on
/// test); each entry stays well below the minority-class row count.
const K_GRID: [usize; 5] = [5, 10, 20, 40, 80];
/// A-priori ridge grid, richer than the frozen phase-728 {0.001, 0.1} so heavier
/// shrinkage is available for the small-sample within-class covariance.
const RIDGE_GRID: [f64; 4] = [0.01, 0.1, 1.0, 10.0];

/// Floor for a standard deviation before dividing, so a (near-)constant train
/// column cannot produce a non-finite standardized value.
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

    assert!(count > 0, "class {target} is empty in the training split");

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

/// Absolute pooled-standardized (Cohen's-d) mean difference per feature, the
/// univariate selection score on the train split.
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

/// Indices of the top-`k` features by score, returned in ascending index order
/// (descending score with an ascending-index tie-break selects them).
fn select_top_k(scores: &[f64], k: usize) -> Vec<usize> {
    let mut ranked: Vec<usize> = (0..scores.len()).collect();
    ranked.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    ranked.truncate(k.min(scores.len()));
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

/// The regularized within-class LDA direction `w = S_W⁻¹ (μ_fail − μ_pass)` on
/// the selected feature space.
fn fit_lda_direction(features: &[Vec<f64>], labels: &[f64], ridge: f64) -> Vec<f64> {
    let mu_pass = class_mean(features, labels, 0.0);
    let mu_fail = class_mean(features, labels, 1.0);

    // Center each row by its own class mean → the concatenated rows have ~zero
    // global mean, so the regularized-Mahalanobis fit's total scatter *is* the
    // pooled within-class scatter S_W.
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
        .expect("regularized within-class Mahalanobis fits the selected SECOM train");

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

/// The best `(k, ridge)` configuration found on the validation split, with the
/// selected feature indices and the fitted discriminant kept for the frozen-test
/// evaluation.
struct SweepBest {
    k: usize,
    ridge: f64,
    validation_auroc: f64,
    indices: Vec<usize>,
    direction: Vec<f64>,
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

    let train_raw = imputer
        .transform(&data.select_rows(&split.train))
        .expect("shapes match");
    let validation_raw = imputer
        .transform(&data.select_rows(&split.validation))
        .expect("shapes match");
    let test_raw = imputer
        .transform(&data.select_rows(&split.test))
        .expect("shapes match");

    // Standardize with train statistics (no leakage).
    let standardizer = Standardizer::fit(&train_raw.features);
    let train_features = standardizer.apply(&train_raw.features);
    let validation_features = standardizer.apply(&validation_raw.features);
    let test_features = standardizer.apply(&test_raw.features);

    let feature_scores = univariate_scores(&train_features, &train_raw.targets);
    let available = feature_scores.len();

    let mut records: Vec<BenchRecord> = Vec::new();

    println!("# secom_stable — Follow-up 2: standardized, feature-selected, within-class LDA");
    println!(
        "# baselines — unsupervised {UNSUPERVISED_TEST_AUROC}, lever-3 supervised {LEVER3_SUPERVISED_TEST_AUROC}"
    );
    println!("# imputed features available: {available}");
    println!("# columns: k,ridge,validation_auroc (selection on validation)");

    // Grid search on validation AUROC over (k, ridge), fixed a-priori grids.
    let mut best: Option<SweepBest> = None;

    for &k in &K_GRID
    {
        if k > available
        {
            continue;
        }

        let indices = select_top_k(&feature_scores, k);
        let train_selected = subset_columns(&train_features, &indices);
        let validation_selected = subset_columns(&validation_features, &indices);

        for &ridge in &RIDGE_GRID
        {
            let direction = fit_lda_direction(&train_selected, &train_raw.targets, ridge);
            let validation_auroc = auroc(
                &scores(&direction, &validation_selected),
                &validation_raw.targets,
            )
            .expect("both classes present in validation");

            println!("{k},{ridge},{validation_auroc:.17e}");

            records.push(BenchRecord::new(
                "secom_stable/validation",
                format!("secom/k_{k}/ridge_{ridge}"),
                "within_class_lda",
                0,
                "validation_auroc",
                validation_auroc,
            ));

            if best
                .as_ref()
                .is_none_or(|current| validation_auroc > current.validation_auroc)
            {
                best = Some(SweepBest {
                    k,
                    ridge,
                    validation_auroc,
                    indices: indices.clone(),
                    direction,
                });
            }
        }
    }

    let SweepBest {
        k: selected_k,
        ridge: selected_ridge,
        validation_auroc: selected_validation_auroc,
        indices,
        direction,
    } = best.expect("at least one (k, ridge) configuration is evaluated");

    // Frozen-test evaluation with the validation-selected (k, ridge).
    let test_selected = subset_columns(&test_features, &indices);
    let test_auroc = auroc(&scores(&direction, &test_selected), &test_raw.targets)
        .expect("both classes in test");

    for (metric, quantity) in [
        ("selected_k", selected_k as f64),
        ("selected_ridge", selected_ridge),
        ("validation_auroc", selected_validation_auroc),
        ("test_auroc", test_auroc),
    ]
    {
        records.push(BenchRecord::new(
            "secom_stable/test",
            "secom/temporal_test",
            "within_class_lda",
            0,
            metric,
            quantity,
        ));
    }

    records.push(BenchRecord::new(
        "secom_stable/test",
        "secom/temporal_test",
        "lever3_supervised_baseline",
        0,
        "test_auroc",
        LEVER3_SUPERVISED_TEST_AUROC,
    ));
    records.push(BenchRecord::new(
        "secom_stable/test",
        "secom/temporal_test",
        "unsupervised_baseline",
        0,
        "test_auroc",
        UNSUPERVISED_TEST_AUROC,
    ));

    println!(
        "# selected k={selected_k} ridge={selected_ridge} (validation AUROC \
{selected_validation_auroc:.4}); frozen-test AUROC = {test_auroc:.4} \
vs lever-3 {LEVER3_SUPERVISED_TEST_AUROC} vs unsupervised {UNSUPERVISED_TEST_AUROC}"
    );
    println!(
        "# validation→test drift: {:+.4}",
        test_auroc - selected_validation_auroc
    );

    fs::create_dir_all(&out_dir).expect("results directory is writable");
    fs::write(out_dir.join("secom_stable.jsonl"), to_jsonl(&records))
        .expect("results file is writable");

    println!("# records={}", records.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn standardizer_centres_and_scales_to_unit_variance() {
        let features = vec![vec![0.0], vec![2.0], vec![4.0]];
        let standardizer = Standardizer::fit(&features);
        assert!(approx_eq(standardizer.means[0], 2.0, 1e-12));

        let out = standardizer.apply(&features);
        let mean: f64 = out.iter().map(|row| row[0]).sum::<f64>() / 3.0;
        let var: f64 = out.iter().map(|row| row[0] * row[0]).sum::<f64>() / 3.0;
        assert!(approx_eq(mean, 0.0, 1e-12));
        assert!(approx_eq(var, 1.0, 1e-12));
    }

    #[test]
    fn constant_column_does_not_produce_non_finite_values() {
        let features = vec![vec![7.0], vec![7.0], vec![7.0]];
        let out = Standardizer::fit(&features).apply(&features);
        assert!(out.iter().all(|row| row[0].is_finite()));
    }

    #[test]
    fn univariate_score_ranks_the_separating_feature_first() {
        // Column 0 separates the classes perfectly; column 1 is pure noise.
        let features = vec![
            vec![0.0, 5.0],
            vec![0.0, -5.0],
            vec![10.0, 5.0],
            vec![10.0, -5.0],
        ];
        let labels = vec![0.0, 0.0, 1.0, 1.0];
        let s = univariate_scores(&features, &labels);
        assert!(s[0] > s[1]);
        assert_eq!(select_top_k(&s, 1), vec![0]);
    }

    #[test]
    fn select_top_k_returns_ascending_indices_and_breaks_ties_by_index() {
        // Equal scores → the lowest indices win, returned ascending.
        let scores = vec![1.0, 1.0, 1.0, 0.0];
        assert_eq!(select_top_k(&scores, 2), vec![0, 1]);
        // Distinct scores → the two largest (indices 2, 3), returned ascending.
        let scores = vec![0.1, 0.2, 0.9, 0.8];
        assert_eq!(select_top_k(&scores, 2), vec![2, 3]);
    }

    #[test]
    fn within_class_lda_separates_a_linearly_separable_toy() {
        // Two Gaussians offset along +x; the LDA direction must order fails above
        // passes, giving a perfect in-sample AUROC.
        let mut features = Vec::new();
        let mut labels = Vec::new();
        for i in 0..8
        {
            let jitter = (i as f64) * 0.01;
            features.push(vec![jitter, 1.0 + jitter]);
            labels.push(0.0);
            features.push(vec![5.0 + jitter, 1.0 - jitter]);
            labels.push(1.0);
        }

        let direction = fit_lda_direction(&features, &labels, 0.1);
        let auroc = auroc(&scores(&direction, &features), &labels).unwrap();
        assert!(
            auroc > 0.99,
            "expected near-perfect separation, got {auroc}"
        );
    }
}
