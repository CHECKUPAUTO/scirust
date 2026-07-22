//! Lever 3 — SECOM supervised reformulation.
//!
//! The phase-728 diagnostic showed SECOM yield failures are **not geometric
//! outliers** (regularized-Mahalanobis in-sample AUROC 0.56; failures sit
//! *closer* to the normal centre than passes), so every unsupervised density
//! detector landed at or below chance on the frozen test (best 0.469). The
//! diagnostic's conclusion was that the SECOM null is a **problem-formulation**
//! result, not a detector-power deficit — the labels are simply not an
//! anomaly-detection target. This binary tests that conclusion the honest
//! way: fit a **supervised** linear discriminant that *uses the labels* and
//! see whether it separates failures the unsupervised detectors could not.
//!
//! Method (reusing the frozen phase-728 SECOM protocol): temporal split
//! (train 0.6 / validation 0.2 / test), train-fitted `FittedImputer`
//! (no leakage). On the imputed **train** split only, the regularized linear
//! discriminant direction is
//!
//! ```text
//!   w = Σ⁻¹ (μ_fail − μ_pass)
//! ```
//!
//! where `μ_pass`, `μ_fail` are the class means and `Σ⁻¹` is the
//! ridge-regularized inverse covariance from
//! `scirust_multivariate::FittedDistanceMetric::fit_regularized_mahalanobis`.
//! The score `w · x` is projected onto validation and test; the ridge is
//! **selected on validation AUROC** (from the phase-728 grid {0.001, 0.1}),
//! then frozen and reported on test.
//!
//! Honesty: the supervised discriminant uses labels the unsupervised
//! detectors did not — this is a *reformulation*, not a fairer version of the
//! same task. SECOM remains a hard dataset; the number reported is whatever
//! it is. The claim under test is narrow: if a simple supervised linear
//! direction clears chance on the frozen test where every unsupervised
//! density detector failed, the phase-728 SECOM null was about **problem
//! formulation**, not detector power. Deterministic; run twice, byte-identical.

use std::fs;
use std::path::{Path, PathBuf};

use scirust_bench_schema::{BenchRecord, to_jsonl};
use scirust_multivariate::{FittedDistanceMetric, Matrix as MultivariateMatrix};
use scirust_srcc_bench::{
    FittedImputer, MissingValuePolicy, SplitStrategy, TabularDataset, metrics::auroc, parse_secom,
    sha256_hex, split_dataset,
};
use serde::Deserialize;

const CONFIG_TEXT: &str = include_str!("../../configs/phase728.json");

const SECOM_DATA_SHA: &str = "20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c";
const SECOM_LABELS_SHA: &str = "126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03";

/// The phase-728 unsupervised frozen-test baseline (best detector, from the
/// diagnostic): regularized Mahalanobis AUROC.
const UNSUPERVISED_TEST_AUROC: f64 = 0.469;

#[derive(Deserialize)]
struct Config {
    secom: SecomConfig,
}

#[derive(Deserialize)]
struct SecomConfig {
    train_fraction: f64,
    validation_fraction: f64,
    missing_maximum_fraction: f64,
    mahalanobis_ridge_grid: Vec<f64>,
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

/// Class mean of the rows whose label equals `target`.
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

/// The regularized linear discriminant direction `w = Σ⁻¹ (μ_fail − μ_pass)`.
fn discriminant_direction(
    inverse_scatter: &MultivariateMatrix,
    mu_fail: &[f64],
    mu_pass: &[f64],
) -> Vec<f64> {
    let dimension = mu_fail.len();
    let delta: Vec<f64> = (0..dimension).map(|i| mu_fail[i] - mu_pass[i]).collect();

    (0..dimension)
        .map(|i| {
            (0..dimension)
                .map(|j| inverse_scatter.data[i][j] * delta[j])
                .sum()
        })
        .collect()
}

/// The discriminant score `w · x`.
fn project(direction: &[f64], row: &[f64]) -> f64 {
    direction.iter().zip(row).map(|(w, x)| w * x).sum()
}

fn scores(direction: &[f64], features: &[Vec<f64>]) -> Vec<f64> {
    features.iter().map(|row| project(direction, row)).collect()
}

fn fit_direction(train: &TabularDataset, ridge: f64) -> Vec<f64> {
    let mu_pass = class_mean(&train.features, &train.targets, 0.0);
    let mu_fail = class_mean(&train.features, &train.targets, 1.0);

    let matrix = MultivariateMatrix {
        rows: train.sample_count(),
        cols: train.feature_count(),
        data: train.features.clone(),
    };

    let metric = FittedDistanceMetric::fit_regularized_mahalanobis(&matrix, ridge)
        .expect("regularized Mahalanobis fits the imputed SECOM train");

    let FittedDistanceMetric::RegularizedMahalanobis {
        inverse_scatter, ..
    } = &metric
    else
    {
        panic!("expected a regularized Mahalanobis metric");
    };

    discriminant_direction(inverse_scatter, &mu_fail, &mu_pass)
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

    let mut records: Vec<BenchRecord> = Vec::new();

    println!("# secom_supervised — Lever 3: does a supervised linear discriminant separate SECOM?");
    println!(
        "# unsupervised frozen-test baseline (phase 728, best detector): AUROC={UNSUPERVISED_TEST_AUROC}"
    );
    println!("# columns: ridge,train_auroc,validation_auroc (selection on validation)");

    // Ridge selection on the validation split only.
    let mut best: Option<(f64, f64, Vec<f64>)> = None;

    for &ridge in &secom.mahalanobis_ridge_grid
    {
        let direction = fit_direction(&train, ridge);

        let train_auroc = auroc(&scores(&direction, &train.features), &train.targets)
            .expect("both classes present in train");
        let validation_auroc = auroc(
            &scores(&direction, &validation.features),
            &validation.targets,
        )
        .expect("both classes present in validation");

        println!("{ridge},{train_auroc:.17e},{validation_auroc:.17e}");

        records.push(BenchRecord::new(
            "secom_supervised/validation",
            format!("secom/ridge_{ridge}"),
            "linear_discriminant",
            0,
            "validation_auroc",
            validation_auroc,
        ));
        records.push(BenchRecord::new(
            "secom_supervised/validation",
            format!("secom/ridge_{ridge}"),
            "linear_discriminant",
            0,
            "train_auroc",
            train_auroc,
        ));

        if best
            .as_ref()
            .is_none_or(|(_, best_val, _)| validation_auroc > *best_val)
        {
            best = Some((ridge, validation_auroc, direction));
        }
    }

    let (selected_ridge, selected_validation_auroc, direction) =
        best.expect("the ridge grid is non-empty");

    // Frozen-test evaluation with the validation-selected ridge.
    let test_auroc = auroc(&scores(&direction, &test.features), &test.targets)
        .expect("both classes present in test");

    records.push(BenchRecord::new(
        "secom_supervised/test",
        "secom/temporal_test",
        "linear_discriminant",
        0,
        "selected_ridge",
        selected_ridge,
    ));
    records.push(BenchRecord::new(
        "secom_supervised/test",
        "secom/temporal_test",
        "linear_discriminant",
        0,
        "test_auroc",
        test_auroc,
    ));
    records.push(BenchRecord::new(
        "secom_supervised/test",
        "secom/temporal_test",
        "unsupervised_baseline",
        0,
        "test_auroc",
        UNSUPERVISED_TEST_AUROC,
    ));

    println!(
        "# selected ridge={selected_ridge} (validation AUROC {selected_validation_auroc:.4}); \
frozen-test AUROC = {test_auroc:.4} vs unsupervised {UNSUPERVISED_TEST_AUROC}"
    );

    fs::create_dir_all(&out_dir).expect("results directory is writable");
    fs::write(out_dir.join("secom_supervised.jsonl"), to_jsonl(&records))
        .expect("results file is writable");

    println!("# records={}", records.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discriminant_direction_with_identity_scatter_is_the_mean_difference() {
        // Σ⁻¹ = I → w = μ_fail − μ_pass.
        let inverse_scatter = MultivariateMatrix {
            rows: 2,
            cols: 2,
            data: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };

        let w = discriminant_direction(&inverse_scatter, &[3.0, 5.0], &[1.0, 2.0]);

        assert_eq!(w, vec![2.0, 3.0]);
    }

    #[test]
    fn discriminant_direction_applies_the_inverse_scatter() {
        // Σ⁻¹ = [[2,0],[0,0.5]], δ = [1, 4] → w = [2, 2].
        let inverse_scatter = MultivariateMatrix {
            rows: 2,
            cols: 2,
            data: vec![vec![2.0, 0.0], vec![0.0, 0.5]],
        };

        let w = discriminant_direction(&inverse_scatter, &[1.0, 4.0], &[0.0, 0.0]);

        assert_eq!(w, vec![2.0, 2.0]);
    }

    #[test]
    fn project_is_the_dot_product() {
        assert_eq!(project(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn class_mean_averages_only_the_target_label() {
        let features = vec![vec![0.0, 0.0], vec![2.0, 4.0], vec![10.0, 10.0]];
        let labels = vec![1.0, 1.0, 0.0];

        assert_eq!(class_mean(&features, &labels, 1.0), vec![1.0, 2.0]);
        assert_eq!(class_mean(&features, &labels, 0.0), vec![10.0, 10.0]);
    }
}
