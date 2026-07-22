//! Phase-728 **diagnostic**: why did the contamination-robustness machinery
//! not beat the plain baselines on real industrial data?
//!
//! Phase 728 returned an honest null on its central contamination questions:
//! robust regression was inconclusive (Huber beat OLS on FD003 but not
//! FD001, so replication failed), and every unsupervised anomaly detector
//! landed at or below chance on the frozen SECOM test. Before proposing any
//! *more powerful* robust algorithm, this binary asks whether contamination
//! robustness is even the bottleneck — by decomposing each null into its
//! causal parts. It is exploratory re-analysis, not a new preregistered
//! test: it reuses the frozen phase-728 configuration
//! (`configs/phase728.json`) verbatim so the splits, decimation, imputation
//! and contamination match exactly what phase 728 evaluated.
//!
//! # Regression decomposition (FD001, FD003)
//!
//! For each method and contamination level, the test RMSE of the
//! contaminated fit is compared against two references computed on the same
//! split:
//!
//! - **clean-fit RMSE** — the method fitted on the *uncontaminated* training
//!   split. This is the best a perfectly robust method could recover; the
//!   gap `contaminated − clean_fit` is the *robustness recovery* residual.
//! - **predict-the-mean RMSE** — the trivial model that ignores the
//!   features and returns the training-target mean. `clean_fit / mean`
//!   near 1 means the features barely predict the target: a low task
//!   ceiling, where no estimator (robust or not) has room to win.
//!
//! plus the **prediction shift** `RMS(pred_contaminated − pred_clean)` over
//! the test set — how much the injected contamination actually moved the
//! fitted function, in target units. If OLS's prediction shift is small, the
//! contamination is not a strong attack on this design and robustness has
//! nothing to correct.
//!
//! # SECOM decomposition
//!
//! - **base rates** — failure fraction in train vs test (a shifted base rate
//!   alone can sink a frozen-threshold detector);
//! - **feature drift** — per kept feature, the standardized central shift
//!   `|median_test − median_train| / MAD_train`; reported as mean, max, and
//!   the count of features shifted by more than one train-MAD;
//! - **failure separability** — regularized Mahalanobis distance to the
//!   train-normal centre, split by label, with the distance-vs-label AUROC
//!   computed **in-sample (train)** and on the **frozen test**. Train AUROC
//!   near 0.5 means failures are not geometric outliers even in-sample (a
//!   wrong-tool result); a decent train AUROC that collapses on test means
//!   distribution shift.
//!
//! Output: `results/diagnostic_728.jsonl` (`BenchRecord`s) plus a
//! deterministic stdout summary. No network, checksum-verified inputs, no
//! timings in hashed content. Run twice and compare byte-for-byte.

use std::fs;
use std::path::{Path, PathBuf};

use scirust_bench_schema::{BenchRecord, to_jsonl};
use scirust_learning::{
    RegressionDataset, RobustLoss, RobustRegressionConfig, RobustRegressionMethod,
    fit_robust_regression,
};
use scirust_multivariate::{FittedDistanceMetric, Matrix as MultivariateMatrix};
use scirust_solvers::linalg::Matrix as SolversMatrix;
use scirust_srcc_bench::{
    ContaminationConfig, ContaminationKind, FittedImputer, MissingValuePolicy, SplitStrategy,
    TabularDataset, apply_contamination, metrics::auroc, parse_cmapss_training, parse_secom,
    sha256_hex, split_dataset,
};
use serde::Deserialize;

const CONFIG_TEXT: &str = include_str!("../../configs/phase728.json");

const TRAIN_FD001_SHA: &str = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8";
const TRAIN_FD003_SHA: &str = "2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2";
const SECOM_DATA_SHA: &str = "20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c";
const SECOM_LABELS_SHA: &str = "126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03";

#[derive(Deserialize)]
struct Config {
    cmapss: CmapssConfig,
    cmapss_fd003: CmapssFd003Config,
    secom: SecomConfig,
}

#[derive(Deserialize)]
struct CmapssConfig {
    regression: RegressionConfig,
}

#[derive(Deserialize)]
struct CmapssFd003Config {
    regression: RegressionConfig,
}

#[derive(Deserialize)]
struct RegressionConfig {
    split_seed: u64,
    #[serde(default)]
    decimation_stride: usize,
    missing_maximum_fraction: f64,
    train_fraction: f64,
    validation_fraction: f64,
    contamination_seed: u64,
    fractions: Vec<f64>,
    coherent_feature_offset: f64,
    coherent_target_offset: f64,
    huber_delta: f64,
    trimmed_fraction: f64,
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

/// Fits a method and returns its test predictions (or `None` on a typed fit
/// failure — recorded, never silently zeroed).
fn fit_predict(
    configuration: RobustRegressionConfig,
    train: &TabularDataset,
    test_features: &SolversMatrix,
    test_rows: usize,
) -> Option<Vec<f64>> {
    let dataset = RegressionDataset {
        features: solvers_matrix(&train.features),
        targets: SolversMatrix::from_row_major(train.sample_count(), 1, train.targets.clone()),
        sample_weights: None,
    };

    let report = fit_robust_regression(&dataset, configuration).ok()?;
    let predictions = report.model.predict(test_features).ok()?;

    Some((0..test_rows).map(|row| predictions[(row, 0)]).collect())
}

fn regression_diagnostic(
    label: &str,
    dataset: &TabularDataset,
    config: &RegressionConfig,
    records: &mut Vec<BenchRecord>,
    summary: &mut Vec<String>,
) {
    let decimated = decimate_by_group(dataset, config.decimation_stride);

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
    let test_rows = test.sample_count();

    // Task ceiling: predict the training-target mean.
    let train_mean = train.targets.iter().sum::<f64>() / train.sample_count() as f64;
    let mean_predictions = vec![train_mean; test_rows];
    let predict_mean_rmse = rmse(&mean_predictions, &test.targets);

    let emit =
        |records: &mut Vec<BenchRecord>, method: &str, fraction: f64, metric: &str, value: f64| {
            records.push(BenchRecord::new(
                format!("diagnostic_728/{label}_regression"),
                format!("{label}/coherent_{fraction:.2}"),
                method.to_string(),
                config.contamination_seed,
                metric,
                value,
            ));
        };

    summary.push(format!(
        "# {label}: predict_mean_rmse={predict_mean_rmse:.4} (task ceiling), \
train_rows={}, test_rows={test_rows}",
        train.sample_count(),
    ));

    for (method, configuration) in method_configs(config)
    {
        // Clean-fit reference (contamination fraction 0): the recovery target.
        let Some(clean_predictions) = fit_predict(configuration, &train, &test_features, test_rows)
        else
        {
            emit(records, method, 0.0, "fit_ok", 0.0);
            continue;
        };

        let clean_fit_rmse = rmse(&clean_predictions, &test.targets);
        let task_ceiling_ratio = clean_fit_rmse / predict_mean_rmse;

        emit(records, method, 0.0, "clean_fit_rmse", clean_fit_rmse);
        emit(
            records,
            method,
            0.0,
            "task_ceiling_ratio",
            task_ceiling_ratio,
        );

        summary.push(format!(
            "#   {method}: clean_fit_rmse={clean_fit_rmse:.4} \
task_ceiling_ratio={task_ceiling_ratio:.4}"
        ));

        for &fraction in &config.fractions
        {
            let (contaminated_train, _) = apply_contamination(
                &train,
                &ContaminationConfig {
                    kind: ContaminationKind::CoherentAlternativeCluster {
                        feature_offset: config.coherent_feature_offset,
                        target_offset: config.coherent_target_offset,
                    },
                    fraction,
                    seed: config.contamination_seed,
                },
            )
            .expect("contamination request is valid");

            let Some(contaminated_predictions) = fit_predict(
                configuration,
                &contaminated_train,
                &test_features,
                test_rows,
            )
            else
            {
                emit(records, method, fraction, "fit_ok", 0.0);
                continue;
            };

            let contaminated_rmse = rmse(&contaminated_predictions, &test.targets);
            let recovery_residual = contaminated_rmse - clean_fit_rmse;
            let prediction_shift = rmse(&contaminated_predictions, &clean_predictions);

            emit(
                records,
                method,
                fraction,
                "contaminated_rmse",
                contaminated_rmse,
            );
            emit(
                records,
                method,
                fraction,
                "recovery_residual",
                recovery_residual,
            );
            emit(
                records,
                method,
                fraction,
                "prediction_shift_rms",
                prediction_shift,
            );
        }
    }
}

fn median_and_mad(values: &[f64]) -> (f64, f64) {
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(f64::total_cmp);

    let median = |data: &[f64]| -> f64 {
        let n = data.len();

        if n % 2 == 1
        {
            data[n / 2]
        }
        else
        {
            (data[n / 2 - 1] + data[n / 2]) / 2.0
        }
    };

    let center = median(&sorted);

    let mut deviations: Vec<f64> = values.iter().map(|v| (v - center).abs()).collect();
    deviations.sort_by(f64::total_cmp);

    // Normal-consistency scaling, matching scirust-stats' MAD convention.
    let mad = median(&deviations) * 1.482_602_218_505_602;

    (center, mad)
}

fn secom_diagnostic(
    dataset_raw: &TabularDataset,
    config: &SecomConfig,
    records: &mut Vec<BenchRecord>,
    summary: &mut Vec<String>,
) {
    let split = split_dataset(
        dataset_raw,
        &SplitStrategy::Temporal {
            train_fraction: config.train_fraction,
            validation_fraction: config.validation_fraction,
        },
        0,
        None,
    )
    .expect("temporal split is valid");

    let train_raw = dataset_raw.select_rows(&split.train);
    let test_raw = dataset_raw.select_rows(&split.test);

    let imputer = FittedImputer::fit(
        &train_raw.features,
        MissingValuePolicy {
            maximum_missing_fraction: config.missing_maximum_fraction,
        },
    )
    .expect("secom training keeps varying columns");

    let train = imputer.transform(&train_raw).expect("shapes match");
    let test = imputer.transform(&test_raw).expect("shapes match");

    let emit = |records: &mut Vec<BenchRecord>, metric: &str, value: f64| {
        records.push(BenchRecord::new(
            "diagnostic_728/secom",
            "secom/temporal",
            "analysis",
            0,
            metric,
            value,
        ));
    };

    // Base rates.
    let train_fail =
        train.targets.iter().filter(|&&t| t == 1.0).count() as f64 / train.sample_count() as f64;
    let test_fail =
        test.targets.iter().filter(|&&t| t == 1.0).count() as f64 / test.sample_count() as f64;

    emit(records, "train_failure_rate", train_fail);
    emit(records, "test_failure_rate", test_fail);

    // Feature drift: standardized central shift per kept feature. The
    // exact-constant drop leaves *near*-constant columns (train MAD ≈ 0 when
    // > 50 % of values coincide); a shift / MAD there is uninterpretable, so
    // those columns are counted separately and excluded from the drift
    // statistics rather than allowed to dominate them.
    let feature_count = train.feature_count();
    let mut drift_values = Vec::new();
    let mut near_constant = 0usize;

    for column in 0..feature_count
    {
        let train_column: Vec<f64> = train.features.iter().map(|row| row[column]).collect();
        let test_column: Vec<f64> = test.features.iter().map(|row| row[column]).collect();

        let (train_median, train_mad) = median_and_mad(&train_column);
        let (test_median, _) = median_and_mad(&test_column);

        // "Near constant" relative to the column's own magnitude.
        let floor = 1.0e-6 * (train_median.abs() + 1.0);

        if train_mad <= floor
        {
            near_constant += 1;
            continue;
        }

        drift_values.push((test_median - train_median).abs() / train_mad);
    }

    let informative = drift_values.len().max(1);
    let mean_drift = drift_values.iter().sum::<f64>() / informative as f64;
    let max_drift = drift_values.iter().fold(0.0, |a: f64, &b| a.max(b));
    let shifted_features = drift_values.iter().filter(|&&d| d > 1.0).count();

    emit(records, "kept_feature_count", feature_count as f64);
    emit(records, "near_constant_kept_columns", near_constant as f64);
    emit(
        records,
        "informative_feature_count",
        drift_values.len() as f64,
    );
    emit(records, "mean_standardized_drift", mean_drift);
    emit(records, "max_standardized_drift", max_drift);
    emit(
        records,
        "features_shifted_over_one_mad",
        shifted_features as f64,
    );

    // Failure separability: regularized Mahalanobis to the train-normal
    // centre, in-sample vs frozen test.
    let ridge = config
        .mahalanobis_ridge_grid
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    let train_matrix = MultivariateMatrix {
        rows: train.sample_count(),
        cols: feature_count,
        data: train.features.clone(),
    };

    let metric = FittedDistanceMetric::fit_regularized_mahalanobis(&train_matrix, ridge)
        .expect("regularized Mahalanobis fits the imputed SECOM train");

    let FittedDistanceMetric::RegularizedMahalanobis { location, .. } = &metric
    else
    {
        panic!("expected a regularized Mahalanobis metric");
    };

    let location = location.clone();

    let distances = |data: &TabularDataset| -> Vec<f64> {
        data.features
            .iter()
            .map(|row| {
                metric
                    .distance(row, &location)
                    .expect("finite distance on finite features")
            })
            .collect()
    };

    let train_distances = distances(&train);
    let test_distances = distances(&test);

    let train_auroc = auroc(&train_distances, &train.targets).expect("both classes in train");
    let test_auroc = auroc(&test_distances, &test.targets).expect("both classes in test");

    // Mean distance by label on the test split.
    let mean_by_label = |distances: &[f64], labels: &[f64], target: f64| -> f64 {
        let selected: Vec<f64> = distances
            .iter()
            .zip(labels)
            .filter(|(_, label)| **label == target)
            .map(|(distance, _)| *distance)
            .collect();

        selected.iter().sum::<f64>() / selected.len() as f64
    };

    let test_pass_distance = mean_by_label(&test_distances, &test.targets, 0.0);
    let test_fail_distance = mean_by_label(&test_distances, &test.targets, 1.0);

    emit(records, "mahalanobis_train_auroc", train_auroc);
    emit(records, "mahalanobis_test_auroc", test_auroc);
    emit(
        records,
        "mahalanobis_test_pass_mean_distance",
        test_pass_distance,
    );
    emit(
        records,
        "mahalanobis_test_fail_mean_distance",
        test_fail_distance,
    );
    emit(
        records,
        "mahalanobis_fail_over_pass_ratio",
        test_fail_distance / test_pass_distance,
    );

    summary.push(format!(
        "# secom: base_rate train={train_fail:.4} test={test_fail:.4}; \
kept={feature_count} near_constant={near_constant} informative={}; \
drift mean={mean_drift:.3} max={max_drift:.3} shifted>1MAD={shifted_features}",
        drift_values.len(),
    ));
    summary.push(format!(
        "# secom: mahalanobis AUROC train(in-sample)={train_auroc:.4} test(frozen)={test_auroc:.4}; \
fail/pass distance ratio={:.4}",
        test_fail_distance / test_pass_distance,
    ));
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

    println!(
        "# diagnostic_728 — why robustness did not beat baselines (re-analysis, not evidence)"
    );

    let fd001_text = read_verified(&data_dir.join("cmapss/train_FD001.txt"), TRAIN_FD001_SHA);
    let fd001 = parse_cmapss_training(&fd001_text).expect("FD001 parses");
    regression_diagnostic(
        "cmapss",
        &fd001,
        &config.cmapss.regression,
        &mut records,
        &mut summary,
    );

    let fd003_text = read_verified(&data_dir.join("cmapss/train_FD003.txt"), TRAIN_FD003_SHA);
    let fd003 = parse_cmapss_training(&fd003_text).expect("FD003 parses");
    regression_diagnostic(
        "cmapss_fd003",
        &fd003,
        &config.cmapss_fd003.regression,
        &mut records,
        &mut summary,
    );

    let secom_data = read_verified(&data_dir.join("secom/secom.data"), SECOM_DATA_SHA);
    let secom_labels = read_verified(&data_dir.join("secom/secom_labels.data"), SECOM_LABELS_SHA);
    let secom = parse_secom(&secom_data, &secom_labels).expect("secom parses");
    secom_diagnostic(&secom, &config.secom, &mut records, &mut summary);

    for line in &summary
    {
        println!("{line}");
    }

    fs::create_dir_all(&out_dir).expect("results directory is writable");
    fs::write(out_dir.join("diagnostic_728.jsonl"), to_jsonl(&records))
        .expect("results file is writable");

    println!("# records={}", records.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmse_matches_hand_calculation() {
        // Squared errors 0, 1, 1, 16 → mean 4.5 → sqrt.
        assert_eq!(
            rmse(&[1.0, 2.0, 4.0, 8.0], &[1.0, 1.0, 5.0, 4.0]),
            (4.5f64).sqrt(),
        );
    }

    #[test]
    fn median_and_mad_use_the_sorted_midpoint_and_normal_scaling() {
        // Odd length: median is the central order statistic.
        let (median, _) = median_and_mad(&[3.0, 1.0, 2.0]);
        assert_eq!(median, 2.0);

        // Even length: mean of the two central order statistics.
        let (median, _) = median_and_mad(&[4.0, 1.0, 3.0, 2.0]);
        assert_eq!(median, 2.5);

        // MAD of a symmetric set: deviations {2,1,0,1,2} → median 1 → ×1.4826.
        let (center, mad) = median_and_mad(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(center, 3.0);
        assert_eq!(mad, 1.482_602_218_505_602);
    }

    #[test]
    fn near_constant_column_has_zero_mad() {
        // Majority-identical values give a zero MAD even though not constant.
        let (_, mad) = median_and_mad(&[7.0, 7.0, 7.0, 7.0, 9.0]);
        assert_eq!(mad, 0.0);
    }
}
