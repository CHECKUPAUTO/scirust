//! Third program, direction B — do the two winning levers compose?
//!
//! Axis 2 showed **nonlinearity** helps (degree-2 features broke the SECOM
//! ceiling); axis 3 showed **robustness** helps (Huber beat OLS on real
//! heavy-tailed OBD2 residuals). Each was tested alone. This binary asks whether
//! they **compose** on the same real workload, with a clean **2×2 factorial**:
//!
//! |            | linear features | degree-2 polynomial features |
//! |------------|-----------------|------------------------------|
//! | squared loss | `ols_linear` (axis-3 baseline) | `ols_poly` (nonlinearity only) |
//! | Huber loss   | `huber_linear` (robustness only) | `huber_poly` (both) |
//!
//! The two factors are exactly "polynomial features" (the nonlinearity lever) and
//! "Huber loss" (the robustness lever); nothing else differs. Same real OBD2
//! telemetry, same leave-one-segment-out protocol and bulk metrics as axis 3,
//! same fixed Huber δ = 1.345. Degree-2 features are re-standardized on the
//! training rows (so squares and cross-terms are comparably scaled); no ridge and
//! no tuning. Because degree-2 Huber IRLS re-factorizes a large QR every
//! iteration, the fit uses every `TRAIN_STRIDE`-th training row (heavy tails
//! survive subsampling; all four cells share the identical decimated rows, so the
//! head-to-head is exact); every model is still **scored on the full held-out
//! segment**, so the paired bootstrap keeps its full test-set power.
//!
//! Three composition questions, each a seeded paired bootstrap of per-row
//! absolute error: (1) does `huber_poly` beat `huber_linear`? — nonlinearity on
//! top of robustness; (2) does `huber_poly` beat `ols_poly`? — robustness on top
//! of nonlinearity; (3) is the joint gain super- or sub-additive versus the two
//! single-lever gains? Whether the levers reinforce, merely add, or partly cancel
//! is reported as-is. Deterministic; run twice, byte-identical.

use std::fs;
use std::path::{Path, PathBuf};

use scirust_bench_schema::{BenchRecord, to_jsonl};
use scirust_learning::{
    RegressionDataset, RobustLoss, RobustRegressionConfig, RobustRegressionMethod,
    fit_robust_regression,
};
use scirust_solvers::linalg::Matrix as SolversMatrix;
use scirust_srcc_bench::{paired_bootstrap, paired_differences, parse_obd2, sha256_hex};
use serde::Deserialize;

const CONFIG_TEXT: &str = include_str!("../../configs/phase728.json");

const OBD2_SHA: &str = "229ef4488a89b62be530acce36ec7522421e7b82b1b5279022ffac72f0bb0751";

/// Real engine channels used as regression targets (heaviest native tails first).
const TARGETS: [&str; 3] = ["ENGINE_LOAD", "THROTTLE_POS", "MAF"];

/// Huber transition point in MAD-scale units.
const HUBER_DELTA: f64 = 1.345;
/// Training-row decimation stride (overridable with `--train-stride`). Degree-2
/// Huber IRLS re-factorizes a ~37 k × 77 QR every iteration, and the shared
/// scalar solver makes the full-data fit intractable, so every model is **fit**
/// on each `TRAIN_STRIDE`-th training row — heavy tails survive subsampling — while
/// still being **scored on the full held-out segment**, so the paired bootstrap
/// keeps its full test-set power. All four factorial cells share the same
/// decimated training rows, so the comparison between them is exact.
const TRAIN_STRIDE: usize = 12;
/// Fraction of smallest-|residual| rows kept by the trimmed-RMSE metric.
const TRIMMED_METRIC_FRACTION: f64 = 0.9;
/// Floor for a standard deviation before dividing.
const SD_FLOOR: f64 = 1.0e-12;

/// One factorial cell: a display name, whether it uses the Huber loss, and the
/// polynomial degree of its feature map.
struct Method {
    name: &'static str,
    huber: bool,
    degree: usize,
}

const METHODS: [Method; 4] = [
    Method {
        name: "ols_linear",
        huber: false,
        degree: 1,
    },
    Method {
        name: "huber_linear",
        huber: true,
        degree: 1,
    },
    Method {
        name: "ols_poly",
        huber: false,
        degree: 2,
    },
    Method {
        name: "huber_poly",
        huber: true,
        degree: 2,
    },
];

#[derive(Deserialize)]
struct Config {
    bootstrap: BootstrapConfig,
}

#[derive(Deserialize)]
struct BootstrapConfig {
    resamples: usize,
    level: f64,
    seed: u64,
}

fn read_verified(path: &Path, expected_sha: &str) -> String {
    let bytes = fs::read(path).unwrap_or_else(|error| {
        panic!(
            "cannot read {}: {error}. The OBD2 telemetry ships in-repo under \
examples/obd2_diagnostic/data/.",
            path.display()
        )
    });

    let actual = sha256_hex(&bytes);

    assert!(
        actual == expected_sha,
        "checksum mismatch for {}: expected {expected_sha}, found {actual}",
        path.display()
    );

    String::from_utf8(bytes).expect("dataset file is valid UTF-8")
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

/// Degree-2 polynomial map: original features, their squares, and pairwise
/// products. Width `k(k+3)/2` for `k` inputs. Degree 1 is the identity.
fn expand_polynomial(features: &[Vec<f64>], degree: usize) -> Vec<Vec<f64>> {
    if degree < 2
    {
        return features.to_vec();
    }

    features
        .iter()
        .map(|row| {
            let mut expanded: Vec<f64> = row.clone();

            for (i, &value_i) in row.iter().enumerate()
            {
                expanded.push(value_i * value_i);

                for &value_j in &row[i + 1..]
                {
                    expanded.push(value_i * value_j);
                }
            }

            expanded
        })
        .collect()
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

/// Builds the design matrices (train, test) for a method's degree: expand and, at
/// degree 2, re-standardize the expanded columns on the training rows.
fn design(
    degree: usize,
    train_base: &[Vec<f64>],
    test_base: &[Vec<f64>],
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    if degree < 2
    {
        return (train_base.to_vec(), test_base.to_vec());
    }

    let train_expanded = expand_polynomial(train_base, degree);
    let test_expanded = expand_polynomial(test_base, degree);
    let standardizer = Standardizer::fit(&train_expanded);
    (
        standardizer.apply(&train_expanded),
        standardizer.apply(&test_expanded),
    )
}

fn method_config(huber: bool) -> RobustRegressionConfig {
    let base = RobustRegressionConfig::default();

    if huber
    {
        RobustRegressionConfig {
            method: RobustRegressionMethod::IterativelyReweightedLeastSquares,
            loss: RobustLoss::Huber { delta: HUBER_DELTA },
            ..base
        }
    }
    else
    {
        RobustRegressionConfig {
            method: RobustRegressionMethod::OrdinaryLeastSquares,
            ..base
        }
    }
}

/// Fits one factorial cell on `(design_train, targets)` and predicts `design_test`.
fn fit_predict(
    huber: bool,
    design_train: &[Vec<f64>],
    train_targets: &[f64],
    design_test: &SolversMatrix,
    test_rows: usize,
) -> Vec<f64> {
    let dataset = RegressionDataset {
        features: solvers_matrix(design_train),
        targets: SolversMatrix::from_row_major(design_train.len(), 1, train_targets.to_vec()),
        sample_weights: None,
    };

    let report = fit_robust_regression(&dataset, method_config(huber))
        .expect("regression fits the standardized OBD2 design");
    let predictions = report
        .model
        .predict(design_test)
        .expect("prediction shapes match");

    (0..test_rows).map(|row| predictions[(row, 0)]).collect()
}

fn distinct_sorted(groups: &[u64]) -> Vec<u64> {
    let mut distinct: Vec<u64> = Vec::new();

    for &group in groups
    {
        if !distinct.contains(&group)
        {
            distinct.push(group);
        }
    }

    distinct.sort_unstable();
    distinct
}

fn rmse(residuals: &[f64]) -> f64 {
    (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt()
}

fn mae(residuals: &[f64]) -> f64 {
    residuals.iter().map(|r| r.abs()).sum::<f64>() / residuals.len() as f64
}

fn sorted_absolute(residuals: &[f64]) -> Vec<f64> {
    let mut absolute: Vec<f64> = residuals.iter().map(|r| r.abs()).collect();
    absolute.sort_by(f64::total_cmp);
    absolute
}

fn median_absolute(residuals: &[f64]) -> f64 {
    let absolute = sorted_absolute(residuals);
    let n = absolute.len();

    if n == 0
    {
        return 0.0;
    }

    if n % 2 == 1
    {
        absolute[n / 2]
    }
    else
    {
        0.5 * (absolute[n / 2 - 1] + absolute[n / 2])
    }
}

fn trimmed_rmse(residuals: &[f64], fraction: f64) -> f64 {
    let absolute = sorted_absolute(residuals);
    let keep = ((fraction * absolute.len() as f64).floor() as usize).max(1);
    (absolute[..keep].iter().map(|r| r * r).sum::<f64>() / keep as f64).sqrt()
}

/// Seeded paired-bootstrap mean and CI of the per-row absolute-error reduction
/// `|reference| − |candidate|` (positive = candidate better on the bulk).
fn absolute_error_gain(
    reference: &[f64],
    candidate: &[f64],
    bootstrap: &BootstrapConfig,
) -> (f64, f64, f64) {
    let reference_absolute: Vec<f64> = reference.iter().map(|r| r.abs()).collect();
    let candidate_absolute: Vec<f64> = candidate.iter().map(|r| r.abs()).collect();
    let differences = paired_differences(&reference_absolute, &candidate_absolute)
        .expect("aligned finite per-row absolute errors");
    let report = paired_bootstrap(
        &differences,
        bootstrap.resamples,
        bootstrap.level,
        bootstrap.seed,
    )
    .expect("enough rows for the bootstrap");
    (
        report.mean_difference,
        report.confidence_interval.lo,
        report.confidence_interval.hi,
    )
}

fn verdict(lo: f64, hi: f64) -> &'static str {
    if lo > 0.0
    {
        "wins"
    }
    else if hi < 0.0
    {
        "loses"
    }
    else
    {
        "ties"
    }
}

fn main() {
    let mut data_path = PathBuf::from("examples/obd2_diagnostic/data/opel_corsa_telemetry.csv");
    let mut out_dir = PathBuf::from("results");
    let mut train_stride = TRAIN_STRIDE;

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
            "--data" => data_path = PathBuf::from(value("--data")),
            "--out" => out_dir = PathBuf::from(value("--out")),
            "--train-stride" =>
            {
                train_stride = value("--train-stride")
                    .parse()
                    .expect("--train-stride is a positive integer")
            },
            other => panic!("unknown argument: {other}"),
        }
    }

    assert!(train_stride > 0, "--train-stride must be positive");

    let config: Config = serde_json::from_str(CONFIG_TEXT).expect("embedded config is valid");
    let bootstrap = &config.bootstrap;

    let text = read_verified(&data_path, OBD2_SHA);
    let mut records: Vec<BenchRecord> = Vec::new();

    println!("# obd2_robust_nonlinear — Direction B: do robustness and nonlinearity compose?");
    println!(
        "# 2×2 factorial (linear/poly × OLS/Huber) on real OBD2, leave-one-segment-out; \
Huber δ={HUBER_DELTA}, degree-2 re-standardized, no ridge; train stride={train_stride} (full test)"
    );

    for target in TARGETS
    {
        let data = parse_obd2(&text, target).expect("OBD2 parses for the target channel");
        let groups = data.groups.as_ref().expect("OBD2 carries segment groups");
        let segments = distinct_sorted(groups);
        let rows = data.targets.len();

        let mut pooled: Vec<Vec<f64>> = vec![Vec::with_capacity(rows); METHODS.len()];

        for &held in &segments
        {
            // Training rows decimated by the stride (heavy tails survive
            // subsampling); test rows kept in full for the bootstrap.
            let train_rows: Vec<usize> = (0..rows)
                .filter(|&row| groups[row] != held)
                .step_by(train_stride)
                .collect();
            let test_rows: Vec<usize> = (0..rows).filter(|&row| groups[row] == held).collect();

            let train_features_raw: Vec<Vec<f64>> = train_rows
                .iter()
                .map(|&row| data.features[row].clone())
                .collect();
            let train_targets: Vec<f64> = train_rows.iter().map(|&row| data.targets[row]).collect();
            let test_features_raw: Vec<Vec<f64>> = test_rows
                .iter()
                .map(|&row| data.features[row].clone())
                .collect();
            let test_targets: Vec<f64> = test_rows.iter().map(|&row| data.targets[row]).collect();

            let base_standardizer = Standardizer::fit(&train_features_raw);
            let train_base = base_standardizer.apply(&train_features_raw);
            let test_base = base_standardizer.apply(&test_features_raw);

            for (index, method) in METHODS.iter().enumerate()
            {
                let (design_train, design_test) = design(method.degree, &train_base, &test_base);
                let test_matrix = solvers_matrix(&design_test);
                let predictions = fit_predict(
                    method.huber,
                    &design_train,
                    &train_targets,
                    &test_matrix,
                    test_targets.len(),
                );

                for (prediction, actual) in predictions.iter().zip(&test_targets)
                {
                    pooled[index].push(actual - prediction);
                }
            }
        }

        println!("# {target}:");
        println!("#   cell          rmse       mae    medAE  trim10RMSE");

        for (index, method) in METHODS.iter().enumerate()
        {
            let residuals = &pooled[index];
            let cell_rmse = rmse(residuals);
            let cell_mae = mae(residuals);
            let cell_median = median_absolute(residuals);
            let cell_trimmed = trimmed_rmse(residuals, TRIMMED_METRIC_FRACTION);

            for (metric, quantity) in [
                ("rmse", cell_rmse),
                ("mae", cell_mae),
                ("median_abs_error", cell_median),
                ("trimmed_rmse", cell_trimmed),
            ]
            {
                records.push(BenchRecord::new(
                    "obd2_robust_nonlinear/metrics",
                    format!("obd2/{target}"),
                    method.name,
                    0,
                    metric,
                    quantity,
                ));
            }

            println!(
                "#   {:<12} {cell_rmse:8.4} {cell_mae:8.4} {cell_median:8.4} {cell_trimmed:11.4}",
                method.name
            );
        }

        // Composition analysis, all on per-row absolute error (the bulk metric
        // where robustness helped in axis 3). Index map: 0 ols_linear,
        // 1 huber_linear, 2 ols_poly, 3 huber_poly.
        let single_robust = absolute_error_gain(&pooled[0], &pooled[1], bootstrap);
        let single_nonlinear = absolute_error_gain(&pooled[0], &pooled[2], bootstrap);
        let both = absolute_error_gain(&pooled[0], &pooled[3], bootstrap);
        // Does each lever still add on top of the other?
        let nonlinear_on_robust = absolute_error_gain(&pooled[1], &pooled[3], bootstrap);
        let robust_on_nonlinear = absolute_error_gain(&pooled[2], &pooled[3], bootstrap);
        let additive_expectation = single_robust.0 + single_nonlinear.0;
        let compositional_gap = both.0 - additive_expectation;

        for (name, gain) in [
            ("robust_only_vs_ols", single_robust),
            ("nonlinear_only_vs_ols", single_nonlinear),
            ("both_vs_ols", both),
            ("nonlinear_on_top_of_robust", nonlinear_on_robust),
            ("robust_on_top_of_nonlinear", robust_on_nonlinear),
        ]
        {
            records.push(
                BenchRecord::new(
                    "obd2_robust_nonlinear/composition",
                    format!("obd2/{target}"),
                    name,
                    bootstrap.seed,
                    "abs_error_gain_mean",
                    gain.0,
                )
                .with_ci(scirust_bench_schema::ConfidenceInterval {
                    lo: gain.1,
                    hi: gain.2,
                    level: bootstrap.level,
                }),
            );
        }

        records.push(BenchRecord::new(
            "obd2_robust_nonlinear/composition",
            format!("obd2/{target}"),
            "compositional_gap",
            bootstrap.seed,
            "abs_error_gain_mean",
            compositional_gap,
        ));

        println!(
            "#   robustness alone Δ={:+.4} [{:+.4},{:+.4}] {}",
            single_robust.0,
            single_robust.1,
            single_robust.2,
            verdict(single_robust.1, single_robust.2)
        );
        println!(
            "#   nonlinearity alone Δ={:+.4} [{:+.4},{:+.4}] {}",
            single_nonlinear.0,
            single_nonlinear.1,
            single_nonlinear.2,
            verdict(single_nonlinear.1, single_nonlinear.2)
        );
        println!(
            "#   both Δ={:+.4} [{:+.4},{:+.4}] {} | additive expectation {:+.4} → gap {:+.4} ({})",
            both.0,
            both.1,
            both.2,
            verdict(both.1, both.2),
            additive_expectation,
            compositional_gap,
            if compositional_gap > 0.0
            {
                "super-additive"
            }
            else
            {
                "sub-additive"
            }
        );
        println!(
            "#   nonlinearity on top of robustness: Δ={:+.4} [{:+.4},{:+.4}] {}",
            nonlinear_on_robust.0,
            nonlinear_on_robust.1,
            nonlinear_on_robust.2,
            verdict(nonlinear_on_robust.1, nonlinear_on_robust.2)
        );
        println!(
            "#   robustness on top of nonlinearity: Δ={:+.4} [{:+.4},{:+.4}] {}",
            robust_on_nonlinear.0,
            robust_on_nonlinear.1,
            robust_on_nonlinear.2,
            verdict(robust_on_nonlinear.1, robust_on_nonlinear.2)
        );
    }

    fs::create_dir_all(&out_dir).expect("results directory is writable");
    fs::write(
        out_dir.join("obd2_robust_nonlinear.jsonl"),
        to_jsonl(&records),
    )
    .expect("results file is writable");

    println!("# records={}", records.len());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_polynomial_degree_two_has_expected_width() {
        // 3 inputs → 3 linear + 3 squares + 3 pairwise = 9 = k(k+3)/2 with k=3.
        let expanded = expand_polynomial(&[vec![2.0, 3.0, 5.0]], 2);
        assert_eq!(expanded[0].len(), 9);
        // Original features retained in front.
        assert_eq!(&expanded[0][..3], &[2.0, 3.0, 5.0]);
        // Squares then cross products.
        assert_eq!(expanded[0][3], 4.0); // 2²
        assert_eq!(expanded[0][4], 6.0); // 2·3
        assert_eq!(expanded[0][5], 10.0); // 2·5
    }

    #[test]
    fn expand_polynomial_degree_one_is_identity() {
        let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert_eq!(expand_polynomial(&rows, 1), rows);
    }

    #[test]
    fn bulk_metrics_are_consistent() {
        let residuals = [3.0, -4.0];
        assert!((rmse(&residuals) - (12.5_f64).sqrt()).abs() < 1e-12);
        assert!((mae(&residuals) - 3.5).abs() < 1e-12);
        assert!((median_absolute(&[1.0, -3.0, 2.0, -4.0]) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn verdict_reads_the_interval() {
        assert_eq!(verdict(0.1, 0.2), "wins");
        assert_eq!(verdict(-0.2, -0.1), "loses");
        assert_eq!(verdict(-0.1, 0.1), "ties");
    }
}
