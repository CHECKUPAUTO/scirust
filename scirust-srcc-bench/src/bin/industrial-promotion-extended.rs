//! Follow-up 8 / axis 4 — extended promotion gates on a real shadow deployment.
//!
//! Phase 729's [`scirust_srcc_bench::PromotionGate`] made a single-metric
//! promote/hold decision. This binary drives the **extended** gate
//! ([`scirust_srcc_bench::ExtendedPromotionGate`]) — a *weighted composite* of
//! several metrics, a *switching-cost* deadband, and *temporal shadow windows* —
//! on the real workload of axis 3. The incumbent is OLS, the candidate is
//! Huber-IRLS, both fit leave-one-segment-out on the Opel-Corsa OBD2 telemetry
//! (the same protocol and data as `industrial-obd2-native`). The **seven driving
//! segments are the shadow windows**: a candidate must not only win on pooled
//! data but hold up segment by segment.
//!
//! Axis 3 found a genuinely two-sided result — Huber beats OLS on the bulk
//! (MAE, median error) but loses on tail-sensitive squared error. An operator
//! cannot promote on "it depends"; the extended gate prices the trade-off in
//! advance. The composite weights a **relative** absolute-error improvement 0.75
//! and a **relative** squared-error improvement 0.25 (each scaled by the
//! incumbent's own pooled error magnitude — its known operating point), so the
//! decision variable is a single weighted percentage gain. The gate is then
//! evaluated at two preregistered switching costs: `0.00` (a free switch) and
//! `0.05` (promotion must be worth a 5 % weighted gain to justify the operational
//! cost of changing models). The bulk win is real but modest, so the deadband is
//! exactly what decides whether it is worth acting on.
//!
//! Deterministic: OLS and Huber-IRLS are RNG-free; every decision is a seeded
//! paired bootstrap. Run twice, byte-identical.

use std::fs;
use std::path::{Path, PathBuf};

use scirust_bench_schema::{BenchRecord, to_jsonl};
use scirust_learning::{
    RegressionDataset, RobustLoss, RobustRegressionConfig, RobustRegressionMethod,
    fit_robust_regression,
};
use scirust_solvers::linalg::Matrix as SolversMatrix;
use scirust_srcc_bench::{
    Decision, ExtendedPromotionGate, Orientation, ShadowWindow, WeightedMetric, WindowMetricValues,
    parse_obd2, sha256_hex,
};
use serde::Deserialize;

const CONFIG_TEXT: &str = include_str!("../../configs/phase728.json");

const OBD2_SHA: &str = "229ef4488a89b62be530acce36ec7522421e7b82b1b5279022ffac72f0bb0751";

/// Real engine channels used as regression targets (heaviest native tails first).
const TARGETS: [&str; 3] = ["ENGINE_LOAD", "THROTTLE_POS", "MAF"];

/// Huber transition point in MAD-scale units.
const HUBER_DELTA: f64 = 1.345;
/// Composite weight on the relative absolute-error improvement (bulk).
const ABSOLUTE_WEIGHT: f64 = 0.75;
/// Composite weight on the relative squared-error improvement (tail-sensitive).
const SQUARED_WEIGHT: f64 = 0.25;
/// Preregistered switching costs: a free switch, then a 5 %-gain hurdle.
const SWITCHING_COSTS: [f64; 2] = [0.0, 0.05];
/// No shadow window may regress on the composite.
const MIN_WINDOW_IMPROVEMENT: f64 = 0.0;
/// Floor for a standard deviation before dividing.
const SD_FLOOR: f64 = 1.0e-12;

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

fn method_config(method: &str) -> RobustRegressionConfig {
    let base = RobustRegressionConfig::default();

    match method
    {
        "ols" => RobustRegressionConfig {
            method: RobustRegressionMethod::OrdinaryLeastSquares,
            ..base
        },
        "huber_irls" => RobustRegressionConfig {
            method: RobustRegressionMethod::IterativelyReweightedLeastSquares,
            loss: RobustLoss::Huber { delta: HUBER_DELTA },
            ..base
        },
        other => panic!("unknown method {other}"),
    }
}

fn fit_predict(
    method: &str,
    train_features: &[Vec<f64>],
    train_targets: &[f64],
    test_features: &SolversMatrix,
    test_rows: usize,
) -> Vec<f64> {
    let dataset = RegressionDataset {
        features: solvers_matrix(train_features),
        targets: SolversMatrix::from_row_major(train_features.len(), 1, train_targets.to_vec()),
        sample_weights: None,
    };

    let report = fit_robust_regression(&dataset, method_config(method))
        .expect("regression fits the standardized OBD2 design");
    let predictions = report
        .model
        .predict(test_features)
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

/// One segment's per-row error vectors for both models.
struct SegmentErrors {
    label: String,
    incumbent_absolute: Vec<f64>,
    candidate_absolute: Vec<f64>,
    incumbent_squared: Vec<f64>,
    candidate_squared: Vec<f64>,
}

fn main() {
    let mut data_path = PathBuf::from("examples/obd2_diagnostic/data/opel_corsa_telemetry.csv");
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
            "--data" => data_path = PathBuf::from(value("--data")),
            "--out" => out_dir = PathBuf::from(value("--out")),
            other => panic!("unknown argument: {other}"),
        }
    }

    let config: Config = serde_json::from_str(CONFIG_TEXT).expect("embedded config is valid");
    let bootstrap = &config.bootstrap;

    let text = read_verified(&data_path, OBD2_SHA);
    let mut records: Vec<BenchRecord> = Vec::new();

    println!(
        "# promotion_extended — Axis 4: extended gate (composite + switching cost + shadow windows)"
    );
    println!(
        "# incumbent OLS vs candidate Huber on real OBD2, 7 driving segments = shadow windows; \
composite {ABSOLUTE_WEIGHT}·rel(abs-err) + {SQUARED_WEIGHT}·rel(sq-err)"
    );

    for target in TARGETS
    {
        let data = parse_obd2(&text, target).expect("OBD2 parses for the target channel");
        let groups = data.groups.as_ref().expect("OBD2 carries segment groups");
        let segments = distinct_sorted(groups);
        let rows = data.targets.len();

        let mut per_segment: Vec<SegmentErrors> = Vec::with_capacity(segments.len());
        // Pooled incumbent error magnitudes → composite scales (the incumbent's
        // known operating point, fixed before the candidate is judged).
        let mut incumbent_absolute_sum = 0.0;
        let mut incumbent_squared_sum = 0.0;
        let mut pooled_rows = 0usize;

        for &held in &segments
        {
            let train_rows: Vec<usize> = (0..rows).filter(|&row| groups[row] != held).collect();
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

            let standardizer = Standardizer::fit(&train_features_raw);
            let train_features = standardizer.apply(&train_features_raw);
            let test_features = standardizer.apply(&test_features_raw);
            let test_matrix = solvers_matrix(&test_features);

            let ols = fit_predict(
                "ols",
                &train_features,
                &train_targets,
                &test_matrix,
                test_targets.len(),
            );
            let huber = fit_predict(
                "huber_irls",
                &train_features,
                &train_targets,
                &test_matrix,
                test_targets.len(),
            );

            let mut incumbent_absolute = Vec::with_capacity(test_targets.len());
            let mut candidate_absolute = Vec::with_capacity(test_targets.len());
            let mut incumbent_squared = Vec::with_capacity(test_targets.len());
            let mut candidate_squared = Vec::with_capacity(test_targets.len());

            for ((ols_prediction, huber_prediction), actual) in
                ols.iter().zip(&huber).zip(&test_targets)
            {
                let ols_residual = actual - ols_prediction;
                let huber_residual = actual - huber_prediction;
                incumbent_absolute.push(ols_residual.abs());
                candidate_absolute.push(huber_residual.abs());
                incumbent_squared.push(ols_residual * ols_residual);
                candidate_squared.push(huber_residual * huber_residual);

                incumbent_absolute_sum += ols_residual.abs();
                incumbent_squared_sum += ols_residual * ols_residual;
                pooled_rows += 1;
            }

            per_segment.push(SegmentErrors {
                label: format!("segment_{held}"),
                incumbent_absolute,
                candidate_absolute,
                incumbent_squared,
                candidate_squared,
            });
        }

        let absolute_scale = (incumbent_absolute_sum / pooled_rows as f64).max(SD_FLOOR);
        let squared_scale = (incumbent_squared_sum / pooled_rows as f64).max(SD_FLOOR);

        let windows: Vec<ShadowWindow> = per_segment
            .iter()
            .map(|segment| ShadowWindow {
                label: segment.label.clone(),
                values: vec![
                    WindowMetricValues {
                        metric: "absolute_error".to_string(),
                        incumbent: segment.incumbent_absolute.clone(),
                        candidate: segment.candidate_absolute.clone(),
                    },
                    WindowMetricValues {
                        metric: "squared_error".to_string(),
                        incumbent: segment.incumbent_squared.clone(),
                        candidate: segment.candidate_squared.clone(),
                    },
                ],
            })
            .collect();

        let composite = vec![
            WeightedMetric {
                metric: "absolute_error".to_string(),
                orientation: Orientation::LowerIsBetter,
                weight: ABSOLUTE_WEIGHT,
                scale: absolute_scale,
            },
            WeightedMetric {
                metric: "squared_error".to_string(),
                orientation: Orientation::LowerIsBetter,
                weight: SQUARED_WEIGHT,
                scale: squared_scale,
            },
        ];

        println!(
            "# {target}: incumbent pooled |err|={absolute_scale:.4} err²={squared_scale:.4} (composite scales)"
        );

        for &switching_cost in &SWITCHING_COSTS
        {
            let gate = ExtendedPromotionGate {
                composite: composite.clone(),
                switching_cost,
                min_window_improvement: MIN_WINDOW_IMPROVEMENT,
                resamples: bootstrap.resamples,
                level: bootstrap.level,
                seed: bootstrap.seed,
            };

            let report = gate.decide(&windows).expect("well-formed shadow windows");
            let promoted = report.decision == Decision::Promote;
            let windows_held = report.windows.iter().filter(|w| w.passed).count();

            let cell = format!("obd2/{target}/cost_{switching_cost}");

            records.push(BenchRecord::new(
                "promotion_extended/decision",
                cell.clone(),
                "huber_vs_ols",
                bootstrap.seed,
                "promote",
                if promoted { 1.0 } else { 0.0 },
            ));
            records.push(
                BenchRecord::new(
                    "promotion_extended/composite",
                    cell.clone(),
                    "huber_vs_ols",
                    bootstrap.seed,
                    "pooled_composite_improvement",
                    report.pooled_mean_improvement,
                )
                .with_ci(report.pooled_confidence_interval),
            );
            records.push(BenchRecord::new(
                "promotion_extended/windows",
                cell,
                "huber_vs_ols",
                bootstrap.seed,
                "windows_held",
                windows_held as f64,
            ));

            println!(
                "#   cost={switching_cost:.2} → {:<7} | composite Δ={:+.4} CI=[{:+.4},{:+.4}] \
(cost {switching_cost:.2}) | windows held {windows_held}/{}",
                if promoted { "PROMOTE" } else { "HOLD" },
                report.pooled_mean_improvement,
                report.pooled_confidence_interval.lo,
                report.pooled_confidence_interval.hi,
                report.windows.len(),
            );

            if !report.reasons.is_empty()
            {
                println!("#     reason: {}", report.reasons.join("; "));
            }
        }
    }

    fs::create_dir_all(&out_dir).expect("results directory is writable");
    fs::write(out_dir.join("promotion_extended.jsonl"), to_jsonl(&records))
        .expect("results file is writable");

    println!("# records={}", records.len());
}
