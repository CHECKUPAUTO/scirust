//! Lever 2 — the decisive robustness test: high-leverage contamination.
//!
//! The phase-728 diagnostic showed the coherent-block contamination barely
//! moved the least-squares fit (1–4 RUL), so robustness had nothing to
//! repair and the H2 comparison was inconclusive. This binary runs the
//! **fair** test the diagnostic called for: on the re-framed task (piecewise
//! RUL capped at 125 — lever 1's strong lever, now on master; the paired
//! comparison keeps the phase-728 stride 20 for tractability and direct
//! comparability) it injects a genuinely **high-leverage** contamination — a
//! fraction of
//! training rows pushed far out in feature space with a wrong target
//! (`ContaminationKind::LeveragePoint`) — and re-runs the paired
//! leave-one-engine-out OLS-vs-robust comparison.
//!
//! Two things are reported per subset:
//!
//! - a **leverage confirmation**: the mean OLS prediction shift
//!   `RMS(pred_contaminated − pred_clean)` on the held-out engines. Unlike
//!   the diagnostic's 1–4 RUL, a high-leverage attack should move OLS by a
//!   large margin — otherwise the contamination is still not an attack and
//!   the test is void;
//! - the **paired verdict**: seeded bootstrap CI of the per-engine RMSE
//!   difference `OLS − robust`. A CI strictly above zero means robustness
//!   finally wins under a strong attack (the regime where it matters); a CI
//!   straddling zero on a *confirmed* high-leverage attack is a decisive null
//!   — the current robust regressors do not help even when the baseline is
//!   badly hurt.
//!
//! Split seeds, fractions, missing-value policy, `huber_delta`,
//! `trimmed_fraction` and the bootstrap settings are read verbatim from the
//! frozen `configs/phase728.json`. The leverage knobs are fixed a priori
//! (documented constants), not tuned on outcomes. Deterministic; run twice,
//! byte-identical.

use std::fs;
use std::path::{Path, PathBuf};

use scirust_bench_schema::{BenchRecord, to_jsonl};
use scirust_learning::{
    RegressionDataset, RobustLoss, RobustRegressionConfig, RobustRegressionMethod,
    fit_robust_regression,
};
use scirust_solvers::linalg::Matrix as SolversMatrix;
use scirust_srcc_bench::{
    ContaminationConfig, ContaminationKind, FittedImputer, MissingValuePolicy, SplitStrategy,
    TabularDataset, apply_contamination, clip_rul_targets, paired_bootstrap, paired_differences,
    parse_cmapss_training, sha256_hex, split_dataset,
};
use serde::Deserialize;

const CONFIG_TEXT: &str = include_str!("../../configs/phase728.json");

const TRAIN_FD001_SHA: &str = "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8";
const TRAIN_FD003_SHA: &str = "2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2";

/// Re-framed task: piecewise RUL knee (lever 1's strong lever). The paired
/// leave-one-engine-out comparison uses the phase-728 stride (20) rather than
/// lever 1's stride 5: with 100 engines it already yields 100 paired points
/// for the bootstrap, and stride governs statistical power, not the leverage
/// question under test — keeping it at 20 also makes this test directly
/// comparable to the phase-728 paired result (same framing except piecewise
/// RUL and a high-leverage rather than low-leverage attack), while the
/// stride-5 iterative refits over 100 engines are intractable here.
const R_EARLY: f64 = 125.0;
const STRIDE: usize = 20;

/// High-leverage contamination knobs, fixed a priori: features pushed 20
/// train-MADs out with the target set to twice the RUL knee (clearly outside
/// the piecewise range `[0, 125]`). The **fraction is swept** {0.1, 0.2, 0.3}
/// so the verdict's dependence on attack strength is reported rather than a
/// single point tuned to an outcome — 0.3 is the boundary where trimmed
/// least squares' 0.7 retention should, in principle, help most.
const FEATURE_SHIFT_MADS: f64 = 20.0;
const CORRUPT_TARGET: f64 = 250.0;
const LEVERAGE_FRACTIONS: [f64; 3] = [0.1, 0.2, 0.3];
const LEVERAGE_SEED: u64 = 0x0728_0002;

#[derive(Deserialize)]
struct Config {
    bootstrap: BootstrapConfig,
    cmapss: Section,
    cmapss_fd003: Section,
}

#[derive(Deserialize)]
struct BootstrapConfig {
    resamples: usize,
    level: f64,
    seed: u64,
}

#[derive(Deserialize)]
struct Section {
    regression: RegressionConfig,
}

#[derive(Deserialize)]
struct RegressionConfig {
    missing_maximum_fraction: f64,
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

fn method_config(method: &str, config: &RegressionConfig) -> RobustRegressionConfig {
    let base = RobustRegressionConfig::default();

    match method
    {
        "ols" => RobustRegressionConfig {
            method: RobustRegressionMethod::OrdinaryLeastSquares,
            ..base
        },
        "huber_irls" => RobustRegressionConfig {
            method: RobustRegressionMethod::IterativelyReweightedLeastSquares,
            loss: RobustLoss::Huber {
                delta: config.huber_delta,
            },
            ..base
        },
        "trimmed_ls" => RobustRegressionConfig {
            method: RobustRegressionMethod::TrimmedLeastSquares {
                retained_fraction: config.trimmed_fraction,
            },
            ..base
        },
        other => panic!("unknown method {other}"),
    }
}

fn fit_predict(
    method: &str,
    config: &RegressionConfig,
    train: &TabularDataset,
    test_features: &SolversMatrix,
    test_rows: usize,
) -> Option<Vec<f64>> {
    let dataset = RegressionDataset {
        features: solvers_matrix(&train.features),
        targets: SolversMatrix::from_row_major(train.sample_count(), 1, train.targets.clone()),
        sample_weights: None,
    };

    let report = fit_robust_regression(&dataset, method_config(method, config)).ok()?;
    let predictions = report.model.predict(test_features).ok()?;

    Some((0..test_rows).map(|row| predictions[(row, 0)]).collect())
}

fn distinct_groups(dataset: &TabularDataset) -> Vec<u64> {
    let groups = dataset.groups.as_ref().expect("grouped dataset");
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

/// Per-contamination-fraction accumulator: per-method held-out RMSEs, the
/// OLS prediction shifts (leverage confirmation), and the count of engines
/// dropped for a fit failure on any method (pairing integrity).
struct FractionAccumulator {
    fraction: f64,
    per_method: Vec<(&'static str, Vec<f64>)>,
    prediction_shifts: Vec<f64>,
    dropped: usize,
}

fn evaluate(
    label: &str,
    raw: &TabularDataset,
    config: &RegressionConfig,
    bootstrap: &BootstrapConfig,
    records: &mut Vec<BenchRecord>,
    summary: &mut Vec<String>,
) {
    let reframed = decimate_by_group(&clip_rul_targets(raw, R_EARLY), STRIDE);
    let engines = distinct_groups(&reframed);

    let methods = ["ols", "huber_irls", "trimmed_ls"];

    let mut per_fraction: Vec<FractionAccumulator> = LEVERAGE_FRACTIONS
        .iter()
        .map(|&fraction| FractionAccumulator {
            fraction,
            per_method: methods.iter().map(|&m| (m, Vec::new())).collect(),
            prediction_shifts: Vec::new(),
            dropped: 0,
        })
        .collect();

    for &engine in &engines
    {
        let split = split_dataset(
            &reframed,
            &SplitStrategy::LeaveOneGroupOut {
                held_out_group: engine,
            },
            0,
            Some("unit"),
        )
        .expect("every engine occurs");

        let train_raw = reframed.select_rows(&split.train);
        let test_raw = reframed.select_rows(&split.test);

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

        // Clean OLS reference (fraction-independent) for the leverage
        // confirmation, computed once per engine.
        let clean_ols = fit_predict("ols", config, &train, &test_features, test_rows);

        for accumulator in &mut per_fraction
        {
            let (contaminated_train, _) = apply_contamination(
                &train,
                &ContaminationConfig {
                    kind: ContaminationKind::LeveragePoint {
                        feature_shift_mads: FEATURE_SHIFT_MADS,
                        corrupt_target: CORRUPT_TARGET,
                    },
                    fraction: accumulator.fraction,
                    seed: LEVERAGE_SEED,
                },
            )
            .expect("leverage contamination is valid");

            let mut engine_rmse: Vec<(&str, f64)> = Vec::new();
            let mut engine_ok = true;

            for &method in &methods
            {
                match fit_predict(
                    method,
                    config,
                    &contaminated_train,
                    &test_features,
                    test_rows,
                )
                {
                    Some(predictions) =>
                    {
                        engine_rmse.push((method, rmse(&predictions, &test.targets)));

                        if method == "ols"
                            && let Some(clean) = &clean_ols
                        {
                            accumulator
                                .prediction_shifts
                                .push(rmse(&predictions, clean));
                        }
                    },
                    None => engine_ok = false,
                }
            }

            if engine_ok && engine_rmse.len() == methods.len()
            {
                for (method, value) in engine_rmse
                {
                    accumulator
                        .per_method
                        .iter_mut()
                        .find(|(name, _)| *name == method)
                        .expect("method present")
                        .1
                        .push(value);
                }
            }
            else
            {
                accumulator.dropped += 1;
            }
        }
    }

    for accumulator in &per_fraction
    {
        let fraction = accumulator.fraction;
        let per_method = &accumulator.per_method;
        let prediction_shifts = &accumulator.prediction_shifts;
        let dropped = accumulator.dropped;

        let mean_shift = if prediction_shifts.is_empty()
        {
            0.0
        }
        else
        {
            prediction_shifts.iter().sum::<f64>() / prediction_shifts.len() as f64
        };

        let cell = format!("{label}/piecewise_frac_{fraction:.1}");

        records.push(BenchRecord::new(
            "leverage/confirmation",
            cell.clone(),
            "ols",
            LEVERAGE_SEED,
            "mean_prediction_shift_rms",
            mean_shift,
        ));
        records.push(BenchRecord::new(
            "leverage/confirmation",
            cell.clone(),
            "all_methods",
            LEVERAGE_SEED,
            "dropped_engines",
            dropped as f64,
        ));

        summary.push(format!(
            "# {label} frac={fraction:.1}: OLS prediction shift = {mean_shift:6.2} RUL; \
dropped={dropped}"
        ));

        let ols_rmse = &per_method[0].1;

        for robust in ["huber_irls", "trimmed_ls"]
        {
            let robust_rmse = &per_method
                .iter()
                .find(|(name, _)| *name == robust)
                .expect("method present")
                .1;

            let differences = paired_differences(ols_rmse, robust_rmse)
                .expect("per-engine vectors are aligned and finite");

            let report = paired_bootstrap(
                &differences,
                bootstrap.resamples,
                bootstrap.level,
                bootstrap.seed,
            )
            .expect("enough engines for the bootstrap");

            let verdict = if report.confidence_interval.lo > 0.0
            {
                "robust_wins"
            }
            else if report.confidence_interval.hi < 0.0
            {
                "ols_wins"
            }
            else
            {
                "straddles_zero"
            };

            records.push(
                BenchRecord::new(
                    "leverage/paired",
                    cell.clone(),
                    format!("ols_minus_{robust}"),
                    bootstrap.seed,
                    "rmse_paired_mean_difference",
                    report.mean_difference,
                )
                .with_ci(report.confidence_interval),
            );

            summary.push(format!(
                "#   OLS − {robust:11}: mean Δ={:+.3} CI=[{:+.3},{:+.3}] → {verdict}",
                report.mean_difference,
                report.confidence_interval.lo,
                report.confidence_interval.hi,
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

    println!("# leverage — Lever 2: does robustness win under a high-leverage attack?");
    println!(
        "# task: piecewise RUL (125) + stride {STRIDE}; attack: fractions {LEVERAGE_FRACTIONS:?} \
of train rows pushed {FEATURE_SHIFT_MADS} MADs out, target={CORRUPT_TARGET}"
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
        &config.bootstrap,
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
        &config.bootstrap,
        &mut records,
        &mut summary,
    );

    for line in &summary
    {
        println!("{line}");
    }

    fs::create_dir_all(&out_dir).expect("results directory is writable");
    fs::write(out_dir.join("leverage.jsonl"), to_jsonl(&records))
        .expect("results file is writable");

    println!("# records={}", records.len());
}
