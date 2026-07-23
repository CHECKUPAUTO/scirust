//! Deterministic benchmark for phase 4E.7 — the unified certified decision
//! pipeline, wired end to end.
//!
//! For each scenario we take a synthetic single-feature regression, split the
//! rows three ways (train / calibrate / test) with a seeded shuffle, contaminate
//! the **training rows only**, fit OLS/Huber/LTS/MM on training, and compute each
//! estimator's signed residuals on the calibration and test folds. Those residual
//! sets are the leakage-free evidence handed to [`run_certified_pipeline`], which
//! selects an estimator (or abstains) and certifies coverage.
//!
//! The output is the whole certificate: the selection verdict, the deployed (or
//! provisionally-retained) estimator, the empirical test coverage of its conformal
//! band, and the explicit lists of what the certificate does and does not
//! guarantee. The `heteroscedastic` scenario uses a group label so the pipeline
//! issues a per-group (Mondrian) certificate.
//!
//! Everything is seeded; the program is byte-identical across runs.

use scirust_learning::{
    HighBreakdownConfig, HighBreakdownMethod, RegressionDataset, RobustLoss,
    RobustRegressionConfig, RobustRegressionMethod, fit_high_breakdown, fit_robust_regression,
};
use scirust_solvers::linalg::Matrix;
use scirust_srcc_bench::{
    CertifiedPipelineConfig, CoverageKind, CoverageMode, EstimatorEvidence, EstimatorTournament,
    Orientation, TournamentDecision, run_certified_pipeline,
};
use scirust_stats::SplitMix64;

const TRUE_SLOPE: f64 = 3.0;
const TRUE_INTERCEPT: f64 = 2.0;
const SPLIT_SEED: u64 = 0x00C0_FFEE_5EED;
const TOURNAMENT_SEED: u64 = 0x00CE_5712_F1ED;

#[derive(Clone, Copy)]
enum Estimator {
    Ols,
    Huber,
    Lts,
    Mm,
}

impl Estimator {
    fn name(self) -> &'static str {
        match self
        {
            Self::Ols => "ols",
            Self::Huber => "huber",
            Self::Lts => "lts",
            Self::Mm => "mm",
        }
    }
}

fn base(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n
    {
        let x = i as f64 * 0.25;
        let noise = (((i * 17) % 11) as f64 - 5.0) * 0.05;
        xs.push(x);
        ys.push(TRUE_SLOPE * x + TRUE_INTERCEPT + noise);
    }
    (xs, ys)
}

/// Contaminate training rows only; also widen the noise of the high group in the
/// heteroscedastic scenario (applied to *all* rows since it is a property of the
/// data, not contamination).
fn scenario(name: &str) -> (Vec<f64>, Vec<f64>, usize) {
    let n = 160;
    let (xs, mut ys) = base(n);
    match name
    {
        "clean" =>
        {},
        "bad_leverage" | "heteroscedastic" =>
        {},
        other => panic!("unknown scenario: {other}"),
    }
    if name == "heteroscedastic"
    {
        // The upper half of x is a noisier group: inflate its residual spread.
        for (i, y) in ys.iter_mut().enumerate()
        {
            if xs[i] >= xs[n / 2]
            {
                *y += (((i * 23) % 13) as f64 - 6.0) * 0.5;
            }
        }
    }
    (xs, ys, n)
}

fn contaminate_training(name: &str, xs: &mut [f64], ys: &mut [f64], train: &[usize]) {
    if name == "bad_leverage"
    {
        for (k, &i) in train.iter().take(12).enumerate()
        {
            xs[i] = 60.0 + k as f64;
            ys[i] = TRUE_SLOPE * xs[i] + TRUE_INTERCEPT - 150.0;
        }
    }
}

fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = SplitMix64::new(seed);
    for i in (1..n).rev()
    {
        let draw = (rng.next_f64() * (i + 1) as f64) as usize;
        indices.swap(i, draw.min(i));
    }
    indices
}

/// Fit `estimator` on the training rows; return signed residuals `y − ŷ` on the
/// evaluation rows.
fn signed_residuals(
    estimator: Estimator,
    xs: &[f64],
    ys: &[f64],
    train: &[usize],
    eval: &[usize],
) -> Vec<f64> {
    let train_x: Vec<f64> = train.iter().map(|&i| xs[i]).collect();
    let train_y: Vec<f64> = train.iter().map(|&i| ys[i]).collect();
    let features = Matrix::from_row_major(train_x.len(), 1, train_x);

    let (slope, intercept) = match estimator
    {
        Estimator::Ols | Estimator::Huber =>
        {
            let dataset = RegressionDataset {
                features,
                targets: Matrix::from_row_major(train_y.len(), 1, train_y),
                sample_weights: None,
            };
            let (method, loss) = match estimator
            {
                Estimator::Ols => (
                    RobustRegressionMethod::OrdinaryLeastSquares,
                    RobustLoss::Squared,
                ),
                _ => (
                    RobustRegressionMethod::IterativelyReweightedLeastSquares,
                    RobustLoss::Huber { delta: 1.345 },
                ),
            };
            let model = fit_robust_regression(
                &dataset,
                RobustRegressionConfig {
                    method,
                    loss,
                    ..RobustRegressionConfig::default()
                },
            )
            .expect("regression fits")
            .model;
            (model.coefficients[(0, 0)], model.intercept[0])
        },
        Estimator::Lts | Estimator::Mm =>
        {
            let method = if matches!(estimator, Estimator::Lts)
            {
                HighBreakdownMethod::LeastTrimmedSquares
            }
            else
            {
                HighBreakdownMethod::MmEstimator
            };
            let report = fit_high_breakdown(
                &features,
                &train_y,
                HighBreakdownConfig {
                    method,
                    ..HighBreakdownConfig::default()
                },
            )
            .expect("high-breakdown fits");
            (report.coefficients[0], report.intercept)
        },
    };

    eval.iter()
        .map(|&i| ys[i] - (slope * xs[i] + intercept))
        .collect()
}

fn evidence(
    estimator: Estimator,
    xs: &[f64],
    ys: &[f64],
    train: &[usize],
    calib: &[usize],
    test: &[usize],
) -> EstimatorEvidence {
    EstimatorEvidence::new(
        estimator.name(),
        signed_residuals(estimator, xs, ys, train, calib),
        signed_residuals(estimator, xs, ys, train, test),
    )
}

fn group_labels(xs: &[f64], rows: &[usize], n: usize) -> Vec<u64> {
    let threshold = xs[n / 2];
    rows.iter()
        .map(|&i| u64::from(xs[i] >= threshold))
        .collect()
}

fn run_scenario(name: &str) {
    let (mut xs, mut ys, n) = scenario(name);
    let order = shuffled_indices(n, SPLIT_SEED);
    // 50% train, 25% calibrate, 25% test.
    let train_cut = n / 2;
    let calib_cut = train_cut + n / 4;
    let mut train: Vec<usize> = order[..train_cut].to_vec();
    let mut calib: Vec<usize> = order[train_cut..calib_cut].to_vec();
    let mut test: Vec<usize> = order[calib_cut..].to_vec();
    train.sort_unstable();
    calib.sort_unstable();
    test.sort_unstable();

    contaminate_training(name, &mut xs, &mut ys, &train);

    let incumbent = evidence(Estimator::Ols, &xs, &ys, &train, &calib, &test);
    let challengers = [
        evidence(Estimator::Huber, &xs, &ys, &train, &calib, &test),
        evidence(Estimator::Lts, &xs, &ys, &train, &calib, &test),
        evidence(Estimator::Mm, &xs, &ys, &train, &calib, &test),
    ];

    let coverage_mode = if name == "heteroscedastic"
    {
        CoverageMode::GroupConditional {
            calibration_groups: group_labels(&xs, &calib, n),
            test_groups: group_labels(&xs, &test, n),
        }
    }
    else
    {
        CoverageMode::Marginal
    };

    let config = CertifiedPipelineConfig {
        tournament: EstimatorTournament {
            orientation: Orientation::LowerIsBetter,
            min_improvement: 0.0,
            tie_margin: 0.0,
            quality_floor: None,
            resamples: 4000,
            level: 0.95,
            seed: TOURNAMENT_SEED,
        },
        coverage_level: 0.9,
        coverage_mode,
    };

    let report = run_certified_pipeline(&incumbent, &challengers, &config).expect("pipeline runs");

    let verdict = match &report.selection.decision
    {
        TournamentDecision::Select { winner } => format!("Select({winner})"),
        TournamentDecision::HoldIncumbent => "HoldIncumbent".to_string(),
        TournamentDecision::Tie { contenders } => format!("Tie({})", contenders.join(",")),
        TournamentDecision::Inconclusive => "Inconclusive".to_string(),
        TournamentDecision::RejectAll => "RejectAll".to_string(),
    };

    println!();
    println!("# {name}");
    println!(
        "#   split: train={} calibrate={} test={}",
        train.len(),
        calib.len(),
        test.len()
    );
    println!("#   selection : {verdict}");
    println!(
        "#   deploy    : {}",
        report
            .selected_estimator
            .as_deref()
            .unwrap_or("(none — abstained/rejected)")
    );
    match &report.coverage
    {
        None => println!("#   coverage  : (none certified)"),
        Some(cert) =>
        {
            println!(
                "#   coverage  : '{}' target={:.2} empirical_test={:.3} (n_cal={}, n_test={}){}",
                cert.estimator,
                cert.level,
                cert.empirical_test_coverage,
                cert.calibration_count,
                cert.test_count,
                if cert.provisional
                {
                    " [PROVISIONAL]"
                }
                else
                {
                    ""
                },
            );
            if let CoverageKind::GroupConditional { per_group } = &cert.kind
            {
                for g in per_group
                {
                    println!(
                        "#       group {}: half_width={:.4} empirical_test={:.3} conditional={}",
                        g.key, g.half_width, g.empirical_test_coverage, g.conditionally_valid,
                    );
                }
            }
        },
    }
    for guarantee in &report.guarantees
    {
        println!("#   + {guarantee}");
    }
    for caveat in &report.caveats
    {
        println!("#   - {caveat}");
    }
}

fn main() {
    println!("# industrial_certified_pipeline — phase 4E.7");
    println!("# leakage-free train/calibrate/test; select-or-abstain, then certify coverage.");

    for name in ["clean", "bad_leverage", "heteroscedastic"]
    {
        run_scenario(name);
    }
}
