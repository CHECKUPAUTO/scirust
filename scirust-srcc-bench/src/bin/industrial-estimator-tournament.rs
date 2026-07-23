//! Deterministic benchmark for phase 4E.5 — the adaptive estimator tournament
//! with abstention, wired end to end under a leakage-free protocol.
//!
//! For each scenario we generate a synthetic single-feature regression
//! (`y = 3x + 2 + noise`), split the rows into disjoint **training** and
//! **validation** halves with a seeded shuffle (the validation rows are never
//! seen during fitting — leakage-free), inject contamination **into training
//! rows only**, fit five estimators on training, score each on the *clean*
//! held-out validation rows (per-sample absolute error), and feed those per-unit
//! scores to [`EstimatorTournament`]. OLS is the incumbent; Huber, Tukey,
//! FAST-LTS and MM are the challengers.
//!
//! Contaminating only the training side is the honest robustness question: a
//! model that correctly ignores training outliers should predict the clean
//! validation line well, so its held-out error is low. (Contaminating validation
//! too would just penalise every method for the outliers' own error and hide the
//! effect.)
//!
//! The point is honest *decisions*, not a fixed winner:
//!
//! - `clean` — every method fits the same line; the tournament refuses to switch
//!   (HoldIncumbent / Inconclusive).
//! - `vertical` — vertical training outliers break OLS; all four robust methods
//!   recover and are statistically indistinguishable → Tie.
//! - `bad_leverage` — high-leverage training outliers (the phase-4E.3 regime)
//!   also drag Huber and Tukey off the line; only the multi-start high-breakdown
//!   estimators recover, so the *beaters* are LTS and MM while Huber/Tukey do not
//!   beat the incumbent.
//! - `tiny_validation` — the clean signal but only a handful of validation units,
//!   so the intervals are wide and the tournament abstains (Inconclusive).
//!
//! Everything is seeded; the program is byte-identical across runs.

use scirust_learning::{
    HighBreakdownConfig, HighBreakdownMethod, RegressionDataset, RobustLoss,
    RobustRegressionConfig, RobustRegressionMethod, fit_high_breakdown, fit_robust_regression,
};
use scirust_solvers::linalg::Matrix;
use scirust_srcc_bench::{EstimatorTournament, Orientation, TournamentDecision, TournamentEntry};
use scirust_stats::SplitMix64;

const TRUE_SLOPE: f64 = 3.0;
const TRUE_INTERCEPT: f64 = 2.0;
const SPLIT_SEED: u64 = 0x5350_4C49_5421;
const TOURNAMENT_SEED: u64 = 0x0054_4F55_524E_4559;

/// A candidate estimator, named for the tournament.
#[derive(Clone, Copy)]
enum Estimator {
    OrdinaryLeastSquares,
    Huber,
    Tukey,
    FastLts,
    Mm,
}

impl Estimator {
    fn name(self) -> &'static str {
        match self
        {
            Self::OrdinaryLeastSquares => "ols",
            Self::Huber => "huber",
            Self::Tukey => "tukey",
            Self::FastLts => "lts",
            Self::Mm => "mm",
        }
    }
}

/// Deterministic base signal (no contamination).
fn base(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n
    {
        let x = i as f64 * 0.5;
        // Small, bounded, mean-near-zero deterministic noise.
        let noise = (((i * 17) % 11) as f64 - 5.0) * 0.05;
        xs.push(x);
        ys.push(TRUE_SLOPE * x + TRUE_INTERCEPT + noise);
    }
    (xs, ys)
}

/// Number of rows for a scenario.
fn scenario_rows(name: &str) -> usize {
    if name == "tiny_validation" { 24 } else { 80 }
}

/// Injects a named contamination pattern into the **training rows only**
/// (`xs`/`ys` are mutated in place; validation rows are left clean).
fn contaminate(name: &str, xs: &mut [f64], ys: &mut [f64], train: &[usize]) {
    match name
    {
        "clean" | "tiny_validation" =>
        {},
        "vertical" =>
        {
            // Large vertical outliers on the first ten training rows.
            for &i in train.iter().take(10)
            {
                ys[i] += 45.0;
            }
        },
        "bad_leverage" =>
        {
            // High-leverage training outliers: move the first eight training rows
            // far past the design range in x and drop them far below the line —
            // the phase-4E.3 regime that single-start IRLS cannot escape.
            for (k, &i) in train.iter().take(8).enumerate()
            {
                xs[i] = 45.0 + k as f64;
                ys[i] = TRUE_SLOPE * xs[i] + TRUE_INTERCEPT - 120.0;
            }
        },
        other => panic!("unknown scenario: {other}"),
    }
}

/// A seeded Fisher–Yates shuffle of `0..n` (deterministic).
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

/// Fit `estimator` on the training rows and return per-unit absolute errors on
/// the validation rows. `None` if the fit fails.
fn validation_errors(
    estimator: Estimator,
    xs: &[f64],
    ys: &[f64],
    train: &[usize],
    validation: &[usize],
) -> Option<Vec<f64>> {
    let train_x: Vec<f64> = train.iter().map(|&i| xs[i]).collect();
    let train_y: Vec<f64> = train.iter().map(|&i| ys[i]).collect();
    let features = Matrix::from_row_major(train_x.len(), 1, train_x);

    let (slope, intercept) = match estimator
    {
        Estimator::OrdinaryLeastSquares | Estimator::Huber | Estimator::Tukey =>
        {
            let dataset = RegressionDataset {
                features,
                targets: Matrix::from_row_major(train_y.len(), 1, train_y),
                sample_weights: None,
            };
            let (method, loss) = match estimator
            {
                Estimator::OrdinaryLeastSquares => (
                    RobustRegressionMethod::OrdinaryLeastSquares,
                    RobustLoss::Squared,
                ),
                Estimator::Huber => (
                    RobustRegressionMethod::IterativelyReweightedLeastSquares,
                    RobustLoss::Huber { delta: 1.345 },
                ),
                _ => (
                    RobustRegressionMethod::IterativelyReweightedLeastSquares,
                    RobustLoss::TukeyBisquare { cutoff: 4.685 },
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
            .ok()?
            .model;
            (model.coefficients[(0, 0)], model.intercept[0])
        },
        Estimator::FastLts | Estimator::Mm =>
        {
            let method = if matches!(estimator, Estimator::FastLts)
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
            .ok()?;
            (report.coefficients[0], report.intercept)
        },
    };

    Some(
        validation
            .iter()
            .map(|&i| (slope * xs[i] + intercept - ys[i]).abs())
            .collect(),
    )
}

fn main() {
    println!("# industrial_estimator_tournament — phase 4E.5");
    println!("# leakage-free holdout; per-unit validation MAE; OLS is the incumbent.");
    println!("# a candidate 'beats' only if its bootstrap improvement lower bound > 0.");

    let tournament = EstimatorTournament {
        orientation: Orientation::LowerIsBetter,
        min_improvement: 0.0,
        tie_margin: 0.0,
        quality_floor: None,
        resamples: 4000,
        level: 0.95,
        seed: TOURNAMENT_SEED,
    };

    let challengers = [
        Estimator::Huber,
        Estimator::Tukey,
        Estimator::FastLts,
        Estimator::Mm,
    ];

    for name in ["clean", "vertical", "bad_leverage", "tiny_validation"]
    {
        let n = scenario_rows(name);
        let (mut xs, mut ys) = base(n);
        let order = shuffled_indices(n, SPLIT_SEED);
        let cut = n * 6 / 10;
        let mut train: Vec<usize> = order[..cut].to_vec();
        let mut validation: Vec<usize> = order[cut..].to_vec();
        train.sort_unstable();
        validation.sort_unstable();

        // Leakage-free: contaminate only the training rows; validation stays clean.
        contaminate(name, &mut xs, &mut ys, &train);

        let incumbent = TournamentEntry::new(
            Estimator::OrdinaryLeastSquares.name(),
            validation_errors(
                Estimator::OrdinaryLeastSquares,
                &xs,
                &ys,
                &train,
                &validation,
            )
            .expect("ols fits"),
        );
        let candidates: Vec<TournamentEntry> = challengers
            .iter()
            .map(|&estimator| {
                TournamentEntry::new(
                    estimator.name(),
                    validation_errors(estimator, &xs, &ys, &train, &validation)
                        .expect("challenger fits"),
                )
            })
            .collect();

        let report = tournament
            .evaluate(&incumbent, &candidates)
            .expect("tournament evaluates");

        let verdict = match &report.decision
        {
            TournamentDecision::Select { winner } => format!("Select({winner})"),
            TournamentDecision::HoldIncumbent => "HoldIncumbent".to_string(),
            TournamentDecision::Tie { contenders } => format!("Tie({})", contenders.join(",")),
            TournamentDecision::Inconclusive => "Inconclusive".to_string(),
            TournamentDecision::RejectAll => "RejectAll".to_string(),
        };

        println!();
        println!(
            "# {name:<16} train={} validation={} -> {verdict}",
            train.len(),
            validation.len()
        );
        println!("#   incumbent ols mae={:.4}", mean(&incumbent.scores));
        for finding in &report.findings
        {
            println!(
                "#   {:<6} mae={:.4}  improvement={:+.4} [{:+.4}, {:+.4}]  beats={}",
                finding.name,
                finding.mean_score,
                finding.mean_improvement,
                finding.improvement_interval.lo,
                finding.improvement_interval.hi,
                finding.beats_incumbent,
            );
        }
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}
