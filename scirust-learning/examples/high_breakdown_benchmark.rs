//! Deterministic benchmark for phase 4E.3 — high-breakdown regression.
//!
//! Phase 4E.2 showed Huber recovers the true slope against vertical outliers but
//! collapses under bad leverage. This compares the full estimator family on the
//! same controlled geometries by their **slope-recovery error** `|β̂ − 2|`
//! (ground-truth slope 2):
//!
//! - OLS, Huber, Tukey (the existing IRLS losses);
//! - single-start Trimmed LS and Median-of-Means (the existing high-ish-breakdown
//!   methods);
//! - the new FAST-LTS and MM estimators.
//!
//! The expected story: everything recovers on clean/vertical; only the
//! **multi-start** high-breakdown estimators (LTS, MM) — and possibly single-start
//! Trimmed LS if its one OLS start happens to escape — recover under bad leverage.
//! Run twice, byte-identical.

use scirust_learning::{
    HighBreakdownConfig, HighBreakdownMethod, RegressionDataset, RobustLoss,
    RobustRegressionConfig, RobustRegressionMethod, fit_high_breakdown, fit_robust_regression,
};
use scirust_solvers::linalg::Matrix;

const TRUE_SLOPE: f64 = 2.0;

fn design(xs: &[f64]) -> Matrix {
    Matrix::from_row_major(xs.len(), 1, xs.to_vec())
}

fn clean(n: usize) -> (Vec<f64>, Vec<f64>) {
    let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let ys: Vec<f64> = (0..n)
        .map(|i| TRUE_SLOPE * i as f64 + (((i * 7) % 11) as f64 - 5.0) * 0.2)
        .collect();
    (xs, ys)
}

fn geometry(name: &str) -> (Vec<f64>, Vec<f64>) {
    let (mut xs, mut ys) = clean(50);
    match name
    {
        "clean" =>
        {},
        "vertical" =>
        {
            for k in 0..10
            {
                let x = (k * 4) as f64;
                xs.push(x);
                ys.push(TRUE_SLOPE * x + 70.0);
            }
        },
        "bad_leverage" =>
        {
            for k in 0..10
            {
                let x = 120.0 + k as f64;
                xs.push(x);
                ys.push(TRUE_SLOPE * x - 160.0);
            }
        },
        "mixed" =>
        {
            for k in 0..6
            {
                let x = (k * 5) as f64;
                xs.push(x);
                ys.push(TRUE_SLOPE * x + 70.0);
            }
            for k in 0..6
            {
                let x = 120.0 + k as f64;
                xs.push(x);
                ys.push(TRUE_SLOPE * x - 160.0);
            }
        },
        other => panic!("unknown geometry: {other}"),
    }
    (xs, ys)
}

fn robust_slope(xs: &[f64], ys: &[f64], method: RobustRegressionMethod, loss: RobustLoss) -> f64 {
    let dataset = RegressionDataset {
        features: design(xs),
        targets: Matrix::from_row_major(ys.len(), 1, ys.to_vec()),
        sample_weights: None,
    };
    let config = RobustRegressionConfig {
        method,
        loss,
        ..RobustRegressionConfig::default()
    };
    fit_robust_regression(&dataset, config)
        .expect("regression fits")
        .model
        .coefficients[(0, 0)]
}

fn high_breakdown_slope(xs: &[f64], ys: &[f64], method: HighBreakdownMethod) -> f64 {
    fit_high_breakdown(
        &design(xs),
        ys,
        HighBreakdownConfig {
            method,
            ..HighBreakdownConfig::default()
        },
    )
    .expect("high-breakdown fits")
    .coefficients[0]
}

fn main() {
    println!("# high_breakdown_benchmark — phase 4E.3");
    println!("# ground truth slope = {TRUE_SLOPE}; each cell is |slope_est - 2| (lower = better)");
    println!("# geometry            ols     huber   tukey   trimmed mom     lts     mm");

    for name in ["clean", "vertical", "bad_leverage", "mixed"]
    {
        let (xs, ys) = geometry(name);
        let ols = robust_slope(
            &xs,
            &ys,
            RobustRegressionMethod::OrdinaryLeastSquares,
            RobustLoss::Squared,
        );
        let huber = robust_slope(
            &xs,
            &ys,
            RobustRegressionMethod::IterativelyReweightedLeastSquares,
            RobustLoss::Huber { delta: 1.345 },
        );
        let tukey = robust_slope(
            &xs,
            &ys,
            RobustRegressionMethod::IterativelyReweightedLeastSquares,
            RobustLoss::TukeyBisquare { cutoff: 4.685 },
        );
        let trimmed = robust_slope(
            &xs,
            &ys,
            RobustRegressionMethod::TrimmedLeastSquares {
                retained_fraction: 0.75,
            },
            RobustLoss::Squared,
        );
        let mom = robust_slope(
            &xs,
            &ys,
            RobustRegressionMethod::MedianOfMeans {
                block_count: 5,
                seed: 43,
            },
            RobustLoss::Squared,
        );
        let lts = high_breakdown_slope(&xs, &ys, HighBreakdownMethod::LeastTrimmedSquares);
        let mm = high_breakdown_slope(&xs, &ys, HighBreakdownMethod::MmEstimator);

        let error = |slope: f64| (slope - TRUE_SLOPE).abs();
        println!(
            "{name:<18} {:.4}  {:.4}  {:.4}  {:.4}  {:.4}  {:.4}  {:.4}",
            error(ols),
            error(huber),
            error(tukey),
            error(trimmed),
            error(mom),
            error(lts),
            error(mm),
        );
    }
}
