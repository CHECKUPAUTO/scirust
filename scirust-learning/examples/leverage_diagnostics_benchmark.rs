//! Deterministic benchmark for phase 4E.2 — leverage & influence diagnostics.
//!
//! Direction 3D falsified the *kurtosis* frontier. This asks the sharper question
//! the leverage diagnostics make possible: across controlled geometries, does any
//! cheap diagnostic — OLS-residual excess kurtosis, max hat leverage, the
//! bad-leverage count, or the max robust feature-space distance — line up with
//! whether a robust fit (Huber IRLS) actually recovers the true slope better than
//! OLS?
//!
//! The ground truth is known (slope 2, intercept 0), so "robust benefit" is the
//! *slope-recovery* gain `|β_OLS − 2| − |β_Huber − 2|` (positive = Huber closer to
//! truth). A null or a partial result is a valid outcome — in particular Huber is
//! expected to help against vertical outliers but *not* against bad leverage,
//! which is exactly the gap the diagnostics expose and phase 4E.3 (high-breakdown
//! regression) must close. No RNG; run twice, byte-identical.

use scirust_learning::{
    InfluenceConfig, InfluenceReport, ObservationInfluenceClass, RegressionDataset, RobustLoss,
    RobustRegressionConfig, RobustRegressionMethod, fit_robust_regression,
};
use scirust_solvers::linalg::Matrix;

const TRUE_SLOPE: f64 = 2.0;

fn matrix(rows: usize, cols: usize, data: Vec<f64>) -> Matrix {
    Matrix::from_row_major(rows, cols, data)
}

/// A clean line y = 2x with a small bounded wiggle over x = 0..n.
fn clean(n: usize) -> Vec<(f64, f64)> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            (x, TRUE_SLOPE * x + (((i * 7) % 11) as f64 - 5.0) * 0.2)
        })
        .collect()
}

fn geometry(name: &str) -> Vec<(f64, f64)> {
    let mut points = clean(50);
    match name
    {
        "clean" =>
        {},
        "vertical" =>
        {
            // Ten large vertical outliers at ordinary x positions.
            for k in 0..10
            {
                let x = (k * 4) as f64;
                points.push((x, TRUE_SLOPE * x + 70.0));
            }
        },
        "good_leverage" =>
        {
            // Ten high-leverage points ON the line.
            for k in 0..10
            {
                let x = 120.0 + k as f64;
                points.push((x, TRUE_SLOPE * x));
            }
        },
        "bad_leverage" =>
        {
            // Ten high-leverage points OFF the line (Huber's weak spot).
            for k in 0..10
            {
                let x = 120.0 + k as f64;
                points.push((x, TRUE_SLOPE * x - 160.0));
            }
        },
        "mixed" =>
        {
            for k in 0..6
            {
                let x = (k * 5) as f64;
                points.push((x, TRUE_SLOPE * x + 70.0));
            }
            for k in 0..6
            {
                let x = 120.0 + k as f64;
                points.push((x, TRUE_SLOPE * x - 160.0));
            }
        },
        "nonlinear" =>
        {
            // Model misspecification: a quadratic truth fit by a line.
            points = (0..60)
                .map(|i| {
                    let x = i as f64;
                    (x, TRUE_SLOPE * x + 0.05 * x * x)
                })
                .collect();
        },
        "heteroscedastic" =>
        {
            // Noise amplitude grows with x (bounded, deterministic).
            points = (0..60)
                .map(|i| {
                    let x = i as f64;
                    let noise = (((i * 13) % 17) as f64 - 8.0) * (0.02 * x);
                    (x, TRUE_SLOPE * x + noise)
                })
                .collect();
        },
        other => panic!("unknown geometry: {other}"),
    }
    points
}

fn features_and_targets(points: &[(f64, f64)]) -> (Matrix, Vec<f64>) {
    let n = points.len();
    let features = matrix(n, 1, points.iter().map(|&(x, _)| x).collect());
    let targets: Vec<f64> = points.iter().map(|&(_, y)| y).collect();
    (features, targets)
}

fn fit_slope(
    features: &Matrix,
    targets: &[f64],
    method: RobustRegressionMethod,
    loss: RobustLoss,
) -> f64 {
    let dataset = RegressionDataset {
        features: features.clone(),
        targets: Matrix::from_row_major(targets.len(), 1, targets.to_vec()),
        sample_weights: None,
    };
    let config = RobustRegressionConfig {
        method,
        loss,
        ..RobustRegressionConfig::default()
    };
    let report = fit_robust_regression(&dataset, config).expect("regression fits the benchmark");
    report.model.coefficients[(0, 0)]
}

fn excess_kurtosis(values: &[f64]) -> f64 {
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let second = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count;
    let fourth = values.iter().map(|v| (v - mean).powi(4)).sum::<f64>() / count;
    if second <= 0.0
    {
        0.0
    }
    else
    {
        fourth / (second * second) - 3.0
    }
}

fn main() {
    println!("# leverage_diagnostics_benchmark — phase 4E.2");
    println!("# ground truth slope = {TRUE_SLOPE}; robust_benefit = |b_ols-2| - |b_huber-2|");
    println!(
        "# columns: geometry | kurtosis | max_leverage | bad_leverage | max_robust_dist | \
ols_slope_err | huber_slope_err | robust_benefit"
    );

    for name in [
        "clean",
        "vertical",
        "good_leverage",
        "bad_leverage",
        "mixed",
        "nonlinear",
        "heteroscedastic",
    ]
    {
        let points = geometry(name);
        let (features, targets) = features_and_targets(&points);

        let ols_slope = fit_slope(
            &features,
            &targets,
            RobustRegressionMethod::OrdinaryLeastSquares,
            RobustLoss::Squared,
        );
        let huber_slope = fit_slope(
            &features,
            &targets,
            RobustRegressionMethod::IterativelyReweightedLeastSquares,
            RobustLoss::Huber { delta: 1.345 },
        );
        let ols_error = (ols_slope - TRUE_SLOPE).abs();
        let huber_error = (huber_slope - TRUE_SLOPE).abs();
        let robust_benefit = ols_error - huber_error;

        let report = InfluenceReport::fit(&features, &targets, InfluenceConfig::default()).unwrap();
        let residuals: Vec<f64> = report.records.iter().map(|r| r.residual).collect();
        let kurtosis = excess_kurtosis(&residuals);
        let max_leverage = report
            .records
            .iter()
            .map(|r| r.leverage)
            .fold(0.0_f64, f64::max);
        let bad_leverage = report.count(ObservationInfluenceClass::BadLeverage);
        let max_robust_distance = report
            .records
            .iter()
            .filter_map(|r| r.robust_distance)
            .fold(0.0_f64, f64::max);

        println!(
            "{name:<16} kurtosis {kurtosis:.3} max_leverage {max_leverage:.3} bad_leverage {bad_leverage} \
max_robust_dist {max_robust_distance:.3} ols_slope_err {ols_error:.4} huber_slope_err {huber_error:.4} \
robust_benefit {robust_benefit:.4}"
        );
    }
}
