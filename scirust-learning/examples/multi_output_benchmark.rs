//! Deterministic benchmark for phase 4E.4 — robust multi-output regression.
//!
//! Two sensor channels share a common noise term, so their fit residuals are
//! almost perfectly positively correlated: the residual cloud collapses onto the
//! `r₁ ≈ r₂` line and the `(+,−)` direction carries essentially zero variance.
//! Contamination is injected *against* that correlation — `(+δ, −δ)` on a cluster
//! of rows — which is only **moderate per channel** (below a per-output outlier
//! cutoff) yet **extreme jointly**. This separates the three residual geometries:
//!
//! - `IndependentOutputs` and `Euclidean` see only moderate per-channel /
//!   isotropic residuals and barely down-weight the cluster;
//! - `RobustMahalanobis` models the output covariance (via OGK) and flags the
//!   cluster as far in the near-degenerate direction, rejecting it.
//!
//! The cluster is **centered on the design** (`x̄` of the cluster equals `x̄` of
//! the sample), so the anomaly is slope-neutral and its corruption lands on the
//! **intercepts** — where a marginal method cannot escape it. This is deliberately
//! *low leverage*: it is the geometry `RobustMahalanobis` is built for. It is NOT
//! the bad-leverage regime (contamination at extreme `x`), which no IRLS reweighter
//! can fix and which needs the high-breakdown estimators from phase 4E.3.
//!
//! Ground truth: slopes `(2, 3)`, intercepts `(0, 0)` (the shared noise is
//! recentred to exact sample-mean zero). The metric is the total coefficient error
//! `|β̂₁−2| + |β̂₂−3| + |α̂₁| + |α̂₂|`. OLS is the non-robust baseline. Run twice,
//! byte-identical.

use scirust_learning::{
    MultiOutputResidualGeometry, MultiOutputRobustConfig, RegressionDataset,
    RobustRegressionConfig, RobustRegressionMethod, fit_multi_output_robust, fit_robust_regression,
};
use scirust_solvers::linalg::Matrix;

const N: usize = 60;
const TRUE_SLOPES: [f64; 2] = [2.0, 3.0];

fn base() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut xs = Vec::with_capacity(N);
    let mut shared = Vec::with_capacity(N);
    for i in 0..N
    {
        xs.push(i as f64);
        shared.push(((i * 13) % 21) as f64 - 10.0); // common noise, roughly ±10
    }
    // Recentre the shared noise to exact sample-mean zero, so the true intercepts
    // are 0 and intercept error is directly interpretable.
    let mean = shared.iter().sum::<f64>() / N as f64;
    for s in &mut shared
    {
        *s -= mean;
    }
    let y1: Vec<f64> = (0..N).map(|i| TRUE_SLOPES[0] * xs[i] + shared[i]).collect();
    let y2: Vec<f64> = (0..N).map(|i| TRUE_SLOPES[1] * xs[i] + shared[i]).collect();
    (xs, y1, y2)
}

fn scenario(name: &str) -> (Matrix, Matrix) {
    let (xs, mut y1, mut y2) = base();
    match name
    {
        "clean" =>
        {},
        "per_channel" =>
        {
            // Ten large vertical outliers on channel 1 only.
            for &i in &[0, 6, 12, 18, 24, 30, 36, 42, 48, 54]
            {
                y1[i] += 400.0;
            }
        },
        "anti_correlated" =>
        {
            // A centred cluster shifted against the shared-noise correlation:
            // moderate per channel (±12 on ±10 noise), extreme in the (+,−)
            // direction. Rows 24..36 have mean x = 29.5 = the sample mean, so the
            // anomaly is slope-neutral and corrupts the intercepts.
            for i in 24..36
            {
                y1[i] += 12.0;
                y2[i] -= 12.0;
            }
        },
        other => panic!("unknown scenario: {other}"),
    }
    let features = Matrix::from_row_major(N, 1, xs);
    let mut targets_data = Vec::with_capacity(N * 2);
    for i in 0..N
    {
        targets_data.push(y1[i]);
        targets_data.push(y2[i]);
    }
    (features, Matrix::from_row_major(N, 2, targets_data))
}

fn coefficient_error(slopes: [f64; 2], intercepts: [f64; 2]) -> f64 {
    (slopes[0] - TRUE_SLOPES[0]).abs()
        + (slopes[1] - TRUE_SLOPES[1]).abs()
        + intercepts[0].abs()
        + intercepts[1].abs()
}

fn ols_error(features: &Matrix, targets: &Matrix) -> f64 {
    let dataset = RegressionDataset {
        features: features.clone(),
        targets: targets.clone(),
        sample_weights: None,
    };
    let model = fit_robust_regression(
        &dataset,
        RobustRegressionConfig {
            method: RobustRegressionMethod::OrdinaryLeastSquares,
            ..RobustRegressionConfig::default()
        },
    )
    .expect("ols fits")
    .model;
    coefficient_error(
        [model.coefficients[(0, 0)], model.coefficients[(0, 1)]],
        [model.intercept[0], model.intercept[1]],
    )
}

fn geometry_error(
    features: &Matrix,
    targets: &Matrix,
    geometry: MultiOutputResidualGeometry,
) -> f64 {
    let report = fit_multi_output_robust(
        features,
        targets,
        MultiOutputRobustConfig {
            residual_geometry: geometry,
            ..MultiOutputRobustConfig::default()
        },
    )
    .expect("multi-output fits");
    coefficient_error(
        [report.coefficients[(0, 0)], report.coefficients[(0, 1)]],
        [report.intercepts[0], report.intercepts[1]],
    )
}

fn main() {
    println!("# multi_output_benchmark — phase 4E.4");
    println!("# metric: total coefficient error |b1-2|+|b2-3|+|a1|+|a2| (lower = better)");
    println!("# scenario           ols       independent  euclidean   robust_mahalanobis");

    for name in ["clean", "per_channel", "anti_correlated"]
    {
        let (features, targets) = scenario(name);
        let ols = ols_error(&features, &targets);
        let independent = geometry_error(
            &features,
            &targets,
            MultiOutputResidualGeometry::IndependentOutputs,
        );
        let euclidean = geometry_error(&features, &targets, MultiOutputResidualGeometry::Euclidean);
        let mahalanobis = geometry_error(
            &features,
            &targets,
            MultiOutputResidualGeometry::RobustMahalanobis,
        );
        println!(
            "{name:<18} {ols:.4}    {independent:.4}       {euclidean:.4}      {mahalanobis:.4}"
        );
    }
}
