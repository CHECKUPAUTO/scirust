//! Deterministic benchmark for phase 4E.1 — robust affine scatter (OGK).
//!
//! Compares three scatter estimators — classical covariance, robust *diagonal*
//! (MAD², correlation-blind), and OGK robust scatter — under clean data and four
//! contamination geometries, on two axes the program cares about:
//!
//! - **correlation error** under contamination: how far each estimator's implied
//!   correlation drifts from the clean-data OGK reference (lower = more robust);
//! - **outlier recall**: the fraction of injected outliers that land in the top-`m`
//!   of each estimator's Mahalanobis ranking (higher = better masking resistance).
//!
//! It also runs an OGK dimension sweep (p = 2, 4, 6) reporting reweighting
//! iterations and rejects. No RNG: every dataset is a fixed deterministic pattern,
//! so the whole stdout is a stable fingerprint (`… | sha256sum`, run twice).

use scirust_multivariate::{
    Matrix, RobustScaleMethod, RobustScalerConfig, RobustScatterConfig, RobustScatterMethod,
    RobustScatterModel, RobustUnivariateScale, ZeroScalePolicy,
};

fn mat(data: Vec<Vec<f64>>) -> Matrix {
    let rows = data.len();
    let cols = data[0].len();
    Matrix { rows, cols, data }
}

/// A deterministic, healthy-rank correlated 2-D cloud (slope 0.8 + bounded wiggle).
fn clean(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let x = (i as f64) - (n as f64) / 2.0;
            let wiggle = ((i * 7) % 13) as f64 - 6.0;
            vec![x, 0.8 * x + wiggle]
        })
        .collect()
}

/// Clean cloud plus `m` gross **vertical** outliers (extreme in `y` only).
fn vertical(n: usize, m: usize) -> Vec<Vec<f64>> {
    let mut rows = clean(n);
    for k in 0..m
    {
        let x = (k as f64) - (m as f64) / 2.0;
        rows.push(vec![x, 60.0 + k as f64]);
    }
    rows
}

/// Clean cloud plus `m` **bad-leverage** outliers (extreme in `x`, off the line).
fn leverage(n: usize, m: usize) -> Vec<Vec<f64>> {
    let mut rows = clean(n);
    for k in 0..m
    {
        rows.push(vec![40.0 + k as f64, -30.0 - k as f64]);
    }
    rows
}

/// Clean cloud plus a minority forming a second, **anti-correlated** cluster.
fn correlated_contamination(n: usize, m: usize) -> Vec<Vec<f64>> {
    let mut rows = clean(n);
    for k in 0..m
    {
        let t = k as f64;
        rows.push(vec![10.0 + t, -25.0 - 2.0 * t]);
    }
    rows
}

/// A deterministic p-dimensional correlated cloud for the dimension sweep.
fn cloud_p(n: usize, p: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            let base = (i as f64) - (n as f64) / 2.0;
            (0..p)
                .map(|j| {
                    let wiggle = ((i * (7 + j) + j * 3) % 13) as f64 - 6.0;
                    0.6 * (j as f64 + 1.0) * base + wiggle
                })
                .collect()
        })
        .collect()
}

fn ogk_config(reweight: bool) -> RobustScatterConfig {
    RobustScatterConfig {
        method: RobustScatterMethod::Ogk {
            scale: RobustUnivariateScale::MedianAbsoluteDeviation,
            reweight,
        },
        ridge: 0.0,
        ..RobustScatterConfig::default()
    }
}

fn classical_config() -> RobustScatterConfig {
    RobustScatterConfig {
        method: RobustScatterMethod::Classical,
        ridge: 0.0,
        ..RobustScatterConfig::default()
    }
}

/// Correlation implied by a 2×2 scatter.
fn correlation(model: &RobustScatterModel) -> f64 {
    let s = &model.scatter;
    let denominator = (s.data[0][0] * s.data[1][1]).sqrt();
    if denominator > 0.0
    {
        s.data[0][1] / denominator
    }
    else
    {
        0.0
    }
}

/// Fraction of the injected outliers (the last `m` rows) that fall in the top-`m`
/// of the squared Mahalanobis ranking.
fn outlier_recall(distances: &[f64], m: usize) -> f64 {
    let n = distances.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| distances[b].total_cmp(&distances[a]).then(a.cmp(&b)));
    let injected: std::collections::BTreeSet<usize> = (n - m..n).collect();
    let hits = order
        .iter()
        .take(m)
        .filter(|i| injected.contains(i))
        .count();
    hits as f64 / m as f64
}

/// Squared Mahalanobis distances of every row under a fitted scatter model.
fn model_distances(model: &RobustScatterModel, data: &Matrix) -> Vec<f64> {
    (0..data.rows)
        .map(|i| model.mahalanobis_squared(&data.data[i]).unwrap())
        .collect()
}

/// Robust *diagonal* distances: Σ_j ((x_j − median_j) / MAD_j)². Correlation-blind.
fn diagonal_distances(data: &Matrix) -> Vec<f64> {
    let scaler = scirust_multivariate::RobustScaler::fit(
        data,
        RobustScalerConfig {
            scale_method: RobustScaleMethod::MedianAbsoluteDeviation,
            zero_scale_policy: ZeroScalePolicy::DropDimension,
            ..RobustScalerConfig::default()
        },
    )
    .expect("robust scaler fits the benchmark clouds");
    let transformed = scaler
        .transform(data)
        .expect("transform matches fitted dims");
    (0..transformed.rows)
        .map(|i| transformed.data[i].iter().map(|v| v * v).sum())
        .collect()
}

fn main() {
    println!("# robust_scatter_benchmark — phase 4E.1 (OGK)");
    println!("# estimators: classical covariance | robust diagonal (MAD², corr-blind) | OGK");
    println!(
        "# metric 1: |correlation − clean-OGK reference| under contamination (lower = robust)"
    );
    println!("# metric 2: outlier recall = injected outliers in the top-m Mahalanobis ranking");

    // Reference correlation from OGK on clean data.
    let clean_data = mat(clean(60));
    let reference = RobustScatterModel::fit(&clean_data, ogk_config(true)).unwrap();
    let reference_correlation = correlation(&reference);
    println!("clean ogk reference_correlation {reference_correlation:.6}");

    let conditions: [(&str, Matrix, usize); 4] = [
        ("clean", mat(clean(60)), 0),
        ("vertical", mat(vertical(50, 10)), 10),
        ("leverage", mat(leverage(50, 10)), 10),
        ("correlated", mat(correlated_contamination(50, 10)), 10),
    ];

    for (name, data, outliers) in &conditions
    {
        let classical = RobustScatterModel::fit(data, classical_config()).unwrap();
        let ogk = RobustScatterModel::fit(data, ogk_config(true)).unwrap();

        let classical_error = (correlation(&classical) - reference_correlation).abs();
        let ogk_error = (correlation(&ogk) - reference_correlation).abs();
        println!("{name} classical correlation_error {classical_error:.6}");
        println!("{name} ogk correlation_error {ogk_error:.6}");
        println!(
            "{name} ogk rejected {} iterations {}",
            ogk.report.reweighted_outlier_count, ogk.report.iterations
        );

        if *outliers > 0
        {
            let classical_recall = outlier_recall(&model_distances(&classical, data), *outliers);
            let diagonal_recall = outlier_recall(&diagonal_distances(data), *outliers);
            let ogk_recall = outlier_recall(&model_distances(&ogk, data), *outliers);
            println!("{name} classical outlier_recall {classical_recall:.6}");
            println!("{name} diagonal outlier_recall {diagonal_recall:.6}");
            println!("{name} ogk outlier_recall {ogk_recall:.6}");
        }
    }

    // Dimension sweep — OGK stays positive definite and converges as p grows.
    for p in [2_usize, 4, 6]
    {
        let data = mat(cloud_p(80, p));
        let model = RobustScatterModel::fit(&data, ogk_config(true)).unwrap();
        let positive_definite = (0..p).all(|j| model.scatter.data[j][j] > 0.0);
        println!(
            "dim{p} ogk pd {} converged {} iterations {} effective {}",
            positive_definite,
            model.report.converged,
            model.report.iterations,
            model.report.effective_sample_count
        );
    }
}
