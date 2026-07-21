//! Deterministic mechanism benchmark for fitted scale-aware geometry.
//!
//! One fixed two-cluster dataset (deterministic `SplitMix64` → standard-normal
//! inverse CDF, via `scirust-stats`) is replayed under
//!
//! - **global** rescaling: every coordinate multiplied by a common `λ`;
//! - **anisotropic** rescaling: column `j` multiplied by an independent factor
//!   drawn from `{1, 1e3, 1e6, 1e9}`.
//!
//! For each transformation and each metric (raw Euclidean, relative norm,
//! robust diagonal (MAD), regularized Mahalanobis), the metric is **refit on the
//! transformed data** and compared against the same metric fitted on the
//! unscaled reference:
//!
//! - `knn` — fraction of points whose 5-nearest-neighbour *set* is preserved;
//! - `cluster` — two-cluster recovery accuracy by nearest-medoid assignment
//!   (medoids fixed at the first generated point of each cluster);
//! - `distortion` — the maximum over point pairs of
//!   `|d_scaled / d_reference − 1|` (0 means the refit geometry is unchanged).
//!
//! Scientific content goes to **stdout** with full `{:.17e}` precision and is
//! byte-for-byte reproducible (`cmp` / SHA-256). Wall-clock timings are
//! environment-dependent and therefore go to **stderr**, never into the hashed
//! artifact.
//!
//! # What this does and does not claim
//!
//! Robust-diagonal geometry is invariant to positive per-coordinate rescaling
//! *because it is refit*; it is not rotation- or affine-invariant. The
//! regularized Mahalanobis baseline uses the classical mean/covariance and a
//! fixed ridge: the ridge deliberately breaks exact scale equivariance at
//! extreme anisotropy, and that degradation is part of the honest output. No
//! metric here is claimed to be robust to adversarial contamination.

use std::time::Instant;

use scirust_multivariate::{
    FittedDistanceMetric, Matrix, RobustScaleMethod, RobustScalerConfig, ZeroScalePolicy,
};
use scirust_stats::{Distribution, Normal, SplitMix64};

/// Points generated per cluster.
const CLUSTER_SIZE: usize = 20;
/// Feature count.
const DIMENSIONS: usize = 4;
/// Cluster-centre separation in normalized units.
const SEPARATION: f64 = 8.0;
/// Seed for the base dataset.
const DATA_SEED: u64 = 0x5EED_0722;
/// Neighbourhood size for the kNN-preservation score.
const KNN: usize = 5;
/// Global scale factors replayed.
const GLOBAL_SCALES: [f64; 4] = [1.0, 1.0e3, 1.0e6, 1.0e9];
/// Per-column anisotropic factors (spec ratios 1, 1e3, 1e6, 1e9).
const ANISOTROPIC_FACTORS: [f64; DIMENSIONS] = [1.0, 1.0e3, 1.0e6, 1.0e9];
/// Epsilon for the relative-norm metric.
const EPSILON: f64 = 1.0e-12;
/// Ridge for the regularized Mahalanobis metric.
const RIDGE: f64 = 1.0e-9;

/// The four benchmarked geometries, refit on `data` where applicable.
fn fit_metrics(data: &Matrix) -> Result<Vec<(&'static str, FittedDistanceMetric)>, String> {
    let scaler_config = RobustScalerConfig {
        center: true,
        scale_method: RobustScaleMethod::MedianAbsoluteDeviation,
        zero_scale_policy: ZeroScalePolicy::Error,
        minimum_scale: 0.0,
    };
    Ok(vec![
        ("raw_euclidean", FittedDistanceMetric::RawEuclidean),
        (
            "relative_norm",
            FittedDistanceMetric::RelativeNorm { epsilon: EPSILON },
        ),
        (
            "robust_diagonal",
            FittedDistanceMetric::fit_robust_diagonal(data, scaler_config)
                .map_err(|e| format!("robust diagonal fit: {e}"))?,
        ),
        (
            "regularized_mahalanobis",
            FittedDistanceMetric::fit_regularized_mahalanobis(data, RIDGE)
                .map_err(|e| format!("mahalanobis fit: {e}"))?,
        ),
    ])
}

/// Build the fixed two-cluster dataset in normalized units.
fn base_dataset() -> Matrix {
    let standard = Normal::standard();
    let mut rng = SplitMix64::new(DATA_SEED);
    let mut draw = |offset: f64| -> Vec<f64> {
        (0..DIMENSIONS)
            .map(|_| {
                let u = 1.0e-6 + rng.next_f64() * (1.0 - 2.0e-6);
                offset + standard.quantile(u)
            })
            .collect()
    };
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(2 * CLUSTER_SIZE);
    for _ in 0..CLUSTER_SIZE
    {
        rows.push(draw(0.0));
    }
    for _ in 0..CLUSTER_SIZE
    {
        rows.push(draw(SEPARATION));
    }
    Matrix {
        rows: rows.len(),
        cols: DIMENSIONS,
        data: rows,
    }
}

/// Apply per-column factors (a global scale is the special case of equal
/// factors).
fn rescale(data: &Matrix, factors: &[f64; DIMENSIONS]) -> Matrix {
    let mut out = data.clone();
    for row in &mut out.data
    {
        for (j, &f) in factors.iter().enumerate()
        {
            row[j] *= f;
        }
    }
    out
}

/// All pairwise distances under `metric` (upper triangle, row-major order).
fn pairwise(metric: &FittedDistanceMetric, data: &Matrix) -> Result<Vec<f64>, String> {
    let n = data.rows;
    let mut out = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n
    {
        for j in (i + 1)..n
        {
            out.push(
                metric
                    .distance(&data.data[i], &data.data[j])
                    .map_err(|e| format!("distance ({i}, {j}): {e}"))?,
            );
        }
    }
    Ok(out)
}

/// Index into the upper-triangle vector produced by [`pairwise`].
fn pair_index(i: usize, j: usize, n: usize) -> usize {
    let (a, b) = if i < j { (i, j) } else { (j, i) };
    a * n - a * (a + 1) / 2 + (b - a - 1)
}

/// The `k` nearest neighbours of `point` as a sorted index set, with
/// deterministic tie-breaking by (distance, index) via `total_cmp`.
fn knn_set(distances: &[f64], point: usize, n: usize, k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).filter(|&j| j != point).collect();
    order.sort_by(|&a, &b| {
        distances[pair_index(point, a, n)]
            .total_cmp(&distances[pair_index(point, b, n)])
            .then(a.cmp(&b))
    });
    let mut set: Vec<usize> = order.into_iter().take(k).collect();
    set.sort_unstable();
    set
}

/// Fraction of points whose `KNN`-nearest-neighbour set matches the reference.
fn knn_preservation(reference: &[f64], scaled: &[f64], n: usize) -> f64 {
    let mut preserved = 0usize;
    for point in 0..n
    {
        if knn_set(reference, point, n, KNN) == knn_set(scaled, point, n, KNN)
        {
            preserved += 1;
        }
    }
    preserved as f64 / n as f64
}

/// Two-cluster recovery accuracy: assign every point to the nearer of the two
/// fixed medoids (first generated point of each cluster; distance ties go to
/// the first medoid) and compare with the generating labels.
fn cluster_recovery(distances: &[f64], n: usize) -> f64 {
    let medoid_a = 0usize;
    let medoid_b = CLUSTER_SIZE;
    let mut correct = 0usize;
    for point in 0..n
    {
        let label = usize::from(point >= CLUSTER_SIZE);
        let assigned = if point == medoid_a
        {
            0
        }
        else if point == medoid_b
        {
            1
        }
        else
        {
            let da = distances[pair_index(point, medoid_a, n)];
            let db = distances[pair_index(point, medoid_b, n)];
            usize::from(db < da)
        };
        if assigned == label
        {
            correct += 1;
        }
    }
    correct as f64 / n as f64
}

/// `max |d_scaled / d_reference − 1|` over pairs with `d_reference > 0`.
fn max_distortion(reference: &[f64], scaled: &[f64]) -> f64 {
    let mut worst = 0.0_f64;
    for (&r, &s) in reference.iter().zip(scaled.iter())
    {
        if r > 0.0
        {
            worst = worst.max((s / r - 1.0).abs());
        }
    }
    worst
}

fn run() -> Result<(), String> {
    let base = base_dataset();
    let n = base.rows;

    println!("# robust_geometry deterministic mechanism benchmark");
    println!(
        "# cluster_size={CLUSTER_SIZE} dimensions={DIMENSIONS} separation={SEPARATION} \
knn={KNN} epsilon={EPSILON} ridge={RIDGE}"
    );
    println!("# columns: transform,metric,knn_preservation,cluster_recovery,max_distortion");

    // Reference pairwise distances: each metric fitted on the unscaled data.
    let reference_metrics = fit_metrics(&base)?;
    let mut reference_distances = Vec::with_capacity(reference_metrics.len());
    for (name, metric) in &reference_metrics
    {
        reference_distances.push((*name, pairwise(metric, &base)?));
    }

    let mut transforms: Vec<(String, Matrix)> = Vec::new();
    for &lambda in &GLOBAL_SCALES
    {
        transforms.push((
            format!("global_{lambda:.0e}"),
            rescale(&base, &[lambda; DIMENSIONS]),
        ));
    }
    transforms.push((
        "anisotropic_1_1e3_1e6_1e9".to_string(),
        rescale(&base, &ANISOTROPIC_FACTORS),
    ));

    for (transform_name, transformed) in &transforms
    {
        let started = Instant::now();
        let metrics = fit_metrics(transformed)?;
        for ((name, metric), (reference_name, reference)) in
            metrics.iter().zip(reference_distances.iter())
        {
            assert_eq!(name, reference_name);
            let scaled = pairwise(metric, transformed)?;
            let knn = knn_preservation(reference, &scaled, n);
            let cluster = cluster_recovery(&scaled, n);
            let distortion = max_distortion(reference, &scaled);
            println!("{transform_name},{name},{knn:.17e},{cluster:.17e},{distortion:.17e}");
        }
        // Wall-clock is environment-dependent: stderr only, never hashed.
        eprintln!(
            "# timing {transform_name}: {:.3} ms",
            started.elapsed().as_secs_f64() * 1.0e3
        );
    }

    Ok(())
}

fn main() {
    if let Err(message) = run()
    {
        eprintln!("robust_geometry benchmark failed: {message}");
        std::process::exit(1);
    }
}
