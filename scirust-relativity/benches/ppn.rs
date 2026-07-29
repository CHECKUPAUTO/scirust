//! Wall-clock micro-benchmarks for PPN (gamma, beta) extraction.
//!
//! These measure elapsed time and are therefore machine-dependent and not
//! bit-reproducible — that is inherent to timing; the extractor itself is fully
//! deterministic. They cover metric sampling, the effective-estimator
//! computation, the extrapolation solve, a complete extraction, and its scaling
//! with sample count and polynomial order. Run with
//! `cargo bench -p scirust-relativity --bench ppn`.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use scirust_relativity::IsotropicSchwarzschild;
use scirust_relativity::ppn::{
    IsotropicChartAdapter, PpnDomain, StaticIsotropicMetric, fit_polynomial_intercept,
};

const MASS: f64 = 1.0;

fn radii(sample_count: usize) -> Vec<f64> {
    let (compactness_min, compactness_max) = (0.005, 0.05);
    let last = (sample_count - 1) as f64;
    (0..sample_count)
        .map(|index| {
            let fraction = index as f64 / last;
            let compactness = compactness_min + fraction * (compactness_max - compactness_min);
            MASS / compactness
        })
        .collect()
}

fn ppn(criterion: &mut Criterion) {
    let isotropic = IsotropicSchwarzschild::try_new(MASS).expect("valid mass");
    let adapter = IsotropicChartAdapter::new(&isotropic, MASS).expect("valid adapter");
    let domain = PpnDomain::uniform_compactness(0.005, 0.05, 24);

    // Metric sampling: g_tt and the spatial conformal factor over the domain.
    criterion.bench_function("ppn_metric_sampling_24", |bencher| {
        let sample_radii = radii(24);
        bencher.iter(|| {
            let mut accumulator = 0.0;
            for &radius in &sample_radii
            {
                accumulator += adapter.g_tt(black_box(radius)).expect("g_tt");
                accumulator += adapter
                    .spatial_conformal_factor(black_box(radius))
                    .expect("conformal");
            }
            black_box(accumulator)
        });
    });

    // Effective-estimator computation (sampling + the gamma/beta arithmetic).
    criterion.bench_function("ppn_effective_estimators_24", |bencher| {
        let sample_radii = radii(24);
        bencher.iter(|| {
            let mut accumulator = 0.0;
            for &radius in &sample_radii
            {
                let u = MASS / radius;
                let g_tt = adapter.g_tt(radius).expect("g_tt");
                let conformal = adapter.spatial_conformal_factor(radius).expect("conformal");
                accumulator += (conformal - 1.0) / (2.0 * u);
                accumulator += -(g_tt + 1.0 - 2.0 * u) / (2.0 * u * u);
            }
            black_box(accumulator)
        });
    });

    // The extrapolation solve alone, on fixed synthetic data.
    criterion.bench_function("ppn_extrapolation_solve_deg3_24", |bencher| {
        let xs: Vec<f64> = (0..24).map(|i| 0.005 + 0.045 * i as f64 / 23.0).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 1.0 + 0.75 * x + 0.25 * x * x).collect();
        bencher.iter(|| black_box(fit_polynomial_intercept(black_box(&xs), black_box(&ys), 3)));
    });

    // Diagnostic calculation alone: the four extra fits a full extraction takes
    // per parameter beyond the primary one (window-halved, the two adjacent
    // orders, and resolution-halved) -- the marginal cost the hardened
    // diagnostics add over a bare `fit_polynomial_intercept` call. Uses only the
    // public solver, mirroring `extract_ppn`'s internal sensitivity probes.
    criterion.bench_function("ppn_diagnostic_probes_24_deg3", |bencher| {
        let sample_radii = radii(24);
        let compactness: Vec<f64> = sample_radii.iter().map(|&r| MASS / r).collect();
        let gamma_effective: Vec<f64> = sample_radii
            .iter()
            .map(|&radius| {
                let u = MASS / radius;
                (adapter.spatial_conformal_factor(radius).expect("conformal") - 1.0) / (2.0 * u)
            })
            .collect();
        let half = compactness.len().div_ceil(2);
        let window_u = &compactness[..half];
        let window_y = &gamma_effective[..half];
        let coarse_u: Vec<f64> = compactness.iter().step_by(2).copied().collect();
        let coarse_y: Vec<f64> = gamma_effective.iter().step_by(2).copied().collect();
        bencher.iter(|| {
            black_box(fit_polynomial_intercept(
                black_box(window_u),
                black_box(window_y),
                3,
            ))
            .expect("window probe");
            black_box(fit_polynomial_intercept(&compactness, &gamma_effective, 2))
                .expect("order-1 probe");
            black_box(fit_polynomial_intercept(&compactness, &gamma_effective, 4))
                .expect("order+1 probe");
            black_box(fit_polynomial_intercept(
                black_box(&coarse_u),
                black_box(&coarse_y),
                3,
            ))
            .expect("resolution probe")
        });
    });

    // A complete extraction.
    criterion.bench_function("ppn_full_extraction_24_deg3", |bencher| {
        bencher.iter(|| {
            black_box(
                scirust_relativity::ppn::extract_ppn(&adapter, black_box(&domain), 3)
                    .expect("extraction"),
            )
        });
    });

    // Scaling with sample count.
    let mut by_samples = criterion.benchmark_group("ppn_extraction_by_sample_count");
    for &sample_count in &[12_usize, 24, 48]
    {
        let domain = PpnDomain::uniform_compactness(0.005, 0.05, sample_count);
        by_samples.bench_with_input(
            BenchmarkId::from_parameter(sample_count),
            &domain,
            |bencher, domain| {
                bencher.iter(|| {
                    black_box(
                        scirust_relativity::ppn::extract_ppn(&adapter, domain, 3)
                            .expect("extraction"),
                    )
                });
            },
        );
    }
    by_samples.finish();

    // Scaling with polynomial order.
    let mut by_order = criterion.benchmark_group("ppn_extraction_by_order");
    for &order in &[1_usize, 2, 3, 4]
    {
        by_order.bench_with_input(
            BenchmarkId::from_parameter(order),
            &order,
            |bencher, &order| {
                bencher.iter(|| {
                    black_box(
                        scirust_relativity::ppn::extract_ppn(&adapter, &domain, order)
                            .expect("extraction"),
                    )
                });
            },
        );
    }
    by_order.finish();
}

criterion_group!(benches, ppn);
criterion_main!(benches);
