//! Wall-clock micro-benchmarks for the geometry-core hot paths.
//!
//! Unlike the deterministic experiments (byte-identical output) and the
//! operation-count proxy in the worldline crate's `complexity_scaling`, these
//! measure real elapsed time and are therefore machine-dependent and **not**
//! bit-reproducible — that is inherent to timing. The library functions they
//! call remain fully deterministic; only the measured durations vary. Run with
//! `cargo bench -p scirust-relativity`.
//!
//! The cases span the analytic hot paths (`christoffel`, `invert_metric`), the
//! finite-difference curvature engine, the RK4 transport integrator, and the
//! Newton-shooting world-function / van Vleck primitives, so a change in any of
//! them shows up here.

use std::f64::consts::FRAC_PI_2;
use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use scirust_relativity::{
    Connection, CurvatureTensors, Metric, Schwarzschild, WorldFunctionSettings,
    geodesic_exponential, invert_metric, transport_along_segment, van_vleck_determinant,
    world_function,
};

fn geometry_core(criterion: &mut Criterion) {
    let schwarzschild = Schwarzschild::try_new(1.0).expect("valid Schwarzschild mass");
    let point = [0.0, 10.0, FRAC_PI_2, 0.0];
    let metric = schwarzschild.components(&point);
    let settings = WorldFunctionSettings::default();
    let base = [0.0, 12.0, FRAC_PI_2, 0.0];
    let field = [0.15, 12.4, FRAC_PI_2 + 0.03, 0.05];

    criterion.bench_function("christoffel", |bencher| {
        bencher.iter(|| black_box(schwarzschild.christoffel(black_box(&point))));
    });

    criterion.bench_function("invert_metric", |bencher| {
        bencher.iter(|| black_box(invert_metric(black_box(&metric)).expect("invertible metric")));
    });

    criterion.bench_function("curvature_tensors", |bencher| {
        bencher.iter(|| {
            black_box(
                CurvatureTensors::compute(&schwarzschild, black_box(&point), 1.0e-4)
                    .expect("finite curvature"),
            )
        });
    });

    criterion.bench_function("parallel_transport_segment_100", |bencher| {
        let start = [0.0, 10.0, FRAC_PI_2, 0.0];
        let end = [0.0, 8.0, FRAC_PI_2, 0.5];
        let vector = [1.0, 0.0, 0.1, 0.0];
        bencher.iter(|| {
            black_box(
                transport_along_segment(&schwarzschild, &start, &end, black_box(&vector), 100)
                    .expect("finite transport"),
            )
        });
    });

    criterion.bench_function("geodesic_exponential", |bencher| {
        let velocity = [0.2, 0.15, 0.03, 0.02];
        bencher.iter(|| {
            black_box(
                geodesic_exponential(&schwarzschild, &base, black_box(&velocity), settings.step)
                    .expect("finite exponential map"),
            )
        });
    });

    criterion.bench_function("world_function", |bencher| {
        bencher.iter(|| {
            black_box(
                world_function(&schwarzschild, &base, &field, &settings).expect("finite sigma"),
            )
        });
    });

    criterion.bench_function("van_vleck_determinant", |bencher| {
        bencher.iter(|| {
            black_box(
                van_vleck_determinant(&schwarzschild, &base, &field, &settings)
                    .expect("finite van Vleck determinant"),
            )
        });
    });
}

criterion_group!(benches, geometry_core);
criterion_main!(benches);
