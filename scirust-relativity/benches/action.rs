//! Wall-clock micro-benchmarks for the Einstein-Hilbert action variation.
//!
//! These measure elapsed time and are therefore machine-dependent and not
//! bit-reproducible — that is inherent to timing; the variation itself is fully
//! deterministic. They cover the metric-only Ricci scalar (the enabling nested
//! finite difference) and a complete variation across grid resolutions. Run with
//! `cargo bench -p scirust-relativity --bench action`.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use scirust_relativity::action::{
    ActionDomain, ActionPerturbation, ActionSettings, einstein_hilbert_action_variation,
};
use scirust_relativity::{DeSitter, ricci_scalar_from_metric};
use std::f64::consts::FRAC_PI_2;

const LAMBDA: f64 = 0.03;

fn perturbation() -> ActionPerturbation {
    ActionPerturbation {
        component: (1, 1),
        center: (3.0, FRAC_PI_2),
        half_widths: (1.0, 1.0),
    }
}

fn settings() -> ActionSettings {
    ActionSettings {
        amplitude: 1.0e-3,
        connection_step: 1.0e-3,
        metric_step: 1.0e-3,
        cosmological_constant: LAMBDA,
    }
}

fn action(criterion: &mut Criterion) {
    let de_sitter = DeSitter::try_new(LAMBDA).expect("valid de Sitter");
    let perturbation = perturbation();
    let settings = settings();

    // The metric-only nested-difference Ricci scalar at one point.
    criterion.bench_function("action_ricci_scalar_from_metric", |bencher| {
        bencher.iter(|| {
            black_box(
                ricci_scalar_from_metric(
                    &de_sitter,
                    black_box(&[0.0, 3.0, FRAC_PI_2, 0.0]),
                    1.0e-3,
                    1.0e-3,
                )
                .expect("scalar"),
            )
        });
    });

    // A complete variation at a moderate grid.
    criterion.bench_function("action_variation_de_sitter_31", |bencher| {
        let domain = ActionDomain {
            radial_range: (2.0, 4.0),
            polar_range: (FRAC_PI_2 - 1.0, FRAC_PI_2 + 1.0),
            grid: 31,
        };
        bencher.iter(|| {
            black_box(
                einstein_hilbert_action_variation(
                    &de_sitter,
                    black_box(&perturbation),
                    black_box(&domain),
                    black_box(&settings),
                )
                .expect("variation"),
            )
        });
    });

    // Scaling with grid resolution.
    let mut by_grid = criterion.benchmark_group("action_variation_by_grid");
    for &grid in &[21_usize, 31, 41]
    {
        let domain = ActionDomain {
            radial_range: (2.0, 4.0),
            polar_range: (FRAC_PI_2 - 1.0, FRAC_PI_2 + 1.0),
            grid,
        };
        by_grid.bench_with_input(
            BenchmarkId::from_parameter(grid),
            &domain,
            |bencher, domain| {
                bencher.iter(|| {
                    black_box(
                        einstein_hilbert_action_variation(
                            &de_sitter,
                            &perturbation,
                            domain,
                            &settings,
                        )
                        .expect("variation"),
                    )
                });
            },
        );
    }
    by_grid.finish();
}

criterion_group!(benches, action);
criterion_main!(benches);
