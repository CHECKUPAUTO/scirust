//! Wall-clock micro-benchmarks for homogeneous ADM time evolution (Layer 3.2).
//!
//! These measure elapsed time and are therefore machine-dependent and not
//! bit-reproducible — that is inherent to timing; the evolution itself is fully
//! deterministic. They separate the per-step cost (one right-hand-side
//! evaluation and one constraint evaluation) from a complete integration, and
//! show how the integration scales with the number of steps. The CSV-writing
//! experiment is deliberately **not** benchmarked here: only the numerical work
//! is. Run with `cargo bench -p scirust-relativity --bench adm_homogeneous`.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use scirust_relativity::adm_evolution::AdmEvolutionSettings;
use scirust_relativity::adm_homogeneous::{
    BarotropicFluid, HomogeneousEvolution, HomogeneousState, evolve_homogeneous,
};
use scirust_sim::System;

fn settings() -> AdmEvolutionSettings {
    AdmEvolutionSettings {
        spatial_step: 1.0e-3,
        metric_step: 1.0e-3,
    }
}

fn adm_homogeneous(criterion: &mut Criterion) {
    let evolution = HomogeneousEvolution::new(BarotropicFluid::dust(), settings());
    let initial = HomogeneousState::friedmann(1.0, 0.5).expect("valid initial data");

    // One right-hand-side evaluation: the cost the integrator pays per RK4 stage.
    criterion.bench_function("adm_homogeneous_rhs_evaluation", |bencher| {
        let mut state = vec![0.0_f64; evolution.dim()];
        // Flatten through the public evolution path so the benchmark measures
        // the same code the integrator drives.
        let history = evolve_homogeneous(&evolution, &initial, 0.0, 0.01, 0.01).expect("evolve");
        let sample = history.first().expect("non-empty");
        for i in 0..3
        {
            state[3 * i + i] = sample.scale_factor * sample.scale_factor;
            state[9 + 3 * i + i] =
                -sample.hubble_parameter * sample.scale_factor * sample.scale_factor;
        }
        state[18] = sample.energy_density;
        let mut derivative = vec![0.0_f64; evolution.dim()];
        bencher.iter(|| {
            evolution.derivatives(0.0, black_box(&state), black_box(&mut derivative));
            black_box(derivative[0])
        });
    });

    // One constraint evaluation: the per-sample monitoring cost.
    criterion.bench_function("adm_homogeneous_constraint_monitor", |bencher| {
        bencher.iter(|| {
            black_box(
                evolution
                    .hamiltonian_residual(black_box(&initial))
                    .expect("residual"),
            )
        });
    });

    // A complete integration, and its scaling with the number of steps.
    let mut by_steps = criterion.benchmark_group("adm_homogeneous_evolution_by_steps");
    for &steps in &[50_usize, 100, 200]
    {
        let step = 1.0 / steps as f64;
        by_steps.bench_with_input(
            BenchmarkId::from_parameter(steps),
            &step,
            |bencher, &step| {
                bencher.iter(|| {
                    black_box(
                        evolve_homogeneous(&evolution, &initial, 1.0, 2.0, step).expect("evolves"),
                    )
                });
            },
        );
    }
    by_steps.finish();
}

criterion_group!(benches, adm_homogeneous);
criterion_main!(benches);
