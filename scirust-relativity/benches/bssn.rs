//! Wall-clock micro-benchmarks for the BSSN formulation core (Layer 3.3).
//!
//! These measure elapsed time and are therefore machine-dependent and not
//! bit-reproducible — that is inherent to timing; the operations themselves are
//! fully deterministic. They cover the ADM-to-BSSN conversion, the inverse
//! reconstruction, the algebraic-constraint evaluation, the conformal Ricci
//! decomposition, and the full BSSN right-hand side, and compare the BSSN
//! right-hand-side cost against the corresponding ADM one. CSV serialization is
//! deliberately excluded. Run with
//! `cargo bench -p scirust-relativity --bench bssn`.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use scirust_relativity::Metric;
use scirust_relativity::adm_evolution::{
    AdmEvolutionSettings, AdmSources, SpatialScalarField, SpatialTensorField, SpatialVectorField,
    curvature_evolution_rhs,
};
use scirust_relativity::bssn::{
    BssnGauge, adm_to_bssn, bssn_algebraic_constraints, bssn_evolution_rhs, bssn_to_adm,
    conformal_ricci,
};

const ORIGIN: [f64; 3] = [0.0, 0.0, 0.0];

fn settings() -> AdmEvolutionSettings {
    AdmEvolutionSettings {
        spatial_step: 1.0e-3,
        metric_step: 1.0e-3,
    }
}

struct ConstantMetric([[f64; 3]; 3]);
impl Metric<3> for ConstantMetric {
    fn components(&self, _x: &[f64; 3]) -> [[f64; 3]; 3] {
        self.0
    }
}
struct ConstantTensor([[f64; 3]; 3]);
impl SpatialTensorField for ConstantTensor {
    fn components(&self, _x: &[f64; 3]) -> [[f64; 3]; 3] {
        self.0
    }
}
struct UnitLapse;
impl SpatialScalarField for UnitLapse {
    fn value(&self, _x: &[f64; 3]) -> f64 {
        1.0
    }
}
struct ZeroShift;
impl SpatialVectorField for ZeroShift {
    fn components(&self, _x: &[f64; 3]) -> [f64; 3] {
        [0.0; 3]
    }
}

fn bssn(criterion: &mut Criterion) {
    let metric = [[1.3, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 1.1]];
    let curvature = [[-0.21, 0.0, 0.0], [0.0, -0.09, 0.0], [0.0, 0.0, -0.15]];
    let sources = AdmSources::perfect_fluid(0.02, 0.007, &metric);
    let settings = settings();

    criterion.bench_function("bssn_adm_to_bssn", |bencher| {
        bencher.iter(|| {
            black_box(
                adm_to_bssn(
                    &ConstantMetric(metric),
                    &ConstantTensor(curvature),
                    black_box(&ORIGIN),
                    &settings,
                )
                .expect("converts"),
            )
        });
    });

    let state = adm_to_bssn(
        &ConstantMetric(metric),
        &ConstantTensor(curvature),
        &ORIGIN,
        &settings,
    )
    .expect("converts");

    criterion.bench_function("bssn_to_adm", |bencher| {
        bencher.iter(|| black_box(bssn_to_adm(black_box(&state)).expect("reconstructs")));
    });

    let reconstructed_connection = state.conformal_connection;
    criterion.bench_function("bssn_algebraic_constraints", |bencher| {
        bencher.iter(|| {
            black_box(bssn_algebraic_constraints(
                black_box(&state),
                black_box(&reconstructed_connection),
            ))
        });
    });

    criterion.bench_function("bssn_conformal_ricci", |bencher| {
        bencher.iter(|| {
            black_box(
                conformal_ricci(&ConstantMetric(metric), black_box(&ORIGIN), &settings)
                    .expect("decomposes"),
            )
        });
    });

    criterion.bench_function("bssn_evolution_rhs", |bencher| {
        bencher.iter(|| {
            black_box(
                bssn_evolution_rhs(
                    &ConstantMetric(metric),
                    &ConstantTensor(curvature),
                    black_box(&ORIGIN),
                    &BssnGauge::SYNCHRONOUS,
                    &sources,
                    &settings,
                )
                .expect("rhs"),
            )
        });
    });

    // The corresponding ADM right-hand side, for a like-for-like cost comparison.
    criterion.bench_function("bssn_reference_adm_curvature_rhs", |bencher| {
        bencher.iter(|| {
            black_box(
                curvature_evolution_rhs(
                    &ConstantMetric(metric),
                    &ConstantTensor(curvature),
                    &UnitLapse,
                    &ZeroShift,
                    black_box(&ORIGIN),
                    &sources,
                    &settings,
                )
                .expect("rhs"),
            )
        });
    });
}

criterion_group!(benches, bssn);
criterion_main!(benches);
