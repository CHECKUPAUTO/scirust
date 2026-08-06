//! Wall-clock micro-benchmarks for the ADM constraint and evolution core
//! (Layer 3.1).
//!
//! These measure elapsed time and are therefore machine-dependent and not
//! bit-reproducible — that is inherent to timing; the evaluators themselves
//! are fully deterministic. They cover the Hamiltonian constraint, the
//! momentum constraint (the heaviest quantity: a spatial divergence of a
//! field that itself needs the inverse metric at each neighbor), the metric
//! evolution right-hand side, and the full extrinsic-curvature evolution
//! right-hand side (which additionally needs the spatial Ricci tensor and the
//! lapse Hessian). Run with `cargo bench -p scirust-relativity --bench adm_evolution`.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use scirust_relativity::Metric;
use scirust_relativity::adm_evolution::{
    AdmEvolutionSettings, AdmSources, SpatialScalarField, SpatialTensorField, SpatialVectorField,
    curvature_evolution_rhs, hamiltonian_constraint, metric_evolution_rhs, momentum_constraint,
};

struct FlatSpace;
impl Metric<3> for FlatSpace {
    fn components(&self, _coordinates: &[f64; 3]) -> [[f64; 3]; 3] {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }
}

struct LinearTracelessCurvature {
    epsilon: f64,
}
impl SpatialTensorField for LinearTracelessCurvature {
    fn components(&self, coordinates: &[f64; 3]) -> [[f64; 3]; 3] {
        let k11 = self.epsilon * coordinates[0];
        [
            [k11, 0.0, 0.0],
            [0.0, -0.5 * k11, 0.0],
            [0.0, 0.0, -0.5 * k11],
        ]
    }
}

struct QuadraticLapse {
    coefficient: f64,
}
impl SpatialScalarField for QuadraticLapse {
    fn value(&self, coordinates: &[f64; 3]) -> f64 {
        let [x, y, z] = *coordinates;
        1.0 + self.coefficient * (x * x + y * y + z * z)
    }
}

struct LinearShift {
    matrix: [[f64; 3]; 3],
}
impl SpatialVectorField for LinearShift {
    #[allow(clippy::needless_range_loop)]
    fn components(&self, coordinates: &[f64; 3]) -> [f64; 3] {
        let mut value = [0.0_f64; 3];
        for i in 0..3
        {
            for j in 0..3
            {
                value[i] += self.matrix[i][j] * coordinates[j];
            }
        }
        value
    }
}

fn adm_evolution(criterion: &mut Criterion) {
    let curvature = LinearTracelessCurvature { epsilon: 0.05 };
    let lapse = QuadraticLapse { coefficient: 0.1 };
    let mut matrix = [[0.0_f64; 3]; 3];
    matrix[0][1] = 0.2;
    let shift = LinearShift { matrix };
    let point = [1.0, 2.0, 3.0];
    let settings = AdmEvolutionSettings {
        spatial_step: 1.0e-3,
        metric_step: 1.0e-3,
    };
    let sources = AdmSources::VACUUM;

    criterion.bench_function("adm_evolution_hamiltonian_constraint", |bencher| {
        bencher.iter(|| {
            black_box(
                hamiltonian_constraint(
                    &FlatSpace,
                    &curvature,
                    black_box(&point),
                    &sources,
                    &settings,
                )
                .expect("constraint"),
            )
        });
    });

    criterion.bench_function("adm_evolution_momentum_constraint", |bencher| {
        bencher.iter(|| {
            black_box(
                momentum_constraint(
                    &FlatSpace,
                    &curvature,
                    black_box(&point),
                    &sources,
                    &settings,
                )
                .expect("constraint"),
            )
        });
    });

    criterion.bench_function("adm_evolution_metric_rhs", |bencher| {
        bencher.iter(|| {
            black_box(
                metric_evolution_rhs(
                    &FlatSpace,
                    &curvature,
                    &lapse,
                    &shift,
                    black_box(&point),
                    &settings,
                )
                .expect("rhs"),
            )
        });
    });

    criterion.bench_function("adm_evolution_curvature_rhs", |bencher| {
        bencher.iter(|| {
            black_box(
                curvature_evolution_rhs(
                    &FlatSpace,
                    &curvature,
                    &lapse,
                    &shift,
                    black_box(&point),
                    &sources,
                    &settings,
                )
                .expect("rhs"),
            )
        });
    });
}

criterion_group!(benches, adm_evolution);
criterion_main!(benches);
