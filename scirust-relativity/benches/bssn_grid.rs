//! Benchmarks for the periodic one-dimensional BSSN grid (Layer 3.4).
//!
//! Only numerical work is timed. CSV serialisation is deliberately excluded:
//! it is I/O formatting, not part of the numerical kernel.
//!
//! Wall-clock figures are machine-dependent; the computation itself is
//! deterministic.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use scirust_relativity::adm_evolution::{AdmEvolutionSettings, AdmSources, SpatialTensorField};
use scirust_relativity::bssn::{BssnGauge, adm_to_bssn, conformal_ricci};
use scirust_relativity::bssn_grid::{
    BssnGridState, BssnGridSystem, COMPONENTS_PER_POINT, TransverseTracelessWave,
    bssn_grid_constraints, bssn_grid_rhs, evolve_bssn_grid,
};
use scirust_relativity::grid::{
    UniformGrid1d, UniformGrid2d, UniformGrid3d, periodic_first_derivative_all,
    periodic_mixed_derivative, periodic_second_derivative_all,
};

const RESOLUTIONS: [usize; 3] = [32, 64, 128];
const TWO_PI: f64 = std::f64::consts::TAU;

fn grid_of(points: usize) -> UniformGrid1d {
    UniformGrid1d::new(points, 0.0, 1.0).expect("valid grid")
}

fn wave_state(grid: UniformGrid1d) -> BssnGridState {
    let wave = TransverseTracelessWave::new(1.0e-4, TWO_PI, 0.0);
    BssnGridState::from_adm_fields(grid, &wave.metric_field(), &wave.curvature_field())
        .expect("wave initial data")
}

fn zero_curvature() -> impl SpatialTensorField {
    |_: &[f64; 3]| [[0.0_f64; 3]; 3]
}

/// Periodic centred first derivative over a whole component array.
fn bench_first_derivative(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_periodic_first_derivative");
    for &points in &RESOLUTIONS
    {
        let grid = grid_of(points);
        let samples: Vec<f64> = (0..points)
            .map(|i| (TWO_PI * grid.coordinate(i)).sin())
            .collect();
        let mut out = vec![0.0_f64; points];
        group.bench_with_input(BenchmarkId::from_parameter(points), &points, |b, _| {
            b.iter(|| {
                periodic_first_derivative_all(black_box(&samples), black_box(&grid), &mut out)
                    .expect("derivative")
            });
        });
    }
    group.finish();
}

/// Periodic centred second derivative over a whole component array.
fn bench_second_derivative(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_periodic_second_derivative");
    for &points in &RESOLUTIONS
    {
        let grid = grid_of(points);
        let samples: Vec<f64> = (0..points)
            .map(|i| (TWO_PI * grid.coordinate(i)).sin())
            .collect();
        let mut out = vec![0.0_f64; points];
        group.bench_with_input(BenchmarkId::from_parameter(points), &points, |b, _| {
            b.iter(|| {
                periodic_second_derivative_all(black_box(&samples), black_box(&grid), &mut out)
                    .expect("derivative")
            });
        });
    }
    group.finish();
}

/// Conformal connection reconstruction at one point (`adm_to_bssn` seeds
/// `Gammatilde^i` from the metric, which is the reconstruction being timed).
fn bench_conformal_connection(c: &mut Criterion) {
    let grid = grid_of(64);
    let settings = AdmEvolutionSettings {
        spatial_step: grid.spacing(),
        metric_step: grid.spacing(),
    };
    let state = wave_state(grid);
    let view = state.view();
    let metric = view.metric();
    let curvature = view.curvature();
    let at = [grid.coordinate(7), 0.0, 0.0];
    c.bench_function("grid_conformal_connection_point", |b| {
        b.iter(|| {
            adm_to_bssn(
                black_box(&metric),
                black_box(&curvature),
                black_box(&at),
                black_box(&settings),
            )
            .expect("conversion")
        });
    });
}

/// Conformal Ricci evaluation at one grid point.
fn bench_conformal_ricci(c: &mut Criterion) {
    let grid = grid_of(64);
    let settings = AdmEvolutionSettings {
        spatial_step: grid.spacing(),
        metric_step: grid.spacing(),
    };
    let state = wave_state(grid);
    let view = state.view();
    let metric = view.metric();
    let at = [grid.coordinate(7), 0.0, 0.0];
    c.bench_function("grid_conformal_ricci_point", |b| {
        b.iter(|| {
            conformal_ricci(black_box(&metric), black_box(&at), black_box(&settings))
                .expect("ricci")
        });
    });
}

/// One complete BSSN right-hand side over the whole grid.
///
/// This is the dominant cost of a timestep: RK4 calls it four times.
fn bench_grid_rhs(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_bssn_rhs");
    for &points in &RESOLUTIONS
    {
        let grid = grid_of(points);
        let state = wave_state(grid);
        let view = state.view();
        let mut out = vec![0.0_f64; COMPONENTS_PER_POINT * points];
        group.bench_with_input(BenchmarkId::from_parameter(points), &points, |b, _| {
            b.iter(|| {
                bssn_grid_rhs(
                    black_box(&view),
                    black_box(&BssnGauge::SYNCHRONOUS),
                    black_box(&AdmSources::VACUUM),
                    0.0,
                    &mut out,
                )
                .expect("rhs")
            });
        });
    }
    group.finish();
}

/// One complete constraint-monitoring pass over the grid.
fn bench_constraint_monitor(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_constraint_monitor");
    for &points in &RESOLUTIONS
    {
        let grid = grid_of(points);
        let state = wave_state(grid);
        let view = state.view();
        group.bench_with_input(BenchmarkId::from_parameter(points), &points, |b, _| {
            b.iter(|| {
                bssn_grid_constraints(black_box(&view), black_box(&AdmSources::VACUUM))
                    .expect("constraints")
            });
        });
    }
    group.finish();
}

/// A short RK4 trajectory, so the per-step cost can be read off directly.
fn bench_rk4_trajectory(c: &mut Criterion) {
    let grid = grid_of(32);
    let system = BssnGridSystem::vacuum(grid);
    let initial = wave_state(grid);
    let step = 0.25 * grid.spacing();
    c.bench_function("grid_rk4_ten_steps_n32", |b| {
        b.iter(|| {
            evolve_bssn_grid(
                black_box(&system),
                black_box(&initial),
                0.0,
                10.0 * step,
                step,
            )
            .expect("evolution")
        });
    });
}

/// Flat-state encode and decode.
///
/// Storage *is* the flat layout, so decoding is a borrow and encoding is a
/// contiguous copy; this measures how little that costs relative to the
/// right-hand side.
fn bench_flat_state_round_trip(c: &mut Criterion) {
    let grid = grid_of(128);
    let state = wave_state(grid);
    c.bench_function("grid_flat_state_round_trip_n128", |b| {
        b.iter(|| {
            let flat = black_box(&state).as_slice().to_vec();
            BssnGridState::from_flat(black_box(grid), flat).expect("round trip")
        });
    });
}

/// Building initial data from analytic ADM fields through the production path.
fn bench_initial_data(c: &mut Criterion) {
    let grid = grid_of(64);
    let manufactured = ManufacturedMetric {
        amplitude: 0.01,
        wave_number: TWO_PI,
    };
    c.bench_function("grid_initial_data_from_adm_n64", |b| {
        b.iter(|| {
            BssnGridState::from_adm_fields(
                black_box(grid),
                black_box(&manufactured),
                &zero_curvature(),
            )
            .expect("initial data")
        });
    });
}

struct ManufacturedMetric {
    amplitude: f64,
    wave_number: f64,
}

impl scirust_relativity::Metric<3> for ManufacturedMetric {
    fn components(&self, coordinates: &[f64; 3]) -> [[f64; 3]; 3] {
        let s = (self.wave_number * coordinates[0]).sin();
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0 + self.amplitude * s, 0.0],
            [0.0, 0.0, 1.0 - self.amplitude * s],
        ]
    }
}

/// The mixed second difference — the stencil with no one-dimensional analogue.
fn bench_mixed_derivative(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_periodic_mixed_derivative_2d");
    for &points in &[16_usize, 32, 64]
    {
        let grid =
            UniformGrid2d::from_axes([points, points], [0.0, 0.0], [1.0, 1.0]).expect("grid");
        let samples: Vec<f64> = (0..grid.total_points())
            .map(|linear| {
                let p = grid.position(linear);
                (TWO_PI * p[0]).sin() * (TWO_PI * p[1]).sin()
            })
            .collect();
        group.bench_with_input(BenchmarkId::from_parameter(points), &points, |b, _| {
            b.iter(|| {
                let mut total = 0.0_f64;
                for linear in 0..grid.total_points()
                {
                    total += periodic_mixed_derivative(
                        black_box(&samples),
                        black_box(&grid),
                        linear,
                        0,
                        1,
                    )
                    .expect("mixed");
                }
                total
            });
        });
    }
    group.finish();
}

/// One complete BSSN right-hand side over a two-dimensional grid.
///
/// The point count grows as `N^2`, so this is where the cost of a second
/// dimension actually shows up.
fn bench_grid_rhs_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_bssn_rhs_2d");
    for &points in &[8_usize, 16, 32]
    {
        let grid =
            UniformGrid2d::from_axes([points, points], [0.0, 0.0], [1.0, 1.0]).expect("grid");
        let state = BssnGridState::minkowski(grid);
        let view = state.view();
        let mut out = vec![0.0_f64; COMPONENTS_PER_POINT * grid.total_points()];
        group.bench_with_input(BenchmarkId::from_parameter(points), &points, |b, _| {
            b.iter(|| {
                bssn_grid_rhs(
                    black_box(&view),
                    black_box(&BssnGauge::SYNCHRONOUS),
                    black_box(&AdmSources::VACUUM),
                    0.0,
                    &mut out,
                )
                .expect("rhs")
            });
        });
    }
    group.finish();
}

/// One complete BSSN right-hand side over a three-dimensional grid.
///
/// The point count grows as `N^3` and each point now takes nine mixed second
/// differences rather than four, so this separates the cost of the extra points
/// from the cost of the extra stencils.
fn bench_grid_rhs_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_bssn_rhs_3d");
    for &points in &[4_usize, 8, 16]
    {
        let grid = UniformGrid3d::from_axes([points; 3], [0.0; 3], [1.0; 3]).expect("grid");
        let state = BssnGridState::minkowski(grid);
        let view = state.view();
        let mut out = vec![0.0_f64; COMPONENTS_PER_POINT * grid.total_points()];
        group.bench_with_input(BenchmarkId::from_parameter(points), &points, |b, _| {
            b.iter(|| {
                bssn_grid_rhs(
                    black_box(&view),
                    black_box(&BssnGauge::SYNCHRONOUS),
                    black_box(&AdmSources::VACUUM),
                    0.0,
                    &mut out,
                )
                .expect("rhs")
            });
        });
    }
    group.finish();
}

/// Initial data for the body-diagonal gravitational wave.
///
/// The polarization has every component non-zero, so the conformal
/// decomposition does strictly more work than for the along-`x` wave.
fn bench_diagonal_wave_initial_data(c: &mut Criterion) {
    let grid = UniformGrid3d::from_axes([8; 3], [0.0; 3], [1.0; 3]).expect("grid");
    let wave = TransverseTracelessWave::diagonal(1.0e-6, TWO_PI, 0.0);
    c.bench_function("grid_diagonal_wave_initial_data_n8", |b| {
        b.iter(|| {
            BssnGridState::from_adm_fields(
                black_box(grid),
                &wave.metric_field(),
                &wave.curvature_field(),
            )
            .expect("initial data")
        });
    });
}

criterion_group!(
    benches,
    bench_mixed_derivative,
    bench_grid_rhs_2d,
    bench_grid_rhs_3d,
    bench_diagonal_wave_initial_data,
    bench_first_derivative,
    bench_second_derivative,
    bench_conformal_connection,
    bench_conformal_ricci,
    bench_grid_rhs,
    bench_constraint_monitor,
    bench_rk4_trajectory,
    bench_flat_state_round_trip,
    bench_initial_data,
);
criterion_main!(benches);
