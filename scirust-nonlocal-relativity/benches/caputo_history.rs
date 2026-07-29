//! Wall-clock benchmark for the complete-history Caputo memory pipeline.
//!
//! The complete uniform history retains every past sample, so each accepted
//! step evaluates a Caputo sum over all prior samples and the whole run is
//! `O(N^2)` in the step count `N`. This benchmark times the fixed-step
//! integrator at doubling `N`, so the quadratic growth is visible in the
//! elapsed time (each doubling of `N` should roughly quadruple the time).
//!
//! Timing is machine-dependent and **not** bit-reproducible — that is inherent.
//! The deterministic, reproducible companion is the operation-count proxy in the
//! `complexity_scaling` experiment, which measures the same growth without a
//! wall clock. The integrator itself remains fully deterministic. Run with
//! `cargo bench -p scirust-nonlocal-relativity`.

use std::f64::consts::FRAC_PI_2;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use scirust_nonlocal_relativity::{
    CaputoCoordinateMemory, CompleteUniformHistory, IdentityHistoryTransport, NonlocalConfig,
    NonlocalSimulationPolicy, SemiImplicitEulerStepper, WorldlineState,
    simulate_nonlocal_worldline_with_policy,
};
use scirust_relativity::Schwarzschild;

const MASS: f64 = 1.0;
const RADIUS: f64 = 12.0;
const ALPHA: f64 = 0.55;
const COUPLING: f64 = 0.02;
const STEP: f64 = 0.005;
const NORM_FLOOR: f64 = 1.0e-8;

/// Circular equatorial geodesic initial state (requires `radius > 3 * mass`).
fn circular_state(mass: f64, radius: f64) -> WorldlineState<4> {
    let denominator = (1.0 - 3.0 * mass / radius).sqrt();
    let u_t = 1.0 / denominator;
    let u_phi = (mass / (radius * radius * radius)).sqrt() / denominator;
    WorldlineState::new([0.0, radius, FRAC_PI_2, 0.0], [u_t, 0.0, 0.0, u_phi])
}

fn caputo_history(criterion: &mut Criterion) {
    let background = Schwarzschild::try_new(MASS).expect("valid Schwarzschild mass");
    let initial = circular_state(MASS, RADIUS);

    let mut group = criterion.benchmark_group("caputo_complete_history");
    for &steps in &[100_usize, 200, 400]
    {
        group.bench_with_input(
            BenchmarkId::from_parameter(steps),
            &steps,
            |bencher, &steps| {
                bencher.iter(|| {
                    let config = NonlocalConfig::new(ALPHA, COUPLING, STEP, steps, NORM_FLOOR)
                        .expect("valid config");
                    black_box(
                        simulate_nonlocal_worldline_with_policy(
                            &background,
                            initial,
                            config,
                            NonlocalSimulationPolicy::new(
                                CompleteUniformHistory::<4>::with_capacity(steps + 1),
                                CaputoCoordinateMemory,
                                IdentityHistoryTransport,
                                SemiImplicitEulerStepper,
                            ),
                        )
                        .expect("finite worldline"),
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, caputo_history);
criterion_main!(benches);
