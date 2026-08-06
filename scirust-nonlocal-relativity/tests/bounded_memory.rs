//! Regression tests for the `BoundedShortMemoryHistory` short-memory
//! approximation: its endpoint error against the complete-history oracle
//! shrinks as the window grows, and a window covering every sample reproduces
//! the oracle bit-for-bit. Quantified by
//! `experiments/nonlocal-relativity-v2/bounded_memory_error`.

use scirust_nonlocal_relativity::{
    BoundedShortMemoryHistory, CaputoCoordinateMemory, CompleteUniformHistory,
    IdentityHistoryTransport, NonlocalConfig, NonlocalSimulationPolicy, SemiImplicitEulerStepper,
    WorldlineState, simulate_nonlocal_worldline_with_policy,
};
use scirust_relativity::Schwarzschild;
use std::f64::consts::FRAC_PI_2;

const ALPHA: f64 = 0.55;
const COUPLING: f64 = 0.05;
const STEP: f64 = 0.01;
const STEPS: usize = 128;

fn circular_schwarzschild_state(mass: f64, radius: f64) -> WorldlineState<4> {
    let denominator = (1.0 - 3.0 * mass / radius).sqrt();
    let u_t = 1.0 / denominator;
    let u_phi = (mass / (radius * radius * radius)).sqrt() / denominator;
    WorldlineState::new([0.0, radius, FRAC_PI_2, 0.0], [u_t, 0.0, 0.0, u_phi])
}

fn initial_state() -> WorldlineState<4> {
    let mut initial = circular_schwarzschild_state(1.0, 12.0);
    initial.velocity[1] = -0.01;
    initial
}

fn complete_final(background: &Schwarzschild) -> WorldlineState<4> {
    let config = NonlocalConfig::new(ALPHA, COUPLING, STEP, STEPS, 1.0e-8).unwrap();
    *simulate_nonlocal_worldline_with_policy(
        background,
        initial_state(),
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(STEPS + 1),
            CaputoCoordinateMemory,
            IdentityHistoryTransport,
            SemiImplicitEulerStepper,
        ),
    )
    .unwrap()
    .final_state()
    .unwrap()
}

fn bounded_final(background: &Schwarzschild, window: usize) -> WorldlineState<4> {
    let config = NonlocalConfig::new(ALPHA, COUPLING, STEP, STEPS, 1.0e-8).unwrap();
    *simulate_nonlocal_worldline_with_policy(
        background,
        initial_state(),
        config,
        NonlocalSimulationPolicy::new(
            BoundedShortMemoryHistory::<4>::new(window).unwrap(),
            CaputoCoordinateMemory,
            IdentityHistoryTransport,
            SemiImplicitEulerStepper,
        ),
    )
    .unwrap()
    .final_state()
    .unwrap()
}

fn endpoint_error(a: &WorldlineState<4>, b: &WorldlineState<4>) -> f64 {
    (0..4)
        .map(|i| {
            (a.coordinates[i] - b.coordinates[i]).powi(2) + (a.velocity[i] - b.velocity[i]).powi(2)
        })
        .sum::<f64>()
        .sqrt()
}

#[test]
fn window_covering_all_samples_reproduces_the_oracle_bit_for_bit() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let oracle = complete_final(&background);
    // The trajectory has STEPS + 1 samples; a window at least that large retains
    // every sample, so the bounded backend IS the complete history.
    let bounded = bounded_final(&background, STEPS + 1);
    for component in 0..4
    {
        assert_eq!(
            bounded.coordinates[component].to_bits(),
            oracle.coordinates[component].to_bits(),
            "coordinate {component}"
        );
        assert_eq!(
            bounded.velocity[component].to_bits(),
            oracle.velocity[component].to_bits(),
            "velocity {component}"
        );
    }
}

#[test]
fn short_memory_error_decreases_as_the_window_grows() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let oracle = complete_final(&background);

    let mut previous_error = f64::INFINITY;
    for &window in &[4_usize, 8, 16, 32, 64]
    {
        let error = endpoint_error(&bounded_final(&background, window), &oracle);
        assert!(error.is_finite());
        assert!(
            error < previous_error,
            "error did not decrease at window {window}: {error:e} !< {previous_error:e}"
        );
        previous_error = error;
    }

    // A very short window is a genuine approximation: its error is nonzero.
    let short_error = endpoint_error(&bounded_final(&background, 4), &oracle);
    assert!(short_error > 0.0, "expected a nonzero short-memory error");
}
