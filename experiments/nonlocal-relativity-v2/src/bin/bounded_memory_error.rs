//! Phase 9 experiment: bounded short-memory approximation error.
//!
//! `BoundedShortMemoryHistory` retains only the most recent `W` velocity
//! samples and applies the Caputo L1 stencil to that window (the classical
//! "short-memory principle"). It is an explicit `O(N*W)` approximation of the
//! `O(N^2)` complete-history oracle. This experiment quantifies the accuracy
//! cost: for increasing window `W`, it reports the endpoint coordinate and
//! velocity error against the complete-history oracle on the same fixed-step
//! trajectory, and the endpoint memory-vector norm. As `W` approaches the total
//! sample count the window becomes the full history and the error vanishes.
//!
//! This measures the approximation the user opts into by choosing a bounded
//! backend; it is not a validation of the underlying model.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{
    circular_schwarzschild_state, euclidean_distance, print_common_header, require_finite,
};
use scirust_nonlocal_relativity::{
    BoundedShortMemoryHistory, CaputoCoordinateMemory, CompleteUniformHistory,
    IdentityHistoryTransport, NonlocalConfig, NonlocalSimulationPolicy, NonlocalTrajectory,
    SemiImplicitEulerStepper, WorldlineState, simulate_nonlocal_worldline_with_policy,
};
use scirust_relativity::Schwarzschild;

const MASS: f64 = 1.0;
const RADIUS: f64 = 12.0;
const ALPHA: f64 = 0.55;
const COUPLING: f64 = 0.05;
const STEP: f64 = 0.01;
const STEPS: usize = 128;
const WINDOWS: [usize; 6] = [4, 8, 16, 32, 64, 129];

fn main() -> Result<(), String> {
    let background =
        Schwarzschild::try_new(MASS).ok_or_else(|| "invalid Schwarzschild mass".to_string())?;
    let mut initial = circular_schwarzschild_state(MASS, RADIUS);
    initial.velocity[1] = -0.01;

    let oracle = run_complete(&background, initial)?;
    let oracle_final = oracle.final_state().ok_or("empty oracle")?;

    print_common_header("bounded short-memory approximation error");
    println!("# background: Schwarzschild M = {MASS}, exterior near-circular orbit r0 = {RADIUS}");
    println!(
        "# fixed step = {STEP}, steps = {STEPS} (total samples = {}), alpha = {ALPHA}, kappa = {COUPLING}",
        STEPS + 1
    );
    println!(
        "# oracle: CompleteUniformHistory (retains all samples). W >= total samples reproduces it."
    );
    println!("window_W,endpoint_coord_err,endpoint_vel_err,endpoint_memory_l2,retained_samples");

    for &window in &WINDOWS
    {
        let trajectory = run_bounded(&background, initial, window)?;
        let final_state = trajectory.final_state().ok_or("empty trajectory")?;
        let final_diagnostics = trajectory.final_diagnostics().ok_or("no diagnostics")?;
        let retained = trajectory
            .history_diagnostics()
            .last()
            .ok_or("no history diagnostics")?
            .retained_samples;
        let coord_err = euclidean_distance(&final_state.coordinates, &oracle_final.coordinates);
        let vel_err = euclidean_distance(&final_state.velocity, &oracle_final.velocity);
        require_finite(&[
            ("endpoint_coord_err", coord_err),
            ("endpoint_vel_err", vel_err),
            ("endpoint_memory_l2", final_diagnostics.memory_l2_norm),
        ])?;
        println!(
            "{window},{coord_err:.6e},{vel_err:.6e},{:.9e},{retained}",
            final_diagnostics.memory_l2_norm
        );
    }

    println!("# interpretation: the endpoint error against the complete-history oracle decreases");
    println!("# monotonically as the window W grows, and vanishes once W covers every sample (the");
    println!("# window then IS the full history). This is the truncation cost of the short-memory");
    println!("# approximation, the price of its O(N*W) evaluation versus the oracle's O(N^2).");
    Ok(())
}

fn run_complete(
    background: &Schwarzschild,
    initial: WorldlineState<4>,
) -> Result<NonlocalTrajectory<4>, String> {
    let config = NonlocalConfig::new(ALPHA, COUPLING, STEP, STEPS, 1.0e-8).map_err(stringify)?;
    simulate_nonlocal_worldline_with_policy(
        background,
        initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(STEPS + 1),
            CaputoCoordinateMemory,
            IdentityHistoryTransport,
            SemiImplicitEulerStepper,
        ),
    )
    .map_err(stringify)
}

fn run_bounded(
    background: &Schwarzschild,
    initial: WorldlineState<4>,
    window: usize,
) -> Result<NonlocalTrajectory<4>, String> {
    let config = NonlocalConfig::new(ALPHA, COUPLING, STEP, STEPS, 1.0e-8).map_err(stringify)?;
    simulate_nonlocal_worldline_with_policy(
        background,
        initial,
        config,
        NonlocalSimulationPolicy::new(
            BoundedShortMemoryHistory::<4>::new(window).map_err(stringify)?,
            CaputoCoordinateMemory,
            IdentityHistoryTransport,
            SemiImplicitEulerStepper,
        ),
    )
    .map_err(stringify)
}

fn stringify<E: std::fmt::Display>(error: E) -> String {
    error.to_string()
}
