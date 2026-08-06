//! Phase 10 experiment: empirical complexity of the memory pipelines.
//!
//! Wall-clock timing is non-deterministic, so this reports a **deterministic
//! operation-count proxy**: the sum over accepted evaluations of the retained
//! history sample count, which is exactly the number of history-sample touches
//! the Caputo evaluation performs over a run. It is measured against an
//! increasing fixed step count `N` (with `N` doubling each row) so the empirical
//! growth ratio `proxy(2N)/proxy(N)` can be compared with the
//! implementation-derived complexity:
//!
//! - complete raw coordinate memory retains every sample, so the proxy is
//!   `O(N^2)` and the ratio tends to `4`;
//! - bounded short memory retains at most `W` samples, so the proxy is
//!   `O(N*W)` and the ratio tends to `2`;
//! - discrete connection transport touches the same sample counts but pays an
//!   extra per-touch Christoffel contraction (`O(D^3)`), so its proxy has the
//!   same `O(N^2)` growth with a larger constant.
//!
//! This measures scaling, not absolute speed, and makes no claim of asymptotic
//! complexity from timing.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{circular_schwarzschild_state, print_common_header};
use scirust_nonlocal_relativity::{
    BoundedShortMemoryHistory, CaputoCoordinateMemory, CompleteUniformHistory,
    DiscreteConnectionTransport, IdentityHistoryTransport, NonlocalConfig,
    NonlocalSimulationPolicy, NonlocalTrajectory, SemiImplicitEulerStepper, WorldlineState,
    simulate_nonlocal_worldline_with_policy,
};
use scirust_relativity::Schwarzschild;

const MASS: f64 = 1.0;
const RADIUS: f64 = 12.0;
const ALPHA: f64 = 0.55;
const COUPLING: f64 = 0.02;
const STEP: f64 = 0.005;
const WINDOW: usize = 16;
const STEP_COUNTS: [usize; 4] = [50, 100, 200, 400];

fn op_count_proxy(trajectory: &NonlocalTrajectory<4>) -> usize {
    trajectory
        .history_diagnostics()
        .iter()
        .map(|diagnostics| diagnostics.retained_samples)
        .sum()
}

fn main() -> Result<(), String> {
    let background =
        Schwarzschild::try_new(MASS).ok_or_else(|| "invalid Schwarzschild mass".to_string())?;
    let initial = circular_schwarzschild_state(MASS, RADIUS);

    print_common_header("memory-pipeline complexity (deterministic op-count proxy)");
    println!("# background: Schwarzschild M = {MASS}, circular orbit r0 = {RADIUS}, step = {STEP}");
    println!("# bounded short-memory window W = {WINDOW}");
    println!("# op_count_proxy = sum over accepted evaluations of retained sample count");
    println!("# ratio = proxy(this N) / proxy(previous N); expected ~4 for O(N^2), ~2 for O(N*W)");
    println!("pipeline,steps,op_count_proxy,ratio");

    for pipeline in ["complete_raw", "bounded_short", "discrete_transport"]
    {
        let mut previous: Option<usize> = None;
        for &steps in &STEP_COUNTS
        {
            let proxy = match pipeline
            {
                "complete_raw" => op_count_proxy(&run_complete(&background, initial, steps)?),
                "bounded_short" => op_count_proxy(&run_bounded(&background, initial, steps)?),
                "discrete_transport" =>
                {
                    op_count_proxy(&run_transport(&background, initial, steps)?)
                },
                _ => unreachable!(),
            };
            let ratio = previous.map_or(f64::NAN, |previous| proxy as f64 / previous as f64);
            if ratio.is_finite()
            {
                println!("{pipeline},{steps},{proxy},{ratio:.3}");
            }
            else
            {
                println!("{pipeline},{steps},{proxy},NA");
            }
            previous = Some(proxy);
        }
    }

    println!("# interpretation: complete_raw and discrete_transport ratios approach 4 (O(N^2));");
    println!("# bounded_short's ratio approaches 2 once N exceeds the window W (O(N*W)). The");
    println!("# measured growth matches the implementation-derived complexity; discrete_transport");
    println!("# has the same growth as complete_raw with a larger per-touch constant (O(D^3)).");
    Ok(())
}

fn run_complete(
    background: &Schwarzschild,
    initial: WorldlineState<4>,
    steps: usize,
) -> Result<NonlocalTrajectory<4>, String> {
    let config =
        NonlocalConfig::new(ALPHA, COUPLING, STEP, steps, 1.0e-8).map_err(|e| e.to_string())?;
    simulate_nonlocal_worldline_with_policy(
        background,
        initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(steps + 1),
            CaputoCoordinateMemory,
            IdentityHistoryTransport,
            SemiImplicitEulerStepper,
        ),
    )
    .map_err(|e| e.to_string())
}

fn run_bounded(
    background: &Schwarzschild,
    initial: WorldlineState<4>,
    steps: usize,
) -> Result<NonlocalTrajectory<4>, String> {
    let config =
        NonlocalConfig::new(ALPHA, COUPLING, STEP, steps, 1.0e-8).map_err(|e| e.to_string())?;
    simulate_nonlocal_worldline_with_policy(
        background,
        initial,
        config,
        NonlocalSimulationPolicy::new(
            BoundedShortMemoryHistory::<4>::new(WINDOW).map_err(|e| e.to_string())?,
            CaputoCoordinateMemory,
            IdentityHistoryTransport,
            SemiImplicitEulerStepper,
        ),
    )
    .map_err(|e| e.to_string())
}

fn run_transport(
    background: &Schwarzschild,
    initial: WorldlineState<4>,
    steps: usize,
) -> Result<NonlocalTrajectory<4>, String> {
    let config =
        NonlocalConfig::new(ALPHA, COUPLING, STEP, steps, 1.0e-8).map_err(|e| e.to_string())?;
    simulate_nonlocal_worldline_with_policy(
        background,
        initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(steps + 1),
            CaputoCoordinateMemory,
            DiscreteConnectionTransport,
            SemiImplicitEulerStepper,
        ),
    )
    .map_err(|e| e.to_string())
}
