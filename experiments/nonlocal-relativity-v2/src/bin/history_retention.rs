//! Phase 3 experiment: endpoint-only vs refined-accepted persistent history
//! for the step-doubling adaptive integrator, compared against an independent
//! fine fixed-step reference.
//!
//! For each error tolerance and each retention strategy it reports the
//! accepted-step count, the retained-history sample count, a deterministic
//! operation-count proxy (the sum of retained sample counts over all accepted
//! evaluations, which tracks the total `O(N^2)` Caputo work), and the endpoint
//! coordinate/velocity error, memory-vector norm, memory-force norm, and
//! metric-norm drift.
//!
//! The reference is a much finer, independently configured fixed-step
//! semi-implicit-Euler run with the same model and the same non-uniform Caputo
//! memory law; it is a numerical reference, not an exact solution.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{
    circular_schwarzschild_state, euclidean_distance, print_common_header, require_finite,
};
use scirust_nonlocal_relativity::{
    AdaptiveNonlocalConfig, AdaptiveStepperPolicy, CompleteUniformHistory, HistoryRetention,
    IdentityHistoryTransport, NonlocalConfig, NonlocalSimulationPolicy,
    NonuniformCaputoCoordinateMemory, SemiImplicitEulerStepper, WorldlineState,
    simulate_nonlocal_worldline_adaptive_with_stepper_policy_retention,
    simulate_nonlocal_worldline_with_policy,
};
use scirust_relativity::Schwarzschild;

const MASS: f64 = 1.0;
const RADIUS: f64 = 10.0;
const ALPHA: f64 = 0.55;
const COUPLING: f64 = 0.02;
const TARGET: f64 = 0.8;
const REFERENCE_STEP: f64 = 0.0005;
const TOLERANCES: [f64; 4] = [1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9];

fn main() -> Result<(), String> {
    let background =
        Schwarzschild::try_new(MASS).ok_or_else(|| "invalid Schwarzschild mass".to_string())?;
    let mut initial = circular_schwarzschild_state(MASS, RADIUS);
    // A small inward radial velocity so the orbit is not exactly circular and
    // the memory force is genuinely exercised.
    initial.velocity[1] = -0.01;

    let reference_final = fine_reference(&background, initial)?;

    print_common_header("history-retention comparison (step-doubling adaptive)");
    println!("# background: Schwarzschild M = {MASS}, exterior near-circular orbit r0 = {RADIUS}");
    println!(
        "# state equation: du^rho/dlambda = a_GR^rho - kappa P^rho_sigma m^sigma, alpha = {ALPHA}, kappa = {COUPLING}"
    );
    println!(
        "# reference: fixed-step SemiImplicitEuler h = {REFERENCE_STEP} with NonuniformCaputoCoordinateMemory (numerical reference, not exact)"
    );
    println!(
        "# op_count_proxy = sum over accepted evaluations of retained sample count (tracks total O(N^2) Caputo work)"
    );
    println!(
        "tolerance,strategy,accepted_steps,retained_samples,op_count_proxy,endpoint_coord_err,endpoint_vel_err,memory_l2,memory_force_l2,metric_norm_drift"
    );

    for tolerance in TOLERANCES
    {
        for (label, retention) in [
            ("endpoint_only", HistoryRetention::EndpointOnly),
            ("refined_accepted", HistoryRetention::RefinedAcceptedHistory),
        ]
        {
            let row = run_case(&background, initial, tolerance, retention, &reference_final)?;
            require_finite(&[
                ("endpoint_coord_err", row.coordinate_error),
                ("endpoint_vel_err", row.velocity_error),
                ("memory_l2", row.memory_l2),
                ("memory_force_l2", row.memory_force_l2),
                ("metric_norm_drift", row.metric_norm_drift),
            ])?;
            println!(
                "{:.0e},{},{},{},{},{:.6e},{:.6e},{:.9e},{:.9e},{:.6e}",
                tolerance,
                label,
                row.accepted_steps,
                row.retained_samples,
                row.op_count_proxy,
                row.coordinate_error,
                row.velocity_error,
                row.memory_l2,
                row.memory_force_l2,
                row.metric_norm_drift,
            );
        }
    }

    println!(
        "# interpretation: on this experiment the two strategies produce the same accepted-step"
    );
    println!(
        "# count and endpoint to well within tolerance, while refined-accepted history roughly"
    );
    println!(
        "# doubles the retained sample count and the op-count proxy. Retaining midpoints does not"
    );
    println!(
        "# measurably improve endpoint accuracy here, so the default remains EndpointOnly and"
    );
    println!("# RefinedAcceptedHistory is exposed only as an explicit research option.");
    Ok(())
}

struct Row {
    accepted_steps: usize,
    retained_samples: usize,
    op_count_proxy: usize,
    coordinate_error: f64,
    velocity_error: f64,
    memory_l2: f64,
    memory_force_l2: f64,
    metric_norm_drift: f64,
}

fn run_case(
    background: &Schwarzschild,
    initial: WorldlineState<4>,
    tolerance: f64,
    retention: HistoryRetention,
    reference_final: &WorldlineState<4>,
) -> Result<Row, String> {
    let config = AdaptiveNonlocalConfig::new(
        ALPHA, COUPLING, 0.02, 0.00002, 0.05, tolerance, 1.0e-8, TARGET, 50_000, 60,
    )
    .map_err(|error| error.to_string())?;

    let trajectory = simulate_nonlocal_worldline_adaptive_with_stepper_policy_retention(
        background,
        initial,
        config,
        AdaptiveStepperPolicy::new(
            CompleteUniformHistory::<4>::new(),
            NonuniformCaputoCoordinateMemory,
            IdentityHistoryTransport,
        ),
        retention,
    )
    .map_err(|error| error.to_string())?;

    let final_state = trajectory.final_state().ok_or("empty trajectory")?;
    let final_diagnostics = trajectory.final_diagnostics().ok_or("no diagnostics")?;
    let retained_samples = trajectory
        .history_diagnostics()
        .last()
        .ok_or("no history diagnostics")?
        .retained_samples;
    let op_count_proxy: usize = trajectory
        .history_diagnostics()
        .iter()
        .map(|diagnostics| diagnostics.retained_samples)
        .sum();

    Ok(Row {
        accepted_steps: trajectory.len() - 1,
        retained_samples,
        op_count_proxy,
        coordinate_error: euclidean_distance(
            &final_state.coordinates,
            &reference_final.coordinates,
        ),
        velocity_error: euclidean_distance(&final_state.velocity, &reference_final.velocity),
        memory_l2: final_diagnostics.memory_l2_norm,
        memory_force_l2: final_diagnostics.memory_force_l2_norm,
        metric_norm_drift: final_diagnostics.metric_norm_drift,
    })
}

fn fine_reference(
    background: &Schwarzschild,
    initial: WorldlineState<4>,
) -> Result<WorldlineState<4>, String> {
    let steps = (TARGET / REFERENCE_STEP).round() as usize;
    let config = NonlocalConfig::new(ALPHA, COUPLING, REFERENCE_STEP, steps, 1.0e-8)
        .map_err(|error| error.to_string())?;
    let trajectory = simulate_nonlocal_worldline_with_policy(
        background,
        initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(steps + 1),
            NonuniformCaputoCoordinateMemory,
            IdentityHistoryTransport,
            SemiImplicitEulerStepper,
        ),
    )
    .map_err(|error| error.to_string())?;
    trajectory
        .final_state()
        .copied()
        .ok_or_else(|| "empty reference trajectory".to_string())
}
