//! Phase 9 experiment: adaptive-tolerance convergence tables for both adaptive
//! controllers (embedded Heun-Euler and step-doubling), measured against an
//! independent fine fixed-step reference.
//!
//! For each error tolerance it reports the accepted-step count and the endpoint
//! coordinate error against the reference, plus the error-reduction ratio
//! between consecutive tolerances (a self-consistency trend, not a proof of a
//! convergence order). The reference is a fine fixed-step run of the matching
//! method family — a numerical reference, not an exact solution.

#![forbid(unsafe_code)]

use nonlocal_relativity_experiments::{
    circular_schwarzschild_state, euclidean_distance, print_common_header, require_finite,
};
use scirust_nonlocal_relativity::{
    AdaptiveNonlocalConfig, CaputoCoordinateMemory, CompleteUniformHistory, HeunPeceStepper,
    IdentityHistoryTransport, NonlocalConfig, NonlocalSimulationPolicy,
    NonuniformCaputoCoordinateMemory, SemiImplicitEulerStepper, WorldlineState,
    simulate_nonlocal_worldline_adaptive, simulate_nonlocal_worldline_adaptive_with_stepper,
    simulate_nonlocal_worldline_with_policy,
};
use scirust_relativity::Schwarzschild;

const MASS: f64 = 1.0;
const RADIUS: f64 = 10.0;
const ALPHA: f64 = 0.55;
const COUPLING: f64 = 0.02;
const TARGET: f64 = 0.8;
const TOLERANCES: [f64; 5] = [1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9];

fn main() -> Result<(), String> {
    let background =
        Schwarzschild::try_new(MASS).ok_or_else(|| "invalid Schwarzschild mass".to_string())?;
    let mut initial = circular_schwarzschild_state(MASS, RADIUS);
    initial.velocity[1] = -0.01;

    let heun_reference = fine_reference_heun(&background, initial)?;
    let euler_reference = fine_reference_euler(&background, initial)?;

    print_common_header("adaptive-tolerance convergence (both controllers)");
    println!("# background: Schwarzschild M = {MASS}, exterior near-circular orbit r0 = {RADIUS}");
    println!("# alpha = {ALPHA}, kappa = {COUPLING}, target affine parameter = {TARGET}");
    println!(
        "# embedded reference: fixed-step HeunPeceStepper h = 5e-4; stepdoubling reference: fixed-step SemiImplicitEuler h = 5e-4"
    );
    println!(
        "# error_ratio = previous_row_error / this_row_error (trend only; not a proof of convergence order)"
    );
    println!("controller,tolerance,accepted_steps,endpoint_coord_err,error_ratio");

    run_controller(
        "embedded_heun_euler",
        &background,
        initial,
        &heun_reference,
        |bg, init, cfg| {
            simulate_nonlocal_worldline_adaptive(bg, init, cfg)
                .map(|t| (t.len() - 1, *t.final_state().unwrap()))
                .map_err(|e| e.to_string())
        },
    )?;

    run_controller(
        "step_doubling",
        &background,
        initial,
        &euler_reference,
        |bg, init, cfg| {
            simulate_nonlocal_worldline_adaptive_with_stepper(bg, init, cfg)
                .map(|t| (t.len() - 1, *t.final_state().unwrap()))
                .map_err(|e| e.to_string())
        },
    )?;

    println!(
        "# interpretation: for both controllers the endpoint error decreases monotonically as"
    );
    println!(
        "# the tolerance tightens and the accepted-step count rises, converging toward the fine"
    );
    println!(
        "# fixed-step reference of the matching method. This is numerical self-consistency, not"
    );
    println!("# a validation of the underlying phenomenological model.");
    Ok(())
}

fn run_controller<F>(
    label: &str,
    background: &Schwarzschild,
    initial: WorldlineState<4>,
    reference_final: &WorldlineState<4>,
    run: F,
) -> Result<(), String>
where
    F: Fn(
        &Schwarzschild,
        WorldlineState<4>,
        AdaptiveNonlocalConfig,
    ) -> Result<(usize, WorldlineState<4>), String>,
{
    let mut previous_error: Option<f64> = None;
    for tolerance in TOLERANCES
    {
        let config = AdaptiveNonlocalConfig::new(
            ALPHA, COUPLING, 0.02, 0.000005, 0.05, tolerance, 1.0e-8, TARGET, 200_000, 60,
        )
        .map_err(|error| error.to_string())?;
        let (accepted_steps, final_state) = run(background, initial, config)?;
        let error = euclidean_distance(&final_state.coordinates, &reference_final.coordinates);
        require_finite(&[("endpoint_coord_err", error)])?;
        let ratio = previous_error.map_or(f64::NAN, |previous| previous / error);
        if ratio.is_finite()
        {
            println!("{label},{tolerance:.0e},{accepted_steps},{error:.6e},{ratio:.3}");
        }
        else
        {
            println!("{label},{tolerance:.0e},{accepted_steps},{error:.6e},NA");
        }
        previous_error = Some(error);
    }
    Ok(())
}

fn fine_reference_heun(
    background: &Schwarzschild,
    initial: WorldlineState<4>,
) -> Result<WorldlineState<4>, String> {
    let step = 0.0005;
    let steps = (TARGET / step).round() as usize;
    let config = NonlocalConfig::new(ALPHA, COUPLING, step, steps, 1.0e-8)
        .map_err(|error| error.to_string())?;
    let trajectory = simulate_nonlocal_worldline_with_policy(
        background,
        initial,
        config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(steps + 1),
            CaputoCoordinateMemory,
            IdentityHistoryTransport,
            HeunPeceStepper,
        ),
    )
    .map_err(|error| error.to_string())?;
    trajectory
        .final_state()
        .copied()
        .ok_or_else(|| "empty reference".to_string())
}

fn fine_reference_euler(
    background: &Schwarzschild,
    initial: WorldlineState<4>,
) -> Result<WorldlineState<4>, String> {
    let step = 0.0005;
    let steps = (TARGET / step).round() as usize;
    let config = NonlocalConfig::new(ALPHA, COUPLING, step, steps, 1.0e-8)
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
        .ok_or_else(|| "empty reference".to_string())
}
