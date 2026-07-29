//! Runs the nonlocal-memory worldline model on a stationary observer in the
//! Kerr background, at increasing spin, to demonstrate that this crate's
//! worldline and memory machinery works unmodified on a rotating
//! background: the only background-specific code is `Kerr::components` and
//! `Kerr::christoffel` themselves (the latter evaluated by finite
//! differences — see `scirust-relativity/src/kerr.rs`).
//!
//! The initial state here is a **stationary** observer (zero spatial
//! velocity in Boyer-Lindquist coordinates) at a radius well outside both
//! the horizon and the ergosphere, deliberately avoiding any Kerr-specific
//! circular-orbit formula (which, unlike Schwarzschild's, involves a
//! prograde/retrograde asymmetry and an ISCO shift that this crate does not
//! derive or claim). It is not a geodesic, so the trajectory drifts under
//! the ordinary geodesic acceleration once released, exactly like the
//! "near-circular but not exact" states used in this crate's other
//! examples.

use scirust_nonlocal_relativity::{
    CaputoCoordinateMemory, CompleteUniformHistory, HeunPeceStepper, IdentityHistoryTransport,
    NonlocalConfig, NonlocalSimulationPolicy, WorldlineState,
    simulate_nonlocal_worldline_with_policy,
};
use scirust_relativity::{Kerr, Metric};
use std::error::Error;
use std::f64::consts::FRAC_PI_2;

fn stationary_state(background: &Kerr, radius: f64) -> WorldlineState<4> {
    let coordinates = [0.0, radius, FRAC_PI_2, 0.0];
    let metric = background.components(&coordinates);
    let time_component = 1.0 / (-metric[0][0]).sqrt();

    WorldlineState::new(coordinates, [time_component, 0.0, 0.0, 0.0])
}

fn main() -> Result<(), Box<dyn Error>> {
    let mass = 1.0;
    let radius = 15.0;
    let alpha = 0.55;
    let coupling = 0.02;
    let step = 0.02;
    let steps = 80;
    let metric_norm_floor = 1.0e-8;

    println!("spin,lambda,radius,phi,metric_norm_drift,memory_l2_norm,memory_force_l2_norm");

    for spin in [0.0, 0.5, 0.9]
    {
        let background = Kerr::try_new(mass, spin).ok_or("sub-extremal spin required")?;
        let initial = stationary_state(&background, radius);
        let config = NonlocalConfig::new(alpha, coupling, step, steps, metric_norm_floor)?;

        let trajectory = simulate_nonlocal_worldline_with_policy(
            &background,
            initial,
            config,
            NonlocalSimulationPolicy::new(
                CompleteUniformHistory::<4>::with_capacity(steps + 1),
                CaputoCoordinateMemory,
                IdentityHistoryTransport,
                HeunPeceStepper,
            ),
        )?;

        for index in (0..trajectory.len()).step_by(16)
        {
            let state = trajectory.states()[index];
            let diagnostics = trajectory.diagnostics()[index];

            println!(
                "{spin},{:.12e},{:.12},{:.12e},{:.12e},{:.12e},{:.12e}",
                diagnostics.affine_parameter,
                state.coordinates[1],
                state.coordinates[3],
                diagnostics.metric_norm_drift,
                diagnostics.memory_l2_norm,
                diagnostics.memory_force_l2_norm,
            );
        }
    }

    Ok(())
}
