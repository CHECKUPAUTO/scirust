//! Compares the standard affine-parameter Caputo velocity memory against the
//! proper-time-based Caputo velocity memory
//! ([`proper_time_caputo_velocity_memory`]) on the same Schwarzschild
//! exterior trajectory, at increasing refinement.
//!
//! Because the memory-force law in this crate is built to be orthogonal to
//! the four-velocity (`u_rho F_memory^rho = 0`), the metric norm `g(u,u)`
//! stays close to constant along an accepted trajectory, with only a small,
//! refinement-shrinking numerical drift (see the `final_metric_norm_drift`
//! column, which drops by roughly a factor of 4 each time the step halves —
//! consistent with the Heun PECE integrator's second-order local accuracy).
//! Proper time therefore advances at an approximately *constant* rate
//! `c = sqrt(-g(u,u))` relative to the affine parameter, and a Caputo
//! derivative computed against a linearly rescaled parameter differs from the
//! original by a factor of `c^(-alpha)` — a fact about the Caputo operator
//! itself, not a discretization artifact. Because `c` does not depend on the
//! step size, this predicted difference between the two memory values is
//! expected to stay roughly constant under refinement (it does, to within
//! the shrinking numerical drift), unlike the metric-norm drift itself. This
//! is deliberately not a large effect in this model, and the example does
//! not claim otherwise.

use scirust_nonlocal_relativity::{
    CaputoCoordinateMemory, CompleteUniformHistory, HeunPeceStepper, IdentityHistoryTransport,
    NonlocalConfig, NonlocalSimulationPolicy, WorldlineState, coordinate_l2_norm,
    proper_time_caputo_velocity_memory, simulate_nonlocal_worldline_with_policy,
};
use scirust_relativity::Schwarzschild;
use std::error::Error;
use std::f64::consts::FRAC_PI_2;

fn circular_schwarzschild_state(mass: f64, radius: f64) -> WorldlineState<4> {
    let denominator = (1.0 - 3.0 * mass / radius).sqrt();
    let u_t = 1.0 / denominator;
    let u_phi = (mass / (radius * radius * radius)).sqrt() / denominator;

    WorldlineState::new([0.0, radius, FRAC_PI_2, 0.0], [u_t, 0.0, 0.0, u_phi])
}

fn main() -> Result<(), Box<dyn Error>> {
    let mass = 1.0;
    let background = Schwarzschild::try_new(mass).expect("positive mass");
    let mut initial = circular_schwarzschild_state(mass, 10.0);
    initial.velocity[1] = -0.05;

    let alpha = 0.55;
    let coupling = 0.05;
    let metric_norm_floor = 1.0e-8;
    let base_step = 0.02;
    let base_steps = 80;

    println!(
        "refinement_level,step,affine_memory_l2_norm,proper_time_memory_l2_norm,\
         relative_difference,final_metric_norm_drift"
    );

    for (refinement_label, factor) in [("h", 1usize), ("h/2", 2usize), ("h/4", 4usize)]
    {
        let step = base_step / factor as f64;
        let steps = base_steps * factor;
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

        let affine_memory_norm = trajectory
            .final_diagnostics()
            .expect("non-empty trajectory")
            .memory_l2_norm;
        let metric_norm_drift = trajectory
            .final_diagnostics()
            .expect("non-empty trajectory")
            .metric_norm_drift;

        let proper_time_memory =
            proper_time_caputo_velocity_memory(&trajectory, step, config.fractional_order())?;
        let proper_time_memory_norm = coordinate_l2_norm(&proper_time_memory);

        let relative_difference =
            (proper_time_memory_norm - affine_memory_norm).abs() / affine_memory_norm.max(1.0e-300);

        println!(
            "{refinement_label},{step:.12e},{affine_memory_norm:.12e},\
             {proper_time_memory_norm:.12e},{relative_difference:.12e},{metric_norm_drift:.12e}"
        );
    }

    Ok(())
}
