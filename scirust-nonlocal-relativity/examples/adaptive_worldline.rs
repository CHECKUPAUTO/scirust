//! Compares the adaptive-step integrator
//! ([`simulate_nonlocal_worldline_adaptive`]) against the fixed-step
//! `HeunPeceStepper` reference path on the same Schwarzschild exterior
//! trajectory, at a sequence of tightening error tolerances.
//!
//! The adaptive integrator chooses its own non-uniform affine-parameter step
//! via an embedded Heun-Euler pair (see
//! `scirust-nonlocal-relativity/src/adaptive.rs`) and evaluates the Caputo
//! memory force with `scirust_fractional::caputo_l1_nonuniform` directly
//! against that non-uniform history, rather than resampling a
//! uniformly-stepped trajectory after the fact as
//! [`proper_time_caputo_velocity_memory`] does. This example does not claim
//! the adaptive path is more *physically* correct than the fixed-step
//! reference — both discretize the same ordinary state equation with a
//! history-dependent force — only that it reaches a comparable final state
//! with a smaller accepted-step count than a uniform grid fine enough for
//! the same tolerance would need.

use scirust_nonlocal_relativity::{
    AdaptiveNonlocalConfig, CaputoCoordinateMemory, CompleteUniformHistory,
    DiscreteConnectionTransport, HeunPeceStepper, IdentityHistoryTransport, NonlocalConfig,
    NonlocalSimulationPolicy, WorldlineState, coordinate_l2_norm,
    simulate_nonlocal_worldline_adaptive, simulate_nonlocal_worldline_with_policy,
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

fn state_distance(left: &WorldlineState<4>, right: &WorldlineState<4>) -> f64 {
    let mut coordinate_difference = [0.0_f64; 4];
    let mut velocity_difference = [0.0_f64; 4];

    for component in 0..4
    {
        coordinate_difference[component] =
            left.coordinates[component] - right.coordinates[component];
        velocity_difference[component] = left.velocity[component] - right.velocity[component];
    }

    coordinate_l2_norm(&coordinate_difference) + coordinate_l2_norm(&velocity_difference)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mass = 1.0;
    let background = Schwarzschild::try_new(mass).expect("positive mass");
    let mut initial = circular_schwarzschild_state(mass, 10.0);
    initial.velocity[1] = -0.01;

    let alpha = 0.55;
    let coupling = 0.02;
    let metric_norm_floor = 1.0e-8;
    let target: f64 = 1.6;

    // A very fine fixed-step reference, independent of the adaptive path,
    // used only as a stable comparison point.
    let reference_step = 0.001;
    let reference_steps = (target / reference_step).round() as usize;
    let reference_config = NonlocalConfig::new(
        alpha,
        coupling,
        reference_step,
        reference_steps,
        metric_norm_floor,
    )?;
    let reference_trajectory = simulate_nonlocal_worldline_with_policy(
        &background,
        initial,
        reference_config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(reference_steps + 1),
            CaputoCoordinateMemory,
            IdentityHistoryTransport,
            HeunPeceStepper,
        ),
    )?;
    let reference_final = reference_trajectory
        .final_state()
        .expect("non-empty trajectory");

    println!("method,error_tolerance,accepted_steps,final_parameter,distance_to_reference");

    for error_tolerance in [1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9]
    {
        let adaptive_config = AdaptiveNonlocalConfig::new(
            alpha,
            coupling,
            0.02,
            0.00002,
            0.1,
            error_tolerance,
            metric_norm_floor,
            target,
            20_000,
            60,
        )?;
        let adaptive_trajectory =
            simulate_nonlocal_worldline_adaptive(&background, initial, adaptive_config)?;
        let adaptive_final = adaptive_trajectory
            .final_state()
            .expect("non-empty trajectory");
        let adaptive_distance = state_distance(adaptive_final, reference_final);
        let adaptive_steps = adaptive_trajectory.len() - 1;
        let adaptive_final_parameter = adaptive_trajectory
            .final_diagnostics()
            .expect("non-empty trajectory")
            .affine_parameter;

        println!(
            "adaptive,{error_tolerance:.1e},{adaptive_steps},{adaptive_final_parameter:.12e},\
             {adaptive_distance:.12e}"
        );
    }

    // A matched-cost fixed-step comparison: how many *uniform* steps would
    // be needed to reach a comparable distance to the reference as the
    // tightest adaptive run above, using the discrete-transport path for
    // variety.
    let matched_steps = 800;
    let matched_step = target / matched_steps as f64;
    let matched_config = NonlocalConfig::new(
        alpha,
        coupling,
        matched_step,
        matched_steps,
        metric_norm_floor,
    )?;
    let matched_trajectory = simulate_nonlocal_worldline_with_policy(
        &background,
        initial,
        matched_config,
        NonlocalSimulationPolicy::new(
            CompleteUniformHistory::<4>::with_capacity(matched_steps + 1),
            CaputoCoordinateMemory,
            DiscreteConnectionTransport,
            HeunPeceStepper,
        ),
    )?;
    let matched_final = matched_trajectory
        .final_state()
        .expect("non-empty trajectory");
    let matched_distance = state_distance(matched_final, reference_final);

    println!("fixed_uniform,NA,{matched_steps},{target:.12e},{matched_distance:.12e}");

    Ok(())
}
