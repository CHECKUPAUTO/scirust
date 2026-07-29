//! Demonstrates that [`DiscreteConnectionTransport`]'s accumulated numerical
//! transport converges, under path refinement, to the exact closed-form
//! answer given by [`exact_schwarzschild_circular_orbit_transport`] along a
//! circular equatorial Schwarzschild orbit.
//!
//! Unlike `examples/exact_transport_convergence.rs`, this is a genuinely
//! **curved** background: Schwarzschild has nonzero curvature everywhere
//! outside its horizon, so the flat-spacetime path-independence argument
//! does not apply here. What makes an exact answer available instead is a
//! different structural fact: along a circular equatorial orbit, both `r`
//! and `theta` are fixed and the orbit's four-velocity is constant, so the
//! transport equation reduces to a constant-coefficient linear ODE with the
//! exact solution `V(lambda) = exp(-lambda A) V(0)` for the fixed generator
//! `A`. See `scirust-nonlocal-relativity/src/curved_transport.rs` for the
//! full derivation.
//!
//! This does **not** extend to general curved paths: eccentric orbits,
//! inclined orbits, or any trajectory where `r` or `theta` vary, have no
//! such closed-form shortcut in this crate.

use scirust_nonlocal_relativity::{
    DiscreteConnectionTransport, HistoryEntry, coordinate_l2_norm,
    exact_schwarzschild_circular_orbit_transport, schwarzschild_circular_orbit_angular_velocity,
    schwarzschild_circular_orbit_four_velocity, transport_vector_along_polyline,
};
use scirust_relativity::Schwarzschild;
use std::error::Error;
use std::f64::consts::FRAC_PI_2;

fn vector_distance(left: &[f64; 4], right: &[f64; 4]) -> f64 {
    let mut difference = [0.0_f64; 4];

    for component in 0..4
    {
        difference[component] = left[component] - right[component];
    }

    coordinate_l2_norm(&difference)
}

fn orbit_polyline(
    background: &Schwarzschild,
    radius: f64,
    total_lambda: f64,
    waypoint_count: usize,
) -> Result<Vec<HistoryEntry<4>>, Box<dyn Error>> {
    let four_velocity = schwarzschild_circular_orbit_four_velocity(background, radius)?;
    let angular_velocity = schwarzschild_circular_orbit_angular_velocity(background, radius)?;
    let mut waypoints = Vec::with_capacity(waypoint_count + 1);

    for index in 0..=waypoint_count
    {
        let lambda = total_lambda * index as f64 / waypoint_count as f64;
        let time = four_velocity[0] * lambda;
        let phi = angular_velocity * time;
        let coordinates = [time, radius, FRAC_PI_2, phi];
        waypoints.push(HistoryEntry::new(coordinates, four_velocity, lambda));
    }

    Ok(waypoints)
}

fn main() -> Result<(), Box<dyn Error>> {
    let background = Schwarzschild::try_new(1.0).expect("positive mass");
    let radius = 10.0;
    let four_velocity = schwarzschild_circular_orbit_four_velocity(&background, radius)?;
    let orbit_fraction = 0.02;
    let total_lambda = orbit_fraction * 2.0 * std::f64::consts::PI / four_velocity[3];
    let test_vector = [0.1, 0.4, -0.2, 0.05];

    println!("waypoints,segment_step,numerical_error,error_ratio_to_previous");

    let mut previous_error: Option<f64> = None;

    for waypoint_count in [4usize, 8, 16, 32, 64, 128]
    {
        let waypoints = orbit_polyline(&background, radius, total_lambda, waypoint_count)?;

        let numerical = transport_vector_along_polyline(
            &background,
            &DiscreteConnectionTransport,
            test_vector,
            &waypoints,
        )?;
        let exact = exact_schwarzschild_circular_orbit_transport(
            &background,
            radius,
            test_vector,
            total_lambda,
        )?;

        let error = vector_distance(&numerical, &exact);
        let segment_step = total_lambda / waypoint_count as f64;
        let ratio = previous_error.map(|previous| previous / error);

        match ratio
        {
            Some(ratio) => println!("{waypoint_count},{segment_step:.12e},{error:.12e},{ratio:.6}"),
            None => println!("{waypoint_count},{segment_step:.12e},{error:.12e},NA"),
        }

        previous_error = Some(error);
    }

    Ok(())
}
