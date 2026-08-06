//! Demonstrates that [`DiscreteConnectionTransport`]'s accumulated numerical
//! transport converges, under path refinement, to the exact closed-form
//! answer given by [`exact_cylindrical_minkowski_transport`].
//!
//! Parallel transport in flat spacetime is path-independent (zero curvature
//! means trivial holonomy around any contractible loop), so a vector's
//! Cartesian components are unchanged by transport along *any* path; this
//! gives an exact reference with no discretization at all, for this specific
//! flat-spacetime chart pair. This is a genuinely stronger validation than
//! comparing two discretizations to each other (as
//! `examples/coordinate_covariance.rs` does): here the numerical scheme is
//! checked against a known-exact answer.
//!
//! This does **not** extend to curved backgrounds: `Schwarzschild` (or any
//! other non-flat spacetime) has no such closed-form transport in this
//! crate, and `DiscreteConnectionTransport` remains a discrete approximation
//! there.

use scirust_nonlocal_relativity::{
    CylindricalMinkowski, DiscreteConnectionTransport, HistoryEntry,
    cartesian_to_cylindrical_coordinates, cartesian_to_cylindrical_velocity, coordinate_l2_norm,
    cylindrical_to_cartesian_coordinates, exact_cylindrical_minkowski_transport,
    transport_vector_along_polyline,
};
use std::error::Error;

/// Build a polyline of `waypoint_count + 1` evenly sampled cylindrical
/// `HistoryEntry` values along a straight Cartesian path
/// `position(lambda) = start + lambda * cartesian_velocity`.
fn straight_line_polyline(
    waypoint_count: usize,
    total_lambda: f64,
) -> Result<Vec<HistoryEntry<4>>, Box<dyn Error>> {
    let cartesian_start = [0.0, 3.0, 4.0, 0.0];
    let cartesian_velocity = [1.0, 0.3, -0.2, 0.1];
    let mut waypoints = Vec::with_capacity(waypoint_count + 1);

    for index in 0..=waypoint_count
    {
        let lambda = total_lambda * index as f64 / waypoint_count as f64;
        let mut cartesian_coordinates = cartesian_start;

        for component in 0..4
        {
            cartesian_coordinates[component] += lambda * cartesian_velocity[component];
        }

        let cylindrical_coordinates = cartesian_to_cylindrical_coordinates(cartesian_coordinates)?;
        let cylindrical_velocity =
            cartesian_to_cylindrical_velocity(cartesian_coordinates, cartesian_velocity)?;

        waypoints.push(HistoryEntry::new(
            cylindrical_coordinates,
            cylindrical_velocity,
            lambda,
        ));
    }

    Ok(waypoints)
}

fn vector_distance(left: &[f64; 4], right: &[f64; 4]) -> f64 {
    let mut difference = [0.0_f64; 4];

    for component in 0..4
    {
        difference[component] = left[component] - right[component];
    }

    coordinate_l2_norm(&difference)
}

fn main() -> Result<(), Box<dyn Error>> {
    let total_lambda = 2.0;
    let test_vector_cartesian = [0.0, 0.0, 1.0, 0.0];

    println!("waypoints,segment_step,numerical_error,error_ratio_to_previous");

    let mut previous_error: Option<f64> = None;

    for waypoint_count in [4usize, 8, 16, 32, 64, 128]
    {
        let waypoints = straight_line_polyline(waypoint_count, total_lambda)?;
        let start_coordinates_cartesian =
            cylindrical_to_cartesian_coordinates(waypoints[0].coordinates)?;
        let start_vector =
            cartesian_to_cylindrical_velocity(start_coordinates_cartesian, test_vector_cartesian)?;

        let numerical = transport_vector_along_polyline(
            &CylindricalMinkowski,
            &DiscreteConnectionTransport,
            start_vector,
            &waypoints,
        )?;
        let exact = exact_cylindrical_minkowski_transport(
            waypoints[0].coordinates,
            start_vector,
            waypoints.last().expect("non-empty polyline").coordinates,
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
