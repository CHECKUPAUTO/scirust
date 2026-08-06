use scirust_nonlocal_relativity::{
    CylindricalMinkowski, DiscreteConnectionTransport, HistoryEntry, IdentityHistoryTransport,
    NonlocalRelativityError, cartesian_to_cylindrical_coordinates,
    cartesian_to_cylindrical_velocity, exact_cylindrical_minkowski_transport,
    transport_vector_along_polyline,
};

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    let scale = expected.abs().max(1.0);
    let relative_error = (actual - expected).abs() / scale;

    assert!(
        relative_error <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, \
         relative_error={relative_error:.17e}, tolerance={tolerance:.17e}"
    );
}

fn vector_distance(left: &[f64; 4], right: &[f64; 4]) -> f64 {
    let mut sum = 0.0;

    for component in 0..4
    {
        let difference = left[component] - right[component];
        sum += difference * difference;
    }

    sum.sqrt()
}

/// Build a polyline of `waypoints` evenly sampled cylindrical `HistoryEntry`
/// values along a straight Cartesian path
/// `position(lambda) = start + lambda * cartesian_velocity`, together with
/// the (constant) Cartesian velocity converted into cylindrical components at
/// each waypoint.
fn straight_line_polyline(waypoint_count: usize, total_lambda: f64) -> Vec<HistoryEntry<4>> {
    let cartesian_start = [0.0, 3.0, 4.0, 0.0];
    let cartesian_velocity = [1.0, 0.3, -0.2, 0.1];
    let mut waypoints = Vec::with_capacity(waypoint_count + 1);

    for index in 0..=waypoint_count
    {
        let lambda = if waypoint_count == 0
        {
            0.0
        }
        else
        {
            total_lambda * index as f64 / waypoint_count as f64
        };
        let mut cartesian_coordinates = cartesian_start;
        for component in 0..4
        {
            cartesian_coordinates[component] += lambda * cartesian_velocity[component];
        }

        let cylindrical_coordinates =
            cartesian_to_cylindrical_coordinates(cartesian_coordinates).unwrap();
        let cylindrical_velocity =
            cartesian_to_cylindrical_velocity(cartesian_coordinates, cartesian_velocity).unwrap();

        waypoints.push(HistoryEntry::new(
            cylindrical_coordinates,
            cylindrical_velocity,
            lambda,
        ));
    }

    waypoints
}

fn numerical_transport_error(waypoint_count: usize) -> f64 {
    let waypoints = straight_line_polyline(waypoint_count, 2.0);
    let test_vector_cartesian = [0.0, 0.0, 1.0, 0.0];
    let start_cylindrical_velocity = cartesian_to_cylindrical_velocity(
        scirust_nonlocal_relativity::cylindrical_to_cartesian_coordinates(waypoints[0].coordinates)
            .unwrap(),
        test_vector_cartesian,
    )
    .unwrap();

    let numerical = transport_vector_along_polyline(
        &CylindricalMinkowski,
        &DiscreteConnectionTransport,
        start_cylindrical_velocity,
        &waypoints,
    )
    .unwrap();
    let exact = exact_cylindrical_minkowski_transport(
        waypoints[0].coordinates,
        start_cylindrical_velocity,
        waypoints.last().unwrap().coordinates,
    )
    .unwrap();

    vector_distance(&numerical, &exact)
}

#[test]
fn exact_transport_round_trip_recovers_original_vector() {
    let from = [0.0, 5.0, 0.9, 0.0];
    let to = [1.0, 5.4, 1.1, 0.2];
    let vector = [0.2, 0.1, -0.05, 0.3];

    let transported = exact_cylindrical_minkowski_transport(from, vector, to).unwrap();
    let back = exact_cylindrical_minkowski_transport(to, transported, from).unwrap();

    for component in 0..4
    {
        assert_close(back[component], vector[component], 1.0e-12);
    }
}

#[test]
fn exact_transport_is_deterministic_bit_for_bit() {
    let from = [0.0, 5.0, 0.9, 0.0];
    let to = [1.0, 5.4, 1.1, 0.2];
    let vector = [0.2, 0.1, -0.05, 0.3];

    let first = exact_cylindrical_minkowski_transport(from, vector, to).unwrap();
    let second = exact_cylindrical_minkowski_transport(from, vector, to).unwrap();

    for component in 0..4
    {
        assert_eq!(first[component].to_bits(), second[component].to_bits());
    }
}

#[test]
fn discrete_transport_converges_to_the_exact_oracle_under_refinement() {
    let error_coarse = numerical_transport_error(4);
    let error_medium = numerical_transport_error(16);
    let error_fine = numerical_transport_error(64);

    assert!(
        error_medium < error_coarse,
        "error did not shrink: coarse={error_coarse:.6e}, medium={error_medium:.6e}"
    );
    assert!(
        error_fine < error_medium,
        "error did not shrink further: medium={error_medium:.6e}, fine={error_fine:.6e}"
    );
    assert!(
        error_fine < 1.0e-6,
        "finest error was not small: {error_fine:.6e}"
    );
}

#[test]
fn identity_transport_along_polyline_leaves_vector_unchanged() {
    let waypoints = straight_line_polyline(8, 2.0);
    let vector = [0.4, -0.3, 0.2, 0.1];

    let result = transport_vector_along_polyline(
        &CylindricalMinkowski,
        &IdentityHistoryTransport,
        vector,
        &waypoints,
    )
    .unwrap();

    for component in 0..4
    {
        assert_eq!(result[component].to_bits(), vector[component].to_bits());
    }
}

#[test]
fn single_waypoint_polyline_leaves_vector_unchanged() {
    let waypoints = straight_line_polyline(0, 2.0);
    assert_eq!(waypoints.len(), 1);
    let vector = [0.4, -0.3, 0.2, 0.1];

    let result = transport_vector_along_polyline(
        &CylindricalMinkowski,
        &DiscreteConnectionTransport,
        vector,
        &waypoints,
    )
    .unwrap();

    for component in 0..4
    {
        assert_eq!(result[component].to_bits(), vector[component].to_bits());
    }
}

#[test]
fn empty_polyline_is_rejected() {
    let waypoints: Vec<HistoryEntry<4>> = Vec::new();
    let vector = [0.4, -0.3, 0.2, 0.1];

    let result = transport_vector_along_polyline(
        &CylindricalMinkowski,
        &DiscreteConnectionTransport,
        vector,
        &waypoints,
    );

    assert!(matches!(
        result,
        Err(NonlocalRelativityError::EmptyTransportPolyline)
    ));
}

#[test]
fn transported_and_exact_vectors_remain_finite() {
    let waypoints = straight_line_polyline(12, 2.0);
    let vector = [0.4, -0.3, 0.2, 0.1];

    let numerical = transport_vector_along_polyline(
        &CylindricalMinkowski,
        &DiscreteConnectionTransport,
        vector,
        &waypoints,
    )
    .unwrap();
    let exact = exact_cylindrical_minkowski_transport(
        waypoints[0].coordinates,
        vector,
        waypoints.last().unwrap().coordinates,
    )
    .unwrap();

    assert!(numerical.iter().all(|value| value.is_finite()));
    assert!(exact.iter().all(|value| value.is_finite()));
}
