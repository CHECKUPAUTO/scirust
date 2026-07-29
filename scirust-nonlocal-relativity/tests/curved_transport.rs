use scirust_nonlocal_relativity::{
    DiscreteConnectionTransport, HistoryEntry, NonlocalRelativityError,
    exact_schwarzschild_circular_orbit_transport, metric_contraction,
    schwarzschild_circular_orbit_angular_velocity, schwarzschild_circular_orbit_four_velocity,
    transport_vector_along_polyline,
};
use scirust_relativity::{Metric, Schwarzschild};
use std::f64::consts::FRAC_PI_2;

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

#[test]
fn circular_orbit_four_velocity_is_timelike_unit_normalized() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let radius = 10.0;
    let four_velocity = schwarzschild_circular_orbit_four_velocity(&background, radius).unwrap();

    let coordinates = [0.0, radius, FRAC_PI_2, 0.0];
    let metric = background.components(&coordinates);
    let norm = metric_contraction(&metric, &four_velocity, &four_velocity);

    assert_close(norm, -1.0, 1.0e-13);
}

#[test]
fn circular_orbit_angular_velocity_matches_four_velocity_ratio() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let radius = 8.0;

    let angular_velocity =
        schwarzschild_circular_orbit_angular_velocity(&background, radius).unwrap();
    let four_velocity = schwarzschild_circular_orbit_four_velocity(&background, radius).unwrap();

    assert_close(
        four_velocity[3] / four_velocity[0],
        angular_velocity,
        1.0e-13,
    );
}

#[test]
fn exact_transport_preserves_metric_norm_squared() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let radius = 12.0;
    let vector = [0.3, -0.2, 0.4, 0.1];
    let coordinates = [0.0, radius, FRAC_PI_2, 0.0];
    let metric = background.components(&coordinates);
    let initial_norm = metric_contraction(&metric, &vector, &vector);

    let transported =
        exact_schwarzschild_circular_orbit_transport(&background, radius, vector, 3.7).unwrap();
    let final_norm = metric_contraction(&metric, &transported, &transported);

    assert_close(final_norm, initial_norm, 1.0e-11);
}

#[test]
fn exact_transport_preserves_inner_product_with_four_velocity() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let radius = 12.0;
    let vector = [0.3, -0.2, 0.4, 0.1];
    let four_velocity = schwarzschild_circular_orbit_four_velocity(&background, radius).unwrap();
    let coordinates = [0.0, radius, FRAC_PI_2, 0.0];
    let metric = background.components(&coordinates);
    let initial_inner_product = metric_contraction(&metric, &vector, &four_velocity);

    let transported =
        exact_schwarzschild_circular_orbit_transport(&background, radius, vector, 3.7).unwrap();
    let final_inner_product = metric_contraction(&metric, &transported, &four_velocity);

    assert_close(final_inner_product, initial_inner_product, 1.0e-11);
}

#[test]
fn exact_transport_conserves_polar_component_at_the_equator() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let radius = 9.0;
    let vector = [0.1, 0.2, 0.5, -0.3];

    let transported =
        exact_schwarzschild_circular_orbit_transport(&background, radius, vector, 2.1).unwrap();

    assert_close(transported[2], vector[2], 1.0e-12);
}

#[test]
fn exact_transport_is_identity_at_zero_delta_lambda() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let radius = 15.0;
    let vector = [0.2, -0.1, 0.05, 0.4];

    let transported =
        exact_schwarzschild_circular_orbit_transport(&background, radius, vector, 0.0).unwrap();

    for component in 0..4
    {
        assert_eq!(
            transported[component].to_bits(),
            vector[component].to_bits()
        );
    }
}

#[test]
fn exact_transport_is_deterministic_bit_for_bit() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let radius = 11.0;
    let vector = [0.2, -0.1, 0.05, 0.4];

    let first =
        exact_schwarzschild_circular_orbit_transport(&background, radius, vector, 1.3).unwrap();
    let second =
        exact_schwarzschild_circular_orbit_transport(&background, radius, vector, 1.3).unwrap();

    for component in 0..4
    {
        assert_eq!(first[component].to_bits(), second[component].to_bits());
    }
}

#[test]
fn exact_transport_forward_then_backward_recovers_original_vector() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let radius = 13.0;
    let vector = [0.2, -0.1, 0.05, 0.4];

    let forward =
        exact_schwarzschild_circular_orbit_transport(&background, radius, vector, 2.6).unwrap();
    let back =
        exact_schwarzschild_circular_orbit_transport(&background, radius, forward, -2.6).unwrap();

    for component in 0..4
    {
        assert_close(back[component], vector[component], 1.0e-10);
    }
}

#[test]
fn exact_transport_rejects_radius_at_or_below_three_mass() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let vector = [0.0, 1.0, 0.0, 0.0];

    let result = exact_schwarzschild_circular_orbit_transport(&background, 3.0, vector, 1.0);

    assert!(matches!(
        result,
        Err(NonlocalRelativityError::InvalidCircularOrbitRadius(radius)) if radius == 3.0
    ));
}

#[test]
fn exact_transport_rejects_non_finite_delta_lambda() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let vector = [0.0, 1.0, 0.0, 0.0];

    let result = exact_schwarzschild_circular_orbit_transport(&background, 10.0, vector, f64::NAN);

    assert!(matches!(
        result,
        Err(NonlocalRelativityError::InvalidTransportSegmentStep(_))
    ));
}

fn discrete_transport_error_against_exact_oracle(waypoint_count: usize) -> f64 {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let radius = 10.0;
    let four_velocity = schwarzschild_circular_orbit_four_velocity(&background, radius).unwrap();
    let angular_velocity =
        schwarzschild_circular_orbit_angular_velocity(&background, radius).unwrap();
    let orbit_fraction = 0.02;
    let total_lambda = orbit_fraction * 2.0 * std::f64::consts::PI / four_velocity[3];

    let mut waypoints = Vec::with_capacity(waypoint_count + 1);
    for index in 0..=waypoint_count
    {
        let lambda = total_lambda * index as f64 / waypoint_count as f64;
        let time = four_velocity[0] * lambda;
        let phi = angular_velocity * time;
        let coordinates = [time, radius, FRAC_PI_2, phi];
        waypoints.push(HistoryEntry::new(coordinates, four_velocity, lambda));
    }

    let test_vector = [0.1, 0.4, -0.2, 0.05];

    let numerical = transport_vector_along_polyline(
        &background,
        &DiscreteConnectionTransport,
        test_vector,
        &waypoints,
    )
    .unwrap();
    let exact = exact_schwarzschild_circular_orbit_transport(
        &background,
        radius,
        test_vector,
        total_lambda,
    )
    .unwrap();

    vector_distance(&numerical, &exact)
}

#[test]
fn discrete_transport_converges_to_the_exact_circular_orbit_oracle_under_refinement() {
    let error_coarse = discrete_transport_error_against_exact_oracle(4);
    let error_medium = discrete_transport_error_against_exact_oracle(16);
    let error_fine = discrete_transport_error_against_exact_oracle(64);

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
