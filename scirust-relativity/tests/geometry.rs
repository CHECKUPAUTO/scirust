use scirust_relativity::{
    Connection, GeodesicSystem, Metric, Minkowski, RelativityError, invert_metric, metric_norm,
    numerical_christoffel,
};
use scirust_sim::{System, simulate};

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    let scale = expected.abs().max(1.0);
    let relative_error = (actual - expected).abs() / scale;

    assert!(
        relative_error <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, \
         relative_error={relative_error:.17e}, tolerance={tolerance:.17e}"
    );
}

#[test]
fn minkowski_metric_has_expected_signature() {
    let coordinates = [0.0; 4];
    let metric = Minkowski.components(&coordinates);

    assert_eq!(metric[0][0], -1.0);
    assert_eq!(metric[1][1], 1.0);
    assert_eq!(metric[2][2], 1.0);
    assert_eq!(metric[3][3], 1.0);

    for (row_index, row_values) in metric.iter().enumerate()
    {
        for (column_index, &value) in row_values.iter().enumerate()
        {
            if row_index != column_index
            {
                assert_eq!(value, 0.0);
            }
        }
    }
}

#[test]
fn minkowski_metric_is_its_own_inverse() {
    let metric = Minkowski.components(&[0.0; 4]);
    let inverse = invert_metric(&metric).unwrap();

    assert_eq!(inverse, metric);
}

#[test]
fn minkowski_christoffel_symbols_are_zero() {
    let analytic = Minkowski.christoffel(&[1.0, 2.0, 3.0, 4.0]);
    let numerical = numerical_christoffel(&Minkowski, &[1.0, 2.0, 3.0, 4.0], 1.0e-5).unwrap();

    assert_eq!(analytic, [[[0.0; 4]; 4]; 4]);
    assert_eq!(numerical, [[[0.0; 4]; 4]; 4]);
}

#[test]
fn timelike_norm_matches_minkowski_quadratic_form() {
    let metric = Minkowski.components(&[0.0; 4]);
    let vector = [2.0, 1.0, 0.0, 0.0];

    assert_eq!(metric_norm(&metric, &vector), -3.0);
}

#[test]
fn geodesic_rhs_is_constant_velocity_in_flat_spacetime() {
    let system = GeodesicSystem::<_, 4>::new(Minkowski);
    let state = [1.0, 2.0, 3.0, 4.0, 0.5, -0.25, 0.75, 1.5];
    let mut derivative = [0.0; 8];

    system.derivatives(0.0, &state, &mut derivative);

    assert_eq!(&derivative[..4], &state[4..]);
    assert_eq!(&derivative[4..], &[0.0; 4]);
}

#[test]
fn minkowski_geodesic_integrates_to_a_straight_line() {
    let system = GeodesicSystem::<_, 4>::new(Minkowski);
    let initial = [0.0, 1.0, -2.0, 3.0, 1.0, 0.25, -0.5, 2.0];

    let trajectory = simulate(&system, &initial, 0.0, 4.0, 0.01).unwrap();
    let final_state = trajectory.last_state().unwrap();

    let expected_position = [4.0, 2.0, -4.0, 11.0];

    for index in 0..4
    {
        assert_close(final_state[index], expected_position[index], 2.0e-12);
        assert_close(final_state[4 + index], initial[4 + index], 2.0e-12);
    }
}

#[test]
fn invalid_numerical_connection_inputs_are_rejected() {
    assert_eq!(
        numerical_christoffel(&Minkowski, &[0.0; 4], 0.0),
        Err(RelativityError::InvalidDifferenceStep(0.0))
    );

    assert_eq!(
        numerical_christoffel(&Minkowski, &[0.0, f64::NAN, 0.0, 0.0], 1.0e-5,),
        Err(RelativityError::NonFiniteCoordinate(1))
    );
}

#[test]
fn singular_metric_is_rejected() {
    let singular = [[0.0_f64; 3]; 3];

    assert_eq!(
        invert_metric(&singular),
        Err(RelativityError::SingularMetric)
    );
}
