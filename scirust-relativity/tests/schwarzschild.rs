use scirust_relativity::{
    Connection, GeodesicSystem, Metric, Schwarzschild, numerical_christoffel,
};
use scirust_sim::System;
use std::f64::consts::{FRAC_PI_2, PI};

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
fn constructor_rejects_non_positive_or_non_finite_mass() {
    assert!(Schwarzschild::try_new(1.0).is_some());
    assert!(Schwarzschild::try_new(0.0).is_none());
    assert!(Schwarzschild::try_new(-1.0).is_none());
    assert!(Schwarzschild::try_new(f64::NAN).is_none());
    assert!(Schwarzschild::try_new(f64::INFINITY).is_none());
}

#[test]
fn exterior_domain_excludes_horizon_interior_and_polar_axis() {
    let spacetime = Schwarzschild::try_new(2.0).unwrap();

    assert!(spacetime.is_in_exterior(&[0.0, 5.0, FRAC_PI_2, 0.0]));
    assert!(!spacetime.is_in_exterior(&[0.0, 4.0, FRAC_PI_2, 0.0]));
    assert!(!spacetime.is_in_exterior(&[0.0, 3.0, FRAC_PI_2, 0.0]));
    assert!(!spacetime.is_in_exterior(&[0.0, 5.0, 0.0, 0.0]));
    assert!(!spacetime.is_in_exterior(&[0.0, 5.0, PI, 0.0]));
}

#[test]
fn metric_matches_hand_derived_values_at_four_mass_units() {
    let spacetime = Schwarzschild::try_new(1.0).unwrap();
    let metric = spacetime.components(&[0.0, 4.0, FRAC_PI_2, 0.0]);

    assert_close(metric[0][0], -0.5, 0.0);
    assert_close(metric[1][1], 2.0, 0.0);
    assert_close(metric[2][2], 16.0, 0.0);
    assert_close(metric[3][3], 16.0, 1.0e-15);

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
fn analytic_connection_matches_central_finite_differences() {
    let spacetime = Schwarzschild::try_new(1.25).unwrap();
    let coordinates = [0.7, 12.0, 1.1, -0.4];

    let analytic = spacetime.christoffel(&coordinates);
    let numerical = numerical_christoffel(&spacetime, &coordinates, 1.0e-5).unwrap();

    for (rho, rho_values) in analytic.iter().enumerate()
    {
        for (mu, mu_values) in rho_values.iter().enumerate()
        {
            for (nu, &value) in mu_values.iter().enumerate()
            {
                assert_close(numerical[rho][mu][nu], value, 2.0e-8);
            }
        }
    }
}

#[test]
fn initially_static_particle_has_inward_radial_acceleration() {
    let spacetime = Schwarzschild::try_new(1.0).unwrap();
    let system = GeodesicSystem::<_, 4>::new(spacetime);

    let state = [0.0, 10.0, FRAC_PI_2, 0.0, 1.0, 0.0, 0.0, 0.0];
    let mut derivative = [0.0_f64; 8];

    system.derivatives(0.0, &state, &mut derivative);

    assert_close(derivative[5], -0.008, 1.0e-15);
}

#[test]
fn circular_equatorial_orbit_has_zero_initial_radial_acceleration() {
    let mass = 1.0;
    let radius = 10.0;
    let spacetime = Schwarzschild::try_new(mass).unwrap();
    let system = GeodesicSystem::<_, 4>::new(spacetime);

    let normalization = 1.0 - 3.0 * mass / radius;
    let time_velocity = 1.0 / f64::sqrt(normalization);
    let angular_velocity = f64::sqrt(mass / (radius * radius * radius * normalization));

    let state = [
        0.0,
        radius,
        FRAC_PI_2,
        0.0,
        time_velocity,
        0.0,
        0.0,
        angular_velocity,
    ];
    let mut derivative = [0.0_f64; 8];

    system.derivatives(0.0, &state, &mut derivative);

    assert_close(derivative[5], 0.0, 2.0e-17);
}
