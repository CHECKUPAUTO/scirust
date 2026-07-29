use scirust_relativity::{Connection, Kerr, Metric, Schwarzschild};
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

#[test]
fn constructor_rejects_invalid_or_extremal_parameters() {
    assert!(Kerr::try_new(1.0, 0.0).is_some());
    assert!(Kerr::try_new(1.0, 0.5).is_some());
    assert!(Kerr::try_new(1.0, -0.5).is_some());
    assert!(Kerr::try_new(0.0, 0.0).is_none());
    assert!(Kerr::try_new(-1.0, 0.0).is_none());
    assert!(Kerr::try_new(f64::NAN, 0.0).is_none());
    assert!(Kerr::try_new(1.0, f64::NAN).is_none());
    assert!(Kerr::try_new(1.0, f64::INFINITY).is_none());
    // Extremal and super-extremal spins are rejected.
    assert!(Kerr::try_new(1.0, 1.0).is_none());
    assert!(Kerr::try_new(1.0, -1.0).is_none());
    assert!(Kerr::try_new(1.0, 1.5).is_none());
}

#[test]
fn zero_spin_metric_matches_schwarzschild_exactly() {
    // Value equality, not bit-pattern equality: the off-diagonal Kerr term
    // `-2*M*spin*r*sin^2(theta)/Sigma` evaluates to signed negative zero at
    // `spin = 0.0` (`-2.0 * 0.0 = -0.0` in IEEE 754), which compares equal
    // to Schwarzschild's literal positive zero but has a different bit
    // pattern; that is expected floating-point behavior, not a defect.
    let mass = 1.3;
    let rotating = Kerr::try_new(mass, 0.0).unwrap();
    let static_case = Schwarzschild::try_new(mass).unwrap();
    let coordinates = [0.4, 9.0, 1.1, -0.3];

    let rotating_metric = rotating.components(&coordinates);
    let static_metric = static_case.components(&coordinates);

    for row in 0..4
    {
        for column in 0..4
        {
            assert_eq!(rotating_metric[row][column], static_metric[row][column]);
        }
    }
}

#[test]
fn zero_spin_christoffel_matches_schwarzschild_to_finite_difference_tolerance() {
    let mass = 1.3;
    let rotating = Kerr::try_new(mass, 0.0).unwrap();
    let static_case = Schwarzschild::try_new(mass).unwrap();
    let coordinates = [0.4, 9.0, 1.1, -0.3];

    let rotating_symbols = rotating.christoffel(&coordinates);
    let static_symbols = static_case.christoffel(&coordinates);

    for (rho, rho_values) in rotating_symbols.iter().enumerate()
    {
        for (mu, mu_values) in rho_values.iter().enumerate()
        {
            for (nu, &value) in mu_values.iter().enumerate()
            {
                let expected = static_symbols[rho][mu][nu];
                if expected == 0.0
                {
                    assert!(
                        value.abs() < 1.0e-6,
                        "Gamma^{rho}_({mu} {nu}) expected ~0, got {value:e}"
                    );
                }
                else
                {
                    assert_close(value, expected, 1.0e-6);
                }
            }
        }
    }
}

#[test]
fn zero_spin_outer_horizon_matches_schwarzschild_horizon() {
    let mass = 2.0;
    let rotating = Kerr::try_new(mass, 0.0).unwrap();
    let static_case = Schwarzschild::try_new(mass).unwrap();

    assert_close(
        rotating.outer_horizon_radius(),
        static_case.horizon_radius(),
        1.0e-15,
    );
}

#[test]
fn outer_horizon_depends_only_on_spin_magnitude() {
    let mass = 1.0;
    let positive_spin = Kerr::try_new(mass, 0.6).unwrap();
    let negative_spin = Kerr::try_new(mass, -0.6).unwrap();
    let unspun = Kerr::try_new(mass, 0.0).unwrap();

    assert_close(
        positive_spin.outer_horizon_radius(),
        negative_spin.outer_horizon_radius(),
        1.0e-15,
    );
    assert!(positive_spin.outer_horizon_radius() < unspun.outer_horizon_radius());
    assert!(positive_spin.outer_horizon_radius() > mass);
}

#[test]
fn exterior_domain_excludes_horizon_interior_and_polar_axis() {
    let spacetime = Kerr::try_new(2.0, 1.0).unwrap();
    let horizon = spacetime.outer_horizon_radius();

    assert!(spacetime.is_in_exterior(&[0.0, horizon + 1.0, FRAC_PI_2, 0.0]));
    assert!(!spacetime.is_in_exterior(&[0.0, horizon - 0.5, FRAC_PI_2, 0.0]));
    assert!(!spacetime.is_in_exterior(&[0.0, horizon, FRAC_PI_2, 0.0]));
    assert!(!spacetime.is_in_exterior(&[0.0, horizon + 1.0, 0.0, 0.0]));
    assert!(!spacetime.is_in_exterior(&[0.0, horizon + 1.0, std::f64::consts::PI, 0.0]));
}

#[test]
fn metric_is_symmetric() {
    let spacetime = Kerr::try_new(1.0, 0.7).unwrap();
    let coordinates = [0.3, 8.0, 1.0, 0.6];
    let metric = spacetime.components(&coordinates);

    for (row, row_values) in metric.iter().enumerate()
    {
        for (column, &value) in row_values.iter().enumerate()
        {
            assert_eq!(
                value.to_bits(),
                metric[column][row].to_bits(),
                "metric not symmetric at ({row}, {column})"
            );
        }
    }
}

#[test]
fn metric_matches_hand_derived_values_in_the_equatorial_plane() {
    // theta = pi/2 => cos(theta) = 0 => Sigma = r^2 exactly.
    let mass = 1.0;
    let spin = 0.5;
    let radius = 10.0;
    let spacetime = Kerr::try_new(mass, spin).unwrap();
    let metric = spacetime.components(&[0.0, radius, FRAC_PI_2, 0.0]);

    let sigma = radius * radius;
    let delta = radius * radius - 2.0 * mass * radius + spin * spin;
    let expected_time_time = -(1.0 - 2.0 * mass * radius / sigma);
    let expected_time_phi = -2.0 * mass * spin * radius / sigma;
    let expected_radial_radial = sigma / delta;
    let expected_polar_polar = sigma;
    let expected_azimuthal_azimuthal =
        radius * radius + spin * spin + 2.0 * mass * spin * spin * radius / sigma;

    assert_close(metric[0][0], expected_time_time, 1.0e-14);
    assert_close(metric[0][3], expected_time_phi, 1.0e-14);
    assert_close(metric[3][0], expected_time_phi, 1.0e-14);
    assert_close(metric[1][1], expected_radial_radial, 1.0e-14);
    assert_close(metric[2][2], expected_polar_polar, 1.0e-14);
    assert_close(metric[3][3], expected_azimuthal_azimuthal, 1.0e-14);
    assert_eq!(metric[0][1], 0.0);
    assert_eq!(metric[0][2], 0.0);
    assert_eq!(metric[1][2], 0.0);
    assert_eq!(metric[1][3], 0.0);
    assert_eq!(metric[2][3], 0.0);
}

#[test]
fn frame_dragging_term_has_the_expected_sign() {
    // A positive spin should drag co-rotating: the ZAMO angular velocity
    // omega = -g_tphi/g_phiphi must be positive for a > 0 outside the
    // horizon, matching the Lense-Thirring precession direction.
    let spacetime = Kerr::try_new(1.0, 0.8).unwrap();
    let coordinates = [0.0, 15.0, FRAC_PI_2, 0.0];
    let metric = spacetime.components(&coordinates);

    let zamo_angular_velocity = -metric[0][3] / metric[3][3];
    assert!(
        zamo_angular_velocity > 0.0,
        "expected positive (co-rotating) frame dragging, got {zamo_angular_velocity:e}"
    );
}

#[test]
fn christoffel_symbols_are_finite_in_the_exterior() {
    let spacetime = Kerr::try_new(1.0, 0.7).unwrap();
    let horizon = spacetime.outer_horizon_radius();
    let coordinates = [0.0, horizon + 3.0, 1.2, 0.5];

    let symbols = spacetime.christoffel(&coordinates);

    for rho_values in symbols
    {
        for mu_values in rho_values
        {
            for value in mu_values
            {
                assert!(value.is_finite());
            }
        }
    }
}

#[test]
fn accessors_return_constructed_parameters() {
    let spacetime = Kerr::try_new(2.5, 0.9).unwrap();

    assert_close(spacetime.mass(), 2.5, 0.0);
    assert_close(spacetime.spin(), 0.9, 0.0);
}
