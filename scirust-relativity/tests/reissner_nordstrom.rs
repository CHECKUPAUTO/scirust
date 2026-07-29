use scirust_relativity::{
    Connection, Metric, ReissnerNordstrom, Schwarzschild, numerical_christoffel,
};
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
    assert!(ReissnerNordstrom::try_new(1.0, 0.0).is_some());
    assert!(ReissnerNordstrom::try_new(1.0, 0.5).is_some());
    assert!(ReissnerNordstrom::try_new(1.0, -0.5).is_some());
    assert!(ReissnerNordstrom::try_new(0.0, 0.0).is_none());
    assert!(ReissnerNordstrom::try_new(-1.0, 0.0).is_none());
    assert!(ReissnerNordstrom::try_new(f64::NAN, 0.0).is_none());
    assert!(ReissnerNordstrom::try_new(1.0, f64::NAN).is_none());
    assert!(ReissnerNordstrom::try_new(1.0, f64::INFINITY).is_none());
    // Extremal and super-extremal charges are rejected.
    assert!(ReissnerNordstrom::try_new(1.0, 1.0).is_none());
    assert!(ReissnerNordstrom::try_new(1.0, 1.5).is_none());
}

#[test]
fn zero_charge_metric_matches_schwarzschild_exactly() {
    let mass = 1.3;
    let charged = ReissnerNordstrom::try_new(mass, 0.0).unwrap();
    let neutral = Schwarzschild::try_new(mass).unwrap();
    let coordinates = [0.4, 9.0, 1.1, -0.3];

    let charged_metric = charged.components(&coordinates);
    let neutral_metric = neutral.components(&coordinates);

    for row in 0..4
    {
        for column in 0..4
        {
            assert_eq!(
                charged_metric[row][column].to_bits(),
                neutral_metric[row][column].to_bits()
            );
        }
    }
}

#[test]
fn zero_charge_christoffel_matches_schwarzschild_to_machine_precision() {
    // The two implementations compute the shared f(r)-metric formula through
    // different intermediate expressions (Schwarzschild folds mass and
    // radius directly; Reissner-Nordström goes through a separate `lapse`
    // and `lapse_derivative`), so results agree to machine precision rather
    // than bit-for-bit.
    let mass = 1.3;
    let charged = ReissnerNordstrom::try_new(mass, 0.0).unwrap();
    let neutral = Schwarzschild::try_new(mass).unwrap();
    let coordinates = [0.4, 9.0, 1.1, -0.3];

    let charged_symbols = charged.christoffel(&coordinates);
    let neutral_symbols = neutral.christoffel(&coordinates);

    for (rho, rho_values) in charged_symbols.iter().enumerate()
    {
        for (mu, mu_values) in rho_values.iter().enumerate()
        {
            for (nu, &value) in mu_values.iter().enumerate()
            {
                let expected = neutral_symbols[rho][mu][nu];
                if expected == 0.0
                {
                    assert_eq!(value, 0.0);
                }
                else
                {
                    assert_close(value, expected, 1.0e-14);
                }
            }
        }
    }
}

#[test]
fn zero_charge_outer_horizon_matches_schwarzschild_horizon() {
    let mass = 2.0;
    let charged = ReissnerNordstrom::try_new(mass, 0.0).unwrap();
    let neutral = Schwarzschild::try_new(mass).unwrap();

    assert_close(
        charged.outer_horizon_radius(),
        neutral.horizon_radius(),
        1.0e-15,
    );
}

#[test]
fn outer_horizon_shrinks_with_increasing_charge() {
    let mass = 1.0;
    let uncharged = ReissnerNordstrom::try_new(mass, 0.0).unwrap();
    let charged = ReissnerNordstrom::try_new(mass, 0.8).unwrap();

    assert!(charged.outer_horizon_radius() < uncharged.outer_horizon_radius());
    assert!(charged.outer_horizon_radius() > mass);
}

#[test]
fn exterior_domain_excludes_horizon_interior_and_polar_axis() {
    let spacetime = ReissnerNordstrom::try_new(2.0, 1.0).unwrap();
    let horizon = spacetime.outer_horizon_radius();

    assert!(spacetime.is_in_exterior(&[0.0, horizon + 1.0, FRAC_PI_2, 0.0]));
    assert!(!spacetime.is_in_exterior(&[0.0, horizon - 0.5, FRAC_PI_2, 0.0]));
    assert!(!spacetime.is_in_exterior(&[0.0, horizon, FRAC_PI_2, 0.0]));
    assert!(!spacetime.is_in_exterior(&[0.0, horizon + 1.0, 0.0, 0.0]));
    assert!(!spacetime.is_in_exterior(&[0.0, horizon + 1.0, std::f64::consts::PI, 0.0]));
}

#[test]
fn metric_matches_hand_derived_values() {
    let spacetime = ReissnerNordstrom::try_new(1.0, 0.5).unwrap();
    let radius = 4.0;
    let metric = spacetime.components(&[0.0, radius, FRAC_PI_2, 0.0]);

    // f(4) = 1 - 2*1/4 + 0.25/16 = 1 - 0.5 + 0.015625 = 0.515625
    let expected_lapse = 0.515_625;
    assert_close(metric[0][0], -expected_lapse, 1.0e-14);
    assert_close(metric[1][1], 1.0 / expected_lapse, 1.0e-14);
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
    let spacetime = ReissnerNordstrom::try_new(1.25, 0.6).unwrap();
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
fn christoffel_symbols_are_finite_in_the_exterior() {
    let spacetime = ReissnerNordstrom::try_new(1.0, 0.7).unwrap();
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
    let spacetime = ReissnerNordstrom::try_new(2.5, 1.1).unwrap();

    assert_close(spacetime.mass(), 2.5, 0.0);
    assert_close(spacetime.charge(), 1.1, 0.0);
}
