//! Exact-oracle validation of the spatially flat FLRW background.
//!
//! - Exponential scale factor `a = exp(H t)`: de Sitter space, so
//!   `R = 12 H^2`, `R_(mu nu) = 3 H^2 g_(mu nu)`, `K = 24 H^4`, and its
//!   Kretschmann scalar agrees with static-chart [`DeSitter`] at
//!   `Lambda = 3 H^2` (coordinate independence).
//! - Power-law scale factor `a = (t/t_ref)^p`: a genuinely time-dependent
//!   geometry whose Ricci scalar and Kretschmann invariant match the general
//!   flat-FLRW (Friedmann) formulas in `a`, `a_dot`, `a_ddot`.

use scirust_relativity::{
    CurvatureTensors, DeSitter, ExponentialScaleFactor, Flrw, Metric, PowerLawScaleFactor,
    ScaleFactor,
};
use std::f64::consts::FRAC_PI_2;

const STEP: f64 = 1.0e-4;

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    let scale = expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() / scale <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}"
    );
}

fn relative(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs()
}

// --------------------------------------------------------------------------
// Scale factors
// --------------------------------------------------------------------------

#[test]
fn scale_factor_constructors_validate_inputs() {
    assert!(ExponentialScaleFactor::try_new(0.5).is_some());
    assert!(ExponentialScaleFactor::try_new(0.0).is_none());
    assert!(ExponentialScaleFactor::try_new(-1.0).is_none());
    assert!(ExponentialScaleFactor::try_new(f64::NAN).is_none());

    assert!(PowerLawScaleFactor::try_new(2.0 / 3.0, 1.0).is_some());
    assert!(PowerLawScaleFactor::try_new(0.5, 0.0).is_none());
    assert!(PowerLawScaleFactor::try_new(0.5, -1.0).is_none());
    assert!(PowerLawScaleFactor::try_new(f64::INFINITY, 1.0).is_none());
}

#[test]
fn scale_factor_values_and_derivatives() {
    let exponential = ExponentialScaleFactor::try_new(0.5).unwrap();
    assert_close(exponential.value(0.0), 1.0, 0.0);
    assert_close(exponential.first_derivative(0.0), 0.5, 1.0e-15);
    assert_close(exponential.second_derivative(0.0), 0.25, 1.0e-15);
    assert_close(exponential.cosmological_constant(), 0.75, 1.0e-15);

    // a = (t / 2)^(1/2): a(2) = 1, a_dot = (1/2)(1/t)a.
    let power = PowerLawScaleFactor::try_new(0.5, 2.0).unwrap();
    assert_close(power.value(2.0), 1.0, 1.0e-15);
    assert_close(power.first_derivative(2.0), 0.25, 1.0e-15);
}

#[test]
fn flrw_metric_and_hubble_parameter() {
    let hubble = 0.4;
    let background = Flrw::new(ExponentialScaleFactor::try_new(hubble).unwrap());
    let time = 1.3;
    let scale = (hubble * time).exp();

    let metric = background.components(&[time, 0.2, -0.1, 0.5]);
    assert_close(metric[0][0], -1.0, 0.0);
    assert_close(metric[1][1], scale * scale, 1.0e-12);
    assert_close(metric[2][2], scale * scale, 1.0e-12);
    assert_close(metric[3][3], scale * scale, 1.0e-12);

    // Exponential scale factor has a constant Hubble parameter H.
    assert_close(background.hubble_parameter(time), hubble, 1.0e-12);
}

// --------------------------------------------------------------------------
// Exponential FLRW == de Sitter
// --------------------------------------------------------------------------

#[test]
fn exponential_flrw_is_de_sitter() {
    let hubble = 0.5;
    let lambda = 3.0 * hubble * hubble; // 0.75
    let background = Flrw::new(ExponentialScaleFactor::try_new(hubble).unwrap());

    for &time in &[-0.5, 0.0, 1.0, 2.0]
    {
        let coordinates = [time, 0.1, -0.2, 0.3];
        let tensors = CurvatureTensors::compute(&background, &coordinates, STEP).unwrap();
        let scale = (hubble * time).exp();

        // Maximally symmetric: R = 4 Lambda, K = 8 Lambda^2 / 3.
        assert_close(tensors.ricci_scalar(), 4.0 * lambda, 1.0e-6);
        assert_close(tensors.kretschmann(), 8.0 * lambda * lambda / 3.0, 1.0e-6);

        // Ricci = Lambda g, Einstein = -Lambda g.
        let metric = background.components(&coordinates);
        for (mu, row) in tensors.ricci().iter().enumerate()
        {
            for (nu, &value) in row.iter().enumerate()
            {
                assert!((value - lambda * metric[mu][nu]).abs() < 1.0e-6);
            }
        }
        for (mu, row) in tensors.einstein().iter().enumerate()
        {
            for (nu, &value) in row.iter().enumerate()
            {
                assert!((value + lambda * metric[mu][nu]).abs() < 1.0e-6);
            }
        }

        // Spot-check the closed-form components explicitly.
        assert_close(tensors.ricci()[0][0], -lambda, 1.0e-6);
        assert_close(tensors.ricci()[1][1], lambda * scale * scale, 1.0e-6);
    }
}

#[test]
fn flrw_de_sitter_kretschmann_agrees_with_static_chart() {
    // Coordinate independence: de Sitter in flat FLRW slicing (parameter H) and
    // in the static chart (parameter Lambda = 3 H^2) are the same geometry, so
    // their Kretschmann scalars must agree.
    let hubble = 0.3;
    let lambda = 3.0 * hubble * hubble;
    let flrw = Flrw::new(ExponentialScaleFactor::try_new(hubble).unwrap());
    let static_chart = DeSitter::try_new(lambda).unwrap();

    let flrw_k = CurvatureTensors::compute(&flrw, &[0.7, 0.0, 0.0, 0.0], STEP)
        .unwrap()
        .kretschmann();
    let static_k = CurvatureTensors::compute(&static_chart, &[0.0, 3.0, FRAC_PI_2, 0.0], STEP)
        .unwrap()
        .kretschmann();

    let oracle = 8.0 * lambda * lambda / 3.0;
    assert!(relative(flrw_k, oracle) < 1.0e-5);
    assert!(relative(static_k, oracle) < 1.0e-5);
    assert!(relative(flrw_k, static_k) < 1.0e-5);
}

// --------------------------------------------------------------------------
// Power-law FLRW: general Friedmann oracle (time-dependent curvature)
// --------------------------------------------------------------------------

#[test]
fn power_law_flrw_matches_friedmann_oracle() {
    // Radiation- (p = 1/2) and matter- (p = 2/3) dominated expansion.
    for &exponent in &[0.5, 2.0 / 3.0]
    {
        let scale_factor = PowerLawScaleFactor::try_new(exponent, 1.0).unwrap();
        let background = Flrw::new(scale_factor);

        for &time in &[1.5, 2.0, 4.0]
        {
            let tensors =
                CurvatureTensors::compute(&background, &[time, 0.0, 0.0, 0.0], STEP).unwrap();

            let scale = scale_factor.value(time);
            let rate = scale_factor.first_derivative(time);
            let acceleration = scale_factor.second_derivative(time);
            let acceleration_ratio = acceleration / scale;
            let hubble = rate / scale;

            // R = 6 (a_ddot/a + (a_dot/a)^2), K = 12 ((a_ddot/a)^2 + (a_dot/a)^4).
            // Radiation (p = 1/2) is traceless, R = 0 exactly, so the Ricci
            // scalar is compared with an absolute floor rather than a ratio.
            let ricci_oracle = 6.0 * (acceleration_ratio + hubble * hubble);
            let kretschmann_oracle =
                12.0 * (acceleration_ratio * acceleration_ratio + hubble.powi(4));

            assert_close(tensors.ricci_scalar(), ricci_oracle, 1.0e-6);
            assert!(relative(tensors.kretschmann(), kretschmann_oracle) < 1.0e-6);

            // The geometry is genuinely curved and time-dependent (not flat):
            // the Kretschmann scalar is a strictly positive, varying quantity
            // even where the Ricci scalar vanishes.
            assert!(tensors.kretschmann() > 1.0e-6);
        }
    }
}
