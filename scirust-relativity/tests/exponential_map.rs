//! Validation of the geodesic exponential and logarithm maps.
//!
//! - Flat spacetime: `exp_p(v) = p + v` and `log_p(q) = q - p` exactly.
//! - Curved spacetimes (Schwarzschild, de Sitter): the maps are local inverses,
//!   `log_p(exp_p(v)) = v` and `exp_p(log_p(q)) = q`, verified as a round-trip
//!   identity.
//! - The maps validate their inputs with typed errors.

use scirust_relativity::{
    DeSitter, Minkowski, RelativityError, Schwarzschild, geodesic_exponential, geodesic_logarithm,
};
use std::f64::consts::FRAC_PI_2;

const STEP: f64 = 0.01;
const JACOBIAN_STEP: f64 = 1.0e-5;
const TOLERANCE: f64 = 1.0e-10;
const MAX_ITERATIONS: usize = 50;

fn assert_close_vector(actual: &[f64; 4], expected: &[f64; 4], tolerance: f64) {
    for i in 0..4
    {
        assert!(
            (actual[i] - expected[i]).abs() <= tolerance,
            "component {i}: actual={:.12e} expected={:.12e}",
            actual[i],
            expected[i]
        );
    }
}

// --------------------------------------------------------------------------
// Flat spacetime: exact
// --------------------------------------------------------------------------

#[test]
fn flat_exponential_is_translation() {
    let position = [0.0, 1.0, 2.0, 0.5];
    let velocity = [1.0, 0.3, -0.2, 0.1];
    let image = geodesic_exponential(&Minkowski, &position, &velocity, 0.05).unwrap();
    let expected = [
        position[0] + velocity[0],
        position[1] + velocity[1],
        position[2] + velocity[2],
        position[3] + velocity[3],
    ];
    assert_close_vector(&image, &expected, 1.0e-12);
}

#[test]
fn flat_logarithm_is_difference() {
    let position = [0.0, 1.0, 2.0, 0.5];
    let target = [2.0, 1.5, 1.0, 0.9];
    let tangent = geodesic_logarithm(
        &Minkowski,
        &position,
        &target,
        0.05,
        JACOBIAN_STEP,
        1.0e-12,
        20,
    )
    .unwrap();
    let expected = [
        target[0] - position[0],
        target[1] - position[1],
        target[2] - position[2],
        target[3] - position[3],
    ];
    assert_close_vector(&tangent, &expected, 1.0e-12);
}

// --------------------------------------------------------------------------
// Curved spacetimes: local inverses (round trip)
// --------------------------------------------------------------------------

fn assert_round_trips<B: scirust_relativity::Connection<4> + Copy>(
    background: &B,
    position: [f64; 4],
    tangent: [f64; 4],
    target: [f64; 4],
) {
    // v -> exp -> log -> v.
    let image = geodesic_exponential(background, &position, &tangent, STEP).unwrap();
    let recovered_tangent = geodesic_logarithm(
        background,
        &position,
        &image,
        STEP,
        JACOBIAN_STEP,
        TOLERANCE,
        MAX_ITERATIONS,
    )
    .unwrap();
    assert_close_vector(&recovered_tangent, &tangent, 1.0e-8);

    // q -> log -> exp -> q.
    let recovered_tangent_for_target = geodesic_logarithm(
        background,
        &position,
        &target,
        STEP,
        JACOBIAN_STEP,
        TOLERANCE,
        MAX_ITERATIONS,
    )
    .unwrap();
    let recovered_target =
        geodesic_exponential(background, &position, &recovered_tangent_for_target, STEP).unwrap();
    assert_close_vector(&recovered_target, &target, 1.0e-8);
}

#[test]
fn exp_log_round_trip_schwarzschild() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    assert_round_trips(
        &background,
        [0.0, 12.0, FRAC_PI_2, 0.0],
        [0.2, 0.15, 0.03, 0.02],
        [0.15, 12.4, FRAC_PI_2 + 0.03, 0.05],
    );
}

#[test]
fn exp_log_round_trip_de_sitter() {
    let background = DeSitter::try_new(0.05).unwrap();
    assert_round_trips(
        &background,
        [0.0, 3.0, FRAC_PI_2, 0.0],
        [0.2, 0.1, 0.05, 0.02],
        [0.1, 3.3, FRAC_PI_2 + 0.04, 0.03],
    );
}

// --------------------------------------------------------------------------
// Error paths
// --------------------------------------------------------------------------

#[test]
fn exponential_and_logarithm_report_typed_errors() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let position = [0.0, 10.0, FRAC_PI_2, 0.0];
    let velocity = [0.1, 0.1, 0.0, 0.0];

    assert_eq!(
        geodesic_exponential(&background, &position, &velocity, 0.0),
        Err(RelativityError::InvalidDifferenceStep(0.0)),
    );
    assert_eq!(
        geodesic_exponential(
            &background,
            &[f64::NAN, 10.0, FRAC_PI_2, 0.0],
            &velocity,
            0.01,
        ),
        Err(RelativityError::NonFiniteCoordinate(0)),
    );

    // A one-iteration cap cannot reach a 1e-10 tolerance on a curved background.
    assert_eq!(
        geodesic_logarithm(
            &background,
            &position,
            &[0.1, 10.4, FRAC_PI_2 + 0.05, 0.03],
            STEP,
            JACOBIAN_STEP,
            1.0e-10,
            1,
        ),
        Err(RelativityError::LogarithmMapDidNotConverge),
    );
}
