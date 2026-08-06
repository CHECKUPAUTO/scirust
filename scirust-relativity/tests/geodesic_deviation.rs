//! Validation of the geodesic-deviation (Jacobi) integrator.
//!
//! The Jacobi field is validated against a **convention-free ground truth**:
//! the actual coordinate separation of two nearby geodesics,
//! `xi(tau) = [x(+eps) - x(-eps)] / (2 eps)`, integrated with the same geodesic
//! solver. Agreement fixes the sign and index convention of the Riemann source
//! in the Jacobi equation without appeal to a hand-derived formula.
//!
//! - Flat spacetime: `xi(tau) = xi_0 + tau xi_dot_0` exactly.
//! - de Sitter and Schwarzschild: the Jacobi field reproduces the geodesic-flow
//!   separation to the integration tolerance.
//! - The Jacobi map is linear in its initial data, and inputs are validated
//!   with typed errors.

use scirust_relativity::{
    Connection, DeSitter, GeodesicSystem, Minkowski, RelativityError, Schwarzschild,
    integrate_geodesic_deviation,
};
use scirust_sim::simulate;
use std::f64::consts::FRAC_PI_2;

const CURVATURE_STEP: f64 = 1.0e-4;
const FLOW_EPSILON: f64 = 1.0e-4;

/// Ground-truth Jacobi field: the central-difference separation of two nearby
/// geodesics, integrated with the ordinary geodesic solver.
fn flow_deviation<B: Connection<4> + Copy>(
    background: B,
    position: [f64; 4],
    velocity: [f64; 4],
    deviation: [f64; 4],
    deviation_velocity: [f64; 4],
    affine_length: f64,
    step: f64,
) -> [f64; 4] {
    let mut plus = [0.0; 8];
    let mut minus = [0.0; 8];
    for i in 0..4
    {
        plus[i] = position[i] + FLOW_EPSILON * deviation[i];
        plus[i + 4] = velocity[i] + FLOW_EPSILON * deviation_velocity[i];
        minus[i] = position[i] - FLOW_EPSILON * deviation[i];
        minus[i + 4] = velocity[i] - FLOW_EPSILON * deviation_velocity[i];
    }
    let system = GeodesicSystem::<_, 4>::new(background);
    let forward = simulate(&system, &plus, 0.0, affine_length, step).unwrap();
    let backward = simulate(&system, &minus, 0.0, affine_length, step).unwrap();
    let forward_end = forward.last_state().unwrap();
    let backward_end = backward.last_state().unwrap();
    let mut result = [0.0; 4];
    for i in 0..4
    {
        result[i] = (forward_end[i] - backward_end[i]) / (2.0 * FLOW_EPSILON);
    }
    result
}

fn max_component(vector: &[f64; 4]) -> f64 {
    vector.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

fn assert_matches_flow(jacobi: &[f64; 4], flow: &[f64; 4]) {
    let scale = max_component(flow);
    for i in 0..4
    {
        let tolerance = 1.0e-5 * scale + 1.0e-9;
        assert!(
            (jacobi[i] - flow[i]).abs() <= tolerance,
            "component {i}: jacobi={:.9e} flow={:.9e}",
            jacobi[i],
            flow[i]
        );
    }
}

// --------------------------------------------------------------------------
// Flat spacetime: exact linear growth
// --------------------------------------------------------------------------

#[test]
fn flat_deviation_grows_linearly() {
    let position = [0.0, 1.0, 2.0, 0.5];
    let velocity = [1.0, 0.1, 0.0, 0.0];
    let deviation = [0.0, 0.3, 0.2, 0.1];
    let deviation_velocity = [0.0, 0.05, -0.02, 0.0];
    let affine_length = 2.0;

    let samples = integrate_geodesic_deviation(
        &Minkowski,
        &position,
        &velocity,
        &deviation,
        &deviation_velocity,
        affine_length,
        0.01,
        CURVATURE_STEP,
    )
    .unwrap();

    // First sample is the initial data; last is at the affine length.
    assert_eq!(samples.first().unwrap().affine_parameter, 0.0);
    assert_eq!(samples.first().unwrap().deviation, deviation);
    assert!((samples.last().unwrap().affine_parameter - affine_length).abs() < 1.0e-12);

    let final_deviation = samples.last().unwrap().deviation;
    for i in 0..4
    {
        let expected = deviation[i] + affine_length * deviation_velocity[i];
        assert!((final_deviation[i] - expected).abs() < 1.0e-9);
    }
}

// --------------------------------------------------------------------------
// Curved spacetimes: agree with the geodesic-flow ground truth
// --------------------------------------------------------------------------

#[test]
fn jacobi_matches_geodesic_flow_de_sitter() {
    let background = DeSitter::try_new(0.05).unwrap();
    let position = [0.0, 3.0, FRAC_PI_2, 0.0];
    let velocity = [1.0, 0.0, 0.0, 0.05];
    let deviation = [0.0, 0.1, 0.05, 0.0];
    let deviation_velocity = [0.0, 0.0, 0.0, 0.0];
    let affine_length = 0.5;
    let step = 0.0025;

    let samples = integrate_geodesic_deviation(
        &background,
        &position,
        &velocity,
        &deviation,
        &deviation_velocity,
        affine_length,
        step,
        CURVATURE_STEP,
    )
    .unwrap();
    let flow = flow_deviation(
        background,
        position,
        velocity,
        deviation,
        deviation_velocity,
        affine_length,
        step,
    );
    assert_matches_flow(&samples.last().unwrap().deviation, &flow);
}

#[test]
fn jacobi_matches_geodesic_flow_schwarzschild() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let position = [0.0, 12.0, FRAC_PI_2, 0.0];
    let velocity = [1.05, 0.0, 0.0, 0.02];
    let deviation = [0.0, 0.1, 0.05, 0.0];
    let deviation_velocity = [0.0, 0.0, 0.0, 0.01];
    let affine_length = 1.0;
    let step = 0.005;

    let samples = integrate_geodesic_deviation(
        &background,
        &position,
        &velocity,
        &deviation,
        &deviation_velocity,
        affine_length,
        step,
        CURVATURE_STEP,
    )
    .unwrap();
    let flow = flow_deviation(
        background,
        position,
        velocity,
        deviation,
        deviation_velocity,
        affine_length,
        step,
    );
    assert_matches_flow(&samples.last().unwrap().deviation, &flow);
}

// --------------------------------------------------------------------------
// Linearity and error paths
// --------------------------------------------------------------------------

#[test]
fn jacobi_map_is_linear_in_initial_data() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let position = [0.0, 10.0, FRAC_PI_2, 0.0];
    let velocity = [1.05, 0.0, 0.0, 0.02];
    let affine_length = 1.0;
    let step = 0.01;

    let end = |deviation: [f64; 4], deviation_velocity: [f64; 4]| {
        integrate_geodesic_deviation(
            &background,
            &position,
            &velocity,
            &deviation,
            &deviation_velocity,
            affine_length,
            step,
            CURVATURE_STEP,
        )
        .unwrap()
        .last()
        .unwrap()
        .deviation
    };

    let first = end([0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.02, 0.0]);
    let second = end([0.0, 0.0, 0.05, 0.0], [0.0, 0.01, 0.0, 0.0]);
    let (a, b) = (1.5, -0.7);
    let combined = end(
        [0.0, a * 0.1, b * 0.05, 0.0],
        [0.0, b * 0.01, a * 0.02, 0.0],
    );

    for i in 0..4
    {
        let linear = a * first[i] + b * second[i];
        assert!((combined[i] - linear).abs() < 1.0e-10);
    }
}

#[test]
fn geodesic_deviation_reports_typed_errors() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let position = [0.0, 10.0, FRAC_PI_2, 0.0];
    let velocity = [1.05, 0.0, 0.0, 0.02];
    let deviation = [0.0, 0.1, 0.0, 0.0];
    let zero = [0.0; 4];

    assert_eq!(
        integrate_geodesic_deviation(
            &background,
            &position,
            &velocity,
            &deviation,
            &zero,
            1.0,
            0.0,
            CURVATURE_STEP,
        ),
        Err(RelativityError::InvalidDifferenceStep(0.0)),
    );
    assert_eq!(
        integrate_geodesic_deviation(
            &background,
            &position,
            &velocity,
            &deviation,
            &zero,
            0.0,
            0.01,
            CURVATURE_STEP,
        ),
        Err(RelativityError::InvalidAffineLength(0.0)),
    );
    assert_eq!(
        integrate_geodesic_deviation(
            &background,
            &position,
            &velocity,
            &[f64::NAN, 0.0, 0.0, 0.0],
            &zero,
            1.0,
            0.01,
            CURVATURE_STEP,
        ),
        Err(RelativityError::NonFiniteDeviationVector),
    );
}
