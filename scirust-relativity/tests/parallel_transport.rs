//! Validation of the parallel-transport engine against exact identities.
//!
//! - Flat Cartesian spacetime: transport is the exact identity.
//! - Metric compatibility: the metric inner product of a transported vector is
//!   preserved along the path (to RK4 tolerance).
//! - Flat curvilinear coordinates: transport around a closed loop has zero
//!   holonomy.
//! - Curved spacetime: the holonomy defect around a small loop equals
//!   `-R^rho_(sigma mu nu) V^sigma A^mu B^nu` to leading order in the loop size
//!   (checked against the numerical Riemann tensor from [`CurvatureTensors`]),
//!   cross-validating the transport engine and the curvature engine.
//! - Linearity and typed error paths.

use scirust_relativity::{
    Connection, CurvatureTensors, DeSitter, Metric, Minkowski, MinkowskiSpherical, RelativityError,
    Schwarzschild, holonomy_defect, metric_norm, transport_along_polyline, transport_along_segment,
};
use std::f64::consts::FRAC_PI_2;

fn max_abs(vector: &[f64; 4]) -> f64 {
    vector
        .iter()
        .fold(0.0_f64, |worst, value| worst.max(value.abs()))
}

// --------------------------------------------------------------------------
// Flatness / exactness
// --------------------------------------------------------------------------

#[test]
fn flat_cartesian_transport_is_exact_identity() {
    // Minkowski Christoffel symbols vanish, so the transport ODE is dV/ds = 0
    // and RK4 returns the input bit-for-bit, along any path.
    let vector = [0.3, -0.5, 0.2, 0.1];
    let path = [
        [0.0, 1.0, 2.0, 3.0],
        [1.0, -2.0, 0.5, 4.0],
        [-3.0, 2.0, 1.0, 0.0],
    ];
    let transported = transport_along_polyline(&Minkowski, &path, &vector, 8).unwrap();
    assert_eq!(transported, vector);
}

// --------------------------------------------------------------------------
// Metric compatibility (norm preservation)
// --------------------------------------------------------------------------

fn assert_norm_preserved<B: Metric<4> + Connection<4>>(
    background: &B,
    start: [f64; 4],
    end: [f64; 4],
    vector: [f64; 4],
) {
    let transported = transport_along_segment(background, &start, &end, &vector, 200).unwrap();
    let initial_norm = metric_norm(&background.components(&start), &vector);
    let final_norm = metric_norm(&background.components(&end), &transported);
    let relative = (final_norm - initial_norm).abs() / initial_norm.abs().max(1.0e-12);
    assert!(
        relative < 1.0e-8,
        "norm drift {relative:.3e}: initial={initial_norm:.12e} final={final_norm:.12e}"
    );
}

#[test]
fn transport_preserves_metric_norm_schwarzschild() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    assert_norm_preserved(
        &background,
        [0.0, 10.0, FRAC_PI_2, 0.0],
        [0.0, 6.0, FRAC_PI_2, 1.0],
        [0.2, 0.1, 0.03, 0.02],
    );
}

#[test]
fn transport_preserves_metric_norm_de_sitter() {
    let background = DeSitter::try_new(0.03).unwrap();
    assert_norm_preserved(
        &background,
        [0.0, 3.0, FRAC_PI_2, 0.0],
        [0.0, 5.0, 1.2, 0.7],
        [0.15, 0.2, 0.05, 0.04],
    );
}

// --------------------------------------------------------------------------
// Holonomy
// --------------------------------------------------------------------------

#[test]
fn flat_spherical_closed_loop_has_zero_holonomy() {
    // Spherical-chart Minkowski is genuinely flat, so transport around any
    // closed loop returns the vector (zero holonomy) despite the non-zero
    // Christoffel symbols of the chart.
    let background = MinkowskiSpherical;
    let base = [0.0, 5.0, FRAC_PI_2, 0.3];
    let edge = 0.4;
    let loop_path = [
        base,
        [base[0], base[1] + edge, base[2], base[3]],
        [base[0], base[1] + edge, base[2] + edge, base[3]],
        [base[0], base[1], base[2] + edge, base[3]],
        base,
    ];
    let vector = [0.1, 0.3, 0.05, 0.2];
    let defect = holonomy_defect(&background, &loop_path, &vector, 200).unwrap();
    assert!(
        max_abs(&defect) < 1.0e-9,
        "flat holonomy {:.3e}",
        max_abs(&defect)
    );
}

#[test]
fn holonomy_matches_riemann_curvature_de_sitter() {
    // The holonomy defect of transport around a small parallelogram loop
    // spanned by A = eps e_(mu0), B = eps e_(nu0) is, to leading order,
    //   Delta V^rho = - R^rho_(sigma mu0 nu0) V^sigma eps^2.
    // This checks the transport engine against the independently computed
    // Riemann tensor -- a cross-validation of two separate numerical engines.
    let background = DeSitter::try_new(0.03).unwrap();
    let point = [0.0, 3.0, FRAC_PI_2, 0.2];
    let vector = [0.1, 0.2, 0.05, 0.03];
    let mu0 = 1; // radial
    let nu0 = 2; // polar

    let riemann = *CurvatureTensors::compute(&background, &point, 1.0e-4)
        .unwrap()
        .riemann();

    let eps = 1.0e-2;
    let mut a_end = point;
    a_end[mu0] += eps;
    let mut ab_end = a_end;
    ab_end[nu0] += eps;
    let mut b_end = point;
    b_end[nu0] += eps;
    let loop_path = [point, a_end, ab_end, b_end, point];

    let defect = holonomy_defect(&background, &loop_path, &vector, 400).unwrap();

    for rho in 0..4
    {
        let mut prediction = 0.0;
        for sigma in 0..4
        {
            prediction -= riemann[rho][sigma][mu0][nu0] * vector[sigma];
        }
        prediction *= eps * eps;

        // Leading-order identity: the next-order correction is O(eps), so at
        // eps = 1e-2 the significant components agree to well under 2%.
        let tolerance = 2.0e-2 * prediction.abs() + 1.0e-9;
        assert!(
            (defect[rho] - prediction).abs() <= tolerance,
            "rho={rho}: defect={:.6e} prediction={:.6e}",
            defect[rho],
            prediction
        );
    }

    // The loop lies in the (r, theta) plane, so the holonomy must actually
    // rotate the vector there: at least one component is non-trivially non-zero.
    assert!(max_abs(&defect) > 1.0e-8);
}

// --------------------------------------------------------------------------
// Linearity
// --------------------------------------------------------------------------

#[test]
fn transport_is_linear() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let start = [0.0, 9.0, FRAC_PI_2, 0.0];
    let end = [0.0, 7.0, 1.3, 0.5];
    let v = [0.2, 0.1, 0.03, 0.02];
    let w = [-0.1, 0.25, 0.0, 0.05];
    let (a, b) = (1.5, -0.7);

    let mut combination = [0.0; 4];
    for i in 0..4
    {
        combination[i] = a * v[i] + b * w[i];
    }

    let transported_combination =
        transport_along_segment(&background, &start, &end, &combination, 100).unwrap();
    let transported_v = transport_along_segment(&background, &start, &end, &v, 100).unwrap();
    let transported_w = transport_along_segment(&background, &start, &end, &w, 100).unwrap();

    for i in 0..4
    {
        let linear = a * transported_v[i] + b * transported_w[i];
        assert!((transported_combination[i] - linear).abs() < 1.0e-12);
    }
}

// --------------------------------------------------------------------------
// Error paths
// --------------------------------------------------------------------------

#[test]
fn transport_reports_typed_errors() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let start = [0.0, 8.0, FRAC_PI_2, 0.0];
    let end = [0.0, 7.0, FRAC_PI_2, 0.1];
    let vector = [0.1, 0.2, 0.0, 0.0];

    assert_eq!(
        transport_along_segment(&background, &start, &end, &vector, 0),
        Err(RelativityError::InvalidTransportResolution),
    );
    assert_eq!(
        transport_along_segment(
            &background,
            &[0.0, f64::NAN, FRAC_PI_2, 0.0],
            &end,
            &vector,
            10,
        ),
        Err(RelativityError::NonFiniteCoordinate(1)),
    );
    assert_eq!(
        transport_along_segment(
            &background,
            &start,
            &end,
            &[f64::INFINITY, 0.0, 0.0, 0.0],
            10
        ),
        Err(RelativityError::NonFiniteTransportedVector),
    );
    assert_eq!(
        transport_along_polyline(&background, &[start, end], &vector, 0),
        Err(RelativityError::InvalidTransportResolution),
    );
}
