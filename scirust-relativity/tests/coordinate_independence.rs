//! Coordinate-independence validation of scalar curvature invariants.
//!
//! Scalar curvature invariants (the Ricci scalar `R` and the Kretschmann scalar
//! `K = R_{abcd} R^{abcd}`) are geometric: they do not depend on the coordinate
//! chart used to compute them. This suite checks that invariance directly, by
//! computing the invariants of the *same physical geometry* in two different
//! charts and requiring agreement:
//!
//! - **Flat spacetime** in Cartesian coordinates ([`Minkowski`], curvature
//!   exactly zero) versus spherical spatial coordinates
//!   ([`MinkowskiSpherical`], non-zero Christoffel symbols but numerically zero
//!   curvature).
//! - **Schwarzschild** in areal coordinates ([`Schwarzschild`], `K = 48 M^2 /
//!   r^6`) versus isotropic coordinates ([`IsotropicSchwarzschild`], whose
//!   Kretschmann scalar must equal `48 M^2 / r^6` with `r` the *areal* radius,
//!   not the isotropic radius).
//!
//! Tolerances are set from the measured finite-difference accuracy of the
//! curvature engine. The isotropic-Schwarzschild connection is itself a finite
//! difference (see [`IsotropicSchwarzschild`]), so its curvature is a nested
//! finite difference and correspondingly less accurate than the
//! analytic-connection backgrounds; the `~1e-5` tolerances below reflect that
//! honestly rather than hiding it.

use scirust_relativity::{
    CurvatureTensors, IsotropicSchwarzschild, Metric, Minkowski, MinkowskiSpherical, Schwarzschild,
};
use std::f64::consts::FRAC_PI_2;

/// Analytic-connection step (single finite difference of the curvature engine).
const ANALYTIC_STEP: f64 = 1.0e-4;
/// Nested-finite-difference step for the isotropic (numerical-connection)
/// background, where the curvature is a finite difference of a finite
/// difference and a smaller step amplifies roundoff.
const NESTED_STEP: f64 = 1.0e-3;

fn max_abs_tensor4(tensor: &[[[[f64; 4]; 4]; 4]; 4]) -> f64 {
    let mut worst = 0.0_f64;
    for block in tensor
    {
        for plane in block
        {
            for row in plane
            {
                for &value in row
                {
                    worst = worst.max(value.abs());
                }
            }
        }
    }
    worst
}

fn max_abs_tensor2(tensor: &[[f64; 4]; 4]) -> f64 {
    let mut worst = 0.0_f64;
    for row in tensor
    {
        for &value in row
        {
            worst = worst.max(value.abs());
        }
    }
    worst
}

/// Relative error with the conventional `max(|expected|, 1)` scale, for
/// quantities whose expected value may be zero or O(1).
fn relative_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs().max(1.0)
}

/// True relative error `|actual - expected| / |expected|`, for the small but
/// strictly non-zero curved-spacetime Kretschmann values (where the `max(., 1)`
/// floor above would silently turn a relative bound into a weak absolute one).
fn true_relative_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs()
}

// --------------------------------------------------------------------------
// MinkowskiSpherical — flat curvature from non-zero Christoffels
// --------------------------------------------------------------------------

#[test]
fn spherical_minkowski_metric_is_diagonal_lapse_one() {
    let spacetime = MinkowskiSpherical;
    let radius = 4.0;
    let metric = spacetime.components(&[0.3, radius, FRAC_PI_2, -0.7]);

    assert_eq!(metric[0][0], -1.0);
    assert_eq!(metric[1][1], 1.0);
    assert_eq!(metric[2][2], radius * radius);
    assert!((metric[3][3] - radius * radius).abs() < 1.0e-12); // sin^2(pi/2) = 1

    for (row_index, row) in metric.iter().enumerate()
    {
        for (column_index, &value) in row.iter().enumerate()
        {
            if row_index != column_index
            {
                assert_eq!(value, 0.0);
            }
        }
    }
}

#[test]
fn spherical_minkowski_curvature_is_numerically_zero() {
    let spacetime = MinkowskiSpherical;
    let coordinates = [0.3, 4.0, FRAC_PI_2, -0.7];
    assert!(spacetime.is_in_regular_chart(&coordinates));

    let tensors = CurvatureTensors::compute(&spacetime, &coordinates, ANALYTIC_STEP).unwrap();

    // Unlike Cartesian Minkowski, the Christoffel symbols here are non-zero, so
    // this genuinely exercises the connection-quadratic terms of the Riemann
    // tensor; the result is still (numerically) flat.
    assert!(max_abs_tensor4(tensors.riemann()) < 1.0e-6);
    assert!(max_abs_tensor2(tensors.ricci()) < 1.0e-6);
    assert!(max_abs_tensor2(tensors.einstein()) < 1.0e-6);
    assert!(tensors.ricci_scalar().abs() < 1.0e-6);
    assert!(tensors.kretschmann().abs() < 1.0e-9);
}

#[test]
fn spherical_minkowski_rejects_axis_and_origin() {
    let spacetime = MinkowskiSpherical;
    assert!(spacetime.is_in_regular_chart(&[0.0, 2.0, 1.0, 0.0]));
    assert!(!spacetime.is_in_regular_chart(&[0.0, 0.0, 1.0, 0.0])); // origin
    assert!(!spacetime.is_in_regular_chart(&[0.0, 2.0, 0.0, 0.0])); // theta = 0
    assert!(!spacetime.is_in_regular_chart(&[0.0, 2.0, std::f64::consts::PI, 0.0]));
    assert!(!spacetime.is_in_regular_chart(&[0.0, f64::NAN, 1.0, 0.0]));
}

// --------------------------------------------------------------------------
// IsotropicSchwarzschild — the areal-radius oracle
// --------------------------------------------------------------------------

#[test]
fn isotropic_constructor_and_radius_relation() {
    assert!(IsotropicSchwarzschild::try_new(1.0).is_some());
    assert!(IsotropicSchwarzschild::try_new(0.0).is_none());
    assert!(IsotropicSchwarzschild::try_new(-1.0).is_none());
    assert!(IsotropicSchwarzschild::try_new(f64::NAN).is_none());

    let spacetime = IsotropicSchwarzschild::try_new(2.0).unwrap();
    // Horizon: rho = M/2 maps to areal r = 2 M.
    assert_eq!(spacetime.horizon_radius(), 1.0);
    assert!((spacetime.areal_radius(1.0) - 4.0).abs() < 1.0e-12);
    // Far away, rho and r coincide to leading order.
    let large = 1.0e6;
    assert!(relative_error(spacetime.areal_radius(large), large) < 1.0e-5);
}

#[test]
fn isotropic_exterior_domain() {
    let spacetime = IsotropicSchwarzschild::try_new(2.0).unwrap();
    assert!(spacetime.is_in_exterior(&[0.0, 2.0, FRAC_PI_2, 0.0]));
    assert!(!spacetime.is_in_exterior(&[0.0, 1.0, FRAC_PI_2, 0.0])); // horizon
    assert!(!spacetime.is_in_exterior(&[0.0, 0.5, FRAC_PI_2, 0.0])); // interior
    assert!(!spacetime.is_in_exterior(&[0.0, 3.0, 0.0, 0.0])); // polar axis
}

#[test]
fn isotropic_metric_matches_closed_form() {
    let spacetime = IsotropicSchwarzschild::try_new(1.0).unwrap();
    let isotropic_radius = 4.0;
    let metric = spacetime.components(&[0.0, isotropic_radius, FRAC_PI_2, 0.0]);

    // A = (1 - 1/8)/(1 + 1/8) = 7/9; B = 9/8; B^4 = 1.601806640625.
    let lapse = (7.0_f64 / 9.0).powi(2);
    let conformal_fourth = (9.0_f64 / 8.0).powi(4);
    assert!((metric[0][0] + lapse).abs() < 1.0e-12);
    assert!((metric[1][1] - conformal_fourth).abs() < 1.0e-12);
    assert!(
        (metric[2][2] - conformal_fourth * isotropic_radius * isotropic_radius).abs() < 1.0e-10
    );
}

#[test]
fn isotropic_schwarzschild_is_ricci_flat() {
    let spacetime = IsotropicSchwarzschild::try_new(1.0).unwrap();
    let tensors =
        CurvatureTensors::compute(&spacetime, &[0.0, 4.0, FRAC_PI_2, 0.0], NESTED_STEP).unwrap();

    // Vacuum solution: Ricci (hence Einstein and R) vanish to nested-FD tolerance.
    assert!(max_abs_tensor2(tensors.ricci()) < 5.0e-5);
    assert!(tensors.ricci_scalar().abs() < 5.0e-5);
    assert!(max_abs_tensor2(tensors.einstein()) < 5.0e-5);
}

#[test]
fn isotropic_kretschmann_matches_areal_oracle() {
    let mass = 1.0;
    let spacetime = IsotropicSchwarzschild::try_new(mass).unwrap();
    let isotropic_radius = 4.0;
    let areal_radius = spacetime.areal_radius(isotropic_radius);

    let tensors = CurvatureTensors::compute(
        &spacetime,
        &[0.0, isotropic_radius, FRAC_PI_2, 0.0],
        NESTED_STEP,
    )
    .unwrap();

    // K = 48 M^2 / r^6 with r the AREAL radius (~5.0625), not the isotropic
    // radius (4.0). Using the isotropic radius would be off by a large factor.
    let oracle = 48.0 * mass * mass / areal_radius.powi(6);
    assert!(true_relative_error(tensors.kretschmann(), oracle) < 5.0e-5);

    // Guard against an accidental rho-based oracle passing by coincidence: the
    // isotropic-radius oracle is a large factor off (a true relative gap, using
    // the wrong oracle itself as the scale rather than the max(.,1) floor).
    let wrong_oracle = 48.0 * mass * mass / isotropic_radius.powi(6);
    let true_relative_gap = (tensors.kretschmann() - wrong_oracle).abs() / wrong_oracle;
    assert!(
        true_relative_gap > 0.5,
        "Kretschmann must use the areal radius, not the isotropic radius"
    );
}

// --------------------------------------------------------------------------
// Coordinate independence: same geometry, two charts, invariants agree
// --------------------------------------------------------------------------

#[test]
fn kretschmann_agrees_between_areal_and_isotropic_schwarzschild() {
    let mass = 1.0;
    let isotropic = IsotropicSchwarzschild::try_new(mass).unwrap();
    let areal = Schwarzschild::try_new(mass).unwrap();

    // Sweep several isotropic radii; each maps to an areal radius, and the
    // Kretschmann scalar computed in the two charts must agree. The nested-FD
    // isotropic accuracy degrades as K shrinks with radius, so the sweep stops
    // at rho = 6 (true relative agreement stays <= ~3e-6 there); larger radii
    // are reported, without an assertion, by the coordinate_independence
    // experiment.
    for &isotropic_radius in &[3.0, 4.0, 6.0]
    {
        let areal_radius = isotropic.areal_radius(isotropic_radius);
        assert!(areal_radius > 2.0 * mass);

        let isotropic_k = CurvatureTensors::compute(
            &isotropic,
            &[0.0, isotropic_radius, FRAC_PI_2, 0.0],
            NESTED_STEP,
        )
        .unwrap()
        .kretschmann();

        let areal_k =
            CurvatureTensors::compute(&areal, &[0.0, areal_radius, FRAC_PI_2, 0.0], ANALYTIC_STEP)
                .unwrap()
                .kretschmann();

        // Both must equal the analytic invariant, and hence each other.
        let oracle = 48.0 * mass * mass / areal_radius.powi(6);
        assert!(true_relative_error(areal_k, oracle) < 1.0e-6);
        assert!(true_relative_error(isotropic_k, oracle) < 5.0e-5);
        assert!(true_relative_error(isotropic_k, areal_k) < 5.0e-5);
    }
}

#[test]
fn ricci_scalar_agrees_between_cartesian_and_spherical_minkowski() {
    let cartesian =
        CurvatureTensors::compute(&Minkowski, &[0.3, 4.0, FRAC_PI_2, -0.7], ANALYTIC_STEP).unwrap();
    let spherical = CurvatureTensors::compute(
        &MinkowskiSpherical,
        &[0.3, 4.0, FRAC_PI_2, -0.7],
        ANALYTIC_STEP,
    )
    .unwrap();

    // Cartesian is exactly flat; spherical is numerically flat; both scalars
    // agree (at zero) despite very different Christoffel symbols.
    assert_eq!(cartesian.ricci_scalar(), 0.0);
    assert_eq!(cartesian.kretschmann(), 0.0);
    assert!((spherical.ricci_scalar() - cartesian.ricci_scalar()).abs() < 1.0e-6);
    assert!((spherical.kretschmann() - cartesian.kretschmann()).abs() < 1.0e-9);
}
