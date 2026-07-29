//! Exact-oracle validation of the numerical curvature engine.
//!
//! Each background below has a curvature that is known in closed form, so it
//! serves as an analytic oracle for [`CurvatureTensors::compute`]:
//!
//! - Minkowski is flat: every curvature component is *exactly* zero (the
//!   Christoffel symbols are identically zero, so their finite differences are
//!   too — this is an exact, not an approximate, result).
//! - Schwarzschild is Ricci-flat with `K = 48 M^2 / r^6`.
//! - de Sitter / anti-de Sitter are maximally symmetric:
//!   `R_(mu nu) = Lambda g_(mu nu)`, `R = 4 Lambda`,
//!   `G_(mu nu) = -Lambda g_(mu nu)`, `K = 8 Lambda^2 / 3`, and
//!   `R_(abcd) = (Lambda / 3) (g_(ac) g_(bd) - g_(ad) g_(bc))`.
//!
//! The finite-difference engine reproduces these to a stated tolerance; the
//! tolerances are set from the second-order accuracy of a central difference
//! of an analytic connection, not tuned to hide error.

use scirust_relativity::{
    AntiDeSitter, CurvatureTensors, DeSitter, Metric, Minkowski, RelativityError, Schwarzschild,
};
use std::f64::consts::FRAC_PI_2;

/// Central-difference step for the analytic-connection backgrounds. Small
/// enough that the `O(h^2)` truncation error dominates roundoff, large enough
/// that roundoff (`~eps / h`) stays well below the truncation floor.
const STEP: f64 = 1.0e-4;

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    let scale = expected.abs().max(1.0);
    let relative_error = (actual - expected).abs() / scale;
    assert!(
        relative_error <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, \
         relative_error={relative_error:.17e}, tolerance={tolerance:.17e}"
    );
}

fn assert_abs(actual: f64, expected: f64, tolerance: f64) {
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, \
         abs_error={error:.17e}, tolerance={tolerance:.17e}"
    );
}

/// Fully covariant Riemann `R_(rho sigma mu nu) = g_(rho lambda) R^lambda_(sigma mu nu)`.
fn lower_riemann<const D: usize>(
    riemann: &[[[[f64; D]; D]; D]; D],
    metric: &[[f64; D]; D],
) -> [[[[f64; D]; D]; D]; D] {
    let mut lower = [[[[0.0_f64; D]; D]; D]; D];
    for rho in 0..D
    {
        for sigma in 0..D
        {
            for mu in 0..D
            {
                for nu in 0..D
                {
                    let mut value = 0.0;
                    for lambda in 0..D
                    {
                        value += metric[rho][lambda] * riemann[lambda][sigma][mu][nu];
                    }
                    lower[rho][sigma][mu][nu] = value;
                }
            }
        }
    }
    lower
}

// --------------------------------------------------------------------------
// Minkowski — exact flatness
// --------------------------------------------------------------------------

#[test]
fn minkowski_curvature_is_exactly_zero() {
    let coordinates = [0.3, 4.0, FRAC_PI_2, -0.7];
    let tensors = CurvatureTensors::compute(&Minkowski, &coordinates, STEP).unwrap();

    // Minkowski Christoffel symbols are identically zero, so every curvature
    // component is *exactly* zero, with no finite-difference error at all.
    for block in tensors.riemann()
    {
        for plane in block
        {
            for row in plane
            {
                for &value in row
                {
                    assert_eq!(value, 0.0);
                }
            }
        }
    }
    for row in tensors.ricci()
    {
        for &value in row
        {
            assert_eq!(value, 0.0);
        }
    }
    for row in tensors.einstein()
    {
        for &value in row
        {
            assert_eq!(value, 0.0);
        }
    }
    assert_eq!(tensors.ricci_scalar(), 0.0);
    assert_eq!(tensors.kretschmann(), 0.0);
}

// --------------------------------------------------------------------------
// Schwarzschild — Ricci-flat, K = 48 M^2 / r^6
// --------------------------------------------------------------------------

#[test]
fn schwarzschild_is_ricci_flat() {
    let mass = 1.0;
    let spacetime = Schwarzschild::try_new(mass).unwrap();
    let coordinates = [0.0, 8.0, FRAC_PI_2, 0.0];
    let tensors = CurvatureTensors::compute(&spacetime, &coordinates, STEP).unwrap();

    // Vacuum solution: Ricci, Ricci scalar, and Einstein all vanish.
    for row in tensors.ricci()
    {
        for &value in row
        {
            assert_abs(value, 0.0, 1.0e-6);
        }
    }
    assert_abs(tensors.ricci_scalar(), 0.0, 1.0e-6);
    for row in tensors.einstein()
    {
        for &value in row
        {
            assert_abs(value, 0.0, 1.0e-6);
        }
    }
}

#[test]
fn schwarzschild_kretschmann_matches_closed_form() {
    let mass = 1.0;
    let spacetime = Schwarzschild::try_new(mass).unwrap();

    // K = 48 M^2 / r^6; at M = 1, r = 8 this is 48 / 262144 = 1.8310546875e-4,
    // a dyadic rational representable exactly in binary floating point.
    let coordinates = [0.0, 8.0, FRAC_PI_2, 0.0];
    let tensors = CurvatureTensors::compute(&spacetime, &coordinates, STEP).unwrap();
    let expected = 48.0 * mass * mass / 8.0_f64.powi(6);
    assert_eq!(expected, 0.00018310546875);
    assert_close(tensors.kretschmann(), expected, 1.0e-6);

    // A second radius, to confirm the r^-6 scaling rather than a single fit.
    let coordinates = [0.0, 6.0, FRAC_PI_2, 0.0];
    let tensors = CurvatureTensors::compute(&spacetime, &coordinates, STEP).unwrap();
    let expected = 48.0 * mass * mass / 6.0_f64.powi(6);
    assert_close(tensors.kretschmann(), expected, 1.0e-6);
}

#[test]
fn schwarzschild_riemann_is_nontrivial() {
    let spacetime = Schwarzschild::try_new(1.0).unwrap();
    let coordinates = [0.0, 8.0, FRAC_PI_2, 0.0];
    let tensors = CurvatureTensors::compute(&spacetime, &coordinates, STEP).unwrap();

    // Ricci-flat does not mean flat: the Riemann tensor is non-zero.
    let mut max_component = 0.0_f64;
    for block in tensors.riemann()
    {
        for plane in block
        {
            for row in plane
            {
                for &value in row
                {
                    max_component = max_component.max(value.abs());
                }
            }
        }
    }
    assert!(max_component > 1.0e-6, "Riemann should be non-trivial");
}

// --------------------------------------------------------------------------
// de Sitter — maximally symmetric, Lambda > 0
// --------------------------------------------------------------------------

#[test]
fn de_sitter_ricci_is_lambda_times_metric() {
    let lambda = 0.03;
    let spacetime = DeSitter::try_new(lambda).unwrap();
    let coordinates = [0.0, 3.0, FRAC_PI_2, 0.0]; // horizon at r = 10
    assert!(spacetime.is_in_static_patch(&coordinates));

    let metric = spacetime.components(&coordinates);
    let tensors = CurvatureTensors::compute(&spacetime, &coordinates, STEP).unwrap();

    // R_(mu nu) = Lambda g_(mu nu).
    for (mu, row) in tensors.ricci().iter().enumerate()
    {
        for (nu, &value) in row.iter().enumerate()
        {
            assert_abs(value, lambda * metric[mu][nu], 1.0e-6);
        }
    }

    // R = 4 Lambda.
    assert_close(tensors.ricci_scalar(), 4.0 * lambda, 1.0e-5);

    // G_(mu nu) = R_(mu nu) - 1/2 R g_(mu nu) = -Lambda g_(mu nu).
    for (mu, row) in tensors.einstein().iter().enumerate()
    {
        for (nu, &value) in row.iter().enumerate()
        {
            assert_abs(value, -lambda * metric[mu][nu], 1.0e-6);
        }
    }
}

#[test]
fn de_sitter_kretschmann_matches_eight_thirds_lambda_squared() {
    let lambda = 0.03;
    let spacetime = DeSitter::try_new(lambda).unwrap();
    let coordinates = [0.0, 3.0, FRAC_PI_2, 0.0];
    let tensors = CurvatureTensors::compute(&spacetime, &coordinates, STEP).unwrap();

    // K = 8 Lambda^2 / 3, equivalently R^2 / 6 for a 4D maximally symmetric
    // spacetime.
    let expected = 8.0 * lambda * lambda / 3.0;
    assert_close(tensors.kretschmann(), expected, 1.0e-5);
    let scalar = tensors.ricci_scalar();
    assert_close(tensors.kretschmann(), scalar * scalar / 6.0, 1.0e-4);
}

#[test]
fn de_sitter_riemann_is_maximally_symmetric() {
    let lambda = 0.03;
    let spacetime = DeSitter::try_new(lambda).unwrap();
    let coordinates = [0.0, 3.0, FRAC_PI_2, 0.0];
    let metric = spacetime.components(&coordinates);
    let tensors = CurvatureTensors::compute(&spacetime, &coordinates, STEP).unwrap();

    // R_(abcd) = (Lambda / 3) (g_(ac) g_(bd) - g_(ad) g_(bc)).
    let lower = lower_riemann(tensors.riemann(), &metric);
    for a in 0..4
    {
        for b in 0..4
        {
            for c in 0..4
            {
                for d in 0..4
                {
                    let expected = (lambda / 3.0)
                        * (metric[a][c] * metric[b][d] - metric[a][d] * metric[b][c]);
                    assert_abs(lower[a][b][c][d], expected, 1.0e-5);
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// anti-de Sitter — maximally symmetric, Lambda < 0
// --------------------------------------------------------------------------

#[test]
fn anti_de_sitter_ricci_is_negative_lambda_times_metric() {
    let magnitude = 0.05;
    let spacetime = AntiDeSitter::try_new(magnitude).unwrap();
    let lambda = spacetime.cosmological_constant(); // -0.05
    assert_eq!(lambda, -0.05);

    let coordinates = [0.0, 4.0, FRAC_PI_2, 0.0]; // no horizon
    assert!(spacetime.is_in_static_patch(&coordinates));

    let metric = spacetime.components(&coordinates);
    let tensors = CurvatureTensors::compute(&spacetime, &coordinates, STEP).unwrap();

    // R_(mu nu) = Lambda g_(mu nu) with Lambda < 0.
    for (mu, row) in tensors.ricci().iter().enumerate()
    {
        for (nu, &value) in row.iter().enumerate()
        {
            assert_abs(value, lambda * metric[mu][nu], 1.0e-6);
        }
    }

    // R = 4 Lambda < 0.
    assert_close(tensors.ricci_scalar(), 4.0 * lambda, 1.0e-5);
    assert!(tensors.ricci_scalar() < 0.0);

    // K = 8 Lambda^2 / 3 (sign-independent).
    let expected = 8.0 * lambda * lambda / 3.0;
    assert_close(tensors.kretschmann(), expected, 1.0e-5);
}

// --------------------------------------------------------------------------
// Algebraic symmetries of the Riemann tensor (structural, not oracle values)
// --------------------------------------------------------------------------

#[test]
fn riemann_symmetries_and_first_bianchi_hold() {
    // A curved, Ricci-flat background (Schwarzschild) and a curved,
    // maximally symmetric one (de Sitter) exercise the full symmetry group.
    let schwarzschild = Schwarzschild::try_new(1.0).unwrap();
    let de_sitter = DeSitter::try_new(0.03).unwrap();
    let point = [0.0, 5.0, FRAC_PI_2, 0.0];

    for background_lower in [
        {
            let t = CurvatureTensors::compute(&schwarzschild, &point, STEP).unwrap();
            lower_riemann(t.riemann(), &schwarzschild.components(&point))
        },
        {
            let t = CurvatureTensors::compute(&de_sitter, &point, STEP).unwrap();
            lower_riemann(t.riemann(), &de_sitter.components(&point))
        },
    ]
    {
        // The four indices appear in permuted positions across the symmetry
        // relations below (e.g. [a][b][c][d] vs [c][d][a][b]), so an
        // enumerate/iterator rewrite would be strictly less readable than
        // explicit tensor-index ranges here.
        #[allow(clippy::needless_range_loop)]
        for a in 0..4
        {
            for b in 0..4
            {
                for c in 0..4
                {
                    for d in 0..4
                    {
                        // Antisymmetry in the last pair.
                        assert_abs(
                            background_lower[a][b][c][d],
                            -background_lower[a][b][d][c],
                            1.0e-6,
                        );
                        // Antisymmetry in the first pair.
                        assert_abs(
                            background_lower[a][b][c][d],
                            -background_lower[b][a][c][d],
                            1.0e-6,
                        );
                        // Pair-exchange symmetry.
                        assert_abs(
                            background_lower[a][b][c][d],
                            background_lower[c][d][a][b],
                            1.0e-6,
                        );
                        // First Bianchi identity: R_(a[bcd]) = 0.
                        assert_abs(
                            background_lower[a][b][c][d]
                                + background_lower[a][c][d][b]
                                + background_lower[a][d][b][c],
                            0.0,
                            1.0e-6,
                        );
                    }
                }
            }
        }
    }
}

// --------------------------------------------------------------------------
// Error paths
// --------------------------------------------------------------------------

#[test]
fn compute_reports_typed_errors_for_invalid_input() {
    let spacetime = Schwarzschild::try_new(1.0).unwrap();

    let bad_coordinate = [0.0, f64::NAN, FRAC_PI_2, 0.0];
    assert_eq!(
        CurvatureTensors::compute(&spacetime, &bad_coordinate, STEP),
        Err(RelativityError::NonFiniteCoordinate(1)),
    );

    let good = [0.0, 8.0, FRAC_PI_2, 0.0];
    assert_eq!(
        CurvatureTensors::compute(&spacetime, &good, 0.0),
        Err(RelativityError::InvalidDifferenceStep(0.0)),
    );
    assert_eq!(
        CurvatureTensors::compute(&spacetime, &good, -1.0e-4),
        Err(RelativityError::InvalidDifferenceStep(-1.0e-4)),
    );
    assert!(matches!(
        CurvatureTensors::compute(&spacetime, &good, f64::NAN),
        Err(RelativityError::InvalidDifferenceStep(_)),
    ));
}
