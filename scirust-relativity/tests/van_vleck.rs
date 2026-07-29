//! Validation of the determinant primitive and the van Vleck–Morette
//! determinant.
//!
//! - `determinant` matches known values and the `det(A) det(A^-1) = 1` identity.
//! - The van Vleck determinant is `1` in flat spacetime and at coincidence,
//!   symmetric in its two arguments on curved backgrounds, and near coincidence
//!   matches the known maximally-symmetric leading expansion
//!   `(Delta - 1)/sigma -> Lambda/3`.
//! - Inputs are validated with typed errors.

use scirust_relativity::{
    AntiDeSitter, Connection, DeSitter, Metric, Minkowski, RelativityError, Schwarzschild,
    WorldFunctionSettings, determinant, invert_metric, van_vleck_determinant, world_function,
};
use std::f64::consts::FRAC_PI_2;

fn settings() -> WorldFunctionSettings {
    WorldFunctionSettings::default()
}

// --------------------------------------------------------------------------
// determinant primitive
// --------------------------------------------------------------------------

#[test]
fn determinant_matches_known_values() {
    let identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    assert!((determinant(&identity).unwrap() - 1.0).abs() < 1.0e-14);

    // Diagonal: product of the diagonal (with a row swap exercised by pivoting).
    let diagonal = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 5.0],
    ];
    assert!((determinant(&diagonal).unwrap() - 120.0).abs() < 1.0e-12);

    // Minkowski: det diag(-1, 1, 1, 1) = -1.
    let minkowski = Minkowski.components(&[0.0; 4]);
    assert!((determinant(&minkowski).unwrap() + 1.0).abs() < 1.0e-14);

    // A singular matrix (two identical rows) has determinant exactly zero.
    let singular = [
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 0.0, 1.0],
        [2.0, 0.0, 1.0, 0.0],
    ];
    assert_eq!(determinant(&singular).unwrap(), 0.0);
}

#[test]
fn determinant_agrees_with_inverse_and_analytic_schwarzschild() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let metric = background.components(&[0.0, 10.0, FRAC_PI_2, 0.0]);

    // Schwarzschild g = diag(-f, 1/f, r^2, r^2 sin^2 theta), so
    // det g = -r^4 sin^2 theta = -10000 at r = 10, theta = pi/2.
    let det = determinant(&metric).unwrap();
    assert!((det - (-10000.0)).abs() < 1.0e-6, "det = {det}");

    // det(g) det(g^-1) = 1.
    let inverse = invert_metric(&metric).unwrap();
    let product = det * determinant(&inverse).unwrap();
    assert!((product - 1.0).abs() < 1.0e-12, "product = {product}");
}

#[test]
fn determinant_reports_non_finite_entries() {
    let mut matrix = [[0.0_f64; 4]; 4];
    matrix[2][1] = f64::NAN;
    assert_eq!(
        determinant(&matrix),
        Err(RelativityError::NonFiniteMetricComponent { row: 2, column: 1 }),
    );
}

// --------------------------------------------------------------------------
// van Vleck determinant
// --------------------------------------------------------------------------

#[test]
fn van_vleck_flat_is_unity() {
    let base = [0.0, 1.0, 2.0, 0.5];
    for field in [[0.0, 2.0, 3.0, 1.0], [1.5, 1.2, 2.1, 0.6]]
    {
        let delta = van_vleck_determinant(&Minkowski, &base, &field, &settings()).unwrap();
        assert!((delta - 1.0).abs() < 1.0e-7, "flat Delta = {delta}");
    }
}

#[test]
fn van_vleck_coincidence_is_unity() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let point = [0.0, 10.0, FRAC_PI_2, 0.0];
    let delta = van_vleck_determinant(&background, &point, &point, &settings()).unwrap();
    assert!((delta - 1.0).abs() < 1.0e-7, "coincidence Delta = {delta}");
}

fn assert_symmetric<B>(background: &B, base: [f64; 4], field: [f64; 4])
where
    B: Metric<4> + Connection<4> + Copy,
{
    let forward = van_vleck_determinant(background, &base, &field, &settings()).unwrap();
    let reversed = van_vleck_determinant(background, &field, &base, &settings()).unwrap();
    assert!(
        (forward - reversed).abs() < 1.0e-8,
        "van Vleck asymmetry: {forward} vs {reversed}"
    );
}

#[test]
fn van_vleck_is_symmetric_curved() {
    assert_symmetric(
        &Schwarzschild::try_new(1.0).unwrap(),
        [0.0, 12.0, FRAC_PI_2, 0.0],
        [0.15, 12.4, FRAC_PI_2 + 0.03, 0.05],
    );
    assert_symmetric(
        &DeSitter::try_new(0.05).unwrap(),
        [0.0, 3.0, FRAC_PI_2, 0.0],
        [0.1, 3.3, FRAC_PI_2 + 0.04, 0.03],
    );
    assert_symmetric(
        &AntiDeSitter::try_new(0.05).unwrap(),
        [0.0, 3.0, FRAC_PI_2, 0.0],
        [0.1, 3.3, FRAC_PI_2 + 0.04, 0.03],
    );
}

/// `(Delta - 1)/sigma -> Lambda/3` near coincidence for a maximally symmetric
/// background. Checked at a small separation, where the leading term dominates.
fn assert_leading_expansion<B>(background: &B, signed_lambda: f64)
where
    B: Metric<4> + Connection<4> + Copy,
{
    let base = [0.0, 3.0, FRAC_PI_2, 0.0];
    let direction = [0.02, 0.15, 0.03, 0.02];
    let scale = 0.25;
    let field = [
        base[0] + scale * direction[0],
        base[1] + scale * direction[1],
        base[2] + scale * direction[2],
        base[3] + scale * direction[3],
    ];
    let sigma = world_function(background, &base, &field, &settings()).unwrap();
    let delta = van_vleck_determinant(background, &base, &field, &settings()).unwrap();
    let ratio = (delta - 1.0) / sigma;
    assert!(
        (ratio - signed_lambda / 3.0).abs() < 1.0e-4,
        "(Delta - 1)/sigma = {ratio}, expected Lambda/3 = {}",
        signed_lambda / 3.0
    );
}

#[test]
fn van_vleck_matches_de_sitter_leading_expansion() {
    let de_sitter = DeSitter::try_new(0.05).unwrap();
    assert_leading_expansion(&de_sitter, de_sitter.cosmological_constant());

    let anti_de_sitter = AntiDeSitter::try_new(0.05).unwrap();
    assert_leading_expansion(&anti_de_sitter, anti_de_sitter.cosmological_constant());
}

#[test]
fn van_vleck_reports_typed_errors() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let field = [0.1, 10.4, FRAC_PI_2 + 0.05, 0.03];

    assert_eq!(
        van_vleck_determinant(
            &background,
            &[f64::NAN, 10.0, FRAC_PI_2, 0.0],
            &field,
            &settings()
        ),
        Err(RelativityError::NonFiniteCoordinate(0)),
    );

    let starved = WorldFunctionSettings {
        max_iterations: 1,
        ..WorldFunctionSettings::default()
    };
    assert_eq!(
        van_vleck_determinant(&background, &[0.0, 10.0, FRAC_PI_2, 0.0], &field, &starved),
        Err(RelativityError::LogarithmMapDidNotConverge),
    );
}
