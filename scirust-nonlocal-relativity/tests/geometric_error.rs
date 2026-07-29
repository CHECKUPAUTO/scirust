//! Phase 6 tests for the metric-aware `timelike_state_error` diagnostic.
//!
//! The flat-spacetime cases are exact: with the Minkowski metric
//! `diag(-1, 1, 1, 1)` and the static observer `u = (1, 0, 0, 0)`, the
//! temporal magnitude is exactly `|delta^0|` and the spatial magnitude is
//! exactly the Euclidean length of the spatial part of `delta`.

use scirust_nonlocal_relativity::{
    NonlocalRelativityError, TimelikeStateError, build_orthonormal_tetrad, metric_contraction,
    schwarzschild_circular_orbit_four_velocity, tetrad_state_error, timelike_state_error,
};
use scirust_relativity::{Metric, Minkowski, Schwarzschild};
use std::f64::consts::FRAC_PI_2;

const FLOOR: f64 = 1.0e-9;

fn minkowski_metric() -> [[f64; 4]; 4] {
    Minkowski.components(&[0.0, 0.0, 0.0, 0.0])
}

#[test]
fn flat_static_observer_purely_spatial_delta_is_exact_euclidean_length() {
    let metric = minkowski_metric();
    let observer = [1.0, 0.0, 0.0, 0.0];
    let delta = [0.0, 3.0, 4.0, 0.0];

    let TimelikeStateError {
        temporal,
        spatial,
        orthogonality_residual,
    } = timelike_state_error(&metric, &observer, &delta, FLOOR).unwrap();

    assert_eq!(temporal.to_bits(), 0.0_f64.to_bits());
    assert_eq!(spatial.to_bits(), 5.0_f64.to_bits());
    assert_eq!(orthogonality_residual.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn flat_static_observer_purely_temporal_delta_has_zero_spatial() {
    let metric = minkowski_metric();
    let observer = [1.0, 0.0, 0.0, 0.0];
    let delta = [2.0, 0.0, 0.0, 0.0];

    let error = timelike_state_error(&metric, &observer, &delta, FLOOR).unwrap();

    assert_eq!(error.temporal.to_bits(), 2.0_f64.to_bits());
    assert_eq!(error.spatial.to_bits(), 0.0_f64.to_bits());
}

#[test]
fn flat_static_observer_mixed_delta_splits_exactly() {
    let metric = minkowski_metric();
    let observer = [1.0, 0.0, 0.0, 0.0];
    let delta = [1.0, 3.0, 4.0, 0.0];

    let error = timelike_state_error(&metric, &observer, &delta, FLOOR).unwrap();

    assert_eq!(error.temporal.to_bits(), 1.0_f64.to_bits());
    assert_eq!(error.spatial.to_bits(), 5.0_f64.to_bits());
    assert!(error.orthogonality_residual.abs() < 1.0e-15);
}

#[test]
fn geometric_split_differs_from_the_componentwise_euclidean_norm() {
    // A purely temporal error has zero geometric spatial magnitude, but its
    // raw componentwise Euclidean norm is nonzero: the two measures are not
    // the same thing, which is exactly why the geometric diagnostic exists.
    let metric = minkowski_metric();
    let observer = [1.0, 0.0, 0.0, 0.0];
    let delta = [3.0, 0.0, 0.0, 0.0];

    let error = timelike_state_error(&metric, &observer, &delta, FLOOR).unwrap();
    let componentwise_norm = delta.iter().map(|c| c * c).sum::<f64>().sqrt();

    assert_eq!(error.spatial.to_bits(), 0.0_f64.to_bits());
    assert_eq!(error.temporal.to_bits(), 3.0_f64.to_bits());
    assert!(componentwise_norm > 2.9);
}

#[test]
fn moving_observer_projection_is_orthogonal_and_spatial_is_nonnegative() {
    // A boosted observer in flat spacetime: u = gamma (1, v, 0, 0). The
    // projected error must be metric-orthogonal to u (residual ~ 0) and the
    // spatial magnitude non-negative, for a delta with both temporal and
    // spatial parts.
    let metric = minkowski_metric();
    let v = 0.5_f64;
    let gamma = 1.0 / (1.0 - v * v).sqrt();
    let observer = [gamma, gamma * v, 0.0, 0.0];
    let delta = [0.3, -0.2, 0.7, 0.1];

    let error = timelike_state_error(&metric, &observer, &delta, FLOOR).unwrap();

    assert!(
        error.orthogonality_residual.abs() < 1.0e-14,
        "orthogonality residual too large: {}",
        error.orthogonality_residual
    );
    assert!(error.spatial >= 0.0);
    assert!(error.temporal >= 0.0);
    assert!(error.spatial.is_finite() && error.temporal.is_finite());
}

#[test]
fn schwarzschild_circular_orbit_observer_is_handled_correctly() {
    let mass = 1.0;
    let radius = 10.0;
    let background = Schwarzschild::try_new(mass).unwrap();
    let observer = schwarzschild_circular_orbit_four_velocity(&background, radius).unwrap();
    let metric = background.components(&[0.0, radius, FRAC_PI_2, 0.0]);
    let delta = [1.0e-3, -2.0e-3, 0.0, 3.0e-3];

    let error = timelike_state_error(&metric, &observer, &delta, FLOOR).unwrap();

    // The projection is metric-orthogonal to the (curved-background) observer.
    assert!(
        error.orthogonality_residual.abs() < 1.0e-12,
        "orthogonality residual too large: {}",
        error.orthogonality_residual
    );
    assert!(error.spatial >= 0.0 && error.spatial.is_finite());
    assert!(error.temporal >= 0.0 && error.temporal.is_finite());
}

#[test]
fn rejects_non_timelike_observers() {
    let metric = minkowski_metric();
    // Spacelike observer: g(u,u) = +1 > 0.
    let spacelike = [0.0, 1.0, 0.0, 0.0];
    assert!(matches!(
        timelike_state_error(&metric, &spacelike, &[1.0, 0.0, 0.0, 0.0], FLOOR),
        Err(NonlocalRelativityError::NonTimelikeMetricNorm { .. })
    ));
    // Null observer: g(u,u) = 0.
    let null = [1.0, 1.0, 0.0, 0.0];
    assert!(matches!(
        timelike_state_error(&metric, &null, &[1.0, 0.0, 0.0, 0.0], FLOOR),
        Err(NonlocalRelativityError::NonTimelikeMetricNorm { .. })
    ));
}

#[test]
fn rejects_invalid_floor() {
    let metric = minkowski_metric();
    let observer = [1.0, 0.0, 0.0, 0.0];
    for bad in [0.0, -1.0e-9, f64::NAN, f64::INFINITY]
    {
        assert!(matches!(
            timelike_state_error(&metric, &observer, &[1.0, 0.0, 0.0, 0.0], bad),
            Err(NonlocalRelativityError::InvalidMetricNormFloor(_))
        ));
    }
}

// ---- tetrad projection ----

fn assert_orthonormal(metric: &[[f64; 4]; 4], legs: &[[f64; 4]; 4]) {
    // g(e_a, e_b) must equal eta_ab = diag(-1, 1, 1, 1) to machine precision.
    for a in 0..4
    {
        for b in 0..4
        {
            let inner = metric_contraction(metric, &legs[a], &legs[b]);
            let expected = if a == b
            {
                if a == 0 { -1.0 } else { 1.0 }
            }
            else
            {
                0.0
            };
            assert!(
                (inner - expected).abs() < 1.0e-12,
                "g(e_{a}, e_{b}) = {inner}, expected {expected}"
            );
        }
    }
}

#[test]
fn flat_static_observer_tetrad_is_the_standard_basis() {
    let metric = minkowski_metric();
    let observer = [1.0, 0.0, 0.0, 0.0];
    let tetrad = build_orthonormal_tetrad(&metric, &observer, FLOOR).unwrap();
    assert_orthonormal(&metric, tetrad.legs());

    let projected = tetrad_state_error(&metric, &observer, &[2.0, 3.0, 4.0, 0.0], FLOOR).unwrap();
    // Standard-basis frame: components equal the coordinate components.
    assert_eq!(projected.components[0].to_bits(), 2.0_f64.to_bits());
    assert_eq!(projected.components[1].to_bits(), 3.0_f64.to_bits());
    assert_eq!(projected.components[2].to_bits(), 4.0_f64.to_bits());
    assert_eq!(projected.components[3].to_bits(), 0.0_f64.to_bits());
    assert_eq!(projected.temporal.to_bits(), 2.0_f64.to_bits());
    assert_eq!(projected.spatial.to_bits(), 5.0_f64.to_bits());
    assert_eq!(
        projected.reconstruction_residual.to_bits(),
        0.0_f64.to_bits()
    );
}

#[test]
fn tetrad_reconstructs_delta_and_matches_scalar_split_flat_boosted() {
    let metric = minkowski_metric();
    let v = 0.6_f64;
    let gamma = 1.0 / (1.0 - v * v).sqrt();
    let observer = [gamma, gamma * v, 0.0, 0.0];
    let delta = [0.3, -0.2, 0.7, 0.1];

    let tetrad = build_orthonormal_tetrad(&metric, &observer, FLOOR).unwrap();
    assert_orthonormal(&metric, tetrad.legs());

    let projected = tetrad_state_error(&metric, &observer, &delta, FLOOR).unwrap();
    assert!(
        projected.reconstruction_residual < 1.0e-13,
        "reconstruction residual too large: {}",
        projected.reconstruction_residual
    );

    // The tetrad temporal/spatial magnitudes match the scalar split.
    let scalar = timelike_state_error(&metric, &observer, &delta, FLOOR).unwrap();
    assert!((projected.temporal - scalar.temporal).abs() < 1.0e-12);
    assert!((projected.spatial - scalar.spatial).abs() < 1.0e-12);
}

#[test]
fn schwarzschild_orbit_tetrad_is_orthonormal_and_reconstructs() {
    let mass = 1.0;
    let radius = 10.0;
    let background = Schwarzschild::try_new(mass).unwrap();
    let observer = schwarzschild_circular_orbit_four_velocity(&background, radius).unwrap();
    let metric = background.components(&[0.0, radius, FRAC_PI_2, 0.0]);
    let delta = [1.0e-3, -2.0e-3, 5.0e-4, 3.0e-3];

    let tetrad = build_orthonormal_tetrad(&metric, &observer, FLOOR).unwrap();
    assert_orthonormal(&metric, tetrad.legs());

    let projected = tetrad_state_error(&metric, &observer, &delta, FLOOR).unwrap();
    assert!(
        projected.reconstruction_residual < 1.0e-12,
        "reconstruction residual too large: {}",
        projected.reconstruction_residual
    );
    let scalar = timelike_state_error(&metric, &observer, &delta, FLOOR).unwrap();
    assert!((projected.temporal - scalar.temporal).abs() < 1.0e-12);
    assert!((projected.spatial - scalar.spatial).abs() < 1.0e-12);
}

#[test]
fn tetrad_rejects_non_timelike_and_invalid_floor() {
    let metric = minkowski_metric();
    let spacelike = [0.0, 1.0, 0.0, 0.0];
    assert!(matches!(
        build_orthonormal_tetrad(&metric, &spacelike, FLOOR),
        Err(NonlocalRelativityError::NonTimelikeMetricNorm { .. })
    ));
    assert!(matches!(
        tetrad_state_error(&metric, &[1.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0, 0.0], 0.0),
        Err(NonlocalRelativityError::InvalidMetricNormFloor(_))
    ));
}
