//! Analytic-oracle tests: invariants that hold by construction, independent of
//! the numerical reference fixtures.

use scirust_itd::operators::{spatial_mean, vorticity};
use scirust_itd::signature::structural_metrics;
use scirust_itd::{BoundaryMode, Field2, Geometry, StructuralWeights};

fn grid(n: usize, h: f64) -> (Vec<f64>, Geometry) {
    let coords: Vec<f64> = (0..n).map(|k| -1.0 + k as f64 * h).collect();
    (coords, Geometry::isotropic(h).unwrap())
}

/// A rigid rotation `(vx, vy) = (-y, x)` has curl exactly 2 everywhere; for a
/// field linear in the coordinates the second-order differences are exact, so
/// this holds to machine precision even at the edges.
#[test]
fn rigid_rotation_has_constant_vorticity() {
    let n = 11;
    let (c, geom) = grid(n, 0.2);
    let vx = Field2::from_fn(n, n, |i, _| -c[i]);
    let vy = Field2::from_fn(n, n, |_, j| c[j]);
    let w = vorticity(&vx, &vy, &geom, BoundaryMode::Finite).unwrap();
    for &value in w.as_slice()
    {
        assert!((value - 2.0).abs() < 1e-12, "vorticity {value} != 2");
    }
}

/// Pure expansion `(vx, vy) = (x, y)` is irrotational: curl 0 everywhere, and
/// hence the rotational intensity of `ω²·weight` is 0.
#[test]
fn expansion_is_irrotational() {
    let n = 11;
    let (c, geom) = grid(n, 0.2);
    let vx = Field2::from_fn(n, n, |_, j| c[j]);
    let vy = Field2::from_fn(n, n, |i, _| c[i]);
    let w = vorticity(&vx, &vy, &geom, BoundaryMode::Finite).unwrap();
    for &value in w.as_slice()
    {
        assert!(value.abs() < 1e-12, "vorticity {value} != 0");
    }
    let density = w.map(|v| v * v);
    let intensity = spatial_mean(&density, &geom, BoundaryMode::Finite).unwrap();
    assert!(intensity.abs() < 1e-20, "intensity {intensity} != 0");
}

/// A spatially constant vorticity field has zero heterogeneity, localization,
/// roughness and sign-mixing (all deviations, gradients and the mean/mean-abs
/// ratio collapse), hence a zero structure score.
#[test]
fn constant_field_has_zero_structure() {
    let n = 11;
    let (_, geom) = grid(n, 0.2);
    let omega = Field2::from_fn(n, n, |_, _| 1.5);
    let m = structural_metrics(
        &omega,
        &geom,
        None,
        None,
        0.5,
        StructuralWeights::default(),
        BoundaryMode::Finite,
    )
    .unwrap();
    assert!(m.heterogeneity.abs() < 1e-12);
    assert!(m.localization.abs() < 1e-12);
    assert!(m.roughness.abs() < 1e-12);
    assert!(m.sign_mixing.abs() < 1e-12);
    assert!(m.temporal_deformation.abs() < 1e-15);
    assert!(m.structure_score.abs() < 1e-12);
}

/// A zero field falls into the early-return branch: all metrics are zero.
#[test]
fn zero_field_has_zero_metrics() {
    let n = 11;
    let (_, geom) = grid(n, 0.2);
    let omega = Field2::zeros(n, n);
    let m = structural_metrics(
        &omega,
        &geom,
        None,
        None,
        0.5,
        StructuralWeights::default(),
        BoundaryMode::Finite,
    )
    .unwrap();
    assert_eq!(m.structure_score, 0.0);
    assert_eq!(m.heterogeneity, 0.0);
}
