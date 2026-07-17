//! Analytic-oracle tests: invariants that hold by construction, independent of
//! the numerical reference fixtures.

use scirust_itd::operators::{spatial_mean, vorticity};
use scirust_itd::signature::structural_metrics;
use scirust_itd::{
    BoundaryMode, Field2, Geometry, Interpolation, StructuralWeights, Trajectory,
    transport_previous_vorticity,
};

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

/// Zero transport velocity leaves the field unchanged: every departure point is
/// its own grid node, so the exact-snap path returns an identical copy — for
/// both interpolation schemes.
#[test]
fn zero_velocity_transport_is_identity() {
    let nx = 8;
    let ny = 6;
    let hx = 0.5;
    let hy = 0.4;
    let xc: Vec<f64> = (0..nx).map(|k| k as f64 * hx).collect();
    let yc: Vec<f64> = (0..ny).map(|k| k as f64 * hy).collect();
    let prev = Field2::from_fn(ny, nx, |i, j| {
        (0.7 * j as f64).sin() + 0.3 * (i as f64).cos()
    });
    let zero = |_x: f64, _y: f64, _t: f64| (0.0, 0.0);

    for interp in [
        Interpolation::BilinearPeriodic,
        Interpolation::CubicPeriodic,
    ]
    {
        let out = transport_previous_vorticity(
            &prev,
            &xc,
            &yc,
            0.0,
            0.5,
            zero,
            interp,
            Trajectory::MidpointTimeVelocity,
        )
        .unwrap();
        assert_eq!(out, prev, "zero-velocity transport must be identity");
    }
}

/// A constant velocity advecting exactly one cell per step shifts the field by
/// one grid column: the departure points land exactly on the neighbouring
/// nodes, so periodic sampling reproduces the shifted field to machine
/// precision.
#[test]
fn integer_cell_shift_transport() {
    let nx = 8;
    let ny = 5;
    let h = 0.5;
    let dt = 1.0;
    let xc: Vec<f64> = (0..nx).map(|k| k as f64 * h).collect();
    let yc: Vec<f64> = (0..ny).map(|k| k as f64 * h).collect();
    let prev = Field2::from_fn(ny, nx, |i, j| (i * nx + j) as f64);
    // vx = h/dt advects one full cell to the left over one step.
    let drift = move |_x: f64, _y: f64, _t: f64| (h / dt, 0.0);

    let out = transport_previous_vorticity(
        &prev,
        &xc,
        &yc,
        0.0,
        dt,
        drift,
        Interpolation::BilinearPeriodic,
        Trajectory::MidpointTimeVelocity,
    )
    .unwrap();

    for i in 0..ny
    {
        for j in 0..nx
        {
            let src = (j + nx - 1) % nx;
            assert!(
                (out.get(i, j) - prev.get(i, src)).abs() < 1e-12,
                "cell ({i},{j}) not shifted"
            );
        }
    }
}
