//! Analytic-oracle tests: invariants that hold by construction, independent of
//! the numerical reference fixtures.

use scirust_itd::operators::{spatial_mean, vorticity};
use scirust_itd::signature::structural_metrics;
use scirust_itd::{
    BoundaryMode, Field2, Geometry, Interpolation, Scenario, StructuralWeights, Trajectory,
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

// --- field-geometry modules ----------------------------------------------

use scirust_itd::covariance::{
    galilean_source_coordinates, inverse_scale_coordinates, scale_length, subtract_frame_velocity,
};
use scirust_itd::material::material_vorticity_interval;
use scirust_itd::multiscale::{MultiscaleReference, derive_multiscale_profile};
use scirust_itd::transforms::{BilinearTransformPlan, Orthogonal2};

/// The identity rotation maps every node onto itself, so the plan uses the exact
/// node permutation and returns the field byte-for-byte unchanged.
#[test]
fn identity_transform_is_exact_and_unchanged() {
    let coords = vec![0.0, 1.0, 2.0, 3.0];
    let q = Orthogonal2::rotation(0.0).unwrap();
    let plan = BilinearTransformPlan::new(&coords, &coords, q, [0.0, 0.0], 0.0).unwrap();
    assert!(plan.uses_exact_node_map());
    let f = Field2::from_fn(4, 4, |i, j| (i * 4 + j) as f64 * 0.37 - 1.0);
    assert_eq!(plan.transform_scalar(&f).unwrap(), f);
}

/// A reflection is orthogonal with determinant −1 (not a proper rotation); a
/// shear is rejected as non-orthogonal.
#[test]
fn reflection_is_orthogonal_but_not_rotation() {
    let q = Orthogonal2::new([[1.0, 0.0], [0.0, -1.0]]).unwrap();
    assert!((q.determinant() + 1.0).abs() < 1e-15);
    assert!(!q.is_rotation());
    assert!(Orthogonal2::new([[1.0, 0.5], [0.0, 1.0]]).is_err());
}

/// `x = o + (x' − o)/a` inverts `x' = o + a(x − o)` exactly.
#[test]
fn spatial_scaling_round_trip() {
    let xs = vec![0.1, 0.9, 1.7, 2.5];
    let ys = vec![-0.3, 0.4, 1.2, 2.0];
    let a = 2.3;
    let origin = [0.5, -0.2];
    let (sx, sy) = inverse_scale_coordinates(&xs, &ys, a, origin).unwrap();
    for k in 0..xs.len()
    {
        let fwd_x = origin[0] + a * (sx[k] - origin[0]);
        let fwd_y = origin[1] + a * (sy[k] - origin[1]);
        assert!((fwd_x - xs[k]).abs() < 1e-12);
        assert!((fwd_y - ys[k]).abs() < 1e-12);
    }
    assert!((scale_length(1.5, a).unwrap() - a * 1.5).abs() < 1e-15);
}

/// A Galilean boost evaluated at its reference time leaves coordinates fixed, and
/// subtracting a zero frame velocity leaves the field unchanged.
#[test]
fn galilean_at_reference_time_is_identity() {
    let xs = vec![0.0, 1.0, 2.0];
    let ys = vec![0.5, 1.5, 2.5];
    let (sx, sy) = galilean_source_coordinates(&xs, &ys, 1.0, [0.7, -0.3], 1.0).unwrap();
    assert_eq!(sx, xs);
    assert_eq!(sy, ys);
    let vx = Field2::from_fn(3, 3, |i, j| (i + j) as f64);
    let vy = Field2::from_fn(3, 3, |i, j| i as f64 - j as f64);
    let (ux, uy) = subtract_frame_velocity(&vx, &vy, [0.0, 0.0]).unwrap();
    assert_eq!(ux, vx);
    assert_eq!(uy, vy);
}

/// The material tendency is, by construction, the exact sum of the Eulerian and
/// advective tendencies at every node.
#[test]
fn material_tendency_is_sum_of_parts() {
    let n = 5;
    let geom = Geometry::isotropic(0.25).unwrap();
    let prev = Field2::from_fn(n, n, |i, j| (0.3 * i as f64).sin() * (0.2 * j as f64).cos());
    let cur = Field2::from_fn(n, n, |i, j| {
        (0.3 * i as f64 + 0.1).sin() * (0.2 * j as f64).cos()
    });
    let vx = Field2::from_fn(n, n, |_, j| 0.2 + 0.05 * j as f64);
    let vy = Field2::from_fn(n, n, |i, _| -0.1 + 0.05 * i as f64);
    let r = material_vorticity_interval(&prev, &cur, &vx, &vy, &geom, 0.5, BoundaryMode::Finite)
        .unwrap();
    for k in 0..r.material_tendency.as_slice().len()
    {
        let m = r.material_tendency.as_slice()[k];
        let t = r.temporal_tendency.as_slice()[k];
        let a = r.advective_tendency.as_slice()[k];
        assert!((m - (t + a)).abs() < 1e-15, "cell {k}");
    }
}

/// The raw roughness index is exactly linear in the structural length, and the
/// temporal-deformation signature component is scale-independent.
#[test]
fn multiscale_roughness_scales_linearly() {
    let reference = MultiscaleReference {
        intensity_rate: vec![1.0, 1.2, 0.9, 1.1],
        heterogeneity: vec![0.2, 0.3, 0.25, 0.28],
        localization: vec![0.5, 0.4, 0.6, 0.55],
        unit_roughness: vec![0.7, 0.8, 0.75, 0.9],
        sign_mixing: vec![0.1, 0.15, 0.12, 0.14],
        temporal_deformation_interval: vec![0.3, 0.35, 0.32],
        interval_dt: vec![0.25, 0.25, 0.25],
        weights: [0.2, 0.2, 0.2, 0.2, 0.2],
        intensity_index: 1.05,
        temporal_deformation_index: 0.33,
    };
    let lengths = vec![1.0, 2.0, 3.0];
    let profile = derive_multiscale_profile(&reference, &lengths).unwrap();
    let base = profile.raw_roughness_indices[0];
    for (k, &ell) in lengths.iter().enumerate()
    {
        assert!((profile.raw_roughness_indices[k] - ell * base).abs() < 1e-12);
    }
    for k in 1..lengths.len()
    {
        assert!((profile.signatures[k][4] - profile.signatures[0][4]).abs() < 1e-15);
    }
}

use scirust_itd::material::{
    AdvectionSource, interpolate_interval_series_to_nodes, simulate_material_deformation,
    simulate_material_deformation_with_advection,
};
use scirust_itd::simulate::{SimConfig, simulate};

/// Interval-to-node interpolation: constant extrapolation at both ends, and on a
/// uniform grid each interior node is the mean of its two adjacent intervals. A
/// constant series stays constant.
#[test]
fn interval_interpolation_properties() {
    let times = [0.0, 1.0, 2.0, 3.0, 4.0];
    let iv = [2.0, 4.0, 8.0, 6.0];
    let nodes = interpolate_interval_series_to_nodes(&times, &iv).unwrap();
    assert_eq!(nodes.len(), 5);
    assert!(
        (nodes[0] - iv[0]).abs() < 1e-15,
        "left constant extrapolation"
    );
    assert!(
        (nodes[4] - iv[3]).abs() < 1e-15,
        "right constant extrapolation"
    );
    for k in 1..4
    {
        let mean = 0.5 * (iv[k - 1] + iv[k]);
        assert!(
            (nodes[k] - mean).abs() < 1e-15,
            "node {k} is the adjacent mean"
        );
    }

    let constant = [3.5; 4];
    let nodes = interpolate_interval_series_to_nodes(&times, &constant).unwrap();
    assert!(nodes.iter().all(|&v| (v - 3.5).abs() < 1e-15));
}

/// With a zero advection field the advective rate vanishes, the material rate
/// equals the Eulerian rate, the consistency certification holds, and the
/// baseline is byte-identical to a direct `simulate` run.
#[test]
fn material_deformation_zero_advection() {
    let n = 9;
    let xc: Vec<f64> = (0..n).map(|k| -1.0 + 0.25 * k as f64).collect();
    let yc = xc.clone();
    let times = [0.0, 0.4, 1.0, 1.5];
    let geometry = Geometry::isotropic(0.25).unwrap();
    let sim = SimConfig::default();
    let velocity = |x: &[f64], y: &[f64], t: f64| Scenario::Multi.velocity(x, y, t);
    let curvature = |x: &[f64], y: &[f64], _t: f64| Field2::from_fn(y.len(), x.len(), |_, _| 0.1);

    let r = simulate_material_deformation_with_advection(
        "multi",
        velocity,
        curvature,
        |x: &[f64], y: &[f64], _t: f64| {
            (
                Field2::zeros(y.len(), x.len()),
                Field2::zeros(y.len(), x.len()),
            )
        },
        &xc,
        &yc,
        &times,
        &geometry,
        0.5,
        &sim,
    )
    .unwrap();
    assert_eq!(r.advection_source, AdvectionSource::AdvectionVelocityField);
    assert!(r.advective_rate_interval.iter().all(|&v| v.abs() < 1e-15));
    for j in 0..r.eulerian_rate_interval.len()
    {
        assert!(
            (r.material_deformation_interval[j] - r.eulerian_rate_interval[j]).abs() < 1e-12,
            "zero advection: material == eulerian at interval {j}"
        );
    }
    assert!(r.eulerian_consistency_error < 1e-12);

    let direct = simulate(
        "multi", velocity, curvature, &xc, &yc, &times, &geometry, 0.5, &sim,
    )
    .unwrap();
    assert_eq!(
        r.baseline, direct,
        "baseline must be the untouched eulerian run"
    );

    // The default-advection wrapper reports the reference default source.
    let r = simulate_material_deformation(
        "multi", velocity, curvature, &xc, &yc, &times, &geometry, 0.5, &sim,
    )
    .unwrap();
    assert_eq!(r.advection_source, AdvectionSource::VelocityField);
    assert!(r.eulerian_consistency_error < 1e-12);
}
