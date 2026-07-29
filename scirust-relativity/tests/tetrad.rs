//! Validation of the orthonormal-frame (tetrad) construction.
//!
//! - A static observer in flat spacetime yields the standard basis.
//! - The frame is orthonormal, `g(e_a, e_b) = eta_ab`, on flat and curved
//!   backgrounds (checked for a boosted Schwarzschild observer).
//! - The frame spans the tangent space: any vector reconstructs exactly from
//!   its frame components.
//! - Non-timelike vectors and invalid floors are rejected with typed errors.

use scirust_relativity::{
    Metric, Minkowski, OrthonormalTetrad, RelativityError, Schwarzschild, orthonormal_tetrad,
    transport_along_segment,
};
use std::f64::consts::FRAC_PI_2;

const FLOOR: f64 = 1.0e-9;

fn inner(metric: &[[f64; 4]; 4], left: &[f64; 4], right: &[f64; 4]) -> f64 {
    let mut value = 0.0;
    for a in 0..4
    {
        for b in 0..4
        {
            value += metric[a][b] * left[a] * right[b];
        }
    }
    value
}

fn assert_orthonormal(metric: &[[f64; 4]; 4], legs: &[[f64; 4]; 4]) {
    for a in 0..4
    {
        for b in 0..4
        {
            let expected = if a == b
            {
                OrthonormalTetrad::<4>::signature(a)
            }
            else
            {
                0.0
            };
            assert!(
                (inner(metric, &legs[a], &legs[b]) - expected).abs() < 1.0e-10,
                "g(e_{a}, e_{b}) = {:.3e}, expected {expected}",
                inner(metric, &legs[a], &legs[b])
            );
        }
    }
}

#[test]
fn flat_static_observer_is_standard_basis() {
    let metric = Minkowski.components(&[0.0; 4]);
    let tetrad = orthonormal_tetrad(&metric, &[1.0, 0.0, 0.0, 0.0], FLOOR).unwrap();
    let expected = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    assert_eq!(*tetrad.legs(), expected);
}

#[test]
fn boosted_schwarzschild_observer_frame_is_orthonormal_and_reconstructs() {
    let spacetime = Schwarzschild::try_new(1.0).unwrap();
    let coordinates = [0.0, 10.0, FRAC_PI_2, 0.0];
    let metric = spacetime.components(&coordinates);

    // A boosted (t, r) observer: g(u, u) = -f a^2 + b^2 / f < 0 with f = 0.8.
    let four_velocity = [1.3, 0.1, 0.0, 0.0];
    assert!(inner(&metric, &four_velocity, &four_velocity) < 0.0);

    let tetrad = orthonormal_tetrad(&metric, &four_velocity, FLOOR).unwrap();
    assert_orthonormal(&metric, tetrad.legs());

    // Reconstruction: delta = sum_a (eta_aa g(delta, e_a)) e_a.
    let delta = [0.2, -0.1, 0.05, 0.3];
    let mut reconstructed = [0.0; 4];
    for (a, leg) in tetrad.legs().iter().enumerate()
    {
        let component = OrthonormalTetrad::<4>::signature(a) * inner(&metric, &delta, leg);
        for (slot, &leg_component) in reconstructed.iter_mut().zip(leg.iter())
        {
            *slot += component * leg_component;
        }
    }
    for (value, expected) in reconstructed.iter().zip(delta.iter())
    {
        assert!((value - expected).abs() < 1.0e-10);
    }
}

#[test]
fn tetrad_rejects_non_timelike_and_invalid_floor() {
    let metric = Minkowski.components(&[0.0; 4]);

    // Spacelike vector: g(u, u) = +1 > -floor.
    assert!(matches!(
        orthonormal_tetrad(&metric, &[0.0, 1.0, 0.0, 0.0], FLOOR),
        Err(RelativityError::NonTimelikeFrameVector { .. }),
    ));

    // Invalid floors.
    assert_eq!(
        orthonormal_tetrad(&metric, &[1.0, 0.0, 0.0, 0.0], 0.0),
        Err(RelativityError::InvalidTetradFloor(0.0)),
    );
    assert!(matches!(
        orthonormal_tetrad(&metric, &[1.0, 0.0, 0.0, 0.0], f64::NAN),
        Err(RelativityError::InvalidTetradFloor(_)),
    ));
}

#[test]
fn parallel_transport_preserves_frame_orthonormality() {
    // Parallel transport is a metric isometry (nabla g = 0): transporting every
    // leg of an orthonormal frame along a curve keeps the frame orthonormal, so
    // the transported frame is a valid observer frame carried along the path.
    let spacetime = Schwarzschild::try_new(1.0).unwrap();
    let start = [0.0, 10.0, FRAC_PI_2, 0.0];
    let end = [0.0, 8.0, FRAC_PI_2, 0.5];

    let start_metric = spacetime.components(&start);
    let tetrad = orthonormal_tetrad(&start_metric, &[1.3, 0.1, 0.0, 0.0], FLOOR).unwrap();

    // Transport each leg along the same coordinate segment with the shared RK4
    // engine.
    let substeps = 400;
    let mut transported = [[0.0; 4]; 4];
    for (leg, slot) in tetrad.legs().iter().zip(transported.iter_mut())
    {
        *slot = transport_along_segment(&spacetime, &start, &end, leg, substeps).unwrap();
    }

    // g(e_a, e_b) is preserved along the transport, so at the endpoint (using
    // the endpoint metric) the frame is still eta_ab. RK4 keeps this to near
    // machine precision here (~2e-14), far below the 1e-12 bound.
    let end_metric = spacetime.components(&end);
    let mut worst = 0.0_f64;
    for a in 0..4
    {
        for b in 0..4
        {
            let expected = if a == b
            {
                OrthonormalTetrad::<4>::signature(a)
            }
            else
            {
                0.0
            };
            worst =
                worst.max((inner(&end_metric, &transported[a], &transported[b]) - expected).abs());
        }
    }
    assert!(
        worst < 1.0e-12,
        "transported frame lost orthonormality: max defect {worst:.3e}"
    );
}
