//! Validation of covector and rank-2 covariant-tensor parallel transport.
//!
//! - Flat spacetime: transport is the exact identity.
//! - Metric compatibility (`nabla g = 0`) gives two exact identities to the
//!   integration tolerance:
//!   - the metric transports to itself, `transport(g_start) = g_end`;
//!   - index lowering commutes with transport,
//!     `lower(transport(V)) = transport(lower(V))`;
//!   - and the covector-vector contraction `W_a V^a` is preserved when both are
//!     transported.

use scirust_relativity::{
    Connection, DeSitter, Metric, Minkowski, RelativityError, Schwarzschild,
    transport_along_segment, transport_covariant_tensor_along_polyline,
    transport_covariant_tensor_along_segment, transport_covector_along_polyline,
    transport_covector_along_segment,
};
use std::f64::consts::FRAC_PI_2;

fn lower(metric: &[[f64; 4]; 4], vector: &[f64; 4]) -> [f64; 4] {
    let mut covector = [0.0; 4];
    for (a, slot) in covector.iter_mut().enumerate()
    {
        for (b, &component) in vector.iter().enumerate()
        {
            *slot += metric[a][b] * component;
        }
    }
    covector
}

fn max_tensor_gap(left: &[[f64; 4]; 4], right: &[[f64; 4]; 4]) -> f64 {
    let mut worst = 0.0_f64;
    for a in 0..4
    {
        for b in 0..4
        {
            worst = worst.max((left[a][b] - right[a][b]).abs());
        }
    }
    worst
}

// --------------------------------------------------------------------------
// Flat spacetime: exact identity
// --------------------------------------------------------------------------

#[test]
fn flat_covector_transport_is_identity() {
    let covector = [0.3, -0.5, 0.2, 0.1];
    let path = [
        [0.0, 1.0, 2.0, 3.0],
        [1.0, 0.0, 1.0, 2.0],
        [-1.0, 2.0, 0.0, 1.0],
    ];
    let transported = transport_covector_along_polyline(&Minkowski, &path, &covector, 8).unwrap();
    assert_eq!(transported, covector);
}

#[test]
fn flat_tensor_transport_is_identity() {
    let tensor = [
        [1.0, 0.2, 0.0, 0.0],
        [0.2, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.1],
        [0.0, 0.0, 0.1, 4.0],
    ];
    let path = [[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 1.0, 2.0]];
    let transported =
        transport_covariant_tensor_along_polyline(&Minkowski, &path, &tensor, 8).unwrap();
    assert_eq!(transported, tensor);
}

// --------------------------------------------------------------------------
// Metric compatibility
// --------------------------------------------------------------------------

fn assert_metric_self_transports<B: Metric<4> + Connection<4>>(
    background: &B,
    start: [f64; 4],
    end: [f64; 4],
) {
    let metric_start = background.components(&start);
    let metric_end = background.components(&end);
    let transported =
        transport_covariant_tensor_along_segment(background, &start, &end, &metric_start, 200)
            .unwrap();
    assert!(
        max_tensor_gap(&transported, &metric_end) < 1.0e-8,
        "metric self-transport gap {:.3e}",
        max_tensor_gap(&transported, &metric_end)
    );
}

#[test]
fn metric_transports_to_itself_schwarzschild() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    assert_metric_self_transports(
        &background,
        [0.0, 10.0, FRAC_PI_2, 0.0],
        [0.0, 8.0, FRAC_PI_2, 0.5],
    );
}

#[test]
fn metric_transports_to_itself_de_sitter() {
    let background = DeSitter::try_new(0.05).unwrap();
    assert_metric_self_transports(
        &background,
        [0.0, 3.0, FRAC_PI_2, 0.0],
        [0.0, 4.0, 1.3, 0.4],
    );
}

#[test]
fn index_lowering_commutes_with_transport() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let start = [0.0, 10.0, FRAC_PI_2, 0.0];
    let end = [0.0, 7.0, FRAC_PI_2, 0.6];
    let vector = [0.2, 0.1, 0.03, 0.02];

    // Transport the vector, then lower with the endpoint metric.
    let vector_transported =
        transport_along_segment(&background, &start, &end, &vector, 400).unwrap();
    let lowered_after = lower(&background.components(&end), &vector_transported);

    // Lower at the start, then transport the covector.
    let lowered_before = lower(&background.components(&start), &vector);
    let covector_transported =
        transport_covector_along_segment(&background, &start, &end, &lowered_before, 400).unwrap();

    for i in 0..4
    {
        assert!((lowered_after[i] - covector_transported[i]).abs() < 1.0e-9);
    }
}

#[test]
fn covector_vector_contraction_is_preserved() {
    // W_a V^a is a scalar; if both are parallel-transported it is constant.
    let background = DeSitter::try_new(0.05).unwrap();
    let start = [0.0, 3.0, FRAC_PI_2, 0.0];
    let end = [0.0, 4.0, 1.2, 0.5];
    let vector = [0.2, 0.1, 0.05, 0.02];
    let covector = [0.3, -0.1, 0.2, 0.05];

    let contraction_start: f64 = (0..4).map(|i| covector[i] * vector[i]).sum();

    let vector_end = transport_along_segment(&background, &start, &end, &vector, 200).unwrap();
    let covector_end =
        transport_covector_along_segment(&background, &start, &end, &covector, 200).unwrap();
    let contraction_end: f64 = (0..4).map(|i| covector_end[i] * vector_end[i]).sum();

    assert!((contraction_end - contraction_start).abs() < 1.0e-8);
}

// --------------------------------------------------------------------------
// Error paths
// --------------------------------------------------------------------------

#[test]
fn covariant_transport_reports_typed_errors() {
    let background = Schwarzschild::try_new(1.0).unwrap();
    let start = [0.0, 8.0, FRAC_PI_2, 0.0];
    let end = [0.0, 7.0, FRAC_PI_2, 0.1];
    let covector = [0.1, 0.2, 0.0, 0.0];
    let tensor = [[1.0, 0.0, 0.0, 0.0]; 4];

    assert_eq!(
        transport_covector_along_segment(&background, &start, &end, &covector, 0),
        Err(RelativityError::InvalidTransportResolution),
    );
    assert_eq!(
        transport_covector_along_segment(
            &background,
            &[0.0, f64::NAN, FRAC_PI_2, 0.0],
            &end,
            &covector,
            10,
        ),
        Err(RelativityError::NonFiniteCoordinate(1)),
    );
    assert_eq!(
        transport_covariant_tensor_along_segment(&background, &start, &end, &tensor, 0),
        Err(RelativityError::InvalidTransportResolution),
    );
}
