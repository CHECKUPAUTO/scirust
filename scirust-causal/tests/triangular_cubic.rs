use scirust_causal::{CausalError, TriangularCubicFlow};
use scirust_solvers::Matrix;

fn test_weights() -> Matrix {
    Matrix::from_row_major(
        4,
        4,
        vec![
            0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, -0.50, 0.75, 0.0, 0.0, 0.20, -0.10, 0.40, 0.0,
        ],
    )
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    let error = (actual - expected).abs();

    assert!(
        error <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, \
         error={error:.17e}, tolerance={tolerance:.17e}"
    );
}

fn scalar_objective(flow: &TriangularCubicFlow, x: &[f64], upstream: &[f64]) -> f64 {
    flow.forward(x)
        .expect("forward evaluation must succeed")
        .iter()
        .zip(upstream)
        .map(|(output, gradient)| output * gradient)
        .sum()
}

#[test]
fn rejects_non_square_weights() {
    let weights = Matrix::zeros(2, 3);

    assert_eq!(
        TriangularCubicFlow::new(weights),
        Err(CausalError::NotSquare { rows: 2, cols: 3 })
    );
}

#[test]
fn rejects_nonzero_diagonal_or_upper_triangle() {
    let diagonal = Matrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 0.0]);

    assert!(matches!(
        TriangularCubicFlow::new(diagonal),
        Err(CausalError::NonStrictLowerTriangular { row: 0, col: 0, .. })
    ));

    let upper = Matrix::from_row_major(2, 2, vec![0.0, 0.5, 0.0, 0.0]);

    assert!(matches!(
        TriangularCubicFlow::new(upper),
        Err(CausalError::NonStrictLowerTriangular { row: 0, col: 1, .. })
    ));
}

#[test]
fn exact_inverse_round_trip_is_numerically_stable() {
    let flow = TriangularCubicFlow::new(test_weights()).unwrap();
    let x = vec![0.5, -1.25, 0.75, 2.0];

    let y = flow.forward(&x).unwrap();
    let reconstructed = flow.inverse(&y).unwrap();

    for (actual, expected) in reconstructed.iter().zip(&x)
    {
        assert_close(*actual, *expected, 1.0e-12);
    }
}

#[test]
fn inverse_forward_round_trip_is_numerically_stable() {
    let flow = TriangularCubicFlow::new(test_weights()).unwrap();
    let y = vec![-0.25, 1.5, -0.75, 0.125];

    let x = flow.inverse(&y).unwrap();
    let reconstructed = flow.forward(&x).unwrap();

    for (actual, expected) in reconstructed.iter().zip(&y)
    {
        assert_close(*actual, *expected, 1.0e-12);
    }
}

#[test]
fn jacobian_is_unit_lower_triangular() {
    let flow = TriangularCubicFlow::new(test_weights()).unwrap();
    let x = vec![0.5, -1.25, 0.75, 2.0];
    let jacobian = flow.jacobian(&x).unwrap();

    for row in 0..flow.dim()
    {
        assert_close(jacobian[(row, row)], 1.0, 0.0);

        for col in (row + 1)..flow.dim()
        {
            assert_close(jacobian[(row, col)], 0.0, 0.0);
        }
    }

    let determinant = jacobian.determinant().unwrap();
    assert_close(determinant, 1.0, 1.0e-12);
    assert_close(flow.log_abs_det_jacobian(), 0.0, 0.0);
}

#[test]
fn analytical_input_gradient_matches_central_difference() {
    let flow = TriangularCubicFlow::new(test_weights()).unwrap();
    let x = vec![0.5, -1.25, 0.75, 2.0];
    let upstream = vec![0.2, -0.7, 1.1, 0.4];
    let epsilon = 1.0e-6;

    let (analytical, _) = flow.backward(&x, &upstream).unwrap();

    for coordinate in 0..x.len()
    {
        let mut plus = x.clone();
        let mut minus = x.clone();

        plus[coordinate] += epsilon;
        minus[coordinate] -= epsilon;

        let numerical = (scalar_objective(&flow, &plus, &upstream)
            - scalar_objective(&flow, &minus, &upstream))
            / (2.0 * epsilon);

        assert_close(analytical[coordinate], numerical, 2.0e-8);
    }
}

#[test]
fn analytical_weight_gradient_matches_central_difference() {
    let flow = TriangularCubicFlow::new(test_weights()).unwrap();
    let x = vec![0.5, -1.25, 0.75, 2.0];
    let upstream = vec![0.2, -0.7, 1.1, 0.4];
    let epsilon = 1.0e-6;
    let (_, analytical) = flow.backward(&x, &upstream).unwrap();

    let dim = flow.dim();
    let base = flow.weights().data().to_vec();

    for row in 0..dim
    {
        for col in 0..row
        {
            let index = row * dim + col;
            let mut plus_data = base.clone();
            let mut minus_data = base.clone();

            plus_data[index] += epsilon;
            minus_data[index] -= epsilon;

            let plus_flow = TriangularCubicFlow::from_row_major(dim, plus_data).unwrap();
            let minus_flow = TriangularCubicFlow::from_row_major(dim, minus_data).unwrap();

            let numerical = (scalar_objective(&plus_flow, &x, &upstream)
                - scalar_objective(&minus_flow, &x, &upstream))
                / (2.0 * epsilon);

            assert_close(analytical[(row, col)], numerical, 2.0e-8);
        }
    }
}

#[test]
fn evaluation_is_bit_reproducible_on_one_build() {
    let flow = TriangularCubicFlow::new(test_weights()).unwrap();
    let x = vec![0.5, -1.25, 0.75, 2.0];

    let first = flow.forward(&x).unwrap();
    let second = flow.forward(&x).unwrap();

    assert_eq!(first, second);
}

// ─── Dimension and structure rejection ──────────────────────────────────────

#[test]
fn rejects_zero_dimension() {
    assert_eq!(
        TriangularCubicFlow::new(Matrix::zeros(0, 0)),
        Err(CausalError::ZeroDimension)
    );
}

#[test]
fn rejects_wrong_row_major_storage_length() {
    let result = TriangularCubicFlow::from_row_major(2, vec![1.0, 2.0, 3.0]);
    assert!(matches!(result, Err(CausalError::DimensionMismatch { .. })));
}

// ─── Non-finite weight rejection ────────────────────────────────────────────

fn non_finite_weight_test(value: f64) {
    let data = vec![0.0, 0.0, value, 0.0];
    let result = TriangularCubicFlow::from_row_major(2, data);
    assert!(matches!(result, Err(CausalError::NonFiniteWeight { .. })));
}

#[test]
fn rejects_nan_weight() {
    non_finite_weight_test(f64::NAN);
}

#[test]
fn rejects_pos_inf_weight() {
    non_finite_weight_test(f64::INFINITY);
}

#[test]
fn rejects_neg_inf_weight() {
    non_finite_weight_test(f64::NEG_INFINITY);
}

// ─── Non-finite input rejection ─────────────────────────────────────────────

fn assert_non_finite_input<F>(operation: &str, call: F)
where
    F: Fn() -> Result<(), CausalError>,
{
    let result = call();
    assert!(
        matches!(&result, Err(CausalError::NonFiniteInput { .. })),
        "expected NonFiniteInput for {operation}: got {result:?}"
    );
}

fn non_finite_flow() -> TriangularCubicFlow {
    TriangularCubicFlow::from_row_major(2, vec![0.0, 0.0, 1.0, 0.0]).unwrap()
}

#[test]
fn rejects_non_finite_input_forward() {
    let flow = non_finite_flow();
    assert_non_finite_input("forward", || flow.forward(&[f64::NAN, 0.0]).map(|_| ()));
    assert_non_finite_input("forward", || {
        flow.forward(&[f64::INFINITY, 0.0]).map(|_| ())
    });
    assert_non_finite_input("forward", || {
        flow.forward(&[0.0, f64::NEG_INFINITY]).map(|_| ())
    });
}

#[test]
fn rejects_non_finite_input_inverse() {
    let flow = non_finite_flow();
    assert_non_finite_input("inverse", || flow.inverse(&[f64::NAN, 0.0]).map(|_| ()));
}

#[test]
fn rejects_non_finite_input_jacobian() {
    let flow = non_finite_flow();
    assert_non_finite_input("jacobian", || {
        flow.jacobian(&[f64::INFINITY, 0.0]).map(|_| ())
    });
}

#[test]
fn rejects_non_finite_input_backward() {
    let flow = non_finite_flow();
    assert_non_finite_input("backward x", || {
        flow.backward(&[f64::NAN, 0.0], &[1.0, 1.0]).map(|_| ())
    });
    assert_non_finite_input("backward upstream", || {
        flow.backward(&[1.0, 1.0], &[0.0, f64::NEG_INFINITY])
            .map(|_| ())
    });
}

// ─── Overflow rejection ─────────────────────────────────────────────────────

#[test]
fn rejects_overflow_cubic() {
    let flow = non_finite_flow();
    // A[1,0] = 1.0, x[0] = 1e103  =>  z[1] = 1e103  =>  cube = 1e309 > f64::MAX
    let x = vec![1e103, 0.0];
    let result = flow.forward(&x);
    assert!(
        matches!(&result, Err(CausalError::NonFiniteComputation { .. })),
        "expected overflow error, got {result:?}"
    );
}

// ─── Identity behavior ──────────────────────────────────────────────────────

#[test]
fn identity_when_all_weights_zero() {
    let flow = TriangularCubicFlow::from_row_major(1, vec![0.0]).unwrap();
    let x = vec![-3.25];
    let y = flow.forward(&x).unwrap();
    assert_eq!(y, x);

    let reconstructed = flow.inverse(&y).unwrap();
    assert_eq!(reconstructed, x);

    let j = flow.jacobian(&x).unwrap();
    assert_eq!(j[(0, 0)], 1.0);
}

// ─── Hand-derived 2D oracle ─────────────────────────────────────────────────

#[test]
fn hand_derived_two_dimensional_oracle() {
    // A = [[0, 0], [2, 0]],  x = [3, 5]
    // z = A x = [0, 6]
    // y = x + z^{∘3} = [3, 5 + 216] = [3, 221]
    let flow = TriangularCubicFlow::from_row_major(2, vec![0.0, 0.0, 2.0, 0.0]).unwrap();
    let x = vec![3.0, 5.0];
    let expected_y = vec![3.0, 221.0];

    let y = flow.forward(&x).unwrap();
    for (a, e) in y.iter().zip(&expected_y)
    {
        assert_close(*a, *e, 0.0);
    }

    let reconstructed = flow.inverse(&y).unwrap();
    for (a, e) in reconstructed.iter().zip(&x)
    {
        assert_close(*a, *e, 0.0);
    }
}

// ─── Jacobian via finite differences on 2D oracle ───────────────────────────

#[test]
fn jacobian_matches_central_difference() {
    // 2D oracle: A = [[0, 0], [2, 0]],  x = [3, 5]
    // Analytical Jacobian:
    //   J[0,0] = 1,  J[0,1] = 0
    //   J[1,0] = 3 * 6^2 * 2 = 216,  J[1,1] = 1
    let flow = TriangularCubicFlow::from_row_major(2, vec![0.0, 0.0, 2.0, 0.0]).unwrap();
    let x = vec![3.0, 5.0];
    let epsilon = 1.0e-6;

    let j_analytical = flow.jacobian(&x).unwrap();

    for col in 0..2
    {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[col] += epsilon;
        x_minus[col] -= epsilon;

        let y_plus = flow.forward(&x_plus).unwrap();
        let y_minus = flow.forward(&x_minus).unwrap();

        for row in 0..2
        {
            let numerical = (y_plus[row] - y_minus[row]) / (2.0 * epsilon);
            let analytical = j_analytical[(row, col)];
            assert_close(analytical, numerical, 2.0e-7);
        }
    }

    // Check specific entries
    assert_close(j_analytical[(0, 0)], 1.0, 0.0);
    assert_close(j_analytical[(0, 1)], 0.0, 0.0);
    assert_close(j_analytical[(1, 0)], 216.0, 1.0e-12);
    assert_close(j_analytical[(1, 1)], 1.0, 0.0);
}

// ─── Inverse round-trip over several vectors ────────────────────────────────

#[test]
fn inverse_round_trip_multiple_vectors() {
    let flow = TriangularCubicFlow::new(test_weights()).unwrap();
    let vectors = vec![
        vec![0.0, 0.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0, 1.0],
        vec![0.5, -1.25, 0.75, 2.0],
        vec![-3.0, 2.5, -1.0, 0.25],
        vec![100.0, -50.0, 25.0, -10.0],
    ];

    for x in &vectors
    {
        let y = flow.forward(x).unwrap();
        let reconstructed = flow.inverse(&y).unwrap();
        for (a, e) in reconstructed.iter().zip(x.iter())
        {
            assert_close(*a, *e, 1.0e-12);
        }
    }
}

// ─── Gradient mask: diagonal and upper triangle are zero ────────────────────

#[test]
fn backward_gradient_mask_upper_triangular_zero() {
    let flow = TriangularCubicFlow::new(test_weights()).unwrap();
    let x = vec![0.5, -1.25, 0.75, 2.0];
    let upstream = vec![0.2, -0.7, 1.1, 0.4];
    let (_, grad_weights) = flow.backward(&x, &upstream).unwrap();
    let dim = flow.dim();

    for row in 0..dim
    {
        for col in row..dim
        {
            assert_eq!(
                grad_weights[(row, col)],
                0.0,
                "grad_weights[({row}, {col})] must be zero"
            );
        }
    }
}

// ─── Public API returns errors, never panics ────────────────────────────────

#[test]
fn public_api_returns_error_not_panic() {
    // New with non-square matrix
    assert!(TriangularCubicFlow::new(Matrix::zeros(2, 3)).is_err());

    // New with zero dimension
    assert!(TriangularCubicFlow::new(Matrix::zeros(0, 0)).is_err());

    // from_row_major with wrong length
    assert!(TriangularCubicFlow::from_row_major(2, vec![1.0]).is_err());

    // forward with wrong dimension
    let flow = non_finite_flow();
    assert!(flow.forward(&[1.0]).is_err());

    // inverse with wrong dimension
    assert!(flow.inverse(&[1.0, 2.0, 3.0]).is_err());

    // jacobian with wrong dimension
    assert!(flow.jacobian(&[]).is_err());

    // backward with wrong dimension
    assert!(flow.backward(&[1.0, 2.0, 3.0], &[1.0, 2.0]).is_err());
    assert!(flow.backward(&[1.0, 2.0], &[1.0, 2.0, 3.0]).is_err());

    // backward with mismatched x and upstream dimensions
    assert!(flow.backward(&[1.0, 2.0], &[1.0, 2.0]).is_ok());
}
