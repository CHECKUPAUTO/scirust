use scirust_causal::{CausalError, PolynomialAcyclicity};
use scirust_solvers::Matrix;

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, \
         error={error:.17e}, tolerance={tolerance:.17e}"
    );
}

// ─── Zero matrix gives zero ──────────────────────────────────────────────────

#[test]
fn zero_matrix_gives_zero() {
    let a = Matrix::zeros(3, 3);
    let h = PolynomialAcyclicity::value(&a).unwrap();
    assert_eq!(h, 0.0);

    let (h2, grad) = PolynomialAcyclicity::value_and_gradient(&a).unwrap();
    assert_eq!(h2, 0.0);
    for i in 0..3
    {
        for j in 0..3
        {
            assert_eq!(grad[(i, j)], 0.0);
        }
    }
}

// ─── Strictly triangular matrices give zero (h = 0 since B = 0 for strictly triangular) ──

#[test]
fn strictly_triangular_gives_zero() {
    let a = Matrix::from_row_major(3, 3, vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0]);
    let h = PolynomialAcyclicity::value(&a).unwrap();
    assert_close(h, 0.0, 1.0e-15);

    let (h2, grad) = PolynomialAcyclicity::value_and_gradient(&a).unwrap();
    assert_close(h2, 0.0, 1.0e-15);

    for i in 0..3
    {
        for j in 0..3
        {
            assert_eq!(grad[(i, j)], 0.0);
        }
    }
}

// ─── Two-node directed cycle ─────────────────────────────────────────────────

#[test]
fn two_node_cycle() {
    // A = [[0, 1], [1, 0]]
    // B = A∘A = [[0, 1], [1, 0]]
    // B^1 = [[0, 1], [1, 0]], trace = 0
    // B^2 = [[1, 0], [0, 1]], trace = 2
    // h = trace(B^2)/2! = 2/2 = 1
    let a = Matrix::from_row_major(2, 2, vec![0.0, 1.0, 1.0, 0.0]);

    let h = PolynomialAcyclicity::value(&a).unwrap();
    assert_close(h, 1.0, 1.0e-14);

    let (h2, _grad) = PolynomialAcyclicity::value_and_gradient(&a).unwrap();
    assert_close(h2, 1.0, 1.0e-14);
}

// ─── Three-node directed cycle ───────────────────────────────────────────────

#[test]
fn three_node_cycle() {
    // A = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    // B = A∘A (all 1s structurally)
    // B^1 = cyclic shift, trace = 0
    // B^2 = double cyclic shift, trace = 0
    // B^3 = identity, trace = 3
    // h = trace(B^3)/3! = 3/6 = 0.5
    let a = Matrix::from_row_major(3, 3, vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);

    let h = PolynomialAcyclicity::value(&a).unwrap();
    assert_close(h, 0.5, 1.0e-14);

    let (h2, _grad) = PolynomialAcyclicity::value_and_gradient(&a).unwrap();
    assert_close(h2, 0.5, 1.0e-14);
}

// ─── Diagonal self-loop ──────────────────────────────────────────────────────

#[test]
fn diagonal_self_loop() {
    // A = [[2, 0], [0, 0]]
    // B = [[4, 0], [0, 0]]
    // B^1 = [[4, 0], [0, 0]], trace = 4
    // B^2 = [[16, 0], [0, 0]], trace = 16
    // h = 4/1! + 16/2! = 4 + 8 = 12
    let a = Matrix::from_row_major(2, 2, vec![2.0, 0.0, 0.0, 0.0]);

    let h = PolynomialAcyclicity::value(&a).unwrap();
    assert_close(h, 12.0, 1.0e-12);

    let (h2, _grad) = PolynomialAcyclicity::value_and_gradient(&a).unwrap();
    assert_close(h2, 12.0, 1.0e-12);
}

// ─── Gradient against finite differences ─────────────────────────────────────

#[test]
fn gradient_matches_finite_difference() {
    let a = Matrix::from_row_major(2, 2, vec![0.5, 1.0, -0.5, 0.3]);
    let epsilon = 1.0e-6;

    let (_, analytical) = PolynomialAcyclicity::value_and_gradient(&a).unwrap();
    let dim = a.rows();

    for row in 0..dim
    {
        for col in 0..dim
        {
            let mut plus_data = a.data().to_vec();
            let mut minus_data = a.data().to_vec();
            let idx = row * dim + col;
            plus_data[idx] += epsilon;
            minus_data[idx] -= epsilon;

            let plus = Matrix::from_row_major(dim, dim, plus_data);
            let minus = Matrix::from_row_major(dim, dim, minus_data);

            let val_plus = PolynomialAcyclicity::value(&plus).unwrap();
            let val_minus = PolynomialAcyclicity::value(&minus).unwrap();
            let numerical = (val_plus - val_minus) / (2.0 * epsilon);

            assert_close(analytical[(row, col)], numerical, 2.0e-7);
        }
    }
}

// ─── Asymmetric oracle catches transpose mistakes ────────────────────────────

#[test]
fn asymmetric_oracle_catches_transpose_mistakes() {
    // A is asymmetric: A = [[0, 0], [2, 0]]
    // B = [[0, 0], [4, 0]]
    // B is strictly lower triangular (B^2 = 0)
    // h = 0
    let a = Matrix::from_row_major(2, 2, vec![0.0, 0.0, 2.0, 0.0]);

    let h = PolynomialAcyclicity::value(&a).unwrap();
    assert_close(h, 0.0, 1.0e-15);

    // Transpose A: A^T = [[0, 2], [0, 0]]
    // B = [[0, 4], [0, 0]] (strictly upper triangular)
    // h = 0
    let a_t = a.transpose();
    let h_t = PolynomialAcyclicity::value(&a_t).unwrap();
    assert_close(h_t, 0.0, 1.0e-15);

    let a_cycle = Matrix::from_row_major(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
    let h_cycle = PolynomialAcyclicity::value(&a_cycle).unwrap();
    assert!(h_cycle > 0.0);

    let a_cycle_t = a_cycle.transpose();
    let h_cycle_t = PolynomialAcyclicity::value(&a_cycle_t).unwrap();
    assert_close(h_cycle, h_cycle_t, 1.0e-15);
}

// ─── Non-finite failures ─────────────────────────────────────────────────────

#[test]
fn rejects_non_finite_coefficients() {
    let a = Matrix::from_row_major(2, 2, vec![0.0, f64::NAN, 1.0, 0.0]);
    assert!(matches!(
        PolynomialAcyclicity::value(&a),
        Err(CausalError::NonFiniteWeight { .. })
    ));
    assert!(matches!(
        PolynomialAcyclicity::value_and_gradient(&a),
        Err(CausalError::NonFiniteWeight { .. })
    ));
}

// ─── Dimension failures ──────────────────────────────────────────────────────

#[test]
fn rejects_non_square() {
    let a = Matrix::zeros(2, 3);
    assert!(matches!(
        PolynomialAcyclicity::value(&a),
        Err(CausalError::NotSquare { .. })
    ));
}

#[test]
fn rejects_zero_dimension() {
    let a = Matrix::zeros(0, 0);
    assert!(matches!(
        PolynomialAcyclicity::value(&a),
        Err(CausalError::ZeroDimension)
    ));
}

// ─── Deterministic repeated evaluation ───────────────────────────────────────

#[test]
fn deterministic_evaluation() {
    let a = Matrix::from_row_major(2, 2, vec![0.5, 1.0, -0.5, 0.3]);

    let h1 = PolynomialAcyclicity::value(&a).unwrap();
    let h2 = PolynomialAcyclicity::value(&a).unwrap();
    assert_eq!(h1, h2);

    let (h3, g1) = PolynomialAcyclicity::value_and_gradient(&a).unwrap();
    let (h4, g2) = PolynomialAcyclicity::value_and_gradient(&a).unwrap();
    assert_eq!(h3, h4);
    assert_eq!(g1.data(), g2.data());
}

// ─── One-dimensional case ────────────────────────────────────────────────────

#[test]
fn one_dimension() {
    // A = [[a]], B = [[a^2]]
    // h = trace(B^1)/1! = a^2
    // grad = 2*a * (B^0)^T/0! = 2*a * 1 = 2a
    // Actually grad: 2*A ∘ [I] = 2*A, so grad[0,0] = 2*a
    let a = Matrix::from_row_major(1, 1, vec![3.0]);
    let h = PolynomialAcyclicity::value(&a).unwrap();
    assert_close(h, 9.0, 1.0e-14);

    let (h2, grad) = PolynomialAcyclicity::value_and_gradient(&a).unwrap();
    assert_close(h2, 9.0, 1.0e-14);
    assert_close(grad[(0, 0)], 6.0, 1.0e-14);
}
