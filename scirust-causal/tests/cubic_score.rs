use scirust_causal::{CausalError, CubicCausalScore};
use scirust_solvers::Matrix;

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, \
         error={error:.17e}, tolerance={tolerance:.17e}"
    );
}

fn objective_via_loss(samples: &Matrix, interactions: &Matrix) -> f64 {
    CubicCausalScore::loss(samples, interactions).unwrap()
}

// ─── One-variable oracle ────────────────────────────────────────────────────

#[test]
fn one_variable_oracle() {
    // d = 1, m = 3. A = [[0]] (zero interaction).
    // Z = X @ A^T = 0, R = X, loss = ||X||^2 / (2m)
    // Gradient: 3 * (R ∘ Z^2)^T @ X / m = 0
    let samples = Matrix::from_row_major(3, 1, vec![1.0, -2.0, 0.5]);
    let interactions = Matrix::from_row_major(1, 1, vec![0.0]);

    let loss = CubicCausalScore::loss(&samples, &interactions).unwrap();
    let expected_loss = (1.0 + 4.0 + 0.25) / (2.0 * 3.0);
    assert_close(loss, expected_loss, 1.0e-14);

    let (loss_g, grad) = CubicCausalScore::loss_and_gradient(&samples, &interactions).unwrap();
    assert_close(loss_g, expected_loss, 1.0e-14);
    assert_eq!(grad[(0, 0)], 0.0);
}

// ─── Two-variable hand-derived oracle ────────────────────────────────────────

#[test]
fn two_variable_hand_oracle() {
    // A = [[0, 0], [0.5, 0]]
    // X = [[1, 2], [3, 4], [5, 6]]
    // Z = X @ A^T = X @ [[0, 0.5], [0, 0]] = [[1*0+2*0, 1*0.5+2*0], ...]
    //   = [[0, 0.5], [0, 1.5], [0, 2.5]]  = [[0, 0.5], [0, 1.5], [0, 2.5]]... wait
    // Z[i,j] = sum_k X[i,k] * A[j,k]
    // A^T = [[0, 0.5], [0, 0]]
    // Z[0,0] = X[0,0]*0 + X[0,1]*0 = 0
    // Z[0,1] = X[0,0]*0.5 + X[0,1]*0 = 0.5
    // Z[1,0] = X[1,0]*0 + X[1,1]*0 = 0
    // Z[1,1] = X[1,0]*0.5 + X[1,1]*0 = 1.5
    // Z[2,0] = X[2,0]*0 + X[2,1]*0 = 0
    // Z[2,1] = X[2,0]*0.5 + X[2,1]*0 = 2.5
    //
    // Z = [[0, 0.5], [0, 1.5], [0, 2.5]]
    // Z^{∘3} = [[0, 0.125], [0, 3.375], [0, 15.625]]
    // R = X + Z^{∘3} = [[1, 2.125], [3, 7.375], [5, 21.625]]
    //
    // ||R||^2 = sum of all entries squared
    // = 1 + 2.125^2 + 9 + 7.375^2 + 25 + 21.625^2
    // = 1 + 4.515625 + 9 + 54.390625 + 25 + 467.640625
    // = 561.546875
    //
    // loss = 561.546875 / (2*3) = 93.5911458333...
    //
    // Gradient: 3 * (R ∘ Z^{∘2})^T @ X / m
    // Z^{∘2} = [[0, 0.25], [0, 2.25], [0, 6.25]]
    // R ∘ Z^{∘2} = [[0, 2.125*0.25=0.53125], [0, 7.375*2.25=16.59375], [0, 21.625*6.25=135.15625]]
    //           = [[0, 0.53125], [0, 16.59375], [0, 135.15625]]
    // (R ∘ Z^{∘2})^T = [[0, 0, 0], [0.53125, 16.59375, 135.15625]]
    // X = [[1, 2], [3, 4], [5, 6]]
    // (R ∘ Z^{∘2})^T @ X =
    //   [0,0]: 0*1 + 0*3 + 0*5 = 0
    //   [0,1]: 0*2 + 0*4 + 0*6 = 0
    //   [1,0]: 0.53125*1 + 16.59375*3 + 135.15625*5 = 0.53125 + 49.78125 + 675.78125 = 726.09375
    //   [1,1]: 0.53125*2 + 16.59375*4 + 135.15625*6 = 1.0625 + 66.375 + 810.9375 = 878.375
    // grad = 3/m * [[0, 0], [726.09375, 878.375]]
    //      = [[0, 0], [726.09375, 878.375]]
    // Wait: 3/3 = 1, so grad = [[0, 0], [726.09375, 878.375]]

    let samples = Matrix::from_row_major(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let interactions = Matrix::from_row_major(2, 2, vec![0.0, 0.0, 0.5, 0.0]);

    let loss = CubicCausalScore::loss(&samples, &interactions).unwrap();
    let expected_loss = 561.546875 / 6.0;
    assert_close(loss, expected_loss, 1.0e-12);

    let (loss_g, grad) = CubicCausalScore::loss_and_gradient(&samples, &interactions).unwrap();
    assert_close(loss_g, expected_loss, 1.0e-12);

    assert_close(grad[(0, 0)], 0.0, 0.0);
    assert_close(grad[(0, 1)], 0.0, 0.0);
    assert_close(grad[(1, 0)], 726.09375, 1.0e-10);
    assert_close(grad[(1, 1)], 878.375, 1.0e-10);
}

// ─── Zero-interaction identity case ──────────────────────────────────────────

#[test]
fn zero_interaction_identity() {
    // A = 0 => Z = 0 => R = X => loss = ||X||^2 / (2m)
    // Gradient = 0
    let samples = Matrix::from_row_major(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let interactions = Matrix::zeros(2, 2);

    let loss = CubicCausalScore::loss(&samples, &interactions).unwrap();
    let expected = (1.0 + 4.0 + 9.0 + 16.0) / (2.0 * 2.0);
    assert_close(loss, expected, 1.0e-14);

    let (loss_g, grad) = CubicCausalScore::loss_and_gradient(&samples, &interactions).unwrap();
    assert_close(loss_g, expected, 1.0e-14);
    for i in 0..2
    {
        for j in 0..2
        {
            assert_eq!(grad[(i, j)], 0.0);
        }
    }
}

// ─── Analytical gradient against central finite differences ─────────────────

#[test]
fn gradient_matches_central_difference() {
    let samples = Matrix::from_row_major(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let interactions = Matrix::from_row_major(2, 2, vec![0.0, 0.0, 0.5, 0.0]);
    let epsilon = 1.0e-6;

    let (_, analytical) = CubicCausalScore::loss_and_gradient(&samples, &interactions).unwrap();
    let dim = interactions.rows();

    for row in 0..dim
    {
        for col in 0..dim
        {
            let mut plus_data = interactions.data().to_vec();
            let mut minus_data = interactions.data().to_vec();
            let idx = row * dim + col;
            plus_data[idx] += epsilon;
            minus_data[idx] -= epsilon;

            let plus = Matrix::from_row_major(dim, dim, plus_data);
            let minus = Matrix::from_row_major(dim, dim, minus_data);

            let loss_plus = objective_via_loss(&samples, &plus);
            let loss_minus = objective_via_loss(&samples, &minus);

            let numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
            assert_close(analytical[(row, col)], numerical, 2.0e-7);
        }
    }
}

// ─── Shape failures ─────────────────────────────────────────────────────────

#[test]
fn rejects_zero_samples() {
    let samples = Matrix::zeros(0, 2);
    let interactions = Matrix::zeros(2, 2);
    assert!(matches!(
        CubicCausalScore::loss(&samples, &interactions),
        Err(CausalError::ZeroSamples)
    ));
    assert!(matches!(
        CubicCausalScore::loss_and_gradient(&samples, &interactions),
        Err(CausalError::ZeroSamples)
    ));
}

#[test]
fn rejects_zero_variables() {
    let samples = Matrix::zeros(3, 0);
    let interactions = Matrix::zeros(0, 0);
    assert!(matches!(
        CubicCausalScore::loss(&samples, &interactions),
        Err(CausalError::ZeroDimension)
    ));
    assert!(matches!(
        CubicCausalScore::loss_and_gradient(&samples, &interactions),
        Err(CausalError::ZeroDimension)
    ));
}

#[test]
fn rejects_non_square_interactions() {
    let samples = Matrix::zeros(3, 2);
    let interactions = Matrix::zeros(2, 3);
    assert!(matches!(
        CubicCausalScore::loss(&samples, &interactions),
        Err(CausalError::NotSquare { .. })
    ));
}

#[test]
fn rejects_incompatible_dimensions() {
    let samples = Matrix::zeros(3, 2);
    let interactions = Matrix::zeros(3, 3);
    assert!(matches!(
        CubicCausalScore::loss(&samples, &interactions),
        Err(CausalError::DimensionMismatch { .. })
    ));
}

// ─── Non-finite failures ────────────────────────────────────────────────────

#[test]
fn rejects_non_finite_samples() {
    let samples = Matrix::from_row_major(2, 2, vec![1.0, f64::NAN, 3.0, 4.0]);
    let interactions = Matrix::zeros(2, 2);
    assert!(matches!(
        CubicCausalScore::loss(&samples, &interactions),
        Err(CausalError::NonFiniteInput { .. })
    ));
    assert!(matches!(
        CubicCausalScore::loss_and_gradient(&samples, &interactions),
        Err(CausalError::NonFiniteInput { .. })
    ));
}

#[test]
fn rejects_non_finite_weights() {
    let samples = Matrix::zeros(3, 2);
    let interactions = Matrix::from_row_major(2, 2, vec![0.0, f64::INFINITY, 0.5, 0.0]);
    assert!(matches!(
        CubicCausalScore::loss(&samples, &interactions),
        Err(CausalError::NonFiniteWeight { .. })
    ));
    assert!(matches!(
        CubicCausalScore::loss_and_gradient(&samples, &interactions),
        Err(CausalError::NonFiniteWeight { .. })
    ));
}

// ─── Deterministic evaluation ────────────────────────────────────────────────

#[test]
fn deterministic_evaluation() {
    let samples = Matrix::from_row_major(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let interactions = Matrix::from_row_major(2, 2, vec![0.0, 0.0, 0.5, 0.0]);

    let loss1 = CubicCausalScore::loss(&samples, &interactions).unwrap();
    let loss2 = CubicCausalScore::loss(&samples, &interactions).unwrap();
    assert_eq!(loss1, loss2);

    let (l1, g1) = CubicCausalScore::loss_and_gradient(&samples, &interactions).unwrap();
    let (l2, g2) = CubicCausalScore::loss_and_gradient(&samples, &interactions).unwrap();
    assert_eq!(l1, l2);
    assert_eq!(g1.data(), g2.data());
}

// ─── No mutation of inputs ───────────────────────────────────────────────────

#[test]
fn does_not_mutate_inputs() {
    let samples_data = vec![1.0, 2.0, 3.0, 4.0];
    let inter_data = vec![0.0, 0.0, 0.5, 0.0];
    let samples = Matrix::from_row_major(2, 2, samples_data.clone());
    let interactions = Matrix::from_row_major(2, 2, inter_data.clone());

    let _ = CubicCausalScore::loss_and_gradient(&samples, &interactions).unwrap();

    assert_eq!(samples.data(), &samples_data);
    assert_eq!(interactions.data(), &inter_data);
}
