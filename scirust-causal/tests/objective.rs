use scirust_causal::{AugmentedLagrangianConfig, CausalError, CausalObjective};
use scirust_solvers::Matrix;

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, \
         error={error:.17e}, tolerance={tolerance:.17e}"
    );
}

fn default_config() -> AugmentedLagrangianConfig {
    AugmentedLagrangianConfig::new(0.01, 0.0, 1.0, 1.0e-8).unwrap()
}

// ─── Config validation ──────────────────────────────────────────────────────

#[test]
fn rejects_negative_lambda_l1() {
    assert!(matches!(
        AugmentedLagrangianConfig::new(-1.0, 0.0, 1.0, 1.0e-8),
        Err(CausalError::InvalidConfiguration { .. })
    ));
}

#[test]
fn rejects_negative_alpha() {
    assert!(matches!(
        AugmentedLagrangianConfig::new(0.0, -1.0, 1.0, 1.0e-8),
        Err(CausalError::InvalidConfiguration { .. })
    ));
}

#[test]
fn rejects_non_positive_rho() {
    assert!(matches!(
        AugmentedLagrangianConfig::new(0.0, 0.0, 0.0, 1.0e-8),
        Err(CausalError::InvalidConfiguration { .. })
    ));
    assert!(matches!(
        AugmentedLagrangianConfig::new(0.0, 0.0, -1.0, 1.0e-8),
        Err(CausalError::InvalidConfiguration { .. })
    ));
}

#[test]
fn rejects_non_positive_smooth_epsilon() {
    assert!(matches!(
        AugmentedLagrangianConfig::new(0.0, 0.0, 1.0, 0.0),
        Err(CausalError::InvalidConfiguration { .. })
    ));
    assert!(matches!(
        AugmentedLagrangianConfig::new(0.0, 0.0, 1.0, -1.0),
        Err(CausalError::InvalidConfiguration { .. })
    ));
}

#[test]
fn rejects_non_finite_config() {
    assert!(AugmentedLagrangianConfig::new(f64::NAN, 0.0, 1.0, 1.0e-8).is_err());
    assert!(AugmentedLagrangianConfig::new(0.0, f64::INFINITY, 1.0, 1.0e-8).is_err());
    assert!(AugmentedLagrangianConfig::new(0.0, 0.0, f64::NEG_INFINITY, 1.0e-8).is_err());
    assert!(AugmentedLagrangianConfig::new(0.0, 0.0, 1.0, f64::NAN).is_err());
}

// ─── Total objective as exact sum of components ─────────────────────────────

#[test]
fn total_is_sum_of_components() {
    let config = AugmentedLagrangianConfig::new(0.1, 0.5, 2.0, 1.0e-8).unwrap();
    let samples = Matrix::from_row_major(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let interactions = Matrix::from_row_major(2, 2, vec![0.0, 0.0, 0.5, 0.0]);

    let eval = CausalObjective::evaluate(&samples, &interactions, &config).unwrap();

    let expected_total = eval.data_loss
        + eval.sparsity_penalty
        + config.alpha * eval.acyclicity
        + eval.augmented_penalty;

    assert_close(eval.total, expected_total, 1.0e-14);
}

// ─── Gradient matches finite differences on full objective ──────────────────

#[test]
fn gradient_matches_finite_difference() {
    let config = AugmentedLagrangianConfig::new(0.05, 0.1, 0.5, 1.0e-8).unwrap();
    let samples = Matrix::from_row_major(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let interactions = Matrix::from_row_major(2, 2, vec![0.0, 0.0, 0.3, 0.0]);
    let epsilon = 1.0e-6;

    let eval = CausalObjective::evaluate(&samples, &interactions, &config).unwrap();
    let dim = interactions.rows();

    fn obj_value(
        samples: &Matrix,
        interactions: &Matrix,
        config: &AugmentedLagrangianConfig,
    ) -> f64 {
        CausalObjective::evaluate(samples, interactions, config)
            .unwrap()
            .total
    }

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

            let numerical = (obj_value(&samples, &plus, &config)
                - obj_value(&samples, &minus, &config))
                / (2.0 * epsilon);

            assert_close(eval.gradient[(row, col)], numerical, 2.0e-7);
        }
    }
}

// ─── Zero penalties → gradient equals data gradient ─────────────────────────

#[test]
fn zero_penalties_matches_data_gradient() {
    let config = AugmentedLagrangianConfig::new(0.0, 0.0, 1.0, 1.0e-8).unwrap();
    let samples = Matrix::from_row_major(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let interactions = Matrix::from_row_major(2, 2, vec![0.0, 0.0, 0.5, 0.0]);

    let eval = CausalObjective::evaluate(&samples, &interactions, &config).unwrap();

    // data gradient should match the eval gradient when penalties are zero
    let (_data_loss, grad_data) =
        scirust_causal::CubicCausalScore::loss_and_gradient(&samples, &interactions).unwrap();

    for i in 0..2
    {
        for j in 0..2
        {
            assert_close(eval.gradient[(i, j)], grad_data[(i, j)], 1.0e-14);
        }
    }
}

// ─── Shape failures propagate ────────────────────────────────────────────────

#[test]
fn rejects_zero_samples() {
    let config = default_config();
    let samples = Matrix::zeros(0, 2);
    let interactions = Matrix::zeros(2, 2);
    assert!(CausalObjective::evaluate(&samples, &interactions, &config).is_err());
}

#[test]
fn rejects_dimension_mismatch() {
    let config = default_config();
    let samples = Matrix::zeros(3, 2);
    let interactions = Matrix::zeros(3, 3);
    assert!(CausalObjective::evaluate(&samples, &interactions, &config).is_err());
}

// ─── Deterministic evaluation ────────────────────────────────────────────────

#[test]
fn deterministic_evaluation() {
    let config = default_config();
    let samples = Matrix::from_row_major(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let interactions = Matrix::from_row_major(2, 2, vec![0.0, 0.0, 0.5, 0.0]);

    let e1 = CausalObjective::evaluate(&samples, &interactions, &config).unwrap();
    let e2 = CausalObjective::evaluate(&samples, &interactions, &config).unwrap();

    assert_eq!(e1.total, e2.total);
    assert_eq!(e1.gradient.data(), e2.gradient.data());
}
