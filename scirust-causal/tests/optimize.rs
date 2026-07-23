use scirust_causal::{CausalError, OptimizerConfig, optimize_causal};
use scirust_solvers::Matrix;

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    let error = (actual - expected).abs();
    assert!(
        error <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, \
         error={error:.17e}, tolerance={tolerance:.17e}"
    );
}

// ─── Config validation ──────────────────────────────────────────────────────

#[test]
fn rejects_zero_inner_iter() {
    assert!(matches!(
        OptimizerConfig::new(0, 3, 1.0e-6, 1.0e-8),
        Err(CausalError::InvalidConfiguration { .. })
    ));
}

#[test]
fn rejects_zero_outer_iter() {
    assert!(matches!(
        OptimizerConfig::new(50, 0, 1.0e-6, 1.0e-8),
        Err(CausalError::InvalidConfiguration { .. })
    ));
}

#[test]
fn rejects_non_positive_tolerances() {
    assert!(OptimizerConfig::new(50, 3, 0.0, 1.0e-8).is_err());
    assert!(OptimizerConfig::new(50, 3, 1.0e-6, 0.0).is_err());
    assert!(OptimizerConfig::new(50, 3, -1.0, 1.0e-8).is_err());
    assert!(OptimizerConfig::new(50, 3, f64::NAN, 1.0e-8).is_err());
}

// ─── Small deterministic problem ────────────────────────────────────────────

#[test]
fn optimize_small_problem() {
    // Tiny known DAG: 2 variables, A[1,0] = 0.5
    // Generate data through the forward flow
    let true_a = Matrix::from_row_major(2, 2, vec![0.0, 0.0, 0.5, 0.0]);
    let flow = scirust_causal::TriangularCubicFlow::new(true_a.clone()).unwrap();

    // Generate synthetic samples
    let noise = vec![
        vec![0.1, 0.2],
        vec![-0.3, 0.5],
        vec![0.7, -0.1],
        vec![-0.4, -0.6],
        vec![0.2, 0.8],
    ];

    let mut samples_data = Vec::new();
    for n in &noise
    {
        let x = flow.inverse(n).unwrap();
        samples_data.extend(x);
    }

    let samples = Matrix::from_row_major(5, 2, samples_data);
    let initial = Matrix::zeros(2, 2);

    let config = OptimizerConfig::new(30, 5, 1.0e-5, 1.0e-8);

    let result = optimize_causal(
        &samples,
        &initial,
        0.001,
        1.0e-8,
        0.0,
        1.0,
        &config.unwrap(),
    )
    .unwrap();

    assert!(result.objective.is_finite());
    assert!(result.gradient_norm.is_finite());
    assert!(result.acyclicity.is_finite());
    assert!(result.outer_iterations > 0 || result.inner_iterations > 0);

    // Check that the optimization didn't cause non-finite gradients
    for i in 0..2
    {
        for j in 0..2
        {
            assert!(result.interactions[(i, j)].is_finite());
        }
    }

    // Diagonal should remain near zero
    assert_close(result.interactions[(0, 0)], 0.0, 1.0e-4);
    assert_close(result.interactions[(1, 1)], 0.0, 1.0e-4);
}

// ─── Zero interactions, zero sparsity, clean data ───────────────────────────

#[test]
fn zero_interaction_optimization() {
    // Data where A = 0 is optimal: X = noise, no coupling.
    let samples = Matrix::from_row_major(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let initial = Matrix::zeros(2, 2);

    let config = OptimizerConfig::new(10, 2, 1.0e-4, 1.0e-8);
    let result =
        optimize_causal(&samples, &initial, 0.0, 1.0e-8, 0.0, 1.0, &config.unwrap()).unwrap();

    assert!(result.objective.is_finite());
    assert!(result.gradient_norm.is_finite());
    assert!(matches!(
        result.termination,
        scirust_causal::TerminationReason::MaxOuterIterations
            | scirust_causal::TerminationReason::Converged
    ));
}

// ─── Dimension mismatch propagates ──────────────────────────────────────────

#[test]
fn rejects_dimension_mismatch() {
    let samples = Matrix::zeros(3, 2);
    let initial = Matrix::zeros(3, 3);
    let config = OptimizerConfig::new(10, 2, 1.0e-6, 1.0e-8).unwrap();

    let result = optimize_causal(&samples, &initial, 0.0, 1.0e-8, 0.0, 1.0, &config);
    assert!(result.is_err());
}
