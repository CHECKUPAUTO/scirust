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

/// Deterministic 2-variable data from the true DAG `A[1,0] = 0.5`.
fn two_variable_samples() -> Matrix {
    let true_a = Matrix::from_row_major(2, 2, vec![0.0, 0.0, 0.5, 0.0]);
    let flow = scirust_causal::TriangularCubicFlow::new(true_a).unwrap();
    let noise = [
        [0.1, 0.2],
        [-0.3, 0.5],
        [0.7, -0.1],
        [-0.4, -0.6],
        [0.2, 0.8],
    ];
    let mut samples_data = Vec::new();
    for n in &noise
    {
        samples_data.extend(flow.inverse(n).unwrap());
    }
    Matrix::from_row_major(5, 2, samples_data)
}

#[test]
fn zero_init_is_reported_as_stationary_not_converged() {
    // The all-zeros interaction matrix is a saddle of the cubic score (its
    // gradient vanishes there). The optimizer must NOT report this as a found
    // minimum: it returns StationaryAtInitialPoint with the empty graph unchanged.
    let samples = two_variable_samples();
    let initial = Matrix::zeros(2, 2);
    let config = OptimizerConfig::new(30, 5, 1.0e-5, 1.0e-8).unwrap();

    let result = optimize_causal(&samples, &initial, 0.001, 1.0e-8, 0.0, 1.0, &config).unwrap();

    assert_eq!(
        result.termination,
        scirust_causal::TerminationReason::StationaryAtInitialPoint
    );
    assert_eq!(result.inner_iterations, 0);
    // The result is the initial guess, untouched.
    assert_eq!(result.interactions.data(), initial.data());
    assert!(
        result.warnings.iter().any(|w| w.contains("stationary")),
        "a stationarity warning must be surfaced"
    );
}

#[test]
fn recovers_structure_from_a_nonzero_init() {
    // Initialized OFF the zero saddle, the optimizer must actually descend and
    // recover the true coupling A[1,0] ≈ 0.5 (not stay at the 0.1 start).
    //
    // `lambda_l1 = 0.0` here: with only 5 samples the raw data-term gradient
    // near the true value has magnitude ~1e-4, so even a small L1 sparsity
    // penalty (e.g. 0.001) dominates it and pulls the fit back to exactly
    // zero — a real, honest optimum of *that* penalized objective, just not
    // the one this test is checking. Disabling L1 isolates the thing being
    // tested: does the score + acyclicity machinery, started off the saddle,
    // actually move toward the generating structure. 30 outer iterations are
    // enough for this small problem to certify `Converged` outright (verified
    // empirically), not merely exhaust its iteration budget.
    let samples = two_variable_samples();
    let initial = Matrix::from_row_major(2, 2, vec![0.0, 0.0, 0.1, 0.0]);
    let config = OptimizerConfig::new(60, 30, 1.0e-6, 1.0e-10).unwrap();

    let result = optimize_causal(&samples, &initial, 0.0, 1.0e-8, 0.0, 1.0, &config).unwrap();

    assert_eq!(
        result.termination,
        scirust_causal::TerminationReason::Converged,
        "expected a certified minimum, got {:?}",
        result.termination
    );
    assert!(
        result.inner_iterations > 0,
        "the optimizer must take descent steps"
    );
    for i in 0..2
    {
        for j in 0..2
        {
            assert!(result.interactions[(i, j)].is_finite());
        }
    }
    // Structure recovered: the coupling lands close to the true 0.5, well past
    // its 0.1 start (empirically ≈ 0.530).
    assert_close(result.interactions[(1, 0)], 0.5, 0.1);
    // The reverse and diagonal entries stay near zero (empirically ≤ 3.1e-4).
    assert_close(result.interactions[(0, 1)], 0.0, 0.05);
    assert_close(result.interactions[(0, 0)], 0.0, 0.05);
    assert_close(result.interactions[(1, 1)], 0.0, 0.05);
}

// ─── Zero interactions, zero sparsity, clean data ───────────────────────────

#[test]
fn zero_init_on_uncoupled_data_is_stationary() {
    // A = 0 is optimal here; a zero start is already stationary, so the honest
    // report is StationaryAtInitialPoint (no descent step taken), not Converged.
    let samples = Matrix::from_row_major(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let initial = Matrix::zeros(2, 2);

    let config = OptimizerConfig::new(10, 2, 1.0e-4, 1.0e-8);
    let result =
        optimize_causal(&samples, &initial, 0.0, 1.0e-8, 0.0, 1.0, &config.unwrap()).unwrap();

    assert!(result.objective.is_finite());
    assert!(result.gradient_norm.is_finite());
    assert_eq!(
        result.termination,
        scirust_causal::TerminationReason::StationaryAtInitialPoint
    );
    assert_eq!(result.inner_iterations, 0);
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
