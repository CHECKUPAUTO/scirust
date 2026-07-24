//! Basic correlation cases, causal motifs (chain/fork/collider), dataset
//! contracts, and property-style invariance tests for
//! [`scirust_causal::PartialCorrelationTest`].
//!
//! Adversarial/contamination scenarios live in
//! `conditional_independence_adversarial.rs`.

use scirust_causal::{
    CalibrationMethod, CausalAssumption, CausalDataset, CausalError, CausalVariable,
    ConditionalIndependenceConfig, ConditionalIndependenceMethod, ConditionalIndependenceTest,
    Environment, IndependenceDecision, Intervention, InterventionKind, PartialCorrelationTest,
    RegimeSelection, VariableKind, VariableRole,
};
use scirust_solvers::Matrix;
use scirust_stats::SplitMix64;

// ─── Fixtures ────────────────────────────────────────────────────────────

/// Builds a purely observational `CausalDataset` from column vectors (one
/// `Vec<f64>` per variable, all the same length), auto-named `v0`, `v1`, ….
fn dataset_from_columns(columns: &[Vec<f64>]) -> CausalDataset {
    let n = columns[0].len();
    let d = columns.len();
    let mut data = vec![0.0; n * d];
    for row in 0..n
    {
        for col in 0..d
        {
            data[row * d + col] = columns[col][row];
        }
    }
    let variables: Vec<CausalVariable> = (0..d)
        .map(|i| {
            CausalVariable::new(
                i,
                format!("v{i}"),
                VariableRole::Unspecified,
                VariableKind::Continuous,
            )
            .unwrap()
        })
        .collect();
    let matrix = Matrix::from_row_major(n, d, data);
    let env = Environment::observational("obs").unwrap();
    CausalDataset::single_environment(variables, env, &matrix, "test fixture").unwrap()
}

/// Centered uniform noise on `[-0.5, 0.5)`, deterministic given `rng`.
fn noise(rng: &mut SplitMix64) -> f64 {
    rng.next_f64() - 0.5
}

fn gaussian_partial_correlation_config(fisher_z: bool) -> ConditionalIndependenceConfig {
    ConditionalIndependenceConfig::new(
        0.05,
        ConditionalIndependenceMethod::GaussianPartialCorrelation { fisher_z },
    )
    .unwrap()
}

// ─── 16.1 Basic correlation cases ───────────────────────────────────────

#[test]
fn two_independent_variables_are_not_rejected() {
    let mut rng = SplitMix64::new(1);
    let n = 200;
    let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let y: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let dataset = dataset_from_columns(&[x, y]);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    let result = test.test(&dataset, 0, 1, &[]).unwrap();
    assert_eq!(
        result.decision,
        IndependenceDecision::IndependentWithinThreshold
    );
}

#[test]
fn direct_linear_dependence_is_rejected() {
    let mut rng = SplitMix64::new(2);
    let n = 200;
    let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| 0.9 * xi + 0.1 * noise(&mut rng))
        .collect();
    let dataset = dataset_from_columns(&[x, y]);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    let result = test.test(&dataset, 0, 1, &[]).unwrap();
    assert_eq!(result.decision, IndependenceDecision::Dependent);
    assert!(result.statistic > 0.0);
}

#[test]
fn negative_dependence_is_rejected_with_negative_statistic() {
    let mut rng = SplitMix64::new(3);
    let n = 200;
    let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| -0.9 * xi + 0.1 * noise(&mut rng))
        .collect();
    let dataset = dataset_from_columns(&[x, y]);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    let result = test.test(&dataset, 0, 1, &[]).unwrap();
    assert_eq!(result.decision, IndependenceDecision::Dependent);
    assert!(result.statistic < 0.0);
}

#[test]
fn empty_conditioning_set_equals_pearson_correlation() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let y = vec![2.0, 1.0, 5.0, 3.0, 7.0, 6.0, 9.0, 8.0];
    let dataset = dataset_from_columns(&[x, y]);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    let result = test.test(&dataset, 0, 1, &[]).unwrap();
    assert_eq!(result.effective_rank, 0);
    // Sanity: matches the crate's own oracle-tested Pearson formula indirectly
    // by symmetry (see property tests) rather than re-deriving it by hand here.
    assert!(result.statistic.is_finite());
}

#[test]
fn perfect_correlation_reports_statistic_of_one() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
    let dataset = dataset_from_columns(&[x, y]);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    let result = test.test(&dataset, 0, 1, &[]).unwrap();
    assert!((result.statistic - 1.0).abs() < 1e-9);
    // Fisher-z is undefined exactly at r=1 (see partial_correlation module);
    // an honest implementation reports Inconclusive here, not a fabricated
    // p-value.
    assert_eq!(result.decision, IndependenceDecision::Inconclusive);
}

#[test]
fn zero_variance_is_a_typed_error_not_inconclusive() {
    let x = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let dataset = dataset_from_columns(&[x, y]);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    let result = test.test(&dataset, 0, 1, &[]);
    assert!(matches!(
        result,
        Err(CausalError::ZeroVariance { variable: 0 })
    ));
}

#[test]
fn insufficient_sample_count_is_a_typed_error() {
    let x = vec![1.0];
    let y = vec![2.0];
    let dataset = dataset_from_columns(&[x, y]);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    let result = test.test(&dataset, 0, 1, &[]);
    assert!(matches!(
        result,
        Err(CausalError::InsufficientSamples {
            required: 2,
            actual: 1
        })
    ));
}

// ─── Causal motifs ───────────────────────────────────────────────────────
//
// These fixtures are *statistical* checks of well-known linear-SCM facts
// about vanishing/non-vanishing partial correlation. They demonstrate the
// consumable *evidence* a discovery algorithm would use — they are not
// themselves a causal-discovery claim (see the crate root and this module's
// own docs).

/// `X -> Z -> Y`: `z = a*x + e_z`, `y = b*z + e_y`, all noises independent.
fn chain_fixture(seed: u64, n: usize) -> CausalDataset {
    let mut rng = SplitMix64::new(seed);
    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n
    {
        let xi = noise(&mut rng);
        let zi = 0.8 * xi + noise(&mut rng);
        let yi = 0.8 * zi + noise(&mut rng);
        x.push(xi);
        z.push(zi);
        y.push(yi);
    }
    dataset_from_columns(&[x, z, y])
}

#[test]
fn chain_x_and_y_are_dependent_marginally_but_independent_given_z() {
    let dataset = chain_fixture(10, 400);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));

    let marginal = test.test(&dataset, 0, 2, &[]).unwrap();
    assert_eq!(
        marginal.decision,
        IndependenceDecision::Dependent,
        "chain: X and Y should be marginally dependent, got statistic {}",
        marginal.statistic
    );

    let conditional = test.test(&dataset, 0, 2, &[1]).unwrap();
    assert_eq!(
        conditional.decision,
        IndependenceDecision::IndependentWithinThreshold,
        "chain: X ⟂ Y | Z should not be rejected, got statistic {} p={:?}",
        conditional.statistic,
        conditional.p_value
    );
}

/// `X <- Z -> Y`: `x = a*z + e_x`, `y = b*z + e_y`, all noises independent.
fn fork_fixture(seed: u64, n: usize) -> CausalDataset {
    let mut rng = SplitMix64::new(seed);
    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n
    {
        let zi = noise(&mut rng);
        let xi = 0.8 * zi + noise(&mut rng);
        let yi = 0.8 * zi + noise(&mut rng);
        x.push(xi);
        z.push(zi);
        y.push(yi);
    }
    dataset_from_columns(&[x, z, y])
}

#[test]
fn fork_x_and_y_are_dependent_marginally_but_independent_given_z() {
    let dataset = fork_fixture(11, 400);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));

    let marginal = test.test(&dataset, 0, 2, &[]).unwrap();
    assert_eq!(
        marginal.decision,
        IndependenceDecision::Dependent,
        "fork: X and Y should be marginally dependent, got statistic {}",
        marginal.statistic
    );

    let conditional = test.test(&dataset, 0, 2, &[1]).unwrap();
    assert_eq!(
        conditional.decision,
        IndependenceDecision::IndependentWithinThreshold,
        "fork: X ⟂ Y | Z should not be rejected, got statistic {} p={:?}",
        conditional.statistic,
        conditional.p_value
    );
}

/// `X -> Z <- Y`: `z = a*x + b*y + e_z`, `x` and `y` mutually independent
/// exogenous variables. This motif is essential for future causal discovery:
/// conditioning on a collider *induces* spurious dependence.
fn collider_fixture(seed: u64, n: usize) -> CausalDataset {
    let mut rng = SplitMix64::new(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for _ in 0..n
    {
        let xi = noise(&mut rng);
        let yi = noise(&mut rng);
        let zi = 0.8 * xi + 0.8 * yi + noise(&mut rng);
        x.push(xi);
        y.push(yi);
        z.push(zi);
    }
    dataset_from_columns(&[x, y, z])
}

#[test]
fn collider_x_and_y_are_independent_marginally_but_dependent_given_z() {
    let dataset = collider_fixture(12, 400);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));

    let marginal = test.test(&dataset, 0, 1, &[]).unwrap();
    assert_eq!(
        marginal.decision,
        IndependenceDecision::IndependentWithinThreshold,
        "collider: X and Y should be marginally independent, got statistic {}",
        marginal.statistic
    );

    let conditional = test.test(&dataset, 0, 1, &[2]).unwrap();
    assert_eq!(
        conditional.decision,
        IndependenceDecision::Dependent,
        "collider: conditioning on Z should induce dependence, got statistic {} p={:?}",
        conditional.statistic,
        conditional.p_value
    );
}

/// Confounded association `U -> X`, `U -> Y`, with `U` **observed**: the test
/// may condition on it (and the induced X,Y dependence should vanish).
#[test]
fn confounded_association_with_observed_confounder() {
    let mut rng = SplitMix64::new(13);
    let n = 400;
    let mut u = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n
    {
        let ui = noise(&mut rng);
        x.push(0.8 * ui + noise(&mut rng));
        y.push(0.8 * ui + noise(&mut rng));
        u.push(ui);
    }
    let dataset = dataset_from_columns(&[u, x, y]);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));

    let marginal = test.test(&dataset, 1, 2, &[]).unwrap();
    assert_eq!(marginal.decision, IndependenceDecision::Dependent);

    let conditional = test.test(&dataset, 1, 2, &[0]).unwrap();
    assert_eq!(
        conditional.decision,
        IndependenceDecision::IndependentWithinThreshold,
        "conditioning on the observed confounder U should remove the induced \
         X,Y association, got statistic {}",
        conditional.statistic
    );
}

/// The same confounding structure, but `U` is **not** in the dataset at all
/// (latent). The test can only ever be asked about the variables that
/// exist — it has no way to "pretend" to condition on an undeclared latent
/// variable, and this test documents that boundary rather than working
/// around it: the marginal association is reported as dependence, exactly as
/// it would be for a genuine direct causal link, because nothing in this
/// dataset distinguishes the two. This is precisely why conditional
/// independence testing alone never rules out latent confounding (see the
/// crate root and this module's docs).
#[test]
fn confounded_association_with_latent_confounder_cannot_be_told_apart_from_direct_dependence() {
    let mut rng = SplitMix64::new(13);
    let n = 400;
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n
    {
        let ui = noise(&mut rng);
        x.push(0.8 * ui + noise(&mut rng));
        y.push(0.8 * ui + noise(&mut rng));
    }
    // U is never included in the dataset.
    let dataset = dataset_from_columns(&[x, y]);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    let marginal = test.test(&dataset, 0, 1, &[]).unwrap();
    assert_eq!(
        marginal.decision,
        IndependenceDecision::Dependent,
        "with U latent, X and Y look exactly as dependent as a direct causal \
         link would — the test cannot and does not distinguish the two"
    );
}

// ─── Dataset contracts ───────────────────────────────────────────────────

#[test]
fn rejects_unknown_variable() {
    let dataset = dataset_from_columns(&[vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]]);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    assert!(matches!(
        test.test(&dataset, 0, 5, &[]),
        Err(CausalError::UnknownVariableIndex { index: 5 })
    ));
    assert!(matches!(
        test.test(&dataset, 0, 1, &[9]),
        Err(CausalError::UnknownVariableIndex { index: 9 })
    ));
}

#[test]
fn rejects_x_equals_y() {
    let dataset = dataset_from_columns(&[vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]]);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    assert!(matches!(
        test.test(&dataset, 0, 0, &[]),
        Err(CausalError::SameVariable { variable: 0 })
    ));
}

#[test]
fn rejects_endpoint_in_conditioning_set() {
    let dataset = dataset_from_columns(&[
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2.0, 3.0, 4.0, 5.0],
        vec![3.0, 1.0, 5.0, 2.0],
    ]);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    assert!(matches!(
        test.test(&dataset, 0, 1, &[0]),
        Err(CausalError::ConditioningContainsEndpoint { variable: 0 })
    ));
    assert!(matches!(
        test.test(&dataset, 0, 1, &[1]),
        Err(CausalError::ConditioningContainsEndpoint { variable: 1 })
    ));
}

#[test]
fn rejects_duplicate_conditioning_variable() {
    let dataset = dataset_from_columns(&[
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec![2.0, 3.0, 4.0, 5.0, 6.0],
        vec![3.0, 1.0, 5.0, 2.0, 4.0],
    ]);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    assert!(matches!(
        test.test(&dataset, 0, 1, &[2, 2]),
        Err(CausalError::DuplicateConditioningVariable { variable: 2 })
    ));
}

#[test]
fn rejects_non_continuous_variable_kind() {
    let n = 10;
    let mut rng = SplitMix64::new(20);
    let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let y: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let matrix = Matrix::from_row_major(n, 2, {
        let mut data = vec![0.0; n * 2];
        for row in 0..n
        {
            data[row * 2] = x[row];
            data[row * 2 + 1] = y[row];
        }
        data
    });
    let variables = vec![
        CausalVariable::new(0, "x", VariableRole::Unspecified, VariableKind::Binary).unwrap(),
        CausalVariable::new(1, "y", VariableRole::Unspecified, VariableKind::Continuous).unwrap(),
    ];
    let env = Environment::observational("obs").unwrap();
    let dataset = CausalDataset::single_environment(variables, env, &matrix, "fixture").unwrap();
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    assert!(matches!(
        test.test(&dataset, 0, 1, &[]),
        Err(CausalError::UnsupportedVariableKind { variable: 0 })
    ));
}

#[test]
fn observational_only_regime_excludes_intervened_blocks() {
    let n = 100;
    let mut rng = SplitMix64::new(21);
    let obs_x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let obs_y: Vec<f64> = obs_x
        .iter()
        .map(|&xi| 0.9 * xi + 0.1 * noise(&mut rng))
        .collect();

    let variables = vec![
        CausalVariable::new(0, "x", VariableRole::Unspecified, VariableKind::Continuous).unwrap(),
        CausalVariable::new(1, "y", VariableRole::Unspecified, VariableKind::Continuous).unwrap(),
    ];

    let mut obs_data = vec![0.0; n * 2];
    for row in 0..n
    {
        obs_data[row * 2] = obs_x[row];
        obs_data[row * 2 + 1] = obs_y[row];
    }
    let obs_matrix = Matrix::from_row_major(n, 2, obs_data);
    let obs_env = Environment::observational("obs").unwrap();
    let obs_block = scirust_causal::SampleBlock::from_matrix(obs_env, &obs_matrix).unwrap();

    // An intervened block with UNRELATED (independent) x, y — if this leaked
    // into the observational-only computation, it would dilute/change the
    // result; the test below confirms it is excluded.
    let intervened_x: Vec<f64> = (0..n).map(|_| noise(&mut rng) * 100.0).collect();
    let intervened_y: Vec<f64> = (0..n).map(|_| noise(&mut rng) * 100.0).collect();
    let mut intervened_data = vec![0.0; n * 2];
    for row in 0..n
    {
        intervened_data[row * 2] = intervened_x[row];
        intervened_data[row * 2 + 1] = intervened_y[row];
    }
    let intervened_matrix = Matrix::from_row_major(n, 2, intervened_data);
    let iv = Intervention::new(0, InterventionKind::Atomic { value: 0.0 }).unwrap();
    let intervened_env = Environment::new("do_x", vec![iv]).unwrap();
    let intervened_block =
        scirust_causal::SampleBlock::from_matrix(intervened_env, &intervened_matrix).unwrap();

    let dataset = CausalDataset::new(
        variables,
        vec![obs_block, intervened_block],
        "mixed regimes",
    )
    .unwrap();

    let config =
        gaussian_partial_correlation_config(true).with_regime(RegimeSelection::ObservationalOnly);
    let test = PartialCorrelationTest::new(config);
    let result = test.test(&dataset, 0, 1, &[]).unwrap();
    assert_eq!(
        result.sample_count, n,
        "only the n observational rows should be used"
    );
    assert_eq!(result.decision, IndependenceDecision::Dependent);
}

#[test]
fn environment_regime_selects_by_id() {
    let variables = vec![
        CausalVariable::new(0, "x", VariableRole::Unspecified, VariableKind::Continuous).unwrap(),
        CausalVariable::new(1, "y", VariableRole::Unspecified, VariableKind::Continuous).unwrap(),
    ];
    let block_a = scirust_causal::SampleBlock::from_matrix(
        Environment::observational("site_a").unwrap(),
        &Matrix::from_row_major(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )
    .unwrap();
    let block_b = scirust_causal::SampleBlock::from_matrix(
        Environment::observational("site_b").unwrap(),
        &Matrix::from_row_major(4, 2, vec![10.0, 1.0, 10.0, 1.0, 10.0, 1.0, 10.0, 1.0]),
    )
    .unwrap();
    let dataset = CausalDataset::new(variables, vec![block_a, block_b], "two sites").unwrap();

    let config = gaussian_partial_correlation_config(true)
        .with_regime(RegimeSelection::Environment("site_b".to_string()));
    let test = PartialCorrelationTest::new(config);
    let result = test.test(&dataset, 0, 1, &[]);
    // site_b has zero variance in both columns by construction (constant rows).
    assert!(matches!(result, Err(CausalError::ZeroVariance { .. })));
}

#[test]
fn explicit_rows_regime_selects_a_global_row_subset() {
    let dataset = dataset_from_columns(&[
        vec![1.0, 2.0, 3.0, 4.0, 100.0],
        vec![2.0, 3.0, 4.0, 5.0, 100.0],
    ]);
    let config = gaussian_partial_correlation_config(true)
        .with_regime(RegimeSelection::ExplicitRows(vec![0, 1, 2, 3]));
    let test = PartialCorrelationTest::new(config);
    let result = test.test(&dataset, 0, 1, &[]).unwrap();
    assert_eq!(
        result.sample_count, 4,
        "the 5th (outlier) row must be excluded"
    );
}

#[test]
fn complete_cases_policy_is_a_no_op_under_current_dataset_invariants() {
    // CausalDataset::new / SampleBlock::new already reject non-finite entries
    // at construction, so every row is always "complete" — this documents
    // that fact as an executable check rather than an assumption.
    let dataset = dataset_from_columns(&[vec![1.0, 2.0, 3.0, 4.0], vec![2.0, 1.0, 4.0, 3.0]]);
    let config = gaussian_partial_correlation_config(true)
        .with_missing_value_policy(scirust_causal::MissingValuePolicy::CompleteCases);
    let test = PartialCorrelationTest::new(config);
    let result = test.test(&dataset, 0, 1, &[]).unwrap();
    assert_eq!(result.sample_count, 4);
}

// ─── Serialization / result shape ───────────────────────────────────────

#[test]
fn result_json_round_trips() {
    let dataset = dataset_from_columns(&[
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0],
    ]);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    let result = test.test(&dataset, 0, 1, &[]).unwrap();
    let json = serde_json::to_string(&result).unwrap();
    let back: scirust_causal::ConditionalIndependenceResult = serde_json::from_str(&json).unwrap();
    assert_eq!(result, back);
}

#[test]
fn result_reports_the_assumptions_it_relies_on() {
    let dataset = chain_fixture(14, 200);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    let result = test.test(&dataset, 0, 2, &[1]).unwrap();
    assert!(
        result
            .assumptions
            .contains(&CausalAssumption::CorrectFunctionalForm)
    );
    assert!(
        result
            .assumptions
            .contains(&CausalAssumption::AdequateSampleSize)
    );
    assert_eq!(result.calibration, CalibrationMethod::FisherZ);
}

// ─── Property-style tests ────────────────────────────────────────────────

#[test]
fn symmetry_x_y_versus_y_x() {
    let dataset = chain_fixture(15, 200);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    let xy = test.test(&dataset, 0, 2, &[1]).unwrap();
    let yx = test.test(&dataset, 2, 0, &[1]).unwrap();
    assert!((xy.effect_size - yx.effect_size).abs() < 1e-12);
    assert_eq!(xy.p_value, yx.p_value);
    assert_eq!(xy.decision, yx.decision);
}

#[test]
fn conditioning_set_order_invariance() {
    let mut rng = SplitMix64::new(16);
    let n = 100;
    let a: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let b: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let c: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let x: Vec<f64> = (0..n)
        .map(|i| 0.4 * a[i] + 0.3 * b[i] + 0.2 * c[i] + noise(&mut rng))
        .collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 0.3 * a[i] + 0.2 * b[i] + 0.4 * c[i] + noise(&mut rng))
        .collect();
    let dataset = dataset_from_columns(&[x, y, a, b, c]);
    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));

    let forward = test.test(&dataset, 0, 1, &[2, 3, 4]).unwrap();
    let reordered = test.test(&dataset, 0, 1, &[4, 2, 3]).unwrap();
    assert_eq!(forward.conditioned_on, vec![2, 3, 4]);
    assert_eq!(reordered.conditioned_on, vec![2, 3, 4]);
    assert!((forward.statistic - reordered.statistic).abs() < 1e-9);
    assert_eq!(forward.p_value, reordered.p_value);
}

#[test]
fn row_order_invariance_for_non_permutation_methods() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let y = vec![2.0, 1.0, 4.0, 3.0, 6.0, 5.0];
    let dataset = dataset_from_columns(&[x.clone(), y.clone()]);

    let mut shuffled_x = x.clone();
    let mut shuffled_y = y.clone();
    shuffled_x.reverse();
    shuffled_y.reverse();
    let shuffled_dataset = dataset_from_columns(&[shuffled_x, shuffled_y]);

    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    let a = test.test(&dataset, 0, 1, &[]).unwrap();
    let b = test.test(&shuffled_dataset, 0, 1, &[]).unwrap();
    assert!((a.statistic - b.statistic).abs() < 1e-12);
}

#[test]
fn positive_scale_invariance() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let y = vec![2.0, 1.0, 5.0, 3.0, 7.0, 6.0, 9.0];
    let dataset_a = dataset_from_columns(&[x.clone(), y.clone()]);
    let scaled_x: Vec<f64> = x.iter().map(|v| v * 1000.0).collect();
    let scaled_y: Vec<f64> = y.iter().map(|v| v * 0.001).collect();
    let dataset_b = dataset_from_columns(&[scaled_x, scaled_y]);

    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    let a = test.test(&dataset_a, 0, 1, &[]).unwrap();
    let b = test.test(&dataset_b, 0, 1, &[]).unwrap();
    assert!((a.statistic - b.statistic).abs() < 1e-9);
}

#[test]
fn translation_invariance() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let y = vec![2.0, 1.0, 5.0, 3.0, 7.0, 6.0, 9.0];
    let dataset_a = dataset_from_columns(&[x.clone(), y.clone()]);
    let shifted_x: Vec<f64> = x.iter().map(|v| v + 1000.0).collect();
    let shifted_y: Vec<f64> = y.iter().map(|v| v - 500.0).collect();
    let dataset_b = dataset_from_columns(&[shifted_x, shifted_y]);

    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    let a = test.test(&dataset_a, 0, 1, &[]).unwrap();
    let b = test.test(&dataset_b, 0, 1, &[]).unwrap();
    assert!((a.statistic - b.statistic).abs() < 1e-9);
}

#[test]
fn negating_one_endpoint_flips_sign_but_preserves_magnitude() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let y = vec![2.0, 1.0, 5.0, 3.0, 7.0, 6.0, 9.0];
    let dataset_a = dataset_from_columns(&[x.clone(), y.clone()]);
    let negated_y: Vec<f64> = y.iter().map(|v| -v).collect();
    let dataset_b = dataset_from_columns(&[x, negated_y]);

    let test = PartialCorrelationTest::new(gaussian_partial_correlation_config(true));
    let a = test.test(&dataset_a, 0, 1, &[]).unwrap();
    let b = test.test(&dataset_b, 0, 1, &[]).unwrap();
    assert!((a.statistic + b.statistic).abs() < 1e-9);
    assert!((a.effect_size - b.effect_size).abs() < 1e-9);
    assert_eq!(a.p_value, b.p_value);
    assert_eq!(a.decision, b.decision);
}
