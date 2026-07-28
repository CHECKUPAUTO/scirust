//! Correctness of backdoor-adjusted effect estimation against known
//! generating coefficients, plus the certificate contract it produces.
//!
//! Adversarial scenarios (latent confounding, collider bias, over-control,
//! CPDAG ambiguity) live in `effect_estimation_adversarial.rs`.

use scirust_causal::{
    AdjustmentStrategy, CausalAssumption, CausalDataset, CausalError, CausalVariable,
    EffectEstimationConfig, Environment, IdentifiabilityStatus, VariableKind, VariableRole,
    estimate_effect_from_dag,
};
use scirust_graph::dag::CausalDag;
use scirust_solvers::Matrix;
use scirust_stats::SplitMix64;

// ─── Fixtures ────────────────────────────────────────────────────────────

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

fn noise(rng: &mut SplitMix64) -> f64 {
    rng.next_f64() - 0.5
}

fn dag_from_edges(n: usize, edges: &[(usize, usize)]) -> CausalDag {
    let mut dag = CausalDag::new(n);
    for &(u, v) in edges
    {
        dag.add_directed_edge(u, v).unwrap();
    }
    dag
}

/// Confounded structure with **known** coefficients:
///   `U(0) = e`, `X(1) = a·U + e`, `Y(2) = b·X + c·U + e`.
/// The true causal effect of `X` on `Y` is exactly `b`; the *unadjusted*
/// regression of `Y` on `X` is biased by the `U` path.
fn confounded_dataset(
    seed: u64,
    n: usize,
    a: f64,
    b: f64,
    c: f64,
    noise_scale: f64,
) -> CausalDataset {
    let mut rng = SplitMix64::new(seed);
    let (mut u, mut x, mut y) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n
    {
        let ui = noise(&mut rng);
        let xi = a * ui + noise_scale * noise(&mut rng);
        let yi = b * xi + c * ui + noise_scale * noise(&mut rng);
        u.push(ui);
        x.push(xi);
        y.push(yi);
    }
    dataset_from_columns(&[u, x, y])
}

// ─── Recovering a known coefficient ──────────────────────────────────────

#[test]
fn adjusting_for_the_confounder_recovers_the_true_coefficient() {
    // True effect b = 0.7, confounding via U with a = 0.9, c = 0.8.
    let dataset = confounded_dataset(40, 4000, 0.9, 0.7, 0.8, 0.3);
    let dag = dag_from_edges(3, &[(0, 1), (0, 2), (1, 2)]);

    let result = estimate_effect_from_dag(
        &dataset,
        &dag,
        1,
        2,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();

    assert_eq!(
        result.certificate.status(),
        IdentifiabilityStatus::Identifiable
    );
    assert_eq!(result.adjustment_set, Some(vec![0]));
    let estimate = result.estimate.unwrap();
    assert!(
        (estimate - 0.7).abs() < 0.02,
        "adjusted estimate {estimate} should recover the true effect 0.7"
    );
}

#[test]
fn omitting_the_confounder_is_visibly_biased_away_from_the_truth() {
    // The same data, deliberately adjusting for nothing. The backdoor
    // criterion rejects this, so no estimate is produced at all -- the point
    // being that the honest answer is abstention, not a biased number.
    let dataset = confounded_dataset(40, 4000, 0.9, 0.7, 0.8, 0.3);
    let dag = dag_from_edges(3, &[(0, 1), (0, 2), (1, 2)]);

    let result = estimate_effect_from_dag(
        &dataset,
        &dag,
        1,
        2,
        &AdjustmentStrategy::Explicit(vec![]),
        &EffectEstimationConfig::new(),
    )
    .unwrap();

    assert_eq!(
        result.certificate.status(),
        IdentifiabilityStatus::NotIdentifiable
    );
    assert_eq!(result.estimate, None);
    assert_eq!(result.certificate.estimate(), None);
    assert!(!result.warnings.is_empty());
}

#[test]
fn an_unconfounded_effect_needs_no_adjustment_and_is_recovered() {
    // X(0) -> Y(1) with coefficient 0.5 and nothing else.
    let mut rng = SplitMix64::new(41);
    let n = 3000;
    let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| 0.5 * xi + 0.3 * noise(&mut rng))
        .collect();
    let dataset = dataset_from_columns(&[x, y]);
    let dag = dag_from_edges(2, &[(0, 1)]);

    let result = estimate_effect_from_dag(
        &dataset,
        &dag,
        0,
        1,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();

    assert_eq!(result.adjustment_set, Some(Vec::new()));
    let estimate = result.estimate.unwrap();
    assert!(
        (estimate - 0.5).abs() < 0.02,
        "estimate {estimate} should recover 0.5"
    );
}

#[test]
fn a_zero_effect_is_estimated_as_approximately_zero() {
    // X(1) and Y(2) share the confounder U(0) but X does NOT cause Y.
    // Adjusting for U must drive the estimate to ~0 -- the "no effect" case
    // a method that merely reports association would get wrong.
    let dataset = confounded_dataset(42, 4000, 0.9, 0.0, 0.8, 0.3);
    let dag = dag_from_edges(3, &[(0, 1), (0, 2), (1, 2)]);

    let result = estimate_effect_from_dag(
        &dataset,
        &dag,
        1,
        2,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();

    let estimate = result.estimate.unwrap();
    assert!(
        estimate.abs() < 0.02,
        "estimate {estimate} should be ~0 when the true effect is 0"
    );
}

#[test]
fn a_negative_effect_recovers_its_sign_and_magnitude() {
    let dataset = confounded_dataset(43, 4000, 0.9, -0.6, 0.8, 0.3);
    let dag = dag_from_edges(3, &[(0, 1), (0, 2), (1, 2)]);

    let result = estimate_effect_from_dag(
        &dataset,
        &dag,
        1,
        2,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();

    let estimate = result.estimate.unwrap();
    assert!(
        (estimate + 0.6).abs() < 0.02,
        "estimate {estimate} should recover -0.6"
    );
}

#[test]
fn an_explicit_valid_adjustment_set_matches_the_canonical_one() {
    let dataset = confounded_dataset(40, 4000, 0.9, 0.7, 0.8, 0.3);
    let dag = dag_from_edges(3, &[(0, 1), (0, 2), (1, 2)]);
    let config = EffectEstimationConfig::new();

    let canonical = estimate_effect_from_dag(
        &dataset,
        &dag,
        1,
        2,
        &AdjustmentStrategy::CanonicalParents,
        &config,
    )
    .unwrap();
    let explicit = estimate_effect_from_dag(
        &dataset,
        &dag,
        1,
        2,
        &AdjustmentStrategy::Explicit(vec![0]),
        &config,
    )
    .unwrap();

    assert_eq!(canonical.estimate, explicit.estimate);
    assert_eq!(canonical.standard_error, explicit.standard_error);
}

// ─── Standard error behaviour ────────────────────────────────────────────

#[test]
fn the_standard_error_shrinks_as_the_sample_grows() {
    let dag = dag_from_edges(3, &[(0, 1), (0, 2), (1, 2)]);
    let config = EffectEstimationConfig::new();

    let small = estimate_effect_from_dag(
        &confounded_dataset(44, 200, 0.9, 0.7, 0.8, 0.3),
        &dag,
        1,
        2,
        &AdjustmentStrategy::CanonicalParents,
        &config,
    )
    .unwrap();
    let large = estimate_effect_from_dag(
        &confounded_dataset(44, 8000, 0.9, 0.7, 0.8, 0.3),
        &dag,
        1,
        2,
        &AdjustmentStrategy::CanonicalParents,
        &config,
    )
    .unwrap();

    assert!(
        large.standard_error.unwrap() < small.standard_error.unwrap(),
        "a larger sample must not report more uncertainty"
    );
}

#[test]
fn the_true_effect_lies_within_a_few_standard_errors() {
    let dataset = confounded_dataset(45, 4000, 0.9, 0.7, 0.8, 0.3);
    let dag = dag_from_edges(3, &[(0, 1), (0, 2), (1, 2)]);
    let result = estimate_effect_from_dag(
        &dataset,
        &dag,
        1,
        2,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();

    let estimate = result.estimate.unwrap();
    let se = result.standard_error.unwrap();
    assert!(se > 0.0, "a real sample must report a positive uncertainty");
    assert!(
        (estimate - 0.7).abs() < 4.0 * se,
        "true effect 0.7 should be within 4 standard errors of {estimate} (se={se})"
    );
}

// ─── Certificate contract ────────────────────────────────────────────────

#[test]
fn the_certificate_mirrors_the_structured_estimate_exactly() {
    let dataset = confounded_dataset(46, 2000, 0.9, 0.7, 0.8, 0.3);
    let dag = dag_from_edges(3, &[(0, 1), (0, 2), (1, 2)]);
    let result = estimate_effect_from_dag(
        &dataset,
        &dag,
        1,
        2,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();

    // The certificate is authoritative; the struct fields are a view of the
    // same values and must never diverge from it.
    assert_eq!(result.certificate.estimate(), result.estimate);
    assert_eq!(result.certificate.uncertainty(), result.standard_error);
    assert!(result.certificate.method().is_some());
    assert!(result.certificate.sensitivity_note().is_some());
    assert!(!result.certificate.fingerprint().is_empty());
}

#[test]
fn the_certificate_names_the_assumptions_the_estimate_rests_on() {
    let dataset = confounded_dataset(47, 2000, 0.9, 0.7, 0.8, 0.3);
    let dag = dag_from_edges(3, &[(0, 1), (0, 2), (1, 2)]);
    let result = estimate_effect_from_dag(
        &dataset,
        &dag,
        1,
        2,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();

    let used = result.certificate.assumptions_used();
    for required in [
        CausalAssumption::Acyclicity,
        CausalAssumption::CausalSufficiency,
        CausalAssumption::CorrectFunctionalForm,
        CausalAssumption::Positivity,
    ]
    {
        assert!(
            used.contains(&required),
            "certificate must name {required:?} among its assumptions"
        );
    }
}

#[test]
fn identical_inputs_produce_an_identical_certificate_fingerprint() {
    let dataset = confounded_dataset(48, 2000, 0.9, 0.7, 0.8, 0.3);
    let dag = dag_from_edges(3, &[(0, 1), (0, 2), (1, 2)]);
    let config = EffectEstimationConfig::new();

    let first = estimate_effect_from_dag(
        &dataset,
        &dag,
        1,
        2,
        &AdjustmentStrategy::CanonicalParents,
        &config,
    )
    .unwrap();
    let second = estimate_effect_from_dag(
        &dataset,
        &dag,
        1,
        2,
        &AdjustmentStrategy::CanonicalParents,
        &config,
    )
    .unwrap();

    assert_eq!(first, second);
    assert_eq!(
        first.certificate.fingerprint(),
        second.certificate.fingerprint()
    );
}

#[test]
fn result_json_round_trips() {
    let dataset = confounded_dataset(49, 1000, 0.9, 0.7, 0.8, 0.3);
    let dag = dag_from_edges(3, &[(0, 1), (0, 2), (1, 2)]);
    let result = estimate_effect_from_dag(
        &dataset,
        &dag,
        1,
        2,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();

    let json = serde_json::to_string(&result).unwrap();
    let round_tripped: scirust_causal::EffectEstimate = serde_json::from_str(&json).unwrap();
    assert_eq!(result, round_tripped);
}

// ─── Malformed queries ───────────────────────────────────────────────────

#[test]
fn malformed_queries_are_typed_errors() {
    let dataset = confounded_dataset(50, 500, 0.9, 0.7, 0.8, 0.3);
    let dag = dag_from_edges(3, &[(0, 1), (0, 2), (1, 2)]);
    let config = EffectEstimationConfig::new();

    assert!(matches!(
        estimate_effect_from_dag(
            &dataset,
            &dag,
            1,
            1,
            &AdjustmentStrategy::CanonicalParents,
            &config
        ),
        Err(CausalError::SameVariable { variable: 1 })
    ));
    assert!(matches!(
        estimate_effect_from_dag(
            &dataset,
            &dag,
            1,
            9,
            &AdjustmentStrategy::CanonicalParents,
            &config
        ),
        Err(CausalError::UnknownVariableIndex { index: 9 })
    ));
}

#[test]
fn a_graph_and_dataset_that_disagree_on_size_is_a_typed_error() {
    let dataset = confounded_dataset(51, 500, 0.9, 0.7, 0.8, 0.3);
    let dag = dag_from_edges(5, &[(0, 1)]);
    assert!(matches!(
        estimate_effect_from_dag(
            &dataset,
            &dag,
            1,
            2,
            &AdjustmentStrategy::CanonicalParents,
            &EffectEstimationConfig::new()
        ),
        Err(CausalError::DimensionMismatch {
            expected: 3,
            got: 5
        })
    ));
}
