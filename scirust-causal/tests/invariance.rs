//! Invariant Causal Prediction: recovery of true direct causes from
//! multi-environment data, the three-way outcome discipline, and the headline
//! contrast — a misspecification ICP *detects* and backdoor adjustment
//! structurally cannot.

use scirust_causal::{
    AdjustmentStrategy, CausalDataset, CausalError, CausalVariable, EffectEstimationConfig,
    Environment, IdentifiabilityStatus, Intervention, InterventionKind, InvarianceConfig,
    InvariantPredictionOutcome, SampleBlock, VariableKind, VariableRole, estimate_effect_from_dag,
    invariant_causal_prediction,
};
use scirust_graph::dag::CausalDag;
use scirust_stats::SplitMix64;

// ─── Fixtures ────────────────────────────────────────────────────────────

fn variables(d: usize) -> Vec<CausalVariable> {
    (0..d)
        .map(|i| {
            CausalVariable::new(
                i,
                format!("v{i}"),
                VariableRole::Unspecified,
                VariableKind::Continuous,
            )
            .unwrap()
        })
        .collect()
}

/// Builds a row-major block from per-variable columns.
fn block(environment: Environment, columns: &[Vec<f64>]) -> SampleBlock {
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
    SampleBlock::new(environment, n, d, data).unwrap()
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

/// Four variables: `v0` is the **only** direct cause of the target `v3`.
///
/// - `v0` — the true cause; the second environment **rescales** it.
/// - `v1` — independent noise, deliberately left untouched, so it cannot act
///   as an environment indicator and absorb a difference on another subset's
///   behalf.
/// - `v2` — a *child* of the target (`v3 -> v2`), strongly correlated with it
///   but not a cause.
/// - `v3` — the target: `v3 = 1.5·v0 + ε`, an invariant mechanism.
///
/// The intervention rescales rather than shifts `v0`, and that choice is
/// load-bearing. Under a pure **mean shift**, the target and its child move by
/// the same amount, so a pooled regression of the target on the child picks a
/// slope near 1 that absorbs the shift exactly — leaving no detectable mean or
/// variance difference, and the child is (correctly, given that evidence)
/// accepted as invariant. A **scale** change breaks that collinearity: the
/// within-environment target-on-child slope depends on the target's variance,
/// which now differs between environments, so no single pooled slope fits
/// both. This is a real property of invariance testing, not an artefact —
/// which environments you have determines what they can distinguish.
///
/// The child's own noise (`2.0·ε`) is likewise load-bearing. Make it small and
/// the child becomes a near-perfect proxy for the target: the pooled slope
/// goes to 1, the residual collapses to the child's *own* invariant noise, and
/// the target's differing variance never reaches the residual — so the child
/// survives. Both of these were found by running the test, not by reasoning
/// about it.
fn one_true_cause_dataset(seed: u64, n: usize) -> CausalDataset {
    let mut rng = SplitMix64::new(seed);
    let mut build = |scale: f64, count: usize| {
        let (mut v0, mut v1, mut v2, mut v3) = (
            Vec::with_capacity(count),
            Vec::with_capacity(count),
            Vec::with_capacity(count),
            Vec::with_capacity(count),
        );
        for _ in 0..count
        {
            let a = scale * noise(&mut rng);
            let b = noise(&mut rng);
            let y = 1.5 * a + 0.3 * noise(&mut rng);
            let c = y + 2.0 * noise(&mut rng);
            v0.push(a);
            v1.push(b);
            v2.push(c);
            v3.push(y);
        }
        vec![v0, v1, v2, v3]
    };

    let observational = block(
        Environment::observational("baseline").unwrap(),
        &build(1.0, n),
    );
    let rescaled = block(
        Environment::new(
            "v0_rescaled",
            vec![
                Intervention::new(
                    0,
                    InterventionKind::MechanismChange {
                        description: "v0 variance increased".to_string(),
                    },
                )
                .unwrap(),
            ],
        )
        .unwrap(),
        &build(4.0, n),
    );
    CausalDataset::new(variables(4), vec![observational, rescaled], "test fixture").unwrap()
}

// ─── Recovering the true cause ───────────────────────────────────────────

#[test]
fn icp_recovers_the_single_true_cause() {
    let dataset = one_true_cause_dataset(80, 600);
    let result = invariant_causal_prediction(
        &dataset,
        3,
        &[0, 1, 2],
        &InvarianceConfig::new(0.05).unwrap(),
    )
    .unwrap();

    assert_eq!(
        result.outcome,
        InvariantPredictionOutcome::CausalPredictorsIdentified,
        "accepted sets: {:?}",
        result.accepted_sets
    );
    assert_eq!(
        result.causal_predictors,
        vec![0],
        "v0 is the only direct cause; accepted sets were {:?}",
        result.accepted_sets
    );
    assert_eq!(
        result.certificate.status(),
        IdentifiabilityStatus::Identifiable
    );
    assert_eq!(result.environments, vec!["baseline", "v0_rescaled"]);
    // Every accepted set must contain the true cause, by construction of the
    // intersection -- checked directly rather than inferred.
    assert!(
        result
            .accepted_sets
            .iter()
            .all(|s| s.predictors.contains(&0)),
        "accepted sets: {:?}",
        result.accepted_sets
    );
    assert!(
        result.sets_tested == 8,
        "2^3 subsets, got {}",
        result.sets_tested
    );
}

#[test]
fn the_certificate_states_the_subset_caveat_and_names_invariance() {
    let dataset = one_true_cause_dataset(80, 600);
    let result = invariant_causal_prediction(
        &dataset,
        3,
        &[0, 1, 2],
        &InvarianceConfig::new(0.05).unwrap(),
    )
    .unwrap();

    let note = result.certificate.sensitivity_note().unwrap();
    assert!(
        note.contains("SUBSET"),
        "the certificate must state that this is a subset, not a complete parent set: {note}"
    );
    assert!(
        result
            .certificate
            .assumptions_used()
            .iter()
            .any(|a| matches!(
                a,
                scirust_causal::CausalAssumption::InvarianceAcrossEnvironments
            )),
        "invariance must be named among the assumptions"
    );
    // No estimate is attached: ICP identifies *which* variables are causes,
    // never *how large* the effect is.
    assert_eq!(result.certificate.estimate(), None);
}

// ─── The headline: ICP detects what backdoor adjustment cannot ──────────

/// A hidden confounder `U` drives both `X` and `Y`, and the second environment
/// intervenes on `X` (severing it from `U`). `U` is never recorded.
///
/// Columns are `[X, Y]`. True effect of `X` on `Y` is `0.7`.
fn hidden_confounder_with_intervention(seed: u64, n: usize) -> CausalDataset {
    let mut rng = SplitMix64::new(seed);

    // Baseline: X is driven by the hidden U.
    let (mut x_obs, mut y_obs) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n
    {
        let u = noise(&mut rng);
        let x = 0.9 * u + 0.3 * noise(&mut rng);
        let y = 0.7 * x + 0.8 * u + 0.3 * noise(&mut rng);
        x_obs.push(x);
        y_obs.push(y);
    }

    // Intervened: X is set independently of U, so the X->Y relationship the
    // data shows is the *causal* 0.7 rather than the confounded ~1.5.
    let (mut x_int, mut y_int) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n
    {
        let u = noise(&mut rng);
        let x = 2.0 * noise(&mut rng);
        let y = 0.7 * x + 0.8 * u + 0.3 * noise(&mut rng);
        x_int.push(x);
        y_int.push(y);
    }

    let observational = block(
        Environment::observational("baseline").unwrap(),
        &[x_obs, y_obs],
    );
    let intervened = block(
        Environment::new(
            "x_intervened",
            vec![Intervention::new(0, InterventionKind::Unspecified).unwrap()],
        )
        .unwrap(),
        &[x_int, y_int],
    );
    CausalDataset::new(
        variables(2),
        vec![observational, intervened],
        "test fixture",
    )
    .unwrap()
}

#[test]
fn icp_flags_a_misspecification_that_backdoor_adjustment_certifies() {
    let dataset = hidden_confounder_with_intervention(81, 1500);
    let true_effect = 0.7;

    // Backdoor adjustment, given the plausible-but-wrong graph "X -> Y", uses
    // the observational rows and confidently certifies a badly wrong number.
    // It has no way to notice, because the confounder is not in its graph.
    let backdoor = estimate_effect_from_dag(
        &dataset,
        &dag_from_edges(2, &[(0, 1)]),
        0,
        1,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();
    assert_eq!(
        backdoor.certificate.status(),
        IdentifiabilityStatus::Identifiable
    );
    let biased = backdoor.estimate.unwrap();
    assert!(
        biased > 1.2,
        "precondition: backdoor should report a badly inflated estimate, got {biased}"
    );
    assert!(
        (biased - true_effect).abs() > 10.0 * backdoor.standard_error.unwrap(),
        "precondition: the bias should dwarf the reported uncertainty"
    );

    // ICP, on the *same* variables with no graph at all, reports that no
    // subset is invariant -- positive evidence the model is misspecified.
    let icp = invariant_causal_prediction(&dataset, 1, &[0], &InvarianceConfig::new(0.05).unwrap())
        .unwrap();

    assert_eq!(
        icp.outcome,
        InvariantPredictionOutcome::AssumptionsViolated,
        "ICP should detect the misspecification; accepted sets were {:?}",
        icp.accepted_sets
    );
    assert!(icp.accepted_sets.is_empty());
    assert_eq!(
        icp.certificate.status(),
        IdentifiabilityStatus::Inconclusive,
        "a detected misspecification concludes nothing about the effect"
    );
    assert!(
        icp.warnings
            .iter()
            .any(|w| w.contains("positive evidence of model misspecification")),
        "the finding must be stated as evidence, not as an absence of findings: {:?}",
        icp.warnings
    );
}

// ─── An empty intersection is a first-class outcome ─────────────────────

#[test]
fn interchangeable_proxies_yield_an_empty_intersection() {
    // v0 and v1 are near-identical proxies for the same latent signal, and the
    // target is that signal plus noise. Both {v0} and {v1} explain the target
    // invariantly, so no single variable is required by every surviving
    // explanation -- an honest "cannot tell", not "nothing is causal".
    let mut rng = SplitMix64::new(82);
    let n = 600;
    let mut build = |shift: f64, count: usize| {
        let (mut v0, mut v1, mut v2) = (
            Vec::with_capacity(count),
            Vec::with_capacity(count),
            Vec::with_capacity(count),
        );
        for _ in 0..count
        {
            let signal = noise(&mut rng) + shift;
            v0.push(signal + 0.01 * noise(&mut rng));
            v1.push(signal + 0.01 * noise(&mut rng));
            v2.push(signal + 0.3 * noise(&mut rng));
        }
        vec![v0, v1, v2]
    };

    let dataset = CausalDataset::new(
        variables(3),
        vec![
            block(
                Environment::observational("baseline").unwrap(),
                &build(0.0, n),
            ),
            block(
                Environment::new(
                    "shifted",
                    vec![Intervention::new(0, InterventionKind::Shift { delta: 3.0 }).unwrap()],
                )
                .unwrap(),
                &build(3.0, n),
            ),
        ],
        "test fixture",
    )
    .unwrap();

    let result =
        invariant_causal_prediction(&dataset, 2, &[0, 1], &InvarianceConfig::new(0.05).unwrap())
            .unwrap();

    assert_eq!(
        result.outcome,
        InvariantPredictionOutcome::NoPredictorConfirmed,
        "accepted sets: {:?}",
        result.accepted_sets
    );
    assert!(result.causal_predictors.is_empty());
    assert!(
        !result.accepted_sets.is_empty(),
        "subsets did survive; it is their intersection that is empty"
    );
    assert_eq!(
        result.certificate.status(),
        IdentifiabilityStatus::Inconclusive
    );
}

// ─── Bounded search honesty ──────────────────────────────────────────────

#[test]
fn a_bounded_search_that_finds_an_intersection_says_it_was_bounded() {
    let dataset = one_true_cause_dataset(80, 600);
    let result = invariant_causal_prediction(
        &dataset,
        3,
        &[0, 1, 2],
        &InvarianceConfig::new(0.05)
            .unwrap()
            .with_max_predictor_set_size(1),
    )
    .unwrap();

    assert!(
        result.sets_tested < 8,
        "a bounded search must examine fewer subsets, tested {}",
        result.sets_tested
    );
    if result.outcome == InvariantPredictionOutcome::CausalPredictorsIdentified
    {
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("not a complete search")),
            "a bounded search that reports an intersection must say it was bounded: {:?}",
            result.warnings
        );
    }
}

// ─── Contracts ───────────────────────────────────────────────────────────

#[test]
fn a_single_environment_is_a_typed_error() {
    // ICP's entire mechanism is comparison across environments.
    let mut rng = SplitMix64::new(83);
    let n = 200;
    let v0: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let v1: Vec<f64> = v0
        .iter()
        .map(|&a| 1.5 * a + 0.3 * noise(&mut rng))
        .collect();
    let dataset = CausalDataset::new(
        variables(2),
        vec![block(
            Environment::observational("only").unwrap(),
            &[v0, v1],
        )],
        "test fixture",
    )
    .unwrap();

    assert!(matches!(
        invariant_causal_prediction(&dataset, 1, &[0], &InvarianceConfig::new(0.05).unwrap()),
        Err(CausalError::InvalidContract { .. })
    ));
}

#[test]
fn malformed_queries_are_typed_errors() {
    let dataset = one_true_cause_dataset(80, 200);
    let config = InvarianceConfig::new(0.05).unwrap();

    assert!(matches!(
        invariant_causal_prediction(&dataset, 3, &[0, 3], &config),
        Err(CausalError::SameVariable { variable: 3 })
    ));
    assert!(matches!(
        invariant_causal_prediction(&dataset, 3, &[0, 9], &config),
        Err(CausalError::UnknownVariableIndex { index: 9 })
    ));
    assert!(matches!(
        invariant_causal_prediction(&dataset, 3, &[0, 0], &config),
        Err(CausalError::DuplicateConditioningVariable { variable: 0 })
    ));
    assert!(matches!(
        invariant_causal_prediction(&dataset, 9, &[0], &config),
        Err(CausalError::UnknownVariableIndex { index: 9 })
    ));
}

#[test]
fn an_invalid_significance_level_is_rejected() {
    for bad in [0.0, 1.0, -0.1, f64::NAN]
    {
        assert!(
            matches!(
                InvarianceConfig::new(bad),
                Err(CausalError::InvalidConfiguration {
                    name: "significance_level",
                    ..
                })
            ),
            "significance level {bad} should be rejected"
        );
    }
}

#[test]
fn results_are_deterministic_and_json_round_trip() {
    let dataset = one_true_cause_dataset(80, 400);
    let config = InvarianceConfig::new(0.05).unwrap();
    let first = invariant_causal_prediction(&dataset, 3, &[0, 1, 2], &config).unwrap();
    let second = invariant_causal_prediction(&dataset, 3, &[0, 1, 2], &config).unwrap();
    assert_eq!(first, second);

    let json = serde_json::to_string(&first).unwrap();
    let round_tripped: scirust_causal::InvariantPredictionResult =
        serde_json::from_str(&json).unwrap();
    assert_eq!(first, round_tripped);
}

#[test]
fn a_stricter_level_accepts_at_least_as_many_subsets() {
    // Rejecting at a smaller alpha is harder, so the accepted family can only
    // grow -- a monotonicity the method must respect.
    let dataset = one_true_cause_dataset(80, 600);
    let lenient = invariant_causal_prediction(
        &dataset,
        3,
        &[0, 1, 2],
        &InvarianceConfig::new(0.10).unwrap(),
    )
    .unwrap();
    let strict = invariant_causal_prediction(
        &dataset,
        3,
        &[0, 1, 2],
        &InvarianceConfig::new(0.01).unwrap(),
    )
    .unwrap();
    assert!(
        strict.accepted_sets.len() >= lenient.accepted_sets.len(),
        "a smaller alpha must accept at least as many subsets: {} vs {}",
        strict.accepted_sets.len(),
        lenient.accepted_sets.len()
    );
}
