//! Adversarial scenarios for backdoor-adjusted effect estimation: the
//! failure modes the method genuinely has, demonstrated rather than asserted.
//!
//! Correctness on well-specified graphs lives in `effect_estimation.rs`.

use scirust_causal::{
    AdjustmentStrategy, CausalDataset, CausalError, CausalVariable, ConditionalIndependenceConfig,
    ConditionalIndependenceMethod, EffectEstimationConfig, Environment, EquivalenceClassConfig,
    EquivalenceClassDiscovery, IdentifiabilityStatus, PartialCorrelationTest, PcStable,
    VariableKind, VariableRole, estimate_effect_from_cpdag, estimate_effect_from_dag,
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

fn fisher_z_test() -> PartialCorrelationTest {
    PartialCorrelationTest::new(
        ConditionalIndependenceConfig::new(
            0.05,
            ConditionalIndependenceMethod::GaussianPartialCorrelation { fisher_z: true },
        )
        .unwrap(),
    )
}

// ─── The headline negative result: latent confounding ────────────────────

#[test]
fn a_latent_confounder_yields_a_confident_and_badly_wrong_estimate() {
    // Truth: U -> X, U -> Y, X -> Y with the true effect of X on Y being 0.7.
    // But U is NEVER MEASURED: the dataset contains only X and Y, and the
    // graph handed to the estimator is the (wrong, but entirely plausible)
    // "X -> Y, nothing else".
    //
    // For THAT graph the backdoor criterion is satisfied by the empty set --
    // X has no parents, so there is no backdoor path to block. The estimator
    // therefore certifies the effect Identifiable and reports a number. The
    // number is badly wrong, and nothing in the data or the criterion can
    // reveal that. This is the causal-sufficiency assumption failing, and it
    // is exactly why the certificate carries a sensitivity note rather than
    // presenting its estimate as a verified fact.
    let mut rng = SplitMix64::new(60);
    let n = 4000;
    let (mut x, mut y) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n
    {
        let u = noise(&mut rng);
        let xi = 0.9 * u + 0.3 * noise(&mut rng);
        let yi = 0.7 * xi + 0.8 * u + 0.3 * noise(&mut rng);
        x.push(xi);
        y.push(yi);
    }
    let dataset = dataset_from_columns(&[x, y]); // U is absent by construction
    let misspecified = dag_from_edges(2, &[(0, 1)]);

    let result = estimate_effect_from_dag(
        &dataset,
        &misspecified,
        0,
        1,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();

    // The framework certifies it -- correctly, *for the graph it was given*.
    assert_eq!(
        result.certificate.status(),
        IdentifiabilityStatus::Identifiable
    );
    let estimate = result.estimate.unwrap();
    let standard_error = result.standard_error.unwrap();

    // And the answer is wrong by far more than its own stated uncertainty:
    // the omitted U path inflates it to roughly 1.5 against a truth of 0.7.
    assert!(
        estimate > 1.2,
        "expected a badly inflated estimate from the latent confounder, got {estimate}"
    );
    assert!(
        (estimate - 0.7).abs() > 10.0 * standard_error,
        "the bias ({}) should dwarf the reported uncertainty ({standard_error}) -- a tight \
         confidence interval around a wrong number is precisely the failure mode",
        (estimate - 0.7).abs()
    );

    // The one honest signal available: the certificate says so in words.
    assert!(
        result
            .certificate
            .sensitivity_note()
            .unwrap()
            .contains("latent confounding"),
        "the certificate must name latent confounding as an unchecked precondition"
    );
}

// ─── Collider bias and over-control ──────────────────────────────────────

#[test]
fn adjusting_for_a_collider_is_rejected_rather_than_silently_biasing() {
    // M-structure: 0 -> 2 <- 1, 0 -> 3(treatment), 1 -> 4(outcome).
    // The empty set is valid here; conditioning on the collider 2 opens a
    // path and must be refused.
    let mut rng = SplitMix64::new(61);
    let n = 2000;
    let (mut a, mut b, mut m, mut t, mut o) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n
    {
        let ai = noise(&mut rng);
        let bi = noise(&mut rng);
        m.push(0.8 * ai + 0.8 * bi + 0.3 * noise(&mut rng));
        t.push(0.8 * ai + 0.3 * noise(&mut rng));
        o.push(0.8 * bi + 0.3 * noise(&mut rng));
        a.push(ai);
        b.push(bi);
    }
    let dataset = dataset_from_columns(&[a, b, m, t, o]);
    let dag = dag_from_edges(5, &[(0, 2), (1, 2), (0, 3), (1, 4)]);
    let config = EffectEstimationConfig::new();

    let valid = estimate_effect_from_dag(
        &dataset,
        &dag,
        3,
        4,
        &AdjustmentStrategy::Explicit(vec![]),
        &config,
    )
    .unwrap();
    assert_eq!(
        valid.certificate.status(),
        IdentifiabilityStatus::Identifiable
    );

    let collider_adjusted = estimate_effect_from_dag(
        &dataset,
        &dag,
        3,
        4,
        &AdjustmentStrategy::Explicit(vec![2]),
        &config,
    )
    .unwrap();
    assert_eq!(
        collider_adjusted.certificate.status(),
        IdentifiabilityStatus::NotIdentifiable
    );
    assert_eq!(collider_adjusted.estimate, None);
}

#[test]
fn adjusting_for_a_mediator_is_rejected_as_a_descendant_of_the_treatment() {
    // 0(treatment) -> 1(mediator) -> 2(outcome). Adjusting for the mediator
    // would estimate the direct effect, not the total effect asked for, and
    // condition 1 of the backdoor criterion refuses it outright.
    let mut rng = SplitMix64::new(62);
    let n = 2000;
    let (mut x, mut m, mut y) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n
    {
        let xi = noise(&mut rng);
        let mi = 0.8 * xi + 0.3 * noise(&mut rng);
        let yi = 0.8 * mi + 0.3 * noise(&mut rng);
        x.push(xi);
        m.push(mi);
        y.push(yi);
    }
    let dataset = dataset_from_columns(&[x, m, y]);
    let dag = dag_from_edges(3, &[(0, 1), (1, 2)]);
    let config = EffectEstimationConfig::new();

    let over_controlled = estimate_effect_from_dag(
        &dataset,
        &dag,
        0,
        2,
        &AdjustmentStrategy::Explicit(vec![1]),
        &config,
    )
    .unwrap();
    assert_eq!(
        over_controlled.certificate.status(),
        IdentifiabilityStatus::NotIdentifiable
    );
    assert_eq!(over_controlled.estimate, None);

    // The correct query (no adjustment) recovers the total effect 0.8*0.8=0.64.
    let total = estimate_effect_from_dag(
        &dataset,
        &dag,
        0,
        2,
        &AdjustmentStrategy::CanonicalParents,
        &config,
    )
    .unwrap();
    let estimate = total.estimate.unwrap();
    assert!(
        (estimate - 0.64).abs() < 0.03,
        "total effect estimate {estimate} should recover 0.64"
    );
}

// ─── CPDAG ambiguity gate (composition with Phase 5C.3) ─────────────────

#[test]
fn a_cpdag_with_an_unoriented_edge_at_the_treatment_refuses_to_estimate() {
    // A chain 0 -> 1 -> 2 is discovered as a fully *undirected* CPDAG (it is
    // Markov-equivalent to two forks and another chain), so the treatment's
    // parent set differs across the class and no single effect is determined.
    let mut rng = SplitMix64::new(63);
    let n = 1500;
    let (mut v0, mut v1, mut v2) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n
    {
        let a = noise(&mut rng);
        let b = 0.8 * a + noise(&mut rng);
        let c = 0.8 * b + noise(&mut rng);
        v0.push(a);
        v1.push(b);
        v2.push(c);
    }
    let dataset = dataset_from_columns(&[v0, v1, v2]);

    let cpdag = PcStable::new(EquivalenceClassConfig::new())
        .discover(&dataset, &fisher_z_test())
        .unwrap()
        .cpdag;
    assert!(
        cpdag.is_undirected(0, 1),
        "precondition: the chain's edges are unoriented in its CPDAG"
    );

    let result =
        estimate_effect_from_cpdag(&dataset, &cpdag, 0, 2, &EffectEstimationConfig::new()).unwrap();

    assert_eq!(
        result.certificate.status(),
        IdentifiabilityStatus::EquivalenceClassOnly
    );
    assert_eq!(result.estimate, None);
    assert_eq!(result.certificate.estimate(), None);
    assert!(
        !result.certificate.unresolved_alternatives().is_empty(),
        "the ambiguous orientation must be recorded as an unresolved alternative"
    );
}

#[test]
fn a_cpdag_fully_oriented_at_the_treatment_does_estimate() {
    // A collider 0 -> 2 <- 1 is fully oriented by PC-Stable, so the
    // treatment's parent set is the same in every member of the class and the
    // effect is identified without choosing a representative DAG.
    let mut rng = SplitMix64::new(64);
    let n = 1500;
    let (mut v0, mut v1, mut v2) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n
    {
        let a = noise(&mut rng);
        let b = noise(&mut rng);
        let c = 0.8 * a + 0.8 * b + 0.3 * noise(&mut rng);
        v0.push(a);
        v1.push(b);
        v2.push(c);
    }
    let dataset = dataset_from_columns(&[v0, v1, v2]);

    let cpdag = PcStable::new(EquivalenceClassConfig::new())
        .discover(&dataset, &fisher_z_test())
        .unwrap()
        .cpdag;
    assert!(cpdag.is_directed(0, 2) && cpdag.is_directed(1, 2));

    let result =
        estimate_effect_from_cpdag(&dataset, &cpdag, 0, 2, &EffectEstimationConfig::new()).unwrap();

    assert_eq!(
        result.certificate.status(),
        IdentifiabilityStatus::Identifiable
    );
    let estimate = result.estimate.unwrap();
    assert!(
        (estimate - 0.8).abs() < 0.03,
        "estimate {estimate} should recover the true coefficient 0.8"
    );
}

// ─── Degenerate numerics: honest non-answers, not crashes ───────────────

#[test]
fn exhausted_degrees_of_freedom_report_inconclusive_with_no_estimate() {
    // 3 rows, adjusting for 1 variable: the model Y ~ 1 + X + Z has 3
    // parameters and 3 observations, leaving zero residual df. The point
    // estimate is arithmetically available but its uncertainty is not, so the
    // honest answer is Inconclusive rather than an unquantified number.
    let dataset = dataset_from_columns(&[
        vec![0.0, 1.0, 2.0],
        vec![0.5, 2.5, 1.5],
        vec![1.0, 3.0, 0.5],
    ]);
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
        IdentifiabilityStatus::Inconclusive
    );
    assert_eq!(result.estimate, None);
    assert_eq!(result.certificate.estimate(), None);
    assert!(
        result
            .warnings
            .iter()
            .any(|w| w.contains("degrees of freedom")),
        "the reason must be stated, not silently swallowed: {:?}",
        result.warnings
    );
}

#[test]
fn a_treatment_fully_determined_by_the_adjustment_set_is_a_typed_error() {
    // X is an exact copy of Z, so after adjusting for Z no variation in X
    // remains to identify any effect.
    let mut rng = SplitMix64::new(65);
    let n = 500;
    let z: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let x = z.clone();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| 0.5 * xi + 0.3 * noise(&mut rng))
        .collect();
    let dataset = dataset_from_columns(&[z, x, y]);
    let dag = dag_from_edges(3, &[(0, 1), (0, 2), (1, 2)]);

    assert!(matches!(
        estimate_effect_from_dag(
            &dataset,
            &dag,
            1,
            2,
            &AdjustmentStrategy::CanonicalParents,
            &EffectEstimationConfig::new()
        ),
        Err(CausalError::ZeroVariance { variable: 1 })
    ));
}

#[test]
fn a_collinear_adjustment_set_is_a_typed_error_not_a_pseudo_inverse() {
    // Two adjustment variables that are exact duplicates of one another.
    let mut rng = SplitMix64::new(66);
    let n = 500;
    let z: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let z_copy = z.clone();
    let x: Vec<f64> = z
        .iter()
        .map(|&zi| 0.9 * zi + 0.3 * noise(&mut rng))
        .collect();
    let y: Vec<f64> = x
        .iter()
        .zip(&z)
        .map(|(&xi, &zi)| 0.5 * xi + 0.8 * zi + 0.3 * noise(&mut rng))
        .collect();
    let dataset = dataset_from_columns(&[z, z_copy, x, y]);
    // 0 and 1 are duplicate confounders; both point into treatment 2 and
    // outcome 3, so both are legitimately in the parent set.
    let dag = dag_from_edges(4, &[(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);

    assert!(matches!(
        estimate_effect_from_dag(
            &dataset,
            &dag,
            2,
            3,
            &AdjustmentStrategy::CanonicalParents,
            &EffectEstimationConfig::new()
        ),
        Err(CausalError::RankDeficientConditioningSet { .. })
    ));
}

// ─── The structural certificate rule, end to end ────────────────────────

#[test]
fn no_non_identifiable_outcome_ever_carries_an_estimate() {
    // Sweep every non-Identifiable path this module can produce and assert
    // the invariant the certificate layer exists to guarantee.
    let mut rng = SplitMix64::new(67);
    let n = 1200;
    let (mut v0, mut v1, mut v2) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n
    {
        let a = noise(&mut rng);
        let b = 0.8 * a + noise(&mut rng);
        let c = 0.8 * b + noise(&mut rng);
        v0.push(a);
        v1.push(b);
        v2.push(c);
    }
    let chain_dataset = dataset_from_columns(&[v0, v1, v2]);
    let confounded = dag_from_edges(3, &[(0, 1), (0, 2), (1, 2)]);
    let config = EffectEstimationConfig::new();

    let mut outcomes = Vec::new();
    // NotIdentifiable: unblocked backdoor path.
    outcomes.push(
        estimate_effect_from_dag(
            &chain_dataset,
            &confounded,
            1,
            2,
            &AdjustmentStrategy::Explicit(vec![]),
            &config,
        )
        .unwrap(),
    );
    // EquivalenceClassOnly: unoriented edge at the treatment.
    let cpdag = PcStable::new(EquivalenceClassConfig::new())
        .discover(&chain_dataset, &fisher_z_test())
        .unwrap()
        .cpdag;
    outcomes.push(estimate_effect_from_cpdag(&chain_dataset, &cpdag, 0, 2, &config).unwrap());
    // Inconclusive: no residual degrees of freedom.
    let tiny = dataset_from_columns(&[
        vec![0.0, 1.0, 2.0],
        vec![0.5, 2.5, 1.5],
        vec![1.0, 3.0, 0.5],
    ]);
    outcomes.push(
        estimate_effect_from_dag(
            &tiny,
            &confounded,
            1,
            2,
            &AdjustmentStrategy::CanonicalParents,
            &config,
        )
        .unwrap(),
    );

    for outcome in outcomes
    {
        assert_ne!(
            outcome.certificate.status(),
            IdentifiabilityStatus::Identifiable
        );
        assert_eq!(
            outcome.estimate, None,
            "a non-Identifiable outcome must never carry an estimate"
        );
        assert_eq!(outcome.certificate.estimate(), None);
        assert_eq!(outcome.certificate.uncertainty(), None);
        assert_eq!(outcome.adjustment_set, None);
    }
}
