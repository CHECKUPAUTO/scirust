//! Which experiment buys you out of the equivalence class?
//!
//! Phase 5C.7 ended on a bill: two structural models no observational test can
//! separate answer the same counterfactual `3` and `1`. The headline test here
//! closes that loop end to end — run discovery on data from one of them, get
//! the undirected edge back, hand it to the planner, and check that the
//! experiment it names is exactly the one that settles the disagreement.
//!
//! The rest is about the planner's honesty rather than its usefulness: that it
//! reports a *guarantee* rather than a best case, that it says so when an
//! ambiguity is out of reach, and that it never dresses a structural count up
//! as an effect estimate.

use scirust_causal::{
    CausalDataset, CausalError, CausalVariable, ConditionalIndependenceConfig,
    ConditionalIndependenceMethod, CounterfactualQuery, Cpdag, Environment, EquivalenceClassConfig,
    EquivalenceClassDiscovery, ExperimentDesignConfig, ExperimentPlanOutcome,
    IdentifiabilityStatus, InterventionAssignment, LinearScm, PartialCorrelationTest, PcStable,
    VariableKind, VariableRole, greedy_experiment_sequence, plan_next_experiment,
};
use scirust_solvers::Matrix;
use scirust_stats::SplitMix64;

// ─── The 5C.7 pair, unchanged ────────────────────────────────────────────

/// `X(0) → Y(1)` with coefficient 1.
fn scm_x_causes_y() -> LinearScm {
    LinearScm::new(
        Matrix::from_row_major(2, 2, vec![0.0, 0.0, 1.0, 0.0]),
        vec![0.0, 0.0],
    )
    .unwrap()
}

/// `Y(1) → X(0)` with coefficient 0.5.
fn scm_y_causes_x() -> LinearScm {
    LinearScm::new(
        Matrix::from_row_major(2, 2, vec![0.0, 0.5, 0.0, 0.0]),
        vec![0.0, 0.0],
    )
    .unwrap()
}

fn unit_noise(rng: &mut SplitMix64) -> f64 {
    12.0_f64.sqrt() * (rng.next_f64() - 0.5)
}

fn discovered_cpdag_from_scm_a() -> Cpdag {
    let mut rng = SplitMix64::new(92);
    let n = 2000;
    let mut data = vec![0.0; n * 2];
    for row in 0..n
    {
        let noise = [unit_noise(&mut rng), unit_noise(&mut rng)];
        let world = scm_x_causes_y().simulate(&noise, &[]).unwrap();
        data[row * 2] = world[0];
        data[row * 2 + 1] = world[1];
    }
    let variables: Vec<CausalVariable> = (0..2)
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
    let dataset = CausalDataset::single_environment(
        variables,
        Environment::observational("obs").unwrap(),
        &Matrix::from_row_major(n, 2, data),
        "test fixture",
    )
    .unwrap();
    let test = PartialCorrelationTest::new(
        ConditionalIndependenceConfig::new(
            0.05,
            ConditionalIndependenceMethod::GaussianPartialCorrelation { fisher_z: true },
        )
        .unwrap(),
    );
    PcStable::new(EquivalenceClassConfig::new())
        .discover(&dataset, &test)
        .unwrap()
        .cpdag
}

// ─── The headline: naming the experiment that settles 5C.7 ───────────────

#[test]
fn the_planner_names_the_experiment_that_resolves_the_counterfactual_disagreement() {
    // 1. The disagreement, restated. Same unit, same question, two models.
    let query = CounterfactualQuery {
        factual: vec![1.0, 1.0],
        interventions: vec![InterventionAssignment::new(0, 3.0)],
        outcome: 1,
    };
    let under_a = scm_x_causes_y().counterfactual(&query).unwrap();
    let under_b = scm_y_causes_x().counterfactual(&query).unwrap();
    assert!((under_a.counterfactual_value() - 3.0).abs() < 1e-12);
    assert!((under_b.counterfactual_value() - 1.0).abs() < 1e-12);

    // 2. Discovery, run on real simulated data, reports it cannot choose.
    let cpdag = discovered_cpdag_from_scm_a();
    assert!(cpdag.is_undirected(0, 1));

    // 3. The planner takes that CPDAG and names an experiment.
    let plan = plan_next_experiment(&cpdag, &ExperimentDesignConfig::default()).unwrap();
    let ExperimentPlanOutcome::Recommended { target } = plan.outcome
    else
    {
        panic!(
            "an undirected edge should be resolvable, got {:?}",
            plan.outcome
        );
    };

    // 4. And that experiment settles the whole thing, whatever it finds.
    let recommended = plan
        .candidates
        .iter()
        .find(|c| c.target == target)
        .expect("the recommended target must be among the candidates");
    assert_eq!(recommended.guaranteed_orientations, 1);
    assert!(
        recommended.resolves_everything,
        "intervening on either endpoint of the only undirected edge resolves it"
    );
    assert_eq!(plan.undirected_before, 1);

    // The whole point: one experiment converts "3 or 1, no way to tell" into
    // an answer. That is what the equivalence class costs, priced in
    // experiments rather than in prose.
    let sequence = greedy_experiment_sequence(&cpdag, &ExperimentDesignConfig::default()).unwrap();
    assert!(sequence.fully_oriented);
    assert_eq!(sequence.steps.len(), 1);
}

#[test]
fn adding_rule_4_left_the_discovery_pipeline_untouched() {
    // Meek proves R1-R3 complete when every orientation came from a
    // v-structure. This phase added R4 behind a separate entry point on that
    // basis, so a CPDAG from the untouched pipeline must be exactly what it
    // was before: for the two-variable case, one undirected edge and nothing
    // oriented.
    let cpdag = discovered_cpdag_from_scm_a();
    assert_eq!(cpdag.n_edges(), 1);
    assert_eq!(cpdag.undirected_edges(), vec![(0, 1)]);
    assert!(cpdag.directed_edges().is_empty());
}

// ─── Honesty properties ──────────────────────────────────────────────────

#[test]
fn the_ranking_reports_a_guarantee_not_a_best_case() {
    // The undirected triangle: every target orients its own two edges for
    // certain, and the third only when the two land in a chain. A planner
    // that ranked on the best case would claim 3.
    let triangle = Cpdag::from_edges(3, &[], &[(0, 1), (0, 2), (1, 2)]).unwrap();
    let plan = plan_next_experiment(&triangle, &ExperimentDesignConfig::default()).unwrap();
    for candidate in &plan.candidates
    {
        assert_eq!(candidate.guaranteed_orientations, 2);
        assert_eq!(candidate.optimistic_orientations, 3);
        assert!(
            !candidate.resolves_everything,
            "two of the four outcomes leave the third edge genuinely free"
        );
    }
    // And the sequence, which advances on the worst outcome, needs a second
    // experiment because of it.
    let sequence =
        greedy_experiment_sequence(&triangle, &ExperimentDesignConfig::default()).unwrap();
    assert!(sequence.fully_oriented);
    assert_eq!(sequence.steps.len(), 2);
}

#[test]
fn an_unreachable_ambiguity_is_reported_rather_than_rounded_away() {
    // Only variable 0 may be intervened on; the open edge is between 1 and 2.
    let graph = Cpdag::from_edges(3, &[], &[(1, 2)]).unwrap();
    let config = ExperimentDesignConfig {
        feasible_targets: Some(vec![0]),
        ..Default::default()
    };
    let plan = plan_next_experiment(&graph, &config).unwrap();
    assert_eq!(plan.outcome, ExperimentPlanOutcome::NoInformativeExperiment);
    assert_eq!(
        plan.certificate.status(),
        IdentifiabilityStatus::NotIdentifiable
    );
    assert_eq!(plan.certificate.estimate(), None);

    let sequence = greedy_experiment_sequence(&graph, &config).unwrap();
    assert!(!sequence.fully_oriented);
    assert!(sequence.steps.is_empty());
    assert_eq!(sequence.undirected_remaining, 1);
    assert!(
        sequence
            .certificate
            .unresolved_alternatives()
            .iter()
            .any(|a| a.contains("stay undirected"))
    );
}

#[test]
fn a_fully_directed_graph_is_a_result_not_a_failure() {
    let chain = Cpdag::from_edges(3, &[(0, 1), (1, 2)], &[]).unwrap();
    let plan = plan_next_experiment(&chain, &ExperimentDesignConfig::default()).unwrap();
    assert_eq!(plan.outcome, ExperimentPlanOutcome::AlreadyDetermined);
    assert_eq!(
        plan.certificate.status(),
        IdentifiabilityStatus::Identifiable
    );
    assert_eq!(plan.certificate.estimate(), None);

    let sequence = greedy_experiment_sequence(&chain, &ExperimentDesignConfig::default()).unwrap();
    assert!(sequence.fully_oriented);
    assert!(sequence.steps.is_empty());
}

#[test]
fn a_structural_count_is_never_dressed_up_as_an_effect_estimate() {
    let star = Cpdag::from_edges(4, &[], &[(0, 1), (0, 2), (0, 3)]).unwrap();
    let plan = plan_next_experiment(&star, &ExperimentDesignConfig::default()).unwrap();
    assert_eq!(plan.certificate.estimate(), None);
    assert_eq!(plan.certificate.uncertainty(), None);
    assert_eq!(plan.certificate.method(), None);
    let note = plan.certificate.sensitivity_note().unwrap();
    assert!(note.contains("not a claim about any effect size"));
    assert!(note.contains("worth its cost"));
}

#[test]
fn the_enumeration_cap_is_disclosed_and_still_yields_a_valid_lower_bound() {
    let star = Cpdag::from_edges(5, &[], &[(0, 1), (0, 2), (0, 3), (0, 4)]).unwrap();
    let capped = plan_next_experiment(
        &star,
        &ExperimentDesignConfig {
            max_outcome_enumeration: 8, // 2^4 = 16 > 8
            ..Default::default()
        },
    )
    .unwrap();
    let hub = capped.candidates.iter().find(|c| c.target == 0).unwrap();
    assert!(!hub.fully_enumerated);
    assert!(capped.warnings.iter().any(|w| w.contains("lower bound")));

    // The uncapped run must agree that the capped number was not an overclaim.
    let full = plan_next_experiment(&star, &ExperimentDesignConfig::default()).unwrap();
    let full_hub = full.candidates.iter().find(|c| c.target == 0).unwrap();
    assert!(full_hub.fully_enumerated);
    assert!(full_hub.guaranteed_orientations >= hub.guaranteed_orientations);
}

#[test]
fn feasible_targets_are_validated_and_order_insensitive() {
    let star = Cpdag::from_edges(3, &[], &[(0, 1), (0, 2)]).unwrap();
    let ascending = plan_next_experiment(
        &star,
        &ExperimentDesignConfig {
            feasible_targets: Some(vec![0, 1, 2]),
            ..Default::default()
        },
    )
    .unwrap();
    let shuffled_with_duplicates = plan_next_experiment(
        &star,
        &ExperimentDesignConfig {
            feasible_targets: Some(vec![2, 0, 1, 0]),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(ascending.candidates, shuffled_with_duplicates.candidates);
    assert_eq!(ascending.outcome, shuffled_with_duplicates.outcome);

    assert!(matches!(
        plan_next_experiment(
            &star,
            &ExperimentDesignConfig {
                feasible_targets: Some(vec![9]),
                ..Default::default()
            }
        ),
        Err(CausalError::UnknownVariableIndex { index: 9 })
    ));
}

#[test]
fn a_hand_written_graph_is_validated_before_it_is_planned_against() {
    // A pair carrying two edges, and a directed cycle: both rejected, because
    // the orientation rules reason *from* acyclicity and the one-edge
    // invariant.
    assert!(matches!(
        Cpdag::from_edges(2, &[(0, 1)], &[(0, 1)]),
        Err(CausalError::InvalidContract { .. })
    ));
    assert!(matches!(
        Cpdag::from_edges(3, &[(0, 1), (1, 2), (2, 0)], &[]),
        Err(CausalError::CyclicGraph)
    ));
}

#[test]
fn results_are_deterministic_and_json_round_trip() {
    let graph = Cpdag::from_edges(5, &[(3, 4)], &[(0, 1), (0, 2), (1, 2)]).unwrap();
    let config = ExperimentDesignConfig::default();
    let first = plan_next_experiment(&graph, &config).unwrap();
    let second = plan_next_experiment(&graph, &config).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.certificate.fingerprint(),
        second.certificate.fingerprint()
    );

    let encoded = serde_json::to_string(&first).unwrap();
    assert_eq!(
        serde_json::from_str::<scirust_causal::ExperimentPlan>(&encoded).unwrap(),
        first
    );

    let sequence = greedy_experiment_sequence(&graph, &config).unwrap();
    let encoded = serde_json::to_string(&sequence).unwrap();
    assert_eq!(
        serde_json::from_str::<scirust_causal::ExperimentSequence>(&encoded).unwrap(),
        sequence
    );
}
