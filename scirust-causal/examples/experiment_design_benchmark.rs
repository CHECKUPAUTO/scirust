//! Phase 5C.8 deterministic benchmark: ranking experiments by what they would
//! settle, checked against hand-derived oracles.
//!
//! # What this demonstrates
//!
//! The first row closes the loop phase 5C.7 opened. That phase priced the
//! Markov-equivalence class in the units of the question being asked: two
//! models no observational test can separate answered one counterfactual `3`
//! and `1`. Here the same ambiguity — rediscovered from simulated data, not
//! asserted — is handed to the planner, which names the single experiment
//! that settles it.
//!
//! The `undirected_triangle` row is the one worth staring at. Every target
//! orients its own two edges for certain and the third only if they land in a
//! chain, so the guarantee is `2` while the best case is `3`. A planner that
//! ranked on the best case would report an experiment as decisive when two of
//! its four possible outcomes are not.
//!
//! # Reproducibility contract
//!
//! Fixed [`SplitMix64`] seed, no wall-clock/hostname/thread-count input. All
//! scientific content goes to **stdout** in a fixed field order; nothing else
//! does. Two runs hashed with SHA-256 must be byte-identical.
//!
//! On any oracle mismatch this program prints a diagnostic to **stderr** and
//! exits with a non-zero status.

use scirust_causal::{
    CausalDataset, CausalVariable, ConditionalIndependenceConfig, ConditionalIndependenceMethod,
    Cpdag, EquivalenceClassConfig, EquivalenceClassDiscovery, ExperimentDesignConfig,
    ExperimentPlan, ExperimentPlanOutcome, LinearScm, PartialCorrelationTest, PcStable,
    VariableKind, VariableRole, greedy_experiment_sequence, plan_next_experiment,
};
use scirust_solvers::Matrix;
use scirust_stats::SplitMix64;

fn expect(condition: bool, description: String) {
    if !condition
    {
        eprintln!("ORACLE FAILURE: {description}");
        std::process::exit(1);
    }
}

/// Unit-variance noise: `U(-0.5, 0.5)` has variance `1/12`.
fn unit_noise(rng: &mut SplitMix64) -> f64 {
    12.0_f64.sqrt() * (rng.next_f64() - 0.5)
}

/// Rediscovers the 5C.7 ambiguity from simulated data rather than asserting
/// it: `X = ε₀`, `Y = X + ε₁`, run through PC-Stable.
fn discovered_two_variable_cpdag() -> Cpdag {
    let scm = LinearScm::new(
        Matrix::from_row_major(2, 2, vec![0.0, 0.0, 1.0, 0.0]),
        vec![0.0, 0.0],
    )
    .unwrap();
    let mut rng = SplitMix64::new(8001);
    let n = 2000;
    let mut data = vec![0.0; n * 2];
    for row in 0..n
    {
        let world = scm
            .simulate(&[unit_noise(&mut rng), unit_noise(&mut rng)], &[])
            .unwrap();
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
        scirust_causal::Environment::observational("obs").unwrap(),
        &Matrix::from_row_major(n, 2, data),
        "5C.8 benchmark fixture",
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

fn describe_outcome(outcome: &ExperimentPlanOutcome) -> String {
    match outcome
    {
        ExperimentPlanOutcome::Recommended { target } => format!("Recommended(v{target})"),
        ExperimentPlanOutcome::AlreadyDetermined => "AlreadyDetermined".to_string(),
        ExperimentPlanOutcome::NoInformativeExperiment => "NoInformativeExperiment".to_string(),
    }
}

fn report(scenario: &str, plan: &ExperimentPlan) {
    let best = plan.candidates.first();
    println!(
        "scenario={scenario} undirected_before={} forced_by_graph_alone={} outcome={} \
         candidates={} best_target={} edges_at_target={} guaranteed={} optimistic={} \
         fully_enumerated={} resolves_all={} status={:?} estimate={:?} warnings={}",
        plan.undirected_before,
        plan.forced_by_graph_alone,
        describe_outcome(&plan.outcome),
        plan.candidates.len(),
        best.map_or(-1_i64, |c| c.target as i64),
        best.map_or(0, |c| c.edges_at_target),
        best.map_or(0, |c| c.guaranteed_orientations),
        best.map_or(0, |c| c.optimistic_orientations),
        best.is_some_and(|c| c.fully_enumerated),
        best.is_some_and(|c| c.resolves_everything),
        plan.certificate.status(),
        plan.certificate.estimate(),
        plan.warnings.len(),
    );
}

fn main() {
    println!("# Phase 5C.8 deterministic experiment-design benchmark");
    println!(
        "# fields: scenario undirected_before forced_by_graph_alone outcome candidates \
         best_target edges_at_target guaranteed optimistic fully_enumerated resolves_all status \
         estimate warnings"
    );
    println!(
        "# scope: these counts say what an experiment would REVEAL about graph structure under a \
         perfect intervention; they are not effect sizes and carry no cost model"
    );

    let default = ExperimentDesignConfig::default();

    // ── The 5C.7 ambiguity, rediscovered and then priced in experiments ──
    let discovered = discovered_two_variable_cpdag();
    expect(
        discovered.is_undirected(0, 1),
        format!(
            "markov_equivalent_pair: discovery should leave the edge undirected, got {discovered:?}"
        ),
    );
    let plan = plan_next_experiment(&discovered, &default).unwrap();
    report("markov_equivalent_pair", &plan);
    let sequence = greedy_experiment_sequence(&discovered, &default).unwrap();
    println!(
        "# rung_three_price_in_experiments: 5C.7 left this pair answering one counterfactual \
         3 vs 1 with no observational tiebreak; {} experiment(s) close it -- steps={:?} \
         fully_oriented={}",
        sequence.steps.len(),
        sequence.steps,
        sequence.fully_oriented,
    );
    expect(
        plan.outcome == ExperimentPlanOutcome::Recommended { target: 0 },
        format!(
            "markov_equivalent_pair: expected Recommended(v0), got {:?}",
            plan.outcome
        ),
    );
    expect(
        sequence.steps.len() == 1 && sequence.fully_oriented,
        format!(
            "markov_equivalent_pair: one experiment should suffice, got {:?}",
            sequence.steps
        ),
    );

    // ── Guarantee versus best case ───────────────────────────────────────
    let triangle = Cpdag::from_edges(3, &[], &[(0, 1), (0, 2), (1, 2)]).unwrap();
    let plan = plan_next_experiment(&triangle, &default).unwrap();
    report("undirected_triangle", &plan);
    let triangle_sequence = greedy_experiment_sequence(&triangle, &default).unwrap();
    println!(
        "# guarantee_versus_best_case: every target orients 2 edges for certain and 3 only when \
         they land in a chain, so ranking on the best case would call one experiment decisive \
         when 2 of its 4 outcomes are not -- worst-case sequence needs {} experiments",
        triangle_sequence.steps.len(),
    );
    for candidate in &plan.candidates
    {
        expect(
            candidate.guaranteed_orientations == 2
                && candidate.optimistic_orientations == 3
                && !candidate.resolves_everything,
            format!("undirected_triangle: expected (2, 3, false), got {candidate:?}"),
        );
    }
    expect(
        triangle_sequence.steps.len() == 2 && triangle_sequence.fully_oriented,
        format!(
            "undirected_triangle: expected 2 experiments, got {:?}",
            triangle_sequence.steps
        ),
    );

    // ── A hub is worth more than a leaf, and the ranking says so ─────────
    let star = Cpdag::from_edges(4, &[], &[(0, 1), (0, 2), (0, 3)]).unwrap();
    let plan = plan_next_experiment(&star, &default).unwrap();
    report("star_hub_beats_leaf", &plan);
    let leaf = plan.candidates.iter().find(|c| c.target == 1).unwrap();
    println!(
        "# ranking: hub v0 guarantees {} orientation(s), leaf v1 guarantees {}",
        plan.candidates[0].guaranteed_orientations, leaf.guaranteed_orientations,
    );
    expect(
        plan.candidates[0].target == 0 && plan.candidates[0].guaranteed_orientations == 3,
        format!(
            "star_hub_beats_leaf: expected v0 with 3, got {:?}",
            plan.candidates[0]
        ),
    );
    expect(
        leaf.guaranteed_orientations == 1,
        format!("star_hub_beats_leaf: expected leaf guarantee 1, got {leaf:?}"),
    );

    // ── Meek's rule 4 fires where R1-R3 cannot ───────────────────────────
    //
    // `0 - 1`, `0 - 2`, `0 - 3` undirected, `2 -> 3 -> 1`, with `1` and `2`
    // non-adjacent. Intervening on 2 injects an orientation that is not a
    // v-structure, which is precisely the setting R4 exists for.
    let rule_4_shape = Cpdag::from_edges(4, &[(2, 3), (3, 1)], &[(0, 1), (0, 2), (0, 3)]).unwrap();
    let plan = plan_next_experiment(&rule_4_shape, &default).unwrap();
    report("meek_rule_4_closes_the_input", &plan);
    println!(
        "# meek_rule_4: 3 edges were written down as undirected, but R4 forces {} of them from \
         the graph alone, leaving {} for an experiment to settle. Phase 5C.3 left R4 out because \
         nothing injected non-v-structure orientations; planning an intervention does. R1-R3 \
         alone would have credited that edge to an experiment that does not earn it.",
        plan.forced_by_graph_alone, plan.undirected_before,
    );
    expect(
        plan.forced_by_graph_alone == 1 && plan.undirected_before == 2,
        format!(
            "meek_rule_4_closes_the_input: expected 1 forced and 2 open, got {} and {}",
            plan.forced_by_graph_alone, plan.undirected_before
        ),
    );
    expect(
        plan.candidates
            .iter()
            .all(|c| c.guaranteed_orientations <= 2),
        "meek_rule_4_closes_the_input: no candidate may claim the edge the graph settled"
            .to_string(),
    );

    // ── An ambiguity out of reach is reported, not rounded away ──────────
    let restricted = ExperimentDesignConfig {
        feasible_targets: Some(vec![0]),
        ..Default::default()
    };
    let unreachable = Cpdag::from_edges(3, &[], &[(1, 2)]).unwrap();
    let plan = plan_next_experiment(&unreachable, &restricted).unwrap();
    report("ambiguity_out_of_reach", &plan);
    expect(
        plan.outcome == ExperimentPlanOutcome::NoInformativeExperiment,
        format!(
            "ambiguity_out_of_reach: expected NoInformativeExperiment, got {:?}",
            plan.outcome
        ),
    );

    // ── Nothing left to learn is a result ────────────────────────────────
    let settled = Cpdag::from_edges(3, &[(0, 1), (1, 2)], &[]).unwrap();
    let plan = plan_next_experiment(&settled, &default).unwrap();
    report("already_determined", &plan);
    expect(
        plan.outcome == ExperimentPlanOutcome::AlreadyDetermined,
        format!(
            "already_determined: expected AlreadyDetermined, got {:?}",
            plan.outcome
        ),
    );

    // ── The enumeration cap downgrades honestly ──────────────────────────
    let wide = Cpdag::from_edges(5, &[], &[(0, 1), (0, 2), (0, 3), (0, 4)]).unwrap();
    let capped = plan_next_experiment(
        &wide,
        &ExperimentDesignConfig {
            max_outcome_enumeration: 8, // 2^4 = 16 > 8
            ..Default::default()
        },
    )
    .unwrap();
    report("enumeration_capped", &capped);
    let full = plan_next_experiment(&wide, &default).unwrap();
    report("enumeration_full", &full);
    println!(
        "# enumeration_cap: capped guarantee {} vs full guarantee {} -- the capped number is a \
         lower bound, never an overclaim",
        capped.candidates[0].guaranteed_orientations, full.candidates[0].guaranteed_orientations,
    );
    expect(
        !capped.candidates[0].fully_enumerated && !capped.warnings.is_empty(),
        "enumeration_capped: the cap must be disclosed".to_string(),
    );
    expect(
        full.candidates[0].guaranteed_orientations >= capped.candidates[0].guaranteed_orientations,
        format!(
            "enumeration_cap: capped {} must not exceed full {}",
            capped.candidates[0].guaranteed_orientations,
            full.candidates[0].guaranteed_orientations
        ),
    );

    println!("# all oracle checks passed");
}
