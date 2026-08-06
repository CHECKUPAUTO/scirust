//! Structural counterfactuals, and the price of an unoriented edge.
//!
//! The headline test builds two structural causal models — `X → Y` and
//! `Y → X` — with coefficients chosen so their joint distributions are
//! **identical**. That identity is verified empirically here, not asserted;
//! `PcStable` is then run on data from one of them and shown to return the
//! undirected edge that says so. The two models then answer the *same*
//! counterfactual question differently. That gap is what rung 3 costs, and no
//! amount of observational data closes it.

use scirust_causal::{
    CausalDataset, CausalError, CausalVariable, ConditionalIndependenceConfig,
    ConditionalIndependenceMethod, CounterfactualQuery, Environment, EquivalenceClassConfig,
    EquivalenceClassDiscovery, IdentifiabilityStatus, InterventionAssignment, LinearScm,
    PartialCorrelationTest, PcStable, VariableKind, VariableRole,
};
use scirust_solvers::Matrix;
use scirust_stats::SplitMix64;

// ─── The Markov-equivalent pair ──────────────────────────────────────────
//
// SCM A:  X = ε₀            (var 1)      Y = 1·X + ε₁   (var 1)
// SCM B:  Y = φ₁ (var 2)                 X = 0.5·Y + φ₀ (var 0.5)
//
// Both give var(X) = 1, var(Y) = 2, cov(X, Y) = 1 — hand-derived, and checked
// empirically below.

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

/// Unit-variance noise: `U(-0.5, 0.5)` has variance `1/12`, so scale by `√12`.
fn unit_noise(rng: &mut SplitMix64) -> f64 {
    12.0_f64.sqrt() * (rng.next_f64() - 0.5)
}

fn sample_worlds(scm: &LinearScm, scales: [f64; 2], seed: u64, n: usize) -> Vec<Vec<f64>> {
    let mut rng = SplitMix64::new(seed);
    (0..n)
        .map(|_| {
            let noise = [
                scales[0] * unit_noise(&mut rng),
                scales[1] * unit_noise(&mut rng),
            ];
            scm.simulate(&noise, &[]).unwrap()
        })
        .collect()
}

fn moments(worlds: &[Vec<f64>]) -> (f64, f64, f64) {
    let n = worlds.len() as f64;
    let mean_x = worlds.iter().map(|w| w[0]).sum::<f64>() / n;
    let mean_y = worlds.iter().map(|w| w[1]).sum::<f64>() / n;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    let mut cov = 0.0;
    for world in worlds
    {
        let dx = world[0] - mean_x;
        let dy = world[1] - mean_y;
        var_x += dx * dx;
        var_y += dy * dy;
        cov += dx * dy;
    }
    (var_x / (n - 1.0), var_y / (n - 1.0), cov / (n - 1.0))
}

#[test]
fn the_two_models_are_observationally_indistinguishable() {
    // SCM A's noise: ε₀ var 1, ε₁ var 1.
    let a = sample_worlds(&scm_x_causes_y(), [1.0, 1.0], 90, 40_000);
    // SCM B's noise: φ₀ var 0.5, φ₁ var 2.
    let b = sample_worlds(
        &scm_y_causes_x(),
        [0.5_f64.sqrt(), 2.0_f64.sqrt()],
        91,
        40_000,
    );

    let (var_x_a, var_y_a, cov_a) = moments(&a);
    let (var_x_b, var_y_b, cov_b) = moments(&b);

    // Both must match the hand-derived (1, 2, 1) and therefore each other.
    for (label, value, expected) in [
        ("var(X) of A", var_x_a, 1.0),
        ("var(Y) of A", var_y_a, 2.0),
        ("cov of A", cov_a, 1.0),
        ("var(X) of B", var_x_b, 1.0),
        ("var(Y) of B", var_y_b, 2.0),
        ("cov of B", cov_b, 1.0),
    ]
    {
        assert!(
            (value - expected).abs() < 0.05,
            "{label} should be ~{expected}, got {value}"
        );
    }
}

#[test]
fn discovery_reports_the_edge_as_unoriented_because_it_cannot_tell_them_apart() {
    // Data from SCM A. PC-Stable finds the edge but cannot orient it -- with
    // two variables there is no unshielded triple to form a v-structure, which
    // is exactly the ambiguity the two SCMs above embody.
    let worlds = sample_worlds(&scm_x_causes_y(), [1.0, 1.0], 92, 2000);
    let n = worlds.len();
    let mut data = vec![0.0; n * 2];
    for (row, world) in worlds.iter().enumerate()
    {
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
    let cpdag = PcStable::new(EquivalenceClassConfig::new())
        .discover(&dataset, &test)
        .unwrap()
        .cpdag;

    assert!(cpdag.is_adjacent(0, 1), "the dependence must be found");
    assert!(
        cpdag.is_undirected(0, 1),
        "and the direction must be reported as undetermined"
    );
}

#[test]
fn markov_equivalent_models_give_different_counterfactual_answers() {
    // The same observed world, the same question, two models no observational
    // test can separate.
    let factual = vec![1.0, 1.0]; // X = 1, Y = 1
    let query = CounterfactualQuery {
        factual: factual.clone(),
        interventions: vec![InterventionAssignment::new(0, 3.0)], // had X been 3
        outcome: 1,                                               // what would Y have been?
    };

    let under_a = scm_x_causes_y().counterfactual(&query).unwrap();
    let under_b = scm_y_causes_x().counterfactual(&query).unwrap();

    // A: ε₁ = Y − X = 0, so Y would have been 3·1 + 0 = 3.
    assert!(
        (under_a.counterfactual_value() - 3.0).abs() < 1e-12,
        "under X → Y, Y should become 3, got {}",
        under_a.counterfactual_value()
    );
    // B: Y is upstream of X, so intervening on X cannot move it: Y stays 1.
    assert!(
        (under_b.counterfactual_value() - 1.0).abs() < 1e-12,
        "under Y → X, Y should be unmoved at 1, got {}",
        under_b.counterfactual_value()
    );

    // Both are certified -- correctly, each *relative to its own model*.
    assert_eq!(
        under_a.certificate.status(),
        IdentifiabilityStatus::Identifiable
    );
    assert_eq!(
        under_b.certificate.status(),
        IdentifiabilityStatus::Identifiable
    );
    // And both say plainly what that certification rests on.
    for outcome in [&under_a, &under_b]
    {
        let note = outcome.certificate.sensitivity_note().unwrap();
        assert!(
            note.contains("RELATIVE to the supplied structural causal model"),
            "the certificate must scope its claim to the model: {note}"
        );
        assert!(
            !outcome.certificate.unresolved_alternatives().is_empty(),
            "the rival models must be recorded as unresolved"
        );
    }
}

// ─── Rung 2 is not rung 3 ────────────────────────────────────────────────

#[test]
fn a_counterfactual_differs_from_the_population_interventional_mean() {
    // Under SCM A, E[Y | do(X = 3)] = 3 because E[ε₁] = 0. But for the
    // particular unit we observed, ε₁ = 1, so *its* counterfactual is 4.
    // Averaging over units and reasoning about one unit are different
    // questions; only the second needs abduction.
    let scm = scm_x_causes_y();

    let population = scm
        .simulate(&[0.0, 0.0], &[InterventionAssignment::new(0, 3.0)])
        .unwrap();
    assert!((population[1] - 3.0).abs() < 1e-12);

    let outcome = scm
        .counterfactual(&CounterfactualQuery {
            factual: vec![1.0, 2.0], // this unit has ε₁ = 2 − 1 = 1
            interventions: vec![InterventionAssignment::new(0, 3.0)],
            outcome: 1,
        })
        .unwrap();
    assert!(
        (outcome.counterfactual_value() - 4.0).abs() < 1e-12,
        "this unit's counterfactual should be 4, got {}",
        outcome.counterfactual_value()
    );
    assert!((outcome.recovered_noise[1] - 1.0).abs() < 1e-12);
}

// ─── Contracts ───────────────────────────────────────────────────────────

#[test]
fn intervening_on_the_outcome_itself_is_flagged() {
    let scm = scm_x_causes_y();
    let outcome = scm
        .counterfactual(&CounterfactualQuery {
            factual: vec![1.0, 1.0],
            interventions: vec![InterventionAssignment::new(1, 7.0)],
            outcome: 1,
        })
        .unwrap();
    assert!((outcome.counterfactual_value() - 7.0).abs() < 1e-12);
    assert!(
        outcome
            .warnings
            .iter()
            .any(|w| w.contains("itself intervened on")),
        "setting the outcome and then reporting it is not a model consequence: {:?}",
        outcome.warnings
    );
}

#[test]
fn the_certificate_carries_the_counterfactual_value_and_zero_computational_uncertainty() {
    let scm = scm_x_causes_y();
    let outcome = scm
        .counterfactual(&CounterfactualQuery {
            factual: vec![1.0, 1.0],
            interventions: vec![InterventionAssignment::new(0, 3.0)],
            outcome: 1,
        })
        .unwrap();
    assert_eq!(
        outcome.certificate.estimate(),
        Some(outcome.counterfactual_value())
    );
    assert_eq!(outcome.certificate.uncertainty(), Some(0.0));
    assert!(
        outcome
            .certificate
            .sensitivity_note()
            .unwrap()
            .contains("computational, not epistemic"),
        "a zero uncertainty must be explained, not left to be misread"
    );
}

#[test]
fn results_are_deterministic_and_json_round_trip() {
    let scm = scm_x_causes_y();
    let query = CounterfactualQuery {
        factual: vec![1.0, 1.0],
        interventions: vec![InterventionAssignment::new(0, 3.0)],
        outcome: 1,
    };
    let first = scm.counterfactual(&query).unwrap();
    let second = scm.counterfactual(&query).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first.certificate.fingerprint(),
        second.certificate.fingerprint()
    );

    let json = serde_json::to_string(&first).unwrap();
    let round_tripped: scirust_causal::CounterfactualOutcome = serde_json::from_str(&json).unwrap();
    assert_eq!(first, round_tripped);
}

#[test]
fn a_partially_sized_factual_is_a_typed_error() {
    // Abduction needs every variable; there is deliberately no imputation.
    let scm = scm_x_causes_y();
    assert!(matches!(
        scm.counterfactual(&CounterfactualQuery {
            factual: vec![1.0],
            interventions: vec![],
            outcome: 1,
        }),
        Err(CausalError::DimensionMismatch {
            expected: 2,
            got: 1
        })
    ));
}

#[test]
fn a_non_finite_factual_is_a_typed_error() {
    let scm = scm_x_causes_y();
    assert!(matches!(
        scm.counterfactual(&CounterfactualQuery {
            factual: vec![1.0, f64::NAN],
            interventions: vec![],
            outcome: 1,
        }),
        Err(CausalError::NonFiniteInput { index: 1, .. })
    ));
}

#[test]
fn abduction_then_simulation_round_trips_on_a_larger_model() {
    // A 4-variable model with a fork and a collider, given out of topological
    // order to confirm ordering is handled rather than assumed.
    //  v3 -> v1, v3 -> v2, v1 -> v0, v2 -> v0
    let coefficients = Matrix::from_row_major(
        4,
        4,
        vec![
            0.0, 2.0, -1.0, 0.0, //
            0.0, 0.0, 0.0, 0.5, //
            0.0, 0.0, 0.0, 1.5, //
            0.0, 0.0, 0.0, 0.0,
        ],
    );
    let scm = LinearScm::new(coefficients, vec![0.1, -0.2, 0.3, 0.4]).unwrap();

    let noise = vec![0.7, -1.1, 2.3, 0.9];
    let world = scm.simulate(&noise, &[]).unwrap();
    let recovered = scm.abduct(&world).unwrap();
    for (original, recovered) in noise.iter().zip(&recovered)
    {
        assert!(
            (original - recovered).abs() < 1e-12,
            "abduction must invert simulation exactly: {original} vs {recovered}"
        );
    }

    // And a counterfactual on the root propagates to everything downstream.
    let outcome = scm
        .counterfactual(&CounterfactualQuery {
            factual: world.clone(),
            interventions: vec![InterventionAssignment::new(3, 10.0)],
            outcome: 0,
        })
        .unwrap();
    assert!(
        (outcome.counterfactual_value() - outcome.factual_value()).abs() > 1e-6,
        "intervening on an ancestor must move the outcome"
    );
    assert!(
        (outcome.counterfactual_world[3] - 10.0).abs() < 1e-12,
        "the intervened variable must hold its assigned value"
    );
}
