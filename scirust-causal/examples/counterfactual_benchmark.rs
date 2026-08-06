//! Phase 5C.7 deterministic benchmark: structural counterfactuals, checked
//! against hand-derived oracles.
//!
//! # What this demonstrates
//!
//! The headline pair of rows is `markov_equivalent_a` / `markov_equivalent_b`.
//! Two structural causal models — `X → Y` and `Y → X` — have coefficients
//! chosen so their joint distributions are *identical* (verified here by
//! printing both empirical covariance structures side by side). They then
//! answer the same counterfactual question differently: `3` versus `1`. No
//! observational test can choose between them, so that gap is not a
//! statistical error more data would close — it is a modelling commitment.
//!
//! The remaining rows separate rung 2 from rung 3: the population
//! interventional mean and one unit's counterfactual are different questions,
//! and only the second needs abduction.
//!
//! # Reproducibility contract
//!
//! Fixed [`SplitMix64`] seeds, no wall-clock/hostname/thread-count input. All
//! scientific content goes to **stdout** in a fixed field order; nothing else
//! does. Two runs hashed with SHA-256 must be byte-identical.
//!
//! On any oracle mismatch this program prints a diagnostic to **stderr** and
//! exits with a non-zero status.

use scirust_causal::{
    CounterfactualOutcome, CounterfactualQuery, InterventionAssignment, LinearScm,
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

fn scm_x_causes_y() -> LinearScm {
    LinearScm::new(
        Matrix::from_row_major(2, 2, vec![0.0, 0.0, 1.0, 0.0]),
        vec![0.0, 0.0],
    )
    .unwrap()
}

fn scm_y_causes_x() -> LinearScm {
    LinearScm::new(
        Matrix::from_row_major(2, 2, vec![0.0, 0.5, 0.0, 0.0]),
        vec![0.0, 0.0],
    )
    .unwrap()
}

/// Empirical `(var(X), var(Y), cov(X, Y))` from `n` simulated worlds.
fn moments(scm: &LinearScm, scales: [f64; 2], seed: u64, n: usize) -> (f64, f64, f64) {
    let mut rng = SplitMix64::new(seed);
    let worlds: Vec<Vec<f64>> = (0..n)
        .map(|_| {
            let noise = [
                scales[0] * unit_noise(&mut rng),
                scales[1] * unit_noise(&mut rng),
            ];
            scm.simulate(&noise, &[]).unwrap()
        })
        .collect();
    let count = n as f64;
    let mean_x = worlds.iter().map(|w| w[0]).sum::<f64>() / count;
    let mean_y = worlds.iter().map(|w| w[1]).sum::<f64>() / count;
    let (mut var_x, mut var_y, mut cov) = (0.0, 0.0, 0.0);
    for world in &worlds
    {
        let dx = world[0] - mean_x;
        let dy = world[1] - mean_y;
        var_x += dx * dx;
        var_y += dy * dy;
        cov += dx * dy;
    }
    (
        var_x / (count - 1.0),
        var_y / (count - 1.0),
        cov / (count - 1.0),
    )
}

fn report(scenario: &str, outcome: &CounterfactualOutcome) {
    println!(
        "scenario={scenario} outcome_variable={} factual_value={:.12e} \
         counterfactual_value={:.12e} recovered_noise={:?} interventions={:?} status={:?} \
         uncertainty={:?} warnings={}",
        outcome.outcome,
        outcome.factual_value(),
        outcome.counterfactual_value(),
        outcome
            .recovered_noise
            .iter()
            .map(|v| format!("{v:.12e}"))
            .collect::<Vec<_>>(),
        outcome
            .interventions
            .iter()
            .map(|i| format!("do(v{}={:.12e})", i.target, i.value))
            .collect::<Vec<_>>(),
        outcome.certificate.status(),
        outcome.certificate.uncertainty(),
        outcome.warnings.len(),
    );
}

fn main() {
    println!("# Phase 5C.7 deterministic structural-counterfactual benchmark");
    println!(
        "# fields: scenario outcome_variable factual_value counterfactual_value recovered_noise \
         interventions status uncertainty warnings"
    );
    println!(
        "# scope: counterfactuals are identified only RELATIVE to a supplied structural causal \
         model, which is assumed and is not identified by data"
    );

    // ── The two Markov-equivalent models have the same distribution ──────
    let (var_x_a, var_y_a, cov_a) = moments(&scm_x_causes_y(), [1.0, 1.0], 6001, 40_000);
    let (var_x_b, var_y_b, cov_b) = moments(
        &scm_y_causes_x(),
        [0.5_f64.sqrt(), 2.0_f64.sqrt()],
        6002,
        40_000,
    );
    println!(
        "# distribution_identity: model_a var_x={var_x_a:.6e} var_y={var_y_a:.6e} \
         cov={cov_a:.6e} | model_b var_x={var_x_b:.6e} var_y={var_y_b:.6e} cov={cov_b:.6e} \
         -- hand-derived target (1, 2, 1) for both"
    );
    for (label, value, expected) in [
        ("model_a var_x", var_x_a, 1.0),
        ("model_a var_y", var_y_a, 2.0),
        ("model_a cov", cov_a, 1.0),
        ("model_b var_x", var_x_b, 1.0),
        ("model_b var_y", var_y_b, 2.0),
        ("model_b cov", cov_b, 1.0),
    ]
    {
        expect(
            (value - expected).abs() < 0.05,
            format!("distribution_identity: {label} should be ~{expected}, got {value}"),
        );
    }

    // ── …and answer the same counterfactual differently ──────────────────
    let query = CounterfactualQuery {
        factual: vec![1.0, 1.0],
        interventions: vec![InterventionAssignment::new(0, 3.0)],
        outcome: 1,
    };
    let under_a = scm_x_causes_y().counterfactual(&query).unwrap();
    let under_b = scm_y_causes_x().counterfactual(&query).unwrap();
    report("markov_equivalent_a", &under_a);
    report("markov_equivalent_b", &under_b);
    println!(
        "# markov_equivalence_cost: same data, same question, answers {:.12e} vs {:.12e} \
         -- the direction is a modelling commitment, not something more data would settle",
        under_a.counterfactual_value(),
        under_b.counterfactual_value(),
    );
    expect(
        (under_a.counterfactual_value() - 3.0).abs() < 1e-12,
        format!(
            "markov_equivalent_a: expected 3, got {}",
            under_a.counterfactual_value()
        ),
    );
    expect(
        (under_b.counterfactual_value() - 1.0).abs() < 1e-12,
        format!(
            "markov_equivalent_b: expected 1, got {}",
            under_b.counterfactual_value()
        ),
    );

    // ── Rung 2 is not rung 3 ─────────────────────────────────────────────
    let scm = scm_x_causes_y();
    let population = scm
        .simulate(&[0.0, 0.0], &[InterventionAssignment::new(0, 3.0)])
        .unwrap();
    let unit = scm
        .counterfactual(&CounterfactualQuery {
            factual: vec![1.0, 2.0], // this unit carries ε₁ = 1
            interventions: vec![InterventionAssignment::new(0, 3.0)],
            outcome: 1,
        })
        .unwrap();
    report("unit_counterfactual", &unit);
    println!(
        "# rung_two_versus_three: population E[Y|do(X=3)]={:.12e} but this unit's \
         counterfactual={:.12e} -- averaging over units and reasoning about one unit are \
         different questions",
        population[1],
        unit.counterfactual_value(),
    );
    expect(
        (population[1] - 3.0).abs() < 1e-12,
        format!("rung_two: expected 3, got {}", population[1]),
    );
    expect(
        (unit.counterfactual_value() - 4.0).abs() < 1e-12,
        format!(
            "rung_three: expected 4, got {}",
            unit.counterfactual_value()
        ),
    );

    // ── A no-op counterfactual reproduces the factual world ──────────────
    let noop = scm
        .counterfactual(&CounterfactualQuery {
            factual: vec![1.0, 2.0],
            interventions: vec![],
            outcome: 1,
        })
        .unwrap();
    report("no_intervention", &noop);
    expect(
        (noop.counterfactual_value() - noop.factual_value()).abs() < 1e-12,
        "no_intervention: the counterfactual world must equal the factual one".to_string(),
    );
    expect(
        !noop.warnings.is_empty(),
        "no_intervention: the no-op should be flagged".to_string(),
    );

    // ── A four-variable model, given out of topological order ────────────
    let deep = LinearScm::new(
        Matrix::from_row_major(
            4,
            4,
            vec![
                0.0, 2.0, -1.0, 0.0, //
                0.0, 0.0, 0.0, 0.5, //
                0.0, 0.0, 0.0, 1.5, //
                0.0, 0.0, 0.0, 0.0,
            ],
        ),
        vec![0.1, -0.2, 0.3, 0.4],
    )
    .unwrap();
    let world = deep.simulate(&[0.7, -1.1, 2.3, 0.9], &[]).unwrap();
    let deep_outcome = deep
        .counterfactual(&CounterfactualQuery {
            factual: world,
            interventions: vec![InterventionAssignment::new(3, 10.0)],
            outcome: 0,
        })
        .unwrap();
    report("root_intervention_propagates", &deep_outcome);
    expect(
        (deep_outcome.counterfactual_world[3] - 10.0).abs() < 1e-12,
        "root_intervention_propagates: the intervened variable must hold its value".to_string(),
    );
    expect(
        (deep_outcome.counterfactual_value() - deep_outcome.factual_value()).abs() > 1e-6,
        "root_intervention_propagates: intervening on an ancestor must move the outcome"
            .to_string(),
    );

    println!("# all oracle checks passed");
}
