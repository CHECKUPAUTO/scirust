//! Sensitivity analysis against the concrete failure it was built to
//! quantify: the Phase 5C.4 latent-confounding scenario.
//!
//! The decisive test here reconstructs that scenario, measures how strong the
//! confounder *actually* was, and checks that the Cinelli–Hazlett bias bound
//! predicts the observed bias. If it does, the machinery is not merely
//! self-consistent — it recovers a real, independently-known quantity.

use scirust_causal::{
    AdjustmentStrategy, CausalDataset, CausalError, CausalVariable, ConfounderScenario,
    EffectEstimationConfig, Environment, RegimeSelection, VariableKind, VariableRole,
    analyze_sensitivity, benchmark_covariate, bound_effect_under_confounder,
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

/// Regenerates Phase 5C.4's latent-confounding scenario *exactly* (same seed,
/// same coefficients, same sample size), returning both what the analyst sees
/// and the ground truth.
///
/// `U -> X`, `U -> Y`, `X -> Y` with a true effect of `0.7`.
/// - `.0` = `[X, Y]`, the observed dataset — `U` is absent by construction.
/// - `.1` = `[U, X, Y]`, the ground truth, used only to *measure* how strong
///   the confounder really was.
fn latent_confounding_scenario() -> (CausalDataset, CausalDataset) {
    let mut rng = SplitMix64::new(60);
    let n = 4000;
    let (mut u, mut x, mut y) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n
    {
        let ui = noise(&mut rng);
        let xi = 0.9 * ui + 0.3 * noise(&mut rng);
        let yi = 0.7 * xi + 0.8 * ui + 0.3 * noise(&mut rng);
        u.push(ui);
        x.push(xi);
        y.push(yi);
    }
    (
        dataset_from_columns(&[x.clone(), y.clone()]),
        dataset_from_columns(&[u, x, y]),
    )
}

const TRUE_EFFECT: f64 = 0.7;

// ─── The decisive test ───────────────────────────────────────────────────

#[test]
fn the_bias_bound_recovers_the_actual_bias_of_the_5c4_latent_confounder() {
    let (observed, ground_truth) = latent_confounding_scenario();

    // 1. Reproduce 5C.4's wrong-but-certified estimate from the observed data
    //    and the misspecified graph "X -> Y, nothing else".
    let result = estimate_effect_from_dag(
        &observed,
        &dag_from_edges(2, &[(0, 1)]),
        0,
        1,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();
    let estimate = result.estimate.unwrap();
    let actual_bias = estimate - TRUE_EFFECT;
    assert!(
        actual_bias > 0.75 && actual_bias < 0.85,
        "precondition: 5C.4's scenario should reproduce a bias near 0.797, got {actual_bias}"
    );

    // 2. Measure how strong the confounder ACTUALLY was, from the ground-truth
    //    dataset where U is present. This is the one step a real analyst
    //    cannot take -- it is the point of comparison, not part of the method.
    let true_strength = benchmark_covariate(
        &ground_truth,
        1, // treatment X
        2, // outcome Y
        &[0],
        0, // covariate U
        &RegimeSelection::ObservationalOnly,
        1e-9,
    )
    .unwrap();
    assert!(
        (true_strength.treatment_share - 0.90).abs() < 0.02,
        "U should explain ~90% of X's variance, measured {}",
        true_strength.treatment_share
    );
    assert!(
        (true_strength.outcome_share - 0.416).abs() < 0.03,
        "U should explain ~41.6% of Y's residual variance, measured {}",
        true_strength.outcome_share
    );

    // 3. The payoff: had the analyst POSITED a confounder of that strength,
    //    what bias would the sensitivity analysis have predicted?
    let analysis = analyze_sensitivity(&result, 1.0).unwrap();
    let predicted = bound_effect_under_confounder(&analysis, &true_strength).unwrap();

    let relative_error = (predicted.bias_bound - actual_bias).abs() / actual_bias;
    assert!(
        relative_error < 0.05,
        "the predicted bias ({}) should match the actual bias ({actual_bias}) to within 5%, \
         relative error was {relative_error}",
        predicted.bias_bound
    );

    // And the adjusted range must reach the true effect -- the whole point of
    // reporting a range rather than a point.
    //
    // A small tolerance is required, and the reason matters. The
    // Cinelli-Hazlett bias is *exact* for a confounder of exactly the given
    // partial R² values; it is a "bound" only over the unknown *direction*,
    // not conservative in its inputs. Those inputs are measured from finite
    // data here, so they carry their own sampling error, and the prediction
    // can land slightly under the realised bias -- as it does: the range
    // stops about 2% of the bias short of the truth. Asserting exact
    // bracketing would be asserting something the method does not promise.
    let shortfall = predicted.adjusted_estimate_lower - TRUE_EFFECT;
    assert!(
        shortfall <= 0.05 * predicted.bias_bound
            && TRUE_EFFECT <= predicted.adjusted_estimate_upper,
        "the adjusted range [{}, {}] should reach the true effect {TRUE_EFFECT} to within 5% of \
         the bias bound; it fell short by {shortfall}",
        predicted.adjusted_estimate_lower,
        predicted.adjusted_estimate_upper
    );
}

#[test]
fn a_high_robustness_value_is_not_by_itself_reassurance() {
    // The lesson this phase exists to record. On the same scenario, RV_1 is
    // high (~0.93), which reads as "you would need an implausibly strong
    // confounder to overturn this". But the confounder that was actually
    // present was nearly that strong, and it moved the estimate by more than
    // its own magnitude away from the truth. RV is a threshold, not a verdict:
    // it must be read against what is plausible in the domain.
    let (observed, ground_truth) = latent_confounding_scenario();
    let result = estimate_effect_from_dag(
        &observed,
        &dag_from_edges(2, &[(0, 1)]),
        0,
        1,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();
    let analysis = analyze_sensitivity(&result, 1.0).unwrap();

    assert!(
        analysis.robustness_value > 0.9,
        "RV should look reassuringly high here, got {}",
        analysis.robustness_value
    );

    let true_strength = benchmark_covariate(
        &ground_truth,
        1,
        2,
        &[0],
        0,
        &RegimeSelection::ObservationalOnly,
        1e-9,
    )
    .unwrap();

    // The real confounder's treatment share is within striking distance of RV
    // on one axis, and the bias it produced was large -- so "RV is high" was
    // never the right thing to conclude from.
    assert!(
        true_strength.treatment_share > 0.85,
        "the real confounder was nearly as strong as RV on the treatment axis"
    );
    let predicted = bound_effect_under_confounder(&analysis, &true_strength).unwrap();
    assert!(
        predicted.bias_bound > 0.5 * result.estimate.unwrap().abs(),
        "a confounder of the strength actually present moves the estimate substantially"
    );
}

// ─── Behaviour on a genuinely unconfounded estimate ─────────────────────

#[test]
fn a_clean_estimate_is_not_moved_by_a_weak_confounder() {
    // X -> Y with coefficient 0.5 and no confounding at all.
    let mut rng = SplitMix64::new(70);
    let n = 3000;
    let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| 0.5 * xi + 0.3 * noise(&mut rng))
        .collect();
    let dataset = dataset_from_columns(&[x, y]);
    let result = estimate_effect_from_dag(
        &dataset,
        &dag_from_edges(2, &[(0, 1)]),
        0,
        1,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();
    let analysis = analyze_sensitivity(&result, 1.0).unwrap();

    let weak =
        bound_effect_under_confounder(&analysis, &ConfounderScenario::new(0.05, 0.05).unwrap())
            .unwrap();
    assert!(
        weak.sign_survives,
        "a weak confounder must not flip the sign"
    );
    assert!(
        weak.bias_bound < 0.1 * result.estimate.unwrap(),
        "a weak confounder should move a clean estimate very little, moved {}",
        weak.bias_bound
    );
}

#[test]
fn the_robustness_value_scales_with_the_strength_of_the_evidence() {
    // A stronger treatment-outcome relationship needs a stronger confounder
    // to explain it away, so RV must be larger for it.
    let build = |coefficient: f64, seed: u64| {
        let mut rng = SplitMix64::new(seed);
        let n = 3000;
        let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| coefficient * xi + 0.3 * noise(&mut rng))
            .collect();
        let dataset = dataset_from_columns(&[x, y]);
        let result = estimate_effect_from_dag(
            &dataset,
            &dag_from_edges(2, &[(0, 1)]),
            0,
            1,
            &AdjustmentStrategy::CanonicalParents,
            &EffectEstimationConfig::new(),
        )
        .unwrap();
        analyze_sensitivity(&result, 1.0).unwrap().robustness_value
    };

    let weak_relationship = build(0.05, 71);
    let strong_relationship = build(2.0, 71);
    assert!(
        strong_relationship > weak_relationship,
        "RV must increase with the strength of the evidence: weak={weak_relationship}, \
         strong={strong_relationship}"
    );
}

#[test]
fn a_smaller_q_needs_a_weaker_confounder() {
    // Halving an estimate takes less confounding than erasing it entirely.
    let (observed, _) = latent_confounding_scenario();
    let result = estimate_effect_from_dag(
        &observed,
        &dag_from_edges(2, &[(0, 1)]),
        0,
        1,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();

    let explain_away = analyze_sensitivity(&result, 1.0).unwrap().robustness_value;
    let halve = analyze_sensitivity(&result, 0.5).unwrap().robustness_value;
    assert!(
        halve < explain_away,
        "RV_0.5 ({halve}) must be below RV_1.0 ({explain_away})"
    );
}

// ─── Covariate benchmarking ──────────────────────────────────────────────

#[test]
fn benchmarking_an_irrelevant_covariate_reports_near_zero_strength() {
    // W(0) is independent of everything; X(1) -> Y(2).
    let mut rng = SplitMix64::new(72);
    let n = 3000;
    let w: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| 0.5 * xi + 0.3 * noise(&mut rng))
        .collect();
    let dataset = dataset_from_columns(&[w, x, y]);

    let strength = benchmark_covariate(
        &dataset,
        1,
        2,
        &[0],
        0,
        &RegimeSelection::ObservationalOnly,
        1e-9,
    )
    .unwrap();
    assert!(
        strength.treatment_share < 0.01 && strength.outcome_share < 0.01,
        "an irrelevant covariate should register as negligible, got {strength:?}"
    );
}

#[test]
fn benchmarking_requires_the_covariate_to_be_in_the_adjustment_set() {
    let (_, ground_truth) = latent_confounding_scenario();
    assert!(matches!(
        benchmark_covariate(
            &ground_truth,
            1,
            2,
            &[0],
            2, // not in the adjustment set
            &RegimeSelection::ObservationalOnly,
            1e-9,
        ),
        Err(CausalError::InvalidContract { .. })
    ));
}

// ─── Contract: no sensitivity analysis without an identified effect ─────

#[test]
fn sensitivity_analysis_refuses_a_non_identifiable_estimate() {
    // A confounded graph with a deliberately empty adjustment set: 5C.4
    // returns NotIdentifiable with no estimate, and there is nothing here to
    // be sensitive about.
    let mut rng = SplitMix64::new(73);
    let n = 2000;
    let (mut u, mut x, mut y) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n
    {
        let ui = noise(&mut rng);
        let xi = 0.9 * ui + 0.3 * noise(&mut rng);
        let yi = 0.7 * xi + 0.8 * ui + 0.3 * noise(&mut rng);
        u.push(ui);
        x.push(xi);
        y.push(yi);
    }
    let dataset = dataset_from_columns(&[u, x, y]);
    let result = estimate_effect_from_dag(
        &dataset,
        &dag_from_edges(3, &[(0, 1), (0, 2), (1, 2)]),
        1,
        2,
        &AdjustmentStrategy::Explicit(vec![]),
        &EffectEstimationConfig::new(),
    )
    .unwrap();
    assert_eq!(result.estimate, None);

    assert!(matches!(
        analyze_sensitivity(&result, 1.0),
        Err(CausalError::InvalidContract { .. })
    ));
}

#[test]
fn q_outside_its_range_is_a_typed_error() {
    let (observed, _) = latent_confounding_scenario();
    let result = estimate_effect_from_dag(
        &observed,
        &dag_from_edges(2, &[(0, 1)]),
        0,
        1,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();

    for bad in [0.0, -0.5, 1.5, f64::NAN]
    {
        assert!(
            matches!(
                analyze_sensitivity(&result, bad),
                Err(CausalError::InvalidConfiguration { name: "q", .. })
            ),
            "q = {bad} should be rejected"
        );
    }
}

#[test]
fn analysis_and_bound_json_round_trip() {
    let (observed, _) = latent_confounding_scenario();
    let result = estimate_effect_from_dag(
        &observed,
        &dag_from_edges(2, &[(0, 1)]),
        0,
        1,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();
    let analysis = analyze_sensitivity(&result, 1.0).unwrap();
    let bound =
        bound_effect_under_confounder(&analysis, &ConfounderScenario::new(0.3, 0.3).unwrap())
            .unwrap();

    let analysis_json = serde_json::to_string(&analysis).unwrap();
    let bound_json = serde_json::to_string(&bound).unwrap();
    assert_eq!(
        analysis,
        serde_json::from_str::<scirust_causal::SensitivityAnalysis>(&analysis_json).unwrap()
    );
    assert_eq!(
        bound,
        serde_json::from_str::<scirust_causal::AdjustedEffect>(&bound_json).unwrap()
    );
}

#[test]
fn sensitivity_analysis_is_deterministic() {
    let (observed, _) = latent_confounding_scenario();
    let result = estimate_effect_from_dag(
        &observed,
        &dag_from_edges(2, &[(0, 1)]),
        0,
        1,
        &AdjustmentStrategy::CanonicalParents,
        &EffectEstimationConfig::new(),
    )
    .unwrap();
    assert_eq!(
        analyze_sensitivity(&result, 1.0).unwrap(),
        analyze_sensitivity(&result, 1.0).unwrap()
    );
}
