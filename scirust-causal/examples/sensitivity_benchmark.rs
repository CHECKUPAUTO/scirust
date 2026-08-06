//! Phase 5C.5 deterministic benchmark: quantitative sensitivity analysis for
//! unmeasured confounding, checked against an explicit oracle on every row.
//!
//! # What this demonstrates
//!
//! The headline row is `latent_confounder_recovered`. It reconstructs Phase
//! 5C.4's failure — an unmeasured confounder producing a certified-but-wrong
//! estimate — then measures how strong that confounder actually was (using a
//! ground-truth dataset a real analyst would not have) and checks that the
//! Cinelli–Hazlett bias formula predicts the realised bias. It does, to within
//! a couple of percent. That is the phase's central claim, made falsifiable.
//!
//! The `robustness_value_alone_misleads` row records the accompanying
//! caution: on that same scenario the robustness value is high, which reads
//! as reassurance, while the confounder that was actually present was nearly
//! that strong. `RV` is a threshold to compare against domain knowledge, not
//! a verdict on its own.
//!
//! # Reproducibility contract
//!
//! Every scenario is generated from a fixed [`SplitMix64`] seed with no
//! wall-clock, hostname, or thread-count input. All scientific content goes to
//! **stdout** in a fixed field order; nothing else does. Two runs hashed with
//! SHA-256 must be byte-identical — verified in Phase 5C.5's validation, with
//! the hash recorded in the PR and the tracker document.
//!
//! On any oracle mismatch this program prints a diagnostic to **stderr** and
//! exits with a non-zero status.

use scirust_causal::{
    AdjustmentStrategy, CausalDataset, CausalVariable, ConfounderScenario, EffectEstimationConfig,
    Environment, RegimeSelection, VariableKind, VariableRole, analyze_sensitivity,
    benchmark_covariate, bound_effect_under_confounder, estimate_effect_from_dag,
};
use scirust_graph::dag::CausalDag;
use scirust_solvers::Matrix;
use scirust_stats::SplitMix64;

fn noise(rng: &mut SplitMix64) -> f64 {
    rng.next_f64() - 0.5
}

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
    CausalDataset::single_environment(variables, env, &matrix, "benchmark fixture").unwrap()
}

fn dag_from_edges(n: usize, edges: &[(usize, usize)]) -> CausalDag {
    let mut dag = CausalDag::new(n);
    for &(u, v) in edges
    {
        dag.add_directed_edge(u, v).unwrap();
    }
    dag
}

fn expect(condition: bool, description: String) {
    if !condition
    {
        eprintln!("ORACLE FAILURE: {description}");
        std::process::exit(1);
    }
}

/// `U -> X`, `U -> Y`, `X -> Y` with true effect `b`. Returns the observed
/// `[X, Y]` and the ground-truth `[U, X, Y]`.
fn confounded_pair(seed: u64, n: usize, b: f64) -> (CausalDataset, CausalDataset) {
    let mut rng = SplitMix64::new(seed);
    let (mut u, mut x, mut y) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n
    {
        let ui = noise(&mut rng);
        let xi = 0.9 * ui + 0.3 * noise(&mut rng);
        let yi = b * xi + 0.8 * ui + 0.3 * noise(&mut rng);
        u.push(ui);
        x.push(xi);
        y.push(yi);
    }
    (
        dataset_from_columns(&[x.clone(), y.clone()]),
        dataset_from_columns(&[u, x, y]),
    )
}

fn main() {
    println!("# Phase 5C.5 deterministic sensitivity-analysis benchmark");
    println!(
        "# fields: scenario estimate standard_error t_statistic partial_r2_treatment_outcome \
         robustness_value q"
    );
    println!(
        "# scope: how strong an unmeasured confounder would have to be to overturn a given \
         estimate — this quantifies a stated assumption, it does not test whether a confounder \
         exists"
    );

    let config = EffectEstimationConfig::new();
    let true_effect = 0.7;

    // ── 1. The headline: recover a known latent confounder's damage ──────
    let (observed, ground_truth) = confounded_pair(4001, 4000, true_effect);
    let biased = estimate_effect_from_dag(
        &observed,
        &dag_from_edges(2, &[(0, 1)]),
        0,
        1,
        &AdjustmentStrategy::CanonicalParents,
        &config,
    )
    .unwrap();
    let analysis = analyze_sensitivity(&biased, 1.0).unwrap();
    println!(
        "scenario=latent_confounder_recovered estimate={:.12e} standard_error={:.12e} \
         t_statistic={:.12e} partial_r2_treatment_outcome={:.12e} robustness_value={:.12e} q={:.2}",
        analysis.estimate,
        analysis.standard_error,
        analysis.t_statistic,
        analysis.partial_r2_treatment_outcome,
        analysis.robustness_value,
        analysis.q,
    );

    let measured = benchmark_covariate(
        &ground_truth,
        1,
        2,
        &[0],
        0,
        &RegimeSelection::ObservationalOnly,
        1e-9,
    )
    .unwrap();
    let predicted = bound_effect_under_confounder(&analysis, &measured).unwrap();
    let actual_bias = analysis.estimate - true_effect;
    let relative_error = (predicted.bias_bound - actual_bias).abs() / actual_bias;
    println!(
        "# latent_confounder_recovered: true_effect={true_effect:.12e} \
         measured_treatment_share={:.12e} measured_outcome_share={:.12e} \
         predicted_bias={:.12e} actual_bias={actual_bias:.12e} relative_error={relative_error:.12e} \
         adjusted_range=[{:.12e}, {:.12e}]",
        measured.treatment_share,
        measured.outcome_share,
        predicted.bias_bound,
        predicted.adjusted_estimate_lower,
        predicted.adjusted_estimate_upper,
    );
    expect(
        relative_error < 0.05,
        format!(
            "latent_confounder_recovered: predicted bias {} should match actual {actual_bias} \
             to within 5%, relative error {relative_error}",
            predicted.bias_bound
        ),
    );
    expect(
        measured.treatment_share > 0.85 && measured.treatment_share < 0.95,
        format!(
            "latent_confounder_recovered: expected the confounder to explain ~90% of treatment \
             variance, measured {}",
            measured.treatment_share
        ),
    );

    // ── 2. The caution that goes with it ─────────────────────────────────
    println!(
        "# robustness_value_alone_misleads: robustness_value={:.12e} \
         measured_confounder_treatment_share={:.12e} \
         -- RV reads as reassuringly high, yet the confounder actually present was nearly that \
         strong; RV is a threshold to compare against domain knowledge, not a verdict",
        analysis.robustness_value, measured.treatment_share,
    );
    expect(
        analysis.robustness_value > 0.9,
        format!(
            "robustness_value_alone_misleads: expected a high RV, got {}",
            analysis.robustness_value
        ),
    );

    // ── 3. A genuinely clean estimate ────────────────────────────────────
    {
        let mut rng = SplitMix64::new(4002);
        let n = 3000;
        let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 0.5 * xi + 0.3 * noise(&mut rng))
            .collect();
        let dataset = dataset_from_columns(&[x, y]);
        let clean = estimate_effect_from_dag(
            &dataset,
            &dag_from_edges(2, &[(0, 1)]),
            0,
            1,
            &AdjustmentStrategy::CanonicalParents,
            &config,
        )
        .unwrap();
        let clean_analysis = analyze_sensitivity(&clean, 1.0).unwrap();
        println!(
            "scenario=unconfounded estimate={:.12e} standard_error={:.12e} t_statistic={:.12e} \
             partial_r2_treatment_outcome={:.12e} robustness_value={:.12e} q={:.2}",
            clean_analysis.estimate,
            clean_analysis.standard_error,
            clean_analysis.t_statistic,
            clean_analysis.partial_r2_treatment_outcome,
            clean_analysis.robustness_value,
            clean_analysis.q,
        );
        let weak = bound_effect_under_confounder(
            &clean_analysis,
            &ConfounderScenario::new(0.05, 0.05).unwrap(),
        )
        .unwrap();
        println!(
            "# unconfounded: weak_confounder_bias={:.12e} sign_survives={}",
            weak.bias_bound, weak.sign_survives
        );
        expect(
            weak.sign_survives,
            "unconfounded: a weak confounder must not flip the sign".to_string(),
        );
    }

    // ── 4. Weak evidence is easy to overturn ─────────────────────────────
    {
        let mut rng = SplitMix64::new(4003);
        let n = 3000;
        let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 0.03 * xi + 0.3 * noise(&mut rng))
            .collect();
        let dataset = dataset_from_columns(&[x, y]);
        let weak_evidence = estimate_effect_from_dag(
            &dataset,
            &dag_from_edges(2, &[(0, 1)]),
            0,
            1,
            &AdjustmentStrategy::CanonicalParents,
            &config,
        )
        .unwrap();
        let weak_analysis = analyze_sensitivity(&weak_evidence, 1.0).unwrap();
        println!(
            "scenario=weak_evidence estimate={:.12e} standard_error={:.12e} t_statistic={:.12e} \
             partial_r2_treatment_outcome={:.12e} robustness_value={:.12e} q={:.2}",
            weak_analysis.estimate,
            weak_analysis.standard_error,
            weak_analysis.t_statistic,
            weak_analysis.partial_r2_treatment_outcome,
            weak_analysis.robustness_value,
            weak_analysis.q,
        );
        expect(
            weak_analysis.robustness_value < analysis.robustness_value,
            format!(
                "weak_evidence: RV ({}) must be below the strong scenario's ({})",
                weak_analysis.robustness_value, analysis.robustness_value
            ),
        );
    }

    // ── 5. RV_q decreases as less needs explaining away ─────────────────
    {
        let mut previous = f64::INFINITY;
        for q in [1.0, 0.75, 0.5, 0.25]
        {
            let partial = analyze_sensitivity(&biased, q).unwrap();
            println!(
                "scenario=robustness_at_q estimate={:.12e} standard_error={:.12e} \
                 t_statistic={:.12e} partial_r2_treatment_outcome={:.12e} \
                 robustness_value={:.12e} q={q:.2}",
                partial.estimate,
                partial.standard_error,
                partial.t_statistic,
                partial.partial_r2_treatment_outcome,
                partial.robustness_value,
            );
            expect(
                partial.robustness_value < previous,
                format!("robustness_at_q: RV must decrease as q decreases (at q={q})"),
            );
            previous = partial.robustness_value;
        }
    }

    println!("# all oracle checks passed");
}
