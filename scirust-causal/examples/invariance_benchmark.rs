//! Phase 5C.6 deterministic benchmark: Invariant Causal Prediction across a
//! fixed battery of scenarios, each checked against an explicit oracle.
//!
//! # What this demonstrates
//!
//! The headline row is `icp_detects_what_backdoor_certifies`. The same data
//! is handed to two methods. Backdoor adjustment, given a plausible-but-wrong
//! graph, certifies a badly biased effect as `Identifiable` and has no way to
//! notice. ICP, given no graph at all, reports that **no** predictor subset is
//! invariant across the environments — positive evidence that the model is
//! misspecified. That contrast is the reason this phase exists.
//!
//! The other rows exercise the three-way outcome discipline: a true cause
//! recovered, an empty intersection when two variables are interchangeable
//! proxies, and the effect of a bounded subset search.
//!
//! # Reproducibility contract
//!
//! Every scenario is generated from a fixed [`SplitMix64`] seed with no
//! wall-clock, hostname, or thread-count input. All scientific content goes to
//! **stdout** in a fixed field order; nothing else does. Two runs hashed with
//! SHA-256 must be byte-identical.
//!
//! On any oracle mismatch this program prints a diagnostic to **stderr** and
//! exits with a non-zero status.

use scirust_causal::{
    AdjustmentStrategy, CausalDataset, CausalVariable, EffectEstimationConfig, Environment,
    IdentifiabilityStatus, Intervention, InterventionKind, InvarianceConfig,
    InvariantPredictionOutcome, InvariantPredictionResult, SampleBlock, VariableKind, VariableRole,
    estimate_effect_from_dag, invariant_causal_prediction,
};
use scirust_graph::dag::CausalDag;
use scirust_stats::SplitMix64;

fn noise(rng: &mut SplitMix64) -> f64 {
    rng.next_f64() - 0.5
}

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

fn report(scenario: &str, result: &InvariantPredictionResult) {
    println!(
        "scenario={scenario} outcome={:?} causal_predictors={:?} accepted_sets={} \
         sets_tested={} environments={:?} significance_level={:.2} status={:?} warnings={}",
        result.outcome,
        result.causal_predictors,
        result.accepted_sets.len(),
        result.sets_tested,
        result.environments,
        result.significance_level,
        result.certificate.status(),
        result.warnings.len(),
    );
}

/// `v0 -> v3` is the only causal edge; `v1` is untouched noise, `v2` is a
/// child of the target. The second environment **rescales** `v0`. See the
/// crate's `tests/invariance.rs` for why a rescale (rather than a mean shift)
/// and a substantially-noisy child are both required for these environments to
/// discriminate at all.
fn one_true_cause(seed: u64, n: usize) -> CausalDataset {
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
    CausalDataset::new(
        variables(4),
        vec![
            block(
                Environment::observational("baseline").unwrap(),
                &build(1.0, n),
            ),
            block(
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
            ),
        ],
        "benchmark fixture",
    )
    .unwrap()
}

/// A hidden confounder drives both `X`(0) and `Y`(1); the second environment
/// intervenes on `X`. The confounder is never recorded. True effect: `0.7`.
fn hidden_confounder(seed: u64, n: usize) -> CausalDataset {
    let mut rng = SplitMix64::new(seed);
    let (mut x_obs, mut y_obs) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n
    {
        let u = noise(&mut rng);
        let x = 0.9 * u + 0.3 * noise(&mut rng);
        let y = 0.7 * x + 0.8 * u + 0.3 * noise(&mut rng);
        x_obs.push(x);
        y_obs.push(y);
    }
    let (mut x_int, mut y_int) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n
    {
        let u = noise(&mut rng);
        let x = 2.0 * noise(&mut rng);
        let y = 0.7 * x + 0.8 * u + 0.3 * noise(&mut rng);
        x_int.push(x);
        y_int.push(y);
    }
    CausalDataset::new(
        variables(2),
        vec![
            block(
                Environment::observational("baseline").unwrap(),
                &[x_obs, y_obs],
            ),
            block(
                Environment::new(
                    "x_intervened",
                    vec![Intervention::new(0, InterventionKind::Unspecified).unwrap()],
                )
                .unwrap(),
                &[x_int, y_int],
            ),
        ],
        "benchmark fixture",
    )
    .unwrap()
}

/// `v0` and `v1` are near-identical proxies for the same latent signal; the
/// target `v2` is that signal plus noise. Neither proxy is individually
/// required, so the intersection is empty.
fn interchangeable_proxies(seed: u64, n: usize) -> CausalDataset {
    let mut rng = SplitMix64::new(seed);
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
    CausalDataset::new(
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
        "benchmark fixture",
    )
    .unwrap()
}

fn main() {
    println!("# Phase 5C.6 deterministic invariant-causal-prediction benchmark");
    println!(
        "# fields: scenario outcome causal_predictors accepted_sets sets_tested environments \
         significance_level status warnings"
    );
    println!(
        "# scope: which variables are direct causes, from multi-environment data and no graph — \
         a SUBSET of the true parents, never a complete set, and never an effect size"
    );

    let config = InvarianceConfig::new(0.05).unwrap();

    // 1. Recover the single true cause.
    {
        let dataset = one_true_cause(5001, 600);
        let result = invariant_causal_prediction(&dataset, 3, &[0, 1, 2], &config).unwrap();
        report("one_true_cause", &result);
        expect(
            result.outcome == InvariantPredictionOutcome::CausalPredictorsIdentified,
            format!(
                "one_true_cause: expected identification, got {:?}",
                result.outcome
            ),
        );
        expect(
            result.causal_predictors == vec![0],
            format!(
                "one_true_cause: expected [0], got {:?}",
                result.causal_predictors
            ),
        );
    }

    // 2. THE HEADLINE. Backdoor certifies a wrong number; ICP flags the model.
    {
        let dataset = hidden_confounder(5002, 1500);
        let true_effect = 0.7;

        let backdoor = estimate_effect_from_dag(
            &dataset,
            &dag_from_edges(2, &[(0, 1)]),
            0,
            1,
            &AdjustmentStrategy::CanonicalParents,
            &EffectEstimationConfig::new(),
        )
        .unwrap();
        let biased = backdoor.estimate.unwrap();
        let standard_error = backdoor.standard_error.unwrap();
        println!(
            "# icp_detects_what_backdoor_certifies: backdoor_status={:?} \
             backdoor_estimate={biased:.12e} backdoor_standard_error={standard_error:.12e} \
             true_effect={true_effect:.12e} bias_in_standard_errors={:.6e}",
            backdoor.certificate.status(),
            (biased - true_effect).abs() / standard_error,
        );

        let icp = invariant_causal_prediction(&dataset, 1, &[0], &config).unwrap();
        report("icp_detects_what_backdoor_certifies", &icp);

        expect(
            backdoor.certificate.status() == IdentifiabilityStatus::Identifiable,
            "icp_detects_what_backdoor_certifies: backdoor should certify this".to_string(),
        );
        expect(
            biased > 1.2,
            format!(
                "icp_detects_what_backdoor_certifies: expected a biased estimate, got {biased}"
            ),
        );
        expect(
            icp.outcome == InvariantPredictionOutcome::AssumptionsViolated,
            format!(
                "icp_detects_what_backdoor_certifies: ICP should flag misspecification, got {:?}",
                icp.outcome
            ),
        );
        expect(
            icp.certificate.status() == IdentifiabilityStatus::Inconclusive,
            "icp_detects_what_backdoor_certifies: a detected misspecification concludes nothing"
                .to_string(),
        );
    }

    // 3. Interchangeable proxies: subsets survive, intersection is empty.
    {
        let dataset = interchangeable_proxies(5003, 600);
        let result = invariant_causal_prediction(&dataset, 2, &[0, 1], &config).unwrap();
        report("interchangeable_proxies", &result);
        expect(
            result.outcome == InvariantPredictionOutcome::NoPredictorConfirmed,
            format!(
                "interchangeable_proxies: expected an empty intersection, got {:?}",
                result.outcome
            ),
        );
        expect(
            !result.accepted_sets.is_empty(),
            "interchangeable_proxies: subsets should survive; it is their intersection that is \
             empty"
                .to_string(),
        );
    }

    // 4. A bounded search examines strictly fewer subsets.
    {
        let dataset = one_true_cause(5001, 600);
        let bounded = invariant_causal_prediction(
            &dataset,
            3,
            &[0, 1, 2],
            &InvarianceConfig::new(0.05)
                .unwrap()
                .with_max_predictor_set_size(1),
        )
        .unwrap();
        report("bounded_search", &bounded);
        expect(
            bounded.sets_tested < 8,
            format!(
                "bounded_search: expected fewer than 8 subsets, tested {}",
                bounded.sets_tested
            ),
        );
    }

    // 5. A stricter level accepts at least as many subsets.
    {
        let dataset = one_true_cause(5001, 600);
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
        report("level_lenient", &lenient);
        report("level_strict", &strict);
        expect(
            strict.accepted_sets.len() >= lenient.accepted_sets.len(),
            format!(
                "level ordering: a smaller alpha must accept at least as many subsets, {} vs {}",
                strict.accepted_sets.len(),
                lenient.accepted_sets.len()
            ),
        );
    }

    println!("# all oracle checks passed");
}
