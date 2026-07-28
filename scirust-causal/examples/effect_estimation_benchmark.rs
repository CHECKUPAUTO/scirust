//! Phase 5C.4 deterministic benchmark: backdoor-adjusted causal effect
//! estimation across a fixed battery of scenarios, each checked against an
//! explicit oracle on both its *identifiability status* and (where a true
//! coefficient exists by construction) its *estimate*.
//!
//! # What this is, and is not
//!
//! This program runs [`scirust_causal::estimate_effect_from_dag`] and
//! [`scirust_causal::estimate_effect_from_cpdag`] over synthetic data whose
//! true generating coefficients are known by construction, and checks that
//! what comes back matches what the known structure predicts. Several
//! scenarios exist specifically to demonstrate where the method *fails* —
//! most pointedly `latent_confounding`, which reports a confidently wrong
//! number because the causal-sufficiency assumption is violated in a way no
//! amount of data can reveal. That row is printed with its bias made
//! explicit, not hidden.
//!
//! An estimate here is a causal effect only under the assumptions listed on
//! its certificate: correct graph, no latent confounding, positivity, and
//! linearity. None are checked, and none are checkable from the data alone.
//!
//! # Reproducibility contract
//!
//! Every scenario's data is generated from a fixed [`SplitMix64`] seed with
//! no wall-clock, hostname, thread-count, or other non-deterministic input.
//! All "scientific" content is printed to **stdout** in a fixed field order;
//! this program prints nothing else to stdout. Running it twice and hashing
//! (SHA-256) each run's captured stdout must produce byte-identical output —
//! verified as part of Phase 5C.4's validation, with the resulting hash
//! recorded in the PR description and the Program 5 tracker document.
//!
//! On any oracle mismatch this program prints a diagnostic to **stderr** and
//! exits with a non-zero status.

use scirust_causal::{
    AdjustmentStrategy, CausalDataset, CausalVariable, ConditionalIndependenceConfig,
    ConditionalIndependenceMethod, EffectEstimate, EffectEstimationConfig, Environment,
    EquivalenceClassConfig, EquivalenceClassDiscovery, IdentifiabilityStatus,
    PartialCorrelationTest, PcStable, VariableKind, VariableRole, estimate_effect_from_cpdag,
    estimate_effect_from_dag,
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

fn fisher_z_test() -> PartialCorrelationTest {
    PartialCorrelationTest::new(
        ConditionalIndependenceConfig::new(
            0.05,
            ConditionalIndependenceMethod::GaussianPartialCorrelation { fisher_z: true },
        )
        .unwrap(),
    )
}

/// `U(0) -> X(1)`, `U(0) -> Y(2)`, `X(1) -> Y(2)`. True effect of X on Y is
/// `b`; U confounds with strengths `a` (into X) and `c` (into Y).
fn confounded_columns(seed: u64, n: usize, a: f64, b: f64, c: f64) -> Vec<Vec<f64>> {
    let mut rng = SplitMix64::new(seed);
    let (mut u, mut x, mut y) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n
    {
        let ui = noise(&mut rng);
        let xi = a * ui + 0.3 * noise(&mut rng);
        let yi = b * xi + c * ui + 0.3 * noise(&mut rng);
        u.push(ui);
        x.push(xi);
        y.push(yi);
    }
    vec![u, x, y]
}

fn report(scenario: &str, result: &EffectEstimate) {
    let estimate = result
        .estimate
        .map_or_else(|| "none".to_string(), |v| format!("{v:.12e}"));
    let standard_error = result
        .standard_error
        .map_or_else(|| "none".to_string(), |v| format!("{v:.12e}"));
    let adjustment = result
        .adjustment_set
        .as_ref()
        .map_or_else(|| "none".to_string(), |z| format!("{z:?}"));
    println!(
        "scenario={scenario} status={:?} estimate={estimate} standard_error={standard_error} \
         adjustment_set={adjustment} sample_count={} assumptions={} warnings={}",
        result.certificate.status(),
        result.sample_count,
        result.certificate.assumptions_used().len(),
        result.warnings.len(),
    );
}

/// Checks a scenario's status, and — when `truth` is `Some` — that the
/// estimate is within `tolerance_in_standard_errors` of the known generating
/// coefficient.
///
/// The tolerance is deliberately expressed in **standard errors**, not as an
/// absolute number: "the estimate is close to the truth" is only meaningful
/// relative to the estimator's own sampling noise at this sample size, and an
/// absolute bound tighter than a couple of standard errors would fail on
/// ordinary sampling variation rather than on any defect. (An earlier draft
/// of this benchmark used an absolute `0.02`, which is ~1.25 standard errors
/// here, and duly failed on a perfectly correct estimate.)
fn check(
    scenario: &str,
    result: &EffectEstimate,
    expected_status: IdentifiabilityStatus,
    truth: Option<(f64, f64)>,
) {
    expect(
        result.certificate.status() == expected_status,
        format!(
            "scenario {scenario}: expected status {expected_status:?}, got {:?}",
            result.certificate.status()
        ),
    );
    match truth
    {
        Some((expected, tolerance_in_standard_errors)) =>
        {
            let estimate = result.estimate.unwrap_or_else(|| {
                eprintln!("ORACLE FAILURE: scenario {scenario}: expected an estimate, got none");
                std::process::exit(1);
            });
            let standard_error = result.standard_error.unwrap_or(f64::NAN);
            let bound = tolerance_in_standard_errors * standard_error;
            expect(
                (estimate - expected).abs() < bound,
                format!(
                    "scenario {scenario}: estimate {estimate} is {:.3} standard errors from the \
                     true coefficient {expected} (se={standard_error}), exceeding the \
                     {tolerance_in_standard_errors}-standard-error bound",
                    (estimate - expected).abs() / standard_error
                ),
            );
        },
        None =>
        {
            if expected_status != IdentifiabilityStatus::Identifiable
            {
                expect(
                    result.estimate.is_none(),
                    format!(
                        "scenario {scenario}: a non-Identifiable status must carry no estimate"
                    ),
                );
            }
        },
    }
}

fn main() {
    println!("# Phase 5C.4 deterministic backdoor effect-estimation benchmark");
    println!(
        "# fields: scenario status estimate standard_error adjustment_set sample_count \
         assumptions warnings"
    );
    println!(
        "# scope: identification by the backdoor criterion plus linear adjustment only — an \
         estimate is a causal effect only under the assumptions named on its certificate"
    );

    let config = EffectEstimationConfig::new();
    let confounded_dag = dag_from_edges(3, &[(0, 1), (0, 2), (1, 2)]);

    // 1. Confounded, correctly adjusted: recovers the true effect 0.7.
    {
        let dataset = dataset_from_columns(&confounded_columns(3001, 4000, 0.9, 0.7, 0.8));
        let result = estimate_effect_from_dag(
            &dataset,
            &confounded_dag,
            1,
            2,
            &AdjustmentStrategy::CanonicalParents,
            &config,
        )
        .unwrap();
        report("confounded_adjusted", &result);
        check(
            "confounded_adjusted",
            &result,
            IdentifiabilityStatus::Identifiable,
            Some((0.7, 4.0)),
        );
    }

    // 2. The same data with no adjustment: refused, not silently biased.
    {
        let dataset = dataset_from_columns(&confounded_columns(3001, 4000, 0.9, 0.7, 0.8));
        let result = estimate_effect_from_dag(
            &dataset,
            &confounded_dag,
            1,
            2,
            &AdjustmentStrategy::Explicit(vec![]),
            &config,
        )
        .unwrap();
        report("confounded_unadjusted", &result);
        check(
            "confounded_unadjusted",
            &result,
            IdentifiabilityStatus::NotIdentifiable,
            None,
        );
    }

    // 3. No confounding at all: the empty set is already valid.
    {
        let mut rng = SplitMix64::new(3002);
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
            &config,
        )
        .unwrap();
        report("unconfounded", &result);
        check(
            "unconfounded",
            &result,
            IdentifiabilityStatus::Identifiable,
            Some((0.5, 4.0)),
        );
    }

    // 4. A genuinely zero effect, with confounding present: adjustment must
    //    drive the estimate to ~0 rather than reporting the association.
    {
        let dataset = dataset_from_columns(&confounded_columns(3003, 4000, 0.9, 0.0, 0.8));
        let result = estimate_effect_from_dag(
            &dataset,
            &confounded_dag,
            1,
            2,
            &AdjustmentStrategy::CanonicalParents,
            &config,
        )
        .unwrap();
        report("zero_effect", &result);
        check(
            "zero_effect",
            &result,
            IdentifiabilityStatus::Identifiable,
            Some((0.0, 4.0)),
        );
    }

    // 5. A negative effect: sign and magnitude both recovered.
    {
        let dataset = dataset_from_columns(&confounded_columns(3004, 4000, 0.9, -0.6, 0.8));
        let result = estimate_effect_from_dag(
            &dataset,
            &confounded_dag,
            1,
            2,
            &AdjustmentStrategy::CanonicalParents,
            &config,
        )
        .unwrap();
        report("negative_effect", &result);
        check(
            "negative_effect",
            &result,
            IdentifiabilityStatus::Identifiable,
            Some((-0.6, 4.0)),
        );
    }

    // 6. Mediator over-control: rejected by backdoor condition 1.
    {
        let mut rng = SplitMix64::new(3005);
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
        let chain = dag_from_edges(3, &[(0, 1), (1, 2)]);

        let over = estimate_effect_from_dag(
            &dataset,
            &chain,
            0,
            2,
            &AdjustmentStrategy::Explicit(vec![1]),
            &config,
        )
        .unwrap();
        report("mediator_overcontrol", &over);
        check(
            "mediator_overcontrol",
            &over,
            IdentifiabilityStatus::NotIdentifiable,
            None,
        );

        let total = estimate_effect_from_dag(
            &dataset,
            &chain,
            0,
            2,
            &AdjustmentStrategy::CanonicalParents,
            &config,
        )
        .unwrap();
        report("mediator_total_effect", &total);
        check(
            "mediator_total_effect",
            &total,
            IdentifiabilityStatus::Identifiable,
            Some((0.64, 4.0)),
        );
    }

    // 7. M-structure: the empty set is valid, conditioning on the collider
    //    is not.
    {
        let mut rng = SplitMix64::new(3006);
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
        let m_dag = dag_from_edges(5, &[(0, 2), (1, 2), (0, 3), (1, 4)]);

        let valid = estimate_effect_from_dag(
            &dataset,
            &m_dag,
            3,
            4,
            &AdjustmentStrategy::Explicit(vec![]),
            &config,
        )
        .unwrap();
        report("m_structure_unadjusted", &valid);
        check(
            "m_structure_unadjusted",
            &valid,
            IdentifiabilityStatus::Identifiable,
            Some((0.0, 4.0)),
        );

        let collider = estimate_effect_from_dag(
            &dataset,
            &m_dag,
            3,
            4,
            &AdjustmentStrategy::Explicit(vec![2]),
            &config,
        )
        .unwrap();
        report("m_structure_collider_adjusted", &collider);
        check(
            "m_structure_collider_adjusted",
            &collider,
            IdentifiabilityStatus::NotIdentifiable,
            None,
        );
    }

    // 8. THE NEGATIVE RESULT. A latent confounder is omitted from both the
    //    data and the graph. The backdoor criterion is satisfied for the
    //    graph it is given, so an estimate is certified Identifiable — and it
    //    is badly wrong. Nothing available to the method can detect this.
    {
        let mut rng = SplitMix64::new(3007);
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
        let dataset = dataset_from_columns(&[x, y]); // U absent by construction
        let result = estimate_effect_from_dag(
            &dataset,
            &dag_from_edges(2, &[(0, 1)]),
            0,
            1,
            &AdjustmentStrategy::CanonicalParents,
            &config,
        )
        .unwrap();
        report("latent_confounding", &result);

        let estimate = result.estimate.unwrap();
        let standard_error = result.standard_error.unwrap();
        println!(
            "# latent_confounding: true effect=7.000000000000e-1 reported={estimate:.12e} \
             bias={:.12e} bias_in_standard_errors={:.6e} \
             -- certified Identifiable and wrong; causal sufficiency is assumed, never checked",
            estimate - 0.7,
            (estimate - 0.7).abs() / standard_error,
        );
        check(
            "latent_confounding",
            &result,
            IdentifiabilityStatus::Identifiable,
            None,
        );
        expect(
            estimate > 1.2,
            format!(
                "scenario latent_confounding: expected a badly inflated estimate, got {estimate}"
            ),
        );
        expect(
            (estimate - 0.7).abs() > 10.0 * standard_error,
            "scenario latent_confounding: the bias should dwarf the reported uncertainty"
                .to_string(),
        );
    }

    // 9. CPDAG with an unoriented edge at the treatment: abstain.
    {
        let mut rng = SplitMix64::new(3008);
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
        let result = estimate_effect_from_cpdag(&dataset, &cpdag, 0, 2, &config).unwrap();
        report("cpdag_chain_ambiguous", &result);
        check(
            "cpdag_chain_ambiguous",
            &result,
            IdentifiabilityStatus::EquivalenceClassOnly,
            None,
        );
    }

    // 10. CPDAG fully oriented at the treatment: estimate is identified
    //     across the whole equivalence class.
    {
        let mut rng = SplitMix64::new(3009);
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
        let result = estimate_effect_from_cpdag(&dataset, &cpdag, 0, 2, &config).unwrap();
        report("cpdag_collider_oriented", &result);
        check(
            "cpdag_collider_oriented",
            &result,
            IdentifiabilityStatus::Identifiable,
            Some((0.8, 4.0)),
        );
    }

    // 11. No residual degrees of freedom: an honest non-answer.
    {
        let dataset = dataset_from_columns(&[
            vec![0.0, 1.0, 2.0],
            vec![0.5, 2.5, 1.5],
            vec![1.0, 3.0, 0.5],
        ]);
        let result = estimate_effect_from_dag(
            &dataset,
            &confounded_dag,
            1,
            2,
            &AdjustmentStrategy::CanonicalParents,
            &config,
        )
        .unwrap();
        report("exhausted_degrees_of_freedom", &result);
        check(
            "exhausted_degrees_of_freedom",
            &result,
            IdentifiabilityStatus::Inconclusive,
            None,
        );
    }

    println!("# all oracle checks passed");
}
