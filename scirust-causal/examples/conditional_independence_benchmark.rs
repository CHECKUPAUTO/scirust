//! Phase 5C.2 deterministic benchmark: classical (Fisher-z), robust
//! (no p-value), and robust+permutation conditional-independence testing
//! across a fixed battery of scenario families, each checked against an
//! explicit oracle expectation.
//!
//! # What this is, and is not
//!
//! This program produces **statistical evidence**, not causal discovery — it
//! runs [`scirust_causal::PartialCorrelationTest`] over synthetic data whose
//! true generating structure is known by construction, and checks that the
//! reported [`scirust_causal::IndependenceDecision`] matches what that known
//! structure predicts. It does **not** run PC-Stable or any causal
//! graph-discovery algorithm — see the crate root and
//! `scirust_causal::conditional_independence` module docs for the exact
//! scientific scope this output stays within.
//!
//! # Reproducibility contract
//!
//! Every scenario's data is generated from a fixed [`SplitMix64`] seed with
//! no wall-clock, hostname, thread-count, or other non-deterministic input.
//! All "scientific" content is printed to **stdout** in a fixed field order;
//! this program prints nothing else to stdout. Running it twice and hashing
//! (SHA-256) each run's captured stdout must produce byte-identical output —
//! verified as part of Phase 5C.2's validation, with the resulting hash
//! recorded in the PR description and the Program 5 tracker document (this
//! program does not print its own hash, mirroring the
//! `scirust-srcc-bench::industrial_protocol_demo` convention). Wall-clock
//! runtime, if observed at all, is measured externally (e.g. `time cargo run
//! --example ...`) and is never part of this program's stdout.
//!
//! On any oracle mismatch this program prints a diagnostic to **stderr** and
//! exits with a non-zero status.

use scirust_causal::{
    CalibrationMethod, CausalDataset, CausalVariable, ConditionalIndependenceConfig,
    ConditionalIndependenceMethod, ConditionalIndependenceTest, Environment, IndependenceDecision,
    PartialCorrelationTest, RobustCalibration, VariableKind, VariableRole,
};
use scirust_multivariate::RobustScatterConfig;
use scirust_solvers::Matrix;
use scirust_stats::SplitMix64;

fn noise(rng: &mut SplitMix64) -> f64 {
    rng.next_f64() - 0.5
}

fn heavy_tailed_noise(rng: &mut SplitMix64) -> f64 {
    let u = rng.next_f64();
    if u < 0.05
    {
        (rng.next_f64() - 0.5) * 30.0
    }
    else
    {
        rng.next_f64() - 0.5
    }
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

/// One (dataset, query) pair plus an optional oracle on what the classical
/// method's [`IndependenceDecision`] must be — `None` where the true
/// behavior is illustrative rather than a fixed, theoretically-derivable
/// expectation (e.g. under heavy tails, a fixed seed can legitimately land
/// on either side of the threshold; this program does not pretend otherwise).
struct Scenario {
    name: &'static str,
    dataset: CausalDataset,
    x: usize,
    y: usize,
    z: Vec<usize>,
    classical_oracle: Option<IndependenceDecision>,
}

fn expect(condition: bool, description: String) {
    if !condition
    {
        eprintln!("ORACLE FAILURE: {description}");
        std::process::exit(1);
    }
}

fn calibration_fields(calibration: &CalibrationMethod) -> (String, String) {
    match calibration
    {
        CalibrationMethod::Permutation { permutations, seed } =>
        {
            (permutations.to_string(), seed.to_string())
        },
        _ => ("-".to_string(), "-".to_string()),
    }
}

fn report_row(
    scenario_name: &str,
    conditioning_size: usize,
    method_label: &str,
    result: &scirust_causal::ConditionalIndependenceResult,
) {
    let (permutation_count, seed) = calibration_fields(&result.calibration);
    let p_value = result
        .p_value
        .map_or_else(|| "none".to_string(), |p| format!("{p:.12e}"));
    println!(
        "scenario={scenario_name} method={method_label} statistic={:.12e} effect_size={:.12e} \
         p_value={p_value} significance_level={:.2} decision={:?} sample_count={} \
         conditioning_size={conditioning_size} permutation_count={permutation_count} seed={seed} \
         warnings={:?}",
        result.statistic,
        result.effect_size,
        result.significance_level,
        result.decision,
        result.sample_count,
        result.warnings
    );
}

/// Reports either a normal result row or (deliberately, not via a crash) a
/// `scenario=... method=... error=...` row for a method that honestly could
/// not be computed (e.g. a robust fit legitimately singular at a tiny sample
/// size) — a typed [`scirust_causal::CausalError`] here is a valid,
/// reportable scientific outcome, not something to hide or panic on.
fn report_or_error(
    scenario_name: &str,
    conditioning_size: usize,
    method_label: &str,
    result: Result<scirust_causal::ConditionalIndependenceResult, scirust_causal::CausalError>,
) -> Option<scirust_causal::ConditionalIndependenceResult> {
    match result
    {
        Ok(value) =>
        {
            report_row(scenario_name, conditioning_size, method_label, &value);
            Some(value)
        },
        Err(error) =>
        {
            println!("scenario={scenario_name} method={method_label} error={error}");
            None
        },
    }
}

fn chain_like_columns(seed: u64, n: usize, coefficient: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = SplitMix64::new(seed);
    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n
    {
        let xi = noise(&mut rng);
        let zi = coefficient * xi + noise(&mut rng);
        let yi = coefficient * zi + noise(&mut rng);
        x.push(xi);
        z.push(zi);
        y.push(yi);
    }
    (x, z, y)
}

fn fork_columns(seed: u64, n: usize, coefficient: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = SplitMix64::new(seed);
    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n
    {
        let zi = noise(&mut rng);
        let xi = coefficient * zi + noise(&mut rng);
        let yi = coefficient * zi + noise(&mut rng);
        x.push(xi);
        z.push(zi);
        y.push(yi);
    }
    (x, z, y)
}

fn collider_columns(seed: u64, n: usize, coefficient: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = SplitMix64::new(seed);
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for _ in 0..n
    {
        let xi = noise(&mut rng);
        let yi = noise(&mut rng);
        let zi = coefficient * xi + coefficient * yi + noise(&mut rng);
        x.push(xi);
        y.push(yi);
        z.push(zi);
    }
    (x, y, z)
}

fn build_scenarios() -> Vec<Scenario> {
    let mut scenarios = Vec::new();

    // 1. Independent Gaussian-style noise: no relationship at all.
    {
        let mut rng = SplitMix64::new(1001);
        let n = 300;
        let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
        let y: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
        scenarios.push(Scenario {
            name: "independent_gaussian",
            dataset: dataset_from_columns(&[x, y]),
            x: 0,
            y: 1,
            z: vec![],
            classical_oracle: Some(IndependenceDecision::IndependentWithinThreshold),
        });
    }

    // 2. Direct linear dependence.
    {
        let mut rng = SplitMix64::new(1002);
        let n = 300;
        let x: Vec<f64> = (0..n).map(|_| noise(&mut rng)).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&xi| 0.9 * xi + 0.1 * noise(&mut rng))
            .collect();
        scenarios.push(Scenario {
            name: "direct_linear",
            dataset: dataset_from_columns(&[x, y]),
            x: 0,
            y: 1,
            z: vec![],
            classical_oracle: Some(IndependenceDecision::Dependent),
        });
    }

    // 3. Chain X -> Z -> Y: marginally dependent, conditionally independent.
    {
        let (x, z, y) = chain_like_columns(1003, 400, 0.8);
        let dataset = dataset_from_columns(&[x, z, y]);
        scenarios.push(Scenario {
            name: "chain_marginal",
            dataset: dataset.clone(),
            x: 0,
            y: 2,
            z: vec![],
            classical_oracle: Some(IndependenceDecision::Dependent),
        });
        scenarios.push(Scenario {
            name: "chain_conditional",
            dataset,
            x: 0,
            y: 2,
            z: vec![1],
            classical_oracle: Some(IndependenceDecision::IndependentWithinThreshold),
        });
    }

    // 4. Fork X <- Z -> Y: marginally dependent, conditionally independent.
    {
        let (x, z, y) = fork_columns(1004, 400, 0.8);
        let dataset = dataset_from_columns(&[x, z, y]);
        scenarios.push(Scenario {
            name: "fork_marginal",
            dataset: dataset.clone(),
            x: 0,
            y: 2,
            z: vec![],
            classical_oracle: Some(IndependenceDecision::Dependent),
        });
        scenarios.push(Scenario {
            name: "fork_conditional",
            dataset,
            x: 0,
            y: 2,
            z: vec![1],
            classical_oracle: Some(IndependenceDecision::IndependentWithinThreshold),
        });
    }

    // 5. Collider X -> Z <- Y: marginally independent, conditionally
    //    dependent (conditioning on a collider induces spurious association).
    {
        let (x, y, z) = collider_columns(1005, 400, 0.8);
        let dataset = dataset_from_columns(&[x, y, z]);
        scenarios.push(Scenario {
            name: "collider_marginal",
            dataset: dataset.clone(),
            x: 0,
            y: 1,
            z: vec![],
            classical_oracle: Some(IndependenceDecision::IndependentWithinThreshold),
        });
        scenarios.push(Scenario {
            name: "collider_conditional",
            dataset,
            x: 0,
            y: 1,
            z: vec![2],
            classical_oracle: Some(IndependenceDecision::Dependent),
        });
    }

    // 6. Heavy-tailed independent variables: occasional large-magnitude
    //    draws, still no true relationship. Illustrative only — heavy tails
    //    inflate the variance of the Pearson statistic itself, so no fixed
    //    decision is asserted for this one at a single seed.
    {
        let mut rng = SplitMix64::new(1006);
        let n = 300;
        let x: Vec<f64> = (0..n).map(|_| heavy_tailed_noise(&mut rng)).collect();
        let y: Vec<f64> = (0..n).map(|_| heavy_tailed_noise(&mut rng)).collect();
        scenarios.push(Scenario {
            name: "heavy_tailed_independence",
            dataset: dataset_from_columns(&[x, y]),
            x: 0,
            y: 1,
            z: vec![],
            classical_oracle: None,
        });
    }

    // 7. Vertical contamination: a clean linear relationship plus a handful
    //    of y-only outliers unrelated to x. No decision oracle here — the
    //    comparison this scenario exists for (robust vs. classical
    //    effect_size) is checked separately in `main`, not via a fixed
    //    decision.
    {
        let mut rng = SplitMix64::new(100);
        let n_clean = 150;
        let mut x: Vec<f64> = Vec::with_capacity(n_clean + 8);
        let mut y: Vec<f64> = Vec::with_capacity(n_clean + 8);
        for _ in 0..n_clean
        {
            let xi = noise(&mut rng);
            x.push(xi);
            y.push(0.9 * xi + 0.1 * noise(&mut rng));
        }
        for i in 0..8
        {
            x.push(noise(&mut rng));
            y.push(if i % 2 == 0 { 50.0 } else { -50.0 });
        }
        scenarios.push(Scenario {
            name: "vertical_contamination",
            dataset: dataset_from_columns(&[x, y]),
            x: 0,
            y: 1,
            z: vec![],
            classical_oracle: Some(IndependenceDecision::IndependentWithinThreshold),
        });
    }

    // 8. Bad leverage points: independent cloud plus a handful of points
    //    extreme in both coordinates, manufacturing spurious classical
    //    significance.
    {
        let mut rng = SplitMix64::new(101);
        let n_clean = 100;
        let mut x: Vec<f64> = Vec::with_capacity(n_clean + 6);
        let mut y: Vec<f64> = Vec::with_capacity(n_clean + 6);
        for _ in 0..n_clean
        {
            x.push(noise(&mut rng));
            y.push(noise(&mut rng));
        }
        for i in 0..6
        {
            let leverage = 10.0 + i as f64 * 0.3;
            x.push(leverage);
            y.push(leverage);
        }
        scenarios.push(Scenario {
            name: "bad_leverage",
            dataset: dataset_from_columns(&[x, y]),
            x: 0,
            y: 1,
            z: vec![],
            classical_oracle: Some(IndependenceDecision::Dependent),
        });
    }

    // 9. Near-unfaithful chain: a real but tiny-coefficient chain. The
    //    marginal dependence exists but is too weak to reliably detect at
    //    this sample size — the honest response is "not rejected," which is
    //    exactly what makes this an instructive negative result, not a bug.
    {
        let (x, z, y) = chain_like_columns(124, 400, 0.02);
        let dataset = dataset_from_columns(&[x, z, y]);
        scenarios.push(Scenario {
            name: "near_unfaithful_marginal",
            dataset: dataset.clone(),
            x: 0,
            y: 2,
            z: vec![],
            classical_oracle: Some(IndependenceDecision::IndependentWithinThreshold),
        });
        scenarios.push(Scenario {
            name: "near_unfaithful_conditional",
            dataset,
            x: 0,
            y: 2,
            z: vec![1],
            classical_oracle: None,
        });
    }

    // 10. Nonlinear dependence invisible to a linear statistic: Y = X^2 with
    //     X built as exact mirror pairs so Cov(X, X^2) = 0 exactly, not just
    //     in expectation (keeps the outcome independent of seed luck).
    {
        let mut rng = SplitMix64::new(127);
        let n_pairs = 200;
        let mut x = Vec::with_capacity(2 * n_pairs);
        let mut y = Vec::with_capacity(2 * n_pairs);
        for _ in 0..n_pairs
        {
            let magnitude = 1.0 + 3.0 * rng.next_f64();
            x.push(magnitude);
            y.push(magnitude * magnitude + 0.1 * noise(&mut rng));
            x.push(-magnitude);
            y.push(magnitude * magnitude + 0.1 * noise(&mut rng));
        }
        scenarios.push(Scenario {
            name: "nonlinear",
            dataset: dataset_from_columns(&[x, y]),
            x: 0,
            y: 1,
            z: vec![],
            classical_oracle: Some(IndependenceDecision::IndependentWithinThreshold),
        });
    }

    // 11. Small sample right at the Fisher-z degrees-of-freedom boundary:
    //     n=4, |Z|=1 gives residual dof = n - |Z| - 3 = 0, honestly
    //     Inconclusive rather than a fabricated p-value — even though the
    //     statistic itself is perfectly well-defined at this sample size.
    {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 1.0, 4.0, 3.0];
        let z = vec![1.0, 3.0, 2.0, 4.0];
        scenarios.push(Scenario {
            name: "small_sample",
            dataset: dataset_from_columns(&[x, y, z]),
            x: 0,
            y: 1,
            z: vec![2],
            classical_oracle: Some(IndependenceDecision::Inconclusive),
        });
    }

    scenarios
}

fn main() {
    let classical = PartialCorrelationTest::new(
        ConditionalIndependenceConfig::new(
            0.05,
            ConditionalIndependenceMethod::GaussianPartialCorrelation { fisher_z: true },
        )
        .unwrap(),
    );
    let robust_no_p_value = PartialCorrelationTest::new(
        ConditionalIndependenceConfig::new(
            0.05,
            ConditionalIndependenceMethod::RobustPartialCorrelation {
                scatter: RobustScatterConfig::default(),
                calibration: RobustCalibration::NoPValue,
            },
        )
        .unwrap(),
    );
    let robust_permutation = PartialCorrelationTest::new(
        ConditionalIndependenceConfig::new(
            0.05,
            ConditionalIndependenceMethod::RobustPartialCorrelation {
                scatter: RobustScatterConfig::default(),
                calibration: RobustCalibration::Permutation {
                    permutations: 199,
                    seed: 2026,
                },
            },
        )
        .unwrap(),
    );

    println!("# Phase 5C.2 deterministic conditional-independence benchmark");
    println!(
        "# fields: scenario method statistic effect_size p_value significance_level decision \
         sample_count conditioning_size permutation_count seed warnings"
    );
    println!(
        "# scope: statistical conditional-independence evidence only — no causal discovery, no \
         PC-Stable, no CPDAG/PAG construction is performed by this program"
    );

    let scenarios = build_scenarios();
    let mut classical_results = std::collections::BTreeMap::new();
    let mut robust_results = std::collections::BTreeMap::new();

    for scenario in &scenarios
    {
        let classical_result = report_or_error(
            scenario.name,
            scenario.z.len(),
            "classical_fisher_z",
            classical.test(&scenario.dataset, scenario.x, scenario.y, &scenario.z),
        );
        let robust_result = report_or_error(
            scenario.name,
            scenario.z.len(),
            "robust_no_p_value",
            robust_no_p_value.test(&scenario.dataset, scenario.x, scenario.y, &scenario.z),
        );
        report_or_error(
            scenario.name,
            scenario.z.len(),
            "robust_permutation",
            robust_permutation.test(&scenario.dataset, scenario.x, scenario.y, &scenario.z),
        );

        if let (Some(expected), Some(classical_result)) =
            (scenario.classical_oracle, &classical_result)
        {
            expect(
                classical_result.decision == expected,
                format!(
                    "scenario {} expected classical decision {:?}, got {:?} (statistic={})",
                    scenario.name, expected, classical_result.decision, classical_result.statistic
                ),
            );
        }

        if let Some(classical_result) = classical_result
        {
            classical_results.insert(scenario.name, classical_result);
        }
        if let Some(robust_result) = robust_result
        {
            robust_results.insert(scenario.name, robust_result);
        }
    }

    // Comparative oracles: contamination scenarios are about the
    // relationship between the two statistics, not a fixed decision.
    let vertical_classical = &classical_results["vertical_contamination"];
    let vertical_robust = &robust_results["vertical_contamination"];
    expect(
        vertical_robust.effect_size > vertical_classical.effect_size,
        format!(
            "vertical_contamination: expected robust ({}) to preserve more signal than classical \
             ({}) under vertical outliers",
            vertical_robust.effect_size, vertical_classical.effect_size
        ),
    );

    let leverage_classical = &classical_results["bad_leverage"];
    let leverage_robust = &robust_results["bad_leverage"];
    expect(
        leverage_robust.effect_size < leverage_classical.effect_size,
        format!(
            "bad_leverage: expected robust ({}) to be less swayed than classical ({}) by bad \
             leverage points",
            leverage_robust.effect_size, leverage_classical.effect_size
        ),
    );

    println!("# all oracle checks passed");
}
