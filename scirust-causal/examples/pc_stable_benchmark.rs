//! Phase 5C.3 deterministic benchmark: PC-Stable equivalence-class discovery
//! across a fixed battery of scenario families, each checked against an
//! explicit, hand-derived oracle CPDAG.
//!
//! # What this is, and is not
//!
//! This program runs [`scirust_causal::PcStable`] over synthetic data whose
//! true generating structure is known by construction, and checks that the
//! discovered [`scirust_causal::Cpdag`] (which edges are directed, which are
//! left undirected, and which pairs are absent entirely) matches what that
//! known structure predicts under the standard causal-sufficiency,
//! acyclicity, and faithfulness assumptions — see
//! `scirust_causal::equivalence_class`'s module docs for the exact scientific
//! scope and honesty caveats. This program discovers a Markov equivalence
//! class; it does **not** estimate a causal effect, does **not** construct a
//! PAG (latent-confounding-robust discovery is out of scope), and several
//! scenarios below exist specifically to demonstrate where this method's
//! assumptions fail, not to hide that fact.
//!
//! # Reproducibility contract
//!
//! Every scenario's data is generated from a fixed [`SplitMix64`] seed with
//! no wall-clock, hostname, thread-count, or other non-deterministic input.
//! All "scientific" content is printed to **stdout** in a fixed field order;
//! this program prints nothing else to stdout. Running it twice and hashing
//! (SHA-256) each run's captured stdout must produce byte-identical output —
//! verified as part of Phase 5C.3's validation, with the resulting hash
//! recorded in the PR description and the Program 5 tracker document (this
//! program does not print its own hash, mirroring the
//! `conditional_independence_benchmark` and `scirust-srcc-bench::industrial_protocol_demo`
//! convention).
//!
//! On any oracle mismatch this program prints a diagnostic to **stderr** and
//! exits with a non-zero status.

use scirust_causal::{
    CausalDataset, CausalVariable, ConditionalIndependenceConfig, ConditionalIndependenceMethod,
    Environment, EquivalenceClassConfig, EquivalenceClassDiscovery, PartialCorrelationTest,
    PcStable, RobustCalibration, VariableKind, VariableRole,
};
use scirust_multivariate::RobustScatterConfig;
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

fn expect(condition: bool, description: String) {
    if !condition
    {
        eprintln!("ORACLE FAILURE: {description}");
        std::process::exit(1);
    }
}

/// One (dataset, method, config) scenario plus its exact expected CPDAG.
struct Scenario {
    name: &'static str,
    dataset: CausalDataset,
    method: ConditionalIndependenceMethod,
    max_conditioning_set_size: Option<usize>,
    expected_directed: Vec<(usize, usize)>,
    expected_undirected: Vec<(usize, usize)>,
    /// If `true`, at least one warning is expected (checked instead of a
    /// fixed empty-warnings expectation).
    expect_warnings: bool,
}

fn fisher_z() -> ConditionalIndependenceMethod {
    ConditionalIndependenceMethod::GaussianPartialCorrelation { fisher_z: true }
}

fn robust_no_p_value() -> ConditionalIndependenceMethod {
    ConditionalIndependenceMethod::RobustPartialCorrelation {
        scatter: RobustScatterConfig::default(),
        calibration: RobustCalibration::NoPValue,
    }
}

/// `0 -> 1 -> 2`.
fn chain_columns(seed: u64, n: usize) -> Vec<Vec<f64>> {
    let mut rng = SplitMix64::new(seed);
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
    vec![v0, v1, v2]
}

/// `0 <- 1 -> 2` — Markov-equivalent to the chain above.
fn fork_columns(seed: u64, n: usize) -> Vec<Vec<f64>> {
    let mut rng = SplitMix64::new(seed);
    let (mut v0, mut v1, mut v2) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n
    {
        let b = noise(&mut rng);
        let a = 0.8 * b + noise(&mut rng);
        let c = 0.8 * b + noise(&mut rng);
        v0.push(a);
        v1.push(b);
        v2.push(c);
    }
    vec![v0, v1, v2]
}

/// `0 -> 2 <- 1` (a pure collider).
fn collider_columns(seed: u64, n: usize) -> Vec<Vec<f64>> {
    let mut rng = SplitMix64::new(seed);
    let (mut v0, mut v1, mut v2) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n
    {
        let a = noise(&mut rng);
        let b = noise(&mut rng);
        let c = 0.8 * a + 0.8 * b + noise(&mut rng);
        v0.push(a);
        v1.push(b);
        v2.push(c);
    }
    vec![v0, v1, v2]
}

/// `0 -> 1 <- 2`, `1 -> 3`: forces Meek's rule 1 to complete `1 -> 3` after
/// v-structure detection alone leaves it undirected — see `pc_stable.rs`'s
/// hand-derivation.
fn collider_with_chain_tail_columns(seed: u64, n: usize) -> Vec<Vec<f64>> {
    let mut rng = SplitMix64::new(seed);
    let (mut v0, mut v1, mut v2, mut v3) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n
    {
        let a = noise(&mut rng);
        let c = noise(&mut rng);
        let b = 0.8 * a + 0.8 * c + noise(&mut rng);
        let d = 0.8 * b + noise(&mut rng);
        v0.push(a);
        v1.push(b);
        v2.push(c);
        v3.push(d);
    }
    vec![v0, v1, v2, v3]
}

/// `0 -> 2 <- 1`, `2 -> 3 <- 4`: two colliders chained through node 2 — see
/// `pc_stable.rs`'s hand-derivation; resolves fully via v-structures alone.
fn two_colliders_columns(seed: u64, n: usize) -> Vec<Vec<f64>> {
    let mut rng = SplitMix64::new(seed);
    let (mut v0, mut v1, mut v2, mut v3, mut v4) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n
    {
        let a = noise(&mut rng);
        let b = noise(&mut rng);
        let e = noise(&mut rng);
        let c = 0.8 * a + 0.8 * b + noise(&mut rng);
        let d = 0.8 * c + 0.8 * e + noise(&mut rng);
        v0.push(a);
        v1.push(b);
        v2.push(c);
        v3.push(d);
        v4.push(e);
    }
    vec![v0, v1, v2, v3, v4]
}

/// `U` (never placed in the dataset) confounds observed `X`(0)/`Y`(1); `Z`(2)
/// is genuinely unrelated. Demonstrates the causal-sufficiency blind spot:
/// the confounded pair looks exactly like an ambiguous-direction direct edge.
fn latent_confounding_columns(seed: u64, n: usize) -> Vec<Vec<f64>> {
    let mut rng = SplitMix64::new(seed);
    let (mut x, mut y, mut z) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for _ in 0..n
    {
        let u = noise(&mut rng);
        x.push(0.8 * u + noise(&mut rng));
        y.push(0.8 * u + noise(&mut rng));
        z.push(noise(&mut rng));
    }
    vec![x, y, z]
}

/// Four mutually independent variables: with `RobustCalibration::NoPValue`
/// (always `Inconclusive`, never `IndependentWithinThreshold`), no edge can
/// ever be removed — demonstrating the conservative-on-`Inconclusive` design
/// invariant even where the true structure has no dependence at all.
fn four_independent_columns(seed: u64, n: usize) -> Vec<Vec<f64>> {
    let mut rng = SplitMix64::new(seed);
    (0..4)
        .map(|_| (0..n).map(|_| noise(&mut rng)).collect())
        .collect()
}

/// Four variables, each a function of every earlier one: a fully connected
/// true skeleton with no conditional independence anywhere, forcing the
/// search to try every conditioning-set size for every pair.
fn fully_connected_four_variable_columns(seed: u64, n: usize) -> Vec<Vec<f64>> {
    let mut rng = SplitMix64::new(seed);
    let mut columns: Vec<Vec<f64>> = (0..4).map(|_| Vec::with_capacity(n)).collect();
    for _ in 0..n
    {
        let v0 = noise(&mut rng);
        let v1 = 0.6 * v0 + noise(&mut rng);
        let v2 = 0.6 * v0 + 0.6 * v1 + noise(&mut rng);
        let v3 = 0.6 * v0 + 0.6 * v1 + 0.6 * v2 + noise(&mut rng);
        columns[0].push(v0);
        columns[1].push(v1);
        columns[2].push(v2);
        columns[3].push(v3);
    }
    columns
}

fn build_scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "chain",
            dataset: dataset_from_columns(&chain_columns(2001, 500)),
            method: fisher_z(),
            max_conditioning_set_size: None,
            expected_directed: vec![],
            expected_undirected: vec![(0, 1), (1, 2)],
            expect_warnings: false,
        },
        Scenario {
            name: "fork",
            dataset: dataset_from_columns(&fork_columns(2002, 500)),
            method: fisher_z(),
            max_conditioning_set_size: None,
            expected_directed: vec![],
            expected_undirected: vec![(0, 1), (1, 2)],
            expect_warnings: false,
        },
        Scenario {
            name: "collider",
            dataset: dataset_from_columns(&collider_columns(2003, 500)),
            method: fisher_z(),
            max_conditioning_set_size: None,
            expected_directed: vec![(0, 2), (1, 2)],
            expected_undirected: vec![],
            expect_warnings: false,
        },
        Scenario {
            name: "collider_with_chain_tail_rule1",
            dataset: dataset_from_columns(&collider_with_chain_tail_columns(2004, 600)),
            method: fisher_z(),
            max_conditioning_set_size: None,
            expected_directed: vec![(0, 1), (1, 3), (2, 1)],
            expected_undirected: vec![],
            expect_warnings: false,
        },
        Scenario {
            name: "two_chained_colliders",
            dataset: dataset_from_columns(&two_colliders_columns(2005, 600)),
            method: fisher_z(),
            max_conditioning_set_size: None,
            expected_directed: vec![(0, 2), (1, 2), (2, 3), (4, 3)],
            expected_undirected: vec![],
            expect_warnings: false,
        },
        Scenario {
            name: "latent_confounding_negative_result",
            dataset: dataset_from_columns(&latent_confounding_columns(2006, 500)),
            method: fisher_z(),
            max_conditioning_set_size: None,
            expected_directed: vec![],
            expected_undirected: vec![(0, 1)],
            expect_warnings: false,
        },
        Scenario {
            name: "chain_unbounded_conditioning",
            dataset: dataset_from_columns(&chain_columns(2001, 500)),
            method: fisher_z(),
            max_conditioning_set_size: None,
            expected_directed: vec![],
            expected_undirected: vec![(0, 1), (1, 2)],
            expect_warnings: false,
        },
        Scenario {
            name: "chain_bounded_conditioning_incorrectly_retains_an_edge",
            dataset: dataset_from_columns(&chain_columns(2001, 500)),
            method: fisher_z(),
            max_conditioning_set_size: Some(0),
            expected_directed: vec![],
            expected_undirected: vec![(0, 1), (0, 2), (1, 2)],
            expect_warnings: false,
        },
        Scenario {
            name: "inconclusive_never_removes_an_edge",
            dataset: dataset_from_columns(&four_independent_columns(2007, 100)),
            method: robust_no_p_value(),
            max_conditioning_set_size: None,
            expected_directed: vec![],
            expected_undirected: vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            expect_warnings: false,
        },
        Scenario {
            name: "small_sample_untestable_candidates_reported_honestly",
            dataset: dataset_from_columns(&fully_connected_four_variable_columns(2008, 3)),
            method: fisher_z(),
            max_conditioning_set_size: None,
            expected_directed: vec![],
            expected_undirected: vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
            expect_warnings: true,
        },
    ]
}

fn main() {
    println!("# Phase 5C.3 deterministic PC-Stable benchmark");
    println!(
        "# fields: scenario directed_edges undirected_edges tests_performed warning_count \
         assumption_count"
    );
    println!(
        "# scope: Markov-equivalence-class discovery only — no effect estimation, no PAG \
         construction (latent-confounding-robust discovery is out of scope)"
    );

    let classical_test =
        PartialCorrelationTest::new(ConditionalIndependenceConfig::new(0.05, fisher_z()).unwrap());
    let robust_no_p_value_test = PartialCorrelationTest::new(
        ConditionalIndependenceConfig::new(0.05, robust_no_p_value()).unwrap(),
    );

    for scenario in build_scenarios()
    {
        let mut config = EquivalenceClassConfig::new();
        if let Some(max) = scenario.max_conditioning_set_size
        {
            config = config.with_max_conditioning_set_size(max);
        }
        let pc = PcStable::new(config);

        let test: &dyn scirust_causal::ConditionalIndependenceTest = match scenario.method
        {
            ConditionalIndependenceMethod::GaussianPartialCorrelation { .. } => &classical_test,
            ConditionalIndependenceMethod::RobustPartialCorrelation {
                calibration: RobustCalibration::NoPValue,
                ..
            } => &robust_no_p_value_test,
            _ => unreachable!("benchmark only uses the two methods constructed above"),
        };

        let result = pc
            .discover(&scenario.dataset, test)
            .unwrap_or_else(|e| panic!("scenario {} failed to complete: {e}", scenario.name));

        println!(
            "scenario={} directed_edges={:?} undirected_edges={:?} tests_performed={} \
             warning_count={} assumption_count={}",
            scenario.name,
            result.cpdag.directed_edges(),
            result.cpdag.undirected_edges(),
            result.tests_performed,
            result.warnings.len(),
            result.assumptions.len(),
        );

        expect(
            result.cpdag.directed_edges() == scenario.expected_directed,
            format!(
                "scenario {}: expected directed edges {:?}, got {:?}",
                scenario.name,
                scenario.expected_directed,
                result.cpdag.directed_edges()
            ),
        );
        expect(
            result.cpdag.undirected_edges() == scenario.expected_undirected,
            format!(
                "scenario {}: expected undirected edges {:?}, got {:?}",
                scenario.name,
                scenario.expected_undirected,
                result.cpdag.undirected_edges()
            ),
        );
        if scenario.expect_warnings
        {
            expect(
                !result.warnings.is_empty(),
                format!(
                    "scenario {}: expected at least one warning, got none",
                    scenario.name
                ),
            );
        }
        else
        {
            expect(
                result.warnings.is_empty(),
                format!(
                    "scenario {}: expected no warnings, got {:?}",
                    scenario.name, result.warnings
                ),
            );
        }
    }

    println!("# all oracle checks passed");
}
