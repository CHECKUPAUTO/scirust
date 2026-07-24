//! Causal-motif correctness, Meek's-rules end-to-end propagation, and
//! cross-method compatibility for [`scirust_causal::PcStable`].
//!
//! Adversarial scenarios (latent confounding, bounded conditioning-set-size
//! limitations, conservative `Inconclusive` handling, relabeling invariance)
//! live in `pc_stable_adversarial.rs`.

use scirust_causal::{
    CausalDataset, CausalVariable, ConditionalIndependenceConfig, ConditionalIndependenceMethod,
    Environment, EquivalenceClassConfig, EquivalenceClassDiscovery, PartialCorrelationTest,
    PcStable, RobustCalibration, VariableKind, VariableRole,
};
use scirust_multivariate::RobustScatterConfig;
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

fn fisher_z_test() -> PartialCorrelationTest {
    PartialCorrelationTest::new(
        ConditionalIndependenceConfig::new(
            0.05,
            ConditionalIndependenceMethod::GaussianPartialCorrelation { fisher_z: true },
        )
        .unwrap(),
    )
}

fn robust_permutation_test() -> PartialCorrelationTest {
    PartialCorrelationTest::new(
        ConditionalIndependenceConfig::new(
            0.05,
            ConditionalIndependenceMethod::RobustPartialCorrelation {
                scatter: RobustScatterConfig::default(),
                calibration: RobustCalibration::Permutation {
                    permutations: 199,
                    seed: 7,
                },
            },
        )
        .unwrap(),
    )
}

/// `0 -> 1 -> 2` (chain).
fn chain_dataset(seed: u64, n: usize) -> CausalDataset {
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
    dataset_from_columns(&[v0, v1, v2])
}

/// `0 <- 1 -> 2` (fork) — Markov-equivalent to the chain above: same
/// skeleton, same (empty) v-structure pattern, so PC-Stable cannot and must
/// not distinguish them.
fn fork_dataset(seed: u64, n: usize) -> CausalDataset {
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
    dataset_from_columns(&[v0, v1, v2])
}

/// `0 -> 2 <- 1` (a pure collider; 0, 1 mutually independent exogenous).
fn collider_dataset(seed: u64, n: usize) -> CausalDataset {
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
    dataset_from_columns(&[v0, v1, v2])
}

/// `0 -> 1 <- 2`, `1 -> 3`: a collider (0,2 into 1) with node 1's chain tail.
/// Hand-verified expectation: skeleton {0-1, 1-2, 1-3}; v-structure at 1
/// (sepset(0,2) = {}) orients 0->1, 2->1; the edge {1,3} is *not* itself a
/// v-structure (0-1-3 and 2-1-3 are unshielded triples but 1 is in both
/// sepset(0,3) and sepset(2,3), since the only path to 3 runs through 1), so
/// it starts undirected — Meek's rule 1 must then fire (`0 -> 1`, `1 - 3`,
/// `0`/`3` not adjacent) to complete it to `1 -> 3`. A discovery that stops
/// after v-structures alone (no rule propagation) would leave {1,3}
/// undirected — this is the test that catches that gap.
fn collider_with_chain_tail_dataset(seed: u64, n: usize) -> CausalDataset {
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
    dataset_from_columns(&[v0, v1, v2, v3])
}

/// `0 -> 2 <- 1`, `2 -> 3 <- 4`: two colliders chained through node 2/3.
/// Hand-verified expectation: skeleton {0-2, 1-2, 2-3, 3-4}; both v-structures
/// resolve directly (sepset(0,1) = {}, sepset(2,4) = {}), giving a fully
/// directed result {0->2, 1->2, 2->3, 4->3} with **no** Meek-rule propagation
/// needed (every triple through node 2 or 3 involving the mediator edge has
/// the mediator in the relevant sepset, so those are correctly excluded from
/// v-structure detection).
fn two_colliders_dataset(seed: u64, n: usize) -> CausalDataset {
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
    dataset_from_columns(&[v0, v1, v2, v3, v4])
}

// ─── Causal motifs ───────────────────────────────────────────────────────

#[test]
fn chain_cpdag_is_fully_undirected() {
    let dataset = chain_dataset(20, 500);
    let test = fisher_z_test();
    let result = PcStable::new(EquivalenceClassConfig::new())
        .discover(&dataset, &test)
        .unwrap();
    assert!(result.cpdag.is_undirected(0, 1));
    assert!(result.cpdag.is_undirected(1, 2));
    assert!(!result.cpdag.is_adjacent(0, 2));
    assert_eq!(result.cpdag.directed_edges().len(), 0);
}

#[test]
fn fork_cpdag_is_fully_undirected_and_identical_to_chains() {
    let chain = chain_dataset(20, 500);
    let fork = fork_dataset(21, 500);
    let test = fisher_z_test();
    let pc = PcStable::new(EquivalenceClassConfig::new());
    let chain_result = pc.discover(&chain, &test).unwrap();
    let fork_result = pc.discover(&fork, &test).unwrap();

    assert!(fork_result.cpdag.is_undirected(0, 1));
    assert!(fork_result.cpdag.is_undirected(1, 2));
    assert!(!fork_result.cpdag.is_adjacent(0, 2));

    // The headline Markov-equivalence demonstration: chain and fork produce
    // the *same* CPDAG (same skeleton, same — empty — orientation), because
    // no conditional-independence test can distinguish them.
    assert_eq!(
        chain_result.cpdag.directed_edges(),
        fork_result.cpdag.directed_edges()
    );
    assert_eq!(
        chain_result.cpdag.undirected_edges(),
        fork_result.cpdag.undirected_edges()
    );
}

#[test]
fn collider_cpdag_is_fully_and_correctly_oriented() {
    let dataset = collider_dataset(22, 500);
    let test = fisher_z_test();
    let result = PcStable::new(EquivalenceClassConfig::new())
        .discover(&dataset, &test)
        .unwrap();
    assert!(result.cpdag.is_directed(0, 2));
    assert!(result.cpdag.is_directed(1, 2));
    assert!(!result.cpdag.is_adjacent(0, 1));
    assert_eq!(result.cpdag.undirected_edges().len(), 0);
}

#[test]
fn collider_recovered_identically_via_the_robust_permutation_method() {
    // Confirms discovery is generic over `ConditionalIndependenceTest`, not
    // hardcoded to the classical Fisher-z method.
    let dataset = collider_dataset(22, 500);
    let test = robust_permutation_test();
    let result = PcStable::new(EquivalenceClassConfig::new())
        .discover(&dataset, &test)
        .unwrap();
    assert!(result.cpdag.is_directed(0, 2));
    assert!(result.cpdag.is_directed(1, 2));
    assert!(!result.cpdag.is_adjacent(0, 1));
}

// ─── Meek's-rules end-to-end propagation ────────────────────────────────

#[test]
fn collider_with_chain_tail_is_fully_oriented_via_rule_1_propagation() {
    let dataset = collider_with_chain_tail_dataset(23, 600);
    let test = fisher_z_test();
    let result = PcStable::new(EquivalenceClassConfig::new())
        .discover(&dataset, &test)
        .unwrap();

    assert!(result.cpdag.is_directed(0, 1), "v-structure: 0 -> 1");
    assert!(result.cpdag.is_directed(2, 1), "v-structure: 2 -> 1");
    assert!(
        result.cpdag.is_directed(1, 3),
        "rule 1 must propagate 0->1, 1-3, 0/3 non-adjacent into 1->3"
    );
    assert_eq!(
        result.cpdag.undirected_edges().len(),
        0,
        "every edge should end up compelled in this structure"
    );
}

#[test]
fn two_chained_colliders_are_fully_oriented_by_v_structures_alone() {
    let dataset = two_colliders_dataset(24, 600);
    let test = fisher_z_test();
    let result = PcStable::new(EquivalenceClassConfig::new())
        .discover(&dataset, &test)
        .unwrap();

    assert!(result.cpdag.is_directed(0, 2));
    assert!(result.cpdag.is_directed(1, 2));
    assert!(result.cpdag.is_directed(2, 3));
    assert!(result.cpdag.is_directed(4, 3));
    assert_eq!(result.cpdag.undirected_edges().len(), 0);
    assert!(!result.cpdag.is_adjacent(0, 1));
    assert!(!result.cpdag.is_adjacent(2, 4));
    assert!(!result.cpdag.is_adjacent(0, 3));
    assert!(!result.cpdag.is_adjacent(0, 4));
    assert!(!result.cpdag.is_adjacent(1, 3));
    assert!(!result.cpdag.is_adjacent(1, 4));
}

#[test]
fn well_behaved_data_produces_no_warnings() {
    let dataset = two_colliders_dataset(24, 600);
    let test = fisher_z_test();
    let result = PcStable::new(EquivalenceClassConfig::new())
        .discover(&dataset, &test)
        .unwrap();
    assert!(
        result.warnings.is_empty(),
        "unexpected warnings: {:?}",
        result.warnings
    );
}
