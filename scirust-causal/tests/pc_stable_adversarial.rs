//! Adversarial scenarios for [`scirust_causal::PcStable`]: latent
//! confounding, a bounded conditioning-set-size limitation, conservative
//! `Inconclusive` handling, relabeling invariance, determinism, and graceful
//! behavior at small sample sizes.
//!
//! Correctness on the canonical causal motifs and Meek's-rules end-to-end
//! propagation live in `pc_stable.rs`.

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

/// `0 -> 1 -> 2` (chain); `0,2` need conditioning on `{1}` to separate.
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

/// `0 -> 2 <- 1`, `2 -> 3 <- 4` (see `pc_stable.rs` for the hand-verified
/// derivation of its expected CPDAG).
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

/// A latent `U` confounds observed `X` (0) and `Y` (1); `Z` (2) is a genuinely
/// unrelated, independent observed variable. `U` is never placed in the
/// dataset — only `X, Y, Z` are observed.
fn latent_confounding_dataset(seed: u64, n: usize) -> CausalDataset {
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
    dataset_from_columns(&[x, y, z])
}

fn permute_dataset(original: &CausalDataset, perm: &[usize]) -> CausalDataset {
    let block = &original.blocks[0];
    let n_samples = block.n_samples();
    let n = perm.len();
    let mut new_data = vec![0.0; n_samples * n];
    for row in 0..n_samples
    {
        for new_col in 0..n
        {
            let old_col = perm[new_col];
            new_data[row * n + new_col] = block.data()[row * block.n_variables() + old_col];
        }
    }
    let variables: Vec<CausalVariable> = (0..n)
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
    let matrix = Matrix::from_row_major(n_samples, n, new_data);
    let env = Environment::observational("obs").unwrap();
    CausalDataset::single_environment(variables, env, &matrix, "permuted fixture").unwrap()
}

fn relabel_directed(edges: &[(usize, usize)], perm: &[usize]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = edges.iter().map(|&(a, b)| (perm[a], perm[b])).collect();
    out.sort_unstable();
    out
}

fn relabel_undirected(edges: &[(usize, usize)], perm: &[usize]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = edges
        .iter()
        .map(|&(a, b)| {
            let (x, y) = (perm[a], perm[b]);
            if x < y { (x, y) } else { (y, x) }
        })
        .collect();
    out.sort_unstable();
    out
}

// ─── Latent confounding: an undisguised negative result ────────────────

#[test]
fn latent_confounding_produces_an_edge_indistinguishable_from_direct_causation() {
    // The true story is "U confounds X and Y" -- neither causes the other.
    // With U unobserved, no conditioning set among the observed variables
    // can separate X and Y, so PC-Stable (which assumes causal sufficiency)
    // retains an edge {X, Y} it has no way to flag as spurious. This is the
    // documented, unavoidable blind spot of any causal-sufficiency-assuming
    // discovery procedure -- not a defect specific to this implementation.
    let dataset = latent_confounding_dataset(30, 500);
    let test = fisher_z_test();
    let result = PcStable::new(EquivalenceClassConfig::new())
        .discover(&dataset, &test)
        .unwrap();

    assert!(
        result.cpdag.is_adjacent(0, 1),
        "confounded X,Y must appear adjacent: no observed variable separates them"
    );
    // Z is genuinely unrelated: correctly found non-adjacent to both.
    assert!(!result.cpdag.is_adjacent(0, 2));
    assert!(!result.cpdag.is_adjacent(1, 2));
    // No unshielded triple exists to orient {X,Y} one way or the other (Z
    // connects to neither): the misleading edge is left undirected, which
    // in the ordinary CPDAG reading would mean "a direct causal edge whose
    // direction is ambiguous" -- itself already a misreading of what is
    // actually pure confounding.
    assert!(result.cpdag.is_undirected(0, 1));
}

// ─── Bounded conditioning-set-size limitation ───────────────────────────

#[test]
fn bounding_max_conditioning_set_size_incorrectly_retains_an_edge() {
    // The chain 0->1->2 needs Z={1} (size 1) to separate 0 and 2.
    let dataset = chain_dataset(31, 500);
    let test = fisher_z_test();

    let unbounded = PcStable::new(EquivalenceClassConfig::new())
        .discover(&dataset, &test)
        .unwrap();
    assert!(
        !unbounded.cpdag.is_adjacent(0, 2),
        "unbounded search must find the separating set and drop the edge"
    );

    let bounded = PcStable::new(EquivalenceClassConfig::new().with_max_conditioning_set_size(0))
        .discover(&dataset, &test)
        .unwrap();
    assert!(
        bounded.cpdag.is_adjacent(0, 2),
        "a max conditioning-set size of 0 can never test Z={{1}}, so the edge \
         is (documented-ly, incorrectly) retained"
    );
}

// ─── Conservative `Inconclusive` handling ───────────────────────────────

#[test]
fn inconclusive_results_never_remove_an_edge() {
    // RobustCalibration::NoPValue always reports Inconclusive, by design
    // (crate::robust_partial_correlation). Even on data with genuinely no
    // dependence at all, a search that only ever receives "Inconclusive"
    // must retain every edge: no evidence of independence is never treated
    // as evidence of independence.
    let mut rng = SplitMix64::new(32);
    let n = 100;
    let columns: Vec<Vec<f64>> = (0..4)
        .map(|_| (0..n).map(|_| noise(&mut rng)).collect())
        .collect();
    let dataset = dataset_from_columns(&columns);

    let test = PartialCorrelationTest::new(
        ConditionalIndependenceConfig::new(
            0.05,
            ConditionalIndependenceMethod::RobustPartialCorrelation {
                scatter: RobustScatterConfig::default(),
                calibration: RobustCalibration::NoPValue,
            },
        )
        .unwrap(),
    );
    let result = PcStable::new(EquivalenceClassConfig::new())
        .discover(&dataset, &test)
        .unwrap();

    let n_vars = 4;
    let expected_edges = n_vars * (n_vars - 1) / 2;
    assert_eq!(
        result.cpdag.n_edges(),
        expected_edges,
        "every pair must remain adjacent when nothing is ever conclusively independent"
    );
    assert_eq!(result.cpdag.directed_edges().len(), 0);
}

// ─── Relabeling invariance ───────────────────────────────────────────────

#[test]
fn relabeling_variables_does_not_change_the_discovered_structure() {
    let dataset = two_colliders_dataset(24, 600);
    let test = fisher_z_test();
    let pc = PcStable::new(EquivalenceClassConfig::new());
    let original = pc.discover(&dataset, &test).unwrap();

    let perm = [4usize, 3, 2, 1, 0]; // new_variable[i] = old_variable[perm[i]]
    let permuted_dataset = permute_dataset(&dataset, &perm);
    let permuted = pc.discover(&permuted_dataset, &test).unwrap();

    assert_eq!(
        relabel_directed(&permuted.cpdag.directed_edges(), &perm),
        original.cpdag.directed_edges(),
    );
    assert_eq!(
        relabel_undirected(&permuted.cpdag.undirected_edges(), &perm),
        original.cpdag.undirected_edges(),
    );
}

// ─── Determinism ─────────────────────────────────────────────────────────

#[test]
fn discovery_is_deterministic_on_a_five_variable_dataset() {
    let dataset = two_colliders_dataset(24, 600);
    let test = fisher_z_test();
    let pc = PcStable::new(EquivalenceClassConfig::new());
    let first = pc.discover(&dataset, &test).unwrap();
    let second = pc.discover(&dataset, &test).unwrap();
    assert_eq!(first, second);
}

// ─── Small sample sizes: graceful, not silent, not a crash ─────────────

/// Four variables, each a function of *every* earlier one (`v1` of `v0`,
/// `v2` of `v0` and `v1`, `v3` of `v0`, `v1` and `v2`): a fully connected true
/// skeleton with no conditional independence anywhere, so nothing is ever
/// separated and the search is forced to try every conditioning-set size up
/// to the maximum possible (`n_vars - 2 = 2`) for every pair.
fn fully_connected_four_variable_dataset(seed: u64, n: usize) -> CausalDataset {
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
    dataset_from_columns(&columns)
}

#[test]
fn small_sample_count_completes_without_error_and_reports_skipped_tests_honestly() {
    // n=3 samples: a conditioning set of size 2 (the maximum possible on 4
    // variables) needs `2 + 2 = 4` samples, one more than available, and
    // must be reported as a skipped, untestable candidate for every pair --
    // never a hard error, and never a silent, unjustified independence
    // claim that would incorrectly drop a genuinely dependent edge.
    let dataset = fully_connected_four_variable_dataset(33, 3);
    let test = fisher_z_test();
    let result = PcStable::new(EquivalenceClassConfig::new())
        .discover(&dataset, &test)
        .expect("discovery must complete even when some candidates are untestable");

    assert!(
        !result.warnings.is_empty(),
        "expected untestable-candidate warnings"
    );
    for warning in &result.warnings
    {
        assert!(
            warning.contains("untestable"),
            "unexpected warning shape: {warning}"
        );
    }
    // Every pair remains adjacent: no conditioning set was ever conclusively
    // found independent (most were untestable; the rest genuinely dependent).
    assert_eq!(result.cpdag.n_edges(), 6);
}
