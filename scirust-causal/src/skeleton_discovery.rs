//! PC-Stable skeleton discovery: the order-independent adjacency search
//! (Colombo & Maathuis, *Order-Independent Constraint-Based Causal Structure
//! Learning*, JMLR 2014) that turns a complete graph into a causal skeleton
//! by iteratively testing conditional independence.
//!
//! # The "stable" mechanism
//!
//! Classic PC updates each variable's adjacency set as soon as an edge is
//! removed, so which conditioning sets get *tried* for a later pair depends
//! on the order pairs happen to be visited — different variable orderings can
//! then find different (but all "valid" under the same data) separating
//! sets, producing a different skeleton. PC-Stable fixes this: at the start
//! of each conditioning-set size `ℓ`, every variable's adjacency set is
//! frozen into a snapshot; every test at that level consults only the
//! snapshot, never an adjacency set some other pair's test has already
//! shrunk. Edges found separated during the level are removed only after the
//! whole level finishes, so the graph state that level `ℓ + 1` starts from
//! never depends on the order pairs were visited within level `ℓ`.
//!
//! # Conservative on `Inconclusive` and on benign numerical failure
//!
//! A candidate conditioning set only removes an edge on an explicit
//! [`IndependenceDecision::IndependentWithinThreshold`] verdict.
//! [`IndependenceDecision::Inconclusive`] and [`IndependenceDecision::Dependent`]
//! both leave the edge in place — "no evidence of independence" is never
//! treated as "evidence of independence". A handful of
//! [`CausalError`] variants are expected outcomes at large conditioning-set
//! sizes relative to the sample count (numerical rank deficiency,
//! insufficient samples, zero residual variance, a singular robust scatter, a
//! solver failure) — these are recorded as a warning and treated the same as
//! `Inconclusive` (this one candidate set is unusable; try the next one).
//! Every other [`CausalError`] indicates a malformed request that this
//! module's own index bookkeeping should never produce, and is propagated as
//! a genuine `Err`.

use crate::assumptions::CausalAssumption;
use crate::conditional_independence::{ConditionalIndependenceTest, IndependenceDecision};
use crate::cpdag::Cpdag;
use crate::dataset::CausalDataset;
use crate::error::CausalError;
use std::collections::{BTreeMap, BTreeSet};

/// Canonicalizes an unordered pair as `(min, max)`.
fn canon(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

/// `true` iff `error` reflects an expected, benign numerical limit of the
/// specific candidate conditioning set tried (too large relative to the
/// sample count, a degenerate residual, …) rather than a malformed request.
fn is_benign_untestable(error: &CausalError) -> bool {
    matches!(
        error,
        CausalError::RankDeficientConditioningSet { .. }
            | CausalError::InsufficientSamples { .. }
            | CausalError::ZeroVariance { .. }
            | CausalError::ScatterFailure(_)
            | CausalError::SolverFailure { .. }
    )
}

/// All `k`-combinations of `items` (already sorted, deduplicated), in
/// lexicographic order. `k == 0` yields exactly one combination: the empty
/// set. `k > items.len()` yields none.
fn combinations(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k > items.len()
    {
        return Vec::new();
    }
    if k == 0
    {
        return vec![Vec::new()];
    }
    let mut out = Vec::new();
    let mut current = Vec::with_capacity(k);
    combinations_recurse(items, k, 0, &mut current, &mut out);
    out
}

fn combinations_recurse(
    items: &[usize],
    k: usize,
    start: usize,
    current: &mut Vec<usize>,
    out: &mut Vec<Vec<usize>>,
) {
    if current.len() == k
    {
        out.push(current.clone());
        return;
    }
    // Prune: not enough remaining items to reach length k.
    let remaining_needed = k - current.len();
    if items.len() < start + remaining_needed
    {
        return;
    }
    for i in start..items.len()
    {
        current.push(items[i]);
        combinations_recurse(items, k, i + 1, current, out);
        current.pop();
    }
}

/// One skeleton-discovery outcome.
pub(crate) struct SkeletonResult {
    pub(crate) skeleton: Cpdag,
    /// Canonical `(min, max)` key -> the conditioning set that separated this
    /// (now non-adjacent) pair. Present for every pair the search removed,
    /// absent for every pair still adjacent.
    pub(crate) separating_sets: BTreeMap<(usize, usize), Vec<usize>>,
    pub(crate) tests_performed: usize,
    pub(crate) warnings: Vec<String>,
    /// The union, across every completed test call, of the underlying
    /// [`ConditionalIndependenceTest`]'s own reported assumptions — so the
    /// caller's final assumption list is honest about *all* the statistical
    /// evidence that fed the discovered skeleton, not only the discovery
    /// procedure's own three (acyclicity, causal sufficiency, faithfulness).
    pub(crate) test_assumptions: BTreeSet<CausalAssumption>,
}

/// Runs PC-Stable skeleton discovery over all `n_vars` variables of
/// `dataset`, using `test` as the conditional-independence oracle.
/// `max_conditioning_set_size` bounds the largest conditioning set tried
/// (`None` for unbounded); a pair whose true separating set is larger than
/// this bound is **incorrectly retained as an edge** — a known, standard
/// limitation of any bounded-order constraint-based search, not a defect
/// specific to this implementation.
///
/// # Errors
///
/// Any [`CausalError`] from `test` other than the benign, expected-at-large-ℓ
/// variants listed in the module docs (those are recorded as warnings and
/// treated as an unusable candidate, not propagated).
pub(crate) fn discover_skeleton(
    dataset: &CausalDataset,
    test: &dyn ConditionalIndependenceTest,
    n_vars: usize,
    max_conditioning_set_size: Option<usize>,
) -> Result<SkeletonResult, CausalError> {
    let mut skeleton = Cpdag::complete(n_vars);
    let mut separating_sets = BTreeMap::new();
    let mut tests_performed = 0usize;
    let mut warnings = Vec::new();
    let mut test_assumptions: BTreeSet<CausalAssumption> = BTreeSet::new();

    let mut ell = 0usize;
    loop
    {
        // Frozen adjacency snapshot for this entire level: the "stable" fix.
        let snapshot: Vec<Vec<usize>> = (0..n_vars).map(|v| skeleton.neighbors(v)).collect();

        let mut any_candidate_this_level = false;
        let mut to_remove: Vec<(usize, usize, Vec<usize>)> = Vec::new();

        for x in 0..n_vars
        {
            for &y in &snapshot[x]
            {
                if y <= x
                {
                    continue; // each unordered pair handled once, from its lower index
                }
                if !skeleton.is_adjacent(x, y)
                {
                    continue; // already removed by a smaller conditioning set this same level
                }

                let neighbors_x: Vec<usize> =
                    snapshot[x].iter().copied().filter(|&v| v != y).collect();
                let neighbors_y: Vec<usize> =
                    snapshot[y].iter().copied().filter(|&v| v != x).collect();

                let mut separated = false;
                for candidates in [&neighbors_x, &neighbors_y]
                {
                    if separated
                    {
                        break;
                    }
                    if candidates.len() < ell
                    {
                        continue;
                    }
                    any_candidate_this_level = true;
                    for z in combinations(candidates, ell)
                    {
                        tests_performed += 1;
                        match test.test(dataset, x, y, &z)
                        {
                            Ok(result) =>
                            {
                                test_assumptions.extend(result.assumptions.iter().cloned());
                                if result.decision
                                    == IndependenceDecision::IndependentWithinThreshold
                                {
                                    to_remove.push((x, y, z));
                                    separated = true;
                                    break;
                                }
                                // Dependent or Inconclusive: not evidence of independence
                            },
                            Err(e) if is_benign_untestable(&e) =>
                            {
                                warnings.push(format!(
                                    "pair ({x}, {y}): conditioning set {z:?} untestable ({e}); skipped"
                                ));
                            },
                            Err(e) => return Err(e),
                        }
                    }
                }
            }
        }

        for (x, y, z) in to_remove
        {
            skeleton.remove_edge(x, y);
            separating_sets.insert(canon(x, y), z);
        }

        if !any_candidate_this_level
        {
            break;
        }
        if let Some(max) = max_conditioning_set_size
        {
            if ell >= max
            {
                break;
            }
        }
        ell += 1;
    }

    Ok(SkeletonResult {
        skeleton,
        separating_sets,
        tests_performed,
        warnings,
        test_assumptions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conditional_independence::{
        ConditionalIndependenceConfig, ConditionalIndependenceMethod, PartialCorrelationTest,
    };
    use crate::dataset::CausalDataset;
    use crate::environment::Environment;
    use crate::variable::{CausalVariable, VariableKind, VariableRole};
    use scirust_solvers::Matrix;
    use scirust_stats::SplitMix64;

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

    /// `0 -> 1 -> 2`.
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

    #[test]
    fn chain_skeleton_drops_the_endpoint_edge_with_the_middle_node_as_sepset() {
        let dataset = chain_dataset(10, 400);
        let test = fisher_z_test();
        let result = discover_skeleton(&dataset, &test, 3, None).unwrap();
        assert!(result.skeleton.is_adjacent(0, 1));
        assert!(result.skeleton.is_adjacent(1, 2));
        assert!(!result.skeleton.is_adjacent(0, 2));
        assert_eq!(result.separating_sets.get(&(0, 2)), Some(&vec![1]));
    }

    #[test]
    fn collider_skeleton_keeps_both_edges_and_drops_no_pair() {
        let dataset = collider_dataset(11, 400);
        let test = fisher_z_test();
        let result = discover_skeleton(&dataset, &test, 3, None).unwrap();
        assert!(result.skeleton.is_adjacent(0, 2));
        assert!(result.skeleton.is_adjacent(1, 2));
        assert!(!result.skeleton.is_adjacent(0, 1));
        assert_eq!(result.separating_sets.get(&(0, 1)), Some(&vec![]));
    }

    #[test]
    fn max_conditioning_set_size_zero_only_tests_marginal_independence() {
        // With max=0, the chain's (0,2) pair — separated only by conditioning
        // on {1} — can never be tested with a nonempty Z, so it is (correctly,
        // if incompletely) retained as an edge: a documented limitation, not a
        // bug.
        let dataset = chain_dataset(10, 400);
        let test = fisher_z_test();
        let result = discover_skeleton(&dataset, &test, 3, Some(0)).unwrap();
        assert!(result.skeleton.is_adjacent(0, 2));
    }

    #[test]
    fn combinations_of_size_zero_is_one_empty_set() {
        assert_eq!(combinations(&[1, 2, 3], 0), vec![Vec::<usize>::new()]);
        assert_eq!(combinations(&[], 0), vec![Vec::<usize>::new()]);
    }

    #[test]
    fn combinations_size_exceeds_items_is_empty() {
        assert_eq!(combinations(&[1, 2], 3), Vec::<Vec<usize>>::new());
    }

    #[test]
    fn combinations_are_lexicographic_and_complete() {
        let combos = combinations(&[0, 1, 2, 3], 2);
        assert_eq!(
            combos,
            vec![
                vec![0, 1],
                vec![0, 2],
                vec![0, 3],
                vec![1, 2],
                vec![1, 3],
                vec![2, 3],
            ]
        );
    }

    #[test]
    fn combinations_size_equals_all_items_is_the_one_full_set() {
        assert_eq!(combinations(&[5, 6, 7], 3), vec![vec![5, 6, 7]]);
    }

    #[test]
    fn benign_errors_are_classified_correctly() {
        assert!(is_benign_untestable(
            &CausalError::RankDeficientConditioningSet {
                rank: 1,
                columns: 2
            }
        ));
        assert!(is_benign_untestable(&CausalError::InsufficientSamples {
            required: 5,
            actual: 2
        }));
        assert!(is_benign_untestable(&CausalError::ZeroVariance {
            variable: 0
        }));
        assert!(is_benign_untestable(&CausalError::SolverFailure {
            detail: "x".to_string()
        }));
        assert!(!is_benign_untestable(&CausalError::UnknownVariableIndex {
            index: 0
        }));
        assert!(!is_benign_untestable(&CausalError::SameVariable {
            variable: 0
        }));
    }
}
