//! Pearl's **backdoor criterion** and adjustment-set discovery: deciding
//! *whether* a causal effect is identifiable from observational data by
//! covariate adjustment, and *which* covariates to adjust for.
//!
//! # The criterion
//!
//! A set `Z` satisfies the backdoor criterion relative to an ordered pair
//! (treatment `X`, outcome `Y`) in a DAG `G` when:
//!
//! 1. no node in `Z` is a descendant of `X`; and
//! 2. `Z` blocks every path between `X` and `Y` that contains an arrow *into*
//!    `X` (a "backdoor path").
//!
//! When it holds, `P(Y | do(X)) = Σ_z P(Y | X, z) P(z)` — the effect is
//! identified, and (under this crate's additional linearity assumption, see
//! `crate::effect_estimation`) estimable by regression on `X` and `Z`.
//!
//! Condition 2 is checked by a standard reduction rather than by enumerating
//! paths: in the graph `G` with **every edge out of `X` deleted**, the only
//! remaining paths out of `X` are exactly the backdoor paths, so condition 2
//! holds iff `X ⊥d Y | Z` in that mutilated graph. `crate::d_separation`
//! supplies the separation test.
//!
//! # What satisfying the criterion does and does not establish
//!
//! It establishes identifiability **given that `G` is the true causal graph**
//! and that every variable in `Z` is observed and correctly measured. It does
//! **not** establish that `G` is correct, that there is no latent confounder
//! (a confounder absent from `G` is invisible to a criterion evaluated on
//! `G`), that positivity holds, or that the functional form assumed
//! downstream is right. The criterion is a statement about a graph, not a
//! validation of it — see the crate's adversarial tests, which include a
//! latent confounder producing a confidently wrong estimate through a
//! backdoor set that is perfectly valid for the (misspecified) graph it was
//! checked against.

use crate::d_separation::is_d_separated;
use crate::error::CausalError;
use crate::skeleton_discovery::combinations;
use scirust_graph::dag::CausalDag;
use std::collections::BTreeSet;

/// Why a candidate adjustment set fails the backdoor criterion.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BackdoorViolation {
    /// Condition 1: adjusting for a descendant of the treatment can block a
    /// causal path (over-control) or open a collider path, and is never part
    /// of a valid backdoor set.
    ContainsDescendantOfTreatment { variable: usize },
    /// Condition 2: at least one backdoor path from treatment to outcome
    /// remains open after conditioning on the candidate set.
    UnblockedBackdoorPath,
}

/// The outcome of checking a candidate set against the backdoor criterion.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum BackdoorVerdict {
    Satisfied,
    Violated(BackdoorViolation),
}

/// Validates a (treatment, outcome, adjustment) query's indices and
/// disjointness, returning the canonicalized (sorted, deduplicated)
/// adjustment set.
fn validate_query(
    dag: &CausalDag,
    treatment: usize,
    outcome: usize,
    adjustment: &[usize],
) -> Result<Vec<usize>, CausalError> {
    let n = dag.n_nodes();
    if treatment >= n
    {
        return Err(CausalError::UnknownVariableIndex { index: treatment });
    }
    if outcome >= n
    {
        return Err(CausalError::UnknownVariableIndex { index: outcome });
    }
    if treatment == outcome
    {
        return Err(CausalError::SameVariable {
            variable: treatment,
        });
    }
    let mut sorted: Vec<usize> = adjustment.to_vec();
    sorted.sort_unstable();
    for pair in sorted.windows(2)
    {
        if pair[0] == pair[1]
        {
            return Err(CausalError::DuplicateConditioningVariable { variable: pair[0] });
        }
    }
    for &z in &sorted
    {
        if z >= n
        {
            return Err(CausalError::UnknownVariableIndex { index: z });
        }
        if z == treatment || z == outcome
        {
            return Err(CausalError::ConditioningContainsEndpoint { variable: z });
        }
    }
    Ok(sorted)
}

/// `dag` with every edge *out of* `treatment` removed. A subgraph of a DAG is
/// acyclic, so no edge insertion here can be rejected.
fn without_outgoing_edges(dag: &CausalDag, treatment: usize) -> CausalDag {
    let n = dag.n_nodes();
    let mut mutilated = CausalDag::new(n);
    for u in 0..n
    {
        if u == treatment
        {
            continue;
        }
        for &v in dag.children(u)
        {
            mutilated
                .add_directed_edge(u, v)
                .expect("a subgraph of a DAG is acyclic");
        }
    }
    mutilated
}

/// Checks `adjustment` against the backdoor criterion for
/// `treatment → outcome` in `dag`.
///
/// # Errors
///
/// [`CausalError::UnknownVariableIndex`] for an out-of-range index,
/// [`CausalError::SameVariable`] if `treatment == outcome`,
/// [`CausalError::DuplicateConditioningVariable`] for a repeated adjustment
/// variable, and [`CausalError::ConditioningContainsEndpoint`] if the
/// adjustment set contains the treatment or the outcome.
pub fn check_backdoor_criterion(
    dag: &CausalDag,
    treatment: usize,
    outcome: usize,
    adjustment: &[usize],
) -> Result<BackdoorVerdict, CausalError> {
    let adjustment = validate_query(dag, treatment, outcome, adjustment)?;

    // Condition 1: no adjustment variable may be a descendant of the treatment.
    let descendants: BTreeSet<usize> = dag.descendants(treatment).into_iter().collect();
    for &z in &adjustment
    {
        if descendants.contains(&z)
        {
            return Ok(BackdoorVerdict::Violated(
                BackdoorViolation::ContainsDescendantOfTreatment { variable: z },
            ));
        }
    }

    // Condition 2: all backdoor paths blocked ⟺ separated in the graph with
    // the treatment's outgoing edges deleted.
    let mutilated = without_outgoing_edges(dag, treatment);
    if is_d_separated(&mutilated, &[treatment], &[outcome], &adjustment)
    {
        Ok(BackdoorVerdict::Satisfied)
    }
    else
    {
        Ok(BackdoorVerdict::Violated(
            BackdoorViolation::UnblockedBackdoorPath,
        ))
    }
}

/// The parents of `treatment` — the canonical backdoor adjustment set.
///
/// Under causal sufficiency (every parent observed) this always satisfies the
/// criterion: parents of `X` cannot be descendants of `X` in an acyclic
/// graph, and conditioning on all of them blocks every path leaving `X`
/// backwards. The one exception is an `outcome` that is itself a parent of
/// `treatment` — then the query's own direction is contradicted by the graph,
/// and this returns [`CausalError::InvalidContract`] rather than quietly
/// dropping it from the set and returning something that is no longer the
/// parent set.
///
/// # Errors
///
/// As [`check_backdoor_criterion`], plus [`CausalError::InvalidContract`]
/// when `outcome` is a parent of `treatment`.
pub fn canonical_adjustment_set(
    dag: &CausalDag,
    treatment: usize,
    outcome: usize,
) -> Result<Vec<usize>, CausalError> {
    validate_query(dag, treatment, outcome, &[])?;
    let mut parents: Vec<usize> = dag.parents(treatment).to_vec();
    parents.sort_unstable();
    if parents.contains(&outcome)
    {
        return Err(CausalError::InvalidContract {
            detail: "outcome is a parent of treatment; the queried direction contradicts the graph",
        });
    }
    Ok(parents)
}

/// Every **minimal** set satisfying the backdoor criterion, up to
/// `max_size` variables, in increasing size then lexicographic order.
/// Minimal means no proper subset of it is also valid.
///
/// Candidates are drawn from all variables other than the treatment, the
/// outcome, and the treatment's descendants (condition 1 excludes those
/// outright). The search is exponential in the candidate count by nature;
/// `max_size` is the caller's bound on that cost, and a set larger than the
/// bound is simply never found — this returns the minimal sets *within the
/// bound*, never a claim that no larger valid set exists.
///
/// # Errors
///
/// As [`check_backdoor_criterion`].
pub fn find_minimal_adjustment_sets(
    dag: &CausalDag,
    treatment: usize,
    outcome: usize,
    max_size: usize,
) -> Result<Vec<Vec<usize>>, CausalError> {
    validate_query(dag, treatment, outcome, &[])?;

    let descendants: BTreeSet<usize> = dag.descendants(treatment).into_iter().collect();
    let candidates: Vec<usize> = (0..dag.n_nodes())
        .filter(|&v| v != treatment && v != outcome && !descendants.contains(&v))
        .collect();

    let mut minimal: Vec<Vec<usize>> = Vec::new();
    for size in 0..=max_size.min(candidates.len())
    {
        for candidate in combinations(&candidates, size)
        {
            // Enumerating by increasing size means any valid proper subset was
            // already accepted, so a superset of an accepted set is not minimal.
            if minimal
                .iter()
                .any(|accepted| accepted.iter().all(|v| candidate.contains(v)))
            {
                continue;
            }
            if check_backdoor_criterion(dag, treatment, outcome, &candidate)?
                == BackdoorVerdict::Satisfied
            {
                minimal.push(candidate);
            }
        }
    }
    Ok(minimal)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dag_from_edges(n: usize, edges: &[(usize, usize)]) -> CausalDag {
        let mut dag = CausalDag::new(n);
        for &(u, v) in edges
        {
            dag.add_directed_edge(u, v).unwrap();
        }
        dag
    }

    /// Confounder 0 -> treatment 1, 0 -> outcome 2, and 1 -> 2.
    fn confounded() -> CausalDag {
        dag_from_edges(3, &[(0, 1), (0, 2), (1, 2)])
    }

    #[test]
    fn empty_set_fails_when_a_backdoor_path_is_open() {
        let dag = confounded();
        assert_eq!(
            check_backdoor_criterion(&dag, 1, 2, &[]).unwrap(),
            BackdoorVerdict::Violated(BackdoorViolation::UnblockedBackdoorPath)
        );
    }

    #[test]
    fn conditioning_on_the_confounder_satisfies_the_criterion() {
        let dag = confounded();
        assert_eq!(
            check_backdoor_criterion(&dag, 1, 2, &[0]).unwrap(),
            BackdoorVerdict::Satisfied
        );
    }

    #[test]
    fn no_confounding_means_the_empty_set_already_satisfies_it() {
        // 0 -> 1 only.
        let dag = dag_from_edges(2, &[(0, 1)]);
        assert_eq!(
            check_backdoor_criterion(&dag, 0, 1, &[]).unwrap(),
            BackdoorVerdict::Satisfied
        );
    }

    #[test]
    fn a_descendant_of_the_treatment_is_rejected_by_condition_one() {
        // 0 -> 1 -> 2, and the mediator 1 is a descendant of treatment 0.
        let dag = dag_from_edges(3, &[(0, 1), (1, 2)]);
        assert_eq!(
            check_backdoor_criterion(&dag, 0, 2, &[1]).unwrap(),
            BackdoorVerdict::Violated(BackdoorViolation::ContainsDescendantOfTreatment {
                variable: 1
            })
        );
    }

    #[test]
    fn conditioning_on_a_collider_opens_a_path_and_is_rejected() {
        // M-structure: 0 -> 2 <- 1, 0 -> 3(treatment), 1 -> 4(outcome).
        // The empty set is valid; conditioning on the collider 2 is not.
        let dag = dag_from_edges(5, &[(0, 2), (1, 2), (0, 3), (1, 4)]);
        assert_eq!(
            check_backdoor_criterion(&dag, 3, 4, &[]).unwrap(),
            BackdoorVerdict::Satisfied
        );
        assert_eq!(
            check_backdoor_criterion(&dag, 3, 4, &[2]).unwrap(),
            BackdoorVerdict::Violated(BackdoorViolation::UnblockedBackdoorPath)
        );
    }

    #[test]
    fn canonical_set_is_the_treatments_parents_and_is_always_valid() {
        let dag = confounded();
        let z = canonical_adjustment_set(&dag, 1, 2).unwrap();
        assert_eq!(z, vec![0]);
        assert_eq!(
            check_backdoor_criterion(&dag, 1, 2, &z).unwrap(),
            BackdoorVerdict::Satisfied
        );
    }

    #[test]
    fn canonical_set_is_empty_for_an_exogenous_treatment() {
        let dag = dag_from_edges(2, &[(0, 1)]);
        assert_eq!(
            canonical_adjustment_set(&dag, 0, 1).unwrap(),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn canonical_set_rejects_an_outcome_that_is_a_parent_of_the_treatment() {
        // 0 -> 1: asking for the effect of 1 on 0 contradicts the graph.
        let dag = dag_from_edges(2, &[(0, 1)]);
        assert!(matches!(
            canonical_adjustment_set(&dag, 1, 0),
            Err(CausalError::InvalidContract { .. })
        ));
    }

    #[test]
    fn minimal_sets_find_the_single_confounder() {
        let dag = confounded();
        let sets = find_minimal_adjustment_sets(&dag, 1, 2, 3).unwrap();
        assert_eq!(sets, vec![vec![0]]);
    }

    #[test]
    fn minimal_sets_return_the_empty_set_when_nothing_needs_adjusting() {
        // 0 -> 1, with an unrelated isolated node 2.
        let dag = dag_from_edges(3, &[(0, 1)]);
        let sets = find_minimal_adjustment_sets(&dag, 0, 1, 2).unwrap();
        assert_eq!(
            sets,
            vec![Vec::<usize>::new()],
            "the empty set is valid and minimal, so it must be the only one returned"
        );
    }

    #[test]
    fn minimal_sets_exclude_supersets_of_an_already_valid_set() {
        // 0 -> 1(treatment), 0 -> 2(outcome), 1 -> 2, plus an irrelevant
        // extra node 3 that must not appear in any *minimal* set.
        let dag = dag_from_edges(4, &[(0, 1), (0, 2), (1, 2)]);
        let sets = find_minimal_adjustment_sets(&dag, 1, 2, 3).unwrap();
        assert_eq!(sets, vec![vec![0]]);
        assert!(
            !sets.iter().any(|s| s.contains(&3)),
            "an irrelevant variable must not appear in a minimal set"
        );
    }

    #[test]
    fn two_independent_confounders_both_need_adjusting() {
        // 0 -> 2, 0 -> 3, 1 -> 2, 1 -> 3, 2(treatment) -> 3(outcome).
        let dag = dag_from_edges(4, &[(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
        let sets = find_minimal_adjustment_sets(&dag, 2, 3, 3).unwrap();
        assert_eq!(sets, vec![vec![0, 1]]);
    }

    #[test]
    fn max_size_bound_can_miss_a_valid_set_without_claiming_none_exists() {
        let dag = dag_from_edges(4, &[(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);
        // The only valid set has size 2; bounding at 1 finds nothing.
        assert!(
            find_minimal_adjustment_sets(&dag, 2, 3, 1)
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            find_minimal_adjustment_sets(&dag, 2, 3, 2).unwrap(),
            vec![vec![0, 1]]
        );
    }

    #[test]
    fn malformed_queries_are_typed_errors() {
        let dag = confounded();
        assert!(matches!(
            check_backdoor_criterion(&dag, 0, 0, &[]),
            Err(CausalError::SameVariable { variable: 0 })
        ));
        assert!(matches!(
            check_backdoor_criterion(&dag, 0, 9, &[]),
            Err(CausalError::UnknownVariableIndex { index: 9 })
        ));
        assert!(matches!(
            check_backdoor_criterion(&dag, 1, 2, &[1]),
            Err(CausalError::ConditioningContainsEndpoint { variable: 1 })
        ));
        assert!(matches!(
            check_backdoor_criterion(&dag, 1, 2, &[0, 0]),
            Err(CausalError::DuplicateConditioningVariable { variable: 0 })
        ));
    }
}
