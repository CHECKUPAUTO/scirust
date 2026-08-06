//! d-separation on a [`CausalDag`] — the graphical criterion that decides
//! which conditional independencies a DAG entails.
//!
//! # Why the moralization formulation
//!
//! d-separation is usually *stated* as a walk over paths: a path is blocked
//! by `Z` when it contains a chain or fork whose middle node is in `Z`, or a
//! collider whose middle node **and every descendant of it** are outside `Z`.
//! Implementing that literally means enumerating paths and tracking collider
//! state, and the collider-descendant clause is a classic source of subtle,
//! silent bugs.
//!
//! This module instead uses the equivalent characterization of Lauritzen,
//! Dawid, Larsen & Leimer (1990): for **disjoint** sets `X`, `Y`, `Z`,
//!
//! > `X ⊥d Y | Z` in `G` **iff** `X` and `Y` are separated by `Z` in the
//! > *moral* graph of `G` restricted to `An(X ∪ Y ∪ Z)`,
//!
//! where `An(S)` is `S` together with all its ancestors, and moralizing means
//! marrying every pair of parents that share a child and then dropping edge
//! directions. That reduces the whole question to plain undirected
//! reachability, and the collider-descendant clause falls out automatically:
//! a collider is "opened" exactly when it survives into the ancestral set,
//! which happens exactly when it or one of its descendants is in `Z`. No
//! special case is written for it, so none can be written wrong.
//!
//! # Disjointness
//!
//! d-separation is only defined for pairwise-disjoint `X`, `Y`, `Z`.
//! [`is_d_separated`] returns `false` (never separated, the conservative
//! answer) for an overlapping query rather than silently reporting a
//! separation that the definition does not license.

use scirust_graph::dag::CausalDag;
use std::collections::{BTreeSet, VecDeque};

/// `nodes` together with every ancestor of every node in it.
fn ancestors_inclusive(dag: &CausalDag, nodes: &[usize]) -> BTreeSet<usize> {
    let mut seen: BTreeSet<usize> = BTreeSet::new();
    let mut queue: VecDeque<usize> = VecDeque::new();
    for &n in nodes
    {
        if seen.insert(n)
        {
            queue.push_back(n);
        }
    }
    while let Some(node) = queue.pop_front()
    {
        for &parent in dag.parents(node)
        {
            if seen.insert(parent)
            {
                queue.push_back(parent);
            }
        }
    }
    seen
}

/// `true` iff `X ⊥d Y | Z` in `dag`.
///
/// Returns `false` — the conservative "not separated" answer — when `x`, `y`,
/// and `z` are not pairwise disjoint, or when any index is out of range for
/// `dag`, since d-separation is undefined for such a query. Callers that need
/// a malformed query to be an error must validate before calling; the
/// adjustment layer (`crate::adjustment`) does exactly that.
#[must_use]
pub(crate) fn is_d_separated(dag: &CausalDag, x: &[usize], y: &[usize], z: &[usize]) -> bool {
    let n = dag.n_nodes();
    if x.iter().chain(y).chain(z).any(|&v| v >= n)
    {
        return false;
    }

    let x_set: BTreeSet<usize> = x.iter().copied().collect();
    let y_set: BTreeSet<usize> = y.iter().copied().collect();
    let z_set: BTreeSet<usize> = z.iter().copied().collect();
    if !x_set.is_disjoint(&y_set) || !x_set.is_disjoint(&z_set) || !y_set.is_disjoint(&z_set)
    {
        return false;
    }
    if x_set.is_empty() || y_set.is_empty()
    {
        // Nothing to connect: vacuously separated.
        return true;
    }

    // 1. Ancestral set of X ∪ Y ∪ Z.
    let mut seeds: Vec<usize> = x.to_vec();
    seeds.extend_from_slice(y);
    seeds.extend_from_slice(z);
    let ancestral = ancestors_inclusive(dag, &seeds);

    // 2 & 3. Moral graph of the ancestral subgraph, as undirected adjacency.
    //        `An(S)` is ancestrally closed, so every parent of a node in it is
    //        also in it; the `contains` guards below are belt-and-braces.
    let mut adjacency: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for &node in &ancestral
    {
        // Directed edges, direction dropped.
        for &child in dag.children(node)
        {
            if ancestral.contains(&child)
            {
                adjacency[node].insert(child);
                adjacency[child].insert(node);
            }
        }
        // Marry the parents of this node (they share `node` as a child).
        let parents: Vec<usize> = dag
            .parents(node)
            .iter()
            .copied()
            .filter(|p| ancestral.contains(p))
            .collect();
        for i in 0..parents.len()
        {
            for &q in parents.iter().skip(i + 1)
            {
                let p = parents[i];
                adjacency[p].insert(q);
                adjacency[q].insert(p);
            }
        }
    }

    // 4 & 5. Remove Z, then ask whether any Y node is reachable from X.
    let mut visited = vec![false; n];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for &start in &x_set
    {
        if !ancestral.contains(&start)
        {
            continue;
        }
        visited[start] = true;
        queue.push_back(start);
    }
    while let Some(node) = queue.pop_front()
    {
        if y_set.contains(&node)
        {
            return false; // reachable ⇒ not separated
        }
        for &next in &adjacency[node]
        {
            if !visited[next] && !z_set.contains(&next)
            {
                visited[next] = true;
                queue.push_back(next);
            }
        }
    }
    true
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

    #[test]
    fn chain_is_blocked_by_the_middle_node_and_open_without_it() {
        // 0 -> 1 -> 2
        let dag = dag_from_edges(3, &[(0, 1), (1, 2)]);
        assert!(!is_d_separated(&dag, &[0], &[2], &[]));
        assert!(is_d_separated(&dag, &[0], &[2], &[1]));
    }

    #[test]
    fn fork_is_blocked_by_the_common_cause_and_open_without_it() {
        // 0 <- 1 -> 2
        let dag = dag_from_edges(3, &[(1, 0), (1, 2)]);
        assert!(!is_d_separated(&dag, &[0], &[2], &[]));
        assert!(is_d_separated(&dag, &[0], &[2], &[1]));
    }

    #[test]
    fn collider_is_open_only_when_conditioned_on() {
        // 0 -> 1 <- 2
        let dag = dag_from_edges(3, &[(0, 1), (2, 1)]);
        assert!(
            is_d_separated(&dag, &[0], &[2], &[]),
            "an unconditioned collider blocks"
        );
        assert!(
            !is_d_separated(&dag, &[0], &[2], &[1]),
            "conditioning on a collider opens the path"
        );
    }

    #[test]
    fn conditioning_on_a_colliders_descendant_also_opens_the_path() {
        // 0 -> 1 <- 2, 1 -> 3. This is the case a naive path-walk that forgets
        // the collider-descendant clause gets wrong; the moralization
        // formulation handles it with no special case (node 1 enters the
        // ancestral set as an ancestor of 3, and moralizing then marries 0-2).
        let dag = dag_from_edges(4, &[(0, 1), (2, 1), (1, 3)]);
        assert!(is_d_separated(&dag, &[0], &[2], &[]));
        assert!(!is_d_separated(&dag, &[0], &[2], &[3]));
    }

    #[test]
    fn conditioning_on_an_unrelated_node_changes_nothing() {
        // 0 -> 1 <- 2, plus an isolated node 3.
        let dag = dag_from_edges(4, &[(0, 1), (2, 1)]);
        assert!(is_d_separated(&dag, &[0], &[2], &[3]));
    }

    #[test]
    fn direct_edge_is_never_separated_by_any_conditioning_set() {
        // 0 -> 1, with 2 and 3 available to condition on.
        let dag = dag_from_edges(4, &[(0, 1), (2, 0), (3, 1)]);
        assert!(!is_d_separated(&dag, &[0], &[1], &[]));
        assert!(!is_d_separated(&dag, &[0], &[1], &[2]));
        assert!(!is_d_separated(&dag, &[0], &[1], &[2, 3]));
    }

    #[test]
    fn set_valued_queries_separate_only_when_every_pair_does() {
        // 0 -> 2 -> 3, and 1 (isolated from the rest).
        let dag = dag_from_edges(4, &[(0, 2), (2, 3)]);
        // {0} vs {3} is blocked by 2 …
        assert!(is_d_separated(&dag, &[0], &[3], &[2]));
        // … and adding the isolated node 1 to either side keeps it separated.
        assert!(is_d_separated(&dag, &[0, 1], &[3], &[2]));
        // But {0,2} vs {3} is not: 2 -> 3 is a direct edge.
        assert!(!is_d_separated(&dag, &[0], &[2, 3], &[]));
    }

    #[test]
    fn overlapping_sets_are_conservatively_reported_not_separated() {
        let dag = dag_from_edges(3, &[(0, 1), (1, 2)]);
        assert!(!is_d_separated(&dag, &[0], &[0], &[]));
        assert!(!is_d_separated(&dag, &[0], &[2], &[0]));
        assert!(!is_d_separated(&dag, &[1], &[2], &[1]));
    }

    #[test]
    fn out_of_range_indices_are_conservatively_reported_not_separated() {
        let dag = dag_from_edges(3, &[(0, 1), (1, 2)]);
        assert!(!is_d_separated(&dag, &[0], &[7], &[1]));
        assert!(!is_d_separated(&dag, &[0], &[2], &[9]));
    }

    #[test]
    fn empty_side_is_vacuously_separated() {
        let dag = dag_from_edges(3, &[(0, 1), (1, 2)]);
        assert!(is_d_separated(&dag, &[], &[2], &[]));
        assert!(is_d_separated(&dag, &[0], &[], &[]));
    }

    #[test]
    fn m_structure_is_separated_marginally_but_not_given_the_middle() {
        // The classic M-structure: 0 -> 2 <- 1, 1 -> 4 ... written as
        // U1=0, U2=1, X=2 collider of U1,U2 ... use the standard 5-node form:
        // 0 -> 2, 1 -> 2, 0 -> 3, 1 -> 4.
        // 3 and 4 are marginally independent, but conditioning on the
        // collider 2 opens the path 3 <- 0 -> 2 <- 1 -> 4.
        let dag = dag_from_edges(5, &[(0, 2), (1, 2), (0, 3), (1, 4)]);
        assert!(is_d_separated(&dag, &[3], &[4], &[]));
        assert!(!is_d_separated(&dag, &[3], &[4], &[2]));
    }
}
