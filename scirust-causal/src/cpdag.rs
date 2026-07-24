//! [`Cpdag`]: a partially directed acyclic graph — the output shape of
//! constraint-based causal discovery.
//!
//! A CPDAG (completed partially directed acyclic graph) represents a Markov
//! equivalence class: an edge is **directed** when every DAG in the class
//! agrees on its orientation ("compelled"), and **undirected** when the class
//! contains DAGs that orient it both ways ("reversible" — the data cannot
//! distinguish them). See `crate::equivalence_class` for the discovery
//! procedure that produces one, and its docs for what a `Cpdag` does and does
//! not claim about the true causal graph.
//!
//! This module is a plain graph data structure with an invariant: a pair of
//! nodes carries **at most one** edge, which is either directed (in exactly
//! one direction) or undirected, never both and never both directions at
//! once. Fields are private and mutation goes through methods that preserve
//! this invariant; `crate::skeleton_discovery` and `crate::orientation` are
//! the only callers that mutate a `Cpdag` (both `pub(crate)`).

use std::collections::BTreeSet;

/// Canonicalizes an unordered pair as `(min, max)`.
fn canon(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

/// A partially directed graph over a fixed node set `0..n_nodes`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Cpdag {
    n_nodes: usize,
    /// `(from, to)`: an oriented edge `from -> to`.
    directed: BTreeSet<(usize, usize)>,
    /// `(a, b)` with `a < b`: an unoriented edge.
    undirected: BTreeSet<(usize, usize)>,
}

impl Cpdag {
    /// The complete undirected graph over `n_nodes` — the starting point of
    /// PC-Stable skeleton discovery, before any pair has been tested.
    pub(crate) fn complete(n_nodes: usize) -> Self {
        let mut undirected = BTreeSet::new();
        for a in 0..n_nodes
        {
            for b in (a + 1)..n_nodes
            {
                undirected.insert((a, b));
            }
        }
        Self {
            n_nodes,
            directed: BTreeSet::new(),
            undirected,
        }
    }

    /// An edgeless graph over `n_nodes` — used directly by unit tests that
    /// hand-construct a specific partial orientation.
    #[cfg(test)]
    pub(crate) fn empty(n_nodes: usize) -> Self {
        Self {
            n_nodes,
            directed: BTreeSet::new(),
            undirected: BTreeSet::new(),
        }
    }

    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Total edge count (directed plus undirected).
    #[must_use]
    pub fn n_edges(&self) -> usize {
        self.directed.len() + self.undirected.len()
    }

    /// `true` iff `a` and `b` share an edge, directed or not, in either
    /// direction.
    #[must_use]
    pub fn is_adjacent(&self, a: usize, b: usize) -> bool {
        self.directed.contains(&(a, b))
            || self.directed.contains(&(b, a))
            || self.undirected.contains(&canon(a, b))
    }

    /// `true` iff the directed edge `from -> to` is present.
    #[must_use]
    pub fn is_directed(&self, from: usize, to: usize) -> bool {
        self.directed.contains(&(from, to))
    }

    /// `true` iff `a` and `b` share an undirected edge.
    #[must_use]
    pub fn is_undirected(&self, a: usize, b: usize) -> bool {
        self.undirected.contains(&canon(a, b))
    }

    /// Every neighbor of `node` (directed or undirected, either direction),
    /// in ascending order.
    #[must_use]
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        let mut out = BTreeSet::new();
        for &(from, to) in &self.directed
        {
            if from == node
            {
                out.insert(to);
            }
            if to == node
            {
                out.insert(from);
            }
        }
        for &(a, b) in &self.undirected
        {
            if a == node
            {
                out.insert(b);
            }
            if b == node
            {
                out.insert(a);
            }
        }
        out.into_iter().collect()
    }

    /// Removes the (necessarily still-undirected, pre-orientation) edge
    /// `{a, b}`. A no-op if the pair was not an undirected edge.
    pub(crate) fn remove_edge(&mut self, a: usize, b: usize) {
        self.undirected.remove(&canon(a, b));
    }

    /// Orients the undirected edge `{a, b}` as `from -> to` (where
    /// `{from, to} == {a, b}`). Returns `true` iff the edge was undirected
    /// beforehand and is now directed; `false` (a no-op) if it was already
    /// directed (either way) or not an edge at all — orientation is never
    /// silently overwritten.
    pub(crate) fn orient(&mut self, from: usize, to: usize) -> bool {
        let key = canon(from, to);
        if !self.undirected.remove(&key)
        {
            return false;
        }
        self.directed.insert((from, to));
        true
    }

    /// All directed edges `(from, to)`, in ascending order.
    #[must_use]
    pub fn directed_edges(&self) -> Vec<(usize, usize)> {
        self.directed.iter().copied().collect()
    }

    /// All undirected edges `(a, b)` with `a < b`, in ascending order.
    #[must_use]
    pub fn undirected_edges(&self) -> Vec<(usize, usize)> {
        self.undirected.iter().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_graph_has_all_pairs_undirected() {
        let g = Cpdag::complete(4);
        assert_eq!(g.n_edges(), 6); // C(4,2)
        assert!(g.is_adjacent(0, 3));
        assert!(g.is_undirected(0, 3));
        assert!(!g.is_directed(0, 3));
        assert!(!g.is_directed(3, 0));
    }

    #[test]
    fn remove_edge_drops_adjacency() {
        let mut g = Cpdag::complete(3);
        g.remove_edge(0, 1);
        assert!(!g.is_adjacent(0, 1));
        assert!(g.is_adjacent(0, 2));
        assert!(g.is_adjacent(1, 2));
        assert_eq!(g.n_edges(), 2);
    }

    #[test]
    fn orient_converts_undirected_to_directed_and_is_idempotent_safe() {
        let mut g = Cpdag::complete(2);
        assert!(g.orient(0, 1));
        assert!(g.is_directed(0, 1));
        assert!(!g.is_directed(1, 0));
        assert!(!g.is_undirected(0, 1));
        assert!(g.is_adjacent(0, 1));
        // Re-orienting (even consistently) is a no-op: it is already directed.
        assert!(!g.orient(0, 1));
        assert!(!g.orient(1, 0));
        assert!(g.is_directed(0, 1));
        assert!(!g.is_directed(1, 0));
    }

    #[test]
    fn orient_on_non_edge_is_a_safe_no_op() {
        let mut g = Cpdag::empty(3);
        assert!(!g.orient(0, 1));
        assert!(!g.is_adjacent(0, 1));
    }

    #[test]
    fn neighbors_include_both_directed_and_undirected_either_direction() {
        let mut g = Cpdag::empty(4);
        g.directed.insert((0, 1)); // 0 -> 1
        g.undirected.insert(canon(0, 2)); // 0 - 2
        g.undirected.insert(canon(3, 0)); // 3 - 0
        let mut n = g.neighbors(0);
        n.sort_unstable();
        assert_eq!(n, vec![1, 2, 3]);
        assert_eq!(g.neighbors(1), vec![0]);
    }

    #[test]
    fn directed_and_undirected_edge_lists_are_sorted_and_disjoint() {
        let mut g = Cpdag::complete(3);
        g.orient(1, 2);
        assert_eq!(g.directed_edges(), vec![(1, 2)]);
        assert_eq!(g.undirected_edges(), vec![(0, 1), (0, 2)]);
        assert_eq!(g.n_edges(), 3);
    }
}
