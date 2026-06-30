//! Directed acyclic graph (DAG) substrate for causal memory organizations.
//!
//! [`crate::Graph`] is strictly **undirected** — its `add_edge` inserts a
//! symmetric adjacency entry — so it cannot represent a causal influence
//! relation ("`u` causes `v`") or answer the questions a causal memory store
//! needs: ancestry, descendants, topological order, intervention sets. This
//! module adds a small, deterministic directed-acyclic-graph type that
//! inter-operates with the undirected [`crate::Graph`] (via
//! [`CausalDag::to_undirected`]) so existing motif / community algorithms can
//! still run on the projected skeleton.
//!
//! All arithmetic is plain `usize` bookkeeping in fixed order, so a build is
//! bit-for-bit deterministic.

use crate::Graph;
use std::collections::{HashSet, VecDeque};

/// Error returned when a directed edge would close a cycle (breaking the DAG
/// invariant). The offending edge `u -> v` is reported.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleError {
    pub from: usize,
    pub to: usize,
}

/// A directed acyclic graph over `n_nodes` nodes. Edges go `u -> v` ("`u`
/// causes `v`"); the constructor and [`CausalDag::add_directed_edge`] reject any
/// edge that would create a directed cycle, so the graph stays a DAG by
/// construction.
#[derive(Debug, Clone)]
pub struct CausalDag {
    n_nodes: usize,
    /// `children[u]` = nodes `v` such that `u -> v`.
    children: Vec<Vec<usize>>,
    /// `parents[v]` = nodes `u` such that `u -> v`.
    parents: Vec<Vec<usize>>,
    node_labels: Vec<String>,
    edge_labels: Vec<Vec<Option<String>>>,
}

impl CausalDag {
    /// Empty DAG over `n_nodes` nodes.
    pub fn new(n_nodes: usize) -> Self {
        Self {
            n_nodes,
            children: vec![Vec::new(); n_nodes],
            parents: vec![Vec::new(); n_nodes],
            node_labels: vec![String::new(); n_nodes],
            edge_labels: vec![vec![None; n_nodes]; n_nodes],
        }
    }

    /// Number of nodes.
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Number of directed edges.
    pub fn n_edges(&self) -> usize {
        self.children.iter().map(|c| c.len()).sum()
    }

    /// Label a node (for memory-item naming).
    pub fn set_node_label(&mut self, node: usize, label: &str) {
        self.node_labels[node] = label.to_string();
    }

    /// Node label, if any.
    pub fn node_label(&self, node: usize) -> &str {
        &self.node_labels[node]
    }

    /// Add a directed edge `u -> v` ("`u` causes `v`"). Returns
    /// [`CycleError`] if `v` can already reach `u` (the edge would close a
    /// cycle), and is a no-op (returns `Ok`) if the edge already exists or
    /// `u == v` (self-loops are rejected as cycles).
    pub fn add_directed_edge(&mut self, u: usize, v: usize) -> Result<(), CycleError> {
        self.add_directed_edge_labeled(u, v, None)
    }

    /// As [`CausalDag::add_directed_edge`] but also stamps an optional edge
    /// label (e.g. the causal-mechanism name).
    pub fn add_directed_edge_labeled(
        &mut self,
        u: usize,
        v: usize,
        label: Option<&str>,
    ) -> Result<(), CycleError> {
        if u == v || self.can_reach(v, u)
        {
            return Err(CycleError { from: u, to: v });
        }
        if !self.children[u].contains(&v)
        {
            self.children[u].push(v);
            self.parents[v].push(u);
        }
        if let Some(lbl) = label
        {
            self.edge_labels[u][v] = Some(lbl.to_string());
        }
        Ok(())
    }

    /// Directed children of `u` (nodes `u` directly causes).
    pub fn children(&self, u: usize) -> &[usize] {
        &self.children[u]
    }

    /// Directed parents of `v` (nodes that directly cause `v`).
    pub fn parents(&self, v: usize) -> &[usize] {
        &self.parents[v]
    }

    /// Is there a directed path `src -> .. -> dst`?
    pub fn can_reach(&self, src: usize, dst: usize) -> bool {
        if src == dst
        {
            return true;
        }
        let mut visited = vec![false; self.n_nodes];
        let mut queue = VecDeque::new();
        visited[src] = true;
        queue.push_back(src);
        while let Some(node) = queue.pop_front()
        {
            for &child in &self.children[node]
            {
                if child == dst
                {
                    return true;
                }
                if !visited[child]
                {
                    visited[child] = true;
                    queue.push_back(child);
                }
            }
        }
        false
    }

    /// All ancestors of `v` (every `u` with a directed path to `v`), excluding
    /// `v` itself. Deterministic BFS order.
    pub fn ancestors(&self, v: usize) -> Vec<usize> {
        let mut visited = vec![false; self.n_nodes];
        let mut queue = VecDeque::new();
        let mut out = Vec::new();
        visited[v] = true;
        queue.push_back(v);
        while let Some(node) = queue.pop_front()
        {
            for &p in &self.parents[node]
            {
                if !visited[p]
                {
                    visited[p] = true;
                    out.push(p);
                    queue.push_back(p);
                }
            }
        }
        out
    }

    /// All descendants of `u` (every `v` reachable from `u`), excluding `u`
    /// itself. Deterministic BFS order.
    pub fn descendants(&self, u: usize) -> Vec<usize> {
        let mut visited = vec![false; self.n_nodes];
        let mut queue = VecDeque::new();
        let mut out = Vec::new();
        visited[u] = true;
        queue.push_back(u);
        while let Some(node) = queue.pop_front()
        {
            for &c in &self.children[node]
            {
                if !visited[c]
                {
                    visited[c] = true;
                    out.push(c);
                    queue.push_back(c);
                }
            }
        }
        out
    }

    /// Kahn's topological order. Returns `Err` with the residual cycle only if
    /// the invariant was somehow violated (it cannot be, by construction).
    pub fn topo_order(&self) -> Result<Vec<usize>, CycleError> {
        let mut indegree = vec![0usize; self.n_nodes];
        for (v, d) in indegree.iter_mut().enumerate()
        {
            *d = self.parents[v].len();
        }
        let mut queue: VecDeque<usize> = (0..self.n_nodes).filter(|&u| indegree[u] == 0).collect();
        let mut order = Vec::with_capacity(self.n_nodes);
        while let Some(u) = queue.pop_front()
        {
            order.push(u);
            for &v in &self.children[u]
            {
                indegree[v] -= 1;
                if indegree[v] == 0
                {
                    queue.push_back(v);
                }
            }
        }
        if order.len() == self.n_nodes
        {
            Ok(order)
        }
        else
        {
            // Unreachable by construction; surface a best-effort error.
            Err(CycleError { from: 0, to: 0 })
        }
    }

    /// The set of nodes whose value must be fixed when intervening on `u`
    /// (the union of `u`'s ancestors and `u` itself) — the "backdoor
    /// adjustment set" skeleton. Excludes `u`'s descendants (effects of the
    /// intervention).
    pub fn intervention_ancestors(&self, u: usize) -> HashSet<usize> {
        let mut set: HashSet<usize> = self.ancestors(u).into_iter().collect();
        set.insert(u);
        set
    }

    /// Project onto the crate's undirected [`Graph`] (drop edge direction) so
    /// existing motif / community algorithms apply to the causal skeleton.
    pub fn to_undirected(&self) -> Graph {
        let mut g = Graph::new(self.n_nodes);
        for (i, label) in self.node_labels.iter().enumerate()
        {
            if !label.is_empty()
            {
                g.set_node_label(i, label);
            }
        }
        for u in 0..self.n_nodes
        {
            for &v in &self.children[u]
            {
                g.add_edge(u, v);
            }
        }
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_a_cycle_by_construction() {
        // A -> B -> C ; adding C -> A must fail (closes the cycle).
        let mut dag = CausalDag::new(3);
        dag.set_node_label(0, "rain");
        dag.set_node_label(1, "wet");
        dag.set_node_label(2, "slip");
        assert!(dag.add_directed_edge(0, 1).is_ok());
        assert!(dag.add_directed_edge(1, 2).is_ok());
        let err = dag.add_directed_edge(2, 0).unwrap_err();
        assert_eq!(err, CycleError { from: 2, to: 0 });
        // The rejected edge did not land.
        assert_eq!(dag.n_edges(), 2);
    }

    #[test]
    fn rejects_self_loop() {
        let mut dag = CausalDag::new(2);
        assert!(dag.add_directed_edge(0, 0).is_err());
        assert_eq!(dag.n_edges(), 0);
    }

    #[test]
    fn ancestors_descendants_and_reachability() {
        // 0 -> 1 -> 2, 0 -> 2 (diamond-free chain plus a shortcut)
        let mut dag = CausalDag::new(3);
        dag.add_directed_edge(0, 1).unwrap();
        dag.add_directed_edge(1, 2).unwrap();
        dag.add_directed_edge(0, 2).unwrap();
        assert!(dag.can_reach(0, 2));
        assert!(!dag.can_reach(2, 0));
        // Order-insensitive: ancestors/descendants return a deterministic BFS
        // order that follows edge-insertion order, so compare as sorted sets.
        let mut anc = dag.ancestors(2);
        anc.sort();
        assert_eq!(anc, vec![0, 1]);
        let mut des = dag.descendants(0);
        des.sort();
        assert_eq!(des, vec![1, 2]);
    }

    #[test]
    fn topological_order_respects_direction() {
        let mut dag = CausalDag::new(4);
        dag.add_directed_edge(0, 1).unwrap();
        dag.add_directed_edge(0, 2).unwrap();
        dag.add_directed_edge(1, 3).unwrap();
        dag.add_directed_edge(2, 3).unwrap();
        let order = dag.topo_order().unwrap();
        // Each edge u -> v must place u before v.
        for u in 0..4
        {
            for &v in dag.children(u)
            {
                let pu = order.iter().position(|x| *x == u).unwrap();
                let pv = order.iter().position(|x| *x == v).unwrap();
                assert!(pu < pv, "{u} should precede {v}");
            }
        }
    }

    #[test]
    fn to_undirected_interops_with_graph() {
        let mut dag = CausalDag::new(2);
        dag.add_directed_edge(0, 1).unwrap();
        let g = dag.to_undirected();
        assert_eq!(g.n_nodes, 2);
        assert!(g.are_adjacent(0, 1));
        assert_eq!(g.n_edges(), 1);
    }

    #[test]
    fn labeled_edges_round_trip() {
        let mut dag = CausalDag::new(2);
        dag.add_directed_edge_labeled(0, 1, Some("direct_cause"))
            .unwrap();
        assert_eq!(dag.children(0), &[1]);
        assert_eq!(dag.parents(1), &[0]);
    }
}
