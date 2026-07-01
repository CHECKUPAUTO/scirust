//! Causal-aware retrieval — re-rank similarity hits by causal relevance.
//!
//! `scirust-retrieval` ranks purely by cosine similarity ([`crate::DenseIndex`])
//! and BM25 lexical fusion ([`crate::hybrid`]) — there is no notion of *causal*
//! relevance. When memories carry causal structure (a [`CausalDag`] over
//! document ids), a query that asks "what *caused* X" should rank X's ancestors
//! above its descendants even if their embedding similarity is similar. This
//! module adds [`CausalReranker`]: it takes the exact similarity top-k and
//! re-sorts by a combined score of similarity + causal-graph proximity to a
//! set of *focus* nodes, with an optional `intervention_overlap` term.
//!
//! Deterministic, fixed-order arithmetic; the exact similarity hit list is left
//! untouched — only the final ordering changes.

use crate::{DenseIndex, Encoder, Scored};
use scirust_graph::dag::CausalDag;
use std::collections::{HashMap, HashSet};

/// How to weight the causal component against similarity in
/// [`CausalReranker`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CausalWeights {
    /// Weight on cosine similarity (0..=1).
    pub similarity: f32,
    /// Weight on causal-graph proximity to the focus set (0..=1).
    pub causal_proximity: f32,
    /// Weight on intervention-set overlap (0..=1).
    pub intervention_overlap: f32,
}

impl Default for CausalWeights {
    fn default() -> Self {
        Self {
            similarity: 0.6,
            causal_proximity: 0.3,
            intervention_overlap: 0.1,
        }
    }
}

/// A causal-aware retrieval result.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalRelevance {
    pub id: u64,
    pub similarity: f32,
    pub causal_proximity: f32,
    pub intervention_overlap: f32,
    pub combined: f32,
}

/// Re-rank the exact similarity top-k by causal relevance to a `focus` node
/// set in `dag`.
pub struct CausalReranker<'a> {
    dag: &'a CausalDag,
    weights: CausalWeights,
}

impl<'a> CausalReranker<'a> {
    pub fn new(dag: &'a CausalDag) -> Self {
        Self {
            dag,
            weights: CausalWeights::default(),
        }
    }

    pub fn with_weights(mut self, weights: CausalWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Causal proximity of `node` to the `focus` set: `1.0` if `node` is in the
    /// set, decaying by graph distance otherwise (BFS shortest directed path
    /// in either direction — `node` is an ancestor of a focus node, or a
    /// descendant of one), `0.0` if unreachable in either direction.
    fn proximity(&self, node: usize, focus: &HashSet<usize>) -> f32 {
        if node >= self.dag.n_nodes()
        {
            return 0.0;
        }
        if focus.contains(&node)
        {
            return 1.0;
        }
        // Shortest distance, in either direction along the directed graph:
        //   focus -> ... -> node  (node is a descendant of a focus node)
        //   node  -> ... -> focus  (node is an ancestor of a focus node)
        // Take the min over both, over every focus node.
        let d = focus
            .iter()
            .filter_map(|&f| {
                let fwd = self.bfs_dist(f, node); // f -> node
                let bwd = self.bfs_dist(node, f); // node -> f
                [fwd, bwd].into_iter().flatten().min()
            })
            .min()
            .unwrap_or(usize::MAX);
        if d == usize::MAX
        {
            0.0
        }
        else
        {
            1.0 / (1.0 + d as f32)
        }
    }

    /// BFS shortest directed distance from `src` to `dst` (forward, along
    /// child edges). `Some(0)` if `src == dst`, `None` if unreachable.
    fn bfs_dist(&self, src: usize, dst: usize) -> Option<usize> {
        // Out-of-range endpoints are unreachable (and would panic on indexing).
        if src >= self.dag.n_nodes() || dst >= self.dag.n_nodes()
        {
            return None;
        }
        if src == dst
        {
            return Some(0);
        }
        let mut visited = vec![false; self.dag.n_nodes()];
        let mut frontier: Vec<(usize, usize)> = vec![(src, 0)];
        visited[src] = true;
        let mut head = 0;
        while head < frontier.len()
        {
            let (node, d) = frontier[head];
            head += 1;
            for &n in self.dag.children(node)
            {
                if n == dst
                {
                    return Some(d + 1);
                }
                if !visited[n]
                {
                    visited[n] = true;
                    frontier.push((n, d + 1));
                }
            }
        }
        None
    }

    /// Overlap of `node`'s intervention-ancestor set with `focus`'s, in
    /// `[0, 1]` (Jaccard). Documents that share the same causal ancestry as the
    /// focus are likely co-effects of a common cause.
    fn intervention_overlap(&self, node: usize, focus: &HashSet<usize>) -> f32 {
        // Out-of-range nodes have no ancestry (and would panic on indexing).
        let n = self.dag.n_nodes();
        let a = if node < n
        {
            self.dag.intervention_ancestors(node)
        }
        else
        {
            HashSet::new()
        };
        let mut b = HashSet::new();
        for &f in focus.iter().filter(|&&f| f < n)
        {
            for x in self.dag.intervention_ancestors(f)
            {
                b.insert(x);
            }
        }
        if a.is_empty() && b.is_empty()
        {
            return 0.0;
        }
        let inter = a.intersection(&b).count() as f32;
        let union = a.union(&b).count().max(1) as f32;
        inter / union
    }

    /// Re-rank `hits` (the exact similarity top-k) by causal relevance to
    /// `focus` nodes. `id_to_node` maps document ids to `CausalDag` node
    /// indices. Documents with no node mapping keep their similarity only.
    pub fn rerank(
        &self,
        hits: &[Scored],
        focus: &HashSet<usize>,
        id_to_node: &HashMap<u64, usize>,
    ) -> Vec<CausalRelevance> {
        let mut scored: Vec<CausalRelevance> = hits
            .iter()
            .map(|h| {
                let (prox, ov) = match id_to_node.get(&h.id)
                {
                    Some(&node) => (
                        self.proximity(node, focus),
                        self.intervention_overlap(node, focus),
                    ),
                    None => (0.0, 0.0),
                };
                let w = self.weights;
                // Normalise similarity to [0,1] defensively (cosine can be < 0).
                let sim = (h.score + 1.0) / 2.0;
                let combined =
                    w.similarity * sim + w.causal_proximity * prox + w.intervention_overlap * ov;
                CausalRelevance {
                    id: h.id,
                    similarity: h.score,
                    causal_proximity: prox,
                    intervention_overlap: ov,
                    combined,
                }
            })
            .collect();
        // Sort by combined score descending, ties broken by smaller id.
        scored.sort_by(|a, b| {
            b.combined
                .partial_cmp(&a.combined)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.id.cmp(&b.id))
        });
        scored
    }

    /// Convenience: index `index`, query `encoder`, then causal-rerank.
    pub fn retrieve<E: Encoder>(
        &self,
        encoder: &mut E,
        index: &DenseIndex,
        query: &str,
        k: usize,
        focus: &HashSet<usize>,
        id_to_node: &HashMap<u64, usize>,
    ) -> Vec<CausalRelevance> {
        let q = encoder.encode(query);
        let hits = index.search(&q, k);
        self.rerank(&hits, focus, id_to_node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(idx: &mut DenseIndex, id: u64, v: &[f32]) {
        idx.add(id, v).unwrap();
    }

    #[test]
    fn causal_ancestors_outrank_unrelated_on_tie() {
        // Three nodes: 0 -> 1, 0 -> 2. Focus is {2}. Two documents with
        // *identical* similarity: doc 10 (node 0, an ancestor of the focus) and
        // doc 20 (node 1, causally unrelated to the focus). The ancestor must
        // rank first — same similarity, strictly higher causal proximity.
        let mut idx = DenseIndex::new(2);
        doc(&mut idx, 10, &[1.0, 0.0]); // node 0 — ancestor of focus
        doc(&mut idx, 20, &[1.0, 0.0]); // node 1 — unrelated, same vector → tie
        let mut dag = CausalDag::new(3);
        dag.add_directed_edge(0, 1).unwrap();
        dag.add_directed_edge(0, 2).unwrap();
        let mut id_to_node = HashMap::new();
        id_to_node.insert(10, 0);
        id_to_node.insert(20, 1);
        let focus: HashSet<usize> = [2].into_iter().collect();
        let reranker = CausalReranker::new(&dag);
        let ranked = reranker.rerank(&idx.search(&[1.0, 0.0], 2), &focus, &id_to_node);
        // Doc 10 (node 0) reaches focus node 2 in one hop → proximity 0.5;
        // doc 20 (node 1) cannot reach the focus and is not its ancestor → 0.0.
        assert_eq!(ranked[0].id, 10);
        assert!(ranked[0].causal_proximity > ranked[1].causal_proximity);
        assert!((ranked[0].causal_proximity - 0.5).abs() < 1e-6);
        assert!((ranked[1].causal_proximity - 0.0).abs() < 1e-6);
    }

    #[test]
    fn similarity_dominates_when_weights_favour_it() {
        let mut idx = DenseIndex::new(2);
        doc(&mut idx, 1, &[1.0, 0.0]); // very similar to query
        doc(&mut idx, 2, &[0.0, 1.0]); // dissimilar but is the focus itself
        let dag = CausalDag::new(2);
        let mut id_to_node = HashMap::new();
        id_to_node.insert(1, 0);
        id_to_node.insert(2, 1);
        let focus: HashSet<usize> = [1].into_iter().collect();
        let reranker = CausalReranker::new(&dag).with_weights(CausalWeights {
            similarity: 0.95,
            causal_proximity: 0.025,
            intervention_overlap: 0.025,
        });
        let ranked = reranker.rerank(&idx.search(&[1.0, 0.0], 2), &focus, &id_to_node);
        assert_eq!(ranked[0].id, 1); // similarity wins
    }

    #[test]
    fn out_of_range_node_indices_do_not_panic() {
        // Regression: a document mapped to a node index >= dag.n_nodes(), and a
        // focus set containing an out-of-range node, must be treated as having
        // no causal signal (proximity/overlap 0.0) rather than panicking on an
        // out-of-bounds Vec index inside the DAG's BFS / ancestor walks.
        let mut idx = DenseIndex::new(2);
        doc(&mut idx, 1, &[1.0, 0.0]); // maps to a valid node
        doc(&mut idx, 2, &[1.0, 0.0]); // maps to an out-of-range node
        let mut dag = CausalDag::new(2);
        dag.add_directed_edge(0, 1).unwrap();
        let mut id_to_node = HashMap::new();
        id_to_node.insert(1, 0);
        id_to_node.insert(2, 99); // 99 >= n_nodes (2) — out of range
        // Focus references both an in-range and an out-of-range node.
        let focus: HashSet<usize> = [1, 42].into_iter().collect();
        let reranker = CausalReranker::new(&dag);
        let ranked = reranker.rerank(&idx.search(&[1.0, 0.0], 2), &focus, &id_to_node);
        assert_eq!(ranked.len(), 2);
        // The out-of-range document (id 2) gets no causal signal.
        let doc2 = ranked.iter().find(|r| r.id == 2).unwrap();
        assert!((doc2.causal_proximity - 0.0).abs() < 1e-6);
        assert!((doc2.intervention_overlap - 0.0).abs() < 1e-6);
        // The in-range ancestor (id 1, node 0 -> focus node 1) still scores.
        let doc1 = ranked.iter().find(|r| r.id == 1).unwrap();
        assert!((doc1.causal_proximity - 0.5).abs() < 1e-6);
    }
}
