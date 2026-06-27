//! Continuous improvement loop.
//!
//! Retrieval should get *better with use*: every time a user picks the document
//! they actually wanted, that `(query, relevant document)` pair is a free
//! training signal. [`ImprovementLoop`] accumulates those pairs (as base
//! embeddings, so re-training never re-encodes) and periodically re-trains the
//! contrastive [`ProjectionHead`] on **all** accumulated feedback, warm-starting
//! from the current weights. Quality is measured each cycle with the crate's own
//! metrics, so the gain is a number — and the whole loop is deterministic (seeded
//! init, fixed-order training), so the same feedback always yields the same
//! improvement, auditable cycle by cycle.

use crate::DenseIndex;
use crate::contrastive::{ContrastiveConfig, ProjectionHead, train};
use crate::metrics::recall_at_k;
use std::collections::HashSet;

/// A feedback-driven trainer that accumulates relevance pairs and re-trains a
/// projection head on demand.
pub struct ImprovementLoop {
    head: ProjectionHead,
    queries: Vec<Vec<f32>>,
    positives: Vec<Vec<f32>>,
    cfg: ContrastiveConfig,
}

impl ImprovementLoop {
    /// Start a fresh loop: a randomly-initialised head (seeded) projecting
    /// `dim_in` base embeddings to `dim_out`, trained with `cfg` each cycle.
    pub fn new(dim_in: usize, dim_out: usize, seed: u64, cfg: ContrastiveConfig) -> Self {
        Self {
            head: ProjectionHead::new(dim_in, dim_out, seed),
            queries: Vec::new(),
            positives: Vec::new(),
            cfg,
        }
    }

    /// Record one `(query, relevant-document)` pair as **base embeddings**.
    pub fn record(&mut self, query: &[f32], positive: &[f32]) {
        self.queries.push(query.to_vec());
        self.positives.push(positive.to_vec());
    }

    /// Number of accumulated feedback pairs.
    pub fn feedback_len(&self) -> usize {
        self.queries.len()
    }

    /// Whether any feedback has been recorded.
    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }

    /// The current projection head.
    pub fn head(&self) -> &ProjectionHead {
        &self.head
    }

    /// Project a base embedding through the current head (for indexing/querying).
    pub fn project(&self, v: &[f32]) -> Vec<f32> {
        self.head.project(v)
    }

    /// Re-train the head on **all** accumulated feedback (warm-started from the
    /// current weights), returning the per-epoch InfoNCE losses. A no-op
    /// (empty vec) until at least one pair has been recorded.
    pub fn train_cycle(&mut self) -> Vec<f32> {
        if self.queries.is_empty()
        {
            return Vec::new();
        }
        train(&mut self.head, &self.queries, &self.positives, self.cfg)
    }

    /// Mean Recall@`k` of the **current** head over an evaluation set. The
    /// `corpus` is `(id, base-embedding)`; each eval entry is
    /// `(query base-embedding, relevant id)`. Documents and queries are projected
    /// through the head, indexed exactly ([`DenseIndex`]), and scored. Returns
    /// `0.0` for an empty eval set.
    pub fn evaluate_recall_at_k(
        &self,
        eval: &[(Vec<f32>, u64)],
        corpus: &[(u64, Vec<f32>)],
        k: usize,
    ) -> f64 {
        if eval.is_empty()
        {
            return 0.0;
        }
        let mut index = DenseIndex::new(self.head.dim_out());
        for (id, emb) in corpus
        {
            index
                .add(*id, &self.head.project(emb))
                .expect("corpus embedding dimension matches the head");
        }
        let mut sum = 0.0;
        for (query, relevant_id) in eval
        {
            let ranked: Vec<u64> = index
                .search(&self.head.project(query), k)
                .into_iter()
                .map(|s| s.id)
                .collect();
            let relevant: HashSet<u64> = [*relevant_id].into_iter().collect();
            sum += recall_at_k(&ranked, &relevant, k);
        }
        sum / eval.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // "Two views": query i = [eᵢ ; 0], document i = [0 ; eᵢ]. The halves are
    // disjoint, so a query and its document have raw cosine 0 — the head must
    // *learn* the cross-view mapping per class from feedback.
    fn query_view(i: usize, n: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; 2 * n];
        v[i] = 1.0;
        v
    }
    fn doc_view(i: usize, n: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; 2 * n];
        v[n + i] = 1.0;
        v
    }

    #[test]
    fn retrieval_quality_climbs_as_feedback_accumulates() {
        let n = 8;
        let cfg = ContrastiveConfig {
            epochs: 300,
            lr: 0.05,
            temperature: 0.1,
        };
        let corpus: Vec<(u64, Vec<f32>)> = (0..n).map(|i| (i as u64, doc_view(i, n))).collect();
        let eval: Vec<(Vec<f32>, u64)> = (0..n).map(|i| (query_view(i, n), i as u64)).collect();

        let mut loop_ = ImprovementLoop::new(2 * n, 8, 7, cfg);
        let mut curve = Vec::new();
        curve.push(loop_.evaluate_recall_at_k(&eval, &corpus, 1)); // before any feedback

        // Each cycle teaches two more classes; eval is over ALL classes.
        for c in 0..(n / 2)
        {
            loop_.record(&query_view(2 * c, n), &doc_view(2 * c, n));
            loop_.record(&query_view(2 * c + 1, n), &doc_view(2 * c + 1, n));
            loop_.train_cycle();
            curve.push(loop_.evaluate_recall_at_k(&eval, &corpus, 1));
        }

        // Starts near chance (random head, disjoint views), ends perfect.
        assert!(curve[0] < 0.5, "initial recall too high: {curve:?}");
        assert_eq!(
            *curve.last().unwrap(),
            1.0,
            "final recall not perfect: {curve:?}"
        );
        assert!(
            *curve.last().unwrap() > curve[0],
            "no improvement: {curve:?}"
        );
        // More feedback never hurts on this structured task (non-decreasing).
        for w in curve.windows(2)
        {
            assert!(
                w[1] >= w[0] - 1e-9,
                "quality regressed across a cycle: {curve:?}"
            );
        }
    }

    #[test]
    fn a_cycle_reduces_the_contrastive_loss() {
        let n = 5;
        let mut loop_ = ImprovementLoop::new(2 * n, 6, 1, ContrastiveConfig::default());
        for i in 0..n
        {
            loop_.record(&query_view(i, n), &doc_view(i, n));
        }
        let losses = loop_.train_cycle();
        assert!(
            *losses.last().unwrap() < losses[0],
            "loss should fall within a cycle: {} -> {}",
            losses[0],
            losses.last().unwrap()
        );
    }

    #[test]
    fn an_empty_loop_trains_nothing_and_scores_zero() {
        let loop_ = ImprovementLoop::new(4, 4, 0, ContrastiveConfig::default());
        assert!(loop_.is_empty());
        assert_eq!(loop_.evaluate_recall_at_k(&[], &[], 1), 0.0);
        let mut l = loop_;
        assert!(l.train_cycle().is_empty()); // no feedback -> no-op
    }
}
