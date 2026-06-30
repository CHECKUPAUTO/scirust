//! Bounded, decay-aware semantic memory.
//!
//! `scirust-retrieval`'s [`DenseIndex`] / [`LshIndex`] / [`Bm25Index`] grow
//! unbounded — no capacity cap, no recency weighting, no importance-based
//! eviction — and [`ImprovementLoop`] accumulates feedback forever. For an
//! agent that runs indefinitely that is a memory leak. This module adds the
//! bounds/decay layer that turns the exact dense index into a **bounded
//! semantic memory**: every document carries [`DocMeta`] (importance,
//! written-at, access count, last access), a configurable [`DecaySchedule`]
//! down-weights stale entries, and a capacity enforces eviction by a combined
//! score when the store is full. The raw (pre-normalisation) vectors are kept
//! alongside so eviction/rebuild is lossless — `DenseIndex` has no `remove()`.
//!
//! The companion change is in [`crate::feedback`]: `ImprovementLoop` gains a
//! backward-compatible `replay_cap` so the feedback replay buffer also stays
//! finite. Everything is deterministic, fixed-order arithmetic.

use crate::{DenseIndex, RetrievalError, Scored};
use serde::{Deserialize, Serialize};

/// Per-document bookkeeping for [`BoundedSemanticMemory`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocMeta {
    pub id: u64,
    pub written_at: f64,
    pub importance: f32,
    pub access_count: u32,
    pub last_access: f64,
}

/// How the recency component of the retention score decays with age
/// (`now - last_access`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DecaySchedule {
    /// No decay: retention depends only on importance.
    None,
    /// Linear decay: `recency = max(0, 1 - age / half_life)`.
    Linear { half_life: f64 },
    /// Exponential decay: `recency = exp(-age / tau)`.
    Exponential { tau: f64 },
}

impl DecaySchedule {
    /// Recency in `[0, 1]` for a document last touched `age` time-units ago.
    pub fn recency(self, age: f64) -> f32 {
        if age <= 0.0
        {
            return 1.0;
        }
        match self
        {
            DecaySchedule::None => 1.0,
            DecaySchedule::Linear { half_life } =>
            {
                let r = 1.0 - age / half_life.max(1e-12);
                r.max(0.0) as f32
            },
            DecaySchedule::Exponential { tau } => (-(age / tau.max(1e-12))).exp() as f32,
        }
    }
}

/// A capacity-capped, decay-aware wrapper over the exact [`DenseIndex`].
///
/// `add` records a document with an importance and timestamp; if the store is
/// full it evicts the lowest-scoring resident (importance + recency) first.
/// `search` returns the exact top-k by cosine similarity (delegated to
/// [`DenseIndex`]) and bumps the accessed documents' `last_access` /
/// `access_count`. `forget` evicts every resident whose score fell below a
/// threshold.
pub struct BoundedSemanticMemory {
    index: DenseIndex,
    /// Raw (pre-normalisation) vectors, kept parallel to `meta` so eviction /
    /// rebuild never loses a document. `DenseIndex` has no `remove()`.
    raw: Vec<Vec<f32>>,
    meta: Vec<DocMeta>,
    capacity: usize,
    decay: DecaySchedule,
}

impl BoundedSemanticMemory {
    /// New bounded memory of `capacity` slots over `dim`-dimensional embeddings,
    /// using `decay` to weight recency.
    pub fn new(dim: usize, capacity: usize, decay: DecaySchedule) -> Self {
        Self {
            index: DenseIndex::new(dim),
            raw: Vec::with_capacity(capacity),
            meta: Vec::with_capacity(capacity),
            capacity,
            decay,
        }
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.index.dim()
    }

    /// Number of resident documents.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Borrow the metadata (parallel to the index rows, in insertion order).
    pub fn meta(&self) -> &[DocMeta] {
        &self.meta
    }

    /// Retention score of resident `i` at time `now`: importance + recency.
    fn score(&self, i: usize, now: f64) -> f32 {
        let m = &self.meta[i];
        m.importance + self.decay.recency(now - m.last_access)
    }

    /// Index of the lowest-scoring resident at `now` (ties → smallest id).
    #[allow(clippy::needless_range_loop)]
    fn worst_index(&self, now: f64) -> usize {
        let mut worst = 0usize;
        let mut worst_score = self.score(0, now);
        for i in 1..self.meta.len()
        {
            let s = self.score(i, now);
            if s < worst_score || (s == worst_score && self.meta[i].id < self.meta[worst].id)
            {
                worst = i;
                worst_score = s;
            }
        }
        worst
    }

    /// Rebuild the index/meta/raw stores keeping only the residents whose
    /// positions are in `keep` (true = keep).
    fn rebuild(&mut self, keep: &[bool]) {
        let dim = self.index.dim();
        let mut new_index = DenseIndex::new(dim);
        let mut new_raw = Vec::new();
        let mut new_meta = Vec::new();
        for (i, keep_i) in keep.iter().enumerate()
        {
            if *keep_i
            {
                new_index
                    .add(self.meta[i].id, &self.raw[i])
                    .expect("dim unchanged");
                new_raw.push(self.raw[i].clone());
                new_meta.push(self.meta[i].clone());
            }
        }
        self.index = new_index;
        self.raw = new_raw;
        self.meta = new_meta;
    }

    /// Add a document `vector` under `id` with `importance` at time `now`. If
    /// the store is full, evicts the lowest-scoring resident first.
    pub fn add(
        &mut self,
        id: u64,
        vector: &[f32],
        importance: f32,
        now: f64,
    ) -> Result<(), RetrievalError> {
        if self.index.len() >= self.capacity && !self.meta.is_empty()
        {
            let worst = self.worst_index(now);
            let mut keep = vec![true; self.meta.len()];
            keep[worst] = false;
            self.rebuild(&keep);
        }
        self.index.add(id, vector)?;
        self.raw.push(vector.to_vec());
        self.meta.push(DocMeta {
            id,
            written_at: now,
            importance,
            access_count: 0,
            last_access: now,
        });
        Ok(())
    }

    /// Exact top-`k` by cosine similarity (the rank is the exact one from
    /// [`DenseIndex`]); updates `last_access` / `access_count` for the returned
    /// documents.
    pub fn search(&mut self, query: &[f32], k: usize, now: f64) -> Vec<Scored> {
        let hits = self.index.search(query, k);
        for h in &hits
        {
            if let Some(i) = self.meta_idx(h.id)
            {
                self.meta[i].access_count += 1;
                self.meta[i].last_access = now;
            }
        }
        hits
    }

    /// Evict every resident whose score (importance + recency at `now`) is
    /// below `threshold`. Returns the evicted ids (insertion order).
    #[allow(clippy::needless_range_loop)]
    pub fn forget(&mut self, threshold: f32, now: f64) -> Vec<u64> {
        let mut evicted = Vec::new();
        let mut keep = vec![true; self.meta.len()];
        for i in 0..self.meta.len()
        {
            if self.score(i, now) < threshold
            {
                keep[i] = false;
                evicted.push(self.meta[i].id);
            }
        }
        self.rebuild(&keep);
        evicted
    }

    fn meta_idx(&self, id: u64) -> Option<usize> {
        self.meta.iter().position(|m| m.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add_n(mem: &mut BoundedSemanticMemory, n: usize, now: f64) {
        for i in 0..n
        {
            let mut v = vec![0.0f32; 4];
            v[i % 4] = 1.0;
            mem.add(i as u64, &v, 0.5, now).unwrap();
        }
    }

    #[test]
    fn capacity_evicts_lowest_score() {
        let mut mem = BoundedSemanticMemory::new(4, 2, DecaySchedule::None);
        // All importance 0.5, no decay → scores tie → smallest id evicted first.
        mem.add(10, &[1.0, 0.0, 0.0, 0.0], 0.5, 0.0).unwrap();
        mem.add(20, &[0.0, 1.0, 0.0, 0.0], 0.5, 0.0).unwrap();
        mem.add(30, &[0.0, 0.0, 1.0, 0.0], 0.5, 0.0).unwrap();
        assert_eq!(mem.len(), 2);
        // id 10 had the smallest score (tie) and was evicted.
        let ids: Vec<u64> = mem.meta().iter().map(|m| m.id).collect();
        assert_eq!(ids, vec![20, 30]);
    }

    #[test]
    fn importance_protects_a_document_from_eviction() {
        let mut mem = BoundedSemanticMemory::new(4, 2, DecaySchedule::None);
        mem.add(1, &[1.0, 0.0, 0.0, 0.0], 0.1, 0.0).unwrap(); // low importance
        mem.add(2, &[0.0, 1.0, 0.0, 0.0], 5.0, 0.0).unwrap(); // high importance
        mem.add(3, &[0.0, 0.0, 1.0, 0.0], 0.2, 0.0).unwrap();
        let ids: Vec<u64> = mem.meta().iter().map(|m| m.id).collect();
        // The high-importance doc 2 survives; the two low ones compete and the
        // smaller-score (0.1 < 0.2) doc 1 is evicted.
        assert!(ids.contains(&2));
        assert!(!ids.contains(&1));
    }

    #[test]
    fn search_updates_access_metadata() {
        let mut mem = BoundedSemanticMemory::new(4, 8, DecaySchedule::None);
        add_n(&mut mem, 4, 0.0);
        let hits = mem.search(&[1.0, 0.0, 0.0, 0.0], 1, 5.0);
        assert_eq!(hits[0].id, 0);
        let m0 = &mem.meta()[0];
        assert_eq!(m0.access_count, 1);
        assert_eq!(m0.last_access, 5.0);
    }

    #[test]
    fn decay_drops_stale_documents_via_forget() {
        let mut mem = BoundedSemanticMemory::new(4, 8, DecaySchedule::Linear { half_life: 10.0 });
        mem.add(1, &[1.0, 0.0, 0.0, 0.0], 0.5, 0.0).unwrap();
        mem.add(2, &[0.0, 1.0, 0.0, 0.0], 0.5, 0.0).unwrap();
        // Touch doc 1 recently; leave doc 2 stale.
        mem.search(&[1.0, 0.0, 0.0, 0.0], 1, 1.0);
        // At now=100 doc 2 has age ~100 → recency 0 → score 0.5; doc 1 age ~99 →
        // recency 0 → score 0.5 too. Use a threshold that removes both stale
        // docs (score < 0.6).
        let evicted = mem.forget(0.6, 100.0);
        assert_eq!(evicted.len(), 2);
        assert!(mem.is_empty());
    }

    #[test]
    fn rebuild_is_lossless_for_survivors() {
        let mut mem = BoundedSemanticMemory::new(2, 2, DecaySchedule::None);
        mem.add(1, &[1.0, 0.0], 0.5, 0.0).unwrap();
        mem.add(2, &[0.0, 1.0], 0.5, 0.0).unwrap();
        mem.add(3, &[1.0, 1.0], 0.5, 0.0).unwrap(); // evicts id 1 (tie, smallest)
        // Survivors 2 and 3 must still be retrievable.
        let hits = mem.search(&[0.0, 1.0], 2, 0.0);
        let ids: Vec<u64> = hits.iter().map(|h| h.id).collect();
        assert!(ids.contains(&2));
    }
}
