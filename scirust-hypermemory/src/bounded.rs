//! Phase 2 — deterministic bounded, decay-aware memory.
//!
//! Phase 1 defined the [`RetentionPolicy`] interface but ran **no** automatic
//! eviction: the exact [`S16Store`] grows unbounded, and only explicit
//! [`S16Store::remove`] changes residency. This module adds the bounded layer:
//! a capacity-capped memory that, when full, evicts the lowest-retention
//! resident (per a [`RetentionPolicy`]) to make room, and can [`forget`] every
//! resident below a threshold. Access bumps recency, so frequently-queried
//! concepts survive.
//!
//! [`forget`]: S16BoundedMemory::forget
//!
//! Everything is deterministic: eviction picks the minimum retention score with
//! an ascending-[`ConceptId`] tie-break, all time is a caller-supplied logical
//! `u64` tick (never wall-clock), and the store/index stay in lock-step.
//!
//! ## Relation to `scirust_retrieval::BoundedSemanticMemory`
//!
//! `BoundedSemanticMemory` is the workspace's existing bounded store. Phase 2
//! deliberately mirrors its *shape* (importance + recency retention, eviction
//! when full, threshold-based forgetting) while keeping the two intentional
//! Phase 1 differences: a **logical `u64` tick** instead of `f64` wall-clock
//! timestamps (so retention is bit-reproducible), and **generation-safe
//! [`ConceptId`]s** instead of raw `u64` ids. It composes the Phase 1 store and
//! exact index rather than re-deriving a policy.

use crate::error::{HypermemoryError, Result};
use crate::id::ConceptId;
use crate::index::{S16ExactIndex, SearchHit, SimilarityMetric};
use crate::metadata::{NoForgetting, RetentionPolicy};
use crate::record::ConceptRecord;
use crate::store::{ConceptSpec, S16Store};
use scirust_simd::hypercomplex::SedenionSimd;

/// The outcome of a bounded insertion: the new id, and the id evicted to make
/// room (if the memory was full).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Insertion {
    /// The id assigned to the inserted concept.
    pub inserted: ConceptId,
    /// The concept evicted to make room, if any.
    pub evicted: Option<ConceptId>,
}

/// A capacity-capped, decay-aware memory over the Phase 1 [`S16Store`] and
/// [`S16ExactIndex`], parameterized by a [`RetentionPolicy`] (default
/// [`NoForgetting`]).
#[derive(Clone, Debug)]
pub struct S16BoundedMemory<P: RetentionPolicy = NoForgetting> {
    store: S16Store,
    index: S16ExactIndex,
    policy: P,
    capacity: usize,
}

impl<P: RetentionPolicy> S16BoundedMemory<P> {
    /// A new bounded memory of `capacity` slots using `metric` for search and
    /// `policy` for retention/eviction.
    #[must_use]
    pub fn new(capacity: usize, metric: SimilarityMetric, policy: P) -> Self {
        Self {
            store: S16Store::new(),
            index: S16ExactIndex::new(metric),
            policy,
            capacity,
        }
    }

    /// The configured capacity.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of resident concepts.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Whether the memory is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// The active similarity metric.
    #[inline]
    #[must_use]
    pub fn metric(&self) -> SimilarityMetric {
        self.index.metric()
    }

    /// Borrow the retention policy.
    #[inline]
    #[must_use]
    pub const fn policy(&self) -> &P {
        &self.policy
    }

    /// Borrow a resident record, or the precise error if `id` does not resolve.
    pub fn get(&self, id: ConceptId) -> Result<&ConceptRecord> {
        self.store.get(id)
    }

    /// Whether `id` currently resides.
    #[must_use]
    pub fn contains(&self, id: ConceptId) -> bool {
        self.store.contains(id)
    }

    /// Deterministic iteration over residents in slot order.
    pub fn iter(&self) -> impl Iterator<Item = &ConceptRecord> + '_ {
        self.store.iter()
    }

    /// The retention score of `id` at logical tick `now`.
    pub fn retention_score(&self, id: ConceptId, now: u64) -> Result<f32> {
        let record = self.store.get(id)?;
        Ok(self.policy.retention_score(record.metadata(), now))
    }

    /// Insert a concept, evicting the lowest-retention resident (evaluated at
    /// `spec.tick`) first if the memory is full.
    ///
    /// Returns the new id and any evicted id. Fails with
    /// [`HypermemoryError::CapacityExhausted`] only if `capacity == 0`, and
    /// propagates the store's validation errors otherwise (leaving any eviction
    /// already performed — the caller sees the returned error and the memory is
    /// left with room freed, which is safe and deterministic).
    pub fn insert(&mut self, spec: ConceptSpec) -> Result<Insertion> {
        if self.capacity == 0
        {
            return Err(HypermemoryError::CapacityExhausted { capacity: 0 });
        }
        let now = spec.tick;
        let mut evicted = None;
        if self.store.len() >= self.capacity
        {
            evicted = self.evict_lowest(now);
        }
        let inserted = self.store.insert(spec)?;
        // The record is guaranteed present; keep the index in lock-step.
        self.index.insert_concept(self.store.get(inserted)?);
        Ok(Insertion { inserted, evicted })
    }

    /// Exact top-`k` search, then bump the recency of every returned concept at
    /// logical tick `now` (so retrieval keeps a concept alive).
    pub fn search(&mut self, query: &SedenionSimd, k: usize, now: u64) -> Result<Vec<SearchHit>> {
        let hits = self.index.search(query, k)?;
        for hit in &hits
        {
            // A hit always resolves (index and store are in lock-step); ignore a
            // benign miss rather than fail the whole search.
            let _ = self.store.touch(hit.id, now);
        }
        Ok(hits)
    }

    /// Phase 3 — one bounded residual-learning step on a resident concept,
    /// keeping the index in lock-step and bumping the concept's recency at
    /// `now`. Semantics and errors are those of [`S16Store::learn_residual`].
    pub fn learn(
        &mut self,
        id: ConceptId,
        target: &SedenionSimd,
        rate: f32,
        now: u64,
    ) -> Result<crate::LearnOutcome> {
        let outcome = self.store.learn_residual(id, target, rate)?;
        self.index.update_concept(self.store.get(id)?);
        let _ = self.store.touch(id, now);
        Ok(outcome)
    }

    /// Cleanup/denoising against the resident concepts, bumping the recency of
    /// the recognized concept at `now`. Semantics and errors are those of
    /// [`crate::S16ExactIndex::denoise`]; a rejected (below-threshold) input
    /// touches nothing.
    pub fn denoise(
        &mut self,
        noisy: &SedenionSimd,
        threshold: f32,
        now: u64,
    ) -> Result<Option<crate::Denoised>> {
        let outcome = self.index.denoise(noisy, threshold)?;
        if let Some(d) = &outcome
        {
            let _ = self.store.touch(d.id, now);
        }
        Ok(outcome)
    }

    /// Evict every resident whose retention score at `now` is strictly below
    /// `threshold`. Returns the evicted ids in ascending order (deterministic).
    pub fn forget(&mut self, now: u64, threshold: f32) -> Vec<ConceptId> {
        let mut doomed: Vec<ConceptId> = self
            .store
            .iter()
            .filter(|record| self.policy.retention_score(record.metadata(), now) < threshold)
            .map(ConceptRecord::id)
            .collect();
        doomed.sort_unstable();
        for &id in &doomed
        {
            let _ = self.store.remove(id);
            self.index.remove(id);
        }
        doomed
    }

    /// Find and evict the single lowest-retention resident at `now`
    /// (tie-break: ascending [`ConceptId`]). Returns the evicted id, or `None`
    /// if empty.
    fn evict_lowest(&mut self, now: u64) -> Option<ConceptId> {
        let mut best: Option<(f32, ConceptId)> = None;
        for record in self.store.iter()
        {
            let score = self.policy.retention_score(record.metadata(), now);
            let id = record.id();
            best = match best
            {
                None => Some((score, id)),
                Some((bs, bid)) =>
                {
                    // Lower score wins; tie-break to the smaller id. `total_cmp`
                    // keeps this deterministic even with non-finite scores.
                    if score.total_cmp(&bs).is_lt() || (score.total_cmp(&bs).is_eq() && id < bid)
                    {
                        Some((score, id))
                    }
                    else
                    {
                        Some((bs, bid))
                    }
                },
            };
        }
        let (_, victim) = best?;
        let _ = self.store.remove(victim);
        self.index.remove(victim);
        Some(victim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::LinearDecay;
    use scirust_simd::hypercomplex::SedenionSimd;

    fn spec(tag: u8, unit: usize, importance: f32, tick: u64) -> ConceptSpec {
        ConceptSpec::new(vec![tag], SedenionSimd::unit(unit), importance, tick)
    }

    #[test]
    fn insert_below_capacity_never_evicts() {
        let mut mem = S16BoundedMemory::new(4, SimilarityMetric::Cosine, NoForgetting);
        let r = mem.insert(spec(0, 0, 1.0, 0)).unwrap();
        assert_eq!(r.evicted, None);
        assert_eq!(mem.len(), 1);
        assert!(mem.contains(r.inserted));
    }

    #[test]
    fn full_memory_evicts_lowest_importance() {
        // NoForgetting → retention = importance. Filling then inserting evicts
        // the least important resident.
        let mut mem = S16BoundedMemory::new(2, SimilarityMetric::Cosine, NoForgetting);
        let low = mem.insert(spec(0, 0, 0.5, 0)).unwrap().inserted;
        let high = mem.insert(spec(1, 1, 5.0, 0)).unwrap().inserted;
        assert_eq!(mem.len(), 2);
        let ins = mem.insert(spec(2, 2, 3.0, 1)).unwrap();
        assert_eq!(
            ins.evicted,
            Some(low),
            "the least important resident is evicted"
        );
        assert_eq!(mem.len(), 2);
        assert!(!mem.contains(low));
        assert!(mem.contains(high));
        assert!(mem.contains(ins.inserted));
    }

    #[test]
    fn eviction_tie_breaks_by_ascending_id() {
        // Equal importance → the smaller ConceptId is evicted first.
        let mut mem = S16BoundedMemory::new(2, SimilarityMetric::Cosine, NoForgetting);
        let a = mem.insert(spec(0, 0, 1.0, 0)).unwrap().inserted;
        let b = mem.insert(spec(1, 1, 1.0, 0)).unwrap().inserted;
        assert!(a < b);
        let ins = mem.insert(spec(2, 2, 1.0, 0)).unwrap();
        assert_eq!(ins.evicted, Some(a), "tie must evict the smaller id");
        assert!(!mem.contains(a));
        assert!(mem.contains(b));
    }

    #[test]
    fn access_recency_protects_a_concept_under_decay() {
        // LinearDecay(half_life=10): retention = importance + recency. Give two
        // equal-importance concepts; access one so its recency stays high, then
        // insert a third — the un-accessed (staler) one is evicted.
        let mut mem = S16BoundedMemory::new(2, SimilarityMetric::Cosine, LinearDecay::new(10));
        let stale = mem.insert(spec(0, 0, 1.0, 0)).unwrap().inserted;
        let fresh = mem.insert(spec(1, 1, 1.0, 0)).unwrap().inserted;
        // At tick 8, query near `fresh` (unit 1) → it is touched, recency reset.
        let q = SedenionSimd::unit(1);
        let hits = mem.search(&q, 1, 8).unwrap();
        assert_eq!(hits[0].id, fresh);
        // Insert a third at tick 9: `stale` (last access 0, age 9) has lower
        // recency than `fresh` (last access 8, age 1) → `stale` is evicted.
        let ins = mem.insert(spec(2, 2, 1.0, 9)).unwrap();
        assert_eq!(ins.evicted, Some(stale));
        assert!(mem.contains(fresh));
    }

    #[test]
    fn forget_evicts_everything_below_threshold() {
        let mut mem = S16BoundedMemory::new(8, SimilarityMetric::Cosine, NoForgetting);
        let a = mem.insert(spec(0, 0, 0.2, 0)).unwrap().inserted;
        let b = mem.insert(spec(1, 1, 1.0, 0)).unwrap().inserted;
        let c = mem.insert(spec(2, 2, 0.1, 0)).unwrap().inserted;
        // Threshold 0.5 → a (0.2) and c (0.1) go, b (1.0) stays.
        let evicted = mem.forget(0, 0.5);
        assert_eq!(evicted, {
            let mut v = vec![a, c];
            v.sort_unstable();
            v
        });
        assert!(mem.contains(b));
        assert_eq!(mem.len(), 1);
    }

    #[test]
    fn zero_capacity_rejects_insert() {
        let mut mem = S16BoundedMemory::new(0, SimilarityMetric::Cosine, NoForgetting);
        assert_eq!(
            mem.insert(spec(0, 0, 1.0, 0)),
            Err(HypermemoryError::CapacityExhausted { capacity: 0 })
        );
    }

    #[test]
    fn search_stays_consistent_after_eviction() {
        // A scripted workload is reproducible and the index never returns an
        // evicted concept.
        let mut mem = S16BoundedMemory::new(3, SimilarityMetric::Cosine, NoForgetting);
        for i in 0..6u8
        {
            let _ = mem
                .insert(spec(i, (i % 4) as usize, 1.0 + i as f32, i as u64))
                .unwrap();
        }
        assert_eq!(mem.len(), 3);
        let q = SedenionSimd::unit(1);
        let hits = mem.search(&q, 3, 100).unwrap();
        // Every hit still resides.
        for h in &hits
        {
            assert!(mem.contains(h.id));
        }
    }
}
