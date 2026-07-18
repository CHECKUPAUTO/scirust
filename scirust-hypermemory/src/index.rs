//! The exact exhaustive index — the Phase 1 oracle.
//!
//! `S16ExactIndex` scans every effective representation on every query and
//! returns the exact top-k with a deterministic total order and `ConceptId`
//! tie-break. There is no approximation, no randomised structure, no
//! `HashMap`. This is the reference against which every future approximate
//! index will be measured for recall.
//!
//! Storage is a **structure-of-arrays**: a `Vec<ConceptId>` parallel to a
//! contiguous `Vec<SedenionSimd>` of 64-byte-aligned effective vectors — not a
//! `Vec<Vec<f32>>`.

use core::cmp::Ordering;

use scirust_simd::hypercomplex::SedenionSimd;

use crate::error::{HypermemoryError, Result};
use crate::id::ConceptId;
use crate::record::ConceptRecord;
use crate::representation::normalize_array;

/// The similarity metric used for ranking.
///
/// Because every stored effective vector is unit-norm, cosine similarity is the
/// dot product and squared Euclidean distance is `2 − 2·cos`, so the two are
/// **rank-equivalent on Phase 1 data** (a tested property — see
/// `metrics_agree_on_ranking`). Both are provided so the choice is explicit and
/// so later phases (with non-unit vectors) can pick deliberately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimilarityMetric {
    /// Squared Euclidean distance `Σ (qᵢ − vᵢ)²`. Lower is more similar.
    SquaredEuclidean,
    /// Cosine similarity `Σ qᵢ·vᵢ` (a dot product on unit vectors). Higher is
    /// more similar.
    Cosine,
}

impl SimilarityMetric {
    /// Score `query` against `value`, in fixed index order (`0..16`,
    /// left-to-right) for bit-reproducibility.
    #[inline]
    #[must_use]
    pub fn score(self, query: &[f32; 16], value: &[f32; 16]) -> f32 {
        match self
        {
            Self::SquaredEuclidean =>
            {
                let mut acc = 0.0f32;
                for i in 0..16
                {
                    let d = query[i] - value[i];
                    acc += d * d;
                }
                acc
            },
            Self::Cosine =>
            {
                let mut acc = 0.0f32;
                for i in 0..16
                {
                    acc += query[i] * value[i];
                }
                acc
            },
        }
    }

    /// Total order placing the *better* (more similar) score first.
    ///
    /// Uses [`f32::total_cmp`] (a genuine total order — NaN-safe, no
    /// `partial_cmp` unwrap). For [`Self::Cosine`], higher first; for
    /// [`Self::SquaredEuclidean`], lower first.
    #[inline]
    #[must_use]
    pub fn rank_cmp(self, a: f32, b: f32) -> Ordering {
        match self
        {
            // Higher cosine is better → `a` precedes `b` when a > b.
            Self::Cosine => b.total_cmp(&a),
            // Lower distance is better → `a` precedes `b` when a < b.
            Self::SquaredEuclidean => a.total_cmp(&b),
        }
    }
}

/// One search result: a concept id and its score under the active metric.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchHit {
    /// The matched concept.
    pub id: ConceptId,
    /// The metric score (cosine similarity or squared distance).
    pub score: f32,
}

/// The result of one accepted cleanup/denoising step: the recognized concept,
/// its **exact stored** effective code (the denoised vector — a stored
/// prototype, never an interpolation), and the score that justified accepting
/// it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Denoised {
    /// The concept the noisy input was recognized as.
    pub id: ConceptId,
    /// The stored effective vector (unit norm) — the denoised code.
    pub code: SedenionSimd,
    /// The metric score of the noisy input against `code`.
    pub score: f32,
}

/// An exact exhaustive index over effective 16-lane representations.
#[derive(Clone, Debug)]
pub struct S16ExactIndex {
    metric: SimilarityMetric,
    ids: Vec<ConceptId>,
    effective: Vec<SedenionSimd>,
}

impl S16ExactIndex {
    /// A new empty index with the given metric.
    #[must_use]
    pub fn new(metric: SimilarityMetric) -> Self {
        Self {
            metric,
            ids: Vec::new(),
            effective: Vec::new(),
        }
    }

    /// The active metric.
    #[inline]
    #[must_use]
    pub const fn metric(&self) -> SimilarityMetric {
        self.metric
    }

    /// Number of indexed entries.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the index holds no entries.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Whether `id` is present in the index.
    #[must_use]
    pub fn contains(&self, id: ConceptId) -> bool {
        self.ids.contains(&id)
    }

    /// Index a concept by its (already validated, unit-norm) effective vector.
    ///
    /// The record guarantees a finite unit effective vector, so this cannot
    /// introduce a degenerate entry.
    pub fn insert_concept(&mut self, record: &ConceptRecord) {
        self.ids.push(record.id());
        self.effective.push(record.effective());
    }

    /// Refresh the stored effective vector for `record`'s id (Phase 3: after a
    /// residual-learning step). Returns whether the id was present. Only the
    /// addressed entry changes; every other entry is untouched.
    pub fn update_concept(&mut self, record: &ConceptRecord) -> bool {
        if let Some(pos) = self.ids.iter().position(|&x| x == record.id())
        {
            self.effective[pos] = record.effective();
            true
        }
        else
        {
            false
        }
    }

    /// Remove `id` from the index if present; returns whether it was found.
    ///
    /// Uses `swap_remove` (O(1)); results are unaffected because `search` fully
    /// re-sorts with a `ConceptId` tie-break, so the internal order never leaks.
    pub fn remove(&mut self, id: ConceptId) -> bool {
        if let Some(pos) = self.ids.iter().position(|&x| x == id)
        {
            self.ids.swap_remove(pos);
            self.effective.swap_remove(pos);
            true
        }
        else
        {
            false
        }
    }

    /// Rebuild the index from a store snapshot in deterministic slot order.
    pub fn rebuild_from<'a, I>(&mut self, records: I)
    where
        I: IntoIterator<Item = &'a ConceptRecord>,
    {
        self.ids.clear();
        self.effective.clear();
        for record in records
        {
            self.insert_concept(record);
        }
    }

    /// Exact top-`k` most-similar entries to `query`.
    ///
    /// Behaviour on degenerate input is explicit:
    ///
    /// * `k == 0` → `Ok([])` (the caller asked for nothing);
    /// * an empty index → `Ok([])`;
    /// * a zero-norm or non-finite `query` with `k > 0` → typed `Err` (never a
    ///   silently empty result that would conceal invalid input).
    ///
    /// Results are sorted best-first by the metric, with `ConceptId` ascending
    /// as the deterministic tie-break.
    #[must_use = "the search result carries the ranked hits"]
    pub fn search(&self, query: &SedenionSimd, k: usize) -> Result<Vec<SearchHit>> {
        if k == 0
        {
            return Ok(Vec::new());
        }
        // Validate + normalize the query whenever there is a request to satisfy,
        // so an invalid query always errors (even against an empty index).
        let q = normalize_array(&query.to_array())?;
        if self.is_empty()
        {
            return Ok(Vec::new());
        }

        let mut hits: Vec<SearchHit> = self
            .ids
            .iter()
            .zip(self.effective.iter())
            .map(|(&id, v)| SearchHit {
                id,
                score: self.metric.score(&q, &v.to_array()),
            })
            .collect();

        hits.sort_by(|a, b| {
            self.metric
                .rank_cmp(a.score, b.score)
                .then_with(|| a.id.cmp(&b.id))
        });
        hits.truncate(k);
        Ok(hits)
    }

    /// Cleanup/denoising: snap a noisy vector to the **nearest stored
    /// prototype**, if it is convincing enough.
    ///
    /// This is the standard denoising component of vector-symbolic
    /// constructions ("cleanup memory"): after noise, perturbation, or a
    /// composition step, a code is recognized by nearest-neighbour against the
    /// stored concepts and replaced by the *exact stored* effective vector —
    /// never an interpolation, so repeated cleanup is idempotent.
    ///
    /// `threshold` is expressed in the active metric and the match is accepted
    /// iff its score is **at least as good as** `threshold` under that metric's
    /// ordering (cosine: `score ≥ threshold`; squared Euclidean:
    /// `score ≤ threshold`). A below-threshold best match returns `Ok(None)` —
    /// garbage is *rejected*, not silently snapped to an arbitrary concept.
    ///
    /// Errors: a non-finite `threshold` or an invalid noisy input (zero /
    /// non-finite) is a typed error. An empty index returns `Ok(None)`.
    /// Ties between equally-good prototypes resolve to the smallest
    /// [`ConceptId`], as everywhere else.
    #[must_use = "the outcome says whether the input was recognized"]
    pub fn denoise(&self, noisy: &SedenionSimd, threshold: f32) -> Result<Option<Denoised>> {
        if !threshold.is_finite()
        {
            return Err(HypermemoryError::InvalidRepresentation {
                reason: "denoise threshold must be finite",
            });
        }
        let q = normalize_array(&noisy.to_array())?;
        let mut best: Option<(f32, usize)> = None;
        for (pos, v) in self.effective.iter().enumerate()
        {
            let score = self.metric.score(&q, &v.to_array());
            best = match best
            {
                None => Some((score, pos)),
                Some((bs, bp)) =>
                {
                    let better = self.metric.rank_cmp(score, bs).is_lt()
                        || (self.metric.rank_cmp(score, bs).is_eq()
                            && self.ids[pos] < self.ids[bp]);
                    if better
                    {
                        Some((score, pos))
                    }
                    else
                    {
                        Some((bs, bp))
                    }
                },
            };
        }
        let Some((score, pos)) = best
        else
        {
            return Ok(None);
        };
        // Accept iff the score is at least as good as the threshold.
        if self.metric.rank_cmp(score, threshold).is_gt()
        {
            return Ok(None);
        }
        Ok(Some(Denoised {
            id: self.ids[pos],
            code: self.effective[pos],
            score,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{ConceptSpec, S16Store};

    fn build() -> (S16Store, S16ExactIndex) {
        let mut store = S16Store::new();
        let mut index = S16ExactIndex::new(SimilarityMetric::Cosine);
        for i in 0..4u32
        {
            let id = store
                .insert(ConceptSpec::new(
                    vec![i as u8],
                    SedenionSimd::unit(i as usize),
                    1.0,
                    0,
                ))
                .unwrap();
            index.insert_concept(store.get(id).unwrap());
        }
        (store, index)
    }

    #[test]
    fn exact_top1_and_topk() {
        let (_store, index) = build();
        let hits = index.search(&SedenionSimd::unit(2), 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id.slot(), 2);
        assert!((hits[0].score - 1.0).abs() < 1e-6);

        let topk = index.search(&SedenionSimd::unit(2), 3).unwrap();
        assert_eq!(topk.len(), 3);
        assert_eq!(topk[0].id.slot(), 2); // best
    }

    #[test]
    fn k_zero_returns_empty_ok() {
        let (_store, index) = build();
        assert_eq!(index.search(&SedenionSimd::unit(0), 0).unwrap(), Vec::new());
    }

    #[test]
    fn empty_index_returns_empty_ok() {
        let index = S16ExactIndex::new(SimilarityMetric::Cosine);
        assert_eq!(index.search(&SedenionSimd::unit(0), 5).unwrap(), Vec::new());
    }

    #[test]
    fn zero_query_is_typed_error_not_empty() {
        let (_store, index) = build();
        assert!(index.search(&SedenionSimd::ZERO, 3).is_err());
    }

    #[test]
    fn deterministic_tie_break_by_concept_id() {
        // Two identical directions → identical score → lower ConceptId first.
        let mut store = S16Store::new();
        let mut index = S16ExactIndex::new(SimilarityMetric::Cosine);
        let a = store
            .insert(ConceptSpec::new(
                b"a".to_vec(),
                SedenionSimd::unit(0),
                1.0,
                0,
            ))
            .unwrap();
        let b = store
            .insert(ConceptSpec::new(
                b"b".to_vec(),
                SedenionSimd::unit(0).scale(3.0),
                1.0,
                0,
            ))
            .unwrap();
        index.insert_concept(store.get(a).unwrap());
        index.insert_concept(store.get(b).unwrap());
        let hits = index.search(&SedenionSimd::unit(0), 2).unwrap();
        assert_eq!(hits[0].id, a, "tie must break to the smaller ConceptId");
        assert_eq!(hits[1].id, b);
        assert!((hits[0].score - hits[1].score).abs() < 1e-6);
    }

    #[test]
    fn removed_entries_are_not_returned() {
        let (store, mut index) = build();
        let victim = store.ids().nth(2).unwrap();
        assert!(index.remove(victim));
        let hits = index.search(&SedenionSimd::unit(2), 4).unwrap();
        assert!(hits.iter().all(|h| h.id != victim));
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn metrics_agree_on_ranking() {
        // On unit vectors, cosine-desc and squared-euclidean-asc rank identically.
        let mut store = S16Store::new();
        let mut cos = S16ExactIndex::new(SimilarityMetric::Cosine);
        let mut euc = S16ExactIndex::new(SimilarityMetric::SquaredEuclidean);
        for i in 0..6u32
        {
            let anchor =
                SedenionSimd::unit(0).scale((i + 1) as f32) + SedenionSimd::unit(1).scale(i as f32);
            let id = store
                .insert(ConceptSpec::new(vec![i as u8], anchor, 1.0, 0))
                .unwrap();
            cos.insert_concept(store.get(id).unwrap());
            euc.insert_concept(store.get(id).unwrap());
        }
        let q = SedenionSimd::unit(0);
        let cos_order: Vec<_> = cos.search(&q, 6).unwrap().iter().map(|h| h.id).collect();
        let euc_order: Vec<_> = euc.search(&q, 6).unwrap().iter().map(|h| h.id).collect();
        assert_eq!(cos_order, euc_order);
    }

    #[test]
    fn search_is_independent_of_insertion_order() {
        // Determinism: same corpus, different index build order → same ranking.
        let mut store = S16Store::new();
        let mut ids = Vec::new();
        for i in 0..5u32
        {
            let anchor = SedenionSimd::unit((i % 4) as usize).scale((i + 1) as f32);
            ids.push(
                store
                    .insert(ConceptSpec::new(vec![i as u8], anchor, 1.0, 0))
                    .unwrap(),
            );
        }
        let mut fwd = S16ExactIndex::new(SimilarityMetric::Cosine);
        for &id in &ids
        {
            fwd.insert_concept(store.get(id).unwrap());
        }
        let mut rev = S16ExactIndex::new(SimilarityMetric::Cosine);
        for &id in ids.iter().rev()
        {
            rev.insert_concept(store.get(id).unwrap());
        }
        let q = SedenionSimd::unit(1);
        assert_eq!(fwd.search(&q, 5).unwrap(), rev.search(&q, 5).unwrap());
    }
}
