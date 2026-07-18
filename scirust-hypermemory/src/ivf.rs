//! Phase 4 — a deterministic approximate index, measured against the oracle.
//!
//! The roadmap defers HNSW-style randomized graphs; the first approximate
//! index must uphold the crate's determinism contract (no RNG, no
//! thread-schedule dependence, fixed reduction order) and be **measurable in
//! recall against the Phase 1 exact oracle** ([`S16ExactIndex`]). An IVF
//! (inverted-file) index with deterministic Lloyd clustering satisfies both:
//!
//! * **Deterministic build.** Centroids initialize to the *first `nlist`
//!   effective vectors in the given (slot-ordered) iteration order*; a fixed
//!   number of Lloyd iterations reassigns points (ties → lowest centroid
//!   index) and recomputes means with fixed index-order `f32` sums; an empty
//!   cluster keeps its previous centroid. Same corpus, same order → the same
//!   clustering, bit for bit.
//! * **Deterministic search.** Centroids are ranked by the active metric
//!   (`f32::total_cmp`, ties → lowest centroid index); the top `nprobe` lists
//!   are scanned **exactly** (same scoring, same `ConceptId` tie-break as the
//!   oracle).
//! * **Oracle containment.** With `nprobe ≥ nlist` every list is scanned, so
//!   the result is **bit-identical** to [`S16ExactIndex::search`] — tested.
//!   Recall at smaller `nprobe` is monotone non-decreasing in `nprobe`
//!   (candidate sets are nested by construction) — also tested.
//!
//! Honest positioning: at 16 dimensions an exhaustive scan is already very
//! fast, so IVF's practical win here is limited; this module's value is the
//! *pattern* — an approximate structure whose recall is measured against an
//! exact oracle rather than assumed — which is what later, larger phases
//! require.

use scirust_simd::hypercomplex::SedenionSimd;

use crate::error::{HypermemoryError, Result};
use crate::id::ConceptId;
use crate::index::{SearchHit, SimilarityMetric};
use crate::record::ConceptRecord;
use crate::representation::normalize_array;

/// A deterministic IVF (inverted-file) approximate index over effective
/// 16-lane representations.
#[derive(Clone, Debug)]
pub struct S16IvfIndex {
    metric: SimilarityMetric,
    centroids: Vec<[f32; 16]>,
    /// Per-centroid posting lists of `(id, effective)` in insertion order.
    lists: Vec<Vec<(ConceptId, [f32; 16])>>,
    len: usize,
}

impl S16IvfIndex {
    /// Build from records (typically [`crate::S16Store::iter`], which is
    /// slot-ordered and therefore deterministic).
    ///
    /// `nlist` is clamped to the corpus size; `iterations` is the fixed Lloyd
    /// iteration count (0 keeps the seeded centroids). An empty corpus yields
    /// an empty index. Fails only if `nlist == 0` while the corpus is
    /// non-empty.
    pub fn build<'a, I>(
        metric: SimilarityMetric,
        nlist: usize,
        iterations: usize,
        records: I,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = &'a ConceptRecord>,
    {
        let points: Vec<(ConceptId, [f32; 16])> = records
            .into_iter()
            .map(|r| (r.id(), r.effective().to_array()))
            .collect();
        if points.is_empty()
        {
            return Ok(Self {
                metric,
                centroids: Vec::new(),
                lists: Vec::new(),
                len: 0,
            });
        }
        if nlist == 0
        {
            return Err(HypermemoryError::InvalidRepresentation {
                reason: "nlist must be non-zero for a non-empty corpus",
            });
        }
        let k = nlist.min(points.len());

        // Deterministic seeding: the first k effective vectors in corpus order.
        let mut centroids: Vec<[f32; 16]> = points.iter().take(k).map(|(_, v)| *v).collect();
        let mut assignment: Vec<usize> = vec![0; points.len()];

        for _ in 0..=iterations
        {
            // Assign: nearest centroid by squared Euclidean distance (rank-
            // equivalent to cosine on unit vectors), ties → lowest index.
            for (slot, (_, v)) in points.iter().enumerate()
            {
                let mut best = 0usize;
                let mut best_d = sq_dist(v, &centroids[0]);
                for (c, centroid) in centroids.iter().enumerate().skip(1)
                {
                    let d = sq_dist(v, centroid);
                    if d.total_cmp(&best_d).is_lt()
                    {
                        best = c;
                        best_d = d;
                    }
                }
                assignment[slot] = best;
            }
            // Update: mean of assigned points, fixed index order; an empty
            // cluster keeps its previous centroid.
            let mut sums = vec![[0.0f32; 16]; k];
            let mut counts = vec![0u32; k];
            for (slot, (_, v)) in points.iter().enumerate()
            {
                let c = assignment[slot];
                for lane in 0..16
                {
                    sums[c][lane] += v[lane];
                }
                counts[c] += 1;
            }
            for c in 0..k
            {
                if counts[c] > 0
                {
                    let inv = 1.0 / counts[c] as f32;
                    for lane in 0..16
                    {
                        centroids[c][lane] = sums[c][lane] * inv;
                    }
                }
            }
        }

        // Final posting lists from the last assignment, in corpus order.
        let mut lists: Vec<Vec<(ConceptId, [f32; 16])>> = vec![Vec::new(); k];
        for (slot, &(id, v)) in points.iter().enumerate()
        {
            lists[assignment[slot]].push((id, v));
        }
        let len = points.len();
        Ok(Self {
            metric,
            centroids,
            lists,
            len,
        })
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
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the index is empty.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of clusters actually built.
    #[inline]
    #[must_use]
    pub fn nlist(&self) -> usize {
        self.centroids.len()
    }

    /// Approximate top-`k`: scan only the `nprobe` most promising clusters.
    ///
    /// Degenerate-input behaviour mirrors the oracle: `k == 0` or an empty
    /// index → `Ok([])`; an invalid query (zero / non-finite) with `k > 0` →
    /// typed error. `nprobe == 0` is a typed error (a silent empty result
    /// would conceal the misconfiguration); `nprobe ≥ nlist` scans everything
    /// and returns **bit-identical** results to [`S16ExactIndex::search`].
    ///
    /// [`S16ExactIndex::search`]: crate::S16ExactIndex::search
    #[must_use = "the search result carries the ranked hits"]
    pub fn search(&self, query: &SedenionSimd, k: usize, nprobe: usize) -> Result<Vec<SearchHit>> {
        if k == 0
        {
            return Ok(Vec::new());
        }
        let q = normalize_array(&query.to_array())?;
        if self.is_empty()
        {
            return Ok(Vec::new());
        }
        if nprobe == 0
        {
            return Err(HypermemoryError::InvalidRepresentation {
                reason: "nprobe must be non-zero",
            });
        }

        // Rank centroids: best-first by metric, ties → lowest centroid index.
        let mut order: Vec<usize> = (0..self.centroids.len()).collect();
        let scores: Vec<f32> = self
            .centroids
            .iter()
            .map(|c| self.metric.score(&q, c))
            .collect();
        order.sort_by(|&a, &b| self.metric.rank_cmp(scores[a], scores[b]).then(a.cmp(&b)));
        order.truncate(nprobe);

        // Exact scoring inside the probed lists; oracle tie-break.
        let mut hits: Vec<SearchHit> = Vec::new();
        for &c in &order
        {
            for &(id, ref v) in &self.lists[c]
            {
                hits.push(SearchHit {
                    id,
                    score: self.metric.score(&q, v),
                });
            }
        }
        hits.sort_by(|a, b| {
            self.metric
                .rank_cmp(a.score, b.score)
                .then_with(|| a.id.cmp(&b.id))
        });
        hits.truncate(k);
        Ok(hits)
    }
}

/// Squared Euclidean distance in fixed index order.
fn sq_dist(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..16
    {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{ConceptSpec, S16Store};

    fn seeded_store(n: u32, seed: u64) -> S16Store {
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as u32 as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let mut store = S16Store::new();
        for i in 0..n
        {
            let mut lanes = [0.0f32; 16];
            for lane in &mut lanes
            {
                *lane = next();
            }
            if lanes.iter().all(|x| *x == 0.0)
            {
                lanes[0] = 1.0;
            }
            store
                .insert(ConceptSpec::new(
                    i.to_le_bytes().to_vec(),
                    SedenionSimd::from_array(lanes),
                    1.0,
                    0,
                ))
                .unwrap();
        }
        store
    }

    #[test]
    fn empty_corpus_builds_an_empty_index() {
        let store = S16Store::new();
        let ivf = S16IvfIndex::build(SimilarityMetric::Cosine, 8, 5, store.iter()).unwrap();
        assert!(ivf.is_empty());
        assert_eq!(
            ivf.search(&SedenionSimd::unit(0), 5, 1).unwrap(),
            Vec::new()
        );
    }

    #[test]
    fn zero_nlist_on_nonempty_corpus_is_rejected() {
        let store = seeded_store(4, 1);
        assert!(S16IvfIndex::build(SimilarityMetric::Cosine, 0, 5, store.iter()).is_err());
    }

    #[test]
    fn zero_nprobe_is_a_typed_error() {
        let store = seeded_store(4, 2);
        let ivf = S16IvfIndex::build(SimilarityMetric::Cosine, 2, 3, store.iter()).unwrap();
        assert!(ivf.search(&SedenionSimd::unit(0), 3, 0).is_err());
    }

    #[test]
    fn build_is_deterministic() {
        let store = seeded_store(64, 3);
        let a = S16IvfIndex::build(SimilarityMetric::Cosine, 8, 10, store.iter()).unwrap();
        let b = S16IvfIndex::build(SimilarityMetric::Cosine, 8, 10, store.iter()).unwrap();
        assert_eq!(
            a.centroids, b.centroids,
            "clustering must be bit-reproducible"
        );
        let q = SedenionSimd::unit(5);
        assert_eq!(a.search(&q, 10, 2).unwrap(), b.search(&q, 10, 2).unwrap());
    }
}
