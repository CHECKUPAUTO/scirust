//! The real-vector baseline.
//!
//! `Real16Index` is `S16ExactIndex`'s twin over `[f32; 16]` interpreted as a
//! plain real vector — same ids, same effective components, same metric, same
//! tie-break, same corpus order. Its purpose is to answer, honestly: does
//! routing the 16 components through the sedenion algebra buy anything for
//! *retrieval*? The tests show the two indexes return bit-identical rankings
//! (falsification criterion F1), because similarity is a lane-wise operation
//! that never touches sedenion multiplication.

use scirust_simd::hypercomplex::SedenionSimd;

use crate::error::Result;
use crate::id::ConceptId;
use crate::index::{SearchHit, SimilarityMetric};
use crate::record::ConceptRecord;
use crate::representation::normalize_array;

/// An exact exhaustive real-vector index over the same 16 effective components.
#[derive(Clone, Debug)]
pub struct Real16Index {
    metric: SimilarityMetric,
    ids: Vec<ConceptId>,
    vectors: Vec<[f32; 16]>,
}

impl Real16Index {
    /// A new empty baseline index with the given metric.
    #[must_use]
    pub fn new(metric: SimilarityMetric) -> Self {
        Self {
            metric,
            ids: Vec::new(),
            vectors: Vec::new(),
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

    /// Whether `id` is present.
    #[must_use]
    pub fn contains(&self, id: ConceptId) -> bool {
        self.ids.contains(&id)
    }

    /// Index a concept by the same effective components the sedenion index uses.
    pub fn insert_concept(&mut self, record: &ConceptRecord) {
        self.ids.push(record.id());
        self.vectors.push(record.effective().to_array());
    }

    /// Remove `id` if present; returns whether it was found.
    pub fn remove(&mut self, id: ConceptId) -> bool {
        if let Some(pos) = self.ids.iter().position(|&x| x == id)
        {
            self.ids.swap_remove(pos);
            self.vectors.swap_remove(pos);
            true
        }
        else
        {
            false
        }
    }

    /// Rebuild from a store snapshot in deterministic order.
    pub fn rebuild_from<'a, I>(&mut self, records: I)
    where
        I: IntoIterator<Item = &'a ConceptRecord>,
    {
        self.ids.clear();
        self.vectors.clear();
        for record in records
        {
            self.insert_concept(record);
        }
    }

    /// Exact top-`k`, with identical semantics and identical degenerate-input
    /// handling to [`crate::S16ExactIndex::search`]. Accepts the query as a
    /// [`SedenionSimd`] so the *same* query value can be handed to both indexes.
    #[must_use = "the search result carries the ranked hits"]
    pub fn search(&self, query: &SedenionSimd, k: usize) -> Result<Vec<SearchHit>> {
        if k == 0
        {
            return Ok(Vec::new());
        }
        let q = normalize_array(&query.to_array())?;
        if self.is_empty()
        {
            return Ok(Vec::new());
        }
        let mut hits: Vec<SearchHit> = self
            .ids
            .iter()
            .zip(self.vectors.iter())
            .map(|(&id, v)| SearchHit {
                id,
                score: self.metric.score(&q, v),
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::S16ExactIndex;
    use crate::store::{ConceptSpec, S16Store};

    fn corpus(metric: SimilarityMetric) -> (S16ExactIndex, Real16Index, Vec<SedenionSimd>) {
        let mut store = S16Store::new();
        let mut sed = S16ExactIndex::new(metric);
        let mut real = Real16Index::new(metric);
        let mut queries = Vec::new();
        for i in 0..32u32
        {
            // A deterministic mix across several basis directions.
            let anchor = SedenionSimd::unit((i % 16) as usize).scale(1.0 + (i % 5) as f32)
                + SedenionSimd::unit(((i + 3) % 16) as usize).scale((i % 3) as f32);
            let id = store
                .insert(ConceptSpec::new(vec![i as u8], anchor, 1.0, 0))
                .unwrap();
            sed.insert_concept(store.get(id).unwrap());
            real.insert_concept(store.get(id).unwrap());
            if i % 4 == 0
            {
                queries.push(SedenionSimd::unit((i % 16) as usize));
            }
        }
        (sed, real, queries)
    }

    #[test]
    fn sedenion_index_matches_real16_baseline_bitwise_cosine() {
        let (sed, real, queries) = corpus(SimilarityMetric::Cosine);
        for q in &queries
        {
            // Identical ids AND identical scores — the algebra adds nothing to
            // retrieval (F1).
            assert_eq!(sed.search(q, 8).unwrap(), real.search(q, 8).unwrap());
        }
    }

    #[test]
    fn sedenion_index_matches_real16_baseline_bitwise_euclidean() {
        let (sed, real, queries) = corpus(SimilarityMetric::SquaredEuclidean);
        for q in &queries
        {
            assert_eq!(sed.search(q, 8).unwrap(), real.search(q, 8).unwrap());
        }
    }

    #[test]
    fn baseline_degenerate_inputs_match_index() {
        let (_sed, real, _q) = corpus(SimilarityMetric::Cosine);
        assert_eq!(real.search(&SedenionSimd::unit(0), 0).unwrap(), Vec::new());
        assert!(real.search(&SedenionSimd::ZERO, 4).is_err());
    }
}
