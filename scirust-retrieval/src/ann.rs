//! Approximate nearest-neighbour index via random-hyperplane LSH (SimHash).
//!
//! The exact [`crate::DenseIndex`] scans every document per query — `O(N·d)`,
//! which does not scale to large corpora. This index instead buckets documents by
//! the **sign of random projections**: for cosine similarity, the probability that
//! two vectors land on the same side of a random hyperplane is `1 − θ/π`, so
//! cosine-near vectors collide in the hash and a query only needs to **exactly
//! score the documents in its buckets** — a sub-linear candidate set — then
//! re-rank them by exact cosine.
//!
//! It trades a little recall for speed, and [`crate::DenseIndex`] is its
//! **correctness oracle**: every returned score is the *exact* cosine (an LSH hit
//! is always verified), and recall against the exact top-k is measurable and
//! tunable via the number of tables. Determinism: the hyperplanes are seeded and
//! every reduction is fixed-order `f32`, so the index is bit-reproducible.

use crate::vector;
use crate::{RetrievalError, Scored};
use scirust_core::nn::PcgEngine;
use std::collections::HashMap;

/// A random-hyperplane LSH index over `dim`-dimensional embeddings.
///
/// `n_tables` independent hash tables, each with `n_bits` hyperplanes, give the
/// recall/speed trade-off: more tables ⇒ higher recall (more chances to collide),
/// more bits ⇒ smaller, more selective buckets.
pub struct LshIndex {
    dim: usize,
    n_tables: usize,
    n_bits: usize,
    // n_tables * n_bits hyperplanes, each `dim` long, laid out contiguously.
    planes: Vec<f32>,
    tables: Vec<HashMap<u64, Vec<usize>>>,
    ids: Vec<u64>,
    normed: Vec<Vec<f32>>,
}

impl LshIndex {
    /// New empty index. `n_bits` must be in `1..=64`; the hyperplanes are drawn
    /// deterministically from `seed`.
    pub fn new(dim: usize, n_tables: usize, n_bits: usize, seed: u64) -> Self {
        assert!(dim >= 1, "LshIndex: dim must be >= 1");
        assert!(
            (1..=64).contains(&n_bits),
            "LshIndex: n_bits must be in 1..=64"
        );
        assert!(n_tables >= 1, "LshIndex: n_tables must be >= 1");
        let mut rng = PcgEngine::new(seed);
        let mut planes = vec![0.0f32; n_tables * n_bits * dim];
        for p in planes.iter_mut()
        {
            *p = rng.float_signed();
        }
        Self {
            dim,
            n_tables,
            n_bits,
            planes,
            tables: vec![HashMap::new(); n_tables],
            ids: Vec::new(),
            normed: Vec::new(),
        }
    }

    /// The embedding dimension this index expects.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the index holds no documents.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The `n_bits`-bit SimHash of `v` in table `t`: bit `i` is set iff `v` is on
    /// the positive side of hyperplane `i`. Sign of a dot product is scale-free,
    /// so the caller need not normalise for the hash to be consistent.
    fn hash(&self, t: usize, v: &[f32]) -> u64 {
        let mut key = 0u64;
        for bit in 0..self.n_bits
        {
            let base = (t * self.n_bits + bit) * self.dim;
            let plane = &self.planes[base..base + self.dim];
            if vector::dot(v, plane) >= 0.0
            {
                key |= 1u64 << bit;
            }
        }
        key
    }

    /// Add a document `vector` under `id`; it is L2-normalised and bucketed into
    /// every table. [`RetrievalError::DimMismatch`] if its length is wrong.
    pub fn add(&mut self, id: u64, vector: &[f32]) -> Result<(), RetrievalError> {
        if vector.len() != self.dim
        {
            return Err(RetrievalError::DimMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        let idx = self.ids.len();
        let nv = vector::normalized(vector);
        for t in 0..self.n_tables
        {
            let key = self.hash(t, &nv);
            self.tables[t].entry(key).or_default().push(idx);
        }
        self.ids.push(id);
        self.normed.push(nv);
        Ok(())
    }

    /// Candidate document indices for `query`: the union of its bucket in every
    /// table, de-duplicated, in first-seen (deterministic) order.
    fn candidates(&self, query_normed: &[f32]) -> Vec<usize> {
        let mut seen = vec![false; self.ids.len()];
        let mut cands = Vec::new();
        for t in 0..self.n_tables
        {
            let key = self.hash(t, query_normed);
            if let Some(bucket) = self.tables[t].get(&key)
            {
                for &idx in bucket
                {
                    if !seen[idx]
                    {
                        seen[idx] = true;
                        cands.push(idx);
                    }
                }
            }
        }
        cands
    }

    /// Approximate top-`k` by cosine: gather LSH candidates, then **exactly**
    /// score and rank them (score descending, id ascending — the same ordering as
    /// [`crate::DenseIndex`]). Returns empty if the index is empty, `k == 0`, or
    /// the query dimension is wrong.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<Scored> {
        if k == 0 || self.is_empty() || query.len() != self.dim
        {
            return Vec::new();
        }
        let q = vector::normalized(query);
        let mut scored: Vec<Scored> = self
            .candidates(&q)
            .into_iter()
            .map(|idx| Scored {
                id: self.ids[idx],
                score: vector::dot(&q, &self.normed[idx]),
            })
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.id.cmp(&b.id))
        });
        scored.truncate(k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DenseIndex;

    // Deterministic pseudo-random corpus and queries.
    fn rng_vec(rng: &mut PcgEngine, dim: usize) -> Vec<f32> {
        (0..dim).map(|_| rng.float_signed()).collect()
    }

    #[test]
    fn a_query_equal_to_a_document_is_found_first_with_exact_score() {
        // An identical vector hashes identically in every table, so it always
        // shares the document's bucket → it is a guaranteed candidate, and the
        // exact re-rank gives it cosine 1.0.
        let mut rng = PcgEngine::new(1);
        let dim = 16;
        let mut ann = LshIndex::new(dim, 8, 8, 42);
        let mut docs = Vec::new();
        for j in 0..20u64
        {
            let v = rng_vec(&mut rng, dim);
            ann.add(j, &v).unwrap();
            docs.push(v);
        }
        for (j, v) in docs.iter().enumerate()
        {
            let hits = ann.search(v, 1);
            assert_eq!(hits[0].id, j as u64, "doc {j} should retrieve itself");
            assert!(
                (hits[0].score - 1.0).abs() < 1e-5,
                "score {}",
                hits[0].score
            );
        }
    }

    #[test]
    fn returned_scores_equal_the_exact_index_scores() {
        // Whatever the ANN returns, its scores must match DenseIndex (the oracle)
        // for the same ids — the LSH only filters candidates, the score is exact.
        let mut rng = PcgEngine::new(7);
        let dim = 24;
        let mut ann = LshIndex::new(dim, 10, 10, 99);
        let mut exact = DenseIndex::new(dim);
        let mut queries = Vec::new();
        for j in 0..50u64
        {
            let v = rng_vec(&mut rng, dim);
            ann.add(j, &v).unwrap();
            exact.add(j, &v).unwrap();
            if j < 8
            {
                queries.push(v);
            }
        }
        for q in &queries
        {
            for hit in ann.search(q, 5)
            {
                // DenseIndex score for the same id (it ranks all docs).
                let exact_score = exact
                    .search(q, exact.len())
                    .into_iter()
                    .find(|s| s.id == hit.id)
                    .map(|s| s.score)
                    .unwrap();
                assert!(
                    (hit.score - exact_score).abs() < 1e-6,
                    "ann score {} != exact {} for id {}",
                    hit.score,
                    exact_score,
                    hit.id
                );
            }
        }
    }

    #[test]
    fn recall_against_the_exact_oracle_is_high_on_clustered_data() {
        // Recall is only meaningful where true neighbours EXIST: random high-dim
        // vectors are near-orthogonal, so their "top-k" are noise that no ANN can
        // recover. Real embeddings cluster, so we build C clusters of near-
        // duplicate docs; a query near a centre has its true top-k inside its
        // cluster, and the LSH must recover them. Measured vs the exact oracle,
        // deterministic (seeded), so the asserted floor is honest.
        let mut rng = PcgEngine::new(2024);
        let dim = 32;
        let n_clusters = 12usize;
        let per_cluster = 5usize;
        let k = 5usize;
        let mut ann = LshIndex::new(dim, 16, 6, 123);
        let mut exact = DenseIndex::new(dim);
        let mut centers = Vec::new();
        let mut id = 0u64;
        for _ in 0..n_clusters
        {
            let center = rng_vec(&mut rng, dim);
            for _ in 0..per_cluster
            {
                let doc: Vec<f32> = center
                    .iter()
                    .map(|&x| x + 0.1 * rng.float_signed())
                    .collect();
                ann.add(id, &doc).unwrap();
                exact.add(id, &doc).unwrap();
                id += 1;
            }
            centers.push(center);
        }
        // One query per cluster: centre + small noise; its exact top-k are its
        // own cluster's docs, and recall is the fraction the ANN also returns.
        let mut hits = 0usize;
        let mut total = 0usize;
        for center in &centers
        {
            let q: Vec<f32> = center
                .iter()
                .map(|&x| x + 0.1 * rng.float_signed())
                .collect();
            let exact_ids: std::collections::HashSet<u64> =
                exact.search(&q, k).into_iter().map(|s| s.id).collect();
            let ann_ids: std::collections::HashSet<u64> =
                ann.search(&q, k).into_iter().map(|s| s.id).collect();
            hits += exact_ids.intersection(&ann_ids).count();
            total += exact_ids.len();
        }
        let recall = hits as f64 / total as f64;
        assert!(
            recall >= 0.9,
            "ANN Recall@{k} vs exact = {recall}, want >= 0.9 on clustered data"
        );
    }

    #[test]
    fn degenerate_queries_return_empty() {
        let mut ann = LshIndex::new(4, 4, 6, 0);
        ann.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(ann.search(&[1.0, 0.0, 0.0, 0.0], 0).is_empty()); // k = 0
        assert!(ann.search(&[1.0, 0.0], 5).is_empty()); // wrong dim
        assert!(
            LshIndex::new(4, 4, 6, 0)
                .search(&[1.0, 0.0, 0.0, 0.0], 5)
                .is_empty()
        ); // empty
    }

    #[test]
    fn dimension_mismatch_is_reported() {
        let mut ann = LshIndex::new(4, 4, 6, 0);
        assert_eq!(
            ann.add(1, &[1.0, 2.0]),
            Err(RetrievalError::DimMismatch {
                expected: 4,
                got: 2
            })
        );
        assert!(ann.is_empty());
    }
}
