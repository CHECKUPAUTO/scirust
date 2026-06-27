//! Exact dense index: brute-force top-k by cosine similarity.
//!
//! Vectors are stored L2-normalised, so a query scores against each document by a
//! single dot product. `search` returns the **exact** top-k — no approximation,
//! no randomised structure — which makes every ranking deterministic and
//! auditable (the property that distinguishes this from a stochastic RAG stage).

use crate::vector;
use crate::{RetrievalError, Scored};
use serde::{Deserialize, Serialize};

/// A flat exact dense-retrieval index over `dim`-dimensional embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseIndex {
    dim: usize,
    ids: Vec<u64>,
    normed: Vec<Vec<f32>>,
}

impl DenseIndex {
    /// New empty index for `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
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

    /// Add a document `vector` under `id`. The vector is L2-normalised on the way
    /// in. Returns [`RetrievalError::DimMismatch`] if its length is wrong.
    pub fn add(&mut self, id: u64, vector: &[f32]) -> Result<(), RetrievalError> {
        if vector.len() != self.dim
        {
            return Err(RetrievalError::DimMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        self.ids.push(id);
        self.normed.push(vector::normalized(vector));
        Ok(())
    }

    /// Exact top-`k` documents by cosine similarity to `query`, sorted by score
    /// descending with a deterministic id-ascending tie-break. Returns an empty
    /// vector if the index is empty, `k == 0`, or the query dimension is wrong.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<Scored> {
        if k == 0 || self.is_empty() || query.len() != self.dim
        {
            return Vec::new();
        }
        let q = vector::normalized(query);
        let mut scored: Vec<Scored> = self
            .ids
            .iter()
            .zip(&self.normed)
            .map(|(&id, v)| Scored {
                id,
                score: vector::dot(&q, v),
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

    #[test]
    fn search_returns_exact_topk_in_similarity_order() {
        let mut idx = DenseIndex::new(2);
        idx.add(10, &[1.0, 0.0]).unwrap(); // cos with [1,0] = 1.000
        idx.add(20, &[0.0, 1.0]).unwrap(); // cos = 0.000
        idx.add(30, &[0.9, 0.1]).unwrap(); // cos = 0.9939 (0.9/√0.82)
        let hits = idx.search(&[1.0, 0.0], 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, 10);
        assert!(
            (hits[0].score - 1.0).abs() < 1e-6,
            "score {}",
            hits[0].score
        );
        assert_eq!(hits[1].id, 30);
        // 0.9 / sqrt(0.81 + 0.01) = 0.9 / sqrt(0.82) = 0.993884...
        assert!(
            (hits[1].score - 0.993_883_7).abs() < 1e-5,
            "score {}",
            hits[1].score
        );
    }

    #[test]
    fn ties_break_by_ascending_id() {
        let mut idx = DenseIndex::new(2);
        // Two documents with identical direction -> identical score; lower id wins.
        idx.add(42, &[1.0, 0.0]).unwrap();
        idx.add(7, &[2.0, 0.0]).unwrap();
        let hits = idx.search(&[1.0, 0.0], 2);
        assert_eq!(hits[0].id, 7, "tie must go to the smaller id");
        assert_eq!(hits[1].id, 42);
        assert!((hits[0].score - hits[1].score).abs() < 1e-6);
    }

    #[test]
    fn degenerate_queries_return_empty() {
        let mut idx = DenseIndex::new(3);
        idx.add(1, &[1.0, 0.0, 0.0]).unwrap();
        assert!(idx.search(&[1.0, 0.0, 0.0], 0).is_empty()); // k = 0
        assert!(idx.search(&[1.0, 0.0], 5).is_empty()); // wrong dim
        assert!(DenseIndex::new(3).search(&[1.0, 0.0, 0.0], 5).is_empty()); // empty index
    }

    #[test]
    fn dimension_mismatch_is_reported() {
        let mut idx = DenseIndex::new(4);
        assert_eq!(
            idx.add(1, &[1.0, 2.0]),
            Err(RetrievalError::DimMismatch {
                expected: 4,
                got: 2
            })
        );
        assert!(idx.is_empty());
    }

    #[test]
    fn k_larger_than_corpus_returns_all() {
        let mut idx = DenseIndex::new(2);
        idx.add(1, &[1.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0]).unwrap();
        assert_eq!(idx.search(&[1.0, 1.0], 10).len(), 2);
    }
}
