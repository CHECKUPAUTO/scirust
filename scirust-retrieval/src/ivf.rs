//! Deterministic IVF (inverted-file) index, measured against the exact oracle.
//!
//! The generalisation of `scirust-hypermemory`'s Phase 4 pattern to arbitrary
//! dimension: an **approximate** index built to the crate's determinism
//! contract — no RNG anywhere — whose recall is *measured* against the exact
//! [`DenseIndex`](crate::DenseIndex) oracle rather than trusted on faith
//! (`tests/ivf_recall.rs`).
//!
//! Determinism, concretely:
//!
//! * centroid seeding = the first `nlist` vectors in insertion order (no
//!   random init);
//! * Lloyd runs a **fixed** number of iterations; assignment ties go to the
//!   lowest centroid index; an empty cluster keeps its previous centroid;
//!   all reductions accumulate in fixed index order;
//! * search ranks centroids once (`f32::total_cmp`, ties → lowest index),
//!   scans the top `nprobe` posting lists **exactly**, and sorts candidates
//!   with the oracle's comparator.
//!
//! Two properties follow by construction and are pinned by tests:
//! `nprobe = nlist` reproduces [`DenseIndex`](crate::DenseIndex)
//! **bit-for-bit**, and recall is monotone non-decreasing in `nprobe`
//! (candidate sets are nested because the centroid ranking does not depend on
//! `nprobe`).

use crate::vector;
use crate::{RetrievalError, Scored};

fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (x, y) in a.iter().zip(b)
    {
        let d = x - y;
        acc += d * d;
    }
    acc
}

/// Index of the nearest centroid by squared Euclidean distance, ties to the
/// lowest index. `centroids` must be non-empty.
fn nearest_centroid(centroids: &[Vec<f32>], v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_d = squared_euclidean(&centroids[0], v);
    for (c, centroid) in centroids.iter().enumerate().skip(1)
    {
        let d = squared_euclidean(centroid, v);
        if d.total_cmp(&best_d).is_lt()
        {
            best = c;
            best_d = d;
        }
    }
    best
}

/// A deterministic inverted-file index over `dim`-dimensional embeddings,
/// scored by cosine exactly like [`DenseIndex`](crate::DenseIndex).
#[derive(Debug, Clone)]
pub struct IvfIndex {
    dim: usize,
    centroids: Vec<Vec<f32>>,
    lists: Vec<Vec<usize>>,
    ids: Vec<u64>,
    normed: Vec<Vec<f32>>,
}

impl IvfIndex {
    /// Build the index over `entries` with `nlist` clusters refined by
    /// `iterations` Lloyd rounds (0 keeps the seeded centroids).
    ///
    /// `nlist` is clamped to the corpus size. Errors:
    /// [`RetrievalError::InvalidParameter`] for `nlist == 0` with a non-empty
    /// corpus, [`RetrievalError::DimMismatch`] for a wrongly sized vector.
    /// An empty `entries` builds an empty index.
    pub fn build(
        dim: usize,
        nlist: usize,
        iterations: usize,
        entries: &[(u64, Vec<f32>)],
    ) -> Result<Self, RetrievalError> {
        if entries.is_empty()
        {
            return Ok(Self {
                dim,
                centroids: Vec::new(),
                lists: Vec::new(),
                ids: Vec::new(),
                normed: Vec::new(),
            });
        }
        if nlist == 0
        {
            return Err(RetrievalError::InvalidParameter {
                reason: "nlist must be nonzero for a non-empty corpus",
            });
        }
        let mut ids = Vec::with_capacity(entries.len());
        let mut normed = Vec::with_capacity(entries.len());
        for (id, v) in entries
        {
            if v.len() != dim
            {
                return Err(RetrievalError::DimMismatch {
                    expected: dim,
                    got: v.len(),
                });
            }
            ids.push(*id);
            normed.push(vector::normalized(v));
        }

        // Deterministic seeding: the first `nlist` vectors in insertion order.
        let nlist = nlist.min(normed.len());
        let mut centroids: Vec<Vec<f32>> = normed[..nlist].to_vec();

        let mut assignment = vec![0usize; normed.len()];
        for _ in 0..iterations
        {
            for (a, v) in assignment.iter_mut().zip(&normed)
            {
                *a = nearest_centroid(&centroids, v);
            }
            // Recompute each centroid as the fixed-order mean of its members;
            // an empty cluster keeps its previous centroid.
            let mut sums = vec![vec![0.0f32; dim]; nlist];
            let mut counts = vec![0usize; nlist];
            for (&a, v) in assignment.iter().zip(&normed)
            {
                counts[a] += 1;
                for (s, &x) in sums[a].iter_mut().zip(v)
                {
                    *s += x;
                }
            }
            for ((centroid, sum), &count) in centroids.iter_mut().zip(&sums).zip(&counts)
            {
                if count > 0
                {
                    let inv = 1.0 / count as f32;
                    for (c, &s) in centroid.iter_mut().zip(sum)
                    {
                        *c = s * inv;
                    }
                }
            }
        }

        // Final assignment pass fills the posting lists in insertion order.
        let mut lists = vec![Vec::new(); nlist];
        for (i, v) in normed.iter().enumerate()
        {
            lists[nearest_centroid(&centroids, v)].push(i);
        }

        Ok(Self {
            dim,
            centroids,
            lists,
            ids,
            normed,
        })
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

    /// Number of clusters actually built (`nlist` clamped to the corpus size;
    /// 0 for an empty index).
    pub fn nlist(&self) -> usize {
        self.centroids.len()
    }

    /// Top-`k` documents by cosine similarity among the `nprobe` posting lists
    /// whose centroids are nearest the query, sorted exactly like
    /// [`DenseIndex::search`](crate::DenseIndex::search) (score descending,
    /// ties → ascending id). With `nprobe >= nlist` the result is
    /// **bit-identical** to the exact oracle. Returns an empty vector for
    /// `k == 0`, `nprobe == 0`, an empty index, or a wrongly sized query —
    /// mirroring the oracle's degenerate-input behaviour.
    pub fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<Scored> {
        if k == 0 || nprobe == 0 || self.is_empty() || query.len() != self.dim
        {
            return Vec::new();
        }
        let q = vector::normalized(query);

        // Rank all centroids once — independent of nprobe, so candidate sets
        // are nested as nprobe grows (the monotone-recall property).
        let mut order: Vec<usize> = (0..self.centroids.len()).collect();
        order.sort_by(|&a, &b| {
            squared_euclidean(&self.centroids[a], &q)
                .total_cmp(&squared_euclidean(&self.centroids[b], &q))
                .then(a.cmp(&b))
        });

        let mut candidates: Vec<usize> = Vec::new();
        for &c in order.iter().take(nprobe.min(order.len()))
        {
            candidates.extend_from_slice(&self.lists[c]);
        }
        // Score in insertion order so the final sort sees the same sequence
        // the oracle sorts — bit-identical output at full probe.
        candidates.sort_unstable();

        let mut scored: Vec<Scored> = candidates
            .into_iter()
            .map(|i| Scored {
                id: self.ids[i],
                score: vector::dot(&q, &self.normed[i]),
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

    fn corpus() -> Vec<(u64, Vec<f32>)> {
        // Two well-separated directions and a straggler.
        vec![
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.9, 0.1, 0.0]),
            (3, vec![0.0, 0.0, 1.0]),
            (4, vec![0.0, 0.1, 0.9]),
            (5, vec![0.5, 0.5, 0.0]),
        ]
    }

    #[test]
    fn build_rejects_bad_parameters() {
        assert_eq!(
            IvfIndex::build(3, 0, 4, &corpus()).unwrap_err(),
            RetrievalError::InvalidParameter {
                reason: "nlist must be nonzero for a non-empty corpus",
            }
        );
        assert_eq!(
            IvfIndex::build(2, 2, 4, &corpus()).unwrap_err(),
            RetrievalError::DimMismatch {
                expected: 2,
                got: 3
            }
        );
        // An empty corpus is fine, whatever nlist says.
        let empty = IvfIndex::build(3, 0, 4, &[]).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.nlist(), 0);
        assert!(empty.search(&[1.0, 0.0, 0.0], 5, 1).is_empty());
    }

    #[test]
    fn nlist_is_clamped_to_the_corpus_size() {
        let idx = IvfIndex::build(3, 100, 2, &corpus()).unwrap();
        assert_eq!(idx.nlist(), 5);
        assert_eq!(idx.len(), 5);
    }

    #[test]
    fn degenerate_searches_return_empty() {
        let idx = IvfIndex::build(3, 2, 2, &corpus()).unwrap();
        assert!(idx.search(&[1.0, 0.0, 0.0], 0, 2).is_empty()); // k = 0
        assert!(idx.search(&[1.0, 0.0, 0.0], 5, 0).is_empty()); // nprobe = 0
        assert!(idx.search(&[1.0, 0.0], 5, 2).is_empty()); // wrong dim
    }

    #[test]
    fn full_probe_matches_the_exact_oracle_on_a_small_corpus() {
        let entries = corpus();
        let mut oracle = crate::DenseIndex::new(3);
        for (id, v) in &entries
        {
            oracle.add(*id, v).unwrap();
        }
        let idx = IvfIndex::build(3, 2, 4, &entries).unwrap();
        for q in [[1.0, 0.0, 0.0], [0.0, 0.2, 1.0], [0.7, 0.7, 0.1]]
        {
            assert_eq!(idx.search(&q, 5, idx.nlist()), oracle.search(&q, 5));
        }
    }

    #[test]
    fn probing_one_list_returns_only_that_cluster() {
        let idx = IvfIndex::build(3, 2, 4, &corpus()).unwrap();
        let hits = idx.search(&[0.0, 0.0, 1.0], 5, 1);
        assert!(!hits.is_empty());
        assert!(hits.len() < 5, "one probe must not scan the whole corpus");
    }
}
