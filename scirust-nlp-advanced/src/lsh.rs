//! Locality-Sensitive Hashing (LSH) for near-duplicate detection.
//!
//! Built on top of [`crate::similarity::MinHash`]: band-and-bucket a MinHash
//! signature into multiple bands so that pairs whose estimated Jaccard
//! similarity exceeds a threshold collide in at least one band. This turns
//! *O(n²)* pairwise comparison into *O(n)* bucketed lookup — the standard
//! MinHash-LSH pipeline used for dedup at scale (e.g. web/Megathon dedup).
//!
//! ## Threshold
//! For `b` bands of `r` rows each, the collision "S-curve" inflection is at
//! `s* ≈ (1/b)^(1/r)`. Pick `(b, r)` to place `s*` near the desired similarity
//! threshold.

use crate::similarity::MinHash;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// One band: a sub-signature of `rows` consecutive MinHash values, hashed to
/// a bucket key. Two items collide in a band iff their `rows`-length
/// sub-signatures are identical.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LshBand {
    /// Row count (signature slice length for this band).
    pub rows: usize,
    /// Offset into the full MinHash signature where this band starts.
    pub offset: usize,
}

/// A MinHash-LSH index.
///
/// Deterministic given the `seed`: the underlying [`MinHash`] coefficients and
/// the band layout are derived from `seed` and `num_hashes`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinHashLsh {
    /// The MinHash family used to compute signatures.
    pub minhash: MinHash,
    /// Bands layout.
    pub bands: Vec<LshBand>,
    /// `band_index[i]` = map from bucket-key → set of item ids that fell in
    /// that bucket for band `i`.
    pub band_index: Vec<HashMap<u64, Vec<usize>>>,
    /// Optional threshold hint (purely informational; the real curve is set by
    /// the band layout).
    pub threshold: f64,
}

impl MinHashLsh {
    /// Build an LSH index for a target similarity `threshold` and an expected
    /// signature length `num_hashes`. The band layout `(b, r)` is chosen so the
    /// S-curve inflection sits near `threshold`.
    pub fn new(num_hashes: usize, threshold: f64, seed: u64) -> Self {
        let num_hashes = num_hashes.max(8);
        let t = threshold.clamp(0.1, 0.99);
        // Greedy search for (b, r) with b*r <= num_hashes and (1/b)^(1/r) ≈ t.
        // Require r >= 2 (r=1 trivially gives s* = 1/b) and b >= 2 (else a single
        // bucket catches everything). Among equally-good layouts, prefer the one
        // using the most of the signature budget (b*r closest to num_hashes).
        // best stores (b, r, s_star, err, budget).
        let mut best: Option<(usize, usize, f64, f64, usize)> = None;
        for r in 2..=num_hashes
        {
            for b in 2..=num_hashes
            {
                if b * r > num_hashes
                {
                    break;
                }
                let s_star = (1.0_f64 / b as f64).powf(1.0 / r as f64);
                let err = (s_star - t).abs();
                let budget = b * r;
                let better = match &best
                {
                    None => true,
                    Some((_, _, _, prev_err, prev_budget)) =>
                    {
                        err + 1e-9 < *prev_err
                            || ((err - prev_err).abs() < 1e-9 && budget > *prev_budget)
                    },
                };
                if better
                {
                    best = Some((b, r, s_star, err, budget));
                }
            }
        }
        let (b, r, s_star, _, _) = best.unwrap_or((num_hashes / 2, 2, 0.5, 0.0, num_hashes));
        let bands = (0..b)
            .map(|i| LshBand {
                rows: r,
                offset: i * r,
            })
            .collect();
        Self {
            minhash: MinHash::new(num_hashes, seed),
            bands,
            band_index: (0..b).map(|_| HashMap::new()).collect(),
            threshold: s_star,
        }
    }

    /// Compute the MinHash signature of a tokenized document.
    pub fn signature(&self, tokens: &[String]) -> Vec<u64> {
        self.minhash.compute_signature(tokens)
    }

    /// Fold a `rows`-length slice of the signature into a single `u64` bucket
    /// key (FNV-1a over the bytes of the slice).
    fn bucket_key(slice: &[u64]) -> u64 {
        let mut h: u64 = 14695981039346656037;
        for v in slice
        {
            for &byte in v.to_le_bytes().iter()
            {
                h ^= byte as u64;
                h = h.wrapping_mul(1099511628211);
            }
        }
        h
    }

    /// Insert an item (by `id`) under its precomputed `signature`. The item is
    /// added to every band's bucket matching its slice.
    pub fn insert(&mut self, id: usize, signature: &[u64]) {
        for (bi, band) in self.bands.iter().enumerate()
        {
            let end = band.offset + band.rows;
            if end > signature.len()
            {
                continue;
            }
            let key = Self::bucket_key(&signature[band.offset..end]);
            self.band_index[bi].entry(key).or_default().push(id);
        }
    }

    /// Insert a tokenized document: compute its signature then [`insert`](Self::insert).
    pub fn insert_doc(&mut self, id: usize, tokens: &[String]) {
        let sig = self.signature(tokens);
        self.insert(id, &sig);
    }

    /// Candidate near-duplicates of `signature`: every id that shares at least
    /// one band bucket. Caller must still verify with
    /// [`MinHash::estimate_jaccard`] (the LSH only nominates candidates).
    pub fn query(&self, signature: &[u64]) -> Vec<usize> {
        let mut hits: Vec<usize> = Vec::new();
        let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for (bi, band) in self.bands.iter().enumerate()
        {
            let end = band.offset + band.rows;
            if end > signature.len()
            {
                continue;
            }
            let key = Self::bucket_key(&signature[band.offset..end]);
            if let Some(ids) = self.band_index[bi].get(&key)
            {
                for &id in ids
                {
                    if seen.insert(id)
                    {
                        hits.push(id);
                    }
                }
            }
        }
        hits
    }

    /// Query by a tokenized document.
    pub fn query_doc(&self, tokens: &[String]) -> Vec<usize> {
        let sig = self.signature(tokens);
        self.query(&sig)
    }

    /// Number of bands.
    pub fn num_bands(&self) -> usize {
        self.bands.len()
    }

    /// Rows per band (uniform layout).
    pub fn rows_per_band(&self) -> usize {
        self.bands.first().map(|b| b.rows).unwrap_or(0)
    }

    /// The inflection similarity `s*` of the S-curve for the chosen layout.
    pub fn estimated_threshold(&self) -> f64 {
        self.threshold
    }

    /// Find all near-duplicate pairs currently in the index. Returns a vec of
    /// `(id_a, id_b)` with `id_a < id_b`, each pair once.
    pub fn duplicate_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs: std::collections::BTreeSet<(usize, usize)> =
            std::collections::BTreeSet::new();
        for bucket_map in &self.band_index
        {
            for ids in bucket_map.values()
            {
                if ids.len() < 2
                {
                    continue;
                }
                for i in 0..ids.len()
                {
                    for j in (i + 1)..ids.len()
                    {
                        let (a, b) = if ids[i] < ids[j]
                        {
                            (ids[i], ids[j])
                        }
                        else
                        {
                            (ids[j], ids[i])
                        };
                        pairs.insert((a, b));
                    }
                }
            }
        }
        pairs.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<String> {
        s.split_whitespace().map(|w| w.to_lowercase()).collect()
    }

    #[test]
    fn near_duplicates_collide() {
        let mut lsh = MinHashLsh::new(64, 0.8, 1);
        let docs = [
            "the quick brown fox jumps over the lazy dog",
            "the quick brown fox jumps over the lazy dog", // exact dup
            "a completely different sentence about rust programming",
            "the quick brown fox jumps over a lazy dog", // near dup (1 word swap)
        ];
        for (i, d) in docs.iter().enumerate()
        {
            lsh.insert_doc(i, &toks(d));
        }
        let pairs = lsh.duplicate_pairs();
        // Exact duplicates (0,1) must collide; near-dup (0,3) very likely too.
        assert!(pairs.contains(&(0, 1)), "exact dup must collide: {pairs:?}");
    }

    #[test]
    fn unrelated_docs_rarely_collide() {
        let mut lsh = MinHashLsh::new(128, 0.8, 5);
        lsh.insert_doc(0, &toks("alpha beta gamma delta epsilon zeta eta theta"));
        lsh.insert_doc(1, &toks("rust cargo crate module trait impl async await"));
        let pairs = lsh.duplicate_pairs();
        assert!(
            pairs.is_empty(),
            "unrelated docs should not collide: {pairs:?}"
        );
    }

    #[test]
    fn query_returns_candidates() {
        let mut lsh = MinHashLsh::new(64, 0.7, 11);
        lsh.insert_doc(0, &toks("fix the database timeout in the query layer"));
        lsh.insert_doc(
            1,
            &toks("refactor the database query layer timeout handling"),
        );
        lsh.insert_doc(2, &toks("totally unrelated weather forecast for tomorrow"));
        let q = lsh.query_doc(&toks("fix the database timeout in the query layer"));
        assert!(q.contains(&0), "self-query must include self: {q:?}");
        assert!(!q.contains(&2), "unrelated must not be a candidate: {q:?}");
    }

    #[test]
    fn determinism_same_seed_same_buckets() {
        let mut a = MinHashLsh::new(64, 0.8, 42);
        let mut b = MinHashLsh::new(64, 0.8, 42);
        for (i, d) in ["x y z", "x y z", "x y w"].iter().enumerate()
        {
            a.insert_doc(i, &toks(d));
            b.insert_doc(i, &toks(d));
        }
        assert_eq!(a.duplicate_pairs(), b.duplicate_pairs());
    }

    #[test]
    fn threshold_near_target() {
        let lsh = MinHashLsh::new(128, 0.75, 1);
        let s = lsh.estimated_threshold();
        assert!((s - 0.75).abs() < 0.15, "s*={s} should be near 0.75");
    }
}
