//! Document similarity: Cosine, Jaccard, and MinHash (approximate Jaccard).
//!
//! All functions operate on tokenized documents (strings of words).

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Cosine similarity for documents
// ---------------------------------------------------------------------------

/// Cosine similarity between two tokenized documents.
pub fn cosine_similarity(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() || b.is_empty()
    {
        return 0.0;
    }
    let mut vocab: Vec<&str> = a.iter().chain(b.iter()).map(|s| s.as_str()).collect();
    vocab.sort();
    vocab.dedup();
    let n = vocab.len();
    let mut va = vec![0.0f64; n];
    let mut vb = vec![0.0f64; n];
    for tok in a
    {
        if let Some(idx) = vocab.iter().position(|&v| v == tok.as_str())
        {
            va[idx] += 1.0;
        }
    }
    for tok in b
    {
        if let Some(idx) = vocab.iter().position(|&v| v == tok.as_str())
        {
            vb[idx] += 1.0;
        }
    }
    crate::classify::cosine_similarity(&va, &vb)
}

/// Cosine similarity between two documents represented as pre-computed
/// count vectors (must have the same length).
pub fn cosine_similarity_vectors(a: &[f64], b: &[f64]) -> f64 {
    crate::classify::cosine_similarity(a, b)
}

// ---------------------------------------------------------------------------
// Jaccard similarity for documents
// ---------------------------------------------------------------------------

/// Jaccard similarity between two tokenized documents (set-based).
pub fn jaccard_similarity(a: &[String], b: &[String]) -> f64 {
    crate::classify::jaccard_similarity(a, b)
}

/// Weighted Jaccard similarity: intersection / union where each token's
/// weight is its count in the document.
pub fn weighted_jaccard(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() && b.is_empty()
    {
        return 1.0;
    }
    let mut counts_a: HashMap<&str, f64> = HashMap::new();
    let mut counts_b: HashMap<&str, f64> = HashMap::new();
    for tok in a
    {
        *counts_a.entry(tok.as_str()).or_insert(0.0) += 1.0;
    }
    for tok in b
    {
        *counts_b.entry(tok.as_str()).or_insert(0.0) += 1.0;
    }
    let all_keys: HashSet<&str> = counts_a
        .keys()
        .copied()
        .chain(counts_b.keys().copied())
        .collect();
    let mut intersection = 0.0f64;
    let mut union = 0.0f64;
    for key in &all_keys
    {
        let ca = counts_a.get(key).copied().unwrap_or(0.0);
        let cb = counts_b.get(key).copied().unwrap_or(0.0);
        intersection += ca.min(cb);
        union += ca.max(cb);
    }
    if union == 0.0
    {
        0.0
    }
    else
    {
        intersection / union
    }
}

// ---------------------------------------------------------------------------
// MinHash
// ---------------------------------------------------------------------------

/// MinHash signature for approximate Jaccard similarity.
///
/// Uses `k` independent hash functions to produce a signature of length `k`.
/// The estimated Jaccard similarity between two sets is the fraction of
/// signature positions where they agree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinHash {
    /// Number of hash functions (= signature length).
    pub num_hashes: usize,
    /// Coefficients for the hash functions: h(x) = (a*x + b) % p % size.
    pub a: Vec<u64>,
    pub b: Vec<u64>,
    /// Large prime for hashing.
    pub prime: u64,
    /// Modulo for bucketing.
    pub size: u64,
}

impl MinHash {
    /// Create a new MinHash with `num_hashes` hash functions.
    pub fn new(num_hashes: usize, seed: u64) -> Self {
        let prime = 2147483647u64; // Mersenne prime 2^31 - 1
        let size = prime;
        // Deterministic pseudo-random coefficients from seed
        let mut a = Vec::with_capacity(num_hashes);
        let mut b = Vec::with_capacity(num_hashes);
        let mut state = seed;
        for _ in 0..num_hashes
        {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            a.push((state % (prime - 1)) + 1);
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            b.push(state % prime);
        }
        Self {
            num_hashes,
            a,
            b,
            prime,
            size,
        }
    }

    /// Compute the MinHash signature for a set of strings.
    #[allow(clippy::needless_range_loop)]
    pub fn compute_signature(&self, tokens: &[String]) -> Vec<u64> {
        let mut signature = vec![u64::MAX; self.num_hashes];
        for tok in tokens
        {
            let hash = self.hash_string(tok);
            for i in 0..self.num_hashes
            {
                let h =
                    (self.a[i].wrapping_mul(hash).wrapping_add(self.b[i])) % self.prime % self.size;
                if h < signature[i]
                {
                    signature[i] = h;
                }
            }
        }
        signature
    }

    /// Hash a string to a u64 using FNV-1a.
    fn hash_string(&self, s: &str) -> u64 {
        let mut h: u64 = 14695981039346656037; // FNV offset basis
        for byte in s.bytes()
        {
            h ^= byte as u64;
            h = h.wrapping_mul(1099511628211); // FNV prime
        }
        h
    }

    /// Estimate Jaccard similarity from two signatures.
    pub fn estimate_jaccard(sig_a: &[u64], sig_b: &[u64]) -> f64 {
        if sig_a.len() != sig_b.len() || sig_a.is_empty()
        {
            return 0.0;
        }
        let agree = sig_a
            .iter()
            .zip(sig_b.iter())
            .filter(|(a, b)| a == b)
            .count();
        agree as f64 / sig_a.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Document similarity matrix
// ---------------------------------------------------------------------------

/// Compute pairwise cosine similarity matrix for a corpus of tokenized docs.
pub fn cosine_similarity_matrix(documents: &[Vec<String>]) -> Vec<Vec<f64>> {
    let n = documents.len();
    let mut matrix = vec![vec![0.0f64; n]; n];
    for i in 0..n
    {
        matrix[i][i] = 1.0;
        for j in (i + 1)..n
        {
            let sim = cosine_similarity(&documents[i], &documents[j]);
            matrix[i][j] = sim;
            matrix[j][i] = sim;
        }
    }
    matrix
}

/// Compute pairwise Jaccard similarity matrix for a corpus of tokenized docs.
pub fn jaccard_similarity_matrix(documents: &[Vec<String>]) -> Vec<Vec<f64>> {
    let n = documents.len();
    let mut matrix = vec![vec![0.0f64; n]; n];
    for i in 0..n
    {
        matrix[i][i] = 1.0;
        for j in (i + 1)..n
        {
            let sim = jaccard_similarity(&documents[i], &documents[j]);
            matrix[i][j] = sim;
            matrix[j][i] = sim;
        }
    }
    matrix
}

/// Compute pairwise approximate Jaccard matrix using MinHash signatures.
pub fn minhash_similarity_matrix(
    documents: &[Vec<String>],
    num_hashes: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let mh = MinHash::new(num_hashes, seed);
    let signatures: Vec<Vec<u64>> = documents
        .iter()
        .map(|doc| mh.compute_signature(doc))
        .collect();
    let n = documents.len();
    let mut matrix = vec![vec![0.0f64; n]; n];
    for i in 0..n
    {
        matrix[i][i] = 1.0;
        for j in (i + 1)..n
        {
            let sim = MinHash::estimate_jaccard(&signatures[i], &signatures[j]);
            matrix[i][j] = sim;
            matrix[j][i] = sim;
        }
    }
    matrix
}

/// Find the most similar document to a query document.
pub fn most_similar(query: &[String], documents: &[Vec<String>]) -> Option<(usize, f64)> {
    if documents.is_empty()
    {
        return None;
    }
    let mut best_idx = 0;
    let mut best_sim = f64::NEG_INFINITY;
    for (i, doc) in documents.iter().enumerate()
    {
        let sim = cosine_similarity(query, doc);
        if sim > best_sim
        {
            best_sim = sim;
            best_idx = i;
        }
    }
    Some((best_idx, best_sim))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec!["hello".into(), "world".into()];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_disjoint() {
        let a = vec!["hello".into(), "world".into()];
        let b = vec!["foo".into(), "bar".into()];
        assert!(cosine_similarity(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_partial() {
        let a = vec!["hello".into(), "world".into(), "rust".into()];
        let b = vec!["hello".into(), "rust".into(), "is".into(), "great".into()];
        let sim = cosine_similarity(&a, &b);
        // 2 shared out of sqrt(3)*sqrt(4) = sqrt(12) ≈ 3.464
        assert!(sim > 0.5 && sim < 1.0);
    }

    #[test]
    fn test_jaccard_similarity_identical() {
        let a = vec!["a".into(), "b".into(), "c".into()];
        assert!((jaccard_similarity(&a, &a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_similarity_partial() {
        let a = vec!["a".into(), "b".into(), "c".into()];
        let b = vec!["b".into(), "c".into(), "d".into()];
        assert!((jaccard_similarity(&a, &b) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_jaccard() {
        let a = vec!["a".into(), "a".into(), "b".into()];
        let b = vec!["a".into(), "b".into(), "b".into()];
        // intersection: min(2,1) for "a" + min(1,2) for "b" = 1 + 1 = 2
        // union: max(2,1) for "a" + max(1,2) for "b" = 2 + 2 = 4
        assert!((weighted_jaccard(&a, &b) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_jaccard_empty() {
        let a: Vec<String> = vec![];
        let b: Vec<String> = vec![];
        assert!((weighted_jaccard(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_minhash_identical() {
        let mh = MinHash::new(100, 42);
        let a = vec!["hello".into(), "world".into(), "rust".into()];
        let sig_a = mh.compute_signature(&a);
        let sig_b = mh.compute_signature(&a);
        let sim = MinHash::estimate_jaccard(&sig_a, &sig_b);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_minhash_similar() {
        let mh = MinHash::new(200, 42);
        let a: Vec<String> = (0..50).map(|i| format!("word{}", i)).collect();
        let mut b = a.clone();
        // Replace 10% of words
        for i in 0..5
        {
            b[i] = format!("new{}", i);
        }
        let sig_a = mh.compute_signature(&a);
        let sig_b = mh.compute_signature(&b);
        let sim = MinHash::estimate_jaccard(&sig_a, &sig_b);
        // Should be roughly 0.8-1.0 (approximate, not exact)
        assert!(sim > 0.5, "MinHash sim = {}", sim);
    }

    #[test]
    fn test_minhash_disjoint() {
        let mh = MinHash::new(100, 42);
        let a: Vec<String> = (0..20).map(|i| format!("aaa{}", i)).collect();
        let b: Vec<String> = (0..20).map(|i| format!("bbb{}", i)).collect();
        let sig_a = mh.compute_signature(&a);
        let sig_b = mh.compute_signature(&b);
        let sim = MinHash::estimate_jaccard(&sig_a, &sig_b);
        // Should be close to 0
        assert!(sim < 0.3, "MinHash sim = {}", sim);
    }

    #[test]
    fn test_cosine_similarity_matrix() {
        let docs = vec![
            vec!["a".into(), "b".into()],
            vec!["a".into(), "b".into()],
            vec!["c".into(), "d".into()],
        ];
        let matrix = cosine_similarity_matrix(&docs);
        assert_eq!(matrix.len(), 3);
        assert!((matrix[0][1] - 1.0).abs() < 1e-10);
        assert!(matrix[0][2].abs() < 1e-10);
        assert!((matrix[0][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaccard_similarity_matrix() {
        let docs = vec![
            vec!["a".into(), "b".into()],
            vec!["a".into(), "b".into(), "c".into()],
        ];
        let matrix = jaccard_similarity_matrix(&docs);
        // intersection={"a","b"}=2, union={"a","b","c"}=3 → 2/3
        assert!((matrix[0][1] - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_minhash_similarity_matrix() {
        let docs = vec![
            vec!["a".into(), "b".into(), "c".into()],
            vec!["a".into(), "b".into(), "d".into()],
            vec!["x".into(), "y".into(), "z".into()],
        ];
        let matrix = minhash_similarity_matrix(&docs, 100, 42);
        assert_eq!(matrix.len(), 3);
        assert!((matrix[0][0] - 1.0).abs() < 1e-10);
        // Docs 0 and 1 share 2/4 = 0.5 Jaccard
        assert!(matrix[0][1] > 0.3);
        // Docs 0 and 2 share nothing
        assert!(matrix[0][2] < 0.3);
    }

    #[test]
    fn test_most_similar() {
        let docs = vec![
            vec!["hello".into(), "world".into()],
            vec!["hello".into(), "rust".into()],
            vec!["foo".into(), "bar".into()],
        ];
        let query = vec!["hello".into(), "world".into()];
        let result = most_similar(&query, &docs);
        assert!(result.is_some());
        let (idx, sim) = result.unwrap();
        assert_eq!(idx, 0);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_most_similar_empty() {
        let query = vec!["hello".into()];
        assert!(most_similar(&query, &[]).is_none());
    }

    #[test]
    fn test_cosine_similarity_empty() {
        assert!(cosine_similarity(&[], &[]).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity_vectors(&a, &b).abs() < 1e-10);
        assert!((cosine_similarity_vectors(&a, &a) - 1.0).abs() < 1e-10);
    }
}
