//! Late-interaction (ColBERT-style) reranking via MaxSim.
//!
//! A single pooled vector per document is fast but blurs token-level meaning. In
//! **late interaction** each query token keeps its own vector and is matched
//! against its single best-aligning document token; the per-token maxima are
//! summed. This recovers fine-grained term matches that a pooled bi-encoder (and
//! the coarse chunking RAG relies on) misses — the precision edge of a
//! second-stage reranker over the first-stage dense candidates.

use crate::Scored;
use crate::vector;

/// MaxSim score `Σ_q maxₖ cos(q, dₖ)`: for every query-token vector, the maximum
/// cosine similarity over all document-token vectors, summed. Returns `0.0` if
/// either side has no tokens. Higher is more relevant; the score is bounded by
/// the number of query tokens.
pub fn maxsim(query_tokens: &[Vec<f32>], doc_tokens: &[Vec<f32>]) -> f32 {
    if query_tokens.is_empty() || doc_tokens.is_empty()
    {
        return 0.0;
    }
    let mut total = 0.0f32;
    for q in query_tokens
    {
        let mut best = f32::NEG_INFINITY;
        for d in doc_tokens
        {
            let s = vector::cosine(q, d);
            if s > best
            {
                best = s;
            }
        }
        total += best;
    }
    total
}

/// Rerank `candidates` (each a document id with its token-vector matrix) by
/// [`maxsim`] against the query tokens, returning the top-`k` as [`Scored`],
/// score descending with an id-ascending tie-break.
pub fn rerank(
    query_tokens: &[Vec<f32>],
    candidates: &[(u64, Vec<Vec<f32>>)],
    k: usize,
) -> Vec<Scored> {
    let mut scored: Vec<Scored> = candidates
        .iter()
        .map(|(id, doc_tokens)| Scored {
            id: *id,
            score: maxsim(query_tokens, doc_tokens),
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

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::FRAC_1_SQRT_2;

    #[test]
    fn maxsim_sums_per_query_token_best_matches() {
        // Query tokens along the two axes.
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        // Doc has an exact match for each query token (plus a diagonal distractor).
        let doc = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        // best(q1) = cos with [1,0] = 1; best(q2) = cos with [0,1] = 1; sum = 2.
        assert!(
            (maxsim(&q, &doc) - 2.0).abs() < 1e-6,
            "{}",
            maxsim(&q, &doc)
        );

        // A doc whose only token is the diagonal: each query token aligns at 1/√2.
        let diag = vec![vec![1.0, 1.0]];
        assert!(
            (maxsim(&q, &diag) - 2.0 * FRAC_1_SQRT_2).abs() < 1e-6,
            "{}",
            maxsim(&q, &diag)
        );
    }

    #[test]
    fn empty_sides_score_zero() {
        assert_eq!(maxsim(&[], &[vec![1.0, 0.0]]), 0.0);
        assert_eq!(maxsim(&[vec![1.0, 0.0]], &[]), 0.0);
    }

    #[test]
    fn rerank_orders_exact_match_above_partial() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let candidates = vec![
            (100, vec![vec![1.0, 1.0]]),                 // score 2/√2 ≈ 1.414
            (200, vec![vec![1.0, 0.0], vec![0.0, 1.0]]), // score 2.0
        ];
        let ranked = rerank(&q, &candidates, 2);
        assert_eq!(ranked[0].id, 200, "exact-match doc must rank first");
        assert!((ranked[0].score - 2.0).abs() < 1e-6);
        assert_eq!(ranked[1].id, 100);
        assert!((ranked[1].score - 2.0 * FRAC_1_SQRT_2).abs() < 1e-6);
    }
}
