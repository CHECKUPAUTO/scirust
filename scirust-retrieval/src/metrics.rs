//! Ranking-quality metrics for retrieval evaluation.
//!
//! Each takes a ranked list of returned document ids (best first) and a notion of
//! relevance, and returns a score in `[0, 1]`. These let a pure-retrieval system
//! be measured the way RAG benchmarks measure their retrievers — Recall@k, MRR,
//! MAP, nDCG@k — so "challenges RAG" is a number, not a claim.

use std::collections::{HashMap, HashSet};

/// Recall@k: fraction of all relevant docs that appear in the top-`k`.
pub fn recall_at_k(retrieved: &[u64], relevant: &HashSet<u64>, k: usize) -> f64 {
    if relevant.is_empty()
    {
        return 0.0;
    }
    let hits = retrieved
        .iter()
        .take(k)
        .filter(|id| relevant.contains(id))
        .count();
    hits as f64 / relevant.len() as f64
}

/// Precision@k: fraction of the top-`k` returned that are relevant. The
/// denominator is `min(k, |retrieved|)` so a short result list is not penalised
/// for positions that do not exist.
pub fn precision_at_k(retrieved: &[u64], relevant: &HashSet<u64>, k: usize) -> f64 {
    let depth = retrieved.len().min(k);
    if depth == 0
    {
        return 0.0;
    }
    let hits = retrieved
        .iter()
        .take(k)
        .filter(|id| relevant.contains(id))
        .count();
    hits as f64 / depth as f64
}

/// Reciprocal rank: `1 / rank` of the first relevant doc (rank is 1-based), or
/// `0.0` if none of the returned docs are relevant.
pub fn reciprocal_rank(retrieved: &[u64], relevant: &HashSet<u64>) -> f64 {
    for (i, id) in retrieved.iter().enumerate()
    {
        if relevant.contains(id)
        {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

/// Mean reciprocal rank over several `(ranking, relevant-set)` queries.
pub fn mean_reciprocal_rank(queries: &[(Vec<u64>, HashSet<u64>)]) -> f64 {
    if queries.is_empty()
    {
        return 0.0;
    }
    let total: f64 = queries
        .iter()
        .map(|(ranking, relevant)| reciprocal_rank(ranking, relevant))
        .sum();
    total / queries.len() as f64
}

/// Average precision for one query: the mean of the precision values taken at
/// each rank where a relevant doc occurs, divided by the number of relevant docs.
pub fn average_precision(retrieved: &[u64], relevant: &HashSet<u64>) -> f64 {
    if relevant.is_empty()
    {
        return 0.0;
    }
    let mut hits = 0usize;
    let mut sum = 0.0f64;
    for (i, id) in retrieved.iter().enumerate()
    {
        if relevant.contains(id)
        {
            hits += 1;
            sum += hits as f64 / (i as f64 + 1.0);
        }
    }
    sum / relevant.len() as f64
}

/// Discounted cumulative gain of the first `k` ranks: `Σ gainᵢ / log₂(rank+1)`.
fn dcg_at_k(gains: impl Iterator<Item = f64>, k: usize) -> f64 {
    gains
        .take(k)
        .enumerate()
        .map(|(i, g)| g / (i as f64 + 2.0).log2())
        .sum()
}

/// nDCG@k with graded relevance `gains` (use `1.0`/`0.0` for binary relevance):
/// the DCG of the returned ranking divided by the ideal DCG (gains sorted
/// descending). Returns `0.0` when there is no positive gain to recover.
pub fn ndcg_at_k(retrieved: &[u64], gains: &HashMap<u64, f64>, k: usize) -> f64 {
    let actual = dcg_at_k(
        retrieved
            .iter()
            .map(|id| gains.get(id).copied().unwrap_or(0.0)),
        k,
    );
    let mut ideal: Vec<f64> = gains.values().copied().filter(|&g| g > 0.0).collect();
    ideal.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
    let best = dcg_at_k(ideal.into_iter(), k);
    if best <= 0.0
    {
        return 0.0;
    }
    actual / best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(ids: &[u64]) -> HashSet<u64> {
        ids.iter().copied().collect()
    }

    #[test]
    fn recall_and_precision_hand_values() {
        // retrieved top-4 = [1,2,3,4]; relevant = {2,4,9}; hits in top-4 = {2,4}.
        let retrieved = [1, 2, 3, 4];
        let relevant = set(&[2, 4, 9]);
        // recall = 2 / 3 relevant total.
        assert!((recall_at_k(&retrieved, &relevant, 4) - 2.0 / 3.0).abs() < 1e-12);
        // precision@4 = 2 / 4.
        assert!((precision_at_k(&retrieved, &relevant, 4) - 0.5).abs() < 1e-12);
        // precision@2 over [1,2] = 1 hit / 2.
        assert!((precision_at_k(&retrieved, &relevant, 2) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn reciprocal_rank_and_mrr_hand_values() {
        // first relevant at rank 2 -> 1/2.
        assert!((reciprocal_rank(&[5, 2, 7], &set(&[2])) - 0.5).abs() < 1e-12);
        // none relevant -> 0.
        assert_eq!(reciprocal_rank(&[5, 7], &set(&[2])), 0.0);
        // MRR of (rank-1 -> 1) and (rank-2 -> 1/2) = (1 + 0.5)/2 = 0.75.
        let q = vec![(vec![2u64, 9], set(&[2])), (vec![9u64, 2], set(&[2]))];
        assert!((mean_reciprocal_rank(&q) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn average_precision_hand_value() {
        // retrieved = [2,1,4,3]; relevant = {2,4}.
        // rank1 (id 2) relevant -> prec 1/1 = 1; rank3 (id 4) relevant -> prec 2/3.
        // AP = (1 + 2/3) / 2 relevant = 0.8333...
        let ap = average_precision(&[2, 1, 4, 3], &set(&[2, 4]));
        assert!((ap - (1.0 + 2.0 / 3.0) / 2.0).abs() < 1e-12, "ap {ap}");
    }

    #[test]
    fn ndcg_matches_hand_computed_value() {
        // gains: id1=3, id2=2, id3=0, id4=1. ranking [1,4,3,2].
        // DCG  = 3/log2(2) + 1/log2(3) + 0/log2(4) + 2/log2(5)
        //      = 3 + 0.6309298 + 0 + 0.8613531 = 4.4922829
        // IDCG (gains 3,2,1,0) = 3 + 2/log2(3) + 1/log2(4) + 0
        //      = 3 + 1.2618595 + 0.5 = 4.7618595
        // nDCG = 4.4922829 / 4.7618595 = 0.9433884
        let gains: HashMap<u64, f64> = [(1, 3.0), (2, 2.0), (3, 0.0), (4, 1.0)]
            .into_iter()
            .collect();
        let n = ndcg_at_k(&[1, 4, 3, 2], &gains, 4);
        assert!((n - 0.943_388_4).abs() < 1e-6, "ndcg {n}");
        // The ideal ranking scores exactly 1.0.
        let ideal = ndcg_at_k(&[1, 2, 4, 3], &gains, 4);
        assert!((ideal - 1.0).abs() < 1e-12, "ideal ndcg {ideal}");
    }

    #[test]
    fn empty_relevance_is_zero_not_nan() {
        assert_eq!(recall_at_k(&[1, 2], &set(&[]), 2), 0.0);
        assert_eq!(average_precision(&[1, 2], &set(&[]),), 0.0);
        assert_eq!(ndcg_at_k(&[1, 2], &HashMap::new(), 2), 0.0);
    }
}
