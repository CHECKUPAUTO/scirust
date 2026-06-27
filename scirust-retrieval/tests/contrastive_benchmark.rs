//! End-to-end validation of contrastive fine-tuning, measured with the crate's
//! own ranking metrics.
//!
//! The corpus is a deterministic "two views" task: query `i` lives in the first
//! half of the feature space (`[eᵢ ; 0]`) and its relevant document in the second
//! half (`[0 ; eᵢ]`). The two halves are disjoint, so the *raw* cosine between any
//! query and any document is exactly 0 — retrieval is no better than chance until
//! a contrastively-trained projection learns to map matching views together.
//! After fine-tuning, Recall@1 and nDCG@k must reach 1.0.

use std::collections::{HashMap, HashSet};

use scirust_retrieval::contrastive::train;
use scirust_retrieval::metrics::{ndcg_at_k, recall_at_k};
use scirust_retrieval::{
    ContrastiveConfig, DenseIndex, Encoder, ProjectedEncoder, ProjectionHead, SemanticRetriever,
};

const N: usize = 5;
const DIM_IN: usize = 2 * N;
const DIM_OUT: usize = 8;

fn corpus() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut queries = Vec::new();
    let mut docs = Vec::new();
    for i in 0..N
    {
        let mut q = vec![0.0f32; DIM_IN];
        q[i] = 1.0; // [eᵢ ; 0]
        let mut d = vec![0.0f32; DIM_IN];
        d[N + i] = 1.0; // [0 ; eᵢ]
        queries.push(q);
        docs.push(d);
    }
    (queries, docs)
}

/// Index `docs` and retrieve each query, returning (mean Recall@1, mean nDCG@N).
/// Document `j` has id `j` and query `i` is relevant to document `i`.
fn evaluate(docs: &[Vec<f32>], queries: &[Vec<f32>]) -> (f64, f64) {
    let dim = docs[0].len();
    let mut index = DenseIndex::new(dim);
    for (j, d) in docs.iter().enumerate()
    {
        index.add(j as u64, d).unwrap();
    }
    let mut recall_sum = 0.0;
    let mut ndcg_sum = 0.0;
    for (i, q) in queries.iter().enumerate()
    {
        let ranked: Vec<u64> = index.search(q, N).into_iter().map(|s| s.id).collect();
        let relevant: HashSet<u64> = [i as u64].into_iter().collect();
        let gains: HashMap<u64, f64> = [(i as u64, 1.0)].into_iter().collect();
        recall_sum += recall_at_k(&ranked, &relevant, 1);
        ndcg_sum += ndcg_at_k(&ranked, &gains, N);
    }
    let n = queries.len() as f64;
    (recall_sum / n, ndcg_sum / n)
}

#[test]
fn fine_tuning_lifts_recall_and_ndcg_to_perfect() {
    let (queries, docs) = corpus();

    // Baseline: raw embeddings (disjoint subspaces → no cross-view signal).
    let (base_recall, base_ndcg) = evaluate(&docs, &queries);
    assert!(
        base_recall < 0.5,
        "raw Recall@1 should be near chance, got {base_recall}"
    );

    // Fine-tune a projection head on the (query, document) pairs.
    let mut head = ProjectionHead::new(DIM_IN, DIM_OUT, 7);
    let losses = train(
        &mut head,
        &queries,
        &docs,
        ContrastiveConfig {
            epochs: 500,
            lr: 0.05,
            temperature: 0.1,
        },
    );
    assert!(
        *losses.last().unwrap() < 0.4 * losses[0],
        "InfoNCE loss should converge: {} -> {}",
        losses[0],
        losses.last().unwrap()
    );

    // Re-measure on the projected space.
    let proj_docs: Vec<Vec<f32>> = docs.iter().map(|d| head.project(d)).collect();
    let proj_queries: Vec<Vec<f32>> = queries.iter().map(|q| head.project(q)).collect();
    let (tuned_recall, tuned_ndcg) = evaluate(&proj_docs, &proj_queries);

    assert!(
        tuned_recall > base_recall && tuned_ndcg > base_ndcg,
        "fine-tuning must improve the metrics: recall {base_recall}->{tuned_recall}, \
         ndcg {base_ndcg}->{tuned_ndcg}"
    );
    assert_eq!(tuned_recall, 1.0, "Recall@1 after fine-tuning");
    assert!(
        tuned_ndcg > 0.999,
        "nDCG@N after fine-tuning, got {tuned_ndcg}"
    );
}

/// A deterministic stand-in encoder: maps known texts to fixed vectors, so the
/// `SemanticRetriever` path is exercised without the MiniLLM's random weights.
struct MockEncoder {
    table: HashMap<String, Vec<f32>>,
    dim: usize,
}

impl Encoder for MockEncoder {
    fn embedding_dim(&self) -> usize {
        self.dim
    }

    fn encode(&mut self, text: &str) -> Vec<f32> {
        self.table
            .get(text)
            .cloned()
            .unwrap_or_else(|| vec![0.0f32; self.dim])
    }
}

#[test]
fn projected_encoder_drives_the_semantic_retriever() {
    let (queries, docs) = corpus();

    // Train the head, then wrap a base encoder with it.
    let mut head = ProjectionHead::new(DIM_IN, DIM_OUT, 7);
    train(&mut head, &queries, &docs, ContrastiveConfig::default());

    let mut table = HashMap::new();
    for (i, (q, d)) in queries.iter().zip(&docs).enumerate()
    {
        table.insert(format!("q{i}"), q.clone());
        table.insert(format!("d{i}"), d.clone());
    }
    let base = MockEncoder { table, dim: DIM_IN };
    let encoder = ProjectedEncoder::new(base, head);

    // The retriever now operates in the fine-tuned space (dim = DIM_OUT).
    let mut retriever = SemanticRetriever::new(encoder);
    assert_eq!(retriever.index().dim(), DIM_OUT);
    for i in 0..N
    {
        retriever.index_text(i as u64, &format!("d{i}")).unwrap();
    }

    // Each query text retrieves its own document first.
    for i in 0..N
    {
        let hits = retriever.retrieve(&format!("q{i}"), 1);
        assert_eq!(hits[0].id, i as u64, "query q{i} should retrieve d{i}");
    }
}
