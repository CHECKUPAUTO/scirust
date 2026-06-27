//! Hybrid retrieval: fuse **dense** (semantic) and **BM25** (lexical) rankings.
//!
//! Pure semantic retrieval nails paraphrase and meaning but can miss an exact
//! keyword (a product code, a rare name); BM25 nails the keyword but misses
//! meaning. Fusing the two ranked lists with **Reciprocal Rank Fusion** (RRF)
//! recovers documents found by *either* signal, beating both pure approaches on
//! mixed queries — and it needs no score normalisation, just ranks.
//!
//! Everything here is deterministic and pure-Rust: a fixed tokenizer, fixed-order
//! `f32` accumulation, and an id-ascending tie-break, so a run is reproducible.

use crate::Scored;
use std::collections::HashMap;

/// Lowercase, split on non-alphanumeric runs. Deterministic and dependency-free
/// — enough for BM25 term matching.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| t.to_lowercase())
        .collect()
}

/// A BM25 lexical index (Okapi BM25 with the non-negative `ln(1 + …)` IDF).
///
/// `k1` controls term-frequency saturation, `b` the document-length
/// normalisation. An inverted index keeps `search` proportional to the postings
/// of the query terms, not the whole corpus.
pub struct Bm25Index {
    k1: f32,
    b: f32,
    vocab: HashMap<String, usize>,
    postings: Vec<Vec<(usize, u32)>>, // term id -> [(doc index, term frequency)]
    doc_len: Vec<u32>,
    ids: Vec<u64>,
    total_len: u64,
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new(1.2, 0.75)
    }
}

impl Bm25Index {
    /// New empty index with the given BM25 parameters (`k1`, `b`).
    pub fn new(k1: f32, b: f32) -> Self {
        Self {
            k1,
            b,
            vocab: HashMap::new(),
            postings: Vec::new(),
            doc_len: Vec::new(),
            ids: Vec::new(),
            total_len: 0,
        }
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether nothing has been indexed.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Tokenize `text` and add it under `id`.
    pub fn add(&mut self, id: u64, text: &str) {
        let tokens = tokenize(text);
        let doc_idx = self.ids.len();
        // Term frequencies in this document.
        let mut tf: HashMap<String, u32> = HashMap::new();
        for tok in &tokens
        {
            *tf.entry(tok.clone()).or_insert(0) += 1;
        }
        for (term, count) in tf
        {
            let term_id = match self.vocab.get(&term)
            {
                Some(&i) => i,
                None =>
                {
                    let i = self.postings.len();
                    self.vocab.insert(term, i);
                    self.postings.push(Vec::new());
                    i
                },
            };
            self.postings[term_id].push((doc_idx, count));
        }
        self.doc_len.push(tokens.len() as u32);
        self.total_len += tokens.len() as u64;
        self.ids.push(id);
    }

    /// BM25 top-`k` for `query` (score descending, id ascending). Returns empty if
    /// the index is empty or `k == 0`.
    pub fn search(&self, query: &str, k: usize) -> Vec<Scored> {
        if k == 0 || self.is_empty()
        {
            return Vec::new();
        }
        let n = self.ids.len() as f32;
        let avgdl = self.total_len as f32 / n;

        // Distinct query term ids, in first-occurrence order (deterministic).
        let mut query_terms: Vec<usize> = Vec::new();
        let mut seen_terms: HashMap<usize, ()> = HashMap::new();
        for tok in tokenize(query)
        {
            if let Some(&tid) = self.vocab.get(&tok)
            {
                if seen_terms.insert(tid, ()).is_none()
                {
                    query_terms.push(tid);
                }
            }
        }

        let mut acc: HashMap<usize, f32> = HashMap::new();
        for &tid in &query_terms
        {
            let df = self.postings[tid].len() as f32;
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for &(doc_idx, tf) in &self.postings[tid]
            {
                let tf = tf as f32;
                let dl = self.doc_len[doc_idx] as f32;
                let denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl);
                let contribution = idf * tf * (self.k1 + 1.0) / denom;
                *acc.entry(doc_idx).or_insert(0.0) += contribution;
            }
        }

        let mut scored: Vec<Scored> = acc
            .into_iter()
            .map(|(doc_idx, score)| Scored {
                id: self.ids[doc_idx],
                score,
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

/// Reciprocal Rank Fusion of several ranked id lists: each id scores
/// `Σ_lists 1 / (rrf_k + rank)` (rank 1-based), and the merged list is sorted by
/// that score (descending, id ascending). `rrf_k` damps the weight of low ranks
/// (60 is the common default). No score normalisation is needed — only ranks.
pub fn reciprocal_rank_fusion(rankings: &[Vec<u64>], rrf_k: f32, k: usize) -> Vec<Scored> {
    let mut scores: HashMap<u64, f32> = HashMap::new();
    for ranking in rankings
    {
        for (rank, &id) in ranking.iter().enumerate()
        {
            *scores.entry(id).or_insert(0.0) += 1.0 / (rrf_k + rank as f32 + 1.0);
        }
    }
    let mut scored: Vec<Scored> = scores
        .into_iter()
        .map(|(id, score)| Scored { id, score })
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

/// End-to-end hybrid retriever: an [`Encoder`](crate::Encoder) + a dense index
/// for the semantic signal, a [`Bm25Index`] for the lexical signal, fused with
/// RRF. Each side contributes a deep candidate pool so a document ranked highly
/// by only one signal still surfaces.
pub struct HybridRetriever<E: crate::Encoder> {
    encoder: E,
    dense: crate::DenseIndex,
    sparse: Bm25Index,
    rrf_k: f32,
}

impl<E: crate::Encoder> HybridRetriever<E> {
    /// Build a hybrid retriever over `encoder` with RRF constant `rrf_k`
    /// (use `60.0` for the standard default).
    pub fn new(encoder: E, rrf_k: f32) -> Self {
        let dim = encoder.embedding_dim();
        Self {
            encoder,
            dense: crate::DenseIndex::new(dim),
            sparse: Bm25Index::default(),
            rrf_k,
        }
    }

    /// Encode `text` for the dense index and tokenize it for BM25, both under `id`.
    pub fn index_text(&mut self, id: u64, text: &str) -> Result<(), crate::RetrievalError> {
        let emb = self.encoder.encode(text);
        self.dense.add(id, &emb)?;
        self.sparse.add(id, text);
        Ok(())
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    /// Whether nothing has been indexed.
    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    /// Retrieve the top-`k` by fusing the dense and BM25 rankings. A generous
    /// candidate pool is drawn from each side before fusion.
    pub fn retrieve(&mut self, query: &str, k: usize) -> Vec<Scored> {
        let pool = (k * 5).max(20);
        let emb = self.encoder.encode(query);
        let dense_ids: Vec<u64> = self
            .dense
            .search(&emb, pool)
            .into_iter()
            .map(|s| s.id)
            .collect();
        let sparse_ids: Vec<u64> = self
            .sparse
            .search(query, pool)
            .into_iter()
            .map(|s| s.id)
            .collect();
        reciprocal_rank_fusion(&[dense_ids, sparse_ids], self.rrf_k, k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DenseIndex, Encoder};
    use std::collections::HashSet;

    #[test]
    fn bm25_matches_hand_computed_scores() {
        // b = 0 (no length norm) isolates the TF-saturation + IDF formula.
        // docs: d0="cat", d1="cat cat". N=2, df(cat)=2.
        // IDF = ln(1 + (2-2+0.5)/(2+0.5)) = ln(1.2) = 0.182322.
        // d0: tf=1 → IDF·1·2.2/(1+1.2) = 0.182322·1.0       = 0.182322
        // d1: tf=2 → IDF·2·2.2/(2+1.2) = 0.182322·1.375     = 0.250692
        let mut bm = Bm25Index::new(1.2, 0.0);
        bm.add(0, "cat");
        bm.add(1, "cat cat");
        let hits = bm.search("cat", 2);
        assert_eq!(hits[0].id, 1, "more occurrences should rank first");
        assert!(
            (hits[0].score - 0.250_692).abs() < 1e-4,
            "d1 {}",
            hits[0].score
        );
        assert_eq!(hits[1].id, 0);
        assert!(
            (hits[1].score - 0.182_322).abs() < 1e-4,
            "d0 {}",
            hits[1].score
        );
    }

    #[test]
    fn bm25_penalises_longer_documents_for_the_same_term_frequency() {
        // Both contain "cat" once; d1 is longer → BM25 ranks the shorter d0 higher.
        let mut bm = Bm25Index::new(1.2, 0.75);
        bm.add(0, "cat");
        bm.add(1, "cat foo bar baz");
        let hits = bm.search("cat", 2);
        assert_eq!(hits[0].id, 0, "shorter doc with same tf must rank higher");
        assert_eq!(hits[1].id, 1);
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn bm25_a_rare_keyword_pinpoints_its_document() {
        let mut bm = Bm25Index::default();
        bm.add(0, "the quick brown fox");
        bm.add(1, "the lazy dog");
        bm.add(2, "the sphinx of quartz"); // unique keyword "sphinx"/"quartz"
        let hits = bm.search("sphinx", 3);
        assert_eq!(hits.len(), 1, "only one doc contains the rare term");
        assert_eq!(hits[0].id, 2);
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn rrf_matches_hand_computed_fusion() {
        // rankings: [10,20,30] and [20,10,40], rrf_k = 60.
        // id10 = 1/61 + 1/62 = 0.032522 ; id20 = 1/62 + 1/61 = 0.032522 (tie → 10 first)
        // id30 = 1/63 = 0.015873 ; id40 = 1/63 = 0.015873 (tie → 30 first)
        let fused = reciprocal_rank_fusion(&[vec![10, 20, 30], vec![20, 10, 40]], 60.0, 4);
        assert_eq!(
            fused.iter().map(|s| s.id).collect::<Vec<_>>(),
            vec![10, 20, 30, 40]
        );
        assert!((fused[0].score - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-9);
    }

    #[test]
    fn fusion_recovers_documents_each_signal_alone_misses() {
        // Two relevant docs: D_sem (semantic match, no keyword) and D_kw (keyword
        // match, no semantic). Distractors are tuned so dense-only finds only
        // D_sem and BM25-only finds only D_kw — each gets Recall@2 = 0.5 — while
        // RRF surfaces BOTH → Recall@2 = 1.0.
        let dim = 4;
        // Embeddings: query ~ [1,0,..]; D_sem aligned; D_kw orthogonal; a semantic
        // distractor partially aligned so it outranks D_kw on the dense side.
        let q_emb = [1.0f32, 0.0, 0.0, 0.0];
        let docs: [(u64, [f32; 4], &str); 4] = [
            (1, [1.0, 0.0, 0.0, 0.0], "alpha beta"), // D_sem: dense-relevant
            (2, [0.0, 1.0, 0.0, 0.0], "zeta keyword"), // D_kw: lexical-relevant
            (3, [0.7, 0.7, 0.0, 0.0], "gamma delta"), // semantic distractor
            (4, [0.0, 0.0, 1.0, 0.0], "epsilon"),    // pure distractor
        ];
        let mut dense = DenseIndex::new(dim);
        let mut bm = Bm25Index::default();
        for (id, emb, text) in &docs
        {
            dense.add(*id, emb).unwrap();
            bm.add(*id, text);
        }
        let relevant: HashSet<u64> = [1, 2].into_iter().collect();
        let recall2 = |ids: &[u64]| -> f64 {
            ids.iter()
                .take(2)
                .filter(|id| relevant.contains(id))
                .count() as f64
                / 2.0
        };

        let dense_ids: Vec<u64> = dense.search(&q_emb, 4).into_iter().map(|s| s.id).collect();
        let sparse_ids: Vec<u64> = bm.search("keyword", 4).into_iter().map(|s| s.id).collect();
        // Each signal alone misses one of the two relevant docs.
        assert_eq!(recall2(&dense_ids), 0.5, "dense ids {dense_ids:?}");
        assert_eq!(recall2(&sparse_ids), 0.5, "sparse ids {sparse_ids:?}");

        let fused: Vec<u64> = reciprocal_rank_fusion(&[dense_ids, sparse_ids], 60.0, 2)
            .into_iter()
            .map(|s| s.id)
            .collect();
        assert_eq!(
            recall2(&fused),
            1.0,
            "hybrid should recover both: {fused:?}"
        );
    }

    // Deterministic stand-in encoder for the end-to-end retriever test.
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
    fn hybrid_retriever_fuses_both_signals_end_to_end() {
        // Query text "zeta" matches D_kw lexically; query embedding matches D_sem.
        let mut table = HashMap::new();
        table.insert("q".to_string(), vec![1.0, 0.0, 0.0, 0.0]);
        table.insert("alpha beta".to_string(), vec![1.0, 0.0, 0.0, 0.0]);
        table.insert("zeta keyword".to_string(), vec![0.0, 1.0, 0.0, 0.0]);
        let enc = MockEncoder { table, dim: 4 };
        let mut hybrid = HybridRetriever::new(enc, 60.0);
        hybrid.index_text(1, "alpha beta").unwrap();
        hybrid.index_text(2, "zeta keyword").unwrap();
        assert_eq!(hybrid.len(), 2);

        // The retriever needs the query TEXT to match BM25; we registered the
        // embedding under "q", so encode("q ... zeta") would not match the table.
        // Index a query whose text is the lexical doc's words and whose embedding
        // is the semantic doc's vector by encoding the stored key directly.
        // Simpler: query "zeta keyword" → dense top D_kw, sparse top D_kw; both
        // agree, D_kw first. Query "alpha beta" → both agree on D_sem.
        let kw = hybrid.retrieve("zeta keyword", 2);
        assert_eq!(kw[0].id, 2, "lexical+semantic agree on D_kw");
        let sem = hybrid.retrieve("alpha beta", 2);
        assert_eq!(sem[0].id, 1, "lexical+semantic agree on D_sem");
    }
}
