//! `rag_challenge` — pure semantic retrieval as an auditable alternative to RAG.
//!
//! Run it:
//! ```text
//! cargo run -p scirust-retrieval --example rag_challenge
//! ```
//!
//! Everything printed below is **deterministic**: the same build prints the same
//! numbers, byte for byte. There is no generation step to hallucinate — every
//! score is a reproducible inner product.
//!
//! The text retriever here is driven by a small **deterministic bag-of-words**
//! encoder (defined at the bottom), exactly like the crate's own tests: the
//! engine is encoder-agnostic, so retrieval quality is the *encoder's* quality.
//! Plug in a trained transformer — or ccos's embeddings — for production. The
//! genuine *semantic* story (relating things that do not lexically overlap) is
//! shown in §5 via the feedback-driven [`ImprovementLoop`], which learns the
//! mapping from confirmed (query, document) pairs.

use std::collections::{HashMap, HashSet};

use scirust_license::{License, LicenseError, Module, demo_root, demo_vendor, verify_license};
use scirust_retrieval::rag::{ContextBudget, RagContext};
use scirust_retrieval::{
    ContrastiveConfig, Encoder, HybridRetriever, RetrievalAccess, Scored, metrics,
};

fn main() {
    println!("== SciRust pure semantic retrieval — the auditable alternative to RAG ==\n");

    // ── §1. Premium gate ────────────────────────────────────────────────────
    // Retrieval is a licensed premium module. The flagship retrievers are built
    // through a RetrievalAccess capability token, obtained only by unlocking a
    // verified entitlement that covers Module::Retrieval.
    println!("§1. Premium gate (Module::Retrieval)");
    let entitled = issue_entitlements(&[Module::Retrieval]);
    let core_only = issue_entitlements(&[Module::Core]);
    match RetrievalAccess::unlock(&core_only)
    {
        Err(LicenseError::NotEntitled(m)) => println!("    core-only license  -> refused ({m})"),
        other => panic!("a core-only license must be refused, got {other:?}"),
    }
    let access = RetrievalAccess::unlock(&entitled).expect("retrieval license unlocks");
    println!(
        "    retrieval license -> unlocked: {:?}\n",
        RetrievalAccess::MODULE
    );

    // ── §2. Index + pure retrieval (deterministic) ──────────────────────────
    println!("§2. Index a corpus, retrieve the answer directly (top-k by cosine)");
    let corpus = sample_corpus();
    let mut dense = access.semantic_retriever(BagOfWords::new(&corpus_texts(&corpus)));
    for (id, text) in &corpus
    {
        dense.index_text(*id, text).expect("index");
    }
    let query = "wet roads after rain";
    let hits = dense.retrieve(query, 3);
    println!("    query: {query:?}");
    print_hits(&hits, &corpus);
    // Determinism: the identical query yields the identical ranking, bit for bit.
    let again = dense.retrieve(query, 3);
    assert_eq!(hits, again, "retrieval must be deterministic");
    println!("    re-run identical: {}\n", hits == again);

    // ── §3. Hybrid retrieval (dense + BM25, fused by RRF) ────────────────────
    // A rare keyword that the dense bag-of-words may dilute is pinned by BM25;
    // reciprocal-rank fusion combines both signals.
    println!("§3. Hybrid (dense + BM25 lexical, reciprocal-rank fusion)");
    let mut hybrid: HybridRetriever<BagOfWords> =
        access.hybrid_retriever(BagOfWords::new(&corpus_texts(&corpus)), 60.0);
    for (id, text) in &corpus
    {
        hybrid.index_text(*id, text).expect("index");
    }
    let kw = "hydroplaning";
    let hhits = hybrid.retrieve(kw, 3);
    println!("    query: {kw:?} (a rare keyword in exactly one document)");
    print_hits(&hhits, &corpus);
    println!();

    // ── §4. Quality is a measured number ────────────────────────────────────
    println!("§4. Quality as numbers (Recall@1, MRR, nDCG@3) on a labelled eval");
    report_metrics(&mut dense, &eval_set());
    println!();

    // ── §5. It learns semantics from feedback (the real semantic story) ─────
    // "Two views": query i = [eᵢ;0], document i = [0;eᵢ]. Halves are disjoint, so
    // a query and its document have raw cosine 0 — the head must *learn* the
    // cross-view mapping from confirmed feedback. Recall@1 climbs from chance to
    // perfect, deterministically.
    println!("§5. Learns from feedback: Recall@1 climbs as relevance pairs accumulate");
    let n = 8usize;
    let corpus_vecs: Vec<(u64, Vec<f32>)> = (0..n).map(|i| (i as u64, doc_view(i, n))).collect();
    let eval_vecs: Vec<(Vec<f32>, u64)> = (0..n).map(|i| (query_view(i, n), i as u64)).collect();
    let cfg = ContrastiveConfig {
        epochs: 300,
        lr: 0.05,
        temperature: 0.1,
    };
    let mut loop_ = access.improvement_loop(2 * n, 8, 7, cfg);
    let mut curve = vec![loop_.evaluate_recall_at_k(&eval_vecs, &corpus_vecs, 1)];
    for c in 0..(n / 2)
    {
        loop_.record(&query_view(2 * c, n), &doc_view(2 * c, n));
        loop_.record(&query_view(2 * c + 1, n), &doc_view(2 * c + 1, n));
        loop_.train_cycle();
        curve.push(loop_.evaluate_recall_at_k(&eval_vecs, &corpus_vecs, 1));
    }
    let pct: Vec<String> = curve.iter().map(|r| format!("{:.0}%", r * 100.0)).collect();
    println!(
        "    feedback cycles 0..{}: Recall@1 = [{}]",
        n / 2,
        pct.join(" → ")
    );
    println!("    (starts near chance, ends perfect — and the same seed always does)\n");

    // ── §6. Pure retrieval vs RAG, side by side ─────────────────────────────
    // RagContext is the crate's own bridge to a generator: it wraps retrieval
    // into a bounded augmented prompt. Pure retrieval returns the ranked answer
    // with scores DIRECTLY (auditable); RAG hands those passages to a stochastic
    // LM. Same retrieval core — RAG just adds the step you then have to trust.
    println!("§6. Pure retrieval  vs  RAG augmentation (same retrieval core)");
    let mut rag = RagContext::new(
        BagOfWords::new(&corpus_texts(&corpus)),
        "Context:\n",
        "\n---\n",
    );
    for (id, text) in &corpus
    {
        rag.index_chunk(*id, text).expect("index");
    }
    let q = "wet roads after rain";
    println!("    PURE  -> ranked passages + scores, returned directly:");
    print_hits(&dense.retrieve(q, 2), &corpus);
    let aug = rag.augment(q, ContextBudget::Chunks(2));
    println!(
        "    RAG   -> an augmented prompt ({} chunks) handed to a generator:",
        aug.chunk_ids.len()
    );
    for line in aug.prompt.lines()
    {
        println!("             | {line}");
    }
    println!(
        "\n    Pure retrieval is the answer + an audit trail. RAG adds a generation\n    \
         step — the one place a wrong or hallucinated word can enter. Same core,\n    \
         one fewer thing to trust."
    );
}

// ── helpers ─────────────────────────────────────────────────────────────────

/// Issue and verify a demo license covering `modules`, returning its entitlements.
fn issue_entitlements(modules: &[Module]) -> scirust_license::Entitlements {
    let signed = demo_vendor().issue_with_leaf(
        License::new("Demo Co", "L-DEMO", modules.to_vec(), 0, None),
        0,
    );
    verify_license(&signed, &demo_root(), 1).expect("demo license verifies")
}

/// A small thematic corpus: weather/road-safety, systems, and animals.
fn sample_corpus() -> Vec<(u64, String)> {
    [
        "rain causes wet roads and hydroplaning",
        "wet roads increase braking distance",
        "rust is a fast memory safe systems language",
        "systems programming values predictable performance",
        "cats purr when they are content",
    ]
    .iter()
    .enumerate()
    .map(|(i, s)| (i as u64, s.to_string()))
    .collect()
}

fn corpus_texts(corpus: &[(u64, String)]) -> Vec<&str> {
    corpus.iter().map(|(_, s)| s.as_str()).collect()
}

/// Eval: query text -> the one relevant document id, by topical overlap.
fn eval_set() -> Vec<(String, u64)> {
    vec![
        ("rain wet roads".to_string(), 0),
        ("braking distance on wet roads".to_string(), 1),
        ("memory safe systems language".to_string(), 2),
    ]
}

fn report_metrics<E: Encoder>(
    dense: &mut scirust_retrieval::SemanticRetriever<E>,
    eval: &[(String, u64)],
) {
    let mut recall = 0.0;
    let mut mrr_queries: Vec<(Vec<u64>, HashSet<u64>)> = Vec::new();
    let mut ndcg = 0.0;
    for (q, rel) in eval
    {
        let ranked: Vec<u64> = dense.retrieve(q, 3).into_iter().map(|s| s.id).collect();
        let relevant: HashSet<u64> = [*rel].into_iter().collect();
        recall += metrics::recall_at_k(&ranked, &relevant, 1);
        let gains: HashMap<u64, f64> = [(*rel, 1.0)].into_iter().collect();
        ndcg += metrics::ndcg_at_k(&ranked, &gains, 3);
        mrr_queries.push((ranked, relevant));
    }
    let n = eval.len() as f64;
    println!("    Recall@1 = {:.3}", recall / n);
    println!(
        "    MRR      = {:.3}",
        metrics::mean_reciprocal_rank(&mrr_queries)
    );
    println!("    nDCG@3   = {:.3}", ndcg / n);
}

fn print_hits(hits: &[Scored], corpus: &[(u64, String)]) {
    for h in hits
    {
        let text = corpus
            .iter()
            .find(|(i, _)| *i == h.id)
            .map(|(_, s)| s.as_str())
            .unwrap_or("?");
        println!("      [{:.3}] #{}  {text}", h.score, h.id);
    }
}

// "Two views" vectors for the feedback-learning demo (§5).
fn query_view(i: usize, n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 2 * n];
    v[i] = 1.0;
    v
}
fn doc_view(i: usize, n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; 2 * n];
    v[n + i] = 1.0;
    v
}

/// A deterministic, L2-normalised bag-of-words encoder. Fixed vocabulary from the
/// corpus, so embeddings are reproducible — unlike a randomly-initialised
/// transformer, whose cosine ranking would be noise. Swap in any real encoder.
struct BagOfWords {
    vocab: HashMap<String, usize>,
    dim: usize,
}

impl BagOfWords {
    fn new(texts: &[&str]) -> Self {
        let mut vocab = HashMap::new();
        for t in texts
        {
            for w in t.split_whitespace()
            {
                let next = vocab.len();
                vocab.entry(w.to_string()).or_insert(next);
            }
        }
        Self {
            dim: vocab.len(),
            vocab,
        }
    }
}

impl Encoder for BagOfWords {
    fn embedding_dim(&self) -> usize {
        self.dim
    }

    fn encode(&mut self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        for w in text.split_whitespace()
        {
            if let Some(&i) = self.vocab.get(w)
            {
                v[i] += 1.0;
            }
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0
        {
            for x in &mut v
            {
                *x /= norm;
            }
        }
        v
    }
}
