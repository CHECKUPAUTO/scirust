//! **Pure retrieval challenges RAG — measured, deterministically.** SciRust's pure-retrieval
//! algorithms, distilled into `ccos_core::retrieval` (zero extra deps, bit-exact `f32`), run over the
//! embeddings CCOS already owns and are scored the way RAG benchmarks score their retrievers.
//!
//! Eval set = CCOS's own `src/*.rs` (same corpus + ground truth as `rag_crux`): for each file `A`
//! that has cross-file dependencies, the **query** is `A`'s text and the **relevant** docs are `A`'s
//! true dependency files (the file→file edges the AST resolved). We rank every other file and report,
//! side by side, the five standard metrics at k ∈ {1, 5, 10} for:
//!
//!   * **ccos RAG (lexical)** — TF-IDF cosine, the retriever `rag_crux` measures.
//!   * **pure dense** — `SemanticRetriever` over a `CcosEncoder` (TF-IDF) + an exact auditable index.
//!   * **pure hybrid** — `HybridRetriever`: dense ⊕ BM25, fused by reciprocal-rank fusion.
//!
//! Numbers are the REAL output of this run (printed below), not asserted. Run twice → identical,
//! bit for bit: the determinism/auditability a generative RAG stage cannot offer.
//!
//! Run: `cargo run --release --example pure_retrieval_vs_rag`

use ccos_core::embeddings::{tokenize, TfidfEmbedder};
use ccos_core::external_memory::{CcosMemory, ExternalMemory};
use ccos_core::retrieval::{metrics, CcosEncoder, HybridRetriever, SemanticRetriever};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Aggregate metrics for one retriever over all queries.
#[derive(Default)]
struct Agg {
    recall: [f64; 3],
    precision: [f64; 3],
    ndcg: [f64; 3],
    rr: f64,
    ap: f64,
    n: usize,
}
const KS: [usize; 3] = [1, 5, 10];

impl Agg {
    /// Score one query's full `ranking` (best-first, self already excluded) against `relevant`.
    fn add(&mut self, ranking: &[u64], relevant: &HashSet<u64>) {
        let gains: HashMap<u64, f64> = relevant.iter().map(|&id| (id, 1.0)).collect();
        for (i, &k) in KS.iter().enumerate() {
            self.recall[i] += metrics::recall_at_k(ranking, relevant, k);
            self.precision[i] += metrics::precision_at_k(ranking, relevant, k);
            self.ndcg[i] += metrics::ndcg_at_k(ranking, &gains, k);
        }
        self.rr += metrics::reciprocal_rank(ranking, relevant);
        self.ap += metrics::average_precision(ranking, relevant);
        self.n += 1;
    }
    fn row(&self, label: &str) {
        let n = self.n.max(1) as f64;
        let m = |a: [f64; 3]| (100.0 * a[0] / n, 100.0 * a[1] / n, 100.0 * a[2] / n);
        let (r1, r5, r10) = m(self.recall);
        let (p1, p5, p10) = m(self.precision);
        let (n1, n5, n10) = m(self.ndcg);
        println!(
            "  {label:<22} {r1:>4.0} {r5:>4.0} {r10:>4.0} | {p1:>4.0} {p5:>4.0} {p10:>4.0} | {n1:>4.0} {n5:>4.0} {n10:>4.0} | {:>5.3} {:>5.3}",
            self.rr / n,
            self.ap / n,
        );
    }
}

fn main() {
    println!(
        "# Pure retrieval challenges RAG — distilled SciRust retrieval vs ccos's lexical RAG\n"
    );

    // ── Corpus: CCOS's own source files, in sorted order (deterministic ids) ──
    let mut files: Vec<(String, String)> = Vec::new();
    for path in rust_files(Path::new("src")) {
        if let Ok(src) = fs::read_to_string(&path) {
            files.push((path.to_string_lossy().replace('\\', "/"), src));
        }
    }
    let n = files.len();
    let id_of: HashMap<String, usize> = files
        .iter()
        .enumerate()
        .map(|(i, (p, _))| (format!("file:{p}"), i))
        .collect();

    // ── Ground truth: AST-resolved file→file dependencies (relevant[A] = A's deps) ──
    let mut mem = CcosMemory::new();
    for (p, src) in &files {
        mem.ingest_source(p, src);
    }
    let mut relevant: Vec<HashSet<u64>> = vec![HashSet::new(); n];
    for e in mem.graph().edges() {
        if let (Some(&a), Some(&b)) = (id_of.get(&e.source.0), id_of.get(&e.target.0)) {
            if a != b {
                relevant[a].insert(b as u64);
            }
        }
    }
    let queries: Vec<usize> = (0..n).filter(|&a| !relevant[a].is_empty()).collect();
    let total_rel: usize = queries.iter().map(|&a| relevant[a].len()).sum();

    // ── Method 1: ccos RAG (lexical TF-IDF cosine), the rag_crux baseline ──
    let corpus_tokens: Vec<Vec<String>> = files.iter().map(|(_, s)| tokenize(s)).collect();
    let mut tfidf = TfidfEmbedder::new(256);
    tfidf.fit(&corpus_tokens);
    let vecs: Vec<Vec<f32>> = corpus_tokens.iter().map(|t| tfidf.embed(t)).collect();
    let lexical_rank = |a: usize| -> Vec<u64> {
        let mut scored: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != a)
            .map(|j| (j, TfidfEmbedder::cosine(&vecs[a], &vecs[j])))
            .collect();
        scored.sort_by(|x, y| {
            y.1.partial_cmp(&x.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(x.0.cmp(&y.0))
        });
        scored.into_iter().map(|(j, _)| j as u64).collect()
    };

    // ── Methods 2 & 3: pure dense + pure hybrid (distilled SciRust retrieval) ──
    let texts: Vec<String> = files.iter().map(|(_, s)| s.clone()).collect();
    let mut dense = SemanticRetriever::new(CcosEncoder::fit(&texts, 256));
    let mut hybrid = HybridRetriever::new(CcosEncoder::fit(&texts, 256), 60.0);
    for (i, (_, src)) in files.iter().enumerate() {
        dense.index_text(i as u64, src).unwrap();
        hybrid.index_text(i as u64, src).unwrap();
    }
    // Retrieve the full ranking, then drop the query's own id (a file is its own top hit).
    let strip = |a: usize, r: Vec<ccos_core::retrieval::Scored>| -> Vec<u64> {
        r.into_iter()
            .map(|s| s.id)
            .filter(|&id| id != a as u64)
            .collect()
    };

    let (mut rag, mut den, mut hyb) = (Agg::default(), Agg::default(), Agg::default());
    for &a in &queries {
        let rel = &relevant[a];
        rag.add(&lexical_rank(a), rel);
        den.add(&strip(a, dense.retrieve(&files[a].1, n)), rel);
        hyb.add(&strip(a, hybrid.retrieve(&files[a].1, n)), rel);
    }

    println!(
        "files: {n}   queries (files with deps): {}   relevant pairs: {total_rel}\n",
        queries.len()
    );
    println!(
        "  metric (%)              Recall@1/5/10 |  Prec@1/5/10  |  nDCG@1/5/10  |   MRR   MAP"
    );
    println!("  {}", "-".repeat(92));
    rag.row("ccos RAG (lexical)");
    den.row("pure dense (distilled)");
    hyb.row("pure hybrid (dense+BM25)");

    println!(
        "\n→ Reading: pure **dense** is an exact-cosine index over the SAME TF-IDF embedding ccos's RAG\n\
         uses, so it reproduces the lexical baseline — but as a clean, serialisable, auditable index\n\
         (the faithful-distillation check). pure **hybrid** fuses BM25's exact-term/IDF signal with the\n\
         dense ranking by RRF; compare its Recall/nDCG row above to see where exact lexical matching\n\
         recovers (or trades off) dependencies the dense smoothing alone ranks lower.\n\
         The decisive, un-rowable win is DETERMINISM: every number here is bit-for-bit reproducible\n\
         (fixed-order f32, id-tie-broken ranking, zero RNG, zero generative step) — re-run and it is\n\
         identical, and there is no hallucinating generator between query and result. See\n\
         docs/MEASUREMENT_pure_retrieval.md."
    );
}
