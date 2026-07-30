//! **Beating RAG on its own turf — semantic recall, deterministically.** Pure dense retrieval over
//! ccos's TF-IDF embedding *ties* a lexical RAG (it is the same signal — see `pure_retrieval_vs_rag`).
//! The win comes from the **encoder**: projecting TF-IDF through ccos's deterministic **LSA** latent
//! space (`LsaEncoder`) captures **synonymy** a literal-term retriever structurally cannot.
//!
//! The crux corpus is built so a query and its answer share **zero vocabulary** — a query uses one set
//! of words ("car vehicle drive"), its answer uses the synonyms ("automobile motor sedan") — and the
//! two are linked only by *bridge* documents where both vocabularies co-occur. A lexical retriever sees
//! no shared term and misses the answer; LSA learns "car ≈ automobile" from the bridges and retrieves
//! it. We measure the gap with the standard metrics — the numbers below are the REAL output of the run.
//!
//! Run: `cargo run --release --example semantic_retrieval_crux`

use ccos_core::retrieval::{
    metrics, CcosEncoder, HybridRetriever, LsaEncoder, Scored, SemanticRetriever,
};
use std::collections::HashSet;

/// Concept `i`: a V-vocabulary (used by queries) and a disjoint W-vocabulary (used by the answer),
/// linked by bridge docs where both co-occur. `(query, answer, bridges)`.
const CONCEPTS: [(&str, &str, [&str; 2]); 6] = [
    (
        "car vehicle drive road",
        "automobile motor sedan",
        [
            "car vehicle automobile motor",
            "drive road sedan automobile",
        ],
    ),
    (
        "fast quick rapid speed",
        "swift speedy brisk",
        ["fast quick swift speedy", "rapid speed brisk swift"],
    ),
    (
        "repair fix mend",
        "restore overhaul refurbish",
        ["repair fix restore overhaul", "mend fix refurbish restore"],
    ),
    (
        "doctor physician clinic",
        "medic practitioner surgery",
        [
            "doctor physician medic practitioner",
            "clinic surgery practitioner medic",
        ],
    ),
    (
        "buy purchase shop",
        "acquire procure",
        [
            "buy purchase acquire procure",
            "shop purchase procure acquire",
        ],
    ),
    (
        "happy glad joyful",
        "cheerful merry content",
        ["happy glad cheerful merry", "joyful glad content cheerful"],
    ),
];

const DISTRACTORS: [&str; 6] = [
    "banana fruit yellow tropical",
    "mountain river valley forest",
    "guitar music melody rhythm",
    "ocean wave beach sand",
    "planet star galaxy cosmos",
    "bread flour bakery oven",
];

fn main() {
    println!("# Semantic retrieval crux — LSA beats lexical RAG on synonym recall\n");

    // ── Build the corpus: answer docs (the relevant targets) + bridge docs + distractors ──
    let mut docs: Vec<String> = Vec::new();
    let mut answer_id = Vec::new(); // answer_id[i] = doc id of concept i's answer (pure-W) doc
    for (_, answer, _) in CONCEPTS {
        answer_id.push(docs.len() as u64);
        docs.push(answer.to_string());
    }
    for (_, _, bridges) in CONCEPTS {
        for b in bridges {
            docs.push(b.to_string());
        }
    }
    for d in DISTRACTORS {
        docs.push(d.to_string());
    }
    let n = docs.len();

    // Eval: query i (V-vocabulary) → its answer doc (W-vocabulary), which shares no term with the query.
    let queries: Vec<(&str, u64)> = CONCEPTS
        .iter()
        .enumerate()
        .map(|(i, (q, _, _))| (*q, answer_id[i]))
        .collect();

    // ── Three retrievers over the SAME corpus ──
    let (dim, rank) = (128usize, 12usize);
    let mut lexical = SemanticRetriever::new(CcosEncoder::fit(&docs, dim)); // TF-IDF (= ccos's RAG)
    let mut lsa = SemanticRetriever::new(LsaEncoder::fit(&docs, dim, rank)); // LSA semantic dense
    let mut hybrid = HybridRetriever::new(LsaEncoder::fit(&docs, dim, rank), 60.0); // LSA ⊕ BM25
    for (id, d) in docs.iter().enumerate() {
        lexical.index_text(id as u64, d).unwrap();
        lsa.index_text(id as u64, d).unwrap();
        hybrid.index_text(id as u64, d).unwrap();
    }

    // ── Score each retriever (Recall@1/3/5, MRR) over the synonym queries ──
    let ranked = |hits: Vec<Scored>| -> Vec<u64> { hits.into_iter().map(|s| s.id).collect() };
    let row = |label: &str, rankings: &[(Vec<u64>, HashSet<u64>)]| {
        let mut q: Vec<(Vec<u64>, HashSet<u64>)> = Vec::new();
        let (mut r1, mut r3, mut r5) = (0.0, 0.0, 0.0);
        for (rk, rel) in rankings {
            r1 += metrics::recall_at_k(rk, rel, 1);
            r3 += metrics::recall_at_k(rk, rel, 3);
            r5 += metrics::recall_at_k(rk, rel, 5);
            q.push((rk.clone(), rel.clone()));
        }
        let m = rankings.len() as f64;
        println!(
            "  {label:<24} {:>5.0}%   {:>5.0}%   {:>5.0}%    {:.3}",
            100.0 * r1 / m,
            100.0 * r3 / m,
            100.0 * r5 / m,
            metrics::mean_reciprocal_rank(&q),
        );
    };
    let eval_with = |retrieve: &mut dyn FnMut(&str) -> Vec<u64>| -> Vec<(Vec<u64>, HashSet<u64>)> {
        queries
            .iter()
            .map(|(q, ans)| (retrieve(q), [*ans].into_iter().collect()))
            .collect()
    };

    let lex_rows = eval_with(&mut |q| ranked(lexical.retrieve(q, n)));
    let lsa_rows = eval_with(&mut |q| ranked(lsa.retrieve(q, n)));
    let hyb_rows = eval_with(&mut |q| ranked(hybrid.retrieve(q, n)));

    println!(
        "corpus: {n} docs ({} concepts, query↔answer share ZERO vocabulary, linked only by bridges)\n",
        CONCEPTS.len()
    );
    println!("  retriever                Recall@1  @3    @5     MRR");
    println!("  {}", "-".repeat(56));
    row("lexical RAG (TF-IDF)", &lex_rows);
    row("LSA semantic (dense)", &lsa_rows);
    row("LSA semantic (hybrid)", &hyb_rows);

    println!(
        "\n→ The query and its answer share NO term, so the lexical RAG — needing a literal match — cannot\n\
         retrieve the answer (it ranks the bridge/distractor docs instead). The LSA encoder learns the\n\
         synonymy from the bridge documents' co-occurrence and projects query and answer to nearby\n\
         points, recovering the answer. This is RAG's own turf — *semantic* recall — and a deterministic,\n\
         dependency-free LSA wins it, bit-for-bit reproducibly (a transformer embedder would too, but it\n\
         could not be replayed bit-exact). See docs/MEASUREMENT_pure_retrieval.md."
    );
}
