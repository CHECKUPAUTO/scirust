//! **Quarantined neural embedder vs the deterministic encoders** — same synonym crux, same metrics.
//!
//! `semantic_retrieval_crux` shows the deterministic LSA encoder beating a lexical retriever on a
//! corpus where query and answer share **zero** vocabulary. This example adds the third contender:
//! the **quarantined neural embedder** (`neural-embed` feature — a local Ollama-style
//! `/api/embeddings` endpoint). The interesting question it answers *locally, on your machine*:
//! how much of the transformer's semantic-recall advantage does the zero-dependency LSA already
//! capture — and what does the quarantine cost you (bit-exact replay) to close the rest?
//!
//! Requires a local embedding server; without one it prints how to start one and exits cleanly:
//!
//! ```bash
//! ollama pull nomic-embed-text && ollama serve &
//! cargo run --release --features neural-embed --example neural_vs_lsa
//! # env overrides: CCOS_EMBED_ENDPOINT (default http://127.0.0.1:11434),
//! #                CCOS_EMBED_MODEL    (default nomic-embed-text)
//! ```
//!
//! The lexical and LSA rows are bit-for-bit reproducible; the neural row is **not** replay-exact
//! (weights/server/hardware-dependent) — that asymmetry is the point of the quarantine.

use ccos_core::neural_embed::NeuralEncoder;
use ccos_core::retrieval::{metrics, CcosEncoder, Encoder, LsaEncoder, SemanticRetriever};
use std::collections::HashSet;

/// The synonym crux: query (V-vocabulary) → answer (disjoint W-vocabulary), linked only by bridges.
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

fn eval<E: Encoder>(r: &mut SemanticRetriever<E>, answers: &[u64], n: usize) -> (f64, f64, f64) {
    let mut pairs: Vec<(Vec<u64>, HashSet<u64>)> = Vec::new();
    let (mut r1, mut r3) = (0.0, 0.0);
    for (i, (q, _, _)) in CONCEPTS.iter().enumerate() {
        let ranking: Vec<u64> = r.retrieve(q, n).into_iter().map(|s| s.id).collect();
        let rel: HashSet<u64> = [answers[i]].into_iter().collect();
        r1 += metrics::recall_at_k(&ranking, &rel, 1);
        r3 += metrics::recall_at_k(&ranking, &rel, 3);
        pairs.push((ranking, rel));
    }
    let m = CONCEPTS.len() as f64;
    (
        100.0 * r1 / m,
        100.0 * r3 / m,
        metrics::mean_reciprocal_rank(&pairs),
    )
}

fn main() {
    let endpoint = std::env::var("CCOS_EMBED_ENDPOINT")
        .unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());
    let model =
        std::env::var("CCOS_EMBED_MODEL").unwrap_or_else(|_| "nomic-embed-text".to_string());

    println!("# Neural (quarantined) vs deterministic encoders — synonym recall\n");

    // ── Corpus: answers + bridges + distractors (as in semantic_retrieval_crux) ──
    let mut docs: Vec<String> = Vec::new();
    let mut answers: Vec<u64> = Vec::new();
    for (_, ans, _) in &CONCEPTS {
        answers.push(docs.len() as u64);
        docs.push((*ans).to_string());
    }
    for (_, _, bridges) in &CONCEPTS {
        for b in bridges {
            docs.push((*b).to_string());
        }
    }
    for d in &DISTRACTORS {
        docs.push((*d).to_string());
    }
    let n = docs.len();

    // ── The two deterministic rows (always available, bit-for-bit reproducible) ──
    let (dim, rank) = (128usize, 12usize);
    let mut lexical = SemanticRetriever::new(CcosEncoder::fit(&docs, dim));
    let mut lsa = SemanticRetriever::new(LsaEncoder::fit(&docs, dim, rank));
    for (i, d) in docs.iter().enumerate() {
        lexical.index_text(i as u64, d).unwrap();
        lsa.index_text(i as u64, d).unwrap();
    }
    let lex = eval(&mut lexical, &answers, n);
    let lsa_r = eval(&mut lsa, &answers, n);

    // ── The quarantined row (needs the local server; degrade with instructions, exit 0) ──
    let neural = match NeuralEncoder::try_new(&endpoint, &model) {
        Ok(enc) => {
            let mut r = SemanticRetriever::new(enc);
            for (i, d) in docs.iter().enumerate() {
                r.index_text(i as u64, d).unwrap();
            }
            Some(eval(&mut r, &answers, n))
        }
        Err(e) => {
            eprintln!("neural row skipped — {e}");
            eprintln!(
                "start a local server first, e.g.:\n  ollama pull {model} && ollama serve\nthen rerun:\n  \
                 CCOS_EMBED_ENDPOINT={endpoint} CCOS_EMBED_MODEL={model} \\\n  \
                 cargo run --release --features neural-embed --example neural_vs_lsa"
            );
            None
        }
    };

    println!("corpus: {n} docs, 6 synonym queries (query ↔ answer share ZERO vocabulary)\n");
    println!(
        "  {:<28}{:>9}{:>8}{:>8}   replayable?",
        "encoder", "Recall@1", "@3", "MRR"
    );
    println!("  {}", "-".repeat(70));
    let row = |name: &str, (r1, r3, mrr): (f64, f64, f64), replay: &str| {
        println!("  {name:<28}{r1:>8.0}%{r3:>7.0}%{mrr:>8.3}   {replay}");
    };
    row("lexical (TF-IDF)", lex, "bit-for-bit");
    row("LSA (deterministic)", lsa_r, "bit-for-bit");
    match neural {
        Some(nr) => row(&format!("neural ({model})"), nr, "NOT replay-exact"),
        None => println!(
            "  {:<28}{:>8}{:>7}{:>8}   (endpoint unavailable — see stderr)",
            "neural (quarantined)", "—", "—", "—"
        ),
    }

    println!(
        "\n→ The deterministic LSA row is the moat-preserving default: it closes part of the semantic\n\
         gap at zero dependencies and full replayability. The neural row — when a local server is\n\
         present — shows what the rest of the gap costs: better vectors, but no bit-exact replay.\n\
         The quarantine (`neural-embed`, off by default) is what keeps that trade explicit.\n\
         See docs/NEURAL_EMBED.md."
    );
}
