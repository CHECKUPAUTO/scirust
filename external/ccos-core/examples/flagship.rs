//! # CCOS flagship — three things a RAG stack can't do, demonstrated deterministically.
//!
//! One event-sourced session, three acts, each ending in a **measured** verdict:
//!
//! 1. **Replay == live.** An agent's working memory is *event-sourced*: every ingest, failure
//!    signal, recall and belief assertion is logged, so the exact state is reconstructible
//!    bit-for-bit. A probabilistic RAG stack cannot replay the context an agent actually saw.
//! 2. **Contested knowledge.** Evidence carries *polarity*: `Supports` / `Contradicts` are distinct
//!    typed edges, so a lone refutation is surfaced *as* a refutation and `qbelief.conflict` flags a
//!    claim as contested. Vector similarity is polarity-blind — a refutation is "related" to the
//!    claim, so it is retrieved without ever being labelled as opposition.
//! 3. **Beating RAG on its own turf.** On a *semantic* recall task where a query and its answer share
//!    zero vocabulary, CCOS's deterministic LSA encoder recovers the synonymy a lexical retriever
//!    structurally misses — bit-exactly reproducibly (a transformer embedder would too, but could
//!    not be replayed bit-for-bit).
//!
//! Everything printed below is the REAL output of the run — deterministic, zero external deps.
//!
//! Run: `cargo run --release --example flagship`

use ccos_core::agent_session::AgentSession;
use ccos_core::embeddings::{tokenize, TfidfEmbedder};
use ccos_core::external_memory::{ExternalMemory, Recall};
use ccos_core::memory::NodeId;
use ccos_core::retrieval::{metrics, CcosEncoder, Encoder, LsaEncoder, SemanticRetriever};
use std::collections::HashSet;

/// A tiny scheduler service: a causal chain api → repo → db (the deadline bug lives in db.rs).
const WORKSPACE: &[(&str, &str)] = &[
    (
        "src/db.rs",
        "// admission timeout\npub fn timeout_ms() -> i64 { 5 } // BUG: far too low\n",
    ),
    (
        "src/repo.rs",
        "use crate::db;\npub fn budget() -> i64 { db::timeout_ms() * 2 }\n",
    ),
    (
        "src/api.rs",
        "use crate::repo;\npub fn admit() -> i64 { repo::budget() + 1 }\n",
    ),
    ("src/log.rs", "pub fn info(_m: &str) {}\n"),
];

const FAULT: &str = "file:src/api.rs";
const CLAIM: &str = "the scheduler admits every queued job within the deadline";

fn main() {
    println!("# CCOS flagship — three things a RAG stack can't do, all deterministic\n");

    // ═══ Acts I + II run on ONE event-sourced session ═══════════════════════════════════════
    let mut s = AgentSession::new();
    for (uri, src) in WORKSPACE {
        s.ingest(uri, src);
    }
    s.signal_failure(FAULT, 3).ok();
    let _ = s.recall(Recall::around(FAULT), 4000);

    // Four confirmations and one lone refutation about the same claim — recorded as typed edges.
    let support = [
        "nominal load: the scheduler admitted every queued job within the deadline",
        "staging: queued jobs were admitted within the deadline",
        "soak test: the scheduler met the deadline for all admitted jobs",
        "canary: every queued job was admitted within the deadline",
    ];
    let dissent = "burst load: the scheduler starved a queued job and missed its deadline";
    for (i, e) in support.iter().enumerate() {
        s.assert_support(&format!("obs{i}: {e}"), CLAIM, 0.9);
    }
    s.assert_contradiction(&format!("obsX: {dissent}"), CLAIM, 0.9);

    // ── Act I · Replay == live ──────────────────────────────────────────────────────────────
    let replayed = s.replay_to(s.len());
    let live = s.memory();
    let claim_id = NodeId(CLAIM.to_string());
    let q_live = live.graph().qbelief(&claim_id);
    let q_replay = replayed.graph().qbelief(&claim_id);
    let replay_exact = live.stats().nodes == replayed.stats().nodes
        && q_live.belief == q_replay.belief
        && q_live.conflict == q_replay.conflict
        && q_live.support == q_replay.support
        && q_live.contradiction == q_replay.contradiction;

    println!("── Act I · Replay == live (event-sourced, auditable) ───────────────────────────");
    println!(
        "  session: {} ingests + 1 failure + 1 recall + {} assertions  →  {} nodes",
        WORKSPACE.len(),
        support.len() + 1,
        live.stats().nodes,
    );
    println!(
        "  replay_to({}) rebuilds {} nodes and an identical belief state  →  replay == live: {}",
        s.len(),
        replayed.stats().nodes,
        if replay_exact {
            "OK ✓"
        } else {
            "MISMATCH ✗"
        },
    );
    assert!(
        replay_exact,
        "replay must reconstruct the live state bit-for-bit"
    );

    // ── Act II · Contested knowledge (typed polarity vs. blind similarity) ──────────────────
    println!("\n── Act II · Contested knowledge (polarity a similarity score can't represent) ──");
    println!("  claim: \"{CLAIM}\"");
    println!(
        "  evidence: {} confirmations, 1 refutation  →  qbelief  belief {:.2}  conflict {:.2}  (S {:.1}, C {:.1})",
        support.len(),
        q_live.belief,
        q_live.conflict,
        q_live.support,
        q_live.contradiction,
    );
    // A similarity retriever ranks the lone dissent as highly "related" to the claim — inside the
    // confirmation cosine band — so no threshold separates confirmation from refutation.
    let mut corpus: Vec<Vec<String>> = support.iter().map(|t| tokenize(t)).collect();
    corpus.push(tokenize(dissent));
    let mut fit = corpus.clone();
    fit.push(tokenize(CLAIM));
    let mut tf = TfidfEmbedder::new(128);
    tf.fit(&fit);
    let qv = tf.embed(&tokenize(CLAIM));
    let cos: Vec<f64> = corpus
        .iter()
        .map(|t| TfidfEmbedder::cosine(&qv, &tf.embed(t)))
        .collect();
    let dissent_cos = *cos.last().unwrap();
    let (min_s, max_s) = cos[..support.len()]
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &c| {
            (lo.min(c), hi.max(c))
        });
    let inside = dissent_cos >= min_s && dissent_cos <= max_s;
    println!(
        "  RAG view: the refutation's cosine to the claim is {:.2}; confirmations span [{:.2}, {:.2}]",
        dissent_cos, min_s, max_s,
    );
    println!(
        "    → the dissent sits {} the confirmation band → similarity is polarity-blind; only the",
        if inside { "INSIDE" } else { "outside" },
    );
    println!(
        "      typed Contradicts edge (conflict {:.2} > 0) flags the claim as contested.",
        q_live.conflict,
    );

    // ── Act III · Beating RAG on its own turf: semantic recall of synonyms ──────────────────
    // (query, answer, [bridge, bridge]) — the query and its answer share ZERO vocabulary; only the
    // bridge docs, where both vocabularies co-occur, link them. Lexical RAG needs a literal match
    // and misses; the LSA encoder learns "car ≈ automobile" from the bridges and retrieves it.
    let concepts: [(&str, &str, [&str; 2]); 6] = [
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
    let decoys = [
        "banana fruit yellow tropical",
        "mountain river valley forest",
        "guitar music melody rhythm",
        "ocean wave beach sand",
        "planet star galaxy cosmos",
        "bread flour bakery oven",
    ];
    let mut docs: Vec<String> = Vec::new();
    let mut answer_id: Vec<u64> = Vec::new();
    for (_, ans, _) in &concepts {
        answer_id.push(docs.len() as u64);
        docs.push((*ans).to_string());
    }
    for (_, _, bridges) in &concepts {
        for b in bridges {
            docs.push((*b).to_string());
        }
    }
    for d in &decoys {
        docs.push((*d).to_string());
    }
    let n = docs.len();

    let (dim, rank) = (128usize, 12usize);
    let mut lexical = SemanticRetriever::new(CcosEncoder::fit(&docs, dim)); // TF-IDF = a lexical RAG
    let mut lsa = SemanticRetriever::new(LsaEncoder::fit(&docs, dim, rank)); // CCOS's deterministic LSA
    for (id, d) in docs.iter().enumerate() {
        lexical.index_text(id as u64, d).unwrap();
        lsa.index_text(id as u64, d).unwrap();
    }
    let (lex_r1, lex_mrr) = eval(&mut lexical, &concepts, &answer_id, n);
    let (lsa_r1, lsa_mrr) = eval(&mut lsa, &concepts, &answer_id, n);

    println!("\n── Act III · Beating RAG on its own turf — semantic recall (synonymy) ──────────");
    println!(
        "  corpus: {n} docs, {} synonym queries (query ↔ answer share ZERO vocabulary)",
        concepts.len(),
    );
    println!("  {:<24}{:>9}{:>8}", "retriever", "Recall@1", "MRR");
    println!(
        "  {:<24}{:>8.0}%{:>8.3}",
        "lexical RAG (TF-IDF)", lex_r1, lex_mrr
    );
    println!(
        "  {:<24}{:>8.0}%{:>8.3}",
        "CCOS LSA (dense)", lsa_r1, lsa_mrr
    );
    assert!(
        lsa_mrr > lex_mrr,
        "LSA must beat lexical RAG on synonym recall ({lsa_mrr:.3} vs {lex_mrr:.3})"
    );

    // ── Verdict ─────────────────────────────────────────────────────────────────────────────
    println!(
        "\nVerdict — three properties a RAG stack cannot offer, each shown above deterministically:\n  \
         (1) the session replays bit-for-bit (auditable, time-travel-debuggable);\n  \
         (2) a lone refutation is represented as such — conflict {:.2} > 0 — where similarity is blind;\n  \
         (3) deterministic LSA recall ({:.0}% vs {:.0}% Recall@1) beats a lexical retriever on synonymy.\n\
         All pure-Rust, zero external dependencies, replayable bit-exact. See docs/MEASUREMENT_pure_retrieval.md.",
        q_live.conflict, lsa_r1, lex_r1,
    );
}

/// Score a retriever over the synonym queries → `(Recall@1 %, MRR)`. Generic over the encoder so the
/// identical evaluation runs against both the lexical TF-IDF retriever and the LSA dense one.
fn eval<E: Encoder>(
    r: &mut SemanticRetriever<E>,
    concepts: &[(&str, &str, [&str; 2])],
    answer_id: &[u64],
    n: usize,
) -> (f64, f64) {
    let mut pairs: Vec<(Vec<u64>, HashSet<u64>)> = Vec::new();
    let mut r1 = 0.0;
    for (i, (q, _, _)) in concepts.iter().enumerate() {
        let ranking: Vec<u64> = r.retrieve(q, n).into_iter().map(|hit| hit.id).collect();
        let rel: HashSet<u64> = [answer_id[i]].into_iter().collect();
        r1 += metrics::recall_at_k(&ranking, &rel, 1);
        pairs.push((ranking, rel));
    }
    (
        100.0 * r1 / concepts.len() as f64,
        metrics::mean_reciprocal_rank(&pairs),
    )
}
