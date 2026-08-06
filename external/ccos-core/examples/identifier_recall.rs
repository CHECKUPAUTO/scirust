//! Do natural-language queries find code by its **identifier** names? Code is written
//! in `snake_case` / `camelCase`, queries in prose. Before subword tokenization a
//! query like "connection pool acquire" shared **zero** tokens with the identifier
//! `connection_pool_acquire`, so the semantic (TF-IDF) signal was 0 and only the
//! substring lexical fallback could match. The tokenizer now splits identifiers into
//! subwords, so the semantic signal works — this measures the resulting recall.
//!
//! Run: `cargo run --release --example identifier_recall`

use ccos_core::embeddings::tokenize;
use ccos_core::external_memory::{CcosMemory, ExternalMemory, Recall};

/// (file, identifier function name, natural-language query for it).
const CASES: &[(&str, &str, &str)] = &[
    ("db", "connection_pool_acquire", "connection pool acquire"),
    (
        "cache",
        "evict_least_recently_used",
        "evict least recently used",
    ),
    ("auth", "verify_session_token", "verify session token"),
    ("queue", "drain_pending_messages", "drain pending messages"),
    (
        "retry",
        "exponential_backoff_delay",
        "exponential backoff delay",
    ),
    ("parse", "tokenize_source_buffer", "tokenize source buffer"),
];

fn main() {
    println!("# Natural-language → code-identifier recall (subword tokenization)\n");

    // The mechanism, made concrete: a query now shares all its tokens with the identifier.
    let q = tokenize("connection pool acquire");
    let id = tokenize("connection_pool_acquire");
    let overlap = q.iter().filter(|t| id.contains(t)).count();
    println!(
        "token overlap  query{q:?}  vs  identifier{id:?}  = {overlap}/{} (was 0 pre-split)\n",
        q.len()
    );

    let mut mem = CcosMemory::new();
    for (f, ident, _) in CASES {
        mem.ingest_source(
            &format!("src/{f}.rs"),
            &format!("pub fn {ident}() -> u32 {{ 0 }}\n"),
        );
    }
    // A little filler so the ranking has distractors.
    for i in 0..20 {
        mem.ingest_source(
            &format!("src/misc_{i}.rs"),
            &format!("pub fn helper_{i}() {{}}\n"),
        );
    }

    let mut hits = 0usize;
    let mut total_rank = 0usize;
    for (f, _, query) in CASES {
        let win = mem.recall(&Recall::semantic(*query), 4096);
        let target = format!("file:src/{f}.rs");
        let rank = win
            .items
            .iter()
            .position(|i| i.uri == target)
            .map_or(999, |p| p + 1);
        if rank <= 2 {
            hits += 1;
        }
        total_rank += rank;
        println!("  {:32}→ {target} at rank {rank}", format!("\"{query}\""));
    }
    println!(
        "\nrecall@2: {hits}/{}   mean rank: {:.1}   — the semantic signal now resolves identifiers",
        CASES.len(),
        total_rank as f64 / CASES.len() as f64,
    );
    println!("not just the lexical substring fallback. Deterministic; recall stays replayable.");
}
