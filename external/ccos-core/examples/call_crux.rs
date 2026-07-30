//! **Call-graph crux: what do `Calls` edges recover that lexical retrieval can't?** The
//! symbol-level analogue of `rag_crux` (which did it for file→file imports). A function that
//! calls `foo()` literally contains the token `foo`, so **direct** calls are lexically visible —
//! the honest question is **transitive** call dependencies: a root that reaches a deep callee it
//! never names (root → mid → deep). Lexical similarity between root and deep is ~0 (no shared
//! vocabulary); the call graph reaches it by construction.
//!
//! Fixture: two 3-deep call chains (names chosen to share no domain tokens across hops) plus
//! decoys. Ground truth = the `Calls` edges the parser+resolver produced. We measure, for a
//! per-symbol TF-IDF baseline, the rank of the **direct** callee vs the **2-hop transitive**
//! callee among all symbols — and contrast with the call graph, which recovers both.
//!
//! Run: `cargo run --release --example call_crux`

use ccos_core::embeddings::{tokenize, TfidfEmbedder};
use ccos_core::external_memory::{CcosMemory, ExternalMemory};
use ccos_core::memory::EdgeType;
use std::collections::HashMap;

fn main() {
    // Two call chains; each hop's names share no vocabulary with the next, so a root never
    // mentions its deep (2-hop) callee. Globally-unique names ⇒ resolved by the Tier-C ladder.
    let files: &[(&str, &str)] = &[
        (
            "src/handler.rs",
            "pub fn route_request() -> i64 { load_record() }",
        ),
        (
            "src/record.rs",
            "pub fn load_record() -> i64 { open_socket() }",
        ),
        ("src/socket.rs", "pub fn open_socket() -> i64 { 3 }"),
        (
            "src/render.rs",
            "pub fn draw_frame() -> i64 { fetch_pixels() }",
        ),
        (
            "src/pixels.rs",
            "pub fn fetch_pixels() -> i64 { alloc_buffer() }",
        ),
        ("src/buffer.rs", "pub fn alloc_buffer() -> i64 { 9 }"),
        // decoys: unrelated free functions (distractors for the lexical ranking).
        ("src/audit.rs", "pub fn verify_signature() -> bool { true }"),
        ("src/cache.rs", "pub fn evict_entry() -> i64 { 0 }"),
        ("src/parse.rs", "pub fn split_tokens() -> i64 { 1 }"),
    ];

    let mut mem = CcosMemory::new();
    for (p, c) in files {
        mem.ingest_source(p, c);
    }
    let g = mem.graph();

    // All symbol nodes (id + body), for the lexical baseline.
    let mut syms: Vec<(String, String)> = g
        .node_entries()
        .filter(|(id, _)| id.0.starts_with("sym:"))
        .map(|(id, n)| (id.0.clone(), n.content.clone()))
        .collect();
    syms.sort();
    let sidx: HashMap<&str, usize> = syms
        .iter()
        .enumerate()
        .map(|(i, (s, _))| (s.as_str(), i))
        .collect();

    // Ground truth: the caller→callee Calls edges (symbol level).
    let mut calls: Vec<(usize, usize)> = Vec::new();
    for e in g.edges() {
        if e.edge_type == EdgeType::Calls {
            if let (Some(&a), Some(&b)) =
                (sidx.get(e.source.0.as_str()), sidx.get(e.target.0.as_str()))
            {
                calls.push((a, b));
            }
        }
    }
    calls.sort_unstable();
    calls.dedup();

    // adjacency for transitive (2-hop) pairs.
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); syms.len()];
    for &(a, b) in &calls {
        adj[a].push(b);
    }
    let mut two_hop: Vec<(usize, usize)> = Vec::new();
    for a in 0..syms.len() {
        for &m in &adj[a] {
            for &d in &adj[m] {
                if d != a && !adj[a].contains(&d) {
                    two_hop.push((a, d)); // a reaches d in exactly 2 hops, never directly
                }
            }
        }
    }
    two_hop.sort_unstable();
    two_hop.dedup();

    // Lexical baseline: TF-IDF over per-symbol token bags.
    let corpus: Vec<Vec<String>> = syms.iter().map(|(_, body)| tokenize(body)).collect();
    let mut tf = TfidfEmbedder::new(128);
    tf.fit(&corpus);
    let vecs: Vec<Vec<f32>> = corpus.iter().map(|t| tf.embed(t)).collect();
    let rank_of = |src: usize, tgt: usize| -> usize {
        let s = TfidfEmbedder::cosine(&vecs[src], &vecs[tgt]);
        let mut r = 1;
        for j in 0..syms.len() {
            if j != src && j != tgt && TfidfEmbedder::cosine(&vecs[src], &vecs[j]) > s {
                r += 1;
            }
        }
        r
    };
    let report = |label: &str, pairs: &[(usize, usize)]| {
        if pairs.is_empty() {
            println!("  {label:<26} (none)");
            return;
        }
        let (mut at1, mut mrr) = (0usize, 0.0f64);
        for &(s, t) in pairs {
            let r = rank_of(s, t);
            if r == 1 {
                at1 += 1;
            }
            mrr += 1.0 / r as f64;
        }
        let n = pairs.len() as f64;
        println!(
            "  {label:<26} lexical recall@1 {:>3.0}%   MRR {:.2}   (n={})",
            100.0 * at1 as f64 / n,
            mrr / n,
            pairs.len()
        );
    };

    println!("# Call-graph crux — what Calls edges recover that lexical retrieval misses\n");
    println!(
        "fixture: {} symbols, {} resolved Calls edges, {} two-hop transitive call pairs\n",
        syms.len(),
        calls.len(),
        two_hop.len()
    );
    println!(
        "LEXICAL TF-IDF (per-symbol) — rank of the callee among all symbols, by cosine to caller:"
    );
    report("DIRECT calls (1 hop)", &calls);
    report("TRANSITIVE calls (2 hop)", &two_hop);
    println!("\nCALL GRAPH — recovers both by construction (direct = the edge; transitive = its closure).");
    println!(
        "\nReading: a direct call names its callee, so lexical similarity finds it; but a root never\n\
         names its 2-hop callee, so lexical similarity collapses there — exactly the causally-distant\n\
         dependency the call graph reaches by traversal. This is the call-level analogue of the\n\
         import crux: structure recovers the cross-vocabulary links a vector retriever cannot see."
    );
}
