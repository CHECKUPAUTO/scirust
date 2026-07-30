//! Pro **OctaSoma semantic memory** walkthrough — fully offline and deterministic.
//!
//! Shows the whole contract in one run: the community-tier refusal (no silent
//! downgrade, the core stays usable), the Pro unlock, region-scoped semantic
//! anchors, and the end-to-end composition — ingest into the causal graph, build
//! the semantic index from it, and recall with *anchor-first semantic entry →
//! causal expansion*. Uses octasoma's deterministic `HashEmbedder`, so the output
//! is bit-identical across runs and machines — swap in `octasoma::OllamaEmbedder`
//! for real semantics (that trades away replay-exactness; see `src/octa_index.rs`).
//!
//! Run with: `cargo run --example octasoma_semantic --features octasoma`

use ccos_core::external_memory::{CcosMemory, ExternalMemory};
use ccos_core::license::{License, Licensing};
use ccos_core::octa_index::{
    recall_semantic, recall_semantic_calibrated, SemanticFeedback, SemanticMemoryAccess,
};
use octasoma::HashEmbedder;

fn main() {
    let now = 1_000u64;

    // 1. Community tier: the OctaSoma backend is locked, explicitly.
    match SemanticMemoryAccess::unlock(&Licensing::community(), now) {
        Err(e) => println!("[community] refused as designed: {e}"),
        Ok(_) => unreachable!("community tier must not unlock a Pro feature"),
    }

    // 2. Pro tier (in production the license comes from an offline-verified,
    //    ed25519-signed token — see `src/license.rs` and docs/DEPLOYMENT.md §4).
    let pro = Licensing::licensed(License {
        licensee: "demo".into(),
        expires_at: None,
        machine: None,
    });
    let access = SemanticMemoryAccess::unlock(&pro, now).expect("pro tier unlocks octasoma");

    // 3. One small semantic index per causal region — the validated deployment.
    let mut idx = access.sharded_index(HashEmbedder::new(128));
    idx.index_node_in(
        "src/db.rs",
        "sym:src/db.rs:query",
        "fn query(conn: &Conn) -> Rows",
    );
    idx.index_node_in("src/db.rs", "sym:src/db.rs:pool", "fn pool() -> Pool");
    idx.index_node_in("src/cache.rs", "sym:src/cache.rs:get", "fn get(key: &str)");
    println!(
        "[pro] indexed {} nodes across {} causal regions",
        idx.len(),
        idx.regions()
    );

    // 4. Semantic anchors *within* a known causal region (the 99 %-hit path):
    //    CCOS resolves the region causally, OctaSoma reranks inside it, and the
    //    returned URIs are the anchors CCOS expands through the causal graph.
    let anchors = idx.semantic_anchors_in("src/db.rs", "fn query(conn: &Conn) -> Rows", 2);
    for (uri, score) in &anchors {
        println!("[pro] anchor {uri} (score {score:.3})");
    }

    // 5. End to end: ingest real source into the causal graph, build the semantic
    //    index from the graph in one deterministic pass, and recall — OctaSoma
    //    picks the entry node, CCOS assembles the window with its usual
    //    Recall::Around machinery (budgets, proximity decay, replay all intact).
    let mut mem = CcosMemory::new();
    mem.ingest_source(
        "src/db.rs",
        "pub fn query() -> i64 { 1 }\npub fn pool() -> i64 { 2 }\n",
    );
    mem.ingest_source("src/cache.rs", "pub fn get() -> i64 { 3 }\n");

    let graph_idx = access.sharded_index_from_graph(HashEmbedder::new(128), mem.graph());
    let window = recall_semantic(&mem, &graph_idx, "pub fn query() -> i64 { 1 }", 512);
    println!(
        "[pro] '{}' window: {} items, {} tokens",
        window.strategy,
        window.items.len(),
        window.tokens
    );
    for item in window.items.iter().take(3) {
        println!("[pro]   {} ({})", item.uri, item.kind);
    }

    // 6. The explicit feedback channel closes the loop: the agent reports which
    //    anchors actually helped, and the labels certify a conformal score floor —
    //    anchors clearing it contain the relevant one with probability ≥ 1 − α
    //    (for workloads exchangeable with the labels). Anchors below the floor are
    //    refused *visibly* and the window falls back to the free lexical entry.
    let mut fb = SemanticFeedback::new();
    let alpha = 0.25;
    println!(
        "[pro] feedback: 0 labels → certified floor at alpha={alpha}: {:?} (no fake threshold)",
        fb.certified_score_floor(alpha)
    );
    for query in [
        "find the row query",
        "where is the sql query",
        "query the db",
    ] {
        fb.record(query, "sym:src/db.rs:query", 1.0, true);
    }
    let floor = fb
        .certified_score_floor(alpha)
        .expect("3 labels support alpha=0.25");
    println!(
        "[pro] feedback: {} labels → certified floor {floor:.3}",
        fb.len()
    );

    let w = recall_semantic_calibrated(
        &mem,
        &graph_idx,
        &fb,
        "pub fn query() -> i64 { 1 }",
        512,
        alpha,
    );
    println!("[pro] exact query      → '{}'", w.strategy);
    let w = recall_semantic_calibrated(&mem, &graph_idx, &fb, "unrelated gibberish", 512, alpha);
    println!("[pro] off-corpus query → '{}'", w.strategy);
}
