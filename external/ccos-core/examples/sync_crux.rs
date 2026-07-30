//! # Sync crux — the distributed multi-agent store, measured on its four claims.
//!
//! Two agents build **disjoint** knowledge (A owns `db.rs`, B owns `api.rs` which calls into
//! `db.rs`). The crux: no single agent can answer a cross-boundary causal question — and a
//! distributed store is only admissible in CCOS if federation keeps the moat: **no network, no
//! new dependency, chain-verified transport, bit-identical convergence**. This example measures
//! all four:
//!
//! 1. **isolation** — before sync, neither agent's graph holds the `api → db` call edge;
//! 2. **exchange** — plain-JSON bundles travel over any medium (here, a `String`), every link
//!    re-verified on import;
//! 3. **convergence** — after the swap, both merged views are **bit-identical** (same canonical
//!    fingerprint), and both now hold the cross-agent causal edge;
//! 4. **tamper-evidence** — a bundle mutated in transit is *refused*, not merged.
//!
//! Run: `cargo run --release --example sync_crux` — deterministic; two runs print the same bytes.

use ccos_core::agent_session::{AgentSession, SyncBundle, SyncError};
use ccos_core::external_memory::CcosMemory;
use ccos_core::memory::EdgeType;

/// Bit-exact fingerprint of the replayable state — graph, sources, and both logs'
/// chain heads (`CcosMemory::state_fingerprint`, the store's official convergence check).
fn view_hash(m: &CcosMemory) -> String {
    m.state_fingerprint().expect("serializable")
}

fn has_cross_edge(m: &CcosMemory) -> bool {
    m.graph().edges().iter().any(|e| {
        e.edge_type == EdgeType::Calls
            && e.source.0.contains("api.rs")
            && e.target.0.contains("db.rs")
    })
}

fn main() {
    println!("# Sync crux — two agents, disjoint knowledge, one causal question\n");

    // ── The two agents ───────────────────────────────────────────────────────
    let mut a = AgentSession::new();
    a.set_agent("agent-a");
    a.ingest("src/db.rs", "pub fn timeout() -> i64 { 30 }\n");
    a.assert_support("src/db.rs", "claim:timeout-is-tested", 0.9);

    let mut b = AgentSession::new();
    b.set_agent("agent-b");
    b.ingest(
        "src/api.rs",
        "use crate::db;\npub fn handle() -> i64 { db::timeout() }\n",
    );

    // ── 1. Isolation: the cross-boundary edge exists for NOBODY ─────────────
    println!("[1] before sync — can anyone see that api::handle depends on db::timeout?");
    println!("      agent-a alone: {}", has_cross_edge(a.memory()));
    println!("      agent-b alone: {}\n", has_cross_edge(b.memory()));

    // ── 2. Exchange: chain-verified bundles over ANY medium ─────────────────
    let bundle_a = a.export_bundle(0).unwrap().to_json().unwrap();
    let bundle_b = b.export_bundle(0).unwrap().to_json().unwrap();
    println!(
        "[2] exchange — plain JSON bundles ({} B and {} B), any transport incl. sneakernet",
        bundle_a.len(),
        bundle_b.len()
    );
    let got_b = a
        .import_bundle(&SyncBundle::from_json(&bundle_b).unwrap())
        .unwrap();
    let got_a = b
        .import_bundle(&SyncBundle::from_json(&bundle_a).unwrap())
        .unwrap();
    println!("      a imported {got_b} op(s) of b; b imported {got_a} op(s) of a — every link re-verified\n");

    // ── 3. Convergence: bit-identical merged views ───────────────────────────
    let (va, vb) = (a.merged_view(), b.merged_view());
    let (ha, hb) = (view_hash(&va), view_hash(&vb));
    println!("[3] convergence — the shared brain, materialized independently on each side:");
    println!("      view(agent-a): {}", &ha[..24]);
    println!("      view(agent-b): {}", &hb[..24]);
    println!(
        "      bit-identical: {} | cross-agent edge api→db now visible: {} / {}\n",
        ha == hb,
        has_cross_edge(&va),
        has_cross_edge(&vb)
    );

    // ── 4. Tamper-evidence: a mutated bundle is refused, not merged ─────────
    let mut evil: serde_json::Value = serde_json::from_str(&bundle_a).unwrap();
    evil["ops"][0]["Ingest"]["source"] = serde_json::Value::from("pub fn timeout() -> i64 { 0 }\n");
    let refused = b.import_bundle(&SyncBundle::from_json(&evil.to_string()).unwrap());
    println!("[4] a bundle mutated in transit (timeout 30 → 0):");
    match refused {
        Err(SyncError::Tampered(detail)) => println!("      ✗ REFUSED — {detail}"),
        other => println!("      unexpected: {other:?}"),
    }

    println!(
        "\n→ Federation without giving up the moat: no network stack, no consensus protocol, no\n\
         new dependency — an agent's history is one hash-chained log, sharing is the exchange of\n\
         verified segments, and the merged view is a pure function of the known logs, so agents\n\
         holding the same logs converge bit-for-bit. Equivocation (one agent, two histories) is\n\
         caught by the same chain (see agent_session tests). docs/SYNC.md has the full contract."
    );
}
