//! Property-based `replay == live` for the **full** graph state.
//!
//! `replay == live` is CCOS's headline invariant: re-running the recorded op log
//! into a fresh memory must reproduce the live memory exactly. The existing tests
//! check it on hand-written chains and at the *stats* (node/edge count) level
//! (`agent_session::replay_is_deterministic`); this is the randomized,
//! byte-identical counterpart — for *any* sequence of ingests, failures and
//! page-faults, a full state hash of the replayed graph (node set + every score +
//! edge set) must equal the live one's.
//!
//! The default resident cap is 5000, far above these small graphs, so nothing
//! demotes to COLD and the resident-state hash covers the whole graph.

use ccos_core::agent_session::AgentSession;
use ccos_core::external_memory::Recall;
use ccos_core::memory::{MemoryGraph, NodeId};
use proptest::prelude::*;
use sha2::{Digest, Sha256};

/// Deterministic full-state hash of a resident graph: the node set (id, label,
/// content, and every score field) and the edge set, both in sorted order so the
/// hash is independent of the resident `HashMap`'s iteration order.
fn graph_hash(graph: &MemoryGraph) -> String {
    let mut h = Sha256::new();
    h.update(graph.node_count().to_le_bytes());
    h.update(graph.edge_count().to_le_bytes());

    let mut ids: Vec<&NodeId> = graph.node_ids().collect();
    ids.sort_by(|a, b| a.0.cmp(&b.0));
    for id in ids {
        if let Some(n) = graph.node(id) {
            h.update(id.0.as_bytes());
            h.update(n.label.as_bytes());
            h.update(n.content.as_bytes());
            h.update(n.base_importance.to_le_bytes());
            h.update(n.failure_relevance.to_le_bytes());
            h.update(n.recency.to_le_bytes());
        }
    }

    let mut edges: Vec<&ccos_core::memory::GraphEdge> = graph.edges().iter().collect();
    edges.sort_by(|a, b| {
        a.source
            .0
            .cmp(&b.source.0)
            .then(a.target.0.cmp(&b.target.0))
    });
    for e in edges {
        h.update(e.source.0.as_bytes());
        h.update(e.target.0.as_bytes());
        h.update(e.weight.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// For any op stream, replaying the recorded log into a fresh memory must
    /// reproduce the live graph byte-for-byte — same node and edge sets, same
    /// scores. A divergence here would be a determinism bug in ingest / failure /
    /// page-fault, the heart of the moat.
    #[test]
    fn replay_reproduces_live_graph_byte_for_byte(
        ops in prop::collection::vec((0u8..6u8, 0u8..8u8, 0u32..50u32), 1..120)
    ) {
        let mut s = AgentSession::new();
        for (op, file, k) in &ops {
            let path = format!("src/f{}.rs", file);
            match op % 6 {
                0..=2 => {
                    // Ingest a fresh revision — the main graph-building op.
                    let source = format!(
                        "mod m{k};\nuse dep_{k}::lib;\npub fn func_{k}(x: u32) -> u32 {{ x + {k} }}\nstruct S{k};\n"
                    );
                    s.ingest(&path, &source);
                }
                3 => {
                    // Failure on a (maybe-existing) file node; an unknown node is a
                    // no-op `Err` that records nothing, so live and replay stay in
                    // lockstep on exactly the ops that were logged.
                    let _ = s.signal_failure(&format!("file:{path}"), k % 4);
                }
                4 => {
                    // Recall mutates recency/access on the paged region and is itself
                    // logged + replayed — exercise every strategy.
                    let r = match k % 4 {
                        0 => Recall::WorkingSet,
                        1 => Recall::Around(format!("file:{path}")),
                        2 => Recall::Task(format!("func_{k}")),
                        _ => Recall::Semantic(format!("func_{k}")),
                    };
                    s.recall(r, 1024);
                }
                _ => {
                    // Page-fault from a synthetic compiler line naming the file.
                    s.page_fault(&format!("error[E0001]: in {path}:{k}"), 1024);
                }
            }
        }

        // Invariant 1 — replay == live, byte-for-byte over the whole graph.
        let live = graph_hash(s.memory().graph());
        let replayed = graph_hash(s.replay_to(s.len()).graph());
        prop_assert_eq!(&replayed, &live, "replay diverged from live after {} logged ops", s.len());

        // Invariant 2 — replay is itself deterministic: a second replay of the same
        // log hashes identically (no run-to-run nondeterminism in reconstruction).
        let replayed_again = graph_hash(s.replay_to(s.len()).graph());
        prop_assert_eq!(&replayed_again, &replayed, "two replays of the same log diverged");
    }
}
