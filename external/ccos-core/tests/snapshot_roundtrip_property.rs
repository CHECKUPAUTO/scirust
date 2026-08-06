//! Property-based **snapshot round-trip**: for any op stream, deserializing a
//! snapshot and re-serializing it must be a **fixed point** (byte-identical JSON),
//! and the restored graph must hash identically to the original.
//!
//! This pins the persistence path — `CcosMemory::to_json` / `from_json`, the same
//! canonical shape `open` / `checkpoint_to` read and write — against any
//! serialization asymmetry (a field that serializes but doesn't round-trip, a map
//! whose order isn't stable, a default that isn't elided). It is the randomized
//! counterpart to the hand-written create→reload cycle in `tests/persistence.rs`.

use ccos_core::agent_session::AgentSession;
use ccos_core::external_memory::{CcosMemory, Recall};
use ccos_core::memory::{MemoryGraph, NodeId};
use proptest::prelude::*;
use sha2::{Digest, Sha256};

/// Deterministic full-state hash of a resident graph (sorted, so it is independent
/// of the resident `HashMap`'s iteration order).
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

    /// Build a memory from a random op stream, snapshot it, restore it, and require
    /// (1) the restored graph hashes identically and (2) re-serializing the restored
    /// memory yields a byte-identical snapshot.
    #[test]
    fn snapshot_round_trips_byte_for_byte(
        ops in prop::collection::vec((0u8..6u8, 0u8..8u8, 0u32..50u32), 1..120)
    ) {
        let mut s = AgentSession::new();
        for (op, file, k) in &ops {
            let path = format!("src/f{}.rs", file);
            match op % 6 {
                0..=2 => {
                    let source = format!(
                        "mod m{k};\nuse dep_{k}::lib;\npub fn func_{k}(x: u32) -> u32 {{ x + {k} }}\nstruct S{k};\n"
                    );
                    s.ingest(&path, &source);
                }
                3 => {
                    let _ = s.signal_failure(&format!("file:{path}"), k % 4);
                }
                4 => {
                    let r = match k % 4 {
                        0 => Recall::WorkingSet,
                        1 => Recall::Around(format!("file:{path}")),
                        2 => Recall::Task(format!("func_{k}")),
                        _ => Recall::Semantic(format!("func_{k}")),
                    };
                    s.recall(r, 1024);
                }
                _ => {
                    s.page_fault(&format!("error[E0001]: in {path}:{k}"), 1024);
                }
            }
        }

        let original = s.memory();
        let json = original.to_json().expect("serialize snapshot");
        let restored = CcosMemory::from_json(&json).expect("deserialize snapshot");

        // 1. The restored graph is identical to the original.
        prop_assert_eq!(
            graph_hash(restored.graph()),
            graph_hash(original.graph()),
            "restored graph diverged from the original"
        );

        // 2. Re-serializing the restored memory is a fixed point — byte-identical.
        let json2 = restored.to_json().expect("re-serialize snapshot");
        prop_assert!(
            json2 == json,
            "snapshot is not a fixed point: {} bytes vs {} after round-trip",
            json2.len(),
            json.len()
        );
    }
}
