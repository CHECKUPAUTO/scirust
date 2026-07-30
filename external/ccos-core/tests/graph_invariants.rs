//! Structural invariants for the memory graph.
//!
//! These guard against a class of bugs where paging/eviction leaves the edge
//! set inconsistent with the node set. Historically, calling `enforce_paging`
//! re-entrantly from `upsert_node` let `add_edge` attach edges to nodes that
//! had just been evicted, so dangling edges accumulated without bound (the edge
//! count grew linearly with the number of cycles even though the node count was
//! capped). That broke the O(Δ) guarantee and the long-term stability budget.

use ccos_core::incremental::IncrementalGraphEngine;
use ccos_core::memory::{EdgeType, MemoryGraph, NodeType};
use std::collections::HashSet;

/// Drive the engine through many mutation cycles under tight paging pressure
/// and assert the graph never accumulates edges that point at evicted nodes,
/// and that the edge set stays bounded (not O(cycles)).
#[test]
fn no_dangling_edges_under_paging_pressure() {
    let mut graph = MemoryGraph::new(0.1, 200);
    let mut engine = IncrementalGraphEngine::new();

    for cycle in 0..3000usize {
        let file_idx = cycle % 7;
        let source = format!(
            "mod module_{c};\nuse dep_{c}::lib;\npub fn func_{c}(x: u32) -> u32 {{ x + {c} }}\nstruct S{c} {{ x: u32 }}\n",
            c = cycle
        );
        let old = if cycle > 0 {
            Some(format!(
                "mod module_{c};\nuse dep_{c}::lib;\npub fn func_{c}(x: u32) -> u32 {{ x + {c} }}\nstruct S{c} {{ x: u32 }}\n",
                c = cycle - 1
            ))
        } else {
            None
        };
        engine.process_delta(
            &format!("src/module_{}.rs", file_idx),
            old.as_deref(),
            &source,
            &mut graph,
        );
    }

    let ids: HashSet<_> = graph.node_ids().cloned().collect();
    let dangling = graph
        .edges()
        .iter()
        .filter(|e| !ids.contains(&e.source) || !ids.contains(&e.target))
        .count();

    assert_eq!(dangling, 0, "graph leaked {} dangling edges", dangling);

    // Node count is bounded by the paging limit ...
    assert!(graph.node_count() <= 200, "nodes must respect paging cap");
    // ... and so must the edge count: edges ⊆ nodes × nodes, never O(cycles).
    assert!(
        graph.edge_count() <= 200 * 4,
        "edge count {} grew unbounded — likely a dangling-edge leak",
        graph.edge_count()
    );
}

/// `add_edge` must refuse to connect endpoints that do not exist, so callers
/// cannot introduce dangling edges out of band.
#[test]
fn add_edge_rejects_missing_endpoints() {
    let mut graph = MemoryGraph::default();
    graph.upsert_node("a".into(), "A".into(), "".into(), NodeType::Module);

    // target missing
    assert!(!graph.add_edge("a".into(), "ghost".into(), 1.0, EdgeType::DependsOn));
    // source missing
    assert!(!graph.add_edge("ghost".into(), "a".into(), 1.0, EdgeType::DependsOn));
    assert_eq!(graph.edge_count(), 0, "no dangling edge should be created");

    graph.upsert_node("b".into(), "B".into(), "".into(), NodeType::Module);
    assert!(graph.add_edge("a".into(), "b".into(), 1.0, EdgeType::DependsOn));
    assert_eq!(graph.edge_count(), 1);
}

/// Eviction order must be deterministic so replay and snapshot hashes are
/// reproducible: building the same graph twice (with score ties under paging)
/// must keep exactly the same surviving node set.
#[test]
fn eviction_is_deterministic() {
    fn build() -> Vec<String> {
        let mut g = MemoryGraph::new(0.1, 25);
        for i in 0..200 {
            g.upsert_node(
                format!("node:{:04}", i).into(),
                format!("n{}", i),
                "x".into(),
                NodeType::Unknown,
            );
        }
        let mut survivors: Vec<String> = g.node_ids().map(|k| k.0.clone()).collect();
        survivors.sort();
        survivors
    }

    let a = build();
    let b = build();
    assert_eq!(a.len(), 25, "paging cap must hold");
    assert_eq!(
        a, b,
        "eviction must be deterministic across identical builds"
    );
}
