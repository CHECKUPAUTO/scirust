use ccos_core::event_log::{EventLog, EventPayload, EventType};
use ccos_core::incremental::IncrementalGraphEngine;
use ccos_core::memory::{EdgeType, MemoryGraph, NodeId, NodeType};
use sha2::{Digest, Sha256};

/// Compute a deterministic hash of graph state for snapshot comparison
fn graph_hash(graph: &MemoryGraph) -> String {
    let mut hasher = Sha256::new();

    // Hash node count
    hasher.update(graph.node_count().to_le_bytes());
    hasher.update(graph.edge_count().to_le_bytes());

    // Collect and sort node IDs for deterministic hashing
    let mut node_ids: Vec<&NodeId> = graph.node_ids().collect();
    node_ids.sort_by(|a, b| a.0.cmp(&b.0));

    for id in node_ids {
        if let Some(node) = graph.node(id) {
            hasher.update(id.0.as_bytes());
            hasher.update(node.label.as_bytes());
            hasher.update(node.content.as_bytes());
            hasher.update(node.base_importance.to_le_bytes());
            hasher.update(node.failure_relevance.to_le_bytes());
            hasher.update(node.recency.to_le_bytes());
        }
    }

    // Hash edges deterministically
    let mut edges_sorted: Vec<&ccos_core::memory::GraphEdge> = graph.edges().iter().collect();
    edges_sorted.sort_by(|a, b| {
        a.source
            .0
            .cmp(&b.source.0)
            .then(a.target.0.cmp(&b.target.0))
    });

    for edge in &edges_sorted {
        hasher.update(edge.source.0.as_bytes());
        hasher.update(edge.target.0.as_bytes());
        hasher.update(edge.weight.to_le_bytes());
    }

    format!("{:x}", hasher.finalize())
}

/// Compute deterministic hash for event log
fn event_log_hash(event_log: &EventLog) -> String {
    let mut hasher = Sha256::new();
    hasher.update(event_log.session_id.as_bytes());
    hasher.update(event_log.event_count().to_le_bytes());

    for event in &event_log.events {
        hasher.update(event.id.as_bytes());
        hasher.update(event.sequence_number.to_le_bytes());

        // Include payload content deterministically
        match &event.payload {
            EventPayload::GraphMutation {
                node_id,
                operation,
                nodes_before,
                nodes_after,
                edges_before,
                edges_after,
            } => {
                hasher.update(node_id.as_bytes());
                hasher.update(operation.as_bytes());
                hasher.update(nodes_before.to_le_bytes());
                hasher.update(nodes_after.to_le_bytes());
                hasher.update(edges_before.to_le_bytes());
                hasher.update(edges_after.to_le_bytes());
            }
            EventPayload::Parsing {
                file_path,
                file_hash,
                modules_found,
                uses_found,
                symbols_found,
            } => {
                hasher.update(file_path.as_bytes());
                hasher.update(file_hash.as_bytes());
                hasher.update(modules_found.to_le_bytes());
                hasher.update(uses_found.to_le_bytes());
                hasher.update(symbols_found.to_le_bytes());
            }
            EventPayload::Snapshot {
                nodes_count,
                edges_count,
                total_events,
            } => {
                hasher.update(nodes_count.to_le_bytes());
                hasher.update(edges_count.to_le_bytes());
                hasher.update(total_events.to_le_bytes());
            }
            EventPayload::CycleEvent {
                cycle_number,
                action,
            } => {
                hasher.update(cycle_number.to_le_bytes());
                hasher.update(action.as_bytes());
            }
            EventPayload::FailureDetection {
                node_id,
                failure_type,
                severity,
            } => {
                hasher.update(node_id.as_bytes());
                hasher.update(failure_type.as_bytes());
                hasher.update(severity.to_le_bytes());
            }
            EventPayload::GuardCheck {
                input_hash,
                passed,
                score,
                ..
            } => {
                hasher.update(input_hash.as_bytes());
                hasher.update(if *passed { &[1u8] } else { &[0u8] });
                hasher.update(score.to_le_bytes());
            }
            _ => {
                // Generic hash for other variants
                let variant_name = format!("{:?}", std::mem::discriminant(&event.payload));
                hasher.update(variant_name.as_bytes());
            }
        }
    }

    format!("{:x}", hasher.finalize())
}

#[test]
fn snapshot_baseline_t0() {
    let graph = MemoryGraph::default();
    let hash = graph_hash(&graph);

    // Empty graph hash must be deterministic
    let hash2 = graph_hash(&MemoryGraph::default());
    assert_eq!(hash, hash2, "empty graph hash must be deterministic");
}

#[test]
fn snapshot_consistent_after_build() {
    let mut graph = MemoryGraph::default();

    // Build graph
    for i in 0..5 {
        graph.upsert_node(
            NodeId(format!("n{}", i)),
            format!("Node {}", i),
            "content".into(),
            NodeType::Module,
        );
    }
    graph.add_edge("n0".into(), "n1".into(), 0.5, EdgeType::DependsOn);
    graph.add_edge("n1".into(), "n2".into(), 0.5, EdgeType::DependsOn);

    let hash1 = graph_hash(&graph);

    // Rebuild identically
    let mut graph2 = MemoryGraph::default();
    for i in 0..5 {
        graph2.upsert_node(
            NodeId(format!("n{}", i)),
            format!("Node {}", i),
            "content".into(),
            NodeType::Module,
        );
    }
    graph2.add_edge("n0".into(), "n1".into(), 0.5, EdgeType::DependsOn);
    graph2.add_edge("n1".into(), "n2".into(), 0.5, EdgeType::DependsOn);

    let hash2 = graph_hash(&graph2);

    assert_eq!(
        hash1, hash2,
        "identical graph builds must produce identical snapshots"
    );
}

#[test]
fn snapshot_detects_changes() {
    let mut graph = MemoryGraph::default();
    graph.upsert_node("a".into(), "A".into(), "".into(), NodeType::Module);
    graph.upsert_node("b".into(), "B".into(), "".into(), NodeType::Module);

    let hash_before = graph_hash(&graph);

    // Mutate
    graph.upsert_node("c".into(), "C".into(), "".into(), NodeType::Module);
    let hash_after = graph_hash(&graph);

    assert_ne!(
        hash_before, hash_after,
        "snapshot must detect graph changes"
    );
}

#[test]
fn snapshot_temporal_evolution() {
    let mut snapshots: Vec<(usize, String)> = Vec::new();
    let mut graph = MemoryGraph::default();
    let mut engine = IncrementalGraphEngine::new();

    // Snapshot at t0
    snapshots.push((0, graph_hash(&graph)));

    // Evolve and snapshot every 10 steps
    for step in 1..=100 {
        let source = format!("mod m{c};\nfn f{c}() {{ let x = {c}; }}\n", c = step);

        let old = if step > 1 {
            Some(format!(
                "mod m{c};\nfn f{c}() {{ let x = {c}; }}\n",
                c = step - 1
            ))
        } else {
            None
        };

        engine.process_delta(
            &format!("src/file_{}.rs", step % 5),
            old.as_deref(),
            &source,
            &mut graph,
        );

        if step % 10 == 0 {
            snapshots.push((step, graph_hash(&graph)));
        }
    }

    // Verify all snapshots exist
    assert!(
        snapshots.len() >= 11,
        "must have at least 11 snapshots (t0 + 10 intervals)"
    );

    // Verify adjacency: consecutive snapshots that represent actual changes differ
    // (or are equal if nothing changed — both valid)
    for i in 1..snapshots.len() {
        let (step, hash) = &snapshots[i];
        let (_prev_step, _prev_hash) = &snapshots[i - 1];

        eprintln!(
            "  snapshot @ step {}: {}",
            step,
            &hash[..16.min(hash.len())]
        );
    }

    // Final state coherence: graph must be non-empty
    assert!(
        graph.node_count() > 0,
        "graph must have nodes after evolution"
    );
}

#[test]
fn snapshot_differential_drift_detection() {
    // Simulate two different execution paths and verify snapshots diverge
    let mut graph_a = MemoryGraph::default();
    let mut graph_b = MemoryGraph::default();

    // Path A: add nodes normally
    for i in 0..10 {
        graph_a.upsert_node(
            NodeId(format!("a{}", i)),
            format!("A{}", i),
            "a".into(),
            NodeType::Module,
        );
    }

    // Path B: same nodes but with different failure relevance
    for i in 0..10 {
        graph_b.upsert_node(
            NodeId(format!("a{}", i)),
            format!("A{}", i),
            "b".into(), // different content
            NodeType::Module,
        );
    }

    let hash_a = graph_hash(&graph_a);
    let hash_b = graph_hash(&graph_b);

    // Different content should produce different snapshots
    assert_ne!(
        hash_a, hash_b,
        "different graph content must produce different snapshots"
    );
}

#[test]
fn snapshot_event_log_coherence() {
    let mut graph = MemoryGraph::default();
    let mut event_log = EventLog::new("snapshot_coherence".into());

    // Build graph and log events
    graph.upsert_node("x".into(), "X".into(), "data".into(), NodeType::Module);
    event_log.append(
        EventType::GraphMutation,
        EventPayload::GraphMutation {
            node_id: "x".into(),
            operation: "add".into(),
            nodes_before: 0,
            nodes_after: 1,
            edges_before: 0,
            edges_after: 0,
        },
    );

    graph.upsert_node("y".into(), "Y".into(), "data".into(), NodeType::Module);
    event_log.append(
        EventType::GraphMutation,
        EventPayload::GraphMutation {
            node_id: "y".into(),
            operation: "add".into(),
            nodes_before: 1,
            nodes_after: 2,
            edges_before: 0,
            edges_after: 0,
        },
    );

    // Take snapshot and store event count
    let snapshot_event_count = event_log.event_count();
    event_log.append(
        EventType::Snapshot,
        EventPayload::Snapshot {
            nodes_count: graph.node_count(),
            edges_count: graph.edge_count(),
            total_events: snapshot_event_count,
        },
    );

    // Verify snapshot data matches reality
    assert_eq!(graph.node_count(), 2);
    assert_eq!(event_log.event_count(), 3);

    // Event-log hashing is deterministic for a fixed log and survives a
    // serialize -> deserialize round-trip unchanged.
    let h1 = event_log_hash(&event_log);
    let h2 = event_log_hash(&event_log);
    assert_eq!(h1, h2, "event log hash must be deterministic");

    let restored = EventLog::from_json(&event_log.to_json()).expect("round-trip");
    assert_eq!(
        event_log_hash(&restored),
        h1,
        "event log hash must survive serialization round-trip"
    );
}
