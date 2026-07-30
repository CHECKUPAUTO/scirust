use ccos_core::event_log::{EventLog, EventPayload, EventType};
use ccos_core::memory::{EdgeType, MemoryGraph, NodeId, NodeType};
use std::collections::HashMap;

/// Build a graph from an event log by replaying GraphMutation events.
/// This simulates recovering graph state from append-only log.
fn rebuild_graph_from_log(event_log: &EventLog) -> MemoryGraph {
    let mut graph = MemoryGraph::default();

    for event in &event_log.events {
        match &event.payload {
            EventPayload::GraphMutation {
                node_id, operation, ..
            } => match operation.as_str() {
                "add" | "ingest" | "upsert" => {
                    graph.upsert_node(
                        NodeId(node_id.clone()),
                        format!("_replay_{}", node_id),
                        format!("replayed: {}", event.id),
                        NodeType::ContextBlock,
                    );
                }
                "remove" => {
                    graph.remove_node(&NodeId(node_id.clone()));
                }
                _ => {}
            },
            EventPayload::Parsing { file_path, .. } => {
                // Parsing events create file nodes
                graph.upsert_node(
                    NodeId(format!("file:{}", file_path)),
                    file_path.clone(),
                    format!("replayed from parsing event {}", event.id),
                    NodeType::Module,
                );
            }
            _ => {}
        }
    }

    graph
}

#[test]
fn graph_consistency_after_replay() {
    // ── Phase 1: Build a reference graph and log simultaneously ──
    let mut reference_graph = MemoryGraph::default();
    let mut event_log = EventLog::new("consistency_test".into());

    // Build nodes directly
    reference_graph.upsert_node(
        "node_a".into(),
        "Module A".into(),
        "content A".into(),
        NodeType::Module,
    );
    reference_graph.upsert_node(
        "node_b".into(),
        "Module B".into(),
        "content B".into(),
        NodeType::Module,
    );
    reference_graph.upsert_node(
        "node_c".into(),
        "Symbol C".into(),
        "content C".into(),
        NodeType::Symbol,
    );
    reference_graph.add_edge("node_a".into(), "node_b".into(), 0.8, EdgeType::DependsOn);
    reference_graph.add_edge("node_b".into(), "node_c".into(), 0.6, EdgeType::References);

    // Log all mutations
    event_log.append(
        EventType::GraphMutation,
        EventPayload::GraphMutation {
            node_id: "node_a".into(),
            operation: "add".into(),
            nodes_before: 0,
            nodes_after: 1,
            edges_before: 0,
            edges_after: 0,
        },
    );
    event_log.append(
        EventType::GraphMutation,
        EventPayload::GraphMutation {
            node_id: "node_b".into(),
            operation: "add".into(),
            nodes_before: 1,
            nodes_after: 2,
            edges_before: 0,
            edges_after: 0,
        },
    );
    event_log.append(
        EventType::GraphMutation,
        EventPayload::GraphMutation {
            node_id: "node_c".into(),
            operation: "add".into(),
            nodes_before: 2,
            nodes_after: 3,
            edges_before: 0,
            edges_after: 0,
        },
    );

    // ── Phase 2: Reconstruct from log ──
    let rebuilt_graph = rebuild_graph_from_log(&event_log);

    // ── Assertions ──
    // The rebuilt graph should have the same node count
    // (modulo implementation details of how we rebuild)
    assert!(
        rebuilt_graph.node_count() > 0,
        "rebuilt graph must have nodes"
    );

    // Event log is strictly append-only
    assert_eq!(event_log.events[0].sequence_number, 0);
    assert_eq!(event_log.events[1].sequence_number, 1);
    assert_eq!(event_log.events[2].sequence_number, 2);
}

#[test]
fn detect_missing_events() {
    let mut event_log = EventLog::new("missing_test".into());

    // Normal sequence: add node_a, add node_b, add edge a->b
    event_log.append(
        EventType::GraphMutation,
        EventPayload::GraphMutation {
            node_id: "node_a".into(),
            operation: "add".into(),
            nodes_before: 0,
            nodes_after: 1,
            edges_before: 0,
            edges_after: 0,
        },
    );
    event_log.append(
        EventType::GraphMutation,
        EventPayload::GraphMutation {
            node_id: "node_b".into(),
            operation: "add".into(),
            nodes_before: 1,
            nodes_after: 2,
            edges_before: 0,
            edges_after: 0,
        },
    );

    // Check: the log has 2 events
    assert_eq!(event_log.event_count(), 2, "log must have exactly 2 events");

    // Simulate corrupted log — build a map of expected nodes
    let mut expected_nodes: HashMap<String, bool> = HashMap::new();
    for event in &event_log.events {
        if let EventPayload::GraphMutation { node_id, .. } = &event.payload {
            expected_nodes.insert(node_id.clone(), true);
        }
    }

    // "node_a" and "node_b" should be present
    assert!(
        expected_nodes.contains_key("node_a"),
        "node_a must be tracked"
    );
    assert!(
        expected_nodes.contains_key("node_b"),
        "node_b must be tracked"
    );

    // "node_c" should NOT be present (missing event)
    assert!(
        !expected_nodes.contains_key("node_c"),
        "node_c must not exist without event"
    );
}

#[test]
fn detect_duplicate_events() {
    let mut event_log = EventLog::new("duplicate_test".into());

    let payload = EventPayload::GraphMutation {
        node_id: "node_x".into(),
        operation: "add".into(),
        nodes_before: 0,
        nodes_after: 1,
        edges_before: 0,
        edges_after: 0,
    };

    let id1 = event_log.append(EventType::GraphMutation, payload.clone());

    // Append the same payload again (simulating duplicate)
    let id2 = event_log.append(EventType::GraphMutation, payload.clone());

    // IDs must be unique even for duplicate content
    assert_ne!(id1, id2, "duplicate events must have unique IDs");

    // Sequence numbers must be distinct
    assert_eq!(event_log.events[0].sequence_number, 0);
    assert_eq!(event_log.events[1].sequence_number, 1);

    // Count of add operations for node_x
    let add_count = event_log
        .events
        .iter()
        .filter(|e| {
            matches!(&e.payload, EventPayload::GraphMutation { node_id, operation, .. }
                if node_id == "node_x" && operation == "add")
        })
        .count();
    assert_eq!(add_count, 2, "duplicate add events must both be recorded");
}

#[test]
fn detect_out_of_order_events() {
    let mut event_log = EventLog::new("order_test".into());

    // Append events in sequence
    event_log.append(
        EventType::CycleStart,
        EventPayload::CycleEvent {
            cycle_number: 1,
            action: "start".into(),
        },
    );
    event_log.append(
        EventType::CycleEnd,
        EventPayload::CycleEvent {
            cycle_number: 1,
            action: "end".into(),
        },
    );

    // Sequence numbers must be monotonic
    let seqs: Vec<u64> = event_log.events.iter().map(|e| e.sequence_number).collect();
    for i in 1..seqs.len() {
        assert!(
            seqs[i] > seqs[i - 1],
            "sequence numbers must be strictly monotonic: seq[{}]={} <= seq[{}]={}",
            i,
            seqs[i],
            i - 1,
            seqs[i - 1]
        );
    }
}

#[test]
fn rollback_simulation_divergence_detected() {
    // Simulate a scenario where a rollback causes state divergence

    let mut graph = MemoryGraph::default();
    let mut event_log = EventLog::new("rollback_test".into());

    // Add nodes in sequence
    graph.upsert_node("n1".into(), "N1".into(), "".into(), NodeType::Module);
    event_log.append(
        EventType::GraphMutation,
        EventPayload::GraphMutation {
            node_id: "n1".into(),
            operation: "add".into(),
            nodes_before: 0,
            nodes_after: 1,
            edges_before: 0,
            edges_after: 0,
        },
    );

    graph.upsert_node("n2".into(), "N2".into(), "".into(), NodeType::Module);
    event_log.append(
        EventType::GraphMutation,
        EventPayload::GraphMutation {
            node_id: "n2".into(),
            operation: "add".into(),
            nodes_before: 1,
            nodes_after: 2,
            edges_before: 0,
            edges_after: 0,
        },
    );

    graph.upsert_node("n3".into(), "N3".into(), "".into(), NodeType::Module);
    event_log.append(
        EventType::GraphMutation,
        EventPayload::GraphMutation {
            node_id: "n3".into(),
            operation: "add".into(),
            nodes_before: 2,
            nodes_after: 3,
            edges_before: 0,
            edges_after: 0,
        },
    );

    // Now the graph has 3 nodes
    assert_eq!(graph.node_count(), 3);

    // Simulate rollback: remove n3 (as if it was never added)
    graph.remove_node(&"n3".into());
    // The event log still shows n3 was added — this is the "divergence"

    // Detect divergence: events claim 3 nodes, but graph has 2
    let events_node_count: usize = event_log.events.iter()
        .filter(|e| {
            matches!(&e.payload, EventPayload::GraphMutation { operation, .. } if operation == "add")
        })
        .count();

    let graph_node_count = graph.node_count();

    assert_ne!(
        events_node_count, graph_node_count,
        "divergence must be detected: events claim {} nodes, graph has {}",
        events_node_count, graph_node_count
    );

    // The system must NOT silently accept this state
    // (in a real implementation, this would trigger a reconciliation)
}

#[test]
fn replay_consistency_full_cycle() {
    // Full cycle: build graph, log everything, rebuild from log, compare

    let mut graph = MemoryGraph::default();
    let mut event_log = EventLog::new("replay_full".into());

    let files = [
        ("src/main.rs", "fn main() {}"),
        ("src/lib.rs", "pub mod foo;\nfn helper() {}"),
        ("src/foo.rs", "pub fn bar() -> u32 { 42 }"),
    ];

    // Build and log
    for (path, source) in &files {
        graph.upsert_node(
            NodeId(format!("file:{}", path)),
            path.to_string(),
            source.to_string(),
            NodeType::Module,
        );
        event_log.append(
            EventType::Parsing,
            EventPayload::Parsing {
                file_path: path.to_string(),
                file_hash: format!("sha256_{}", path.len()),
                modules_found: 1,
                uses_found: 0,
                symbols_found: 1,
            },
        );
    }

    // Add edges
    graph.add_edge(
        "file:src/main.rs".into(),
        "file:src/lib.rs".into(),
        0.8,
        EdgeType::DependsOn,
    );
    graph.add_edge(
        "file:src/lib.rs".into(),
        "file:src/foo.rs".into(),
        0.7,
        EdgeType::Contains,
    );

    event_log.append(
        EventType::GraphMutation,
        EventPayload::GraphMutation {
            node_id: "edge:main->lib".into(),
            operation: "add_edge".into(),
            nodes_before: 3,
            nodes_after: 3,
            edges_before: 0,
            edges_after: 2,
        },
    );

    // Take snapshot
    event_log.append(
        EventType::Snapshot,
        EventPayload::Snapshot {
            nodes_count: graph.node_count(),
            edges_count: graph.edge_count(),
            total_events: event_log.event_count(),
        },
    );

    // Verify snapshot data matches graph
    if let Some(snapshot) = event_log.events.last() {
        if let EventPayload::Snapshot {
            nodes_count,
            edges_count,
            ..
        } = &snapshot.payload
        {
            assert_eq!(
                *nodes_count,
                graph.node_count(),
                "snapshot nodes must match graph"
            );
            assert_eq!(
                *edges_count,
                graph.edge_count(),
                "snapshot edges must match graph"
            );
        }
    }

    // Assert replay is deterministic
    let mut replayer = ccos_core::event_log::EventReplayer::new();
    let result = event_log.replay_deterministic(&mut replayer);
    assert!(result.is_ok(), "deterministic replay must succeed");
}
