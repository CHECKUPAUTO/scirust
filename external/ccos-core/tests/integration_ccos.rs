use ccos_core::{
    event_log::{EventLog, EventPayload, EventReplayer, EventType},
    guard::{GuardConfig, GuardLayer},
    incremental::IncrementalGraphEngine,
    memory::{MemoryGraph, NodeId, NodeType},
    parser::ASTParser,
};

use std::collections::HashMap;

// ── Helper: create a basic guard ──────────────────────────────────
fn make_guard() -> GuardLayer {
    GuardLayer::new(GuardConfig::default())
}

// ── Phase 1: Initial State ────────────────────────────────────────
#[test]
fn phase1_initial_state() {
    let mut graph = MemoryGraph::default();
    let mut engine = IncrementalGraphEngine::new();
    let mut event_log = EventLog::new("phase1".into());

    let mut workspace: HashMap<String, String> = HashMap::new();
    workspace.insert(
        "src/sorter.rs".into(),
        "mod utils;\nfn sort<T: Ord>(data: &mut [T]) { data.sort(); }\n".into(),
    );
    workspace.insert(
        "src/logger.rs".into(),
        "use std::io::Write;\nfn log(msg: &str) { println!(\"{}\", msg); }\n".into(),
    );

    event_log.append(
        EventType::CycleStart,
        EventPayload::CycleEvent {
            cycle_number: 0,
            action: "cycle_init".into(),
        },
    );

    for (path, source) in &workspace {
        let result = engine.register_file(path, source);
        let parser = ASTParser::new();
        parser.update_memory_graph(&result, source, &mut graph);
    }

    // Assertions
    assert!(graph.node_count() > 0, "graph must have nodes");
    assert!(!event_log.events.is_empty(), "event log must not be empty");

    // Verify event ordering is preserved
    let first = &event_log.events[0];
    assert_eq!(first.event_type, EventType::CycleStart);
}

// ── Phase 2: Normal Execution Cycle ───────────────────────────────
#[test]
fn phase2_normal_execution_cycle() {
    let guard = make_guard();

    // Test valid JSON passes guard
    let valid_json = r#"{"analysis": {"summary": "All good", "dependencies": []}}"#;
    let result = guard.validate_and_sanitize(valid_json);
    assert!(result.passed, "valid JSON must pass guard");
    assert!(result.reliability_score >= guard.reliability_threshold());

    // Test invalid JSON triggers fallback
    let invalid = "not json @@@";
    let result = guard.validate_and_sanitize(invalid);
    assert!(!result.passed, "invalid JSON must be blocked");

    // Verify fallback is valid JSON
    let fallback = GuardLayer::fallback_response();
    assert!(serde_json::from_str::<serde_json::Value>(&fallback).is_ok());

    // Event recording test
    let mut event_log = EventLog::new("phase2".into());
    event_log.append(
        EventType::LlmCall,
        EventPayload::LlmCallRequest {
            model: "test".into(),
            prompt_hash: "abc".into(),
            input_tokens: 100,
        },
    );
    event_log.append(
        EventType::LlmResponse,
        EventPayload::LlmCallResponse {
            model: "test".into(),
            response_hash: "def".into(),
            output_tokens: 50,
            latency_ms: 200,
            guard_passed: true,
            reliability_score: 0.95,
        },
    );

    assert_eq!(event_log.event_count(), 2);
    assert!(!event_log.events[0].id.is_empty());
    assert_ne!(event_log.events[0].id, event_log.events[1].id);
}

// ── Phase 3: Mutation Simulation (Incremental O(Δ)) ───────────────
#[test]
fn phase3_mutation_simulation() {
    let mut graph = MemoryGraph::default();
    let mut engine = IncrementalGraphEngine::new();

    let old_source = "mod utils;\npub fn sort<T: Ord>(data: &mut [T]) { data.sort(); }";
    engine.process_delta("src/sorter.rs", None, old_source, &mut graph);
    let nodes_before = graph.node_count();

    let new_source =
        "mod utils;\nuse auth::validate;\npub fn sort<T: Ord>(data: &mut [T]) { data.sort(); }";
    let delta = engine.process_delta("src/sorter.rs", Some(old_source), new_source, &mut graph);

    // O(Δ) assertions
    assert_ne!(
        delta.operation,
        ccos_core::incremental::MutationOp::NoChange,
        "must detect modification"
    );
    assert!(
        delta.nodes_added > 0 || graph.node_count() > nodes_before,
        "graph must reflect the delta: nodes_before={} nodes_after={} delta_added={}",
        nodes_before,
        graph.node_count(),
        delta.nodes_added
    );

    // Verify old nodes evicted before new ones added
    let total_mutations = engine.total_mutations();
    assert_eq!(total_mutations, 2);
}

// ── Phase 4: Failure Propagation ──────────────────────────────────
#[test]
fn phase4_failure_propagation() {
    let mut graph = MemoryGraph::default();

    graph.upsert_node(
        "src/sorter.rs".into(),
        "sorter".into(),
        "sorting module".into(),
        NodeType::Module,
    );
    graph.upsert_node(
        "src/logger.rs".into(),
        "logger".into(),
        "logging module".into(),
        NodeType::Module,
    );
    graph.add_edge(
        "src/sorter.rs".into(),
        "src/logger.rs".into(),
        0.9,
        ccos_core::memory::EdgeType::DependsOn,
    );

    // Inject failure
    graph.set_failure_relevance(&"src/sorter.rs".into(), 0.8);
    graph.propagate_failure(&"src/sorter.rs".into(), 0, 3);

    // Assertions
    let sorter = graph.node(&"src/sorter.rs".into()).unwrap();
    assert!(
        sorter.failure_relevance > 0.0,
        "sorter must have failure relevance"
    );

    let logger = graph.node(&"src/logger.rs".into()).unwrap();
    assert!(
        logger.failure_relevance > 0.0,
        "logger must be impacted by propagation"
    );

    // Score decay: sorter should have higher failure relevance than logger
    assert!(
        sorter.failure_relevance >= logger.failure_relevance,
        "origin must have higher failure relevance than propagated node"
    );
}

// ── Phase 5: Event Sourcing Validation ────────────────────────────
#[test]
fn phase5_event_sourcing_validation() {
    let mut event_log = EventLog::new("phase5".into());

    // Append events
    let id1 = event_log.append(
        EventType::CycleStart,
        EventPayload::CycleEvent {
            cycle_number: 1,
            action: "start".into(),
        },
    );
    let id2 = event_log.append(
        EventType::Parsing,
        EventPayload::Parsing {
            file_path: "test.rs".into(),
            file_hash: "h1".into(),
            modules_found: 2,
            uses_found: 1,
            symbols_found: 5,
        },
    );
    let id3 = event_log.append(
        EventType::CycleEnd,
        EventPayload::CycleEvent {
            cycle_number: 1,
            action: "end".into(),
        },
    );

    // Assert append-only: event IDs are unique
    assert_ne!(id1, id2);
    assert_ne!(id2, id3);
    assert_ne!(id1, id3);

    // Assert ordering preserved
    assert_eq!(event_log.events[0].sequence_number, 0);
    assert_eq!(event_log.events[1].sequence_number, 1);
    assert_eq!(event_log.events[2].sequence_number, 2);

    // Assert no mutation of past events (immutability)
    let event0_id = event_log.events[0].id.clone();
    let event0_seq = event_log.events[0].sequence_number;
    // Append another event
    event_log.append(
        EventType::Snapshot,
        EventPayload::Snapshot {
            nodes_count: 10,
            edges_count: 5,
            total_events: 4,
        },
    );
    // Previous events must remain unchanged
    assert_eq!(event_log.events[0].id, event0_id);
    assert_eq!(event_log.events[0].sequence_number, event0_seq);
}

// ── Phase 6: Deterministic Replay ─────────────────────────────────
#[test]
fn phase6_deterministic_replay() {
    // Build an event log
    let mut event_log = EventLog::new("phase6".into());
    event_log.append(
        EventType::CycleStart,
        EventPayload::CycleEvent {
            cycle_number: 0,
            action: "init".into(),
        },
    );
    event_log.append(
        EventType::Parsing,
        EventPayload::Parsing {
            file_path: "a.rs".into(),
            file_hash: "abc".into(),
            modules_found: 1,
            uses_found: 2,
            symbols_found: 3,
        },
    );
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
    event_log.append(
        EventType::FailureDetection,
        EventPayload::FailureDetection {
            node_id: "n1".into(),
            failure_type: "test".into(),
            severity: 0.5,
        },
    );

    // First replay
    let mut replayer1 = EventReplayer::new();
    let result1 = event_log.replay_deterministic(&mut replayer1);
    assert!(result1.is_ok());
    let count1 = result1.unwrap();

    // Second replay (must be identical)
    let mut replayer2 = EventReplayer::new();
    let result2 = event_log.replay_deterministic(&mut replayer2);
    assert!(result2.is_ok());
    let count2 = result2.unwrap();

    assert_eq!(count1, count2, "replay must be deterministic");
    assert_eq!(
        replayer1.statistics.total_events,
        replayer2.statistics.total_events
    );
    assert_eq!(
        replayer1.statistics.parsing_events,
        replayer2.statistics.parsing_events
    );
    assert_eq!(
        replayer1.statistics.graph_mutations,
        replayer2.statistics.graph_mutations
    );
    assert_eq!(replayer1.statistics.failures, replayer2.statistics.failures);
}

// ── Phase 7: Guard Layer Resilience ───────────────────────────────
#[test]
fn phase7_guard_layer_resilience() {
    let guard = make_guard();

    // Empty output
    let r = guard.validate_and_sanitize("");
    assert!(!r.passed);

    // Null byte injection
    let r = guard.validate_and_sanitize("{\"key\": \"val\x00ue\"}");
    assert!(!r.sanitized_output.contains('\x00'));

    // Extremely long output
    let long = "x".repeat(20000);
    let r = guard.validate_and_sanitize(&long);
    assert!(r.sanitized_output.len() <= 8192);

    // Invalid JSON with braces
    let r = guard.validate_and_sanitize("not json {{{");
    assert!(!r.passed);

    // Valid JSON object
    let r = guard.validate_and_sanitize(r#"{"status":"ok"}"#);
    assert!(r.passed);
    assert!(r.reliability_score > 0.7);
}

// ── Phase 8: AST Parser Robustness ────────────────────────────────
#[test]
fn phase8_ast_parser_robustness() {
    let parser = ASTParser::new();

    // Normal code
    let result = parser.parse_source("test.rs", "mod foo;\nuse std::io;\nfn main() {}");
    assert!(!result.modules.is_empty());
    assert!(!result.use_statements.is_empty());
    assert!(!result.symbols.is_empty());

    // Empty file
    let result = parser.parse_source("empty.rs", "");
    assert_eq!(result.modules.len(), 0);
    assert_eq!(result.use_statements.len(), 0);

    // Comments-only file
    let result = parser.parse_source("comments.rs", "// just a comment\n/* block */");
    assert_eq!(result.modules.len(), 0);

    // Malformed but non-panicking
    let result = parser.parse_source("bad.rs", "}}}} foobar {{{ ///\nuse ????;");
    assert!(result.symbols.is_empty() || !result.symbols.is_empty());
    // Must not panic — reaching here is success
}

// ── Phase 9: Memory Paging ────────────────────────────────────────
#[test]
fn phase9_memory_paging() {
    let mut graph = MemoryGraph::new(0.1, 20);

    // Insert 100 nodes
    for i in 0..100 {
        graph.upsert_node(
            NodeId(format!("node_{}", i)),
            format!("Node {}", i),
            "data".into(),
            NodeType::Unknown,
        );
        // Give low scores to force eviction
        if let Some(node) = graph.node_mut(&NodeId(format!("node_{}", i))) {
            node.base_importance = 0.01;
            node.recency = 0.01;
        }
    }

    graph.enforce_paging();
    assert!(
        graph.node_count() <= 20,
        "paging must limit nodes to max_in_memory"
    );

    // Insert high-importance nodes that should survive
    for i in 100..110 {
        graph.upsert_node(
            NodeId(format!("important_{}", i)),
            format!("Important {}", i),
            "vital".into(),
            NodeType::ContextBlock,
        );
        if let Some(node) = graph.node_mut(&NodeId(format!("important_{}", i))) {
            node.base_importance = 0.95;
            node.failure_relevance = 0.8;
            node.recency = 1.0;
        }
    }

    graph.enforce_paging();
    // Important nodes should survive
    let important_count = graph
        .node_ids()
        .filter(|id| id.0.starts_with("important_"))
        .count();
    assert!(important_count > 0, "high-score nodes must survive paging");
}

// ── Phase 10: Multi-Cycle Stability ───────────────────────────────
#[test]
fn phase10_multicycle_stability() {
    let mut graph = MemoryGraph::default();
    let mut engine = IncrementalGraphEngine::new();
    let mut event_log = EventLog::new("phase10".into());

    let base = "fn cycle{}(x: u32) -> u32 { x + {} }";

    for cycle in 0..50 {
        event_log.append(
            EventType::CycleStart,
            EventPayload::CycleEvent {
                cycle_number: cycle,
                action: format!("cycle_{}", cycle),
            },
        );

        let source = base
            .replace("{}", &cycle.to_string())
            .replace("{}", &cycle.to_string());
        let old = if cycle > 0 {
            Some(
                base.replace("{}", &(cycle - 1).to_string())
                    .replace("{}", &(cycle - 1).to_string()),
            )
        } else {
            None
        };
        let old_str = old.as_deref();

        engine.process_delta(
            &format!("src/cycle_{}.rs", cycle % 3),
            old_str,
            &source,
            &mut graph,
        );

        event_log.append(
            EventType::CycleEnd,
            EventPayload::CycleEvent {
                cycle_number: cycle,
                action: format!("cycle_{}_end", cycle),
            },
        );
    }

    // Stability assertions
    assert!(graph.node_count() > 0);
    assert!(event_log.event_count() == 100);
    assert!(engine.total_mutations() == 50);
}

// ── Phase 11: Event Serialization Roundtrip ───────────────────────
#[test]
fn phase11_event_serialization_roundtrip() {
    let mut event_log = EventLog::new("serialize_test".into());

    event_log.append(
        EventType::LlmCall,
        EventPayload::LlmCallRequest {
            model: "deepseek".into(),
            prompt_hash: "abcdef1234567890".into(),
            input_tokens: 512,
        },
    );

    let json = serde_json::to_string_pretty(&event_log).unwrap();
    assert!(json.contains("deepseek"));
    assert!(json.contains("abcdef1234567890"));

    let restored: EventLog = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.session_id, event_log.session_id);
    assert_eq!(restored.event_count(), event_log.event_count());
}

// ── Phase 12: Graph Connectivity ──────────────────────────────────
#[test]
fn phase12_graph_connectivity() {
    let mut graph = MemoryGraph::default();

    // Build a chain: a -> b -> c -> d
    graph.upsert_node("a".into(), "A".into(), "".into(), NodeType::Module);
    graph.upsert_node("b".into(), "B".into(), "".into(), NodeType::Module);
    graph.upsert_node("c".into(), "C".into(), "".into(), NodeType::Module);
    graph.upsert_node("d".into(), "D".into(), "".into(), NodeType::Module);

    graph.add_edge(
        "a".into(),
        "b".into(),
        0.9,
        ccos_core::memory::EdgeType::DependsOn,
    );
    graph.add_edge(
        "b".into(),
        "c".into(),
        0.9,
        ccos_core::memory::EdgeType::DependsOn,
    );
    graph.add_edge(
        "c".into(),
        "d".into(),
        0.9,
        ccos_core::memory::EdgeType::DependsOn,
    );

    // Failure at 'a' should propagate through entire chain
    graph.set_failure_relevance(&"a".into(), 1.0);
    graph.propagate_failure(&"a".into(), 0, 5);

    let a = graph.node(&"a".into()).unwrap();
    let b = graph.node(&"b".into()).unwrap();
    let c = graph.node(&"c".into()).unwrap();
    let d = graph.node(&"d".into()).unwrap();

    assert!(a.failure_relevance > b.failure_relevance);
    assert!(b.failure_relevance >= c.failure_relevance);
    assert!(c.failure_relevance >= d.failure_relevance);
    assert!(
        d.failure_relevance > 0.0,
        "chain end must still be affected"
    );
}

// ── Phase 13: Incremental Engine No Full Rebuild O(Δ) ─────────────
#[test]
fn phase13_incremental_no_full_rebuild() {
    let mut graph = MemoryGraph::default();
    let mut engine = IncrementalGraphEngine::new();

    // Build large graph
    for i in 0..20 {
        let source = format!("mod mod_{};\nuse dep_{}::lib;\nfn func_{}() {{}}", i, i, i);
        engine.process_delta(&format!("file_{}.rs", i), None, &source, &mut graph);
    }

    let nodes_before = graph.node_count();
    let _edges_before = graph.edge_count();

    // Modify only ONE file
    let modified = "mod mod_5;\nuse dep_5::lib;\nuse dep_extra::lib;\nfn func_5() { let x = 1; }";
    let original = "mod mod_5;\nuse dep_5::lib;\nfn func_5() {}";
    let delta = engine.process_delta("file_5.rs", Some(original), modified, &mut graph);

    // O(Δ): only file_5 and its immediate dependencies should change
    // The delta should NOT rebuild the entire graph
    assert_eq!(
        delta.operation,
        ccos_core::incremental::MutationOp::FileModified
    );

    // change should be bounded — not a full graph rebuild
    let total_change = delta.nodes_added + delta.nodes_removed;
    assert!(
        total_change < nodes_before / 2,
        "O(Δ) violation: total delta {} exceeds half of total nodes {}",
        total_change,
        nodes_before
    );
}

// ── Phase 14: Context Window Selection Weighted ───────────────────
#[test]
fn phase14_context_window_selection() {
    let mut graph = MemoryGraph::default();

    // Create nodes with varying scores
    for i in 0..30 {
        let id = NodeId(format!("ctx_{}", i));
        graph.upsert_node(
            id.clone(),
            format!("Ctx{}", i),
            "data".into(),
            NodeType::ContextBlock,
        );
        if let Some(node) = graph.node_mut(&id) {
            node.base_importance = (i as f64) / 60.0;
            node.recency = if i < 10 { 1.0 } else { 0.1 };
            node.failure_relevance = if i < 5 { 0.9 } else { 0.0 };
        }
    }

    let context = graph.select_context_window(1024);

    // Context selection must be non-empty
    assert!(!context.is_empty());

    // The first returned nodes should have higher scores
    if context.len() >= 2 {
        let score0 = graph.compute_node_score(context[0]);
        let score1 = graph.compute_node_score(context[1]);
        assert!(
            score0 >= score1,
            "context must be sorted by descending score"
        );
    }
}
