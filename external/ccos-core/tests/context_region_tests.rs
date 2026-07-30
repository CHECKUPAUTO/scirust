//! Integration tests for the Context Region Engine (CCOS v0.3).
//!
//! Covers the eight mandated scenarios: automatic region creation, warming on
//! edit, regional activation under failure propagation, cold eviction,
//! deterministic replay, no drift over 10k cycles, snapshot identity, and chaos
//! resilience (missing node, missing event, corrupted JSON, circular deps).

use ccos_core::context_policy::ContextPolicy;
use ccos_core::context_region::ContextRegion;
use ccos_core::event_log::{EventLog, GraphReconstructor};
use ccos_core::memory::{EdgeType, MemoryGraph, NodeId, NodeType};
use ccos_core::region_engine::{ContextRegionEngine, RegionQuery};

/// Two files (a.rs, b.rs) with internal containment edges.
fn two_file_graph() -> MemoryGraph {
    let mut g = MemoryGraph::new(0.2, 100_000);
    for id in [
        "file:a.rs",
        "sym:a.rs:x",
        "sym:a.rs:y",
        "file:b.rs",
        "sym:b.rs:z",
    ] {
        g.upsert_node(id.into(), id.into(), "".into(), NodeType::Symbol);
    }
    g.add_edge(
        "file:a.rs".into(),
        "sym:a.rs:x".into(),
        0.6,
        EdgeType::Contains,
    );
    g.add_edge(
        "file:a.rs".into(),
        "sym:a.rs:y".into(),
        0.6,
        EdgeType::Contains,
    );
    g.add_edge(
        "file:b.rs".into(),
        "sym:b.rs:z".into(),
        0.6,
        EdgeType::Contains,
    );
    g
}

fn new_engine(graph: &MemoryGraph) -> (ContextRegionEngine, EventLog) {
    let mut engine = ContextRegionEngine::new();
    let mut log = EventLog::new("test".into());
    engine.initialize_regions(graph, &mut log);
    (engine, log)
}

// ── 1. Automatic region creation ───────────────────────────────────
#[test]
fn regions_are_created_automatically() {
    let g = two_file_graph();
    let (engine, log) = new_engine(&g);
    assert_eq!(engine.region_count(), 2, "one region per independent file");
    assert!(engine.regions.contains_key("region:a.rs"));
    assert!(engine.regions.contains_key("region:b.rs"));
    // A RegionCreated event was logged for each region.
    let created = log
        .events
        .iter()
        .filter(|e| {
            matches!(
                e.payload,
                ccos_core::event_log::EventPayload::RegionCreated { .. }
            )
        })
        .count();
    assert_eq!(created, 2);
}

// ── 2. Editing a file warms its region ─────────────────────────────
#[test]
fn editing_a_file_warms_its_region() {
    let mut g = two_file_graph();
    let (mut engine, mut log) = new_engine(&g);

    // Cool everything down: advance the recency clock so all nodes decay.
    for _ in 0..20 {
        g.tick();
    }
    engine.initialize_regions(&g, &mut log);
    let cold_b = engine.regions["region:b.rs"].temperature;

    // "Edit" a.rs: re-upsert one of its symbols → recency resets to 1.0.
    g.upsert_node(
        "sym:a.rs:x".into(),
        "x".into(),
        "edited".into(),
        NodeType::Symbol,
    );
    let mut region_a = engine.regions["region:a.rs"].clone();
    region_a.recompute(&g);

    assert!(
        region_a.temperature > cold_b,
        "edited region ({:.4}) must be warmer than an untouched cold region ({:.4})",
        region_a.temperature,
        cold_b
    );
}

// ── 3. A failure propagates a regional activation ──────────────────
#[test]
fn failure_propagates_a_regional_activation() {
    let mut g = two_file_graph();
    // Cross-file causal edge a.rs::x → b.rs::z merges the two files into one
    // region, so a fault in a.rs can wake b.rs as part of the same zone.
    g.add_edge(
        "sym:a.rs:x".into(),
        "sym:b.rs:z".into(),
        0.9,
        EdgeType::DependsOn,
    );

    let (mut engine, mut log) = new_engine(&g);
    assert_eq!(engine.region_count(), 1, "linked files form one region");

    // Inject a fault at a.rs::x and propagate it across causal edges.
    g.set_failure_relevance(&NodeId("sym:a.rs:x".into()), 0.95);
    g.propagate_failure(&NodeId("sym:a.rs:x".into()), 0, 3);
    assert!(
        g.node(&NodeId("sym:b.rs:z".into()))
            .unwrap()
            .failure_relevance
            > 0.0,
        "failure must reach b.rs::z"
    );

    engine.initialize_regions(&g, &mut log);
    let policy = ContextPolicy::default();
    let win = engine
        .activate_region(
            &g,
            &RegionQuery::Node("sym:a.rs:x".into()),
            &policy,
            &mut log,
        )
        .expect("region exists");

    // The whole zone (both files) is woken, attributed to failure propagation.
    assert!(win.files.contains(&"a.rs".to_string()));
    assert!(win.files.contains(&"b.rs".to_string()));
    assert!(
        win.reason.contains("failure"),
        "activation reason must cite failure propagation: {}",
        win.reason
    );
}

// ── 4. A cold region is evicted ────────────────────────────────────
#[test]
fn cold_regions_are_evicted() {
    let g = two_file_graph();
    let (mut engine, mut log) = new_engine(&g);
    assert_eq!(engine.region_count(), 2);

    // Aggressively cool; everything should drop below the eviction floor.
    let mut evicted_total = 0;
    for _ in 0..50 {
        evicted_total += engine.tick_cooldown(0.5, 0.01, &mut log).len();
        if engine.region_count() == 0 {
            break;
        }
    }
    assert!(evicted_total > 0, "cold regions must be evicted");
    assert_eq!(engine.region_count(), 0, "all cold regions evicted");
}

// ── 5. Deterministic replay reconstructs the engine ────────────────
#[test]
fn replay_reconstructs_identical_engine() {
    let g = two_file_graph();
    let mut log = EventLog::new("replay".into());
    // Record the graph into the log so it can be rebuilt purely from events.
    log.record_graph(&g);

    let mut engine = ContextRegionEngine::new();
    engine.initialize_regions(&g, &mut log);
    let policy = ContextPolicy::default();
    engine.activate_region(
        &g,
        &RegionQuery::Node("sym:a.rs:x".into()),
        &policy,
        &mut log,
    );
    engine.activate_region(&g, &RegionQuery::Hottest, &policy, &mut log);

    // Rebuild the graph from the log, then replay the region events.
    let mut recon = GraphReconstructor::new();
    log.replay_deterministic(&mut recon).unwrap();
    let replayed = ContextRegionEngine::replay_from(&recon.graph, &log);

    assert_eq!(
        engine, replayed,
        "replay must reconstruct an identical engine"
    );
}

// ── 6. No drift after 10_000 cycles ────────────────────────────────
#[test]
fn no_drift_after_10000_cycles() {
    let g = two_file_graph();
    let (mut engine, mut log) = new_engine(&g);
    let policy = ContextPolicy::default();
    let initial_regions = engine.region_count();

    for i in 0..10_000u64 {
        // Alternate activation and cooldown; regions must not multiply or leak.
        if i % 2 == 0 {
            engine.activate_region(&g, &RegionQuery::Hottest, &policy, &mut log);
        } else {
            engine.tick_cooldown(0.999, 0.0, &mut log); // floor 0 → never evicts
        }
    }
    assert_eq!(
        engine.region_count(),
        initial_regions,
        "region count must stay bounded — no drift"
    );
    // Temperatures remain in range.
    for r in engine.regions.values() {
        assert!((0.0..=1.0).contains(&r.temperature));
    }
}

// ── 7. Snapshot before/after is identical ──────────────────────────
#[test]
fn snapshot_roundtrip_is_identical() {
    let g = two_file_graph();
    let (engine, _) = new_engine(&g);

    // serde roundtrip.
    let json = serde_json::to_string(&engine).unwrap();
    let restored: ContextRegionEngine = serde_json::from_str(&json).unwrap();
    assert_eq!(engine, restored, "engine must survive a serde roundtrip");

    // Re-clustering the same graph yields an identical base engine.
    let (engine2, _) = new_engine(&g);
    assert_eq!(engine, engine2, "clustering must be deterministic");
}

// ── 8. Chaos: missing node, missing event, corrupted JSON, cycles ──
#[test]
fn chaos_missing_node_does_not_panic() {
    let mut g = two_file_graph();
    let (engine, _) = new_engine(&g);
    // Remove a node out from under a region, then recompute — no panic, the
    // missing member simply contributes nothing.
    g.remove_node(&NodeId("sym:a.rs:x".into()));
    let mut region = engine.regions["region:a.rs"].clone();
    region.recompute(&g);
    assert!(region.temperature >= 0.0);
}

#[test]
fn chaos_missing_region_events_still_reconstructs() {
    let g = two_file_graph();
    // A log with the graph but NO region events at all.
    let mut log = EventLog::new("partial".into());
    log.record_graph(&g);
    let mut recon = GraphReconstructor::new();
    log.replay_deterministic(&mut recon).unwrap();
    // replay_from re-clusters from the graph even with no region events.
    let engine = ContextRegionEngine::replay_from(&recon.graph, &log);
    assert_eq!(
        engine.region_count(),
        2,
        "regions derive from the graph alone"
    );
}

#[test]
fn chaos_corrupted_json_is_rejected() {
    let bad = r#"{"regions": "not-a-map", "clock": "oops"}"#;
    let parsed: Result<ContextRegionEngine, _> = serde_json::from_str(bad);
    assert!(parsed.is_err(), "corrupted engine JSON must be rejected");
    // A single corrupted region is also rejected.
    let bad_region = r#"{"id": 5}"#;
    assert!(serde_json::from_str::<ContextRegion>(bad_region).is_err());
}

#[test]
fn chaos_circular_dependency_terminates() {
    // a.rs::x → b.rs::y → c.rs::z → a.rs::x  (a cross-file cycle).
    let mut g = MemoryGraph::new(0.2, 10_000);
    for id in ["sym:a.rs:x", "sym:b.rs:y", "sym:c.rs:z"] {
        g.upsert_node(id.into(), id.into(), "".into(), NodeType::Symbol);
    }
    g.add_edge(
        "sym:a.rs:x".into(),
        "sym:b.rs:y".into(),
        0.9,
        EdgeType::DependsOn,
    );
    g.add_edge(
        "sym:b.rs:y".into(),
        "sym:c.rs:z".into(),
        0.9,
        EdgeType::DependsOn,
    );
    g.add_edge(
        "sym:c.rs:z".into(),
        "sym:a.rs:x".into(),
        0.9,
        EdgeType::DependsOn,
    );

    // Clustering must terminate (BFS guards against the cycle) and merge the
    // three mutually-dependent files into a single region.
    let (engine, _) = new_engine(&g);
    assert_eq!(
        engine.region_count(),
        1,
        "the dependency cycle forms one region"
    );
}
