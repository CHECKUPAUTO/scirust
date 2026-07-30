//! CCOS v0.3 — Context Scheduler integration tests.
//! Verifies HOT/WARM/COLD paging on graphs built by the real engine:
//! eviction correctness, priority ordering, and that no node is ever lost.

use ccos_core::incremental::IncrementalGraphEngine;
use ccos_core::memory::MemoryGraph;
use ccos_core::scheduler::{ContextScheduler, MemoryZone};

fn build_graph(files: usize) -> MemoryGraph {
    let mut graph = MemoryGraph::new(0.2, 1_000_000);
    let mut engine = IncrementalGraphEngine::new();
    for i in 0..files {
        let src = format!(
            "mod m{i};\nuse dep_{i}::lib;\npub fn func_{i}(x: u32) -> u32 {{ x + {i} }}\nstruct S{i};\n"
        );
        engine.process_delta(&format!("src/f{i}.rs"), None, &src, &mut graph);
    }
    graph
}

#[test]
fn schedules_real_graph_within_budget() {
    let graph = build_graph(50);
    let scheduler = ContextScheduler::from_graph(&graph, 256);

    // HOT tier must fit the budget...
    assert!(scheduler.hot_token_usage() <= 256);
    // ...and every node lands in exactly one zone (nothing lost).
    let zoned = scheduler.hot_context().len()
        + scheduler.warm_context().len()
        + scheduler.cold_context().len();
    assert_eq!(zoned, scheduler.len());
    assert_eq!(scheduler.len(), graph.node_count());
}

#[test]
fn hot_tier_holds_the_highest_priority_nodes() {
    let mut scheduler = ContextScheduler::new(20);
    scheduler.warm_ratio = 0.0; // no warm tier, force HOT vs COLD
    for (i, prio) in [0.1, 0.95, 0.9, 0.2, 0.85].into_iter().enumerate() {
        scheduler.upsert(format!("n{i}").into(), 10, prio);
    }
    scheduler.allocate_context();

    // Budget 20 / cost 10 → 2 HOT slots, taken by the top-2 priorities (n1, n2).
    let hot = scheduler.hot_context();
    assert_eq!(hot.len(), 2);
    assert!(hot.contains(&"n1".into()) && hot.contains(&"n2".into()));
}

#[test]
fn eviction_demotes_without_loss() {
    let graph = build_graph(40);
    let mut scheduler = ContextScheduler::from_graph(&graph, 100_000); // everything HOT
    let before = scheduler.len();
    assert!(!scheduler.hot_context().is_empty());

    // Shrink the budget hard and evict: HOT must contract, total count constant.
    scheduler.token_budget = 64;
    scheduler.evict_context();
    assert!(scheduler.hot_token_usage() <= 64);
    assert_eq!(scheduler.len(), before, "no node lost during eviction");
}

#[test]
fn stress_schedules_large_graph() {
    let graph = build_graph(2_000); // ~10k nodes
    assert!(graph.node_count() > 5_000);
    let scheduler = ContextScheduler::from_graph(&graph, 4_096);

    assert_eq!(scheduler.len(), graph.node_count());
    assert!(scheduler.hot_token_usage() <= 4_096);
    // Cold tier must exist (graph far exceeds the budget).
    assert!(
        scheduler
            .nodes
            .values()
            .any(|n| n.memory_zone == MemoryZone::Cold),
        "a large graph must spill to COLD"
    );
}
