//! CCOS v0.3 CLI commands — the Autonomous Context Runtime surface:
//! `scan`, `agents`, `benchmark` and the `runtime` capstone that wires the
//! scheduler, scanner, agents, and persistence together.

use ccos_core::agents::{Agent, AgentExecutor, CoderAgent, ReviewerAgent, SecurityAgent};
use ccos_core::benchmark::BenchmarkHarness;
use ccos_core::distributed_event_log::DistributedEventLog;
use ccos_core::event_log::EventLog;
use ccos_core::incremental::IncrementalGraphEngine;
use ccos_core::memory::MemoryGraph;
use ccos_core::persistence::{PersistentRuntime, RuntimeState};
use ccos_core::scheduler::ContextScheduler;
use ccos_core::workspace::WorkspaceScanner;
use std::path::Path;
use uuid::Uuid;

fn positional(args: &[String]) -> Option<&str> {
    args.iter()
        .find(|a| !a.starts_with("--"))
        .map(String::as_str)
}

fn flag_value<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

/// Read up to `cap` bytes of `.rs` source under `path` into one context string.
fn read_context(path: &str, cap: usize) -> String {
    let mut files = Vec::new();
    crate::collect_rs_files(Path::new(path), &mut files);
    files.sort();
    let mut context = String::new();
    for f in files {
        if context.len() >= cap {
            break;
        }
        if let Ok(s) = std::fs::read_to_string(&f) {
            context.push_str(&s);
            context.push('\n');
        }
    }
    context
}

/// `ccos scan <path>` — scan a real workspace and ingest the delta into a graph.
pub(crate) async fn run_scan(args: &[String]) -> i32 {
    let path = positional(args).unwrap_or(".");
    let mut scanner = WorkspaceScanner::new(path);
    let mut engine = IncrementalGraphEngine::new();
    let mut graph = MemoryGraph::new_from_env(0.2, 100_000);

    let delta = match scanner.sync(&mut engine, &mut graph).await {
        Ok(d) => d,
        Err(e) => {
            eprintln!("ccos: scan failed: {e}");
            return 1;
        }
    };

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS scan — {:<32}║", crate::truncate(path, 32));
    println!("╚══════════════════════════════════════════════╝\n");
    println!(
        "  Added / Modified / Removed: {} / {} / {}",
        delta.added.len(),
        delta.modified.len(),
        delta.removed.len()
    );
    println!("  Graph nodes:    {}", graph.node_count());
    println!("  Graph edges:    {}", graph.edge_count());
    println!("  Dangling edges: {}", graph.prune_dangling_edges());
    0
}

/// `ccos agents <path>` — run the Coder/Reviewer/Security agents over a
/// workspace, funneling every result through the guard and event log.
pub(crate) async fn run_agents(args: &[String]) -> i32 {
    let path = positional(args).unwrap_or(".");
    let context = read_context(path, 20_000);
    if context.is_empty() {
        eprintln!("ccos: no .rs files found under '{path}'");
        return 1;
    }

    let mut log = EventLog::new(Uuid::new_v4().to_string());
    let executor = AgentExecutor::new();
    let mut agents: Vec<Box<dyn Agent>> = vec![
        Box::new(CoderAgent::new("coder-1")),
        Box::new(ReviewerAgent::new("reviewer-1")),
        Box::new(SecurityAgent::new("security-1")),
    ];
    let results = executor.execute_all(&mut agents, &context, &mut log);

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS agents — {:<30}║", crate::truncate(path, 30));
    println!("╚══════════════════════════════════════════════╝\n");
    for r in &results {
        println!(
            "  [{:<8}] confidence {:.2} | guard {}",
            r.role.as_str(),
            r.confidence,
            if r.guard_passed { "PASS" } else { "BLOCK" }
        );
        println!("    {}", crate::truncate(&r.output, 84));
    }
    println!("\n  Events logged: {}", log.event_count());
    0
}

/// `ccos benchmark [--cycles N] [--cap N] [--out FILE]` — run the cycle
/// benchmark and write a JSON report.
pub(crate) fn run_benchmark(args: &[String]) -> i32 {
    let cycles: usize = flag_value(args, "--cycles")
        .and_then(|v| v.parse().ok())
        .unwrap_or(100_000);
    let cap: usize = flag_value(args, "--cap")
        .and_then(|v| v.parse().ok())
        .unwrap_or(200);
    let out = flag_value(args, "--out").unwrap_or("benchmark_report.json");

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS benchmark                              ║");
    println!("╚══════════════════════════════════════════════╝\n");
    println!("  Running {cycles} cycles (paging cap {cap})…\n");

    let report = BenchmarkHarness::new().with_paging_cap(cap).run(cycles);

    println!("  Total time:        {:.3}s", report.total_seconds);
    println!("  Avg cycle:         {:.3} µs", report.avg_cycle_us);
    println!(
        "  Throughput:        {:.0} cycles/s",
        report.cycles_per_second
    );
    println!(
        "  Final nodes/edges: {} / {}",
        report.final_nodes, report.final_edges
    );
    println!(
        "  Peak nodes/edges:  {} / {}",
        report.peak_nodes, report.peak_edges
    );
    println!("  Node drift:        {}", report.node_drift);
    println!("  Dangling edges:    {} (must be 0)", report.dangling_edges);

    match report.save(out) {
        Ok(()) => println!("\n  Report → {out}"),
        Err(e) => {
            eprintln!("ccos: failed to write {out}: {e}");
            return 1;
        }
    }
    if report.dangling_edges == 0 {
        0
    } else {
        1
    }
}

/// `ccos runtime <path> [--state DIR]` — the capstone: scan a workspace, page
/// the context with the scheduler, run the agents over the HOT tier, and
/// persist the full runtime so it can resume after a restart.
pub(crate) async fn run_runtime(args: &[String]) -> i32 {
    let path = positional(args).unwrap_or(".");
    let state_dir = flag_value(args, "--state").unwrap_or("data");
    let budget: usize = flag_value(args, "--budget")
        .and_then(|v| v.parse().ok())
        .unwrap_or(2048);

    println!("╔══════════════════════════════════════════════╗");
    println!("║  CCOS runtime — Autonomous Context Runtime   ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // 1. Scan the real workspace into a causal graph.
    let mut scanner = WorkspaceScanner::new(path);
    let mut engine = IncrementalGraphEngine::new();
    let mut graph = MemoryGraph::new_from_env(0.2, 100_000);
    let mut event_log = EventLog::new(Uuid::new_v4().to_string());
    let mut dist_log = DistributedEventLog::new();

    let delta = match scanner.sync(&mut engine, &mut graph).await {
        Ok(d) => d,
        Err(e) => {
            eprintln!("ccos: scan failed: {e}");
            return 1;
        }
    };
    dist_log.append(
        format!("scan:{}", delta.changed_count()),
        "workspace".into(),
    );
    println!(
        "  [1/4] Scanned {path}: {} nodes, {} edges",
        graph.node_count(),
        graph.edge_count()
    );

    // 2. Page the context with the scheduler.
    let scheduler = ContextScheduler::from_graph(&graph, budget);
    let hot = scheduler.hot_context();
    println!(
        "  [2/4] Scheduled (budget {budget}): HOT {} · WARM {} · COLD {}",
        hot.len(),
        scheduler.warm_context().len(),
        scheduler.cold_context().len()
    );

    // 3. Run the agents over the HOT context.
    let context: String = hot
        .iter()
        .filter_map(|id| graph.node(id))
        .map(|n| format!("{} {}", n.label, n.content))
        .collect::<Vec<_>>()
        .join("\n");
    let executor = AgentExecutor::new();
    let mut agents: Vec<Box<dyn Agent>> = vec![
        Box::new(CoderAgent::new("coder-1")),
        Box::new(ReviewerAgent::new("reviewer-1")),
        Box::new(SecurityAgent::new("security-1")),
    ];
    let results = executor.execute_all(&mut agents, &context, &mut event_log);
    for r in &results {
        dist_log.append(
            format!("agent:{}:{:.2}", r.role.as_str(), r.confidence),
            "agent".into(),
        );
    }
    println!("  [3/4] Ran {} agents over the HOT context", results.len());

    // 4. Persist the runtime so it survives a restart.
    let runtime = PersistentRuntime::new(state_dir);
    let state = RuntimeState::new(graph, event_log, dist_log);
    if let Err(e) = runtime.save_state(&state) {
        eprintln!("ccos: failed to persist runtime: {e}");
        return 1;
    }
    println!(
        "  [4/4] Persisted runtime → {state_dir}/ (graph.snapshot, events.log, memory.snapshot)"
    );

    // Verify the persisted state restores cleanly.
    match runtime.restore_runtime() {
        Ok(restored) => {
            println!(
                "\n  ✓ runtime resumable: {} nodes, {} events, hash-chain valid",
                restored.graph.node_count(),
                restored.event_log.event_count()
            );
            0
        }
        Err(e) => {
            eprintln!("ccos: persisted runtime failed to restore: {e}");
            1
        }
    }
}
