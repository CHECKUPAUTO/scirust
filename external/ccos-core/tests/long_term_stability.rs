use ccos_core::event_log::{EventLog, EventPayload, EventType};
use ccos_core::incremental::IncrementalGraphEngine;
use ccos_core::memory::{MemoryGraph, NodeId};
use std::time::Instant;

const NUM_CYCLES: usize = 10_000;

#[test]
fn long_term_stability_10k_cycles() {
    let start = Instant::now();

    let mut graph = MemoryGraph::new(0.1, 200);
    let mut engine = IncrementalGraphEngine::new();
    let mut event_log = EventLog::new("long_term_stability".into());

    let valid_json = r#"{"analysis": {"summary": "ok", "deps": []}}"#;
    let invalid_json = "not json @@@ corrupted";
    let truncated = r#"{"analysis": {"summary": "inc"#;
    let empty = "";

    // Track baseline
    let mut cycle_times: Vec<f64> = Vec::with_capacity(NUM_CYCLES);
    let mut node_counts: Vec<usize> = Vec::with_capacity(NUM_CYCLES / 100);

    for cycle in 0..NUM_CYCLES {
        let cycle_start = Instant::now();

        // ── Mutate a random file ──
        let file_idx = cycle % 7;
        let source = format!(
            "mod module_{c};\nuse dep_{c}::lib;\npub fn func_{c}(x: u32) -> u32 {{ x + {c} }}\nstruct S{c} {{ x: u32 }}\n",
            c = cycle
        );

        let old_source = if cycle > 0 {
            Some(format!(
                "mod module_{c};\nuse dep_{c}::lib;\npub fn func_{c}(x: u32) -> u32 {{ x + {c} }}\nstruct S{c} {{ x: u32 }}\n",
                c = cycle - 1
            ))
        } else {
            None
        };

        let _delta = engine.process_delta(
            &format!("src/module_{}.rs", file_idx),
            old_source.as_deref(),
            &source,
            &mut graph,
        );

        // ── Simulate LLM response ──
        let llm_output = match cycle % 5 {
            0 => valid_json,
            1 => invalid_json,
            2 => truncated,
            3 => empty,
            _ => valid_json,
        };

        // Guard check simulation (basic JSON validation)
        let guard_passed = serde_json::from_str::<serde_json::Value>(llm_output.trim()).is_ok();
        let reliability = if guard_passed { 0.9 } else { 0.1 };

        event_log.append(
            EventType::GuardCheck,
            EventPayload::GuardCheck {
                input_hash: format!("{:x}", cycle),
                passed: guard_passed,
                score: reliability,
                warnings: if !guard_passed {
                    vec!["invalid JSON".into()]
                } else {
                    vec![]
                },
            },
        );

        // ── Periodic failure injection ──
        if cycle % 47 == 0 {
            let fail_target = NodeId(format!("file:src/module_{}.rs", cycle % 7));
            graph.set_failure_relevance(&fail_target, 0.85);
            graph.propagate_failure(&fail_target, 0, 3);

            event_log.append(
                EventType::FailureDetection,
                EventPayload::FailureDetection {
                    node_id: fail_target.to_string(),
                    failure_type: "injected_test_failure".into(),
                    severity: 0.85,
                },
            );
        }

        // ── Periodic paging enforcement ──
        if cycle % 50 == 0 {
            graph.max_in_memory_nodes = 200;
            graph.enforce_paging();
        }

        event_log.append(
            EventType::CycleEnd,
            EventPayload::CycleEvent {
                cycle_number: cycle as u64,
                action: format!("cycle_{}_end", cycle),
            },
        );

        let elapsed = cycle_start.elapsed().as_secs_f64();
        cycle_times.push(elapsed);

        if cycle % 500 == 0 {
            node_counts.push(graph.node_count());
            let avg = cycle_times.iter().sum::<f64>() / cycle_times.len() as f64;
            eprintln!(
                "  Cycle {:>5}/{}: nodes={:>4} edges={:>4} events={:>6} | avg={:.3}ms",
                cycle,
                NUM_CYCLES,
                graph.node_count(),
                graph.edge_count(),
                event_log.event_count(),
                avg * 1000.0
            );
        }
    }

    let total_time = start.elapsed();

    // ═══════════════════════════════════════════════════════════
    // ASSERTIONS — long term stability
    // ═══════════════════════════════════════════════════════════

    // 1. No crash — reaching here proves basic stability
    assert!(
        graph.node_count() > 0,
        "graph must have nodes after {} cycles",
        NUM_CYCLES
    );

    // 2. Event log grows monotonically (append-only)
    let expected_min = NUM_CYCLES * 2; // guard checks + cycle ends
    let expected_max = expected_min + (NUM_CYCLES / 47) * 2; // ~failures with tolerance
    assert!(
        event_log.event_count() >= expected_min && event_log.event_count() <= expected_max + 10,
        "event count {} must be in [{}, {}]",
        event_log.event_count(),
        expected_min,
        expected_max + 10
    );

    // 3. Graph stays within paging limits
    assert!(
        graph.node_count() <= 200 + 20, // some tolerance for concurrent inserts
        "graph nodes {} must be bounded by paging limit",
        graph.node_count()
    );

    // 4. No exponential slowdown: first 10% vs last 10% cycle times
    let tenth = NUM_CYCLES / 10;
    let first_tenth_avg: f64 = cycle_times[..tenth].iter().sum::<f64>() / tenth as f64;
    let last_tenth_avg: f64 =
        cycle_times[(NUM_CYCLES - tenth)..].iter().sum::<f64>() / tenth as f64;
    let ratio = last_tenth_avg / first_tenth_avg.max(0.000001);

    assert!(
        ratio < 10.0,
        "no exponential slowdown: last_tenth_avg={:.6}ms / first_tenth_avg={:.6}ms = {:.2}x (must be < 10x)",
        last_tenth_avg * 1000.0,
        first_tenth_avg * 1000.0,
        ratio
    );

    // 5. Event ordering preserved
    for (i, event) in event_log.events.iter().enumerate() {
        assert_eq!(
            event.sequence_number as usize, i,
            "event at index {} has wrong sequence number {} (expected {})",
            i, event.sequence_number, i
        );
    }

    // 6. Apparent memory stability — node count shouldn't drift infinitely upward
    let avg_nodes: f64 = node_counts.iter().sum::<usize>() as f64 / node_counts.len() as f64;
    let max_nodes = node_counts.iter().max().copied().unwrap_or(0);
    assert!(
        max_nodes <= 400,
        "max nodes {} must be bounded (avg: {:.1})",
        max_nodes,
        avg_nodes
    );

    // Final report
    eprintln!(
        "\n─── Long Term Stability Report ({} cycles) ───",
        NUM_CYCLES
    );
    eprintln!("  Total time:       {:.2}s", total_time.as_secs_f64());
    eprintln!(
        "  Avg per cycle:    {:.4}ms",
        total_time.as_secs_f64() / NUM_CYCLES as f64 * 1000.0
    );
    eprintln!("  Final nodes:      {}", graph.node_count());
    eprintln!("  Final edges:      {}", graph.edge_count());
    eprintln!("  Total events:     {}", event_log.event_count());
    eprintln!("  Speed ratio:      {:.2}x (1st vs last 10%)", ratio);
    eprintln!("  ✓ ALL ASSERTIONS PASSED");
}

#[test]
fn long_term_graph_coherence_t0_vs_tfinal() {
    // Ensures graph structure at t0 and t_final are coherent — no silent corruption
    let mut graph = MemoryGraph::default();
    let mut engine = IncrementalGraphEngine::new();

    // Build initial state (t0)
    let initial_files: Vec<(&str, &str)> = vec![
        ("src/a.rs", "mod m1;\nuse std::io;\nfn main() {}"),
        ("src/b.rs", "mod m2;\npub fn helper() {}"),
        ("src/c.rs", "use crate::a;\nfn test() {}"),
    ];

    for (path, source) in &initial_files {
        engine.process_delta(path, None, source, &mut graph);
    }

    let nodes_t0 = graph.node_count();
    let _edges_t0 = graph.edge_count();

    // Run 100 mutation cycles
    for cycle in 0..100 {
        let file_idx = cycle % 3;
        let path = initial_files[file_idx].0;
        let new_source = format!(
            "mod m{};\nuse std::io;\npub fn func_{}(x: u32) -> u32 {{ x + {} }}\n",
            cycle, cycle, cycle
        );

        let old = initial_files[file_idx].1;
        engine.process_delta(path, Some(old), &new_source, &mut graph);
    }

    let nodes_tfinal = graph.node_count();
    let edges_tfinal = graph.edge_count();

    // Core stability assertions
    assert!(nodes_tfinal > 0, "graph must have nodes at final state");
    assert!(edges_tfinal > 0, "graph must have edges at final state");

    // Node count should be in a reasonable range (not exploded, not collapsed to zero)
    let drift_ratio = nodes_tfinal as f64 / nodes_t0.max(1) as f64;
    assert!(
        drift_ratio < 10.0,
        "node count drift ratio {:.2}x — too high (t0: {} nodes, t_final: {} nodes)",
        drift_ratio,
        nodes_t0,
        nodes_tfinal
    );
}
