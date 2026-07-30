//! CCOS v0.3 — Benchmark framework integration tests, including a 100k-cycle
//! stress run and an opt-in 1,000,000-cycle long-stability run.

use ccos_core::benchmark::{BenchmarkHarness, BenchmarkReport};

#[test]
fn stress_100k_cycles_stays_bounded() {
    let report = BenchmarkHarness::new().with_paging_cap(200).run(100_000);

    assert_eq!(report.cycles, 100_000);
    assert_eq!(
        report.dangling_edges, 0,
        "no dangling edges over 100k cycles"
    );
    assert!(report.final_nodes <= 200, "nodes bounded by paging cap");
    assert!(
        report.peak_nodes <= 220,
        "peak {} unbounded",
        report.peak_nodes
    );
    assert!(
        report.node_drift.abs() <= 32,
        "node drift {} indicates a leak",
        report.node_drift
    );
    assert!(report.cycles_per_second > 1_000.0, "unexpectedly slow");
}

#[test]
fn report_roundtrips_through_json() {
    let report = BenchmarkHarness::new().run(1_000);
    let json = report.to_json();
    let parsed: BenchmarkReport = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.cycles, 1_000);
    assert_eq!(parsed.dangling_edges, 0);
    assert!(!parsed.samples.is_empty());
}

/// Long-stability run. Opt-in (slow): `cargo test --test benchmark -- --ignored`.
#[test]
#[ignore = "long-running (~1M cycles); run with --ignored"]
fn long_stability_1m_cycles() {
    let report = BenchmarkHarness::new().with_paging_cap(200).run(1_000_000);

    assert_eq!(report.cycles, 1_000_000);
    assert_eq!(report.dangling_edges, 0, "no dangling edges over 1M cycles");
    assert!(report.final_nodes <= 200);
    assert!(
        report.node_drift.abs() <= 64,
        "node count must not drift over 1M cycles: {}",
        report.node_drift
    );
}
