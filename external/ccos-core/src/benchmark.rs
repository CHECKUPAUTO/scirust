//! # Benchmark Framework (CCOS v0.3)
//!
//! Drives the kernel through a large number of edit/analyze cycles and measures
//! throughput, graph evolution and drift, producing a serializable
//! [`BenchmarkReport`] (e.g. `benchmark_report.json`). It is the in-process
//! counterpart to the `criterion` micro-benches in `benches/`.
//!
//! Memory is reported as a node/edge proxy (the kernel's working set) rather
//! than process RSS, to stay dependency-free.

use crate::event_log::{EventLog, EventPayload, EventType};
use crate::incremental::IncrementalGraphEngine;
use crate::memory::{MemoryGraph, NodeId};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// A periodic measurement taken during a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSample {
    pub cycle: usize,
    pub nodes: usize,
    pub edges: usize,
    pub events: usize,
    pub avg_cycle_us: f64,
}

/// The full result of a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub version: String,
    pub cycles: usize,
    pub total_seconds: f64,
    pub avg_cycle_us: f64,
    pub cycles_per_second: f64,
    pub final_nodes: usize,
    pub final_edges: usize,
    pub final_events: usize,
    pub peak_nodes: usize,
    pub peak_edges: usize,
    /// Node-count drift between the first sample and the last (should be ~0 for
    /// a stable, bounded system).
    pub node_drift: i64,
    pub dangling_edges: usize,
    pub paging_cap: usize,
    pub samples: Vec<BenchmarkSample>,
}

impl BenchmarkReport {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }
}

/// Configurable cycle benchmark.
#[derive(Debug, Clone)]
pub struct BenchmarkHarness {
    pub paging_cap: usize,
    pub files: usize,
    pub sample_every: usize,
}

impl Default for BenchmarkHarness {
    fn default() -> Self {
        Self {
            paging_cap: 200,
            files: 8,
            sample_every: 10_000,
        }
    }
}

impl BenchmarkHarness {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_paging_cap(mut self, cap: usize) -> Self {
        self.paging_cap = cap.max(1);
        self
    }

    /// Run `cycles` edit cycles and return the measured report. Each cycle
    /// modifies one file (round-robin), logs events, and periodically injects a
    /// failure — exercising the incremental engine, paging and event log.
    pub fn run(&self, cycles: usize) -> BenchmarkReport {
        let mut graph = MemoryGraph::new(0.1, self.paging_cap);
        let mut engine = IncrementalGraphEngine::new();
        let mut event_log = EventLog::new("benchmark".into());

        let start = Instant::now();
        let mut samples: Vec<BenchmarkSample> = Vec::new();
        let mut peak_nodes = 0usize;
        let mut peak_edges = 0usize;
        let mut baseline_nodes: Option<usize> = None;
        let sample_every = self.sample_every.max(1);

        for cycle in 0..cycles {
            let file_idx = cycle % self.files.max(1);
            let path = format!("src/module_{file_idx}.rs");
            let new_source = format!(
                "mod m{c};\nuse dep_{c}::lib;\npub fn func_{c}(x: u32) -> u32 {{ x + {c} }}\nstruct S{c};\n",
                c = cycle
            );
            // Treat as a modification so the file's subgraph is evicted + rebuilt
            // (the O(Δ) path), keeping the graph bounded.
            engine.process_delta(&path, Some("fn old() {}"), &new_source, &mut graph);

            event_log.append(
                EventType::CycleEnd,
                EventPayload::CycleEvent {
                    cycle_number: cycle as u64,
                    action: "bench_cycle".into(),
                },
            );

            if cycle % 47 == 0 {
                let target = NodeId(format!("file:src/module_{}.rs", cycle % self.files.max(1)));
                graph.set_failure_relevance(&target, 0.85);
                graph.propagate_failure(&target, 0, 3);
            }

            peak_nodes = peak_nodes.max(graph.node_count());
            peak_edges = peak_edges.max(graph.edge_count());

            if cycle % sample_every == 0 {
                let elapsed = start.elapsed().as_secs_f64();
                let avg_us = if cycle > 0 {
                    elapsed / cycle as f64 * 1_000_000.0
                } else {
                    0.0
                };
                if baseline_nodes.is_none() && cycle > 0 {
                    baseline_nodes = Some(graph.node_count());
                }
                samples.push(BenchmarkSample {
                    cycle,
                    nodes: graph.node_count(),
                    edges: graph.edge_count(),
                    events: event_log.event_count(),
                    avg_cycle_us: avg_us,
                });
            }
        }

        let total = start.elapsed().as_secs_f64();
        let dangling = graph.prune_dangling_edges();
        let final_nodes = graph.node_count();
        let node_drift = baseline_nodes
            .map(|b| final_nodes as i64 - b as i64)
            .unwrap_or(0);

        BenchmarkReport {
            version: env!("CARGO_PKG_VERSION").to_string(),
            cycles,
            total_seconds: total,
            avg_cycle_us: if cycles > 0 {
                total / cycles as f64 * 1_000_000.0
            } else {
                0.0
            },
            cycles_per_second: if total > 0.0 {
                cycles as f64 / total
            } else {
                0.0
            },
            final_nodes,
            final_edges: graph.edge_count(),
            final_events: event_log.event_count(),
            peak_nodes,
            peak_edges,
            node_drift,
            dangling_edges: dangling,
            paging_cap: self.paging_cap,
            samples,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_runs_and_stays_bounded() {
        let harness = BenchmarkHarness::new().with_paging_cap(150);
        let report = harness.run(5_000);

        assert_eq!(report.cycles, 5_000);
        assert_eq!(report.dangling_edges, 0, "no dangling edges after a run");
        assert!(
            report.final_nodes <= 150,
            "nodes must respect the paging cap"
        );
        assert!(report.peak_nodes <= 150 + 8, "peak nodes must stay bounded");
        // Bounded system: node count must not drift significantly.
        assert!(
            report.node_drift.abs() <= 32,
            "node drift {} too large",
            report.node_drift
        );
        assert!(report.cycles_per_second > 0.0);
    }

    #[test]
    fn report_serializes_to_json() {
        let report = BenchmarkHarness::new().run(200);
        let json = report.to_json();
        assert!(json.contains("\"cycles\": 200"));
        let parsed: BenchmarkReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.cycles, 200);
    }
}
