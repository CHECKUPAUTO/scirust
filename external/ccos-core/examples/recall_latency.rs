//! Latency micro-benchmark for the recall strategies, to find where (and whether)
//! the per-recall embedding-store rebuild actually becomes a bottleneck — so the
//! perf work is data-driven, not blind.
//!
//! Run: `cargo run --release --example recall_latency`
//!      `cargo run --release --example recall_latency --features learned-embed`
//!
//! `working_set` / `around` touch only the graph; `semantic` / `hybrid` rebuild the
//! TF-IDF (and, under `learned-embed`, the LSA) store from all resident nodes on
//! every call — that's the cost we want to quantify across corpus sizes.

use ccos_core::external_memory::{CcosMemory, ExternalMemory, Recall};
use std::time::Instant;

fn corpus(n: usize) -> CcosMemory {
    let mut mem = CcosMemory::new();
    for i in 0..n {
        // A few distinctive terms per file so TF-IDF/LSA have real structure.
        let src = format!(
            "// module {i}: handles the area_{} workflow with helper_{}.\n\
             pub fn run_{i}() -> u32 {{ {i} }}\n\
             pub fn apply_{}(x: u32) -> u32 {{ x + {} }}\n",
            i % 17,
            i % 23,
            i % 13,
            i % 7
        );
        mem.ingest_source(&format!("src/m{i}.rs"), &src);
    }
    mem
}

fn time_recall(mem: &mut CcosMemory, recall: &Recall, reps: usize) -> f64 {
    // Warm up once (build any lazy state), then time `reps` calls.
    let _ = mem.recall(recall, 2048);
    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = mem.recall(recall, 2048);
    }
    t0.elapsed().as_secs_f64() * 1e6 / reps as f64 // microseconds per recall
}

fn main() {
    let embedder = if cfg!(feature = "learned-embed") {
        "LSA (learned-embed)"
    } else {
        "INT4 TF-IDF (default)"
    };
    println!("# Recall latency (µs/call) — semantic embedder: {embedder}\n");
    print!("{:<8}", "nodes");
    for s in ["working_set", "around", "task", "semantic", "hybrid"] {
        print!("{s:>13}");
    }
    println!();

    for &n in &[200usize, 800, 2000] {
        let mut mem = corpus(n);
        let anchor = "file:src/m0.rs";
        let reps = if n >= 2000 { 20 } else { 50 };
        let cells = [
            time_recall(&mut mem, &Recall::working_set(), reps),
            time_recall(&mut mem, &Recall::around(anchor), reps),
            time_recall(&mut mem, &Recall::task("area workflow helper"), reps),
            time_recall(&mut mem, &Recall::semantic("area workflow helper"), reps),
            time_recall(&mut mem, &Recall::hybrid("area workflow helper"), reps),
        ];
        print!("{n:<8}");
        for c in cells {
            print!("{c:>12.0} ");
        }
        println!();
    }
}
