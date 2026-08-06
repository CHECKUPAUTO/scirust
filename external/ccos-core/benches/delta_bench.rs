//! Micro-benchmarks for the incremental engine.
//!
//! The kernel's headline claim is that a single-file edit costs `O(Δ)` — work
//! proportional to the *change*, not to the size of the repository already in
//! the graph. These benches measure `process_delta` for one file while the
//! surrounding graph holds 0, 500, 2,000 and 8,000 unrelated nodes; the per-edit
//! time should stay roughly flat as the background grows.
//!
//! Run with `cargo bench`.

use ccos_core::incremental::IncrementalGraphEngine;
use ccos_core::memory::MemoryGraph;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn file_source(i: usize) -> String {
    format!(
        "mod m{i};\nuse dep_{i}::lib;\npub fn func_{i}(x: u32) -> u32 {{ x + {i} }}\nstruct S{i} {{ x: u32 }}\n",
        i = i
    )
}

/// Pre-populate `graph` with roughly `n` unrelated nodes from distinct files.
fn populate(engine: &mut IncrementalGraphEngine, graph: &mut MemoryGraph, n: usize) {
    let files = n / 4 + 1; // ~4 nodes per file
    for f in 0..files {
        let path = format!("bg/file_{f}.rs");
        engine.process_delta(&path, None, &file_source(f), graph);
    }
}

fn bench_process_delta(c: &mut Criterion) {
    let mut group = c.benchmark_group("process_delta_one_file");
    for background in [0usize, 500, 2000, 8000] {
        // Large paging cap so the background actually stays resident.
        let mut engine = IncrementalGraphEngine::new();
        let mut graph = MemoryGraph::new(0.0, usize::MAX);
        populate(&mut engine, &mut graph, background);

        let mut version = 0u64;
        group.bench_with_input(
            BenchmarkId::from_parameter(background),
            &background,
            |b, _| {
                b.iter(|| {
                    // Edit a single hot file repeatedly; cost should be O(Δ),
                    // independent of `background`.
                    version += 1;
                    let new =
                        format!("mod hot;\npub fn hot_{version}() {{}}\nstruct H{version};\n");
                    engine.process_delta("hot/file.rs", Some("fn old() {}"), &new, &mut graph);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_process_delta);
criterion_main!(benches);
