// Migration note (scirust-bench-schema): this file has NO random data
// generator to pin -- there is no seed_from_u64/Lcg::new/PcgEngine::new call
// anywhere below. `matmul_activation_graph` builds a purely structural
// `OpGraph` (MatMul/ReLU nodes with no literal tensor data at all), varied
// only by the `depth` argument, which `bench_pattern_fusion` calls with a
// single fixed value (32); there is no randomness to make reproducible in
// the first place. `BenchRecord::seed` is still mandatory, so a conversion
// uses `0` to mean "no RNG -- fixed, non-random input", e.g. (after
// `cargo bench --bench fusion_bench`, reading
// target/criterion/fusion/matmul_relu_depth_32/new/estimates.json):
//   scirust_bench_schema::criterion_estimate_to_record(
//       &estimates_json,
//       "scirust-fusion/pattern_fusion", // kernel
//       "depth=32",                      // dataset (the swept axis)
//       "FusionPipeline::fuse",          // method
//       0,                               // seed: no RNG, structural input only
//   )
// See scirust-bench-schema's crate docs ("Migrating criterion targets") for
// the full pattern.

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use scirust_fusion::{FusionPipeline, OpGraph, OpKind};

fn matmul_activation_graph(depth: usize) -> OpGraph {
    let mut graph = OpGraph::new();
    let input = graph.add_input(OpKind::Input, None);
    let weights = graph.add_input(OpKind::Input, None);
    let mut current = input;
    for _ in 0..depth
    {
        current = graph.add_binary(OpKind::MatMul, current, weights, None);
        current = graph.add_unary(OpKind::ReLU, current, None);
    }
    graph.add_unary(OpKind::Output, current, None);
    graph
}

fn bench_pattern_fusion(c: &mut Criterion) {
    let pipeline = FusionPipeline::new();
    c.bench_function("fusion/matmul_relu_depth_32", |b| {
        b.iter_batched(
            || matmul_activation_graph(32),
            |mut graph| black_box(pipeline.fuse(&mut graph)),
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_pattern_fusion);
criterion_main!(benches);
