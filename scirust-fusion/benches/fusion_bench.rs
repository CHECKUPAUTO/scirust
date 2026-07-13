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
