// examples/benchmarks/benches/io_bench.rs
//
// Benchmarks IO Safetensors de SciRust v11.2
//
// cd examples/benchmarks && cargo bench --bench io_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use scirust_core::autodiff::reverse::Tensor;
use scirust_core::io::safetensors::{serialize_state_dict, deserialize_state_dict};
use std::collections::HashMap;

fn bench_serialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("safetensors_serialize");
    for n_tensors in [1usize, 8, 64] {
        let mut state = HashMap::new();
        for i in 0..n_tensors {
            let name = format!("layer{}.weight", i);
            let t = Tensor::from_vec(vec![0.5f32; 256 * 256], 256, 256);
            state.insert(name, t);
        }
        let meta = HashMap::new();
        group.bench_with_input(BenchmarkId::new("", n_tensors), &n_tensors, |b, _| {
            b.iter(|| { black_box(serialize_state_dict(black_box(&state), black_box(&meta))); })
        });
    }
    group.finish();
}

fn bench_deserialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("safetensors_deserialize");
    for n_tensors in [1usize, 8, 64] {
        let mut state = HashMap::new();
        for i in 0..n_tensors {
            let name = format!("layer{}.weight", i);
            let t = Tensor::from_vec(vec![0.5f32; 256 * 256], 256, 256);
            state.insert(name, t);
        }
        let meta = HashMap::new();
        let bytes = serialize_state_dict(&state, &meta);
        group.bench_with_input(BenchmarkId::new("", n_tensors), &n_tensors, |b, _| {
            b.iter(|| { black_box(deserialize_state_dict(black_box(&bytes)).unwrap()); })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_serialize, bench_deserialize);
criterion_main!(benches);
