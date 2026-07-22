// examples/benchmarks/benches/io_bench.rs
//
// Benchmarks IO Safetensors de SciRust v11.2
//
// cd examples/benchmarks && cargo bench --bench io_bench

// Migration note (scirust-bench-schema): unlike most criterion targets in
// this workspace, this file has NO random data generator to pin -- there is
// no seed_from_u64/Lcg::new/PcgEngine::new call anywhere below. Every tensor
// in both `bench_serialize` and `bench_deserialize` is a fixed-size,
// fixed-value constant (`vec![0.5f32; 256 * 256]`, named "layer{i}.weight"),
// varied only along the swept `n_tensors` axis ([1, 8, 64]); there is no
// randomness to make reproducible in the first place. `BenchRecord::seed`
// is still mandatory, so a conversion uses `0` to mean "no RNG -- fixed,
// non-random input", e.g. (after `cargo bench --bench io_bench`, reading
// target/criterion/safetensors_serialize/64/new/estimates.json):
//   scirust_bench_schema::criterion_estimate_to_record(
//       &estimates_json,
//       "scirust-core/safetensors_serialize", // kernel
//       "n_tensors=64",                       // dataset (the swept axis)
//       "serialize_state_dict",                // method
//       0,                                     // seed: no RNG, constant input
//   )
// See scirust-bench-schema's crate docs ("Migrating criterion targets") for the full pattern.

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
