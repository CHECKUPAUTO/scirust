// Bench-schema migration note: unlike other criterion targets in this
// workspace, this file has NO data-generation seed to pin -- `data` in both
// benchmarks below is a fixed, non-random ramp,
// `(0..10_000).map(|i| i as f32)`, with no `Lcg`/`PcgEngine`/
// `StdRng::seed_from_u64` call anywhere in the file. `BenchRecord::seed` is
// still mandatory, so `0` is used below purely as an explicit
// "no randomness" sentinel, not a discovered seed. Group `benches` contains
// `bench_scalar_add_one` (id `scalar_add_one_10k`) and `bench_simd_add_one`
// (id `simd_add_one_10k`); example after `cargo bench` writes
// `target/criterion/simd_add_one_10k/new/estimates.json`:
//   scirust_bench_schema::criterion_estimate_to_record(
//       &estimates_json, "scirust-core/simd_add_one", "ramp_10000_f32",
//       "simd_add_one", 0)
// See scirust-bench-schema's crate docs ("Migrating criterion targets") for
// the full pattern.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use scirust_core::simd_add_one;

fn bench_scalar_add_one(c: &mut Criterion) {
    let mut data: Vec<f32> = (0..10_000).map(|i| i as f32).collect();
    c.bench_function("scalar_add_one_10k", |b| {
        b.iter(|| {
            for x in data.iter_mut() {
                *x += 1.0;
            }
            black_box(&data);
        })
    });
}

fn bench_simd_add_one(c: &mut Criterion) {
    let mut data: Vec<f32> = (0..10_000).map(|i| i as f32).collect();
    c.bench_function("simd_add_one_10k", |b| {
        b.iter(|| {
            simd_add_one(black_box(&mut data));
        })
    });
}

criterion_group!(benches, bench_scalar_add_one, bench_simd_add_one);
criterion_main!(benches);
