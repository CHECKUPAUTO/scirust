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
