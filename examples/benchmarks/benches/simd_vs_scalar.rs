use criterion::{black_box, criterion_group, criterion_main, Criterion};
use scirust_core::{simd_add_one, simd_map};

fn bench_scalar_add_one(c: &mut Criterion) {
    let mut data: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
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
    let mut data: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    c.bench_function("simd_add_one_10k", |b| {
        b.iter(|| {
            simd_add_one(black_box(&mut data));
        })
    });
}

fn bench_scalar_map(c: &mut Criterion) {
    let mut data: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    c.bench_function("scalar_map_sin_10k", |b| {
        b.iter(|| {
            for x in data.iter_mut() {
                *x = x.sin();
            }
            black_box(&data);
        })
    });
}

fn bench_simd_map(c: &mut Criterion) {
    let mut data: Vec<f64> = (0..10_000).map(|i| i as f64).collect();
    c.bench_function("simd_map_sin_10k", |b| {
        b.iter(|| {
            simd_map(black_box(&mut data), |x| x.sin());
        })
    });
}

criterion_group!(benches, bench_scalar_add_one, bench_simd_add_one, bench_scalar_map, bench_simd_map);
criterion_main!(benches);
