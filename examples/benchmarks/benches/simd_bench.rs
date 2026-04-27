// examples/benchmarks/benches/simd_bench.rs
//
// cargo bench --package benchmarks --features scirust-core/portable-simd

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use scirust_core::matrix::{
    view::{MatrixView, MatrixViewMut},
    backend::{best_backend, ScalarBackend, SimdBackend},
};

// ------------------------------------------------------------------ //
//  Bench dot product                                                  //
// ------------------------------------------------------------------ //
fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_f32");

    for size in [256usize, 1024, 4096, 16384, 65536] {
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i as f32) * 0.5).collect();

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bh, _| {
            let backend = ScalarBackend;
            bh.iter(|| backend.sdot_f32(black_box(&a), black_box(&b)));
        });

        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bh, _| {
            let backend = best_backend();
            bh.iter(|| backend.sdot_f32(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

// ------------------------------------------------------------------ //
//  Bench AXPY                                                         //
// ------------------------------------------------------------------ //
fn bench_axpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("saxpy_f32");

    for size in [1024usize, 16384, 262144] {
        let x: Vec<f32> = (0..size).map(|i| i as f32).collect();

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bh, _| {
            let mut y = vec![0.0f32; size];
            let backend = ScalarBackend;
            bh.iter(|| backend.saxpy_f32(black_box(2.0), black_box(&x), &mut y));
        });

        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bh, _| {
            let mut y = vec![0.0f32; size];
            let backend = best_backend();
            bh.iter(|| backend.saxpy_f32(black_box(2.0), black_box(&x), &mut y));
        });
    }
    group.finish();
}

// ------------------------------------------------------------------ //
//  Bench GEMM                                                         //
// ------------------------------------------------------------------ //
fn bench_gemm(c: &mut Criterion) {
    let mut group = c.benchmark_group("sgemm_f32");

    for n in [32usize, 64, 128] {
        let a: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.01).collect();

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |bh, &n| {
            let mut c_data = vec![0.0f32; n * n];
            let backend = ScalarBackend;
            bh.iter(|| {
                let av = MatrixView::from_slice(black_box(&a), n, n);
                let bv = MatrixView::from_slice(black_box(&b), n, n);
                let cv = MatrixViewMut::from_slice(&mut c_data, n, n);
                backend.sgemm_f32(1.0, av, bv, 0.0, cv);
            });
        });

        group.bench_with_input(BenchmarkId::new("simd", n), &n, |bh, &n| {
            let mut c_data = vec![0.0f32; n * n];
            let backend = best_backend();
            bh.iter(|| {
                let av = MatrixView::from_slice(black_box(&a), n, n);
                let bv = MatrixView::from_slice(black_box(&b), n, n);
                let cv = MatrixViewMut::from_slice(&mut c_data, n, n);
                backend.sgemm_f32(1.0, av, bv, 0.0, cv);
            });
        });
    }
    group.finish();
}

// ------------------------------------------------------------------ //
//  Bench ReLU                                                         //
// ------------------------------------------------------------------ //
fn bench_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("relu_f32");

    for size in [4096usize, 65536, 1048576] {
        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bh, &size| {
            let mut v: Vec<f32> = (0..size).map(|i| i as f32 - (size / 2) as f32).collect();
            let backend = ScalarBackend;
            bh.iter(|| backend.relu_f32(black_box(&mut v)));
        });

        group.bench_with_input(BenchmarkId::new("simd", size), &size, |bh, &size| {
            let mut v: Vec<f32> = (0..size).map(|i| i as f32 - (size / 2) as f32).collect();
            let backend = best_backend();
            bh.iter(|| backend.relu_f32(black_box(&mut v)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_dot, bench_axpy, bench_gemm, bench_relu);
criterion_main!(benches);
