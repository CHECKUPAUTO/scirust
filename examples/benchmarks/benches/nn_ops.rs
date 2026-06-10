// examples/benchmarks/benches/nn_ops.rs
//
// Benchmarks des opérations NN et autodiff de SciRust v11.2
//
// cd examples/benchmarks && cargo bench --bench nn_ops

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::loss::CrossEntropyLoss;
use scirust_core::nn::module::Module;
use scirust_core::nn::Loss;
use scirust_core::nn::Linear;
use scirust_core::nn::tt_decompose;
use scirust_core::nn::init::{KaimingNormal, Zeros};
use scirust_core::nn::rng::PcgEngine;
use scirust_core::tn::factorize::auto_factorize;

// ------------------------------------------------------------------ //
//  Bench sum_axis                                                      //
// ------------------------------------------------------------------ //
fn bench_sum_axis(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_axis");
    for (rows, cols) in [(64, 784), (256, 256), (1024, 1024)] {
        let t = Tensor::from_vec(vec![1.0f32; rows * cols], rows, cols);
        group.bench_with_input(BenchmarkId::new("axis0", format!("{}x{}", rows, cols)), &(rows, cols), |b, _| {
            b.iter(|| { black_box(&t).sum_axis(0); })
        });
        group.bench_with_input(BenchmarkId::new("axis1", format!("{}x{}", rows, cols)), &(rows, cols), |b, _| {
            b.iter(|| { black_box(&t).sum_axis(1); })
        });
    }
    group.finish();
}

// ------------------------------------------------------------------ //
//  Bench broadcast_to                                                  //
// ------------------------------------------------------------------ //
fn bench_broadcast(c: &mut Criterion) {
    let mut group = c.benchmark_group("broadcast");
    for (from, to) in [((1, 256), (64, 256)), ((64, 1), (64, 256)), ((1, 1), (1024, 1024))] {
        let t = Tensor::from_vec(vec![1.0f32; from.0 * from.1], from.0, from.1);
        group.bench_with_input(
            BenchmarkId::new("to", format!("{}x{}->{}x{}", from.0, from.1, to.0, to.1)),
            &(from, to),
            |b, _| { b.iter(|| { black_box(&t).broadcast_to(to.0, to.1); }) }
        );
    }
    group.finish();
}

// ------------------------------------------------------------------ //
//  Bench softmax / log_softmax                                         //
// ------------------------------------------------------------------ //
fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");
    for (rows, cols) in [(64, 10), (256, 256), (1024, 1024)] {
        let t = Tensor::from_vec(vec![1.0f32; rows * cols], rows, cols);
        group.bench_with_input(BenchmarkId::new("softmax", format!("{}x{}", rows, cols)), &(rows, cols), |b, _| {
            b.iter(|| { black_box(&t).softmax(1); })
        });
        group.bench_with_input(BenchmarkId::new("log_softmax", format!("{}x{}", rows, cols)), &(rows, cols), |b, _| {
            b.iter(|| { black_box(&t).softmax(1).log(); })
        });
    }
    group.finish();
}

// ------------------------------------------------------------------ //
//  Bench Linear forward                                                //
// ------------------------------------------------------------------ //
fn bench_linear_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_forward");
    for (batch, in_f, out_f) in [(64, 784, 256), (256, 256, 128), (1024, 512, 512)] {
        let mut rng = PcgEngine::new(42);
        let mut lin = Linear::new(in_f, out_f, &KaimingNormal, &Zeros, &mut rng);
        group.bench_with_input(
            BenchmarkId::new("", format!("{}x{}->{}", batch, in_f, out_f)),
            &(batch, in_f, out_f),
            |b, _| {
                b.iter(|| {
                    let tape = Tape::new();
                    let x = tape.input(Tensor::from_vec(vec![0.5f32; batch * in_f], batch, in_f));
                    black_box(&mut lin).forward(&tape, x);
                })
            }
        );
    }
    group.finish();
}

// ------------------------------------------------------------------ //
//  Bench CrossEntropy forward + backward                               //
// ------------------------------------------------------------------ //
fn bench_cross_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_entropy");
    for (batch, classes) in [(64, 10), (256, 100), (1024, 1000)] {
        let loss_fn = CrossEntropyLoss::new();

        group.bench_with_input(
            BenchmarkId::new("forward", format!("{}x{}", batch, classes)),
            &(batch, classes),
            |b, _| {
                b.iter(|| {
                    let tape = Tape::new();
                    let pred = tape.input(Tensor::from_vec(vec![0.5f32; batch * classes], batch, classes));
                    let mut target_data = vec![0.0f32; batch * classes];
                    for b in 0..batch { target_data[b * classes] = 1.0; }
                    let target = tape.input(Tensor::from_vec(target_data, batch, classes));
                    black_box(&loss_fn).forward(&tape, pred, target);
                })
            }
        );

        group.bench_with_input(
            BenchmarkId::new("forward+backward", format!("{}x{}", batch, classes)),
            &(batch, classes),
            |b, _| {
                b.iter(|| {
                    let tape = Tape::new();
                    let pred = tape.input(Tensor::from_vec(vec![0.5f32; batch * classes], batch, classes));
                    let mut target_data = vec![0.0f32; batch * classes];
                    for b in 0..batch { target_data[b * classes] = 1.0; }
                    let target = tape.input(Tensor::from_vec(target_data, batch, classes));
                    let loss = loss_fn.forward(&tape, pred, target);
                    loss.backward();
                })
            }
        );
    }
    group.finish();
}

// ------------------------------------------------------------------ //
//  Bench matmul (via Tensor)                                           //
// ------------------------------------------------------------------ //
fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    for n in [32usize, 128, 512] {
        let a = Tensor::from_vec(vec![0.01f32; n * n], n, n);
        let b = Tensor::from_vec(vec![0.01f32; n * n], n, n);
        group.bench_with_input(BenchmarkId::new("", n), &n, |bh, _| {
            bh.iter(|| { black_box(&a).matmul(black_box(&b)); })
        });
    }
    group.finish();
}

// ------------------------------------------------------------------ //
//  Bench TT-Linear forward vs dense Linear                            //
// ------------------------------------------------------------------ //
fn bench_tt_linear_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("tt_linear_forward");
    let mut rng = PcgEngine::new(42);

    // Sizes: (in_features, out_features, ndims, max_rank)
    // ndims controls how many factors in the TT decomposition
    let configs: &[(usize, usize, usize, usize)] = &[
        (48, 96, 2, 4),    // small, rank 4
        (256, 128, 2, 8),  // medium, rank 8
        (512, 512, 2, 16), // large, rank 16
    ];

    for &(in_f, out_f, ndims, max_rank) in configs {
        let mut linear = Linear::new(in_f, out_f, &KaimingNormal, &Zeros, &mut rng);
        let in_dims = auto_factorize(in_f, ndims);
        let out_dims = auto_factorize(out_f, ndims);
        let tt = tt_decompose(&linear, &in_dims, &out_dims, max_rank, 0.0);

        group.bench_with_input(
            BenchmarkId::new("linear", format!("{}x{}", in_f, out_f)),
            &(in_f, out_f),
            |b, _| {
                b.iter(|| {
                    let tape = Tape::new();
                    let x = tape.input(Tensor::from_vec(vec![0.5f32; 64 * in_f], 64, in_f));
                    black_box(&mut linear).forward(&tape, x);
                })
            }
        );
        group.bench_with_input(
            BenchmarkId::new("tt_linear", format!("{}x{}_r{}", in_f, out_f, max_rank)),
            &(in_f, out_f),
            |b, _| {
                let mut tt = tt.clone();
                b.iter(|| {
                    let tape = Tape::new();
                    let x = tape.input(Tensor::from_vec(vec![0.5f32; 64 * in_f], 64, in_f));
                    black_box(&mut tt).forward(&tape, x);
                })
            }
        );
    }
    group.finish();
}

criterion_group!(benches, bench_sum_axis, bench_broadcast, bench_softmax, bench_linear_forward, bench_tt_linear_forward, bench_cross_entropy, bench_matmul);
criterion_main!(benches);
