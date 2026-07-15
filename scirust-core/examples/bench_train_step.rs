//! Forward + backward matmul step (exercises the tape's `Op::MatMul` forward and
//! both transposed backward GEMMs). Run with `--release`; compare threaded vs
//! single-threaded with `--no-default-features` (disables the `rayon` feature).

use scirust_core::autodiff::reverse::{Tape, Tensor};
use std::time::Instant;

fn bench(batch: usize, in_dim: usize, out_dim: usize, iters: usize) {
    let xd = vec![0.02f32; batch * in_dim];
    let wd = vec![0.01f32; in_dim * out_dim];
    // Warmup.
    {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(xd.clone(), batch, in_dim));
        let w = tape.input(Tensor::from_vec(wd.clone(), in_dim, out_dim));
        let loss = x.matmul(w).sum();
        loss.backward();
    }
    let t = Instant::now();
    for _ in 0..iters
    {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(xd.clone(), batch, in_dim));
        let w = tape.input(Tensor::from_vec(wd.clone(), in_dim, out_dim));
        let loss = x.matmul(w).sum();
        loss.backward();
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    // 3 GEMMs of ~batch·in·out each (forward + 2 backward).
    let gflops = (3.0 * 2.0 * batch as f64 * in_dim as f64 * out_dim as f64) / (ms * 1e6);
    println!(
        "fwd+bwd  {batch}x{in_dim} @ {in_dim}x{out_dim} : {ms:8.3} ms/step  ({gflops:7.1} GFLOP/s)"
    );
}

fn main() {
    println!(
        "threads = {}",
        std::thread::available_parallelism().map_or(0, |n| n.get())
    );
    bench(512, 1024, 1024, 20);
    bench(256, 512, 512, 40);
}
