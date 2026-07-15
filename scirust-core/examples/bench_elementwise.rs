//! Elementwise op throughput (add / hadamard) on a large tensor.

use scirust_core::autodiff::reverse::Tensor;
use std::time::Instant;

fn main() {
    let (rows, cols) = (2048usize, 2048usize);
    let a = Tensor::from_vec(vec![1.0009f32; rows * cols], rows, cols);
    let b = Tensor::from_vec(vec![0.9991f32; rows * cols], rows, cols);
    let iters = 200;

    // Warmup.
    let mut acc = 0.0f32;
    acc += a.add(&b).data[0];
    acc += a.hadamard(&b).data[0];

    let t = Instant::now();
    for _ in 0..iters
    {
        acc += a.add(&b).data[0];
    }
    let add_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let t = Instant::now();
    for _ in 0..iters
    {
        acc += a.hadamard(&b).data[0];
    }
    let had_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let mb = (rows * cols * 4) as f64 / 1e6;
    println!(
        "{rows}x{cols} ({mb:.0} MB/operand)  add: {add_ms:.3} ms   hadamard: {had_ms:.3} ms   (sink {acc})"
    );
}
