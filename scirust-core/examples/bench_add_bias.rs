//! Bias add: old path (materialize a full broadcast replicate, then add) vs the
//! fused single-pass `zip_broadcasted`. Row-vector bias over a (batch × feat)
//! activation — the Linear-layer bias pattern.

use scirust_core::autodiff::reverse::Tensor;
use std::time::Instant;

fn main() {
    let (rows, cols) = (4096usize, 4096usize);
    let a = Tensor::from_vec(vec![1.0f32; rows * cols], rows, cols);
    let bias = Tensor::from_vec((0..cols).map(|i| i as f32 * 1e-3).collect(), 1, cols);
    let iters = 100;

    // Warmup.
    let mut sink = 0.0f32;
    sink += a.add(&bias.broadcast_to(rows, cols)).data[0];
    sink += a.zip_broadcasted(&bias, |x, y| x + y).data[0];

    let t = Instant::now();
    for _ in 0..iters
    {
        sink += a.add(&bias.broadcast_to(rows, cols)).data[1];
    }
    let old_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let t = Instant::now();
    for _ in 0..iters
    {
        sink += a.zip_broadcasted(&bias, |x, y| x + y).data[1];
    }
    let new_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    println!(
        "bias add {rows}x{cols}:  old (broadcast_to + add) {old_ms:.3} ms   fused {new_ms:.3} ms   ({:.2}x)   sink {sink}",
        old_ms / new_ms
    );
}
