//! Cost of registering an input (weight/activation) on the tape. Each call used
//! to clone the tensor twice; now once.

use scirust_core::autodiff::reverse::{Tape, Tensor};
use std::time::Instant;

fn main() {
    let (rows, cols) = (2048usize, 2048usize);
    let w = Tensor::from_vec(vec![0.5f32; rows * cols], rows, cols);
    let iters = 300;

    // Warmup.
    {
        let tape = Tape::new();
        let _ = tape.input(w.clone());
    }

    let t = Instant::now();
    for _ in 0..iters
    {
        // Fresh tape each time = the per-forward pattern (modules re-inject
        // weights onto a new tape every step).
        let tape = Tape::new();
        let _ = tape.input(w.clone());
        std::hint::black_box(&tape);
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let mb = (rows * cols * 4) as f64 / 1e6;
    println!("tape.input of {rows}x{cols} ({mb:.0} MB): {ms:.3} ms/call");
}
