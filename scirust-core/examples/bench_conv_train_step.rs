//! Conv2d forward + backward step. The backward reuses the im2col matrix cached
//! by the forward instead of rebuilding it; this measures the whole fwd+bwd cost.

use scirust_core::autodiff::reverse::{Tape, Tensor};
use std::time::Instant;

fn main() {
    let (batch, in_c, out_c, h, w, k) = (128usize, 3usize, 32usize, 32usize, 32usize, 3usize);
    let xd = vec![0.02f32; batch * in_c * h * w];
    let wd = vec![0.01f32; out_c * in_c * k * k];

    // Warmup.
    {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(xd.clone(), batch, in_c * h * w));
        let weight = tape.input(Tensor::from_vec(wd.clone(), out_c, in_c * k * k));
        let y = x.conv2d_forward(weight, None, batch, in_c, h, w, out_c, k, 1, 1);
        y.sum().backward();
    }

    let iters = 10;
    let t = Instant::now();
    for _ in 0..iters
    {
        let tape = Tape::new();
        let x = tape.input(Tensor::from_vec(xd.clone(), batch, in_c * h * w));
        let weight = tape.input(Tensor::from_vec(wd.clone(), out_c, in_c * k * k));
        let y = x.conv2d_forward(weight, None, batch, in_c, h, w, out_c, k, 1, 1);
        y.sum().backward();
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("Conv2d fwd+bwd ({batch}x{in_c}x{h}x{w} -> {out_c} filters): {ms:.2} ms/step");
}
