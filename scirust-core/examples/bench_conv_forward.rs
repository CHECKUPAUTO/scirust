use scirust_core::autodiff::reverse::{Tape, Tensor};
use std::time::Instant;

fn main() {
    let tape = Tape::new();
    // Conv layer similar to the one in benchmarks (32 filters, 3x3 kernel)
    let batch = 128;
    let in_c = 3;
    let out_c = 32;
    let h = 32;
    let w = 32;
    let k = 3;

    let x = tape.input(Tensor::zeros(batch, in_c * h * w));
    let weight = tape.input(Tensor::zeros(out_c, in_c * k * k));

    let t = Instant::now();
    for _ in 0..10
    {
        let _ = x.conv2d_forward(weight, None, batch, in_c, h, w, out_c, k, 1, 1);
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / 10.0;
    println!(
        "Conv2d forward ({}x{}x{}x{} -> {} filters): {:.2} ms/forward",
        batch, in_c, h, w, out_c, ms
    );
}
