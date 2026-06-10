use scirust_core::autodiff::reverse::Tensor;
use scirust_core::nn::conv_utils::{col2im_raw, im2col_raw};
use std::time::Instant;
fn bench<F: Fn()>(label: &str, f: F) {
    let t = Instant::now();
    for _ in 0..10
    {
        f();
    }
    println!(
        "{label}: {:.2} ms/appel",
        t.elapsed().as_secs_f64() * 1000.0 / 10.0
    );
}
fn main() {
    // conv1: b=128, c=3, 32x32, k=3, pad=1
    let in1 = Tensor::zeros(128, 3 * 32 * 32);
    let col1 = Tensor::zeros(27, 128 * 32 * 32);
    bench("im2col conv1 (3->32)", || {
        im2col_raw(&in1, 128, 3, 32, 32, 3, 1, 1);
    });
    bench("col2im conv1", || {
        col2im_raw(&col1, 128, 3, 32, 32, 3, 1, 1);
    });
    // conv2: b=128, c=32, 16x16, k=3, pad=1
    let in2 = Tensor::zeros(128, 32 * 16 * 16);
    let col2 = Tensor::zeros(288, 128 * 16 * 16);
    bench("im2col conv2 (32->64)", || {
        im2col_raw(&in2, 128, 32, 16, 16, 3, 1, 1);
    });
    bench("col2im conv2", || {
        col2im_raw(&col2, 128, 32, 16, 16, 3, 1, 1);
    });
}
