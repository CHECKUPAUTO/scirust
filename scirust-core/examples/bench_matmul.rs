use scirust_core::autodiff::reverse::Tensor;
use std::time::Instant;
fn main() {
    for (m, k, n) in [
        (128, 784, 256),
        (128, 256, 256),
        (128, 256, 10),
        (512, 512, 512),
    ]
    {
        let a = Tensor::from_vec(vec![0.5f32; m * k], m, k);
        let b = Tensor::from_vec(vec![0.3f32; k * n], k, n);
        let t = Instant::now();
        for _ in 0..20
        {
            let _ = a.matmul(&b);
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / 20.0;
        println!("{}x{} @ {}x{} : {:.2} ms/matmul", m, k, k, n, ms);
    }
}
