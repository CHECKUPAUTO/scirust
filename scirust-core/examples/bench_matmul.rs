use scirust_core::autodiff::reverse::Tensor;
use std::time::Instant;

fn bench(m: usize, k: usize, n: usize, iters: usize) {
    let a = Tensor::from_vec(vec![0.5f32; m * k], m, k);
    let b = Tensor::from_vec(vec![0.3f32; k * n], k, n);
    // Warmup (kernel init, thread pool spin-up).
    let _ = a.matmul(&b);
    let t = Instant::now();
    for _ in 0..iters
    {
        let _ = a.matmul(&b);
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let gflops = (2.0 * m as f64 * k as f64 * n as f64) / (ms * 1e6);
    println!("{m:>5}x{k:<5} @ {k:>5}x{n:<5} : {ms:8.3} ms/matmul  ({gflops:7.1} GFLOP/s)");
}

fn main() {
    println!(
        "threads = {}",
        std::thread::available_parallelism().map_or(0, |n| n.get())
    );
    // Small (should stay single-threaded, below the parallel threshold).
    bench(128, 256, 10, 200);
    bench(128, 256, 256, 100);
    // Medium / large (parallelized).
    bench(128, 784, 256, 100);
    bench(512, 512, 512, 40);
    bench(1024, 1024, 1024, 10);
}
