//! **How much does the resident GPU path actually buy on this device?** Times
//! the VRAM-resident GPU compute against the deterministic CPU oracle on the
//! same work, and prints the speedup. On a Jetson Thor's Blackwell GPU (or any
//! real adapter) this is the honest, on-device answer; on Mesa *lavapipe* it is
//! correct but slow (software rasteriser), so the "speedup" there is not
//! meaningful — the point is the mechanism and its measurement.
//!
//!   cargo run -p scirust-gpu --features wgpu --release --example gpu_bench
//!
//! Note: `--release` matters a lot — the CPU oracle is a scalar triple loop.

use std::time::Instant;

use scirust_gpu::ops::{cpu_scale_causal_mask, cpu_softmax};
use scirust_gpu::{CpuBackend, GpuChain, RawComputeBackend};

/// Wall-clock ms per iteration of a chained square GEMM, GPU (resident) vs CPU.
/// Chaining `X ← X·B` keeps the intermediate in VRAM (the residency win) and
/// avoids re-uploading each iteration; a single download at the end flushes the
/// GPU queue so the timing includes real completion.
fn bench_matmul(chain: &GpuChain, n: usize, iters: usize) -> (f64, f64) {
    let a: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
    let b: Vec<f32> = (0..n * n)
        .map(|i| ((i % 5) as f32 - 2.0) / n as f32)
        .collect();

    let ga = chain.upload(&a, n, n);
    let gb = chain.upload(&b, n, n);
    for _ in 0..3
    {
        let _ = chain.download(&chain.matmul(&ga, &gb).unwrap()).unwrap(); // warm up
    }
    let t = Instant::now();
    let mut cur = chain.matmul(&ga, &gb).unwrap();
    for _ in 1..iters
    {
        cur = chain.matmul(&cur, &gb).unwrap();
    }
    std::hint::black_box(&chain.download(&cur).unwrap());
    let gpu_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

    let t = Instant::now();
    let mut c = CpuBackend.gemm_f32(&a, &b, n, n, n).unwrap();
    for _ in 1..iters
    {
        c = CpuBackend.gemm_f32(&c, &b, n, n, n).unwrap();
    }
    std::hint::black_box(&c);
    let cpu_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    (gpu_ms, cpu_ms)
}

/// Single-head resident attention `softmax((Q·Kᵀ)/√d + mask)·V` vs a CPU
/// reference, ms per iteration.
fn bench_attention(chain: &GpuChain, t_len: usize, d: usize, iters: usize) -> (f64, f64) {
    let mk = |phase: f32, n: usize| -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.01 + phase).sin()).collect()
    };
    let (q, k, v) = (mk(0.0, t_len * d), mk(1.0, t_len * d), mk(2.0, t_len * d));
    let (gq, gk, gv) = (
        chain.upload(&q, t_len, d),
        chain.upload(&k, t_len, d),
        chain.upload(&v, t_len, d),
    );
    for _ in 0..3
    {
        let _ = chain
            .download(&chain.attention(&gq, &gk, &gv, true).unwrap())
            .unwrap();
    }
    let t = Instant::now();
    for _ in 0..iters
    {
        std::hint::black_box(
            &chain
                .download(&chain.attention(&gq, &gk, &gv, true).unwrap())
                .unwrap(),
        );
    }
    let gpu_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

    // CPU reference: S = Q·Kᵀ → scale+mask → softmax → ·V.
    let scale = 1.0 / (d as f32).sqrt();
    let t = Instant::now();
    for _ in 0..iters
    {
        let mut kt = vec![0.0f32; d * t_len];
        for r in 0..t_len
        {
            for c in 0..d
            {
                kt[c * t_len + r] = k[r * d + c];
            }
        }
        let s = CpuBackend.gemm_f32(&q, &kt, t_len, d, t_len).unwrap();
        let s = cpu_scale_causal_mask(&s, t_len, t_len, scale, true);
        let w = cpu_softmax(&s, t_len, t_len);
        std::hint::black_box(&CpuBackend.gemm_f32(&w, &v, t_len, t_len, d).unwrap());
    }
    let cpu_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    (gpu_ms, cpu_ms)
}

fn main() {
    let Some(chain) = GpuChain::new()
    else
    {
        eprintln!("no GPU adapter available. Install a Vulkan ICD or run on the Jetson Thor.");
        std::process::exit(2);
    };
    println!("GPU adapter: {}\n", chain.adapter_name());
    println!("Resident GPU compute vs the CPU oracle (ms per iteration, --release).");
    println!("On lavapipe the GPU is a software rasteriser — the speedup is only");
    println!("meaningful on a real adapter (e.g. the Thor's Blackwell).\n");

    println!(
        "{:<28}  {:>11}  {:>11}  {:>9}",
        "workload", "GPU ms", "CPU ms", "speedup"
    );
    for &n in &[128usize, 256, 512]
    {
        let (g, c) = bench_matmul(&chain, n, 20);
        println!(
            "{:<28}  {:>11.3}  {:>11.3}  {:>8.1}x",
            format!("matmul {n}×{n}·{n}×{n}"),
            g,
            c,
            c / g.max(f64::MIN_POSITIVE)
        );
    }
    for &(t_len, d) in &[(128usize, 64usize), (256, 64), (512, 64)]
    {
        let (g, c) = bench_attention(&chain, t_len, d, 20);
        println!(
            "{:<28}  {:>11.3}  {:>11.3}  {:>8.1}x",
            format!("attention t={t_len} d={d}"),
            g,
            c,
            c / g.max(f64::MIN_POSITIVE)
        );
    }
    println!(
        "\nGPU numbers include the input upload amortised over the chain and one\n\
         final download to flush the queue; CPU is the deterministic scalar oracle."
    );
}
