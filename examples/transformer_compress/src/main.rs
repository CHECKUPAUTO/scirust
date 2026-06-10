//! Benchmark skeleton: measure memory + reconstruction error + decomposition
//! time for transformer-style FFN projections.
//!
//! Run with `cargo run --release -p transformer_compress` or enable as a
//! proper criterion bench in `Cargo.toml`. The current Cargo.toml has the
//! `[[bench]]` section commented; uncomment it once `criterion` is added as
//! a dev-dependency.
//!
//! Intended to run on the Jetson Thor for realistic numbers. CPU-only here
//! since Phase 1 doesn't yet wire to scirust-gpu / cudarc.

use scirust_core::nn::{Linear, tt_decompose, tt_decompose_auto};
use std::time::Instant;

fn frob_norm(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn frob_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn fill_weight(linear: &mut Linear, seed: u32) {
    let n = linear.in_features * linear.out_features;
    // Deterministic pseudo-random fill, fine for a benchmark
    let mut state = seed as u64;
    for k in 0..n
    {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let v = ((state >> 33) as i32 as f32) / (i32::MAX as f32);
        linear.weight.data[k] = v;
        let _ = k;
    }
    for k in 0..linear.out_features
    {
        linear.bias.data[k] = 0.01 * (k as f32);
    }
}

/// Bench one (in, out, n_factors, max_rank) configuration.
fn bench_one(in_features: usize, out_features: usize, n_factors: usize, max_rank: usize, tol: f32) {
    let mut rng = scirust_core::nn::rng::PcgEngine::new(42);
    let mut linear = Linear::new(
        in_features,
        out_features,
        &scirust_core::nn::init::Zeros,
        &scirust_core::nn::init::Zeros,
        &mut rng,
    );
    fill_weight(&mut linear, 42);

    let t0 = Instant::now();
    let tt = tt_decompose_auto(&linear, n_factors, max_rank, tol);
    let decomp_time = t0.elapsed();

    let t1 = Instant::now();
    let w_recon = tt.reconstruct_weight();
    let recon_time = t1.elapsed();

    let rel_err = frob_err(&linear.weight.data, &w_recon.data) / frob_norm(&linear.weight.data);

    println!("  {in_features}x{out_features}, d={n_factors}, max_rank={max_rank}, tol={tol:.0e}",);
    println!("    in_dims = {:?}", tt.in_dims);
    println!("    out_dims = {:?}", tt.out_dims);
    println!("    ranks = {:?}", tt.ranks);
    println!(
        "    params: dense={}  TT={}  ratio={:.2}x",
        tt.dense_params(),
        tt.num_params(),
        tt.compression_ratio()
    );
    println!("    decompose time: {decomp_time:?}");
    println!("    reconstruct time: {recon_time:?}");
    println!("    relative Frobenius error: {rel_err:.4e}");
    println!();
}

fn main() {
    println!("=== SciRust: transformer FFN compression bench ===\n");

    // Typical transformer FFN shapes (small models, scaled down for fast bench)
    // GPT-2 small style FFN: 768 -> 3072
    bench_one(768, 3072, 3, 32, 1e-3);
    bench_one(768, 3072, 3, 64, 1e-4);
    bench_one(768, 3072, 2, 64, 1e-4);

    // GPT-2 small attention QKV proj: 768 -> 768
    bench_one(768, 768, 3, 16, 1e-3);
    bench_one(768, 768, 3, 32, 1e-4);

    // Smaller model FFN: 256 -> 1024
    bench_one(256, 1024, 3, 16, 1e-3);
    bench_one(256, 1024, 2, 32, 1e-4);

    let _ = tt_decompose; // silence unused import if manual path is removed
}
