//! MultiHeadAttention forward+backward step.
//!
//! Optional CLI args override the shape: `d_model n_heads batch seq [iters]`.
//! The default is the small-GEMM regime (seq=128, d_head=32); pass a larger
//! `seq` to move into the regime where the per-head score/context GEMMs
//! dominate and batching them matters most.

use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::init::{KaimingNormal, Zeros};
use scirust_core::nn::rng::PcgEngine;
use scirust_core::nn::transformer::attention::MultiHeadAttention;
use scirust_core::tensor::tensor3d::Var3D;
use std::time::Instant;

fn main() {
    let a: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    let d_model = a.first().copied().unwrap_or(256);
    let n_heads = a.get(1).copied().unwrap_or(8);
    let batch = a.get(2).copied().unwrap_or(16);
    let seq = a.get(3).copied().unwrap_or(128);
    let iters = a.get(4).copied().unwrap_or(20);

    let mut rng = PcgEngine::new(1);
    let mut mha =
        MultiHeadAttention::new(d_model, n_heads, 0, true, &KaimingNormal, &Zeros, &mut rng);
    let xd: Vec<f32> = (0..batch * seq * d_model)
        .map(|i| (i as f32 * 1e-4).sin())
        .collect();

    // Warmup.
    {
        let tape = Tape::new();
        let xv = tape.input(Tensor::from_vec(xd.clone(), batch * seq, d_model));
        let out = mha.forward_3d(&tape, Var3D::from_var(xv, batch, seq, d_model));
        out.as_var().sum().backward();
    }

    let t = Instant::now();
    for _ in 0..iters
    {
        let tape = Tape::new();
        let xv = tape.input(Tensor::from_vec(xd.clone(), batch * seq, d_model));
        let out = mha.forward_3d(&tape, Var3D::from_var(xv, batch, seq, d_model));
        out.as_var().sum().backward();
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let d_head = d_model / n_heads;
    println!(
        "MHA fwd+bwd (batch={batch} seq={seq} d_model={d_model} heads={n_heads} d_head={d_head} causal): {ms:.2} ms/step"
    );
}
