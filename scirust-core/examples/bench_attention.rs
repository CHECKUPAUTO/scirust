//! MultiHeadAttention forward+backward step.

use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::init::{KaimingNormal, Zeros};
use scirust_core::nn::rng::PcgEngine;
use scirust_core::nn::transformer::attention::MultiHeadAttention;
use scirust_core::tensor::tensor3d::Var3D;
use std::time::Instant;

fn main() {
    let (d_model, n_heads, batch, seq) = (256usize, 8usize, 16usize, 128usize);
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

    let iters = 20;
    let t = Instant::now();
    for _ in 0..iters
    {
        let tape = Tape::new();
        let xv = tape.input(Tensor::from_vec(xd.clone(), batch * seq, d_model));
        let out = mha.forward_3d(&tape, Var3D::from_var(xv, batch, seq, d_model));
        out.as_var().sum().backward();
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!(
        "MHA fwd+bwd (batch={batch} seq={seq} d_model={d_model} heads={n_heads} causal): {ms:.2} ms/step"
    );
}
