//! **Route B B4g — training throughput: bf16 Tensor cores vs fp32 wgpu.**
//!
//! Times one *full AdamW training step* (forward → cross-entropy grad → backward →
//! optimizer, all resident) on both backends over a fixed batch and reports tok/s +
//! the speedup — the training-side analogue of `cuda_forward_bench` (which measures
//! the forward-only number). Route A is `ResidentModel::train_step` (fp32 wgpu CUDA
//! cores); Route B is `CudaTrainer::train_step` (cuBLASLt bf16 Tensor cores + fp32
//! master weights). Both run the whole `embed → N×GQA → RMSNorm → tied head`
//! backward and an AdamW update of every weight; only the precision/hardware differ.
//!
//! Random weights (throughput is weight-independent). Size via env:
//! `SCIAGENT_D_MODEL` (512), `SCIAGENT_LAYERS` (8), `SCIAGENT_HEADS` (8),
//! `SCIAGENT_KV_HEADS` (2), `SCIAGENT_FF` (1408), `SCIAGENT_SEQ` (128),
//! `SCIAGENT_ITERS` (20).
//!
//! ```text
//! SCIAGENT_D_MODEL=1024 SCIAGENT_LAYERS=24 SCIAGENT_FF=4096 SCIAGENT_SEQ=256 \
//!   cargo run -p scirust-sciagent --features gpu,cuda --release --example cuda_train_bench
//! ```
//!
//! Needs both a Vulkan adapter and a CUDA device — i.e. the Thor.

use std::time::Instant;

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::cuda_model::CudaTrainer;
use scirust_sciagent::gpu::ResidentModel;
use scirust_sciagent::model::SciAgentModel;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let vocab = 256usize;
    let config = SciAgentConfig {
        vocab_size: vocab,
        d_model: env_usize("SCIAGENT_D_MODEL", 512),
        n_layers: env_usize("SCIAGENT_LAYERS", 8),
        n_heads: env_usize("SCIAGENT_HEADS", 8),
        n_kv_heads: env_usize("SCIAGENT_KV_HEADS", 2),
        d_ff: env_usize("SCIAGENT_FF", 1408),
        max_seq_len: env_usize("SCIAGENT_MAX_SEQ", 2048),
        rope_theta: 10_000.0,
        tie_embeddings: true,
        use_bias: false,
        eps: 1e-5,
    };
    let model = SciAgentModel::new(&config);
    let seq = env_usize("SCIAGENT_SEQ", 128).min(config.max_seq_len);
    let iters = env_usize("SCIAGENT_ITERS", 20).max(1);
    let tokens: Vec<u32> = (0..seq)
        .map(|i| (i as u32 * 7 + 1) % vocab as u32)
        .collect();
    let targets: Vec<u32> = (0..seq)
        .map(|i| (i as u32 * 5 + 2) % vocab as u32)
        .collect();
    let (lr, betas, eps, wd) = (1e-4f32, (0.9f32, 0.999f32), 1e-8f32, 0.01f32);
    println!(
        "training throughput @ seq {seq}, {iters} iters — d_model {} · {} layers · ff {}\n",
        config.d_model, config.n_layers, config.d_ff
    );

    // Route A — wgpu fp32 CUDA cores.
    let a_tps = match ResidentModel::from_model(&model)
    {
        Some(mut rm) =>
        {
            let _ = rm.train_step(&tokens, &targets, lr, betas, eps, wd); // warmup
            let t = Instant::now();
            for _ in 0..iters
            {
                let _ = rm.train_step(&tokens, &targets, lr, betas, eps, wd);
            }
            let s = t.elapsed().as_secs_f64();
            let tps = (seq * iters) as f64 / s;
            println!(
                "  Route A (wgpu fp32):      {tps:>8.1} tok/s   [{}]",
                rm.adapter_name()
            );
            Some(tps)
        },
        None =>
        {
            println!("  Route A: no Vulkan adapter");
            None
        },
    };

    // Route B — cuBLASLt bf16 Tensor cores + fp32 master weights.
    let b_tps = match CudaTrainer::from_model(&model)
    {
        Some(mut ct) =>
        {
            let _ = ct.train_step(&tokens, &targets, lr, betas, eps, wd); // warmup
            let t = Instant::now();
            for _ in 0..iters
            {
                let _ = ct.train_step(&tokens, &targets, lr, betas, eps, wd);
            }
            let s = t.elapsed().as_secs_f64();
            let tps = (seq * iters) as f64 / s;
            println!("  Route B (CUDA bf16 TC):   {tps:>8.1} tok/s");
            Some(tps)
        },
        None =>
        {
            println!("  Route B: no CUDA device");
            None
        },
    };

    if let (Some(a), Some(b)) = (a_tps, b_tps)
    {
        println!("\n  Route B / Route A training speedup: {:.1}×", b / a);
    }
    println!(
        "\nFull training step (forward + cross-entropy grad + backward + AdamW),\n\
         entirely resident. bf16 rounds inputs (rel_err ~2% fwd / a few % on grads,\n\
         see cuda_parity); the fp32 master weights + moments preserve convergence."
    );
}
