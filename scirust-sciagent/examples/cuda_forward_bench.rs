//! **Route B B5 — forward throughput: bf16 Tensor cores vs fp32 wgpu.**
//!
//! Times the *same* full decoder forward on both resident backends over one
//! sequence and reports tok/s + the speedup — the model-level realization of the
//! B0 microbench (which measured 12.9× for a bare bf16 GEMM). Both run the whole
//! `embed → N×GQA → final RMSNorm → tied head` on the Jetson Thor; only the
//! precision/hardware differs (wgpu fp32 CUDA cores vs cuBLASLt bf16 Tensor cores).
//!
//! Random weights (throughput is weight-independent). Size via env:
//! `SCIAGENT_D_MODEL` (512), `SCIAGENT_LAYERS` (8), `SCIAGENT_HEADS` (8),
//! `SCIAGENT_KV_HEADS` (2), `SCIAGENT_FF` (1408), `SCIAGENT_SEQ` (128),
//! `SCIAGENT_ITERS` (20).
//!
//! ```text
//! SCIAGENT_D_MODEL=1024 SCIAGENT_LAYERS=24 SCIAGENT_FF=4096 SCIAGENT_SEQ=256 \
//!   cargo run -p scirust-sciagent --features gpu,cuda --release --example cuda_forward_bench
//! ```
//!
//! Needs both a Vulkan adapter and a CUDA device — i.e. the Thor.

use std::time::Instant;

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::cuda_model::CudaModel;
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
    println!(
        "forward throughput @ seq {seq}, {iters} iters — d_model {} · {} layers · ff {}\n",
        config.d_model, config.n_layers, config.d_ff
    );

    // Route A — wgpu fp32 CUDA cores.
    let a_tps = match ResidentModel::from_model(&model)
    {
        Some(rm) =>
        {
            let _ = rm.forward(&tokens); // warmup
            let t = Instant::now();
            for _ in 0..iters
            {
                let _ = rm.forward(&tokens);
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

    // Route B — cuBLASLt bf16 Tensor cores.
    let b_tps = match CudaModel::from_model(&model)
    {
        Some(cm) =>
        {
            let _ = cm.forward(&tokens); // warmup
            let t = Instant::now();
            for _ in 0..iters
            {
                let _ = cm.forward(&tokens);
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
        println!("\n  Route B / Route A forward speedup: {:.1}×", b / a);
    }
    println!(
        "\nForward (prefill-like, m=seq) throughput — the compute-bound regime where\n\
         Tensor cores pay off. bf16 rounds inputs (rel_err ~2% vs fp32, see\n\
         cuda_parity); this is the precision/throughput trade Route B buys."
    );
}
