//! **Resident inference micro-benchmark** — measure the on-device generation path
//! (fully-resident KV cache + batched prefill) on real hardware, so the
//! prefill/decode split is backed by numbers, not just asserted.
//!
//! Two regimes, both timed through the public `generate_cached` API (no dead A/B
//! code): `generate_cached(prompt, 1)` runs the batched **prefill** plus a single
//! argmax (the final decode is skipped when `max_new == 1`), so it *is* the
//! time-to-first-token; a longer run adds the incremental **decode** steps.
//!
//! - **Prefill** ingests the whole prompt in one wide forward — reported as
//!   *ingestion tok/s* (`P / t`). It stays high as `P` grows because it's one
//!   `m = P` forward, not `P` single-row ones (that's the infer-4 win).
//! - **Decode** is one `m = 1` forward per token — reported as *decode tok/s*.
//!   The gap between the two rates is the batching advantage.
//!
//! Weights are random (throughput is weight-independent — identical FLOPs), so no
//! checkpoint is needed. Size via env (defaults to a small model so it runs fast;
//! bump for 350M-class numbers):
//! `SCIAGENT_D_MODEL` (512), `SCIAGENT_LAYERS` (8), `SCIAGENT_HEADS` (8),
//! `SCIAGENT_KV_HEADS` (2), `SCIAGENT_FF` (1408), `SCIAGENT_MAX_SEQ` (1024),
//! `SCIAGENT_DECODE_N` (64).
//!
//! ```text
//! SCIAGENT_D_MODEL=1024 SCIAGENT_LAYERS=24 SCIAGENT_FF=4096 \
//!   cargo run -p scirust-sciagent --features gpu --release --example resident_infer_bench
//! ```
//!
//! Exit code 2 means no GPU adapter was found — run on the Thor or install a
//! Vulkan ICD.

use std::time::Instant;

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::gpu::ResidentModel;
use scirust_sciagent::model::SciAgentModel;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// A synthetic prompt of `len` in-vocab token ids.
fn prompt_of(len: usize, vocab: usize) -> Vec<u32> {
    (0..len)
        .map(|i| (i as u32 * 7 + 1) % vocab as u32)
        .collect()
}

/// Minimum wall-clock (ms) of `reps` runs of `f` — min rejects scheduler noise.
fn best_ms(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps
    {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
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
        max_seq_len: env_usize("SCIAGENT_MAX_SEQ", 1024),
        rope_theta: 10_000.0,
        tie_embeddings: true,
        use_bias: false,
        eps: 1e-5,
    };
    let model = SciAgentModel::new(&config);
    let Some(rm) = ResidentModel::from_model(&model)
    else
    {
        eprintln!("no GPU adapter available. Install a Vulkan ICD or run on the Jetson Thor.");
        std::process::exit(2);
    };
    println!("resident inference bench on: {}", rm.adapter_name());
    println!(
        "config: d_model {} · {} layers · {}/{} heads · ff {} · vocab {} · max_seq {}\n",
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.d_ff,
        vocab,
        config.max_seq_len
    );

    // Warm up (pipeline compile / first-alloc costs out of the measured region).
    let _ = rm.generate_cached(&prompt_of(8, vocab), 2);

    // --- Prefill: time-to-first-token vs prompt length ---------------------
    println!("prefill (batched, one m=P forward) — time to first token:");
    println!("   P    latency_ms   ingest_tok/s");
    let max_p = config.max_seq_len.saturating_sub(2);
    for &p in &[16usize, 64, 128, 256, 512]
    {
        if p > max_p
        {
            continue;
        }
        let prompt = prompt_of(p, vocab);
        let ms = best_ms(3, || {
            let _ = rm.generate_cached(&prompt, 1);
        });
        println!("{p:>5}   {ms:>9.2}   {:>11.1}", p as f64 / (ms / 1e3));
    }

    // --- Decode: incremental tok/s after a fixed short prompt --------------
    let decode_n = env_usize("SCIAGENT_DECODE_N", 64).max(2);
    let base = prompt_of(8, vocab);
    let prefill_ms = best_ms(3, || {
        let _ = rm.generate_cached(&base, 1);
    });
    let total_ms = best_ms(2, || {
        let _ = rm.generate_cached(&base, decode_n);
    });
    let decode_ms = (total_ms - prefill_ms).max(f64::MIN_POSITIVE);
    let steps = (decode_n - 1) as f64; // final token needs no forward
    println!(
        "\ndecode (incremental, one m=1 forward/token) over {decode_n} tokens:\n   \
         {:.2} ms total − {:.2} ms prefill = {:.2} ms  →  {:.1} tok/s",
        total_ms,
        prefill_ms,
        decode_ms,
        steps / (decode_ms / 1e3)
    );
    println!(
        "\nprefill ingests many tokens per forward; decode is one/forward — the\n\
         ratio is the batched-prefill advantage (infer-4)."
    );
}
