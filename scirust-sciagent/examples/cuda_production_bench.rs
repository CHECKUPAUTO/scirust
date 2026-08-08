//! Production-oriented Route-B benchmark for the Jetson AGX Thor.
//!
//! Unlike the historical microbenches, this uses the real `sciagent_350m` shape
//! (~304M parameters), the production `CudaTrainer::pretrain` path (including the
//! B23-B38 kernels/cache/batching/telemetry semantics), and reports an ETA for one
//! pass over the known 1.03B-token v4 corpus. It also compares the naive CUDA
//! generator with B31's resident KV-cache path.
//!
//! No CUDA result is baked into the source. Run it on the Thor and preserve stdout
//! as the hardware record:
//!
//! ```text
//! SCIAGENT_BENCH_BATCHES=1,2,4,8 SCIAGENT_BENCH_SEQ=512 \
//! SCIAGENT_BENCH_STEPS=8 cargo run -p scirust-sciagent --features cuda \
//!   --release --example cuda_production_bench
//! ```

use std::time::Instant;

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::cuda_model::{CudaModel, CudaPretrainConfig, CudaTrainer};
use scirust_sciagent::generate::SamplingParams;
use scirust_sciagent::model::SciAgentModel;

const V4_CORPUS_TOKENS: u64 = 1_029_492_639;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn batch_list() -> Vec<usize> {
    let raw = std::env::var("SCIAGENT_BENCH_BATCHES").unwrap_or_else(|_| "1,2,4,8".into());
    let mut out: Vec<usize> = raw
        .split(',')
        .filter_map(|v| v.trim().parse::<usize>().ok())
        .filter(|&v| v > 0)
        .collect();
    out.sort_unstable();
    out.dedup();
    if out.is_empty()
    {
        out.push(1);
    }
    out
}

fn synthetic_tokens(len: usize, vocab: usize) -> Vec<u32> {
    // Deterministic, cheap, non-constant stream. Quality is irrelevant here: this
    // bench measures the exact compute path without requiring the 4.1 GB shard set.
    (0..len)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (x % vocab as u64) as u32
        })
        .collect()
}

fn bench_training(config: &SciAgentConfig, batch: usize, seq: usize, steps: usize) -> Option<f64> {
    let mut model = SciAgentModel::new(config);
    let Some(mut trainer) = CudaTrainer::from_model(&model)
    else
    {
        eprintln!("no CUDA device available — this benchmark must run on the Thor");
        return None;
    };

    // Enough independent windows to avoid wrapping during the warmup+measurement.
    let windows = batch.saturating_mul(steps.saturating_add(4)).max(batch + 1);
    let tokens = synthetic_tokens(
        windows.saturating_mul(seq).saturating_add(1),
        config.vocab_size,
    );

    // One untimed production step warms NVRTC/cuBLASLt/allocation paths. No eval or
    // checkpoint I/O is enabled; telemetry is flushed at the end of this one step.
    let warm = CudaPretrainConfig {
        base_lr: 1e-4,
        min_lr: 1e-4,
        warmup_steps: 1,
        total_steps: 1,
        start_step: 0,
        seq_len: seq,
        batch_size: batch,
        telemetry_interval: 1,
        log_interval: 0,
        save_interval: 0,
        val_frac: 0.0,
        eval_interval: 0,
        shuffle: false,
        ..Default::default()
    };
    let warm_losses = trainer.pretrain(&tokens, &mut model, config, &warm);
    if warm_losses.is_empty()
    {
        eprintln!("warmup produced no step for B{batch}×T{seq}");
        return None;
    }

    // Production timing: diagnostics stay resident for the whole measured block and
    // flush once at the end, matching the B30 long-run behavior rather than forcing
    // a host barrier after every optimizer step.
    let measured = CudaPretrainConfig {
        base_lr: 1e-4,
        min_lr: 1e-4,
        warmup_steps: 1,
        total_steps: 1 + steps,
        start_step: 1,
        seq_len: seq,
        batch_size: batch,
        telemetry_interval: steps.max(1),
        log_interval: 0,
        save_interval: 0,
        val_frac: 0.0,
        eval_interval: 0,
        shuffle: false,
        ..Default::default()
    };
    let t0 = Instant::now();
    let losses = trainer.pretrain(&tokens, &mut model, config, &measured);
    let secs = t0.elapsed().as_secs_f64().max(1e-9);
    if losses.len() != steps
    {
        eprintln!(
            "B{batch}×T{seq}: expected {steps} measured steps, got {}",
            losses.len()
        );
        return None;
    }
    let processed = batch.saturating_mul(seq).saturating_mul(steps) as f64;
    let tok_s = processed / secs;
    let last_loss = losses.last().copied().unwrap_or(f32::NAN);
    println!(
        "SCIAGENT_THOR_TRAIN batch={batch} seq={seq} steps={steps} tokens={} seconds={secs:.6} tok_s={tok_s:.3} last_loss={last_loss:.6}",
        processed as u64
    );
    Some(tok_s)
}

fn bench_decode(config: &SciAgentConfig, prompt_len: usize, max_new: usize) {
    let model = SciAgentModel::new(config);
    let Some(cuda) = CudaModel::from_model(&model)
    else
    {
        return;
    };
    let prompt = synthetic_tokens(prompt_len.max(1), config.vocab_size);
    let params = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
        repetition_penalty: 1.0,
        repetition_window: 64,
    };

    // Small untimed calls ensure kernels/BLAS paths are hot before either timing.
    let _ = cuda.generate_cached(&prompt, 1, &params, 0x5343_4941);
    let _ = cuda.generate(&prompt, 1, &params, 0x5343_4941);

    let seed = 0x5343_4941_4745_4E54;
    let t0 = Instant::now();
    let cached = cuda.generate_cached(&prompt, max_new, &params, seed);
    let cached_secs = t0.elapsed().as_secs_f64().max(1e-9);
    let cached_new = cached.len().saturating_sub(prompt.len());
    let cached_tps = cached_new as f64 / cached_secs;

    let t1 = Instant::now();
    let naive = cuda.generate(&prompt, max_new, &params, seed);
    let naive_secs = t1.elapsed().as_secs_f64().max(1e-9);
    let naive_new = naive.len().saturating_sub(prompt.len());
    let naive_tps = naive_new as f64 / naive_secs;

    let parity = cached == naive;
    let speedup = if naive_tps > 0.0
    {
        cached_tps / naive_tps
    }
    else
    {
        f64::NAN
    };
    println!(
        "SCIAGENT_THOR_DECODE prompt={prompt_len} requested_new={max_new} cached_new={cached_new} cached_seconds={cached_secs:.6} cached_tok_s={cached_tps:.3} naive_new={naive_new} naive_seconds={naive_secs:.6} naive_tok_s={naive_tps:.3} speedup={speedup:.3} parity={parity}"
    );
    if !parity
    {
        eprintln!("ERROR: cached CUDA decoding changed greedy tokens");
        std::process::exit(3);
    }
}

fn main() {
    let config = SciAgentConfig::sciagent_350m();
    let seq = env_usize("SCIAGENT_BENCH_SEQ", 512).min(config.max_seq_len);
    let steps = env_usize("SCIAGENT_BENCH_STEPS", 8).max(1);
    let corpus_tokens = env_u64("SCIAGENT_BENCH_CORPUS_TOKENS", V4_CORPUS_TOKENS);
    let prompt_len = env_usize("SCIAGENT_BENCH_PROMPT", 128).max(1);
    let decode_new = env_usize("SCIAGENT_BENCH_DECODE_NEW", 8).max(1);
    let batches = batch_list();

    println!(
        "SCIAGENT_THOR_CONFIG params={} vocab={} d_model={} layers={} q_heads={} kv_heads={} max_seq={} bench_seq={} batches={:?}",
        config.total_parameters(),
        config.vocab_size,
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.max_seq_len,
        seq,
        batches
    );

    let mut best: Option<(usize, f64)> = None;
    for batch in batches
    {
        println!("--- training sweep B{batch}×T{seq} ---");
        let Some(tok_s) = bench_training(&config, batch, seq, steps)
        else
        {
            std::process::exit(2);
        };
        let days = corpus_tokens as f64 / tok_s / 86_400.0;
        println!(
            "SCIAGENT_THOR_PASS batch={batch} corpus_tokens={corpus_tokens} tok_s={tok_s:.3} one_pass_days={days:.3}"
        );
        if best.is_none_or(|(_, best_tps)| tok_s > best_tps)
        {
            best = Some((batch, tok_s));
        }
    }
    if let Some((batch, tok_s)) = best
    {
        println!("SCIAGENT_THOR_BEST_TRAIN batch={batch} tok_s={tok_s:.3}");
    }

    println!("--- decode cache parity/throughput ---");
    bench_decode(&config, prompt_len, decode_new);
}
