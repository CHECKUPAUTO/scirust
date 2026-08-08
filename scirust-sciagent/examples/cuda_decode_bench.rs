//! I250 batch-one CUDA decode benchmark.
//!
//! This intentionally excludes the training sweep from `cuda_production_bench` so a
//! future Thor validation window can measure inference alone. The benchmark keeps
//! B49's cached decoder as the token-parity oracle and reports the fused path against
//! the user-facing 250 tok/s target.
//!
//! The benchmark also prints a simple weight-stream roofline. Batch-one autoregressive
//! decode is normally dominated by reading the model weights for every generated token;
//! the roofline therefore reports how much of the configured DRAM bandwidth a target
//! would consume if every weight byte were read exactly once and all other traffic were
//! free. It is an optimistic bound, not a throughput prediction.

use std::time::Instant;

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::cuda_decode::CudaDecodeModel;
use scirust_sciagent::cuda_model::CudaModel;
use scirust_sciagent::generate::SamplingParams;
use scirust_sciagent::model::SciAgentModel;

const THOR_T5000_PEAK_MEMORY_GBPS: f64 = 273.0;
const BF16_BYTES_PER_PARAMETER: f64 = 2.0;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_bool(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .is_some_and(|v| matches!(v.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn synthetic_tokens(len: usize, vocab: usize) -> Vec<u32> {
    (0..len)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (x % vocab as u64) as u32
        })
        .collect()
}

fn print_roofline(params: usize, memory_gbps: f64, target_tps: f64, stretch_tps: f64) {
    let weight_bytes = params as f64 * BF16_BYTES_PER_PARAMETER;
    let bytes_per_gb = 1_000_000_000.0;
    let bf16_roof_tps = memory_gbps * bytes_per_gb / weight_bytes;
    let target_gbps = target_tps * weight_bytes / bytes_per_gb;
    let stretch_gbps = stretch_tps * weight_bytes / bytes_per_gb;

    println!(
        "SCIAGENT_I250_ROOFLINE params={params} precision=bf16 weight_bytes={:.0} memory_peak_gbps={memory_gbps:.3} bf16_weight_stream_roof_tok_s={bf16_roof_tps:.3} target_tok_s={target_tps:.3} target_min_weight_gbps={target_gbps:.3} target_peak_fraction={:.4} stretch_tok_s={stretch_tps:.3} stretch_min_weight_gbps={stretch_gbps:.3} stretch_peak_fraction={:.4}",
        weight_bytes,
        target_gbps / memory_gbps,
        stretch_gbps / memory_gbps,
    );
}

fn main() {
    let config = SciAgentConfig::sciagent_350m();
    let prompt_len = env_usize("SCIAGENT_DECODE_PROMPT", 128).max(1);
    let max_new = env_usize("SCIAGENT_DECODE_NEW", 64).max(1);
    let target_tps = env_f64("SCIAGENT_DECODE_TARGET_TPS", 250.0);
    let stretch_tps = env_f64("SCIAGENT_DECODE_STRETCH_TPS", 750.0);
    let memory_gbps = env_f64(
        "SCIAGENT_DECODE_MEMORY_GBPS",
        THOR_T5000_PEAK_MEMORY_GBPS,
    );
    let require_target = env_bool("SCIAGENT_DECODE_REQUIRE_TARGET");
    assert!(memory_gbps > 0.0, "decode memory bandwidth must be positive");
    assert!(
        prompt_len + max_new <= config.max_seq_len,
        "benchmark request exceeds max_seq_len"
    );

    print_roofline(
        config.total_parameters(),
        memory_gbps,
        target_tps,
        stretch_tps,
    );

    let model = SciAgentModel::new(&config);
    let Some(oracle) = CudaModel::from_model(&model) else {
        eprintln!("no CUDA Route-B runtime available");
        std::process::exit(2);
    };
    let Some(fast) = CudaDecodeModel::from_model(&model) else {
        eprintln!("no fused CUDA decode runtime available");
        std::process::exit(2);
    };

    let prompt = synthetic_tokens(prompt_len, config.vocab_size);
    let params = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
        repetition_penalty: 1.0,
        repetition_window: 64,
    };
    let seed = 0x4932_3530_5448_4F52u64;

    // Warm NVRTC/cuBLASLt and both decode implementations outside the measurement.
    let _ = fast.generate(&prompt, 1, &params, seed);
    let _ = oracle.generate_cached(&prompt, 1, &params, seed);

    let t0 = Instant::now();
    let fast_tokens = fast.generate(&prompt, max_new, &params, seed);
    let fast_secs = t0.elapsed().as_secs_f64().max(1e-9);
    let fast_new = fast_tokens.len().saturating_sub(prompt.len());
    let fast_tps = fast_new as f64 / fast_secs;

    let t1 = Instant::now();
    let oracle_tokens = oracle.generate_cached(&prompt, max_new, &params, seed);
    let oracle_secs = t1.elapsed().as_secs_f64().max(1e-9);
    let oracle_new = oracle_tokens.len().saturating_sub(prompt.len());
    let oracle_tps = oracle_new as f64 / oracle_secs;

    let parity = fast_tokens == oracle_tokens;
    let speedup = if oracle_tps > 0.0 {
        fast_tps / oracle_tps
    } else {
        f64::NAN
    };
    let target_met = fast_tps >= target_tps;
    let stretch_met = fast_tps >= stretch_tps;

    println!(
        "SCIAGENT_I250_DECODE params={} prompt={} requested_new={} fast_new={} fast_seconds={:.6} fast_tok_s={:.3} b49_new={} b49_seconds={:.6} b49_tok_s={:.3} speedup={:.3} target_tok_s={:.3} target_met={} stretch_tok_s={:.3} stretch_met={} parity={}",
        config.total_parameters(),
        prompt_len,
        max_new,
        fast_new,
        fast_secs,
        fast_tps,
        oracle_new,
        oracle_secs,
        oracle_tps,
        speedup,
        target_tps,
        target_met,
        stretch_tps,
        stretch_met,
        parity
    );

    if !parity {
        eprintln!("ERROR: fused CUDA decode diverged from the B49 cached oracle");
        std::process::exit(3);
    }
    if require_target && !target_met {
        eprintln!(
            "ERROR: fused CUDA decode {:.3} tok/s is below required {:.3} tok/s",
            fast_tps, target_tps
        );
        std::process::exit(4);
    }
}
