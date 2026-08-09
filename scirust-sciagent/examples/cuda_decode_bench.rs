//! I250 batch-one CUDA decode benchmark.
//!
//! One resident `CudaDecodeModel` is exercised under multiple implementation modes
//! so A/B numbers are not polluted by different CUDA contexts or weight uploads.

use std::time::Instant;

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::cuda_decode::{
    CudaDecodeDownMode, CudaDecodeFfnMode, CudaDecodeLmHeadMode, CudaDecodeModel, CudaDecodeModes,
};
use scirust_sciagent::cuda_model::CudaModel;
use scirust_sciagent::generate::SamplingParams;
use scirust_sciagent::model::SciAgentModel;

const BF16_BYTES_PER_PARAMETER: f64 = 2.0;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_optional_f64(key: &str) -> Option<f64> {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
}

fn env_bool(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn synthetic_tokens(len: usize, vocab: usize) -> Vec<u32> {
    (0..len)
        .map(|index| {
            let mixed = (index as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (mixed % vocab as u64) as u32
        })
        .collect()
}

fn print_roofline(params: usize, memory_gbps: f64, target_tps: f64, stretch_tps: f64) {
    let weight_bytes = params as f64 * BF16_BYTES_PER_PARAMETER;
    let bytes_per_gb = 1_000_000_000.0;
    let roof_tps = memory_gbps * bytes_per_gb / weight_bytes;
    let target_gbps = target_tps * weight_bytes / bytes_per_gb;
    let stretch_gbps = stretch_tps * weight_bytes / bytes_per_gb;
    println!(
        "SCIAGENT_I250_ROOFLINE provenance=explicit_env params={params} precision=bf16 weight_bytes={:.0} memory_gbps={memory_gbps:.3} bf16_weight_stream_roof_tok_s={roof_tps:.3} target_tok_s={target_tps:.3} target_min_weight_gbps={target_gbps:.3} target_bandwidth_fraction={:.4} stretch_tok_s={stretch_tps:.3} stretch_min_weight_gbps={stretch_gbps:.3} stretch_bandwidth_fraction={:.4}",
        weight_bytes,
        target_gbps / memory_gbps,
        stretch_gbps / memory_gbps,
    );
}

fn throughput(new_tokens: usize, seconds: f64) -> f64 {
    new_tokens as f64 / seconds.max(1e-9)
}

fn timed_generate(
    model: &CudaDecodeModel,
    prompt: &[u32],
    max_new: usize,
    modes: CudaDecodeModes,
) -> (Vec<u32>, f64, f64) {
    let started = Instant::now();
    let tokens = model.generate_greedy_device_feedback_with_modes(prompt, max_new, modes);
    let seconds = started.elapsed().as_secs_f64();
    let new_tokens = tokens.len().saturating_sub(prompt.len());
    (tokens, seconds, throughput(new_tokens, seconds))
}

fn main() {
    let config = SciAgentConfig::sciagent_350m();
    let prompt_len = env_usize("SCIAGENT_DECODE_PROMPT", 128).max(1);
    let max_new = env_usize("SCIAGENT_DECODE_NEW", 64).max(1);
    let target_tps = env_f64("SCIAGENT_DECODE_TARGET_TPS", 250.0);
    let stretch_tps = env_f64("SCIAGENT_DECODE_STRETCH_TPS", 750.0);
    let memory_gbps = env_optional_f64("SCIAGENT_DECODE_MEMORY_GBPS");
    let require_target = env_bool("SCIAGENT_DECODE_REQUIRE_TARGET");
    assert!(prompt_len + max_new <= config.max_seq_len);

    if let Some(memory_gbps) = memory_gbps
    {
        print_roofline(
            config.total_parameters(),
            memory_gbps,
            target_tps,
            stretch_tps,
        );
    }
    else
    {
        println!(
            "SCIAGENT_I250_ROOFLINE provenance=unknown status=omitted hint=SCIAGENT_DECODE_MEMORY_GBPS"
        );
    }

    let model = SciAgentModel::new(&config);
    let Some(oracle) = CudaModel::from_model(&model)
    else
    {
        eprintln!("no CUDA Route-B runtime available");
        std::process::exit(2);
    };
    let Some(fast) = CudaDecodeModel::from_model(&model)
    else
    {
        eprintln!("no I250 CUDA decode runtime available");
        std::process::exit(2);
    };

    let prompt = synthetic_tokens(prompt_len, config.vocab_size);
    let greedy = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
        repetition_penalty: 1.0,
        repetition_window: 64,
    };
    let seed = 0x4932_3530_5448_4F52u64;

    let fastest = CudaDecodeModes::default();
    let ffn_baseline = CudaDecodeModes {
        ffn: CudaDecodeFfnMode::CublasLt,
        down: CudaDecodeDownMode::CublasLt,
        lm_head: CudaDecodeLmHeadMode::FusedArgmax,
    };
    let lm_baseline = CudaDecodeModes {
        ffn: CudaDecodeFfnMode::FusedGemv,
        down: CudaDecodeDownMode::CublasLt,
        lm_head: CudaDecodeLmHeadMode::FullLogits,
    };
    let dense_i250 = CudaDecodeModes {
        ffn: CudaDecodeFfnMode::CublasLt,
        down: CudaDecodeDownMode::CublasLt,
        lm_head: CudaDecodeLmHeadMode::FullLogits,
    };
    let down_candidate = CudaDecodeModes {
        ffn: CudaDecodeFfnMode::FusedGemv,
        down: CudaDecodeDownMode::TiledGemv,
        lm_head: CudaDecodeLmHeadMode::FusedArgmax,
    };

    // Warm all implementation modes outside measurement windows.
    let _ = fast.generate_greedy_device_feedback_with_modes(&prompt, 1, fastest);
    let _ = fast.generate_greedy_device_feedback_with_modes(&prompt, 1, ffn_baseline);
    let _ = fast.generate_greedy_device_feedback_with_modes(&prompt, 1, lm_baseline);
    let _ = fast.generate_greedy_device_feedback_with_modes(&prompt, 1, dense_i250);
    let _ = fast.generate_greedy_device_feedback_with_modes(&prompt, 1, down_candidate);
    let _ = oracle.generate_cached(&prompt, 1, &greedy, seed);

    let (fast_tokens, fast_seconds, fast_tps) = timed_generate(&fast, &prompt, max_new, fastest);
    let (ffn_tokens, ffn_seconds, ffn_tps) = timed_generate(&fast, &prompt, max_new, ffn_baseline);
    let (lm_tokens, lm_seconds, lm_tps) = timed_generate(&fast, &prompt, max_new, lm_baseline);
    let (dense_tokens, dense_seconds, dense_tps) =
        timed_generate(&fast, &prompt, max_new, dense_i250);
    let (down_tokens, down_seconds, down_tps) =
        timed_generate(&fast, &prompt, max_new, down_candidate);

    let oracle_started = Instant::now();
    let oracle_tokens = oracle.generate_cached(&prompt, max_new, &greedy, seed);
    let oracle_seconds = oracle_started.elapsed().as_secs_f64();
    let oracle_new = oracle_tokens.len().saturating_sub(prompt.len());
    let oracle_tps = throughput(oracle_new, oracle_seconds);

    let parity = fast_tokens == oracle_tokens;
    let ffn_parity = ffn_tokens == oracle_tokens;
    let lm_parity = lm_tokens == oracle_tokens;
    let dense_parity = dense_tokens == oracle_tokens;
    let down_parity = down_tokens == oracle_tokens;
    let ffn_gain = fast_tps / ffn_tps.max(1e-9);
    let lm_gain = fast_tps / lm_tps.max(1e-9);
    let stack_gain = fast_tps / dense_tps.max(1e-9);
    let down_gain = down_tps / fast_tps.max(1e-9);
    let speedup = fast_tps / oracle_tps.max(1e-9);
    let target_met = fast_tps >= target_tps;
    let stretch_met = fast_tps >= stretch_tps;

    println!(
        "SCIAGENT_I250_DECODE params={} prompt={} requested_new={} fast_mode=ffn_fused_gemv+lm_fused_argmax fast_seconds={:.6} fast_tok_s={:.3} ffn_baseline=cublaslt+lm_fused_argmax ffn_seconds={:.6} ffn_tok_s={:.3} ffn_gain={:.3} lm_baseline=ffn_fused_gemv+full_logits lm_seconds={:.6} lm_tok_s={:.3} lm_gain={:.3} dense_i250=cublaslt+full_logits dense_seconds={:.6} dense_tok_s={:.3} stack_gain={:.3} b49_seconds={:.6} b49_tok_s={:.3} speedup={:.3} generated_h2d_bytes_per_token=0 generated_d2h_bytes_per_token=0 final_readback_bytes={} target_tok_s={:.3} target_met={} stretch_tok_s={:.3} stretch_met={} parity={} ffn_parity={} lm_parity={} dense_parity={}",
        config.total_parameters(),
        prompt_len,
        max_new,
        fast_seconds,
        fast_tps,
        ffn_seconds,
        ffn_tps,
        ffn_gain,
        lm_seconds,
        lm_tps,
        lm_gain,
        dense_seconds,
        dense_tps,
        stack_gain,
        oracle_seconds,
        oracle_tps,
        speedup,
        max_new * core::mem::size_of::<u32>(),
        target_tps,
        target_met,
        stretch_tps,
        stretch_met,
        parity,
        ffn_parity,
        lm_parity,
        dense_parity,
    );

    println!(
        "SCIAGENT_I250_DOWN baseline_tok_s={:.3} tiled_tok_s={:.3} gain={:.3} seconds={:.6} parity={}",
        fast_tps, down_tps, down_gain, down_seconds, down_parity
    );

    if !parity || !ffn_parity || !lm_parity || !dense_parity || !down_parity
    {
        eprintln!("ERROR: an I250 A/B mode diverged from the B49 cached oracle");
        std::process::exit(3);
    }
    if require_target && !target_met
    {
        eprintln!(
            "ERROR: fastest I250 CUDA decode {:.3} tok/s is below required {:.3} tok/s",
            fast_tps, target_tps
        );
        std::process::exit(4);
    }
}
