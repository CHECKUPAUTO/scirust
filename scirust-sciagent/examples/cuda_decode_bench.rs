//! I250 batch-one CUDA decode benchmark.
//!
//! The primary metric is now greedy device-feedback generation: argmax, token
//! feedback and generated-token accumulation stay on CUDA and the host performs one
//! compact `u32[max_new]` readback after the burst. The historical I250 host-sampler
//! path and B49 cached path are measured as diagnostics/oracles.

use std::time::Instant;

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::cuda_decode::{CudaDecodeFfnMode, CudaDecodeModel};
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
    let Some(fast) =
        CudaDecodeModel::from_model_with_ffn_mode(&model, CudaDecodeFfnMode::FusedGemv)
    else
    {
        eprintln!("no fused-GEMV CUDA decode runtime available");
        std::process::exit(2);
    };
    let Some(cublas) =
        CudaDecodeModel::from_model_with_ffn_mode(&model, CudaDecodeFfnMode::CublasLt)
    else
    {
        eprintln!("no cuBLASLt I250 decode baseline available");
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

    // Warm NVRTC/cuBLASLt and all measured paths outside the timing windows.
    let _ = fast.generate_greedy_device_feedback(&prompt, 1);
    let _ = cublas.generate_greedy_device_feedback(&prompt, 1);
    let _ = oracle.generate_cached(&prompt, 1, &greedy, seed);

    let device_started = Instant::now();
    let device_tokens = fast.generate_greedy_device_feedback(&prompt, max_new);
    let device_seconds = device_started.elapsed().as_secs_f64();
    let device_new = device_tokens.len().saturating_sub(prompt.len());
    let device_tps = throughput(device_new, device_seconds);

    let cublas_started = Instant::now();
    let cublas_tokens = cublas.generate_greedy_device_feedback(&prompt, max_new);
    let cublas_seconds = cublas_started.elapsed().as_secs_f64();
    let cublas_new = cublas_tokens.len().saturating_sub(prompt.len());
    let cublas_tps = throughput(cublas_new, cublas_seconds);

    let oracle_started = Instant::now();
    let oracle_tokens = oracle.generate_cached(&prompt, max_new, &greedy, seed);
    let oracle_seconds = oracle_started.elapsed().as_secs_f64();
    let oracle_new = oracle_tokens.len().saturating_sub(prompt.len());
    let oracle_tps = throughput(oracle_new, oracle_seconds);

    let device_parity = device_tokens == oracle_tokens;
    let cublas_parity = cublas_tokens == oracle_tokens;
    let speedup = if oracle_tps > 0.0
    {
        device_tps / oracle_tps
    }
    else
    {
        f64::NAN
    };
    let ffn_gain = if cublas_tps > 0.0
    {
        device_tps / cublas_tps
    }
    else
    {
        f64::NAN
    };
    let target_met = device_tps >= target_tps;
    let stretch_met = device_tps >= stretch_tps;

    println!(
        "SCIAGENT_I250_DECODE params={} prompt={} requested_new={} fast_mode=device_feedback_greedy ffn_fast=fused_gemv fast_new={} fast_seconds={:.6} fast_tok_s={:.3} ffn_baseline=cublaslt cublas_new={} cublas_seconds={:.6} cublas_tok_s={:.3} ffn_gain={:.3} b49_new={} b49_seconds={:.6} b49_tok_s={:.3} speedup={:.3} generated_h2d_bytes_per_token=0 generated_d2h_bytes_per_token=0 final_readback_bytes={} target_tok_s={:.3} target_met={} stretch_tok_s={:.3} stretch_met={} parity={} cublas_parity={}",
        config.total_parameters(),
        prompt_len,
        max_new,
        device_new,
        device_seconds,
        device_tps,
        cublas_new,
        cublas_seconds,
        cublas_tps,
        ffn_gain,
        oracle_new,
        oracle_seconds,
        oracle_tps,
        speedup,
        max_new * core::mem::size_of::<u32>(),
        target_tps,
        target_met,
        stretch_tps,
        stretch_met,
        device_parity,
        cublas_parity,
    );

    if !device_parity || !cublas_parity
    {
        eprintln!("ERROR: I250 CUDA decode diverged from the B49 cached oracle");
        std::process::exit(3);
    }
    if require_target && !target_met
    {
        eprintln!(
            "ERROR: device-feedback CUDA decode {:.3} tok/s is below required {:.3} tok/s",
            device_tps, target_tps
        );
        std::process::exit(4);
    }
}
