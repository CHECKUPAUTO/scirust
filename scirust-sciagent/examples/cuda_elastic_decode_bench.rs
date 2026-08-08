//! I250-B reconstruction-free ElasticKV batch-one CUDA benchmark.
//!
//! `SCIAGENT_ELASTIC_KEY_RANK=0` selects the full-rank identity oracle. A positive
//! even rank selects the native complete-RoPE-pair prefix. Reduced-rank output is
//! intentionally not labelled exact dense parity; it remains quality-gated.

use std::time::Instant;

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::cuda_elastic_decode::CudaElasticDecodeModel;
use scirust_sciagent::cuda_model::CudaModel;
use scirust_sciagent::generate::SamplingParams;
use scirust_sciagent::model::SciAgentModel;

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

fn main() {
    let config = SciAgentConfig::sciagent_350m();
    let prompt_len = env_usize("SCIAGENT_DECODE_PROMPT", 128).max(1);
    let max_new = env_usize("SCIAGENT_DECODE_NEW", 64).max(1);
    let key_rank = env_usize("SCIAGENT_ELASTIC_KEY_RANK", 0);
    let value_rank = env_usize(
        "SCIAGENT_ELASTIC_VALUE_RANK",
        if key_rank == 0 { 0 } else { key_rank },
    );
    let target_tps = env_f64("SCIAGENT_DECODE_TARGET_TPS", 250.0);
    let require_target = env_bool("SCIAGENT_DECODE_REQUIRE_TARGET");
    assert!(prompt_len + max_new <= config.max_seq_len);

    let model = SciAgentModel::new(&config);
    let Some(oracle) = CudaModel::from_model(&model)
    else
    {
        eprintln!("no CUDA Route-B runtime available");
        std::process::exit(2);
    };
    let elastic = if key_rank == 0
    {
        CudaElasticDecodeModel::from_model_identity(&model)
    }
    else
    {
        CudaElasticDecodeModel::from_model_native_pair_prefix(&model, key_rank, value_rank)
    }
    .unwrap_or_else(|error| {
        eprintln!("Elastic CUDA decode construction failed: {error}");
        std::process::exit(2);
    });

    let prompt = synthetic_tokens(prompt_len, config.vocab_size);
    let params = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
        repetition_penalty: 1.0,
        repetition_window: 64,
    };
    let seed = 0x454C_4153_5449_4332u64;

    let _ = elastic.generate(&prompt, 1, &params, seed);
    let _ = oracle.generate_cached(&prompt, 1, &params, seed);

    let started = Instant::now();
    let elastic_tokens = elastic.generate(&prompt, max_new, &params, seed);
    let elastic_seconds = started.elapsed().as_secs_f64().max(1e-9);
    let elastic_new = elastic_tokens.len().saturating_sub(prompt.len());
    let elastic_tps = elastic_new as f64 / elastic_seconds;

    let oracle_started = Instant::now();
    let oracle_tokens = oracle.generate_cached(&prompt, max_new, &params, seed);
    let oracle_seconds = oracle_started.elapsed().as_secs_f64().max(1e-9);
    let oracle_new = oracle_tokens.len().saturating_sub(prompt.len());
    let oracle_tps = oracle_new as f64 / oracle_seconds;

    let stream_match = elastic_tokens == oracle_tokens;
    let exact_dense_mode = elastic.is_dense_equivalent_identity();
    let speedup = if oracle_tps > 0.0
    {
        elastic_tps / oracle_tps
    }
    else
    {
        f64::NAN
    };
    let target_met = elastic_tps >= target_tps;

    println!(
        "SCIAGENT_I250_ELASTIC params={} prompt={} requested_new={} key_rank={} value_rank={} exact_dense_mode={} elastic_new={} elastic_seconds={:.6} elastic_tok_s={:.3} b49_new={} b49_seconds={:.6} b49_tok_s={:.3} speedup={:.3} target_tok_s={:.3} target_met={} b49_stream_match={}",
        config.total_parameters(),
        prompt_len,
        max_new,
        elastic.key_rank(),
        elastic.value_rank(),
        exact_dense_mode,
        elastic_new,
        elastic_seconds,
        elastic_tps,
        oracle_new,
        oracle_seconds,
        oracle_tps,
        speedup,
        target_tps,
        target_met,
        stream_match
    );

    if exact_dense_mode && !stream_match
    {
        eprintln!("ERROR: full-rank Elastic CUDA decode diverged from B49");
        std::process::exit(3);
    }
    if require_target && !target_met
    {
        eprintln!(
            "ERROR: Elastic CUDA decode {:.3} tok/s is below required {:.3} tok/s",
            elastic_tps, target_tps
        );
        std::process::exit(4);
    }
}
