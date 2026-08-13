//! Full-logit diagnostic for the first four I250 generated tokens.
//!
//! Every path uses the same model, prompt, sampler and CUDA device. Downloads are
//! intentional; this example is a correctness probe and never a performance claim.

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::cuda_decode::{
    CudaDecodeDownMode, CudaDecodeFfnMode, CudaDecodeLmHeadMode, CudaDecodeModel, CudaDecodeModes,
};
use scirust_sciagent::cuda_model::CudaModel;
use scirust_sciagent::generate::SamplingParams;
use scirust_sciagent::model::SciAgentModel;

#[derive(Debug)]
struct DiffStats {
    different: usize,
    max_abs: f32,
    mean_abs: f64,
    rms: f64,
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
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

fn diff_stats(left: &[f32], right: &[f32]) -> DiffStats {
    assert_eq!(left.len(), right.len());
    let mut different = 0usize;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    for (&left, &right) in left.iter().zip(right)
    {
        let abs = (left - right).abs();
        if left.to_bits() != right.to_bits()
        {
            different += 1;
        }
        max_abs = max_abs.max(abs);
        sum_abs += f64::from(abs);
        sum_sq += f64::from(abs) * f64::from(abs);
    }
    let count = left.len() as f64;
    DiffStats {
        different,
        max_abs,
        mean_abs: sum_abs / count,
        rms: (sum_sq / count).sqrt(),
    }
}

fn top(logits: &[f32], count: usize) -> Vec<(usize, f32)> {
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&left, &right| {
        logits[right]
            .total_cmp(&logits[left])
            .then_with(|| left.cmp(&right))
    });
    indices
        .into_iter()
        .take(count)
        .map(|index| (index, logits[index]))
        .collect()
}

fn generated_token(tokens: &[u32], prompt_len: usize, step: usize) -> u32 {
    tokens[prompt_len + step]
}

fn main() {
    let config = SciAgentConfig::sciagent_350m();
    let prompt_len = env_usize("SCIAGENT_DECODE_PROMPT", 128).max(1);
    let trace_tokens = env_usize("SCIAGENT_DECODE_TRACE_TOKENS", 4).max(1);
    assert!(prompt_len + trace_tokens <= config.max_seq_len);

    let model = SciAgentModel::new(&config);
    let Some(route_b) = CudaModel::from_model(&model)
    else
    {
        eprintln!("no CUDA Route-B runtime available");
        std::process::exit(2);
    };
    let Some(i250) = CudaDecodeModel::from_model(&model)
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
    let fused = CudaDecodeModes {
        ffn: CudaDecodeFfnMode::FusedGemv,
        down: CudaDecodeDownMode::CublasLt,
        lm_head: CudaDecodeLmHeadMode::FullLogits,
    };
    let cublaslt = CudaDecodeModes {
        ffn: CudaDecodeFfnMode::CublasLt,
        down: CudaDecodeDownMode::CublasLt,
        lm_head: CudaDecodeLmHeadMode::FullLogits,
    };

    let (route_tokens, route_logits) =
        route_b.generate_cached_trace(&prompt, trace_tokens, &greedy, seed);
    let (fused_tokens, fused_logits) =
        i250.generate_trace_with_modes(&prompt, trace_tokens, &greedy, seed, fused);
    let (cublas_tokens, cublas_logits) =
        i250.generate_trace_with_modes(&prompt, trace_tokens, &greedy, seed, cublaslt);

    let available = route_logits
        .len()
        .min(fused_logits.len())
        .min(cublas_logits.len());
    for step in 0..available
    {
        let fused_diff = diff_stats(&fused_logits[step], &route_logits[step]);
        let cublas_diff = diff_stats(&cublas_logits[step], &route_logits[step]);
        println!(
            "SCIAGENT_I250_TOKEN_TRACE step={} route_token={} fused_token={} cublaslt_token={} fused_different={} fused_max_abs={:.9e} fused_mean_abs={:.9e} fused_rms={:.9e} cublaslt_different={} cublaslt_max_abs={:.9e} cublaslt_mean_abs={:.9e} cublaslt_rms={:.9e} route_top={:?} fused_top={:?} cublaslt_top={:?}",
            step + 1,
            generated_token(&route_tokens, prompt_len, step),
            generated_token(&fused_tokens, prompt_len, step),
            generated_token(&cublas_tokens, prompt_len, step),
            fused_diff.different,
            fused_diff.max_abs,
            fused_diff.mean_abs,
            fused_diff.rms,
            cublas_diff.different,
            cublas_diff.max_abs,
            cublas_diff.mean_abs,
            cublas_diff.rms,
            top(&route_logits[step], 4),
            top(&fused_logits[step], 4),
            top(&cublas_logits[step], 4),
        );
    }

    if available < trace_tokens
    {
        eprintln!(
            "ERROR: trace ended after {available} tokens before requested {trace_tokens} tokens"
        );
        std::process::exit(3);
    }
}
