//! Phase 28 end-to-end baseline for resident WGPU device-feedback generation.
//!
//! This benchmark times the public `WgpuResidentDeviceFeedbackMiniLlm`
//! generation boundary. Model/runtime construction and the CPU oracle are kept
//! outside the timed region. Prompt-only and prompt+decode medians are reported
//! separately; their difference is only an incremental decode proxy, not a
//! direct kernel-time measurement. No performance threshold is embedded here.

use std::env;
use std::time::Instant;

use scirust_core::nn::sampling::SamplingConfig;
use scirust_core::nn::transformer::mini_llm::{CharTokenizer, MiniLLM, MiniLLMConfig};
use scirust_gpu::{WgpuLatentHeadBasis, WgpuLatentLayerBasis, WgpuResidentDeviceFeedbackMiniLlm};

const DEFAULT_VOCAB: usize = 4096;
const DEFAULT_PROMPT_TOKENS: usize = 16;
const DEFAULT_DECODE_TOKENS: &str = "8,32";
const DEFAULT_TOP_KS: &str = "50";
const DEFAULT_REPEATS: usize = 7;
const DEFAULT_WARMUP: usize = 3;
const DEFAULT_SEED: u64 = 0x28_00_00_01;
const D_MODEL: usize = 64;
const N_HEADS: usize = 4;
const N_LAYERS: usize = 2;
const D_FF: usize = 128;
const TEMPERATURE: f32 = 0.9;
const TOP_P: f32 = 0.92;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vocab = parse_positive("SCIRUST_RESIDENT_BENCH_VOCAB", DEFAULT_VOCAB)?;
    let prompt_tokens = parse_positive(
        "SCIRUST_RESIDENT_BENCH_PROMPT_TOKENS",
        DEFAULT_PROMPT_TOKENS,
    )?;
    let decode_tokens = parse_positive_list(
        "SCIRUST_RESIDENT_BENCH_DECODE_TOKENS",
        DEFAULT_DECODE_TOKENS,
    )?;
    let top_ks = parse_list("SCIRUST_RESIDENT_BENCH_TOP_KS", DEFAULT_TOP_KS)?;
    let repeats = parse_positive("SCIRUST_RESIDENT_BENCH_REPEATS", DEFAULT_REPEATS)?;
    let warmup = parse_non_negative("SCIRUST_RESIDENT_BENCH_WARMUP", DEFAULT_WARMUP)?;
    let seed = parse_u64("SCIRUST_RESIDENT_BENCH_SEED", DEFAULT_SEED)?;

    if vocab < 4
    {
        return Err("SCIRUST_RESIDENT_BENCH_VOCAB must be at least 4".into());
    }
    for &top_k in &top_ks
    {
        if top_k >= vocab && top_k != 0
        {
            return Err(format!("top_k={top_k} must be zero or smaller than vocab={vocab}").into());
        }
    }

    let max_decode = *decode_tokens
        .iter()
        .max()
        .ok_or("decode-token list must not be empty")?;
    let max_seq_len = prompt_tokens
        .checked_add(max_decode)
        .ok_or("Phase 28 max_seq_len overflows usize")?;
    let tokenizer = synthetic_tokenizer(vocab)?;
    let config = MiniLLMConfig {
        vocab_size: vocab,
        d_model: D_MODEL,
        n_heads: N_HEADS,
        n_layers: N_LAYERS,
        d_ff: D_FF,
        max_seq_len,
    };
    let prompt = deterministic_prompt(vocab, prompt_tokens);

    println!(
        "vocab_size,d_model,n_heads,n_layers,d_ff,prompt_tokens,requested_decode_tokens,generated_tokens,top_k,temperature,top_p,seed,repeats,warmup,prompt_median_ns,end_to_end_median_ns,incremental_decode_proxy_ns,generated_tokens_per_second_end_to_end,exact_cpu_wgpu,deterministic_replay,output_fingerprint,resident_bytes,prompt_upload_bytes_per_token,generated_upload_bytes_per_token,generated_download_bytes_per_token,last_burst_readback_bytes,sampling_draws"
    );

    for &top_k in &top_ks
    {
        for &requested_decode in &decode_tokens
        {
            emit_case(
                &tokenizer,
                &config,
                &prompt,
                requested_decode,
                top_k,
                repeats,
                warmup,
                seed,
            )?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn emit_case(
    tokenizer: &CharTokenizer,
    config: &MiniLLMConfig,
    prompt: &[usize],
    requested_decode: usize,
    top_k: usize,
    repeats: usize,
    warmup: usize,
    seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let sampling = SamplingConfig {
        temperature: TEMPERATURE,
        top_k,
        top_p: TOP_P,
    };
    let mut cpu = MiniLLM::new(config.clone(), tokenizer.clone());
    let gpu = MiniLLM::new(config.clone(), tokenizer.clone());
    let expected = cpu.generate_ids_cached_sampled(prompt, requested_decode, &sampling, seed);
    let generated_tokens = expected.len().saturating_sub(prompt.len());
    if generated_tokens == 0
    {
        return Err("Phase 28 expected at least one generated token".into());
    }

    let d_head = config.d_model / config.n_heads;
    let basis = identity_basis(d_head);
    let heads = vec![
        WgpuLatentHeadBasis {
            key: &basis,
            value: &basis,
        };
        config.n_heads
    ];
    let layers = vec![WgpuLatentLayerBasis { heads: &heads }; config.n_layers];
    let mut resident = WgpuResidentDeviceFeedbackMiniLlm::new(
        gpu.inference_snapshot(),
        config.max_seq_len,
        d_head,
        &layers,
        sampling,
        seed,
    )?;

    for _ in 0..warmup
    {
        let prompt_only = resident.generate_ids_resident(prompt, 0)?;
        if prompt_only != prompt
        {
            return Err("prompt-only resident generation changed the prompt".into());
        }
        let full = resident.generate_ids_resident(prompt, requested_decode)?;
        if full != expected
        {
            return Err(sequence_mismatch(
                "warmup",
                top_k,
                requested_decode,
                prompt.len(),
                &expected,
                &full,
            )
            .into());
        }
        std::hint::black_box(full);
    }

    let mut prompt_elapsed = Vec::with_capacity(repeats);
    let mut full_elapsed = Vec::with_capacity(repeats);
    let mut replay = true;
    for _ in 0..repeats
    {
        let start = Instant::now();
        let prompt_only = resident.generate_ids_resident(prompt, 0)?;
        prompt_elapsed.push(start.elapsed().as_nanos());
        if prompt_only != prompt
        {
            return Err("measured prompt-only generation changed the prompt".into());
        }

        let start = Instant::now();
        let full = resident.generate_ids_resident(prompt, requested_decode)?;
        full_elapsed.push(start.elapsed().as_nanos());
        if full != expected
        {
            return Err(sequence_mismatch(
                "measured",
                top_k,
                requested_decode,
                prompt.len(),
                &expected,
                &full,
            )
            .into());
        }
        replay &= full == expected;
        std::hint::black_box(full);
    }

    let prompt_median_ns = median(&mut prompt_elapsed);
    let end_to_end_median_ns = median(&mut full_elapsed);
    let incremental_decode_proxy_ns = end_to_end_median_ns as i128 - prompt_median_ns as i128;
    let generated_tokens_per_second_end_to_end =
        generated_tokens as f64 * 1_000_000_000.0 / end_to_end_median_ns.max(1) as f64;
    let telemetry = resident.telemetry();
    let exact_cpu_wgpu = replay;

    if telemetry.prompt_upload_bytes_per_token != core::mem::size_of::<u32>()
    {
        return Err("unexpected Phase 28 prompt upload size".into());
    }
    if telemetry.generated_upload_bytes_per_token != 0
        || telemetry.generated_download_bytes_per_token != 0
    {
        return Err("resident generation performed a per-token generated transfer".into());
    }
    let expected_readback = (4usize + requested_decode)
        .checked_mul(core::mem::size_of::<u32>())
        .ok_or("Phase 28 expected readback size overflows usize")?;
    if telemetry.last_burst_readback_bytes != expected_readback
    {
        return Err(format!(
            "unexpected final readback: got {}, expected {expected_readback}",
            telemetry.last_burst_readback_bytes
        )
        .into());
    }
    if telemetry.sampling_draws != generated_tokens
    {
        return Err(format!(
            "sampling draws {} differ from generated tokens {generated_tokens}",
            telemetry.sampling_draws
        )
        .into());
    }

    println!(
        "{},{},{},{},{},{},{},{},{},{:.6},{:.6},{},{},{},{},{},{},{:.6},{},{},{:016x},{},{},{},{},{},{}",
        config.vocab_size,
        config.d_model,
        config.n_heads,
        config.n_layers,
        config.d_ff,
        prompt.len(),
        requested_decode,
        generated_tokens,
        top_k,
        TEMPERATURE,
        TOP_P,
        seed,
        repeats,
        warmup,
        prompt_median_ns,
        end_to_end_median_ns,
        incremental_decode_proxy_ns,
        generated_tokens_per_second_end_to_end,
        u8::from(exact_cpu_wgpu),
        u8::from(replay),
        fingerprint(&expected),
        telemetry.resident_bytes,
        telemetry.prompt_upload_bytes_per_token,
        telemetry.generated_upload_bytes_per_token,
        telemetry.generated_download_bytes_per_token,
        telemetry.last_burst_readback_bytes,
        telemetry.sampling_draws,
    );

    Ok(())
}

fn sequence_mismatch(
    stage: &str,
    top_k: usize,
    requested_decode: usize,
    prompt_len: usize,
    expected: &[usize],
    actual: &[usize],
) -> String {
    let first_index = expected
        .iter()
        .zip(actual)
        .position(|(expected_token, actual_token)| expected_token != actual_token)
        .unwrap_or(expected.len().min(actual.len()));
    let expected_token = expected
        .get(first_index)
        .map_or_else(|| "end".to_owned(), usize::to_string);
    let actual_token = actual
        .get(first_index)
        .map_or_else(|| "end".to_owned(), usize::to_string);
    format!(
        "Phase 28 {stage} CPU/WGPU sequence mismatch: top_k={top_k}, requested_decode={requested_decode}, first_index={first_index}, generated_offset={}, expected_token={expected_token}, actual_token={actual_token}, expected_len={}, actual_len={}, expected_fingerprint={:016x}, actual_fingerprint={:016x}",
        first_index.saturating_sub(prompt_len),
        expected.len(),
        actual.len(),
        fingerprint(expected),
        fingerprint(actual),
    )
}

fn synthetic_tokenizer(vocab_size: usize) -> Result<CharTokenizer, Box<dyn std::error::Error>> {
    let needed = vocab_size
        .checked_sub(2)
        .ok_or("vocab_size must reserve EOS and unknown ids")?;
    let mut corpus = String::new();
    let mut inserted = 0usize;
    let mut scalar = 0x1000u32;
    while inserted < needed
    {
        let character = char::from_u32(scalar)
            .ok_or("requested synthetic vocabulary exceeds Unicode scalar range")?;
        scalar = scalar
            .checked_add(1)
            .ok_or("synthetic vocabulary scalar counter overflowed")?;
        if character == '\0' || character == '�'
        {
            continue;
        }
        corpus.push(character);
        inserted += 1;
    }

    let tokenizer = CharTokenizer::new(&[corpus.as_str()]);
    if tokenizer.vocab_size != vocab_size
    {
        return Err(format!(
            "synthetic tokenizer built vocab {}, expected {vocab_size}",
            tokenizer.vocab_size
        )
        .into());
    }
    Ok(tokenizer)
}

fn deterministic_prompt(vocab_size: usize, tokens: usize) -> Vec<usize> {
    (0..tokens)
        .map(|index| 2 + (index.wrapping_mul(17).wrapping_add(11) % (vocab_size - 2)))
        .collect()
}

fn identity_basis(dimension: usize) -> Vec<f32> {
    let mut basis = vec![0.0; dimension * dimension];
    for index in 0..dimension
    {
        basis[index * dimension + index] = 1.0;
    }
    basis
}

fn median(values: &mut [u128]) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn fingerprint(tokens: &[usize]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &token in tokens
    {
        for byte in (token as u64).to_le_bytes()
        {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

fn parse_list(name: &str, default: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let raw = env::var(name).unwrap_or_else(|_| default.to_owned());
    let values = raw
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty()
    {
        return Err(format!("{name} must contain comma-separated integers").into());
    }
    Ok(values)
}

fn parse_positive_list(
    name: &str,
    default: &str,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let values = parse_list(name, default)?;
    if values.contains(&0)
    {
        return Err(format!("{name} must contain positive integers").into());
    }
    Ok(values)
}

fn parse_positive(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    let value = env::var(name)
        .ok()
        .map(|raw| raw.parse::<usize>())
        .transpose()?
        .unwrap_or(default);
    if value == 0
    {
        return Err(format!("{name} must be positive").into());
    }
    Ok(value)
}

fn parse_non_negative(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(env::var(name)
        .ok()
        .map(|raw| raw.parse::<usize>())
        .transpose()?
        .unwrap_or(default))
}

fn parse_u64(name: &str, default: u64) -> Result<u64, Box<dyn std::error::Error>> {
    Ok(env::var(name)
        .ok()
        .map(|raw| raw.parse::<u64>())
        .transpose()?
        .unwrap_or(default))
}
