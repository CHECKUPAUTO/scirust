//! Phase 28 diagnostic: isolate one device-feedback burst in a fresh WGPU process.

use std::env;

use scirust_core::nn::sampling::SamplingConfig;
use scirust_core::nn::transformer::mini_llm::{CharTokenizer, MiniLLM, MiniLLMConfig};
use scirust_gpu::{
    WgpuLatentHeadBasis, WgpuLatentLayerBasis, WgpuResidentDeviceFeedbackMiniLlm,
    WgpuResidentSampledMiniLlm,
};

const VOCAB_SIZE: usize = 4096;
const PROMPT_TOKENS: usize = 16;
const MAX_SEQ_LEN: usize = 64;
const D_MODEL: usize = 64;
const N_HEADS: usize = 4;
const N_LAYERS: usize = 2;
const D_FF: usize = 128;
const SEED: u64 = 0x28_00_00_01;
const DEFAULT_LIMIT: usize = 1;
const DEFAULT_TOP_K: usize = 0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let limit = env::var("SCIRUST_RESIDENT_PROBE_LIMIT")
        .ok()
        .map(|raw| raw.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_LIMIT);
    let top_k = env::var("SCIRUST_RESIDENT_PROBE_TOP_K")
        .ok()
        .map(|raw| raw.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_TOP_K);
    if limit == 0 || limit > MAX_SEQ_LEN - PROMPT_TOKENS
    {
        return Err("SCIRUST_RESIDENT_PROBE_LIMIT is outside the probe capacity".into());
    }
    if top_k >= VOCAB_SIZE && top_k != 0
    {
        return Err("SCIRUST_RESIDENT_PROBE_TOP_K must be zero or smaller than vocab".into());
    }

    let tokenizer = synthetic_tokenizer(VOCAB_SIZE)?;
    let config = MiniLLMConfig {
        vocab_size: VOCAB_SIZE,
        d_model: D_MODEL,
        n_heads: N_HEADS,
        n_layers: N_LAYERS,
        d_ff: D_FF,
        max_seq_len: MAX_SEQ_LEN,
    };
    let prompt = deterministic_prompt(VOCAB_SIZE, PROMPT_TOKENS);
    let sampling = SamplingConfig {
        temperature: 0.9,
        top_k,
        top_p: 0.92,
    };

    let mut cpu = MiniLLM::new(config.clone(), tokenizer.clone());
    let host_gpu = MiniLLM::new(config.clone(), tokenizer.clone());
    let feedback_gpu = MiniLLM::new(config.clone(), tokenizer.clone());
    let expected = cpu.generate_ids_cached_sampled(&prompt, limit, &sampling, SEED);
    let expected_first = *expected
        .get(prompt.len())
        .ok_or("CPU oracle did not generate a first token")?;

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

    let mut host_resident = WgpuResidentSampledMiniLlm::new(
        host_gpu.inference_snapshot(),
        config.max_seq_len,
        d_head,
        &layers,
        sampling,
        SEED,
    )?;
    for (position, &token) in prompt.iter().enumerate()
    {
        host_resident.ingest_at(token, position)?;
    }
    let host_first = host_resident.sample_next()?;
    if host_first != expected_first
    {
        return Err(format!(
            "host-stepped first-token mismatch: top_k={top_k}, cpu={expected_first}, wgpu={host_first}"
        )
        .into());
    }
    println!(
        "phase28_burst_probe,mode=host-stepped,limit={limit},top_k={top_k},cpu_first={expected_first},wgpu_first={host_first},match=1,sampling_draws={}",
        host_resident.telemetry().sampling_draws,
    );

    let mut feedback_resident = WgpuResidentDeviceFeedbackMiniLlm::new(
        feedback_gpu.inference_snapshot(),
        config.max_seq_len,
        d_head,
        &layers,
        sampling,
        SEED,
    )?;
    let ids = feedback_resident.generate_ids_resident(&prompt, limit)?;
    let first = ids
        .get(prompt.len())
        .map_or_else(|| "end".to_owned(), usize::to_string);
    let telemetry = feedback_resident.telemetry();
    let exact = ids == expected;
    println!(
        "phase28_burst_probe,mode=device-feedback,limit={limit},top_k={top_k},cpu_first={expected_first},wgpu_first={first},match={},expected_len={},actual_len={},sampling_draws={},readback_bytes={},expected_fingerprint={:016x},actual_fingerprint={:016x}",
        u8::from(exact),
        expected.len(),
        ids.len(),
        telemetry.sampling_draws,
        telemetry.last_burst_readback_bytes,
        fingerprint(&expected),
        fingerprint(&ids),
    );
    if !exact
    {
        return Err(format!(
            "device-feedback burst mismatch at top_k={top_k}, limit={limit}: expected_len={}, actual_len={}, expected_fingerprint={:016x}, actual_fingerprint={:016x}",
            expected.len(),
            ids.len(),
            fingerprint(&expected),
            fingerprint(&ids),
        )
        .into());
    }

    Ok(())
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
        let character =
            char::from_u32(scalar).ok_or("synthetic vocabulary exceeds Unicode scalar range")?;
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

fn fingerprint(tokens: &[usize]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &token in tokens
    {
        for byte in (token as u64).to_le_bytes()
        {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}
