//! Phase 28 diagnostic: isolate first-token parity before device feedback.

use scirust_core::nn::sampling::SamplingConfig;
use scirust_core::nn::transformer::mini_llm::{CharTokenizer, MiniLLM, MiniLLMConfig};
use scirust_gpu::{WgpuLatentHeadBasis, WgpuLatentLayerBasis, WgpuResidentSampledMiniLlm};

const VOCAB_SIZE: usize = 4096;
const PROMPT_TOKENS: usize = 16;
const MAX_SEQ_LEN: usize = 48;
const D_MODEL: usize = 64;
const N_HEADS: usize = 4;
const N_LAYERS: usize = 2;
const D_FF: usize = 128;
const SEED: u64 = 0x28_00_00_01;

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
        top_k: 0,
        top_p: 0.92,
    };

    let mut cpu = MiniLLM::new(config.clone(), tokenizer.clone());
    let gpu = MiniLLM::new(config.clone(), tokenizer);
    let expected = cpu.generate_ids_cached_sampled(&prompt, 1, &sampling, SEED);
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
    let mut resident = WgpuResidentSampledMiniLlm::new(
        gpu.inference_snapshot(),
        config.max_seq_len,
        d_head,
        &layers,
        sampling,
        SEED,
    )?;

    for (position, &token) in prompt.iter().enumerate()
    {
        resident.ingest_at(token, position)?;
    }
    let actual_first = resident.sample_next()?;

    println!(
        "phase28_first_token_probe,top_k=0,cpu_first={expected_first},wgpu_host_stepped_first={actual_first},match={},sampling_draws={},sample_ready={}",
        u8::from(actual_first == expected_first),
        resident.telemetry().sampling_draws,
        resident.telemetry().sample_ready,
    );

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
        let character = char::from_u32(scalar)
            .ok_or("synthetic vocabulary exceeds Unicode scalar range")?;
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
