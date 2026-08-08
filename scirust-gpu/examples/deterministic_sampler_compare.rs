//! Phase 25 real-adapter comparison of the exact sequential and parallel
//! bounded-top-k WGPU samplers.
//!
//! Both implementations are timed through their public `sample()` boundary,
//! so each measurement includes logits upload, dispatch, synchronization and
//! one sampled-token readback. Shader compilation/construction is outside the
//! timed region. No speedup threshold is embedded in this program.

use std::env;
use std::time::Instant;

use scirust_core::nn::sampling::SamplingConfig;
use scirust_gpu::{WgpuDeterministicSampler, WgpuParallelTopKSampler};

const DEFAULT_VOCABS: &str = "4096";
const DEFAULT_TOP_KS: &str = "5,50,200";
const DEFAULT_REPEATS: usize = 7;
const DEFAULT_WARMUP: usize = 3;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vocabs = parse_list("SCIRUST_SAMPLER_COMPARE_VOCABS", DEFAULT_VOCABS)?;
    let top_ks = parse_list("SCIRUST_SAMPLER_COMPARE_TOP_KS", DEFAULT_TOP_KS)?;
    let repeats = parse_positive("SCIRUST_SAMPLER_COMPARE_REPEATS", DEFAULT_REPEATS)?;
    let warmup = parse_non_negative("SCIRUST_SAMPLER_COMPARE_WARMUP", DEFAULT_WARMUP)?;

    println!(
        "vocab_size,top_k,repeats,sequential_median_ns,parallel_median_ns,speedup_parallel_vs_sequential,exact_stream_match,sequential_fingerprint,parallel_fingerprint,sequential_ranking_passes,parallel_ranking_passes,parallel_lanes"
    );

    for vocab in vocabs {
        let values = deterministic_logits(vocab);
        for &top_k in &top_ks {
            emit_case(vocab, top_k, repeats, warmup, &values)?;
        }
    }
    Ok(())
}

fn emit_case(
    vocab: usize,
    top_k: usize,
    repeats: usize,
    warmup: usize,
    values: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let config = SamplingConfig {
        temperature: 0.9,
        top_k,
        top_p: 0.92,
    };
    let seed = 0x25_00_00_00_u64 ^ ((vocab as u64) << 16) ^ top_k as u64;
    let mut sequential = WgpuDeterministicSampler::new(vocab, config, seed)?;
    let mut parallel = WgpuParallelTopKSampler::new(vocab, config, seed)?;

    for _ in 0..warmup {
        std::hint::black_box(sequential.sample(values)?);
        std::hint::black_box(parallel.sample(values)?);
    }
    sequential.reset()?;
    parallel.reset()?;

    let (seq_ns, seq_tokens) = measure(repeats, || sequential.sample(values))?;
    let (par_ns, par_tokens) = measure(repeats, || parallel.sample(values))?;
    let exact = seq_tokens == par_tokens;
    let speedup = seq_ns as f64 / par_ns.max(1) as f64;

    println!(
        "{vocab},{top_k},{repeats},{seq_ns},{par_ns},{speedup:.6},{},{:016x},{:016x},{},{},{}",
        u8::from(exact),
        fingerprint(&seq_tokens),
        fingerprint(&par_tokens),
        sequential.ranking_passes_per_sample(),
        parallel.ranking_passes_per_sample(),
        parallel.ranking_lanes_per_sample(),
    );
    Ok(())
}

fn measure<F>(
    repeats: usize,
    mut sample: F,
) -> Result<(u128, Vec<usize>), Box<dyn std::error::Error>>
where
    F: FnMut() -> Result<usize, scirust_gpu::WgpuDeterministicSamplerError>,
{
    let mut elapsed = Vec::with_capacity(repeats);
    let mut tokens = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        tokens.push(sample()?);
        elapsed.push(start.elapsed().as_nanos());
    }
    elapsed.sort_unstable();
    Ok((elapsed[elapsed.len() / 2], tokens))
}

fn deterministic_logits(vocab: usize) -> Vec<f32> {
    (0..vocab)
        .map(|index| {
            let x = index as f32;
            (x * 0.019).sin() * 1.7 + (x * 0.007).cos() * 0.4 + (index % 17) as f32 * 0.001
        })
        .collect()
}

fn fingerprint(tokens: &[usize]) -> u64 {
    let mut hash = FNV_OFFSET;
    for &token in tokens {
        for byte in (token as u64).to_le_bytes() {
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
    if values.is_empty() || (name.ends_with("VOCABS") && values.contains(&0)) {
        return Err(format!("{name} must contain positive comma-separated integers").into());
    }
    Ok(values)
}

fn parse_positive(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    let value = env::var(name)
        .ok()
        .map(|raw| raw.parse::<usize>())
        .transpose()?
        .unwrap_or(default);
    if value == 0 {
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
