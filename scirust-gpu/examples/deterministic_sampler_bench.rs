//! Reproducible Phase 23 characterization of the deterministic WGPU sampler.
//!
//! This benchmark measures the public `sample()` boundary: one logits upload,
//! one WGPU dispatch, synchronization, and one `u32` result readback. It also
//! reports the exact number of selection-sort comparisons implied by the
//! sampler's `ranking_passes_per_sample()` value.
//!
//! Wall-clock values are meaningful only for the adapter that executes them.
//! In particular, Mesa lavapipe is a software Vulkan implementation and its
//! timing must not be presented as real-GPU throughput.

use std::env;
use std::time::Instant;

use scirust_core::nn::sampling::SamplingConfig;
use scirust_gpu::WgpuDeterministicSampler;

const DEFAULT_VOCABS: &str = "1024,4096";
const DEFAULT_TOP_KS: &str = "0,1,5,50,200";
const DEFAULT_REPEATS: usize = 7;
const DEFAULT_WARMUP: usize = 2;
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vocabs = parse_list("SCIRUST_SAMPLER_BENCH_VOCABS", DEFAULT_VOCABS)?;
    let top_ks = parse_list("SCIRUST_SAMPLER_BENCH_TOP_KS", DEFAULT_TOP_KS)?;
    let repeats = parse_positive("SCIRUST_SAMPLER_BENCH_REPEATS", DEFAULT_REPEATS)?;
    let warmup = parse_non_negative("SCIRUST_SAMPLER_BENCH_WARMUP", DEFAULT_WARMUP)?;

    println!(
        "vocab_size,top_k,ranking_passes,selection_comparisons,comparison_fraction_of_full,fast_path,repeats,median_sample_ns,samples_per_second,deterministic,output_fingerprint,resident_bytes,upload_bytes_per_sample,download_bytes_per_sample"
    );

    for vocab_size in vocabs
    {
        let logits = deterministic_logits(vocab_size);
        for &top_k in &top_ks
        {
            emit_measurement(vocab_size, top_k, repeats, warmup, &logits)?;
        }
    }
    Ok(())
}

fn emit_measurement(
    vocab_size: usize,
    top_k: usize,
    repeats: usize,
    warmup: usize,
    logits: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let config = SamplingConfig {
        temperature: 0.9,
        top_k,
        top_p: 1.0,
    };
    let seed = 0x23_00_00_00_u64 ^ ((vocab_size as u64) << 16) ^ top_k as u64;
    let mut sampler = WgpuDeterministicSampler::new(vocab_size, config, seed)?;

    for _ in 0..warmup
    {
        std::hint::black_box(sampler.sample(logits)?);
    }
    sampler.reset()?;

    let mut durations = Vec::with_capacity(repeats);
    let mut outputs = Vec::with_capacity(repeats);
    for _ in 0..repeats
    {
        let started = Instant::now();
        let token = sampler.sample(logits)?;
        durations.push(started.elapsed().as_nanos());
        outputs.push(token);
    }

    sampler.reset()?;
    let mut deterministic = true;
    for &expected in &outputs
    {
        deterministic &= sampler.sample(logits)? == expected;
    }

    durations.sort_unstable();
    let median_ns = durations[durations.len() / 2];
    let ranking_passes = sampler.ranking_passes_per_sample();
    let comparisons = selection_comparisons(vocab_size, ranking_passes);
    let full_comparisons = selection_comparisons(vocab_size, vocab_size);
    let fraction = if full_comparisons == 0
    {
        0.0
    }
    else
    {
        comparisons as f64 / full_comparisons as f64
    };
    let samples_per_second = 1.0e9 / (median_ns.max(1) as f64);
    let telemetry = sampler.telemetry();

    println!(
        "{vocab_size},{top_k},{ranking_passes},{comparisons},{fraction:.9},{},{repeats},{median_ns},{samples_per_second:.6},{},{:016x},{},{},{}",
        u8::from(sampler.uses_bounded_top_k_fast_path()),
        u8::from(deterministic),
        fingerprint(&outputs),
        telemetry.resident_bytes,
        telemetry.upload_bytes_per_sample,
        telemetry.download_bytes_per_sample,
    );
    Ok(())
}

fn selection_comparisons(vocab_size: usize, ranking_passes: usize) -> u128 {
    let passes = ranking_passes.min(vocab_size) as u128;
    let vocab = vocab_size as u128;
    passes * (2 * vocab - passes - 1) / 2
}

fn deterministic_logits(vocab_size: usize) -> Vec<f32> {
    (0..vocab_size)
        .map(|index| {
            let x = index as f32;
            (x * 0.019).sin() * 1.7 + (x * 0.007).cos() * 0.4 + (index % 17) as f32 * 0.001
        })
        .collect()
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
    if values.is_empty() || values.contains(&0) && name.ends_with("VOCABS")
    {
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
