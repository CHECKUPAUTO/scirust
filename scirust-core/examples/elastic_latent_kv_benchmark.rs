//! Reproducible long-context comparison harness for Elastic Latent KV.
//!
//! This is an attention-layer/cache microbenchmark, not an end-to-end language-model
//! benchmark. It compares the dense numeric decode backend with the validated Phase 7,
//! Phase 8, and adaptive Phase 9 policy materialized through the Phase 8 backend.

use scirust_core::nn::adaptive_latent_kv::{
    AdaptiveKvPolicyConfig, AdaptiveQualityProfile, estimate_channel_bytes, select_adaptive_plan,
};
use scirust_core::nn::adaptive_latent_kv_backend::AdaptiveResidualLatentBackend;
use scirust_core::nn::init::{KaimingNormal, Zeros};
use scirust_core::nn::kv_backend::{AttentionBackend, PlainKvCache, decode_step};
use scirust_core::nn::latent_kv_backend::LatentQuantizedBackend;
use scirust_core::nn::latent_kv_cache::LatentStorageFormat;
use scirust_core::nn::residual_latent_kv_backend::ResidualLatentQuantizedBackend;
use scirust_core::nn::rng::PcgEngine;
use scirust_core::nn::transformer::attention::MultiHeadAttention;
use std::env;
use std::time::Instant;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const DEFAULT_CONTEXTS: &str = "1024,4096,16384,32768";
const DEFAULT_REPEATS: usize = 3;
const DEFAULT_ADAPTIVE_BUDGET_BPS: usize = 3_000;
const D_MODEL: usize = 64;
const HEADS: usize = 4;
const MAX_RESIDUAL_SLOTS: usize = 4;

#[derive(Clone, Copy)]
enum Variant {
    Dense,
    Phase7,
    Phase8,
    Adaptive,
}

impl Variant {
    const ALL: [Self; 4] = [Self::Dense, Self::Phase7, Self::Phase8, Self::Adaptive];

    const fn label(self) -> &'static str {
        match self {
            Self::Dense => "dense-kv",
            Self::Phase7 => "phase7-latent-int4",
            Self::Phase8 => "phase8-residual-int4",
            Self::Adaptive => "adaptive-policy",
        }
    }
}

struct BackendSet {
    backends: Vec<Box<dyn AttentionBackend>>,
    planned_persistent_bytes: usize,
    quality_bps: u16,
    plan_fingerprint: u64,
}

struct Measurement {
    variant: Variant,
    context_tokens: usize,
    allocated_bytes: usize,
    planned_persistent_bytes: usize,
    quality_bps: u16,
    plan_fingerprint: u64,
    cache_prefill_ns: u128,
    decode_ns: u128,
    output: Vec<f32>,
    output_fingerprint: u64,
    deterministic: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let contexts = parse_contexts()?;
    let repeats = parse_positive_usize("SCIRUST_ELASTIC_KV_BENCH_REPEATS", DEFAULT_REPEATS)?;
    let adaptive_budget_bps = parse_positive_usize(
        "SCIRUST_ELASTIC_KV_BENCH_BUDGET_BPS",
        DEFAULT_ADAPTIVE_BUDGET_BPS,
    )?;
    if adaptive_budget_bps > 10_000 {
        return Err("SCIRUST_ELASTIC_KV_BENCH_BUDGET_BPS must be <= 10000".into());
    }

    let mut rng = PcgEngine::new(0x13_00_00_01);
    let attention = MultiHeadAttention::new(D_MODEL, HEADS, false, &KaimingNormal, &Zeros, &mut rng);

    let mut cold_decode_ns = [0_u128; 4];
    for (index, variant) in Variant::ALL.into_iter().enumerate() {
        cold_decode_ns[index] = measure(
            &attention,
            variant,
            0,
            repeats,
            adaptive_budget_bps,
        )?
        .decode_ns;
    }

    println!(
        "variant,context_tokens,resident_tokens,d_model,heads,d_head,allocated_bytes,planned_persistent_bytes,dense_allocated_bytes,compression_ratio,memory_saved_percent,cache_prefill_ns,cache_prefill_tokens_per_second,attention_ttft_proxy_ns,decode_latency_ns,attention_tokens_per_second,max_output_absolute_error,deterministic,quality_bps,plan_fingerprint,output_fingerprint"
    );

    for context_tokens in contexts {
        let dense = measure(
            &attention,
            Variant::Dense,
            context_tokens,
            repeats,
            adaptive_budget_bps,
        )?;
        emit(&dense, &dense.output, dense.allocated_bytes, cold_decode_ns[0]);

        for (index, variant) in Variant::ALL.into_iter().enumerate().skip(1) {
            let measurement = measure(
                &attention,
                variant,
                context_tokens,
                repeats,
                adaptive_budget_bps,
            )?;
            emit(
                &measurement,
                &dense.output,
                dense.allocated_bytes,
                cold_decode_ns[index],
            );
        }
    }

    Ok(())
}

fn measure(
    attention: &MultiHeadAttention,
    variant: Variant,
    context_tokens: usize,
    repeats: usize,
    adaptive_budget_bps: usize,
) -> Result<Measurement, Box<dyn std::error::Error>> {
    let token = deterministic_token(context_tokens, attention.d_model);
    let mut prefill_samples = Vec::with_capacity(repeats);
    let mut decode_samples = Vec::with_capacity(repeats);
    let mut first_output = None;
    let mut first_fingerprint = None;
    let mut deterministic = true;
    let mut allocated_bytes = 0;
    let mut planned_persistent_bytes = 0;
    let mut quality_bps = 10_000;
    let mut plan_fingerprint = 0;

    for _ in 0..repeats {
        let mut set = build_backends(
            variant,
            context_tokens + 1,
            attention.d_head,
            attention.n_heads,
            adaptive_budget_bps,
        )?;

        let prefill_start = Instant::now();
        prefill_backends(&mut set.backends, context_tokens, attention.d_head);
        prefill_samples.push(prefill_start.elapsed().as_nanos());

        let decode_start = Instant::now();
        let output = decode_step(attention, &token, &mut set.backends);
        decode_samples.push(decode_start.elapsed().as_nanos());

        allocated_bytes = set.backends.iter().map(|backend| backend.packed_bytes()).sum();
        planned_persistent_bytes = set.planned_persistent_bytes;
        quality_bps = set.quality_bps;
        plan_fingerprint = set.plan_fingerprint;

        let fingerprint = fingerprint(&output);
        if let Some(expected) = first_fingerprint {
            deterministic &= expected == fingerprint;
        } else {
            first_fingerprint = Some(fingerprint);
            first_output = Some(output);
        }
    }

    Ok(Measurement {
        variant,
        context_tokens,
        allocated_bytes,
        planned_persistent_bytes,
        quality_bps,
        plan_fingerprint,
        cache_prefill_ns: median(&mut prefill_samples),
        decode_ns: median(&mut decode_samples),
        output: first_output.expect("repeats is non-zero"),
        output_fingerprint: first_fingerprint.expect("repeats is non-zero"),
        deterministic,
    })
}

fn build_backends(
    variant: Variant,
    capacity_tokens: usize,
    dimension: usize,
    heads: usize,
    adaptive_budget_bps: usize,
) -> Result<BackendSet, Box<dyn std::error::Error>> {
    let rank = (dimension / 2).max(1);
    match variant {
        Variant::Dense => Ok(BackendSet {
            backends: (0..heads)
                .map(|_| Box::new(PlainKvCache::new(dimension)) as Box<dyn AttentionBackend>)
                .collect(),
            planned_persistent_bytes: 0,
            quality_bps: 10_000,
            plan_fingerprint: 0,
        }),
        Variant::Phase7 => {
            let mut backends = Vec::with_capacity(heads);
            for _ in 0..heads {
                backends.push(Box::new(LatentQuantizedBackend::new_symmetric(
                    capacity_tokens,
                    dimension,
                    rank,
                    LatentStorageFormat::Int4,
                    identity_prefix(dimension, rank),
                )?) as Box<dyn AttentionBackend>);
            }
            Ok(BackendSet {
                backends,
                planned_persistent_bytes: 0,
                quality_bps: 0,
                plan_fingerprint: 0,
            })
        },
        Variant::Phase8 => {
            let mut backends = Vec::with_capacity(heads);
            for _ in 0..heads {
                backends.push(Box::new(ResidualLatentQuantizedBackend::new_symmetric(
                    capacity_tokens,
                    dimension,
                    rank,
                    LatentStorageFormat::Int4,
                    identity_prefix(dimension, rank),
                    2.min(dimension),
                    LatentStorageFormat::Int4,
                )?) as Box<dyn AttentionBackend>);
            }
            Ok(BackendSet {
                backends,
                planned_persistent_bytes: 0,
                quality_bps: 0,
                plan_fingerprint: 0,
            })
        },
        Variant::Adaptive => {
            let full_basis = identity(dimension);
            let rank_quality = rank_quality_profile(dimension);
            let residual_gain = residual_gain_profile(MAX_RESIDUAL_SLOTS);
            let profile = AdaptiveQualityProfile {
                key_rank_quality_bps: &rank_quality,
                value_rank_quality_bps: &rank_quality,
                key_residual_gain_bps: &residual_gain,
                value_residual_gain_bps: &residual_gain,
            };
            let dense_per_head = capacity_tokens
                .saturating_mul(dimension)
                .saturating_mul(2)
                .saturating_mul(core::mem::size_of::<f32>());
            let target_budget = dense_per_head
                .saturating_mul(adaptive_budget_bps)
                .saturating_div(10_000);
            let minimum_rank = (dimension / 4).max(1);
            let minimum_budget = estimate_channel_bytes(
                capacity_tokens,
                dimension,
                minimum_rank,
                0,
                LatentStorageFormat::Int4,
                LatentStorageFormat::Int4,
            )
            .saturating_mul(2);
            let per_head_budget = target_budget.max(minimum_budget);
            let plan = select_adaptive_plan(
                AdaptiveKvPolicyConfig {
                    capacity_tokens,
                    dimension,
                    minimum_rank,
                    maximum_rank: dimension,
                    maximum_residual_slots: MAX_RESIDUAL_SLOTS.min(dimension),
                    budget_bytes: per_head_budget,
                },
                profile,
            )?;
            let mut backends = Vec::with_capacity(heads);
            for _ in 0..heads {
                backends.push(Box::new(AdaptiveResidualLatentBackend::new(
                    capacity_tokens,
                    dimension,
                    &full_basis,
                    &full_basis,
                    plan,
                )?) as Box<dyn AttentionBackend>);
            }
            let plan_fingerprint = (0..heads).fold(FNV_OFFSET, |state, _| {
                (state ^ plan.fingerprint).wrapping_mul(FNV_PRIME)
            });
            Ok(BackendSet {
                backends,
                planned_persistent_bytes: plan.persistent_bytes.saturating_mul(heads),
                quality_bps: plan.worst_quality_bps,
                plan_fingerprint,
            })
        },
    }
}

fn prefill_backends(backends: &mut [Box<dyn AttentionBackend>], tokens: usize, dimension: usize) {
    let mut key = vec![0.0_f32; dimension];
    let mut value = vec![0.0_f32; dimension];
    for position in 0..tokens {
        for (head, backend) in backends.iter_mut().enumerate() {
            fill_head_vector(&mut key, position, head, 0x9e37_0001);
            fill_head_vector(&mut value, position, head, 0x9e37_0002);
            backend.append(&key, &value);
        }
    }
}

fn fill_head_vector(output: &mut [f32], position: usize, head: usize, salt: u64) {
    for (coordinate, scalar) in output.iter_mut().enumerate() {
        let index = position
            .wrapping_mul(131)
            .wrapping_add(head.wrapping_mul(17))
            .wrapping_add(coordinate);
        *scalar = sample(salt, index) * 0.84_f32.powi(coordinate as i32);
    }
}

fn deterministic_token(step: usize, dimension: usize) -> Vec<f32> {
    (0..dimension)
        .map(|index| {
            let phase = (step.wrapping_mul(dimension).wrapping_add(index)) as f32;
            (phase * 0.071).sin() * 0.55 + (phase * 0.037).cos() * 0.25
        })
        .collect()
}

fn rank_quality_profile(dimension: usize) -> Vec<u16> {
    (1..=dimension)
        .map(|rank| ((rank * 10_000) / dimension).min(10_000) as u16)
        .collect()
}

fn residual_gain_profile(maximum_slots: usize) -> Vec<u16> {
    (0..=maximum_slots)
        .map(|slots| (slots.saturating_mul(450)).min(1_800) as u16)
        .collect()
}

fn identity(dimension: usize) -> Vec<f32> {
    let mut basis = vec![0.0; dimension * dimension];
    for diagonal in 0..dimension {
        basis[diagonal * dimension + diagonal] = 1.0;
    }
    basis
}

fn identity_prefix(dimension: usize, rank: usize) -> Vec<f32> {
    let mut basis = vec![0.0; dimension * rank];
    for diagonal in 0..rank {
        basis[diagonal * rank + diagonal] = 1.0;
    }
    basis
}

fn sample(seed: u64, index: usize) -> f32 {
    let mut value = seed.wrapping_add((index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    let unit = (value >> 40) as f32 / ((1_u32 << 24) - 1) as f32;
    unit * 2.0 - 1.0
}

fn emit(
    measurement: &Measurement,
    dense_output: &[f32],
    dense_allocated_bytes: usize,
    ttft_proxy_ns: u128,
) {
    let compression_ratio = if measurement.allocated_bytes == 0 {
        0.0
    } else {
        dense_allocated_bytes as f64 / measurement.allocated_bytes as f64
    };
    let memory_saved_percent = if dense_allocated_bytes == 0 {
        0.0
    } else {
        100.0 * (1.0 - measurement.allocated_bytes as f64 / dense_allocated_bytes as f64)
    };
    let prefill_tokens_per_second = rate(measurement.context_tokens, measurement.cache_prefill_ns);
    let decode_tokens_per_second = rate(1, measurement.decode_ns);
    let maximum_error = max_absolute_error(dense_output, &measurement.output);
    println!(
        "{},{},{},{D_MODEL},{HEADS},{},{},{},{},{compression_ratio:.9e},{memory_saved_percent:.9e},{},{prefill_tokens_per_second:.9e},{ttft_proxy_ns},{},{decode_tokens_per_second:.9e},{maximum_error:.9e},{},{},{:016x},{:016x}",
        measurement.variant.label(),
        measurement.context_tokens,
        measurement.context_tokens + 1,
        D_MODEL / HEADS,
        measurement.allocated_bytes,
        measurement.planned_persistent_bytes,
        dense_allocated_bytes,
        measurement.cache_prefill_ns,
        measurement.decode_ns,
        u8::from(measurement.deterministic),
        measurement.quality_bps,
        measurement.plan_fingerprint,
        measurement.output_fingerprint,
    );
}

fn rate(tokens: usize, nanoseconds: u128) -> f64 {
    if nanoseconds == 0 {
        return 0.0;
    }
    tokens as f64 * 1_000_000_000.0 / nanoseconds as f64
}

fn max_absolute_error(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0_f32, f32::max)
}

fn median(samples: &mut [u128]) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn fingerprint(values: &[f32]) -> u64 {
    let mut state = FNV_OFFSET;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            state ^= u64::from(byte);
            state = state.wrapping_mul(FNV_PRIME);
        }
    }
    state
}

fn parse_contexts() -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let raw = env::var("SCIRUST_ELASTIC_KV_BENCH_CONTEXTS")
        .unwrap_or_else(|_| DEFAULT_CONTEXTS.to_owned());
    let mut contexts = Vec::new();
    for item in raw.split(',') {
        let value: usize = item.trim().parse()?;
        if value == 0 {
            return Err("benchmark contexts must be non-zero".into());
        }
        contexts.push(value);
    }
    if contexts.is_empty() {
        return Err("at least one benchmark context is required".into());
    }
    Ok(contexts)
}

fn parse_positive_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    let value = match env::var(name) {
        Ok(raw) => raw.parse()?,
        Err(_) => default,
    };
    if value == 0 {
        return Err(format!("{name} must be non-zero").into());
    }
    Ok(value)
}
