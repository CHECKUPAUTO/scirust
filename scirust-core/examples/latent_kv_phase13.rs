//! Deterministic Phase 13 integrated-runtime harness.

use scirust_core::nn::adaptive_latent_kv::AdaptiveQualityProfile;
use scirust_core::nn::elastic_latent_runtime::{
    ElasticLatentDecodeRuntime, ElasticLatentRuntimeConfig, HeadCalibration,
};
use scirust_core::nn::init::{KaimingNormal, Zeros};
use scirust_core::nn::latent_kv_cache::LatentStorageFormat;
use scirust_core::nn::latent_kv_kernels::LatentKernelKind;
use scirust_core::nn::latent_kv_lifecycle::{CompressionTier, LifecycleConfig};
use scirust_core::nn::rng::PcgEngine;
use scirust_core::nn::transformer::attention::MultiHeadAttention;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const RANK_QUALITY: [u16; 4] = [4_800, 7_400, 9_100, 10_000];
    const RESIDUAL_GAIN: [u16; 3] = [0, 450, 800];

    let mut rng = PcgEngine::new(13);
    let attention = MultiHeadAttention::new(16, 4, false, &KaimingNormal, &Zeros, &mut rng);
    let basis = identity(attention.d_head);
    let quality = AdaptiveQualityProfile {
        key_rank_quality_bps: &RANK_QUALITY,
        value_rank_quality_bps: &RANK_QUALITY,
        key_residual_gain_bps: &RESIDUAL_GAIN,
        value_residual_gain_bps: &RESIDUAL_GAIN,
    };
    let calibration = HeadCalibration {
        full_key_basis: &basis,
        full_value_basis: &basis,
        quality,
        basis_version: 7,
    };
    let calibrations = [calibration; 4];
    let capacity = 64;
    let hot = CompressionTier {
        coefficient_format: LatentStorageFormat::F32,
        residual_format: LatentStorageFormat::F32,
        maximum_residual_slots: 2,
        rank_divisor: 1,
    };
    let warm = CompressionTier {
        coefficient_format: LatentStorageFormat::Int8,
        residual_format: LatentStorageFormat::Int8,
        maximum_residual_slots: 1,
        rank_divisor: 1,
    };
    let cold = CompressionTier {
        coefficient_format: LatentStorageFormat::Int4,
        residual_format: LatentStorageFormat::Int4,
        maximum_residual_slots: 1,
        rank_divisor: 2,
    };
    let mut runtime = ElasticLatentDecodeRuntime::new(
        &attention,
        ElasticLatentRuntimeConfig {
            capacity_tokens: capacity,
            minimum_rank: 2,
            maximum_rank: 4,
            maximum_residual_slots: 2,
            persistent_budget_bytes: 8_192,
            allocated_ceiling_bytes: 65_536,
            lifecycle: LifecycleConfig {
                capacity_tokens: capacity,
                hot_tokens: 8,
                warm_tokens: 16,
                hot,
                warm,
                cold,
            },
            kernel: LatentKernelKind::Block4,
        },
        &calibrations,
    )?;

    let mut fingerprint = FNV_OFFSET;
    for step in 0..32
    {
        let token = deterministic_token(step, attention.d_model);
        let output = runtime.decode_step(&attention, &token)?;
        for value in output
        {
            fingerprint ^= u64::from(value.to_bits());
            fingerprint = fingerprint.wrapping_mul(FNV_PRIME);
        }
    }

    let telemetry = runtime.telemetry();
    let plan_fingerprint = runtime.plans().iter().fold(FNV_OFFSET, |state, plan| {
        (state ^ plan.fingerprint).wrapping_mul(FNV_PRIME)
    });
    println!(
        "steps,planned_persistent_bytes,allocated_bytes,worst_quality_bps,last_lifecycle_transitions,plan_fingerprint,output_fingerprint"
    );
    println!(
        "{},{},{},{},{},{},{}",
        telemetry.steps,
        telemetry.planned_persistent_bytes,
        telemetry.allocated_bytes,
        telemetry.worst_quality_bps,
        telemetry.last_lifecycle_transitions,
        plan_fingerprint,
        fingerprint,
    );
    Ok(())
}

fn identity(dimension: usize) -> Vec<f32> {
    let mut basis = vec![0.0; dimension * dimension];
    for index in 0..dimension
    {
        basis[index * dimension + index] = 1.0;
    }
    basis
}

fn deterministic_token(step: usize, dimension: usize) -> Vec<f32> {
    (0..dimension)
        .map(|index| {
            let phase = (step * dimension + index) as f32;
            (phase * 0.071).sin() * 0.55 + (phase * 0.037).cos() * 0.25
        })
        .collect()
}
