//! Deterministic Phase 8 harness for sparse-residual latent attention.

use scirust_core::nn::latent_kv_cache::LatentStorageFormat;
use scirust_core::nn::paged_attention::contiguous_attention;
use scirust_core::nn::residual_latent_kv_cache::{
    ResidualLatentAttentionScratch, ResidualQuantizedLatentKvCache, SparseResidualConfig,
};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy)]
struct Scenario {
    name: &'static str,
    residual_slots: usize,
    residual_format: LatentStorageFormat,
    maximum_error: f32,
}

struct Outcome {
    used_bytes: usize,
    allocated_bytes: usize,
    maximum_error: f32,
    output: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scenarios = [
        Scenario {
            name: "baseline-no-residual",
            residual_slots: 0,
            residual_format: LatentStorageFormat::F32,
            maximum_error: 1.0,
        },
        Scenario {
            name: "residual-f32",
            residual_slots: 2,
            residual_format: LatentStorageFormat::F32,
            maximum_error: 2.0e-6,
        },
        Scenario {
            name: "residual-int8",
            residual_slots: 2,
            residual_format: LatentStorageFormat::Int8,
            maximum_error: 0.01,
        },
        Scenario {
            name: "residual-int4",
            residual_slots: 2,
            residual_format: LatentStorageFormat::Int4,
            maximum_error: 0.15,
        },
        Scenario {
            name: "residual-int4-four-slots",
            residual_slots: 4,
            residual_format: LatentStorageFormat::Int4,
            maximum_error: 0.15,
        },
    ];

    let tokens = 96;
    let dimension = 16;
    let rank = 8;
    let keys = generated_matrix(tokens, dimension, 0x8100_0001);
    let values = generated_matrix(tokens, dimension, 0x8100_0002);
    let query = generated_query(dimension, 0x8100_0003);
    let oracle = contiguous_attention(&keys, &values, &query, dimension, tokens);
    let baseline = evaluate(
        scenarios[0],
        tokens,
        dimension,
        rank,
        &keys,
        &values,
        &query,
        &oracle,
    )?;

    println!(
        "scenario,tokens,dimension,rank,coefficient_format,residual_slots,residual_format,dense_bytes,used_bytes,allocated_bytes,compression_ratio,max_absolute_error,baseline_max_absolute_error,improvement_ratio,quality_guard_met,output_fingerprint"
    );
    for scenario in scenarios
    {
        let outcome = evaluate(
            scenario, tokens, dimension, rank, &keys, &values, &query, &oracle,
        )?;
        let dense_bytes = tokens * dimension * 2 * core::mem::size_of::<f32>();
        let compression_ratio = dense_bytes as f64 / outcome.allocated_bytes as f64;
        let improvement_ratio =
            baseline.maximum_error as f64 / f64::from(outcome.maximum_error.max(f32::MIN_POSITIVE));
        let quality_guard_met = outcome.maximum_error <= scenario.maximum_error;
        let fingerprint = fingerprint(scenario, &outcome.output);
        println!(
            "{},{tokens},{dimension},{rank},f32,{},{},{dense_bytes},{},{},{compression_ratio:.9e},{:.9e},{:.9e},{improvement_ratio:.9e},{},{fingerprint:016x}",
            scenario.name,
            scenario.residual_slots,
            scenario.residual_format.label(),
            outcome.used_bytes,
            outcome.allocated_bytes,
            outcome.maximum_error,
            baseline.maximum_error,
            u8::from(quality_guard_met),
        );
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn evaluate(
    scenario: Scenario,
    tokens: usize,
    dimension: usize,
    rank: usize,
    keys: &[f32],
    values: &[f32],
    query: &[f32],
    oracle: &[f32],
) -> Result<Outcome, Box<dyn std::error::Error>> {
    let basis = identity_prefix(dimension, rank);
    let residual = SparseResidualConfig::new(scenario.residual_slots, scenario.residual_format);
    let mut cache = ResidualQuantizedLatentKvCache::new(
        tokens,
        dimension,
        rank,
        rank,
        LatentStorageFormat::F32,
        LatentStorageFormat::F32,
        basis.clone(),
        basis,
        residual,
        residual,
    )?;
    for token in 0..tokens
    {
        let start = token * dimension;
        cache.append(
            &keys[start..start + dimension],
            &values[start..start + dimension],
        )?;
    }

    let mut output = vec![0.0; dimension];
    let mut scratch = ResidualLatentAttentionScratch::new(tokens, rank, rank);
    cache.attention_into(query, &mut output, &mut scratch)?;
    let maximum_error = oracle
        .iter()
        .zip(&output)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0_f32, f32::max);

    Ok(Outcome {
        used_bytes: cache.used_bytes(),
        allocated_bytes: cache.allocated_bytes(),
        maximum_error,
        output,
    })
}

fn identity_prefix(dimension: usize, rank: usize) -> Vec<f32> {
    let mut basis = vec![0.0; dimension * rank];
    for diagonal in 0..rank
    {
        basis[diagonal * rank + diagonal] = 1.0;
    }
    basis
}

fn generated_matrix(rows: usize, columns: usize, seed: u64) -> Vec<f32> {
    let mut matrix = vec![0.0; rows * columns];
    for row in 0..rows
    {
        for column in 0..8
        {
            matrix[row * columns + column] =
                sample(seed, row * columns + column) * coordinate_scale(column);
        }
        matrix[row * columns + 8] = sample(seed ^ 0x81, row * 2) * 0.9;
        matrix[row * columns + 9] = sample(seed ^ 0x18, row * 2 + 1) * -0.75;
    }
    matrix
}

fn generated_query(columns: usize, seed: u64) -> Vec<f32> {
    let mut query = vec![0.0; columns];
    for (column, scalar) in query.iter_mut().enumerate().take(8)
    {
        *scalar = sample(seed, column) * coordinate_scale(column);
    }
    query[8] = 0.85;
    query[9] = -0.65;
    query
}

fn coordinate_scale(column: usize) -> f32 {
    0.82_f32.powi(column as i32)
}

fn sample(seed: u64, index: usize) -> f32 {
    let mut value = seed.wrapping_add((index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    let unit = (value >> 40) as f32 / ((1_u32 << 24) - 1) as f32;
    unit * 2.0 - 1.0
}

fn fingerprint(scenario: Scenario, output: &[f32]) -> u64 {
    let mut state = FNV_OFFSET;
    hash_bytes(&mut state, scenario.name.as_bytes());
    hash_bytes(&mut state, &(scenario.residual_slots as u64).to_le_bytes());
    hash_bytes(&mut state, scenario.residual_format.label().as_bytes());
    for value in output
    {
        hash_bytes(&mut state, &value.to_bits().to_le_bytes());
    }
    state
}

fn hash_bytes(state: &mut u64, bytes: &[u8]) {
    for byte in bytes
    {
        *state ^= u64::from(*byte);
        *state = state.wrapping_mul(FNV_PRIME);
    }
}
