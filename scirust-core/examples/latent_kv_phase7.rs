//! Deterministic Phase 7 runtime harness for quantized latent attention.

use scirust_core::nn::latent_kv_cache::{
    LatentAttentionScratch, LatentStorageFormat, QuantizedLatentKvCache,
};
use scirust_core::nn::paged_attention::contiguous_attention;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy)]
struct Scenario {
    name: &'static str,
    rank: usize,
    format: LatentStorageFormat,
    max_error: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let scenarios = [
        Scenario {
            name: "full-f32",
            rank: 16,
            format: LatentStorageFormat::F32,
            max_error: 2.0e-6,
        },
        Scenario {
            name: "full-int8",
            rank: 16,
            format: LatentStorageFormat::Int8,
            max_error: 0.02,
        },
        Scenario {
            name: "full-int4",
            rank: 16,
            format: LatentStorageFormat::Int4,
            max_error: 0.20,
        },
        Scenario {
            name: "rank8-int4",
            rank: 8,
            format: LatentStorageFormat::Int4,
            max_error: 0.25,
        },
    ];

    println!(
        "scenario,tokens,dimension,rank,format,dense_bytes,used_bytes,allocated_bytes,compression_ratio,max_absolute_error,quality_guard_met,output_fingerprint"
    );
    for scenario in scenarios
    {
        run_scenario(scenario)?;
    }
    Ok(())
}

fn run_scenario(scenario: Scenario) -> Result<(), Box<dyn std::error::Error>> {
    let tokens = 96;
    let dimension = 16;
    let keys = generated_matrix(tokens, dimension, 0x91a7_2001);
    let values = generated_matrix(tokens, dimension, 0x91a7_2002);
    let query = generated_vector(dimension, 0x91a7_2003);
    let expected = contiguous_attention(&keys, &values, &query, dimension, tokens);

    let basis = identity_prefix(dimension, scenario.rank);
    let mut cache = QuantizedLatentKvCache::new(
        tokens,
        dimension,
        scenario.rank,
        scenario.rank,
        scenario.format,
        scenario.format,
        basis.clone(),
        basis,
    )?;
    for token in 0..tokens
    {
        let offset = token * dimension;
        cache.append(
            &keys[offset..offset + dimension],
            &values[offset..offset + dimension],
        )?;
    }

    let mut actual = vec![0.0; dimension];
    let mut scratch = LatentAttentionScratch::new(tokens, scenario.rank, scenario.rank);
    cache.attention_into(&query, &mut actual, &mut scratch)?;

    let max_absolute_error = expected
        .iter()
        .zip(&actual)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    let quality_guard_met = max_absolute_error <= scenario.max_error;
    let dense_bytes = tokens * dimension * 2 * core::mem::size_of::<f32>();
    let allocated_bytes = cache.allocated_bytes();
    let compression_ratio = dense_bytes as f64 / allocated_bytes as f64;
    let fingerprint = fingerprint(scenario, &actual);

    println!(
        "{},{tokens},{dimension},{},{},{dense_bytes},{},{allocated_bytes},{compression_ratio:.9e},{max_absolute_error:.9e},{},{fingerprint:016x}",
        scenario.name,
        scenario.rank,
        scenario.format.label(),
        cache.used_bytes(),
        u8::from(quality_guard_met),
    );
    Ok(())
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
    let mut matrix = Vec::with_capacity(rows * columns);
    for row in 0..rows
    {
        for column in 0..columns
        {
            matrix.push(sample(seed, row * columns + column) * coordinate_scale(column));
        }
    }
    matrix
}

fn generated_vector(columns: usize, seed: u64) -> Vec<f32> {
    (0..columns)
        .map(|column| sample(seed, column) * coordinate_scale(column))
        .collect()
}

fn coordinate_scale(column: usize) -> f32 {
    0.78_f32.powi(column as i32)
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
    hash_bytes(&mut state, &(scenario.rank as u64).to_le_bytes());
    hash_bytes(&mut state, scenario.format.label().as_bytes());
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
