#![cfg(feature = "wgpu")]

use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::init::{KaimingNormal, Zeros};
use scirust_core::nn::rng::PcgEngine;
use scirust_core::nn::transformer::attention::MultiHeadAttention;
use scirust_gpu::{WgpuLatentHeadBasis, WgpuResidentLatentMha, WgpuResidentLatentMhaError};

fn identity_basis(dimension: usize) -> Vec<f32> {
    let mut basis = vec![0.0; dimension * dimension];
    for index in 0..dimension
    {
        basis[index * dimension + index] = 1.0;
    }
    basis
}

fn prefix_basis(dimension: usize, rank: usize) -> Vec<f32> {
    let mut basis = vec![0.0; dimension * rank];
    for index in 0..rank
    {
        basis[index * rank + index] = 1.0;
    }
    basis
}

fn linear(input: &[f32], weight: &[f32], bias: &[f32], width: usize) -> Vec<f32> {
    let mut output = bias.to_vec();
    for column in 0..width
    {
        for row in 0..width
        {
            output[column] += input[row] * weight[row * width + column];
        }
    }
    output
}

fn project(vector: &[f32], basis: &[f32], dimension: usize, rank: usize) -> Vec<f32> {
    let mut output = vec![0.0; rank];
    for latent in 0..rank
    {
        for index in 0..dimension
        {
            output[latent] += vector[index] * basis[index * rank + latent];
        }
    }
    output
}

fn latent_attention(
    query: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    basis: &[f32],
    dimension: usize,
    rank: usize,
) -> Vec<f32> {
    let query_latent = project(query, basis, dimension, rank);
    let scale = 1.0 / (dimension as f32).sqrt();
    let mut weights = Vec::with_capacity(keys.len());
    for key in keys
    {
        let key_latent = project(key, basis, dimension, rank);
        let mut score = 0.0;
        for latent in 0..rank
        {
            score += query_latent[latent] * key_latent[latent];
        }
        weights.push(score * scale);
    }

    let maximum = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut denominator = 0.0;
    for weight in &mut weights
    {
        *weight = (*weight - maximum).exp();
        denominator += *weight;
    }
    for weight in &mut weights
    {
        *weight /= denominator;
    }

    let mut latent_context = vec![0.0; rank];
    for (weight, value) in weights.iter().zip(values)
    {
        let value_latent = project(value, basis, dimension, rank);
        for latent in 0..rank
        {
            latent_context[latent] += *weight * value_latent[latent];
        }
    }

    let mut output = vec![0.0; dimension];
    for index in 0..dimension
    {
        for latent in 0..rank
        {
            output[index] += latent_context[latent] * basis[index * rank + latent];
        }
    }
    output
}

fn cpu_sliding_latent_mha(
    mha: &MultiHeadAttention,
    input: &[f32],
    key_history: &mut Vec<Vec<f32>>,
    value_history: &mut Vec<Vec<f32>>,
    capacity: usize,
    basis: &[f32],
    rank: usize,
) -> Vec<f32> {
    let q = linear(input, &mha.w_q.weight.data, &mha.w_q.bias.data, mha.d_model);
    let k = linear(input, &mha.w_k.weight.data, &mha.w_k.bias.data, mha.d_model);
    let v = linear(input, &mha.w_v.weight.data, &mha.w_v.bias.data, mha.d_model);
    key_history.push(k);
    value_history.push(v);
    if key_history.len() > capacity
    {
        key_history.remove(0);
        value_history.remove(0);
    }

    let mut combined = vec![0.0; mha.d_model];
    for head in 0..mha.n_heads
    {
        let start = head * mha.d_head;
        let query = &q[start..start + mha.d_head];
        let keys: Vec<Vec<f32>> = key_history
            .iter()
            .map(|row| row[start..start + mha.d_head].to_vec())
            .collect();
        let values: Vec<Vec<f32>> = value_history
            .iter()
            .map(|row| row[start..start + mha.d_head].to_vec())
            .collect();
        let context = latent_attention(query, &keys, &values, basis, mha.d_head, rank);
        combined[start..start + mha.d_head].copy_from_slice(&context);
    }

    linear(
        &combined,
        &mha.w_o.weight.data,
        &mha.w_o.bias.data,
        mha.d_model,
    )
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate()
    {
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "index {index}: actual={actual}, expected={expected}, error={error}, tolerance={tolerance}"
        );
    }
}

#[test]
fn fused_full_rank_mha_matches_dense_incremental_oracle() {
    let mut rng = PcgEngine::new(41);
    let mha = MultiHeadAttention::new(8, 2, true, &KaimingNormal, &Zeros, &mut rng);
    let mut dense = mha.clone();
    let basis = identity_basis(4);
    let heads = vec![
        WgpuLatentHeadBasis {
            key: &basis,
            value: &basis,
        };
        2
    ];
    let mut resident = WgpuResidentLatentMha::new(&mha, 8, 4, &heads)
        .expect("Phase 15 validation requires an available WGPU adapter");
    let tape = Tape::new();

    for pos in 0..5
    {
        let input: Vec<f32> = (0..8)
            .map(|index| ((pos * 8 + index) as f32 * 0.19).sin() * 0.35)
            .collect();
        let token = tape.input(Tensor::from_vec(input.clone(), 1, 8));
        let expected_var = dense.infer_step(&tape, token, pos);
        let expected = tape.value(expected_var.idx());
        let actual = resident.infer_step_at(&input, pos).unwrap();
        assert_close(&actual, &expected.data, 6.0e-4);
    }

    let telemetry = resident.telemetry();
    assert_eq!(telemetry.steps, 5);
    assert_eq!(telemetry.resident_tokens, 5);
    assert_eq!(telemetry.rank, 4);
    assert_eq!(telemetry.upload_bytes_per_step, 8 * 4);
    assert_eq!(telemetry.download_bytes_per_step, 8 * 4);
}

#[test]
fn fused_lower_rank_ring_matches_independent_cpu_sliding_oracle() {
    let mut rng = PcgEngine::new(53);
    let mha = MultiHeadAttention::new(8, 2, true, &KaimingNormal, &Zeros, &mut rng);
    let basis = prefix_basis(4, 2);
    let heads = vec![
        WgpuLatentHeadBasis {
            key: &basis,
            value: &basis,
        };
        2
    ];
    let capacity = 2;
    let mut resident = WgpuResidentLatentMha::new(&mha, capacity, 2, &heads)
        .expect("Phase 15 validation requires an available WGPU adapter");
    let resident_bytes = resident.telemetry().resident_bytes;
    let mut keys = Vec::new();
    let mut values = Vec::new();

    for pos in 0..5
    {
        let input: Vec<f32> = (0..8)
            .map(|index| ((pos * 11 + index) as f32 * 0.13).cos() * 0.4)
            .collect();
        let expected =
            cpu_sliding_latent_mha(&mha, &input, &mut keys, &mut values, capacity, &basis, 2);
        let actual = resident.infer_step_at(&input, pos).unwrap();
        assert_close(&actual, &expected, 6.0e-4);
        assert_eq!(resident.telemetry().resident_bytes, resident_bytes);
    }

    let telemetry = resident.telemetry();
    assert_eq!(telemetry.steps, 5);
    assert_eq!(telemetry.resident_tokens, 2);
    assert_eq!(telemetry.next_write_slot, 1);
    assert_eq!(telemetry.resident_bytes, resident_bytes);
}

#[test]
fn fused_mha_rejects_position_before_mutating_ring_and_reset_reuses_storage() {
    let mut rng = PcgEngine::new(67);
    let mha = MultiHeadAttention::new(8, 2, true, &KaimingNormal, &Zeros, &mut rng);
    let basis = identity_basis(4);
    let heads = vec![
        WgpuLatentHeadBasis {
            key: &basis,
            value: &basis,
        };
        2
    ];
    let mut resident = WgpuResidentLatentMha::new(&mha, 3, 4, &heads)
        .expect("Phase 15 validation requires an available WGPU adapter");
    let resident_bytes = resident.telemetry().resident_bytes;
    let input = vec![0.125; 8];

    let error = resident.infer_step_at(&input, 1).unwrap_err();
    assert!(matches!(
        error,
        WgpuResidentLatentMhaError::PositionMismatch {
            expected: 0,
            actual: 1
        }
    ));
    assert_eq!(resident.telemetry().steps, 0);
    assert_eq!(resident.telemetry().resident_tokens, 0);

    resident.infer_step_at(&input, 0).unwrap();
    assert_eq!(resident.telemetry().steps, 1);
    resident.reset().unwrap();
    let telemetry = resident.telemetry();
    assert_eq!(telemetry.steps, 0);
    assert_eq!(telemetry.resident_tokens, 0);
    assert_eq!(telemetry.next_write_slot, 0);
    assert_eq!(telemetry.resident_bytes, resident_bytes);
}
