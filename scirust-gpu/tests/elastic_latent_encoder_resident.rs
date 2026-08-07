#![cfg(feature = "wgpu")]

use scirust_core::autodiff::reverse::{Tape, Tensor};
use scirust_core::nn::init::{KaimingNormal, Zeros};
use scirust_core::nn::rng::PcgEngine;
use scirust_core::nn::transformer::{block::TransformerBlock, encoder::TransformerEncoder};
use scirust_gpu::{
    WgpuLatentHeadBasis, WgpuLatentLayerBasis, WgpuResidentTransformerEncoder,
    WgpuResidentTransformerEncoderError,
};

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

fn linear(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    in_width: usize,
    out_width: usize,
) -> Vec<f32> {
    let mut output = bias.to_vec();
    for column in 0..out_width
    {
        for row in 0..in_width
        {
            output[column] += input[row] * weight[row * out_width + column];
        }
    }
    output
}

fn layer_norm(input: &[f32], gamma: &[f32], beta: &[f32], eps: f32) -> Vec<f32> {
    let width = input.len();
    let mean = input.iter().sum::<f32>() / width as f32;
    let mut variance = 0.0;
    for value in input
    {
        let delta = *value - mean;
        variance += delta * delta;
    }
    variance /= width as f32;
    let inv_std = 1.0 / (variance + eps).sqrt();
    input
        .iter()
        .zip(gamma)
        .zip(beta)
        .map(|((value, scale), shift)| (value - mean) * inv_std * scale + shift)
        .collect()
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
    let mut scores = Vec::with_capacity(keys.len());
    for key in keys
    {
        let key_latent = project(key, basis, dimension, rank);
        let mut score = 0.0;
        for latent in 0..rank
        {
            score += query_latent[latent] * key_latent[latent];
        }
        scores.push(score * scale);
    }

    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut denominator = 0.0;
    for score in &mut scores
    {
        *score = (*score - maximum).exp();
        denominator += *score;
    }
    for score in &mut scores
    {
        *score /= denominator;
    }

    let mut latent_context = vec![0.0; rank];
    for (weight, value) in scores.iter().zip(values)
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

fn cpu_sliding_latent_block(
    block: &TransformerBlock,
    input: &[f32],
    key_history: &mut Vec<Vec<f32>>,
    value_history: &mut Vec<Vec<f32>>,
    capacity: usize,
    basis: &[f32],
    rank: usize,
) -> Vec<f32> {
    let d_model = block.d_model;
    let ln1 = layer_norm(
        input,
        &block.ln1.gamma.data,
        &block.ln1.beta.data,
        block.ln1.eps,
    );
    let q = linear(
        &ln1,
        &block.mha.w_q.weight.data,
        &block.mha.w_q.bias.data,
        d_model,
        d_model,
    );
    let k = linear(
        &ln1,
        &block.mha.w_k.weight.data,
        &block.mha.w_k.bias.data,
        d_model,
        d_model,
    );
    let v = linear(
        &ln1,
        &block.mha.w_v.weight.data,
        &block.mha.w_v.bias.data,
        d_model,
        d_model,
    );
    key_history.push(k);
    value_history.push(v);
    if key_history.len() > capacity
    {
        key_history.remove(0);
        value_history.remove(0);
    }

    let mut combined = vec![0.0; d_model];
    for head in 0..block.n_heads
    {
        let start = head * block.mha.d_head;
        let query = &q[start..start + block.mha.d_head];
        let keys: Vec<Vec<f32>> = key_history
            .iter()
            .map(|row| row[start..start + block.mha.d_head].to_vec())
            .collect();
        let values: Vec<Vec<f32>> = value_history
            .iter()
            .map(|row| row[start..start + block.mha.d_head].to_vec())
            .collect();
        let context = latent_attention(query, &keys, &values, basis, block.mha.d_head, rank);
        combined[start..start + block.mha.d_head].copy_from_slice(&context);
    }

    let attention = linear(
        &combined,
        &block.mha.w_o.weight.data,
        &block.mha.w_o.bias.data,
        d_model,
        d_model,
    );
    let x1: Vec<f32> = input
        .iter()
        .zip(attention)
        .map(|(left, right)| left + right)
        .collect();
    let ln2 = layer_norm(
        &x1,
        &block.ln2.gamma.data,
        &block.ln2.beta.data,
        block.ln2.eps,
    );
    let mut hidden = linear(
        &ln2,
        &block.ffn1.weight.data,
        &block.ffn1.bias.data,
        d_model,
        block.d_ff,
    );
    for value in &mut hidden
    {
        *value = value.max(0.0);
    }
    let ffn = linear(
        &hidden,
        &block.ffn2.weight.data,
        &block.ffn2.bias.data,
        block.d_ff,
        d_model,
    );
    x1.iter()
        .zip(ffn)
        .map(|(left, right)| left + right)
        .collect()
}

fn cpu_sliding_latent_encoder(
    encoder: &TransformerEncoder,
    input: &[f32],
    key_histories: &mut [Vec<Vec<f32>>],
    value_histories: &mut [Vec<Vec<f32>>],
    capacity: usize,
    basis: &[f32],
    rank: usize,
) -> Vec<f32> {
    let mut hidden = input.to_vec();
    for layer in 0..encoder.blocks.len()
    {
        hidden = cpu_sliding_latent_block(
            &encoder.blocks[layer],
            &hidden,
            &mut key_histories[layer],
            &mut value_histories[layer],
            capacity,
            basis,
            rank,
        );
    }
    layer_norm(
        &hidden,
        &encoder.final_ln.gamma.data,
        &encoder.final_ln.beta.data,
        encoder.final_ln.eps,
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
fn full_rank_resident_encoder_matches_legacy_incremental_oracle() {
    let mut rng = PcgEngine::new(79);
    let encoder = TransformerEncoder::new(2, 8, 2, 16, true, &KaimingNormal, &Zeros, &mut rng);
    let mut dense = encoder.clone();
    let basis = identity_basis(4);
    let heads = vec![
        WgpuLatentHeadBasis {
            key: &basis,
            value: &basis,
        };
        2
    ];
    let layers = vec![WgpuLatentLayerBasis { heads: &heads }; 2];
    let mut resident = WgpuResidentTransformerEncoder::new(&encoder, 8, 4, &layers)
        .expect("Phase 17 validation requires an available WGPU adapter");
    let tape = Tape::new();

    for pos in 0..5
    {
        let input: Vec<f32> = (0..8)
            .map(|index| ((pos * 17 + index) as f32 * 0.13).sin() * 0.3)
            .collect();
        let token = tape.input(Tensor::from_vec(input.clone(), 1, 8));
        let expected_var = dense.infer_step(&tape, token, pos);
        let expected = tape.value(expected_var.idx());
        let actual = resident.infer_step_at(&input, pos).unwrap();
        assert_close(&actual, &expected.data, 3.0e-3);
    }

    let telemetry = resident.telemetry();
    assert_eq!(telemetry.steps, 5);
    assert_eq!(telemetry.resident_tokens, 5);
    assert_eq!(telemetry.n_layers, 2);
    assert_eq!(telemetry.rank, 4);
    assert_eq!(telemetry.upload_bytes_per_step, 8 * 4);
    assert_eq!(telemetry.download_bytes_per_step, 8 * 4);
}

#[test]
fn lower_rank_resident_encoder_matches_cpu_oracle_after_ring_wrap() {
    let mut rng = PcgEngine::new(83);
    let encoder = TransformerEncoder::new(2, 8, 2, 16, true, &KaimingNormal, &Zeros, &mut rng);
    let basis = prefix_basis(4, 2);
    let heads = vec![
        WgpuLatentHeadBasis {
            key: &basis,
            value: &basis,
        };
        2
    ];
    let layers = vec![WgpuLatentLayerBasis { heads: &heads }; 2];
    let capacity = 2;
    let mut resident = WgpuResidentTransformerEncoder::new(&encoder, capacity, 2, &layers)
        .expect("Phase 17 validation requires an available WGPU adapter");
    let resident_bytes = resident.telemetry().resident_bytes;
    let mut keys = vec![Vec::new(); 2];
    let mut values = vec![Vec::new(); 2];

    for pos in 0..5
    {
        let input: Vec<f32> = (0..8)
            .map(|index| ((pos * 9 + index) as f32 * 0.1).cos() * 0.25)
            .collect();
        let expected = cpu_sliding_latent_encoder(
            &encoder,
            &input,
            &mut keys,
            &mut values,
            capacity,
            &basis,
            2,
        );
        let actual = resident.infer_step_at(&input, pos).unwrap();
        assert_close(&actual, &expected, 3.0e-3);
        assert_eq!(resident.telemetry().resident_bytes, resident_bytes);
    }

    let telemetry = resident.telemetry();
    assert_eq!(telemetry.steps, 5);
    assert_eq!(telemetry.resident_tokens, 2);
    assert_eq!(telemetry.next_write_slot, 1);
    assert_eq!(telemetry.resident_bytes, resident_bytes);
}

#[test]
fn resident_encoder_rejects_position_and_reset_reuses_storage() {
    let mut rng = PcgEngine::new(89);
    let encoder = TransformerEncoder::new(2, 8, 2, 16, true, &KaimingNormal, &Zeros, &mut rng);
    let basis = identity_basis(4);
    let heads = vec![
        WgpuLatentHeadBasis {
            key: &basis,
            value: &basis,
        };
        2
    ];
    let layers = vec![WgpuLatentLayerBasis { heads: &heads }; 2];
    let mut resident = WgpuResidentTransformerEncoder::new(&encoder, 3, 4, &layers)
        .expect("Phase 17 validation requires an available WGPU adapter");
    let resident_bytes = resident.telemetry().resident_bytes;
    let input = vec![0.125; 8];

    let error = resident.infer_step_at(&input, 1).unwrap_err();
    assert!(matches!(
        error,
        WgpuResidentTransformerEncoderError::PositionMismatch {
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
