#![cfg(feature = "wgpu")]

use scirust_gpu::WgpuResidentLatentKvCache;

fn basis(dimension: usize, rank: usize) -> Vec<f32> {
    let mut basis = vec![0.0; dimension * rank];
    for index in 0..rank
    {
        basis[index * rank + index] = 1.0;
    }
    basis
}

fn project(vector: &[f32], basis: &[f32], dimension: usize, rank: usize) -> Vec<f32> {
    let mut output = vec![0.0; rank];
    for j in 0..rank
    {
        for i in 0..dimension
        {
            output[j] += vector[i] * basis[i * rank + j];
        }
    }
    output
}

fn cpu_attention(
    query: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    key_basis: &[f32],
    value_basis: &[f32],
    dimension: usize,
    rank: usize,
) -> Vec<f32> {
    let query_latent = project(query, key_basis, dimension, rank);
    let scale = 1.0 / (dimension as f32).sqrt();
    let mut scores = Vec::with_capacity(keys.len());
    for key in keys
    {
        let key_latent = project(key, key_basis, dimension, rank);
        let mut score = 0.0;
        for j in 0..rank
        {
            score += query_latent[j] * key_latent[j];
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
        let value_latent = project(value, value_basis, dimension, rank);
        for j in 0..rank
        {
            latent_context[j] += *weight * value_latent[j];
        }
    }

    let mut output = vec![0.0; dimension];
    for i in 0..dimension
    {
        for j in 0..rank
        {
            output[i] += latent_context[j] * value_basis[i * rank + j];
        }
    }
    output
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
fn resident_lower_rank_attention_matches_cpu_oracle() {
    let dimension = 4;
    let rank = 2;
    let key_basis = basis(dimension, rank);
    let value_basis = basis(dimension, rank);
    let mut cache = WgpuResidentLatentKvCache::new(4, dimension, rank, &key_basis, &value_basis)
        .expect("Phase 14 validation requires an available WGPU adapter");

    let keys = vec![
        vec![0.5, -0.2, 0.9, 0.1],
        vec![-0.4, 0.7, 0.3, -0.8],
        vec![0.9, 0.2, -0.5, 0.4],
    ];
    let values = vec![
        vec![0.2, 0.8, 0.6, -0.1],
        vec![0.5, -0.3, 0.7, 0.9],
        vec![-0.6, 0.4, 0.2, 0.3],
    ];
    for (key, value) in keys.iter().zip(&values)
    {
        cache.append(key, value).unwrap();
    }

    let query = [0.6, -0.1, 0.5, 0.2];
    let expected = cpu_attention(
        &query,
        &keys,
        &values,
        &key_basis,
        &value_basis,
        dimension,
        rank,
    );
    let mut actual = vec![0.0; dimension];
    cache.attention_into(&query, &mut actual).unwrap();
    assert_close(&actual, &expected, 5.0e-5);

    let telemetry = cache.telemetry();
    assert_eq!(telemetry.resident_tokens, 3);
    assert_eq!(telemetry.capacity_tokens, 4);
    assert_eq!(telemetry.rank, rank);
    assert!(telemetry.resident_bytes > 0);
}

#[test]
fn resident_ring_wraps_without_growing_persistent_storage() {
    let dimension = 4;
    let rank = 2;
    let key_basis = basis(dimension, rank);
    let value_basis = basis(dimension, rank);
    let mut cache = WgpuResidentLatentKvCache::new(2, dimension, rank, &key_basis, &value_basis)
        .expect("Phase 14 validation requires an available WGPU adapter");
    let resident_bytes = cache.telemetry().resident_bytes;

    let all_keys = [
        vec![0.1, 0.2, 0.3, 0.4],
        vec![0.5, -0.2, 0.7, 0.1],
        vec![-0.3, 0.9, 0.2, -0.4],
    ];
    let all_values = [
        vec![0.7, 0.1, -0.2, 0.5],
        vec![-0.4, 0.6, 0.3, 0.8],
        vec![0.2, -0.5, 0.9, 0.4],
    ];
    for (key, value) in all_keys.iter().zip(&all_values)
    {
        cache.append(key, value).unwrap();
        assert_eq!(cache.telemetry().resident_bytes, resident_bytes);
    }

    let query = [0.4, 0.3, -0.2, 0.7];
    let expected = cpu_attention(
        &query,
        &all_keys[1..],
        &all_values[1..],
        &key_basis,
        &value_basis,
        dimension,
        rank,
    );
    let actual = cache.attention(&query).unwrap();
    assert_close(&actual, &expected, 5.0e-5);

    let telemetry = cache.telemetry();
    assert_eq!(telemetry.resident_tokens, 2);
    assert_eq!(telemetry.next_write_slot, 1);
    assert_eq!(telemetry.resident_bytes, resident_bytes);
}
