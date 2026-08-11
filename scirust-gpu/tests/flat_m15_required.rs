#![cfg(feature = "flat-autotune")]

use flat_attention::{
    AsymmetricGroupedAttentionShape, AsymmetricRotaryEmbeddingConfig, FlatAttentionConfig,
    forward_reference_projection_grouped_rope_asymmetric,
};
use scirust_gpu::{FlatM11ResidentConfig, WgpuFlatM11Bridge};

const ATOL: f32 = 1.5e-4;
const RTOL: f32 = 1.0e-3;

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.023 + phase;
            x.sin() * 1.875 + (x * 0.41).cos() * 0.28125
        })
        .collect()
}

fn rotate_k_projection(
    raw: &[f32],
    kv_len: usize,
    kv_heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Vec<f32> {
    let mut rotated = raw.to_vec();
    let width = kv_heads * head_dim;
    for position in 0..kv_len
    {
        let absolute_position = position_offset + position;
        for head in 0..kv_heads
        {
            let head_base = position * width + head * head_dim;
            for pair in 0..head_dim / 2
            {
                let dim = 2 * pair;
                let exponent = -2.0 * pair as f32 / head_dim as f32;
                let frequency = theta.powf(exponent);
                let angle = absolute_position as f32 * frequency;
                let (sin, cos) = angle.sin_cos();
                let even = raw[head_base + dim];
                let odd = raw[head_base + dim + 1];
                rotated[head_base + dim] = even * cos - odd * sin;
                rotated[head_base + dim + 1] = even * sin + odd * cos;
            }
        }
    }
    rotated
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate()
    {
        let tolerance = ATOL + RTOL * expected.abs();
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "index {index}: actual={actual}, expected={expected}, abs_error={error}, tolerance={tolerance}"
        );
    }
}

#[test]
fn required_lavapipe_executes_m11_and_m15_against_the_same_oracle() {
    let bridge = WgpuFlatM11Bridge::new()
        .unwrap_or_else(|error| panic!("WGPU adapter is required for M15 parity: {error}"));
    assert!(
        bridge.m15_available(),
        "FLAT M15 pipeline must compile on the required WGPU adapter"
    );

    let config = FlatM11ResidentConfig {
        batch: 1,
        q_heads: 8,
        kv_heads: 2,
        query_len: 1,
        kv_len: 17,
        head_dim: 64,
        causal: true,
        softmax_scale: None,
        query_position_offset: 16,
        theta: 10_000.0,
        query_rope_position_offset: 16,
        kv_rope_position_offset: 0,
    };
    let shape = AsymmetricGroupedAttentionShape {
        batch: config.batch,
        q_heads: config.q_heads,
        kv_heads: config.kv_heads,
        query_len: config.query_len,
        kv_len: config.kv_len,
        head_dim: config.head_dim,
        query_position_offset: config.query_position_offset,
    };
    let attention = FlatAttentionConfig {
        causal: config.causal,
        softmax_scale: config.softmax_scale,
    };
    let rotary = AsymmetricRotaryEmbeddingConfig {
        theta: config.theta,
        query_position_offset: config.query_rope_position_offset,
        kv_position_offset: config.kv_rope_position_offset,
    };

    let q = fixture(config.q_heads * config.head_dim, 0.25);
    let raw_k = fixture(config.kv_len * config.kv_heads * config.head_dim, 0.85);
    let v = fixture(config.kv_len * config.kv_heads * config.head_dim, 1.45);
    let rotated_k = rotate_k_projection(
        &raw_k,
        config.kv_len,
        config.kv_heads,
        config.head_dim,
        config.theta,
        config.kv_rope_position_offset,
    );

    let q_gpu = bridge
        .context()
        .upload(&q, 1, config.q_heads * config.head_dim);
    let k_gpu = bridge.context().upload(
        &rotated_k,
        config.kv_len,
        config.kv_heads * config.head_dim,
    );
    let v_gpu = bridge
        .context()
        .upload(&v, config.kv_len, config.kv_heads * config.head_dim);

    let m11_output = bridge
        .forward_pre_rotated_k(&q_gpu, &k_gpu, &v_gpu, config)
        .expect("generic FLAT M11 decode must execute");
    let m15_output = bridge
        .forward_pre_rotated_k_m15(&q_gpu, &k_gpu, &v_gpu, config)
        .expect("specialized FLAT M15 decode must execute");
    let m11 = bridge
        .context()
        .download(&m11_output)
        .expect("M11 output download must succeed");
    let m15 = bridge
        .context()
        .download(&m15_output)
        .expect("M15 output download must succeed");
    let expected = forward_reference_projection_grouped_rope_asymmetric(
        &q, &raw_k, &v, shape, attention, rotary,
    )
    .expect("FLAT scalar oracle must accept the fixture");

    assert_close(&m11, &expected.output);
    assert_close(&m15, &expected.output);
    assert_close(&m15, &m11);
}
