#![cfg(feature = "flat-attention")]

use scirust_gpu::{FlatM11ResidentConfig, GpuMatrix, WgpuFlatM11Bridge};

fn pre_rotated_decode_contract(
    bridge: &WgpuFlatM11Bridge,
    q: &GpuMatrix,
    k: &GpuMatrix,
    v: &GpuMatrix,
    config: FlatM11ResidentConfig,
) {
    let _ = bridge.forward_pre_rotated_k(q, k, v, config);
}

#[test]
fn sciagent_flat_feature_exposes_m15_prerotated_decode_boundary() {
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

    // The real resident SciAgent cache stores RoPE(K) and raw V. P6 therefore
    // requires the M15 pre-rotated-K entry point, with one decode query at the
    // current absolute position attending over the complete resident prefix.
    assert_eq!(config.query_len, 1);
    assert_eq!(config.kv_len, config.query_position_offset + 1);
    assert_eq!(
        config.query_rope_position_offset,
        config.query_position_offset
    );
    assert_eq!(config.kv_rope_position_offset, 0);
    assert!(config.causal);

    // Keep the method in the compile-time contract without requiring a GPU in
    // ordinary CI. A later P6 slice routes ResidentModel::decode_step through it
    // and supplies same-adapter end-to-end parity/latency evidence.
    let _contract: fn(
        &WgpuFlatM11Bridge,
        &GpuMatrix,
        &GpuMatrix,
        &GpuMatrix,
        FlatM11ResidentConfig,
    ) = pre_rotated_decode_contract;
}
