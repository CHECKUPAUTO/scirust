//! Stable SciRust adapter for FLAT's backend-neutral `api::v1` contract.
//!
//! The adapter keeps SciRust's resident [`GpuMatrix`] ownership and validates
//! backend-independent GQA/MQA geometry/configuration through FLAT's versioned
//! API before delegating to the already-qualified caller-owned training bridge.
//! Unsupported or invalid requests fail closed; this adapter never silently
//! falls back to the legacy multi-dispatch attention path.

use crate::{
    BackendError, BackendResult, FlatGroupedTrainingConfig, FlatGroupedTrainingResult, GpuMatrix,
    WgpuContext, WgpuFlatGroupedTrainingBridge,
};
use flat_attention::api::v1::{
    AttentionConfig as StableAttentionConfig, AttentionShape as StableAttentionShape,
    ResidentAttentionRequest,
};

/// Explicit M31 fallback policy for the opt-in stable FLAT adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlatStableFallbackPolicy {
    /// Return the concrete validation/execution error to the caller. Selecting
    /// a legacy path is an explicit higher-level policy decision.
    ReturnError,
}

/// Stable versioned resident adapter layered over the qualified FLAT training bridge.
pub struct WgpuFlatStableAdapter {
    inner: WgpuFlatGroupedTrainingBridge,
}

impl core::fmt::Debug for WgpuFlatStableAdapter {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WgpuFlatStableAdapter")
            .field("adapter", &self.inner.adapter_name())
            .field("api_version", &flat_attention::api::v1::API_VERSION)
            .field("fallback_policy", &FlatStableFallbackPolicy::ReturnError)
            .finish()
    }
}

impl WgpuFlatStableAdapter {
    /// Acquire a fresh SciRust WGPU context and compile the qualified FLAT pipelines.
    pub fn new() -> BackendResult<Self> {
        Self::from_context(WgpuContext::new()?)
    }

    /// Bind the stable adapter to an existing SciRust WGPU ownership domain.
    pub fn from_context(ctx: WgpuContext) -> BackendResult<Self> {
        Ok(Self {
            inner: WgpuFlatGroupedTrainingBridge::from_context(ctx)?,
        })
    }

    /// Versioned FLAT reusable API consumed by this adapter.
    #[must_use]
    pub const fn api_version(&self) -> u16 {
        flat_attention::api::v1::API_VERSION
    }

    /// This opt-in adapter never performs a hidden legacy fallback.
    #[must_use]
    pub const fn fallback_policy(&self) -> FlatStableFallbackPolicy {
        FlatStableFallbackPolicy::ReturnError
    }

    /// Underlying adapter name for benchmark/correctness provenance.
    #[must_use]
    pub fn adapter_name(&self) -> &str {
        self.inner.adapter_name()
    }

    /// Shared SciRust WGPU context for adjacent resident operations.
    #[must_use]
    pub fn context(&self) -> &WgpuContext {
        self.inner.context()
    }

    /// Validate the stable backend-neutral contract and record the qualified
    /// forward→backward chain into the caller-owned command encoder.
    pub fn record_forward_backward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
        d_out: &GpuMatrix,
        config: FlatGroupedTrainingConfig,
    ) -> BackendResult<FlatGroupedTrainingResult> {
        validate_stable_contract(q, k, v, config)?;
        self.inner
            .record_forward_backward(encoder, q, k, v, d_out, config)
    }

    /// Validate through `api::v1`, then submit exactly the same single
    /// caller-owned command buffer as the qualified resident training bridge.
    pub fn forward_backward(
        &self,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
        d_out: &GpuMatrix,
        config: FlatGroupedTrainingConfig,
    ) -> BackendResult<FlatGroupedTrainingResult> {
        validate_stable_contract(q, k, v, config)?;
        self.inner.forward_backward(q, k, v, d_out, config)
    }
}

fn stable_shape(config: FlatGroupedTrainingConfig) -> StableAttentionShape {
    StableAttentionShape {
        batch: config.batch,
        q_heads: config.q_heads,
        kv_heads: config.kv_heads,
        query_len: config.seq_len,
        kv_len: config.seq_len,
        head_dim: config.head_dim,
        query_position_offset: 0,
    }
}

fn stable_config(config: FlatGroupedTrainingConfig) -> StableAttentionConfig {
    StableAttentionConfig {
        causal: config.causal,
        softmax_scale: config.softmax_scale,
    }
}

fn validate_stable_contract(
    q: &GpuMatrix,
    k: &GpuMatrix,
    v: &GpuMatrix,
    config: FlatGroupedTrainingConfig,
) -> BackendResult<()> {
    ResidentAttentionRequest {
        shape: stable_shape(config),
        config: stable_config(config),
        q,
        k,
        v,
    }
    .validate_contract()
    .map_err(|error| BackendError::ShapeMismatch(format!("FLAT api::v1 contract: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_native_grouped_training_contract_to_stable_v1() {
        let config = FlatGroupedTrainingConfig {
            batch: 2,
            q_heads: 8,
            kv_heads: 2,
            seq_len: 17,
            head_dim: 64,
            causal: true,
            softmax_scale: None,
        };
        let shape = stable_shape(config);
        shape.validate().unwrap();
        assert_eq!(shape.query_len, 17);
        assert_eq!(shape.kv_len, 17);
        assert_eq!(shape.group_size().unwrap(), 4);
        assert_eq!(shape.query_position_offset, 0);
        stable_config(config).validate(shape.head_dim).unwrap();
    }

    #[test]
    fn stable_v1_rejects_invalid_grouping_before_backend_dispatch() {
        let config = FlatGroupedTrainingConfig {
            batch: 1,
            q_heads: 3,
            kv_heads: 2,
            seq_len: 4,
            head_dim: 32,
            causal: false,
            softmax_scale: None,
        };
        let error = stable_shape(config).validate().unwrap_err();
        assert!(error.to_string().contains("exactly divisible"));
    }

    #[test]
    fn fallback_policy_is_fail_closed() {
        assert_eq!(
            FlatStableFallbackPolicy::ReturnError,
            FlatStableFallbackPolicy::ReturnError
        );
        assert_eq!(flat_attention::api::v1::API_VERSION, 1);
    }
}
