//! Integrated Phase 13 Elastic Latent KV inference runtime.
//!
//! This session layer combines Phase 9 planning, Phase 10 basis versions,
//! Phase 11 lifecycle metadata, Phase 12 projection kernels, and the validated
//! Phase 8 residual-latent attention backend.

#![allow(clippy::needless_range_loop)]

use crate::autodiff::reverse::Tensor;
use crate::nn::adaptive_latent_kv::{
    AdaptiveKvPlan, AdaptiveKvPolicyConfig, AdaptiveKvPolicyError, AdaptiveQualityProfile,
    select_adaptive_plan,
};
use crate::nn::adaptive_latent_kv_backend::{
    AdaptiveLatentBackendError, AdaptiveResidualLatentBackend,
};
use crate::nn::kv_backend::AttentionBackend;
use crate::nn::latent_kv_kernels::{LatentKernelDispatch, LatentKernelKind};
use crate::nn::latent_kv_lifecycle::{
    CacheTemperature, LatentKvLifecycle, LifecycleAction, LifecycleConfig, LifecycleError,
};
use crate::nn::transformer::attention::MultiHeadAttention;
use core::fmt;

/// Per-head calibration inputs frozen for one decode session.
#[derive(Clone, Copy)]
pub struct HeadCalibration<'a> {
    pub full_key_basis: &'a [f32],
    pub full_value_basis: &'a [f32],
    pub quality: AdaptiveQualityProfile<'a>,
    pub basis_version: u32,
}

/// Integrated runtime configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElasticLatentRuntimeConfig {
    pub capacity_tokens: usize,
    pub minimum_rank: usize,
    pub maximum_rank: usize,
    pub maximum_residual_slots: usize,
    pub persistent_budget_bytes: usize,
    pub allocated_ceiling_bytes: usize,
    pub lifecycle: LifecycleConfig,
    pub kernel: LatentKernelKind,
}

/// Aggregate runtime telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ElasticLatentTelemetry {
    pub steps: usize,
    pub planned_persistent_bytes: usize,
    pub allocated_bytes: usize,
    pub worst_quality_bps: u16,
    pub last_lifecycle_transitions: usize,
}

/// Phase 13 construction or decode errors.
#[derive(Debug)]
pub enum ElasticLatentRuntimeError {
    ZeroHeads,
    HeadCount { expected: usize, actual: usize },
    LifecycleCapacityMismatch,
    TokenLength { expected: usize, actual: usize },
    CapacityExhausted { head: usize, capacity: usize },
    AllocationCeiling { ceiling: usize, actual: usize },
    Policy(AdaptiveKvPolicyError),
    Backend(AdaptiveLatentBackendError),
    Lifecycle(LifecycleError),
}

impl fmt::Display for ElasticLatentRuntimeError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::ZeroHeads => write!(output, "attention must contain at least one head"),
            Self::HeadCount { expected, actual } =>
            {
                write!(
                    output,
                    "head calibration mismatch: expected {expected}, got {actual}"
                )
            },
            Self::LifecycleCapacityMismatch =>
            {
                write!(output, "lifecycle capacity must equal runtime capacity")
            },
            Self::TokenLength { expected, actual } =>
            {
                write!(
                    output,
                    "token length mismatch: expected {expected}, got {actual}"
                )
            },
            Self::CapacityExhausted { head, capacity } =>
            {
                write!(output, "head {head} reached session capacity {capacity}")
            },
            Self::AllocationCeiling { ceiling, actual } =>
            {
                write!(
                    output,
                    "latent runtime allocation {actual} exceeds ceiling {ceiling}"
                )
            },
            Self::Policy(error) => write!(output, "{error}"),
            Self::Backend(error) => write!(output, "{error}"),
            Self::Lifecycle(error) => write!(output, "{error}"),
        }
    }
}

impl std::error::Error for ElasticLatentRuntimeError {}

impl From<AdaptiveKvPolicyError> for ElasticLatentRuntimeError {
    fn from(error: AdaptiveKvPolicyError) -> Self {
        Self::Policy(error)
    }
}

impl From<AdaptiveLatentBackendError> for ElasticLatentRuntimeError {
    fn from(error: AdaptiveLatentBackendError) -> Self {
        Self::Backend(error)
    }
}

impl From<LifecycleError> for ElasticLatentRuntimeError {
    fn from(error: LifecycleError) -> Self {
        Self::Lifecycle(error)
    }
}

/// Session-scoped bounded Elastic Latent KV decoder.
pub struct ElasticLatentDecodeRuntime {
    config: ElasticLatentRuntimeConfig,
    backends: Vec<Box<dyn AttentionBackend>>,
    plans: Vec<AdaptiveKvPlan>,
    basis_versions: Vec<u32>,
    lifecycles: Vec<LatentKvLifecycle>,
    lifecycle_scratch: Vec<LifecycleAction>,
    kernel: LatentKernelDispatch,
    planned_persistent_bytes: usize,
    allocated_bytes: usize,
    worst_quality_bps: u16,
    steps: usize,
    last_lifecycle_transitions: usize,
}

impl ElasticLatentDecodeRuntime {
    /// Selects all per-head plans and constructs a bounded decode session.
    pub fn new(
        attention: &MultiHeadAttention,
        config: ElasticLatentRuntimeConfig,
        calibrations: &[HeadCalibration<'_>],
    ) -> Result<Self, ElasticLatentRuntimeError> {
        if attention.n_heads == 0
        {
            return Err(ElasticLatentRuntimeError::ZeroHeads);
        }
        if calibrations.len() != attention.n_heads
        {
            return Err(ElasticLatentRuntimeError::HeadCount {
                expected: attention.n_heads,
                actual: calibrations.len(),
            });
        }
        if config.lifecycle.capacity_tokens != config.capacity_tokens
        {
            return Err(ElasticLatentRuntimeError::LifecycleCapacityMismatch);
        }

        let base_budget = config.persistent_budget_bytes / attention.n_heads;
        let remainder = config.persistent_budget_bytes % attention.n_heads;
        let mut backends: Vec<Box<dyn AttentionBackend>> = Vec::with_capacity(attention.n_heads);
        let mut plans = Vec::with_capacity(attention.n_heads);
        let mut basis_versions = Vec::with_capacity(attention.n_heads);
        let mut lifecycles = Vec::with_capacity(attention.n_heads);
        let mut planned_persistent_bytes = 0_usize;
        let mut allocated_bytes = 0_usize;
        let mut worst_quality_bps = 10_000_u16;

        for (head, calibration) in calibrations.iter().copied().enumerate()
        {
            let head_budget = base_budget + if head < remainder { 1 } else { 0 };
            let plan = select_adaptive_plan(
                AdaptiveKvPolicyConfig {
                    capacity_tokens: config.capacity_tokens,
                    dimension: attention.d_head,
                    minimum_rank: config.minimum_rank,
                    maximum_rank: config.maximum_rank,
                    maximum_residual_slots: config.maximum_residual_slots,
                    budget_bytes: head_budget,
                },
                calibration.quality,
            )?;
            let backend = AdaptiveResidualLatentBackend::new(
                config.capacity_tokens,
                attention.d_head,
                calibration.full_key_basis,
                calibration.full_value_basis,
                plan,
            )?;
            planned_persistent_bytes =
                planned_persistent_bytes.saturating_add(plan.persistent_bytes);
            allocated_bytes = allocated_bytes.saturating_add(backend.packed_bytes());
            worst_quality_bps = worst_quality_bps.min(plan.worst_quality_bps);
            backends.push(Box::new(backend));
            plans.push(plan);
            basis_versions.push(calibration.basis_version);
            lifecycles.push(LatentKvLifecycle::new(config.lifecycle)?);
        }

        if allocated_bytes > config.allocated_ceiling_bytes
        {
            return Err(ElasticLatentRuntimeError::AllocationCeiling {
                ceiling: config.allocated_ceiling_bytes,
                actual: allocated_bytes,
            });
        }

        let placeholder = LifecycleAction {
            position: 0,
            basis_version: 0,
            from: CacheTemperature::Hot,
            to: CacheTemperature::Hot,
            target: config.lifecycle.hot,
        };
        Ok(Self {
            config,
            backends,
            plans,
            basis_versions,
            lifecycles,
            lifecycle_scratch: vec![placeholder; config.capacity_tokens],
            kernel: LatentKernelDispatch::new(config.kernel),
            planned_persistent_bytes,
            allocated_bytes,
            worst_quality_bps,
            steps: 0,
            last_lifecycle_transitions: 0,
        })
    }

    #[must_use]
    pub fn plans(&self) -> &[AdaptiveKvPlan] {
        &self.plans
    }

    #[must_use]
    pub const fn telemetry(&self) -> ElasticLatentTelemetry {
        ElasticLatentTelemetry {
            steps: self.steps,
            planned_persistent_bytes: self.planned_persistent_bytes,
            allocated_bytes: self.allocated_bytes,
            worst_quality_bps: self.worst_quality_bps,
            last_lifecycle_transitions: self.last_lifecycle_transitions,
        }
    }

    /// Executes one bounded numeric decode step.
    pub fn decode_step(
        &mut self,
        attention: &MultiHeadAttention,
        token: &[f32],
    ) -> Result<Vec<f32>, ElasticLatentRuntimeError> {
        if token.len() != attention.d_model
        {
            return Err(ElasticLatentRuntimeError::TokenLength {
                expected: attention.d_model,
                actual: token.len(),
            });
        }
        for (head, backend) in self.backends.iter().enumerate()
        {
            if backend.len() >= self.config.capacity_tokens
            {
                return Err(ElasticLatentRuntimeError::CapacityExhausted {
                    head,
                    capacity: self.config.capacity_tokens,
                });
            }
        }

        let q = linear_apply_kernel(
            &attention.w_q.weight,
            &attention.w_q.bias,
            token,
            self.kernel,
        );
        let k = linear_apply_kernel(
            &attention.w_k.weight,
            &attention.w_k.bias,
            token,
            self.kernel,
        );
        let v = linear_apply_kernel(
            &attention.w_v.weight,
            &attention.w_v.bias,
            token,
            self.kernel,
        );
        let mut context = vec![0.0_f32; attention.d_model];
        let mut transitions = 0_usize;
        for head in 0..attention.n_heads
        {
            let start = head * attention.d_head;
            let end = start + attention.d_head;
            self.backends[head].append(&k[start..end], &v[start..end]);
            let output = self.backends[head].attention(&q[start..end]);
            context[start..end].copy_from_slice(&output);
            let admission = self.lifecycles[head].admit(self.basis_versions[head]);
            debug_assert!(admission.evicted.is_none());
            transitions = transitions
                .saturating_add(self.lifecycles[head].rebalance_into(&mut self.lifecycle_scratch));
        }
        self.steps = self.steps.saturating_add(1);
        self.last_lifecycle_transitions = transitions;
        Ok(linear_apply_kernel(
            &attention.w_o.weight,
            &attention.w_o.bias,
            &context,
            self.kernel,
        ))
    }
}

fn linear_apply_kernel(
    weight: &Tensor,
    bias: &Tensor,
    input: &[f32],
    kernel: LatentKernelDispatch,
) -> Vec<f32> {
    debug_assert_eq!(input.len(), weight.rows);
    debug_assert_eq!(bias.data.len(), weight.cols);
    let mut output = vec![0.0_f32; weight.cols];
    for column in 0..weight.cols
    {
        output[column] = bias.data[column]
            + kernel.dot_strided(&weight.data, weight.rows, weight.cols, column, input);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{
        ElasticLatentDecodeRuntime, ElasticLatentRuntimeConfig, ElasticLatentRuntimeError,
        HeadCalibration,
    };
    use crate::nn::adaptive_latent_kv::AdaptiveQualityProfile;
    use crate::nn::init::{KaimingNormal, Zeros};
    use crate::nn::kv_backend::{AttentionBackend, PlainKvCache, decode_step};
    use crate::nn::latent_kv_cache::LatentStorageFormat;
    use crate::nn::latent_kv_kernels::LatentKernelKind;
    use crate::nn::latent_kv_lifecycle::{CompressionTier, LifecycleConfig};
    use crate::nn::rng::PcgEngine;
    use crate::nn::transformer::attention::MultiHeadAttention;

    fn identity(dimension: usize) -> Vec<f32> {
        let mut basis = vec![0.0; dimension * dimension];
        for index in 0..dimension
        {
            basis[index * dimension + index] = 1.0;
        }
        basis
    }

    fn attention() -> MultiHeadAttention {
        let mut rng = PcgEngine::new(0);
        MultiHeadAttention::new(8, 2, false, &KaimingNormal, &Zeros, &mut rng)
    }

    fn lifecycle(capacity: usize) -> LifecycleConfig {
        let tier = CompressionTier {
            coefficient_format: LatentStorageFormat::F32,
            residual_format: LatentStorageFormat::F32,
            maximum_residual_slots: 0,
            rank_divisor: 1,
        };
        let hot_tokens = capacity.min(2);
        let warm_tokens = capacity.saturating_sub(hot_tokens).min(2);
        LifecycleConfig {
            capacity_tokens: capacity,
            hot_tokens,
            warm_tokens,
            hot: tier,
            warm: tier,
            cold: tier,
        }
    }

    #[test]
    fn integrated_runtime_matches_plain_full_rank_decode() {
        static QUALITY: [u16; 4] = [2_500, 5_000, 7_500, 10_000];
        static RESIDUAL: [u16; 1] = [0];
        let attention = attention();
        let basis = identity(attention.d_head);
        let profile = AdaptiveQualityProfile {
            key_rank_quality_bps: &QUALITY,
            value_rank_quality_bps: &QUALITY,
            key_residual_gain_bps: &RESIDUAL,
            value_residual_gain_bps: &RESIDUAL,
        };
        let calibrations = [
            HeadCalibration {
                full_key_basis: &basis,
                full_value_basis: &basis,
                quality: profile,
                basis_version: 3,
            },
            HeadCalibration {
                full_key_basis: &basis,
                full_value_basis: &basis,
                quality: profile,
                basis_version: 3,
            },
        ];
        let capacity = 8;
        let mut runtime = ElasticLatentDecodeRuntime::new(
            &attention,
            ElasticLatentRuntimeConfig {
                capacity_tokens: capacity,
                minimum_rank: 4,
                maximum_rank: 4,
                maximum_residual_slots: 0,
                persistent_budget_bytes: 4_096,
                allocated_ceiling_bytes: 16_384,
                lifecycle: lifecycle(capacity),
                kernel: LatentKernelKind::Scalar,
            },
            &calibrations,
        )
        .unwrap();
        let mut plain: Vec<Box<dyn AttentionBackend>> = (0..attention.n_heads)
            .map(|_| Box::new(PlainKvCache::new(attention.d_head)) as Box<dyn AttentionBackend>)
            .collect();
        for step in 0..4
        {
            let token: Vec<f32> = (0..attention.d_model)
                .map(|index| (step * attention.d_model + index) as f32 * 0.01 - 0.2)
                .collect();
            let expected = decode_step(&attention, &token, &mut plain);
            let actual = runtime.decode_step(&attention, &token).unwrap();
            for (left, right) in expected.iter().zip(&actual)
            {
                assert!((left - right).abs() <= 3.0e-6);
            }
        }
        assert_eq!(runtime.telemetry().steps, 4);
        assert!(runtime.telemetry().planned_persistent_bytes <= 4_096);
    }

    #[test]
    fn runtime_stops_before_backend_capacity_panics() {
        static QUALITY: [u16; 4] = [2_500, 5_000, 7_500, 10_000];
        static RESIDUAL: [u16; 1] = [0];
        let attention = attention();
        let basis = identity(attention.d_head);
        let profile = AdaptiveQualityProfile {
            key_rank_quality_bps: &QUALITY,
            value_rank_quality_bps: &QUALITY,
            key_residual_gain_bps: &RESIDUAL,
            value_residual_gain_bps: &RESIDUAL,
        };
        let calibrations = [HeadCalibration {
            full_key_basis: &basis,
            full_value_basis: &basis,
            quality: profile,
            basis_version: 0,
        }; 2];
        let mut runtime = ElasticLatentDecodeRuntime::new(
            &attention,
            ElasticLatentRuntimeConfig {
                capacity_tokens: 2,
                minimum_rank: 4,
                maximum_rank: 4,
                maximum_residual_slots: 0,
                persistent_budget_bytes: 4_096,
                allocated_ceiling_bytes: 16_384,
                lifecycle: lifecycle(2),
                kernel: LatentKernelKind::Block4,
            },
            &calibrations,
        )
        .unwrap();
        let token = vec![0.1; attention.d_model];
        runtime.decode_step(&attention, &token).unwrap();
        runtime.decode_step(&attention, &token).unwrap();
        assert!(matches!(
            runtime.decode_step(&attention, &token),
            Err(ElasticLatentRuntimeError::CapacityExhausted { .. })
        ));
    }
}
