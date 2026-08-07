//! Integrated Phase 13 Elastic Latent KV inference runtime.
//!
//! This session layer combines Phase 9 planning, Phase 10 basis versions,
//! Phase 11 material HOT/WARM/COLD lifecycle storage, Phase 12 projection
//! kernels, and the validated Phase 8 sparse-residual codec.

#![allow(clippy::needless_range_loop)]

use crate::autodiff::reverse::Tensor;
use crate::nn::adaptive_latent_kv::{
    AdaptiveKvPlan, AdaptiveKvPolicyConfig, AdaptiveKvPolicyError, AdaptiveQualityProfile,
    select_adaptive_plan,
};
use crate::nn::adaptive_latent_kv_backend::AdaptiveLatentBackendError;
use crate::nn::kv_backend::AttentionBackend;
use crate::nn::latent_kv_cache::LatentStorageFormat;
use crate::nn::latent_kv_kernels::{LatentKernelDispatch, LatentKernelKind};
use crate::nn::latent_kv_lifecycle::{
    CacheTemperature, CompressionTier, LatentKvLifecycle, LifecycleAction, LifecycleConfig,
    LifecycleError,
};
use crate::nn::tiered_latent_kv_backend::{
    TieredLatentBackendError, TieredResidualLatentBackend,
};
use crate::nn::transformer::attention::MultiHeadAttention;
use core::fmt;

const QUALITY_SCALE: u32 = 10_000;

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
    pub last_lifecycle_evictions: usize,
    pub total_lifecycle_evictions: usize,
}

/// Phase 13 construction or decode errors.
#[derive(Debug)]
pub enum ElasticLatentRuntimeError {
    ZeroHeads,
    HeadCount { expected: usize, actual: usize },
    LifecycleCapacityMismatch,
    TokenLength { expected: usize, actual: usize },
    /// Retained for source compatibility with pre-sliding-window callers.
    CapacityExhausted { head: usize, capacity: usize },
    AllocationCeiling { ceiling: usize, actual: usize },
    Policy(AdaptiveKvPolicyError),
    /// Retained for source compatibility with the original Phase 13 backend.
    Backend(AdaptiveLatentBackendError),
    TieredBackend(TieredLatentBackendError),
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
            Self::TieredBackend(error) => write!(output, "{error}"),
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

impl From<TieredLatentBackendError> for ElasticLatentRuntimeError {
    fn from(error: TieredLatentBackendError) -> Self {
        Self::TieredBackend(error)
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
    last_lifecycle_evictions: usize,
    total_lifecycle_evictions: usize,
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
            let (plan, backend) = select_budgeted_tiered_backend(
                attention.d_head,
                config,
                calibration,
                head_budget,
            )?;
            planned_persistent_bytes = planned_persistent_bytes
                .saturating_add(backend.planned_persistent_bytes());
            allocated_bytes = allocated_bytes.saturating_add(backend.packed_bytes());
            worst_quality_bps = worst_quality_bps.min(lifecycle_worst_quality(
                calibration.quality,
                plan,
                config.lifecycle,
                attention.d_head,
            ));
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
            last_lifecycle_evictions: 0,
            total_lifecycle_evictions: 0,
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
            last_lifecycle_evictions: self.last_lifecycle_evictions,
            total_lifecycle_evictions: self.total_lifecycle_evictions,
        }
    }

    /// Executes one bounded numeric decode step with sliding-window reuse.
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
        let mut evictions = 0_usize;
        for head in 0..attention.n_heads
        {
            let start = head * attention.d_head;
            let end = start + attention.d_head;
            self.backends[head].append(&k[start..end], &v[start..end]);
            let output = self.backends[head].attention(&q[start..end]);
            context[start..end].copy_from_slice(&output);
            let admission = self.lifecycles[head].admit(self.basis_versions[head]);
            evictions = evictions.saturating_add(usize::from(admission.evicted.is_some()));
            transitions = transitions
                .saturating_add(self.lifecycles[head].rebalance_into(&mut self.lifecycle_scratch));
        }
        self.steps = self.steps.saturating_add(1);
        self.last_lifecycle_transitions = transitions;
        self.last_lifecycle_evictions = evictions;
        self.total_lifecycle_evictions = self.total_lifecycle_evictions.saturating_add(evictions);
        Ok(linear_apply_kernel(
            &attention.w_o.weight,
            &attention.w_o.bias,
            &context,
            self.kernel,
        ))
    }
}

fn select_budgeted_tiered_backend(
    dimension: usize,
    config: ElasticLatentRuntimeConfig,
    calibration: HeadCalibration<'_>,
    head_budget: usize,
) -> Result<(AdaptiveKvPlan, TieredResidualLatentBackend), ElasticLatentRuntimeError> {
    let mut planner_budget = head_budget;
    loop
    {
        let plan = select_adaptive_plan(
            AdaptiveKvPolicyConfig {
                capacity_tokens: config.capacity_tokens,
                dimension,
                minimum_rank: config.minimum_rank,
                maximum_rank: config.maximum_rank,
                maximum_residual_slots: config.maximum_residual_slots,
                budget_bytes: planner_budget,
            },
            calibration.quality,
        )?;
        let backend = TieredResidualLatentBackend::new(
            config.capacity_tokens,
            dimension,
            calibration.full_key_basis,
            calibration.full_value_basis,
            plan,
            config.lifecycle,
        )?;
        let actual = backend.planned_persistent_bytes();
        if actual <= head_budget
        {
            return Ok((plan, backend));
        }
        let excess = actual - head_budget;
        if planner_budget <= excess
        {
            return Err(AdaptiveKvPolicyError::BudgetInfeasible.into());
        }
        planner_budget -= excess.max(1);
    }
}

fn lifecycle_worst_quality(
    profile: AdaptiveQualityProfile<'_>,
    plan: AdaptiveKvPlan,
    lifecycle: LifecycleConfig,
    dimension: usize,
) -> u16 {
    let cold_tokens = lifecycle
        .capacity_tokens
        .saturating_sub(lifecycle.hot_tokens.saturating_add(lifecycle.warm_tokens));
    let mut worst = 10_000_u16;
    for (capacity, tier) in [
        (lifecycle.hot_tokens, lifecycle.hot),
        (lifecycle.warm_tokens, lifecycle.warm),
        (cold_tokens, lifecycle.cold),
    ]
    {
        if capacity == 0
        {
            continue;
        }
        let key_rank = tier_rank(plan.key.rank, tier.rank_divisor);
        let value_rank = tier_rank(plan.value.rank, tier.rank_divisor);
        let key_slots = plan
            .key
            .residual_slots
            .min(tier.maximum_residual_slots)
            .min(dimension);
        let value_slots = plan
            .value
            .residual_slots
            .min(tier.maximum_residual_slots)
            .min(dimension);
        let key_quality = channel_quality(
            profile.key_rank_quality_bps,
            profile.key_residual_gain_bps,
            key_rank,
            key_slots,
            tier,
        );
        let value_quality = channel_quality(
            profile.value_rank_quality_bps,
            profile.value_residual_gain_bps,
            value_rank,
            value_slots,
            tier,
        );
        worst = worst.min(key_quality.min(value_quality));
    }
    worst
}

fn channel_quality(
    rank_quality: &[u16],
    residual_gain: &[u16],
    rank: usize,
    residual_slots: usize,
    tier: CompressionTier,
) -> u16 {
    let base = u32::from(rank_quality[rank - 1])
        .saturating_add(u32::from(residual_gain[residual_slots]))
        .min(QUALITY_SCALE);
    let residual_retention = if residual_slots == 0
    {
        QUALITY_SCALE
    }
    else
    {
        format_retention_bps(tier.residual_format)
    };
    let retention = format_retention_bps(tier.coefficient_format).min(residual_retention);
    ((base * retention) / QUALITY_SCALE) as u16
}

fn tier_rank(rank: usize, divisor: usize) -> usize {
    (rank / divisor).max(1)
}

const fn format_retention_bps(format: LatentStorageFormat) -> u32 {
    match format
    {
        LatentStorageFormat::F32 => QUALITY_SCALE,
        LatentStorageFormat::Int8 => 9_975,
        LatentStorageFormat::Int4 => 9_700,
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
    use super::{ElasticLatentDecodeRuntime, ElasticLatentRuntimeConfig, HeadCalibration};
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

    fn calibrations<'a>(basis: &'a [f32]) -> [HeadCalibration<'a>; 2] {
        static QUALITY: [u16; 4] = [2_500, 5_000, 7_500, 10_000];
        static RESIDUAL: [u16; 1] = [0];
        let profile = AdaptiveQualityProfile {
            key_rank_quality_bps: &QUALITY,
            value_rank_quality_bps: &QUALITY,
            key_residual_gain_bps: &RESIDUAL,
            value_residual_gain_bps: &RESIDUAL,
        };
        [HeadCalibration {
            full_key_basis: basis,
            full_value_basis: basis,
            quality: profile,
            basis_version: 3,
        }; 2]
    }

    #[test]
    fn integrated_runtime_matches_plain_full_rank_decode() {
        let attention = attention();
        let basis = identity(attention.d_head);
        let calibrations = calibrations(&basis);
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
        assert_eq!(runtime.telemetry().total_lifecycle_evictions, 0);
    }

    #[test]
    fn runtime_slides_beyond_capacity_without_allocation_growth() {
        let attention = attention();
        let basis = identity(attention.d_head);
        let calibrations = calibrations(&basis);
        let capacity = 2;
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
                kernel: LatentKernelKind::Block4,
            },
            &calibrations,
        )
        .unwrap();
        let allocated = runtime.telemetry().allocated_bytes;
        for step in 0..7
        {
            let token: Vec<f32> = (0..attention.d_model)
                .map(|index| (step * attention.d_model + index) as f32 * 0.013 - 0.25)
                .collect();
            runtime.decode_step(&attention, &token).unwrap();
            assert_eq!(runtime.telemetry().allocated_bytes, allocated);
        }
        assert_eq!(runtime.telemetry().steps, 7);
        assert_eq!(runtime.telemetry().last_lifecycle_evictions, 2);
        assert_eq!(runtime.telemetry().total_lifecycle_evictions, 10);
    }
}
