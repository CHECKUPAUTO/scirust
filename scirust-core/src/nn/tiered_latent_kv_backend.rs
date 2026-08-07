//! Material HOT/WARM/COLD sparse-residual KV backend for Phase 11.
//!
//! Unlike the Phase 11 metadata controller alone, this backend owns three
//! independently encoded fixed-capacity stores. Tokens are deterministically
//! reconstructed into caller-preallocated migration scratch when they age from
//! HOT to WARM or WARM to COLD, then re-encoded with the target tier's rank,
//! formats, and residual-slot cap. When the COLD tier is full, its oldest row is
//! evicted before reuse. No persistent buffer grows after construction.

use crate::nn::adaptive_latent_kv::{AdaptiveKvPlan, estimate_channel_bytes};
use crate::nn::kv_backend::AttentionBackend;
use crate::nn::latent_kv_lifecycle::{CacheTemperature, CompressionTier, LifecycleConfig};
use crate::nn::residual_latent_kv_cache::{
    ResidualLatentCacheError, ResidualQuantizedLatentKvCache, SparseResidualConfig,
};
use core::fmt;
use std::cell::RefCell;

/// Errors returned while materializing lifecycle-aware latent stores.
#[derive(Debug)]
pub enum TieredLatentBackendError {
    /// A full row-major basis was not shaped `[dimension, dimension]`.
    BasisLength {
        /// Human-readable basis name.
        name: &'static str,
        /// Expected full basis element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// Lifecycle windows did not match the requested resident capacity.
    InvalidWindows {
        /// Total resident capacity.
        capacity: usize,
        /// Configured HOT tokens.
        hot: usize,
        /// Configured WARM tokens.
        warm: usize,
    },
    /// A lifecycle rank divisor was zero.
    ZeroRankDivisor(CacheTemperature),
    /// A Phase 8 cache construction or row operation failed.
    Residual(ResidualLatentCacheError),
}

impl fmt::Display for TieredLatentBackendError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::BasisLength {
                name,
                expected,
                actual,
            } => write!(
                output,
                "{name} full basis length mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidWindows {
                capacity,
                hot,
                warm,
            } => write!(
                output,
                "lifecycle windows exceed capacity: hot={hot}, warm={warm}, capacity={capacity}"
            ),
            Self::ZeroRankDivisor(temperature) =>
            {
                write!(output, "{temperature:?} lifecycle rank divisor must be non-zero")
            },
            Self::Residual(error) => write!(output, "{error}"),
        }
    }
}

impl std::error::Error for TieredLatentBackendError {}

impl From<ResidualLatentCacheError> for TieredLatentBackendError {
    fn from(error: ResidualLatentCacheError) -> Self {
        Self::Residual(error)
    }
}

#[derive(Debug)]
struct TieredAttentionScratch {
    scores: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
}

impl TieredAttentionScratch {
    fn new(capacity: usize, dimension: usize) -> Self {
        Self {
            scores: vec![0.0; capacity],
            key: vec![0.0; dimension],
            value: vec![0.0; dimension],
        }
    }

    fn allocated_bytes(&self) -> usize {
        self.scores
            .capacity()
            .saturating_add(self.key.capacity())
            .saturating_add(self.value.capacity())
            .saturating_mul(core::mem::size_of::<f32>())
    }
}

/// Fixed-allocation cache telemetry for one lifecycle-aware head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TieredLatentTelemetry {
    pub hot_tokens: usize,
    pub warm_tokens: usize,
    pub cold_tokens: usize,
    pub evictions: usize,
    pub planned_persistent_bytes: usize,
    pub allocated_bytes: usize,
}

/// Phase 8 storage physically partitioned by the Phase 11 lifecycle windows.
pub struct TieredResidualLatentBackend {
    dimension: usize,
    capacity: usize,
    hot: Option<ResidualQuantizedLatentKvCache>,
    warm: Option<ResidualQuantizedLatentKvCache>,
    cold: Option<ResidualQuantizedLatentKvCache>,
    attention_scratch: RefCell<TieredAttentionScratch>,
    migration_key: Vec<f32>,
    migration_value: Vec<f32>,
    planned_persistent_bytes: usize,
    evictions: usize,
}

impl TieredResidualLatentBackend {
    /// Materializes all non-empty lifecycle tiers up-front.
    pub fn new(
        capacity: usize,
        dimension: usize,
        full_key_basis: &[f32],
        full_value_basis: &[f32],
        plan: AdaptiveKvPlan,
        lifecycle: LifecycleConfig,
    ) -> Result<Self, TieredLatentBackendError> {
        let windows = lifecycle.hot_tokens.saturating_add(lifecycle.warm_tokens);
        if lifecycle.capacity_tokens != capacity || windows > capacity
        {
            return Err(TieredLatentBackendError::InvalidWindows {
                capacity,
                hot: lifecycle.hot_tokens,
                warm: lifecycle.warm_tokens,
            });
        }
        validate_tier(CacheTemperature::Hot, lifecycle.hot)?;
        validate_tier(CacheTemperature::Warm, lifecycle.warm)?;
        validate_tier(CacheTemperature::Cold, lifecycle.cold)?;
        validate_full_basis("key", full_key_basis, dimension)?;
        validate_full_basis("value", full_value_basis, dimension)?;

        let cold_tokens = capacity - windows;
        let hot = build_tier(
            lifecycle.hot_tokens,
            dimension,
            full_key_basis,
            full_value_basis,
            plan,
            lifecycle.hot,
        )?;
        let warm = build_tier(
            lifecycle.warm_tokens,
            dimension,
            full_key_basis,
            full_value_basis,
            plan,
            lifecycle.warm,
        )?;
        let cold = build_tier(
            cold_tokens,
            dimension,
            full_key_basis,
            full_value_basis,
            plan,
            lifecycle.cold,
        )?;
        let planned_persistent_bytes = planned_tier_bytes(
            lifecycle.hot_tokens,
            dimension,
            plan,
            lifecycle.hot,
        )
        .saturating_add(planned_tier_bytes(
            lifecycle.warm_tokens,
            dimension,
            plan,
            lifecycle.warm,
        ))
        .saturating_add(planned_tier_bytes(
            cold_tokens,
            dimension,
            plan,
            lifecycle.cold,
        ));

        Ok(Self {
            dimension,
            capacity,
            hot,
            warm,
            cold,
            attention_scratch: RefCell::new(TieredAttentionScratch::new(capacity, dimension)),
            migration_key: vec![0.0; dimension],
            migration_value: vec![0.0; dimension],
            planned_persistent_bytes,
            evictions: 0,
        })
    }

    /// Returns the concrete cache for deterministic tier inspection.
    #[must_use]
    pub fn tier(&self, temperature: CacheTemperature) -> Option<&ResidualQuantizedLatentKvCache> {
        match temperature
        {
            CacheTemperature::Hot => self.hot.as_ref(),
            CacheTemperature::Warm => self.warm.as_ref(),
            CacheTemperature::Cold => self.cold.as_ref(),
        }
    }

    /// Returns the logical byte estimate used for strict persistent-budget checks.
    #[must_use]
    pub const fn planned_persistent_bytes(&self) -> usize {
        self.planned_persistent_bytes
    }

    /// Returns fixed-allocation and occupancy telemetry.
    #[must_use]
    pub fn telemetry(&self) -> TieredLatentTelemetry {
        TieredLatentTelemetry {
            hot_tokens: tier_len(&self.hot),
            warm_tokens: tier_len(&self.warm),
            cold_tokens: tier_len(&self.cold),
            evictions: self.evictions,
            planned_persistent_bytes: self.planned_persistent_bytes,
            allocated_bytes: self.allocated_bytes(),
        }
    }

    fn allocated_bytes(&self) -> usize {
        tier_allocated(&self.hot)
            .saturating_add(tier_allocated(&self.warm))
            .saturating_add(tier_allocated(&self.cold))
            .saturating_add(self.attention_scratch.borrow().allocated_bytes())
            .saturating_add(
                self.migration_key
                    .capacity()
                    .saturating_add(self.migration_value.capacity())
                    .saturating_mul(core::mem::size_of::<f32>()),
            )
    }

    fn append_inner(&mut self, key: &[f32], value: &[f32]) {
        if self.hot.is_some()
        {
            self.ensure_hot_room();
            self.hot
                .as_mut()
                .expect("HOT tier presence already checked")
                .append(key, value)
                .expect("validated HOT append must fit");
        }
        else if self.warm.is_some()
        {
            self.ensure_warm_room();
            self.warm
                .as_mut()
                .expect("WARM tier presence already checked")
                .append(key, value)
                .expect("validated WARM append must fit");
        }
        else
        {
            self.ensure_cold_room();
            self.cold
                .as_mut()
                .expect("non-zero capacity requires a COLD tier")
                .append(key, value)
                .expect("validated COLD append must fit");
        }
    }

    fn ensure_hot_room(&mut self) {
        let full = self
            .hot
            .as_ref()
            .is_some_and(|cache| cache.len() == cache.capacity());
        if !full
        {
            return;
        }
        if self.warm.is_some()
        {
            self.ensure_warm_room();
        }
        else
        {
            self.ensure_cold_room();
        }
        self.hot
            .as_ref()
            .expect("HOT tier exists")
            .reconstruct_token_into(0, &mut self.migration_key, &mut self.migration_value)
            .expect("oldest HOT row must be resident");
        self.hot
            .as_mut()
            .expect("HOT tier exists")
            .remove_oldest()
            .expect("full HOT tier cannot be empty");
        if let Some(warm) = self.warm.as_mut()
        {
            warm.append(&self.migration_key, &self.migration_value)
                .expect("WARM room was established before migration");
        }
        else if let Some(cold) = self.cold.as_mut()
        {
            cold.append(&self.migration_key, &self.migration_value)
                .expect("COLD room was established before migration");
        }
        else
        {
            self.evictions = self.evictions.saturating_add(1);
        }
    }

    fn ensure_warm_room(&mut self) {
        let full = self
            .warm
            .as_ref()
            .is_some_and(|cache| cache.len() == cache.capacity());
        if !full
        {
            return;
        }
        self.ensure_cold_room();
        self.warm
            .as_ref()
            .expect("WARM tier exists")
            .reconstruct_token_into(0, &mut self.migration_key, &mut self.migration_value)
            .expect("oldest WARM row must be resident");
        self.warm
            .as_mut()
            .expect("WARM tier exists")
            .remove_oldest()
            .expect("full WARM tier cannot be empty");
        if let Some(cold) = self.cold.as_mut()
        {
            cold.append(&self.migration_key, &self.migration_value)
                .expect("COLD room was established before migration");
        }
        else
        {
            self.evictions = self.evictions.saturating_add(1);
        }
    }

    fn ensure_cold_room(&mut self) {
        let full = self
            .cold
            .as_ref()
            .is_some_and(|cache| cache.len() == cache.capacity());
        if full
        {
            self.cold
                .as_mut()
                .expect("COLD tier exists")
                .remove_oldest()
                .expect("full COLD tier cannot be empty");
            self.evictions = self.evictions.saturating_add(1);
        }
    }

    fn append_scores(
        cache: Option<&ResidualQuantizedLatentKvCache>,
        query: &[f32],
        scores: &mut [f32],
        key: &mut [f32],
        value: &mut [f32],
        cursor: &mut usize,
        scale: f32,
    ) {
        let Some(cache) = cache
        else
        {
            return;
        };
        for row in 0..cache.len()
        {
            cache
                .reconstruct_token_into(row, key, value)
                .expect("resident lifecycle row must reconstruct");
            let mut score = 0.0_f32;
            for (left, right) in query.iter().zip(key.iter())
            {
                score += left * right;
            }
            scores[*cursor] = score * scale;
            *cursor += 1;
        }
    }

    fn accumulate_values(
        cache: Option<&ResidualQuantizedLatentKvCache>,
        weights: &[f32],
        key: &mut [f32],
        value: &mut [f32],
        cursor: &mut usize,
        output: &mut [f32],
    ) {
        let Some(cache) = cache
        else
        {
            return;
        };
        for row in 0..cache.len()
        {
            cache
                .reconstruct_token_into(row, key, value)
                .expect("resident lifecycle row must reconstruct");
            let weight = weights[*cursor];
            for (output_scalar, value_scalar) in output.iter_mut().zip(value.iter())
            {
                *output_scalar += weight * value_scalar;
            }
            *cursor += 1;
        }
    }
}

impl AttentionBackend for TieredResidualLatentBackend {
    fn dim(&self) -> usize {
        self.dimension
    }

    fn append(&mut self, key: &[f32], value: &[f32]) {
        assert_eq!(key.len(), self.dimension);
        assert_eq!(value.len(), self.dimension);
        self.append_inner(key, value);
    }

    fn len(&self) -> usize {
        tier_len(&self.cold)
            .saturating_add(tier_len(&self.warm))
            .saturating_add(tier_len(&self.hot))
    }

    fn attention(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(query.len(), self.dimension);
        let resident = self.len();
        assert!(resident > 0, "attention requires a resident lifecycle token");
        let mut scratch = self.attention_scratch.borrow_mut();
        let TieredAttentionScratch {
            scores,
            key,
            value,
        } = &mut *scratch;
        let active_scores = &mut scores[..resident];
        let scale = 1.0 / (self.dimension as f32).sqrt();
        let mut cursor = 0;
        Self::append_scores(
            self.cold.as_ref(),
            query,
            active_scores,
            key,
            value,
            &mut cursor,
            scale,
        );
        Self::append_scores(
            self.warm.as_ref(),
            query,
            active_scores,
            key,
            value,
            &mut cursor,
            scale,
        );
        Self::append_scores(
            self.hot.as_ref(),
            query,
            active_scores,
            key,
            value,
            &mut cursor,
            scale,
        );
        debug_assert_eq!(cursor, resident);

        let mut maximum = active_scores[0];
        for score in &active_scores[1..]
        {
            maximum = maximum.max(*score);
        }
        let mut denominator = 0.0_f32;
        for score in active_scores.iter_mut()
        {
            *score = (*score - maximum).exp();
            denominator += *score;
        }
        for score in active_scores.iter_mut()
        {
            *score /= denominator;
        }

        let mut output = vec![0.0; self.dimension];
        cursor = 0;
        Self::accumulate_values(
            self.cold.as_ref(),
            active_scores,
            key,
            value,
            &mut cursor,
            &mut output,
        );
        Self::accumulate_values(
            self.warm.as_ref(),
            active_scores,
            key,
            value,
            &mut cursor,
            &mut output,
        );
        Self::accumulate_values(
            self.hot.as_ref(),
            active_scores,
            key,
            value,
            &mut cursor,
            &mut output,
        );
        debug_assert_eq!(cursor, resident);
        output
    }

    fn packed_bytes(&self) -> usize {
        self.allocated_bytes()
    }
}

fn validate_full_basis(
    name: &'static str,
    basis: &[f32],
    dimension: usize,
) -> Result<(), TieredLatentBackendError> {
    let expected = dimension.saturating_mul(dimension);
    if basis.len() != expected
    {
        return Err(TieredLatentBackendError::BasisLength {
            name,
            expected,
            actual: basis.len(),
        });
    }
    Ok(())
}

fn validate_tier(
    temperature: CacheTemperature,
    tier: CompressionTier,
) -> Result<(), TieredLatentBackendError> {
    if tier.rank_divisor == 0
    {
        return Err(TieredLatentBackendError::ZeroRankDivisor(temperature));
    }
    Ok(())
}

fn tier_rank(rank: usize, divisor: usize) -> usize {
    (rank / divisor).max(1)
}

fn planned_tier_bytes(
    capacity: usize,
    dimension: usize,
    plan: AdaptiveKvPlan,
    tier: CompressionTier,
) -> usize {
    if capacity == 0
    {
        return 0;
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
    estimate_channel_bytes(
        capacity,
        dimension,
        key_rank,
        key_slots,
        tier.coefficient_format,
        tier.residual_format,
    )
    .saturating_add(estimate_channel_bytes(
        capacity,
        dimension,
        value_rank,
        value_slots,
        tier.coefficient_format,
        tier.residual_format,
    ))
}

fn build_tier(
    capacity: usize,
    dimension: usize,
    full_key_basis: &[f32],
    full_value_basis: &[f32],
    plan: AdaptiveKvPlan,
    tier: CompressionTier,
) -> Result<Option<ResidualQuantizedLatentKvCache>, TieredLatentBackendError> {
    if capacity == 0
    {
        return Ok(None);
    }
    let key_rank = tier_rank(plan.key.rank, tier.rank_divisor);
    let value_rank = tier_rank(plan.value.rank, tier.rank_divisor);
    let key_basis = prefix_basis(full_key_basis, dimension, key_rank);
    let value_basis = prefix_basis(full_value_basis, dimension, value_rank);
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
    Ok(Some(ResidualQuantizedLatentKvCache::new(
        capacity,
        dimension,
        key_rank,
        value_rank,
        tier.coefficient_format,
        tier.coefficient_format,
        key_basis,
        value_basis,
        SparseResidualConfig::new(key_slots, tier.residual_format),
        SparseResidualConfig::new(value_slots, tier.residual_format),
    )?))
}

fn prefix_basis(full_basis: &[f32], dimension: usize, rank: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(dimension.saturating_mul(rank));
    for row in full_basis.chunks_exact(dimension)
    {
        output.extend_from_slice(&row[..rank]);
    }
    output
}

fn tier_len(cache: &Option<ResidualQuantizedLatentKvCache>) -> usize {
    cache.as_ref().map_or(0, ResidualQuantizedLatentKvCache::len)
}

fn tier_allocated(cache: &Option<ResidualQuantizedLatentKvCache>) -> usize {
    cache
        .as_ref()
        .map_or(0, ResidualQuantizedLatentKvCache::allocated_bytes)
}

#[cfg(test)]
mod tests {
    use super::TieredResidualLatentBackend;
    use crate::nn::adaptive_latent_kv::{
        AdaptiveChannelPlan, AdaptiveKvPlan, estimate_channel_bytes,
    };
    use crate::nn::kv_backend::AttentionBackend;
    use crate::nn::latent_kv_cache::LatentStorageFormat;
    use crate::nn::latent_kv_lifecycle::{CacheTemperature, CompressionTier, LifecycleConfig};
    use crate::nn::paged_attention::contiguous_attention;

    fn identity(dimension: usize) -> Vec<f32> {
        let mut basis = vec![0.0; dimension * dimension];
        for index in 0..dimension
        {
            basis[index * dimension + index] = 1.0;
        }
        basis
    }

    fn full_rank_plan(capacity: usize, dimension: usize) -> AdaptiveKvPlan {
        let channel = AdaptiveChannelPlan {
            rank: dimension,
            residual_slots: 0,
            coefficient_format: LatentStorageFormat::F32,
            residual_format: LatentStorageFormat::F32,
            persistent_bytes: estimate_channel_bytes(
                capacity,
                dimension,
                dimension,
                0,
                LatentStorageFormat::F32,
                LatentStorageFormat::F32,
            ),
            quality_bps: 10_000,
        };
        AdaptiveKvPlan {
            key: channel,
            value: channel,
            persistent_bytes: channel.persistent_bytes * 2,
            worst_quality_bps: 10_000,
            fingerprint: 1,
        }
    }

    fn lifecycle(
        capacity: usize,
        hot_tokens: usize,
        warm_tokens: usize,
        hot: CompressionTier,
        warm: CompressionTier,
        cold: CompressionTier,
    ) -> LifecycleConfig {
        LifecycleConfig {
            capacity_tokens: capacity,
            hot_tokens,
            warm_tokens,
            hot,
            warm,
            cold,
        }
    }

    fn f32_tier() -> CompressionTier {
        CompressionTier {
            coefficient_format: LatentStorageFormat::F32,
            residual_format: LatentStorageFormat::F32,
            maximum_residual_slots: 0,
            rank_divisor: 1,
        }
    }

    #[test]
    fn sliding_full_rank_tiers_match_dense_last_window() {
        let capacity = 4;
        let dimension = 4;
        let basis = identity(dimension);
        let tier = f32_tier();
        let mut backend = TieredResidualLatentBackend::new(
            capacity,
            dimension,
            &basis,
            &basis,
            full_rank_plan(capacity, dimension),
            lifecycle(capacity, 1, 1, tier, tier, tier),
        )
        .unwrap();
        let query = [0.25, -0.5, 0.75, 0.1];
        let mut dense_keys = Vec::new();
        let mut dense_values = Vec::new();

        for step in 0..9
        {
            let key: Vec<f32> = (0..dimension)
                .map(|index| (step * dimension + index) as f32 * 0.03 - 0.2)
                .collect();
            let value: Vec<f32> = (0..dimension)
                .map(|index| (step * dimension + index) as f32 * -0.02 + 0.4)
                .collect();
            backend.append(&key, &value);
            dense_keys.extend_from_slice(&key);
            dense_values.extend_from_slice(&value);
            if dense_keys.len() > capacity * dimension
            {
                dense_keys.drain(..dimension);
                dense_values.drain(..dimension);
            }
            let expected = contiguous_attention(
                &dense_keys,
                &dense_values,
                &query,
                dimension,
                dense_keys.len() / dimension,
            );
            let actual = backend.attention(&query);
            for (left, right) in expected.iter().zip(&actual)
            {
                assert!((left - right).abs() <= 3.0e-6, "left={left}, right={right}");
            }
            assert_eq!(backend.len(), (step + 1).min(capacity));
        }
        assert_eq!(backend.telemetry().evictions, 5);
    }

    #[test]
    fn transitions_materialize_target_rank_and_formats() {
        let capacity = 6;
        let dimension = 8;
        let basis = identity(dimension);
        let hot = CompressionTier {
            coefficient_format: LatentStorageFormat::F32,
            residual_format: LatentStorageFormat::F32,
            maximum_residual_slots: 0,
            rank_divisor: 1,
        };
        let warm = CompressionTier {
            coefficient_format: LatentStorageFormat::Int8,
            residual_format: LatentStorageFormat::Int8,
            maximum_residual_slots: 0,
            rank_divisor: 2,
        };
        let cold = CompressionTier {
            coefficient_format: LatentStorageFormat::Int4,
            residual_format: LatentStorageFormat::Int4,
            maximum_residual_slots: 0,
            rank_divisor: 4,
        };
        let mut backend = TieredResidualLatentBackend::new(
            capacity,
            dimension,
            &basis,
            &basis,
            full_rank_plan(capacity, dimension),
            lifecycle(capacity, 1, 2, hot, warm, cold),
        )
        .unwrap();
        let allocated = backend.packed_bytes();
        for step in 0..12
        {
            let key = vec![step as f32 * 0.1; dimension];
            let value = vec![step as f32 * -0.07; dimension];
            backend.append(&key, &value);
            assert_eq!(backend.packed_bytes(), allocated);
        }
        let hot_cache = backend.tier(CacheTemperature::Hot).unwrap();
        let warm_cache = backend.tier(CacheTemperature::Warm).unwrap();
        let cold_cache = backend.tier(CacheTemperature::Cold).unwrap();
        assert_eq!(hot_cache.key_rank(), 8);
        assert_eq!(hot_cache.key_format(), LatentStorageFormat::F32);
        assert_eq!(warm_cache.key_rank(), 4);
        assert_eq!(warm_cache.key_format(), LatentStorageFormat::Int8);
        assert_eq!(cold_cache.key_rank(), 2);
        assert_eq!(cold_cache.key_format(), LatentStorageFormat::Int4);
        assert_eq!(backend.telemetry().hot_tokens, 1);
        assert_eq!(backend.telemetry().warm_tokens, 2);
        assert_eq!(backend.telemetry().cold_tokens, 3);
        assert_eq!(backend.telemetry().evictions, 6);
    }
}
