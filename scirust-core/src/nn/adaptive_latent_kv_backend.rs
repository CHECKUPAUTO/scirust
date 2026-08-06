//! Runtime adapter that materializes a Phase 9 plan as a Phase 8 backend.

use crate::nn::adaptive_latent_kv::AdaptiveKvPlan;
use crate::nn::kv_backend::AttentionBackend;
use crate::nn::residual_latent_kv_backend::ResidualLatentQuantizedBackend;
use crate::nn::residual_latent_kv_cache::{ResidualLatentCacheError, SparseResidualConfig};
use core::fmt;

/// Errors returned while constructing an adaptive backend.
#[derive(Debug)]
pub enum AdaptiveLatentBackendError {
    /// A full row-major basis was not shaped `[dimension, dimension]`.
    BasisLength {
        /// Human-readable basis name.
        name: &'static str,
        /// Expected element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// Phase 8 backend construction failed.
    Residual(ResidualLatentCacheError),
}

impl fmt::Display for AdaptiveLatentBackendError {
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
            Self::Residual(error) => write!(output, "{error}"),
        }
    }
}

impl std::error::Error for AdaptiveLatentBackendError {}

impl From<ResidualLatentCacheError> for AdaptiveLatentBackendError {
    fn from(error: ResidualLatentCacheError) -> Self {
        Self::Residual(error)
    }
}

/// Production-facing backend selected by the deterministic Phase 9 planner.
pub struct AdaptiveResidualLatentBackend {
    plan: AdaptiveKvPlan,
    inner: ResidualLatentQuantizedBackend,
}

impl AdaptiveResidualLatentBackend {
    /// Builds a concrete residual latent backend from a selected adaptive plan.
    pub fn new(
        capacity_tokens: usize,
        dimension: usize,
        full_key_basis: &[f32],
        full_value_basis: &[f32],
        plan: AdaptiveKvPlan,
    ) -> Result<Self, AdaptiveLatentBackendError> {
        let key_basis = prefix_basis("key", full_key_basis, dimension, plan.key.rank)?;
        let value_basis = prefix_basis("value", full_value_basis, dimension, plan.value.rank)?;
        let inner = ResidualLatentQuantizedBackend::new(
            capacity_tokens,
            dimension,
            plan.key.rank,
            plan.value.rank,
            plan.key.coefficient_format,
            plan.value.coefficient_format,
            key_basis,
            value_basis,
            SparseResidualConfig::new(plan.key.residual_slots, plan.key.residual_format),
            SparseResidualConfig::new(plan.value.residual_slots, plan.value.residual_format),
        )?;
        Ok(Self { plan, inner })
    }

    /// Returns the immutable selected plan.
    #[must_use]
    pub const fn plan(&self) -> AdaptiveKvPlan {
        self.plan
    }

    /// Returns the concrete Phase 8 backend for detailed telemetry.
    #[must_use]
    pub const fn inner(&self) -> &ResidualLatentQuantizedBackend {
        &self.inner
    }
}

impl AttentionBackend for AdaptiveResidualLatentBackend {
    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn append(&mut self, key: &[f32], value: &[f32]) {
        self.inner.append(key, value);
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn attention(&self, query: &[f32]) -> Vec<f32> {
        self.inner.attention(query)
    }

    fn packed_bytes(&self) -> usize {
        self.inner.packed_bytes()
    }
}

fn prefix_basis(
    name: &'static str,
    full_basis: &[f32],
    dimension: usize,
    rank: usize,
) -> Result<Vec<f32>, AdaptiveLatentBackendError> {
    let expected = dimension.saturating_mul(dimension);
    if full_basis.len() != expected
    {
        return Err(AdaptiveLatentBackendError::BasisLength {
            name,
            expected,
            actual: full_basis.len(),
        });
    }
    let mut output = Vec::with_capacity(dimension.saturating_mul(rank));
    for row in full_basis.chunks_exact(dimension)
    {
        output.extend_from_slice(&row[..rank]);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::AdaptiveResidualLatentBackend;
    use crate::nn::adaptive_latent_kv::{
        AdaptiveChannelPlan, AdaptiveKvPlan, estimate_channel_bytes,
    };
    use crate::nn::kv_backend::AttentionBackend;
    use crate::nn::latent_kv_cache::LatentStorageFormat;

    fn identity(dimension: usize) -> Vec<f32> {
        let mut basis = vec![0.0; dimension * dimension];
        for (index, row) in basis.chunks_exact_mut(dimension).enumerate()
        {
            row[index] = 1.0;
        }
        basis
    }

    fn plan(capacity: usize, dimension: usize) -> AdaptiveKvPlan {
        let channel = AdaptiveChannelPlan {
            rank: 4,
            residual_slots: 2,
            coefficient_format: LatentStorageFormat::Int8,
            residual_format: LatentStorageFormat::Int8,
            persistent_bytes: estimate_channel_bytes(
                capacity,
                dimension,
                4,
                2,
                LatentStorageFormat::Int8,
                LatentStorageFormat::Int8,
            ),
            quality_bps: 9_500,
        };
        AdaptiveKvPlan {
            key: channel,
            value: channel,
            persistent_bytes: channel.persistent_bytes * 2,
            worst_quality_bps: channel.quality_bps,
            fingerprint: 7,
        }
    }

    #[test]
    fn adaptive_plan_materializes_phase8_backend() {
        let capacity = 16;
        let dimension = 8;
        let basis = identity(dimension);
        let selected = plan(capacity, dimension);
        let backend =
            AdaptiveResidualLatentBackend::new(capacity, dimension, &basis, &basis, selected)
                .unwrap();
        assert_eq!(backend.plan(), selected);
        assert_eq!(backend.dim(), dimension);
        assert_eq!(backend.inner().cache().key_rank(), 4);
        assert_eq!(
            backend
                .inner()
                .cache()
                .key_residual_config()
                .slots_per_token(),
            2
        );
    }
}
