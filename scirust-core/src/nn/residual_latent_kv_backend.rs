//! Adapter from the sparse-residual latent cache to the live numeric decode path.

use crate::nn::kv_backend::AttentionBackend;
use crate::nn::latent_kv_cache::LatentStorageFormat;
use crate::nn::residual_latent_kv_cache::{
    ResidualLatentAttentionScratch, ResidualLatentCacheError, ResidualQuantizedLatentKvCache,
    SparseResidualConfig,
};
use std::cell::RefCell;

/// Reconstruction-free latent backend with fixed-slot sparse residual channels.
pub struct ResidualLatentQuantizedBackend {
    inner: ResidualQuantizedLatentKvCache,
    scratch: RefCell<ResidualLatentAttentionScratch>,
}

impl ResidualLatentQuantizedBackend {
    /// Creates a backend with independent key/value ranks, formats, and residuals.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        capacity_tokens: usize,
        dimension: usize,
        key_rank: usize,
        value_rank: usize,
        key_format: LatentStorageFormat,
        value_format: LatentStorageFormat,
        key_basis: Vec<f32>,
        value_basis: Vec<f32>,
        key_residual: SparseResidualConfig,
        value_residual: SparseResidualConfig,
    ) -> Result<Self, ResidualLatentCacheError> {
        let inner = ResidualQuantizedLatentKvCache::new(
            capacity_tokens,
            dimension,
            key_rank,
            value_rank,
            key_format,
            value_format,
            key_basis,
            value_basis,
            key_residual,
            value_residual,
        )?;
        let scratch = RefCell::new(ResidualLatentAttentionScratch::new(
            capacity_tokens,
            key_rank,
            value_rank,
        ));
        Ok(Self { inner, scratch })
    }

    /// Creates a symmetric key/value backend.
    #[allow(clippy::too_many_arguments)]
    pub fn new_symmetric(
        capacity_tokens: usize,
        dimension: usize,
        rank: usize,
        coefficient_format: LatentStorageFormat,
        basis: Vec<f32>,
        residual_slots_per_token: usize,
        residual_format: LatentStorageFormat,
    ) -> Result<Self, ResidualLatentCacheError> {
        let residual = SparseResidualConfig::new(residual_slots_per_token, residual_format);
        Self::new(
            capacity_tokens,
            dimension,
            rank,
            rank,
            coefficient_format,
            coefficient_format,
            basis.clone(),
            basis,
            residual,
            residual,
        )
    }

    /// Returns the underlying cache for telemetry and deterministic inspection.
    #[must_use]
    pub const fn cache(&self) -> &ResidualQuantizedLatentKvCache {
        &self.inner
    }
}

impl AttentionBackend for ResidualLatentQuantizedBackend {
    fn dim(&self) -> usize {
        self.inner.dimension()
    }

    fn append(&mut self, key: &[f32], value: &[f32]) {
        self.inner
            .append(key, value)
            .expect("ResidualLatentQuantizedBackend append contract violated");
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn attention(&self, query: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.inner.dimension()];
        self.inner
            .attention_into(query, &mut output, &mut self.scratch.borrow_mut())
            .expect("ResidualLatentQuantizedBackend attention contract violated");
        output
    }

    fn packed_bytes(&self) -> usize {
        self.inner.allocated_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::ResidualLatentQuantizedBackend;
    use crate::nn::init::{KaimingNormal, Zeros};
    use crate::nn::kv_backend::{AttentionBackend, PlainKvCache, decode_step};
    use crate::nn::latent_kv_cache::LatentStorageFormat;
    use crate::nn::rng::PcgEngine;
    use crate::nn::transformer::attention::MultiHeadAttention;

    fn identity_prefix(dimension: usize, rank: usize) -> Vec<f32> {
        let mut basis = vec![0.0; dimension * rank];
        for diagonal in 0..rank
        {
            basis[diagonal * rank + diagonal] = 1.0;
        }
        basis
    }

    fn build_attention(d_model: usize, heads: usize) -> MultiHeadAttention {
        let mut rng = PcgEngine::new(0);
        MultiHeadAttention::new(d_model, heads, false, &KaimingNormal, &Zeros, &mut rng)
    }

    fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len());
        for (index, (left_value, right_value)) in left.iter().zip(right).enumerate()
        {
            let error = (left_value - right_value).abs();
            assert!(
                error <= tolerance,
                "index {index}: left={left_value}, right={right_value}, error={error}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn full_rank_residual_backend_matches_plain_decode() {
        let (d_model, heads, capacity) = (12, 3, 16);
        let attention = build_attention(d_model, heads);
        let head_dimension = attention.d_head;
        let mut plain: Vec<Box<dyn AttentionBackend>> = (0..heads)
            .map(|_| Box::new(PlainKvCache::new(head_dimension)) as Box<dyn AttentionBackend>)
            .collect();
        let mut residual: Vec<Box<dyn AttentionBackend>> = (0..heads)
            .map(|_| {
                Box::new(
                    ResidualLatentQuantizedBackend::new_symmetric(
                        capacity,
                        head_dimension,
                        head_dimension,
                        LatentStorageFormat::F32,
                        identity_prefix(head_dimension, head_dimension),
                        2,
                        LatentStorageFormat::Int8,
                    )
                    .unwrap(),
                ) as Box<dyn AttentionBackend>
            })
            .collect();

        for step in 0..8
        {
            let token: Vec<f32> = (0..d_model)
                .map(|index| ((step * d_model + index) as f32) * 0.013 - 0.4)
                .collect();
            let expected = decode_step(&attention, &token, &mut plain);
            let actual = decode_step(&attention, &token, &mut residual);
            assert_close(&expected, &actual, 2.0e-6);
        }
    }

    #[test]
    fn packed_residual_backend_reports_memory_reduction() {
        let capacity = 128;
        let dimension = 16;
        let backend = ResidualLatentQuantizedBackend::new_symmetric(
            capacity,
            dimension,
            4,
            LatentStorageFormat::Int4,
            identity_prefix(dimension, 4),
            2,
            LatentStorageFormat::Int4,
        )
        .unwrap();
        let dense_bytes = capacity * dimension * 2 * core::mem::size_of::<f32>();
        assert!(backend.packed_bytes() < dense_bytes);
        assert_eq!(backend.cache().key_residual_config().slots_per_token(), 2);
        assert_eq!(
            backend.cache().value_residual_config().format(),
            LatentStorageFormat::Int4
        );
    }
}
