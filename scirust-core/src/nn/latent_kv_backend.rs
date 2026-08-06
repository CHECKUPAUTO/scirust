//! Adapter from the quantized latent cache to the live numeric decode path.

use crate::nn::kv_backend::AttentionBackend;
use crate::nn::latent_kv_cache::{
    LatentAttentionScratch, LatentCacheError, LatentStorageFormat, QuantizedLatentKvCache,
};
use std::cell::RefCell;

/// Reconstruction-free quantized latent backend for one attention head.
///
/// The `AttentionBackend` trait returns an owned vector, so the final dense
/// context allocation remains at the trait boundary. Projection, quantized
/// score evaluation, latent value accumulation, and up-projection reuse fixed
/// scratch owned by this adapter.
pub struct LatentQuantizedBackend {
    inner: QuantizedLatentKvCache,
    scratch: RefCell<LatentAttentionScratch>,
}

impl LatentQuantizedBackend {
    /// Creates a backend with independently selected key and value formats.
    pub fn new(
        capacity_tokens: usize,
        dimension: usize,
        key_rank: usize,
        value_rank: usize,
        key_format: LatentStorageFormat,
        value_format: LatentStorageFormat,
        key_basis: Vec<f32>,
        value_basis: Vec<f32>,
    ) -> Result<Self, LatentCacheError> {
        let inner = QuantizedLatentKvCache::new(
            capacity_tokens,
            dimension,
            key_rank,
            value_rank,
            key_format,
            value_format,
            key_basis,
            value_basis,
        )?;
        let scratch = RefCell::new(LatentAttentionScratch::new(
            capacity_tokens,
            key_rank,
            value_rank,
        ));
        Ok(Self { inner, scratch })
    }

    /// Creates a backend using the same rank, format, and basis for keys and values.
    pub fn new_symmetric(
        capacity_tokens: usize,
        dimension: usize,
        rank: usize,
        format: LatentStorageFormat,
        basis: Vec<f32>,
    ) -> Result<Self, LatentCacheError> {
        Self::new(
            capacity_tokens,
            dimension,
            rank,
            rank,
            format,
            format,
            basis.clone(),
            basis,
        )
    }

    /// Returns the underlying cache for telemetry and deterministic inspection.
    #[must_use]
    pub const fn cache(&self) -> &QuantizedLatentKvCache {
        &self.inner
    }
}

impl AttentionBackend for LatentQuantizedBackend {
    fn dim(&self) -> usize {
        self.inner.dimension()
    }

    fn append(&mut self, key: &[f32], value: &[f32]) {
        self.inner
            .append(key, value)
            .expect("LatentQuantizedBackend append contract violated");
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn attention(&self, query: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.inner.dimension()];
        self.inner
            .attention_into(query, &mut output, &mut self.scratch.borrow_mut())
            .expect("LatentQuantizedBackend attention contract violated");
        output
    }

    fn packed_bytes(&self) -> usize {
        self.inner.allocated_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::LatentQuantizedBackend;
    use crate::nn::init::{KaimingNormal, Zeros};
    use crate::nn::kv_backend::{AttentionBackend, PlainKvCache, decode_step};
    use crate::nn::latent_kv_cache::LatentStorageFormat;
    use crate::nn::rng::PcgEngine;
    use crate::nn::transformer::attention::MultiHeadAttention;

    fn identity(dimension: usize) -> Vec<f32> {
        let mut basis = vec![0.0; dimension * dimension];
        for diagonal in 0..dimension {
            basis[diagonal * dimension + diagonal] = 1.0;
        }
        basis
    }

    fn build_attention(d_model: usize, heads: usize) -> MultiHeadAttention {
        let mut rng = PcgEngine::new(0);
        MultiHeadAttention::new(d_model, heads, false, &KaimingNormal, &Zeros, &mut rng)
    }

    fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
        assert_eq!(left.len(), right.len());
        for (index, (left_value, right_value)) in left.iter().zip(right).enumerate() {
            let error = (left_value - right_value).abs();
            assert!(
                error <= tolerance,
                "index {index}: left={left_value}, right={right_value}, error={error}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn full_rank_f32_backend_matches_plain_decode() {
        let (d_model, heads, capacity) = (12, 3, 16);
        let attention = build_attention(d_model, heads);
        let head_dimension = attention.d_head;
        let mut plain: Vec<Box<dyn AttentionBackend>> = (0..heads)
            .map(|_| Box::new(PlainKvCache::new(head_dimension)) as Box<dyn AttentionBackend>)
            .collect();
        let mut latent: Vec<Box<dyn AttentionBackend>> = (0..heads)
            .map(|_| {
                Box::new(
                    LatentQuantizedBackend::new_symmetric(
                        capacity,
                        head_dimension,
                        head_dimension,
                        LatentStorageFormat::F32,
                        identity(head_dimension),
                    )
                    .unwrap(),
                ) as Box<dyn AttentionBackend>
            })
            .collect();

        for step in 0..8 {
            let token: Vec<f32> = (0..d_model)
                .map(|index| ((step * d_model + index) as f32) * 0.013 - 0.4)
                .collect();
            let expected = decode_step(&attention, &token, &mut plain);
            let actual = decode_step(&attention, &token, &mut latent);
            assert_close(&expected, &actual, 2.0e-6);
        }
    }

    #[test]
    fn int8_backend_bounds_decode_error_and_memory() {
        let (d_model, heads, capacity) = (16, 2, 64);
        let attention = build_attention(d_model, heads);
        let head_dimension = attention.d_head;
        let mut plain: Vec<Box<dyn AttentionBackend>> = (0..heads)
            .map(|_| Box::new(PlainKvCache::new(head_dimension)) as Box<dyn AttentionBackend>)
            .collect();
        let mut latent: Vec<Box<dyn AttentionBackend>> = (0..heads)
            .map(|_| {
                Box::new(
                    LatentQuantizedBackend::new_symmetric(
                        capacity,
                        head_dimension,
                        head_dimension,
                        LatentStorageFormat::Int8,
                        identity(head_dimension),
                    )
                    .unwrap(),
                ) as Box<dyn AttentionBackend>
            })
            .collect();

        for step in 0..capacity {
            let token: Vec<f32> = (0..d_model)
                .map(|index| (((step + 3) * (index + 5)) as f32).sin() * 0.25)
                .collect();
            let expected = decode_step(&attention, &token, &mut plain);
            let actual = decode_step(&attention, &token, &mut latent);
            assert_close(&expected, &actual, 0.02);
        }

        let dense_bytes = capacity * head_dimension * 2 * core::mem::size_of::<f32>();
        for backend in &latent {
            assert!(backend.packed_bytes() < dense_bytes);
        }
    }

    #[test]
    fn adapter_reports_cache_state() {
        let dimension = 8;
        let mut backend = LatentQuantizedBackend::new_symmetric(
            4,
            dimension,
            4,
            LatentStorageFormat::Int4,
            {
                let mut basis = vec![0.0; dimension * 4];
                for diagonal in 0..4 {
                    basis[diagonal * 4 + diagonal] = 1.0;
                }
                basis
            },
        )
        .unwrap();
        assert!(backend.is_empty());
        backend.append(&[0.1; 8], &[0.2; 8]);
        assert_eq!(backend.len(), 1);
        assert_eq!(backend.cache().key_rank(), 4);
        assert_eq!(backend.cache().value_rank(), 4);
        assert_eq!(backend.cache().key_format(), LatentStorageFormat::Int4);
    }
}
