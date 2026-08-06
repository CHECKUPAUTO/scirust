//! Reconstruction-free quantized latent KV-cache.
//!
//! The cache projects dense keys and values into fixed latent bases when a token
//! is appended, stores the resulting coefficient rows as FP32, symmetric INT8,
//! or packed symmetric INT4, and evaluates attention directly from those stored
//! coefficients. Keys are never reconstructed. Values are accumulated in latent
//! space and up-projected exactly once per query.
//!
//! Storage and scratch buffers are allocated at construction. Appending tokens
//! and [`QuantizedLatentKvCache::attention_into`] do not grow any buffer.

use core::fmt;

/// Storage format for one latent coefficient matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatentStorageFormat {
    /// Store coefficients as little-endian `f32` values.
    F32,
    /// Store each coefficient as one symmetric signed INT8 code.
    Int8,
    /// Store two symmetric signed INT4 codes per byte in `[-7, 7]`.
    Int4,
}

impl LatentStorageFormat {
    /// Stable human-readable label used by telemetry and tests.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
        }
    }

    const fn quantization_limit(self) -> Option<i8> {
        match self {
            Self::F32 => None,
            Self::Int8 => Some(127),
            Self::Int4 => Some(7),
        }
    }
}

/// Errors returned by the quantized latent cache.
#[derive(Debug, Clone, PartialEq)]
pub enum LatentCacheError {
    /// A required dimension or capacity was zero.
    ZeroDimension {
        /// Human-readable field name.
        field: &'static str,
    },
    /// A rank exceeded the dense head dimension.
    RankTooLarge {
        /// Human-readable rank name.
        name: &'static str,
        /// Supplied rank.
        rank: usize,
        /// Dense head dimension.
        dimension: usize,
    },
    /// A flat input buffer had an unexpected length.
    Length {
        /// Human-readable input name.
        name: &'static str,
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// An input contained a non-finite scalar.
    NonFinite {
        /// Human-readable input name.
        name: &'static str,
        /// Index of the invalid scalar.
        index: usize,
    },
    /// The fixed token capacity was exhausted.
    CapacityExceeded {
        /// Maximum resident token count.
        capacity: usize,
    },
    /// Attention was requested before any token was appended.
    EmptyCache,
    /// A scratch region was smaller than the configured cache requires.
    ScratchTooSmall {
        /// Human-readable scratch region.
        name: &'static str,
        /// Required element count.
        required: usize,
        /// Available element count.
        available: usize,
    },
    /// Checked integer arithmetic overflowed.
    Overflow,
}

impl fmt::Display for LatentCacheError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension { field } => write!(output, "{field} must be non-zero"),
            Self::RankTooLarge {
                name,
                rank,
                dimension,
            } => write!(
                output,
                "{name} rank {rank} exceeds head dimension {dimension}"
            ),
            Self::Length {
                name,
                expected,
                actual,
            } => write!(
                output,
                "{name} length mismatch: expected {expected}, received {actual}"
            ),
            Self::NonFinite { name, index } => {
                write!(output, "{name} contains a non-finite scalar at index {index}")
            }
            Self::CapacityExceeded { capacity } => {
                write!(output, "latent cache token capacity exceeded: {capacity}")
            }
            Self::EmptyCache => write!(output, "attention requires at least one cached token"),
            Self::ScratchTooSmall {
                name,
                required,
                available,
            } => write!(
                output,
                "{name} scratch is too small: required {required}, available {available}"
            ),
            Self::Overflow => write!(output, "latent cache arithmetic overflow"),
        }
    }
}

impl std::error::Error for LatentCacheError {}

/// Reusable scratch storage for reconstruction-free latent attention.
#[derive(Debug, Clone)]
pub struct LatentAttentionScratch {
    scores: Vec<f32>,
    query_latent: Vec<f32>,
    value_latent: Vec<f32>,
}

impl LatentAttentionScratch {
    /// Allocates scratch for the supplied maximum token count and latent ranks.
    #[must_use]
    pub fn new(max_tokens: usize, key_rank: usize, value_rank: usize) -> Self {
        Self {
            scores: vec![0.0; max_tokens],
            query_latent: vec![0.0; key_rank],
            value_latent: vec![0.0; value_rank],
        }
    }

    fn validate(&self, tokens: usize, key_rank: usize, value_rank: usize) -> Result<(), LatentCacheError> {
        require_scratch("score", self.scores.len(), tokens)?;
        require_scratch("query-latent", self.query_latent.len(), key_rank)?;
        require_scratch("value-latent", self.value_latent.len(), value_rank)
    }
}

/// Fixed-capacity quantized latent KV-cache for one attention head.
#[derive(Debug, Clone)]
pub struct QuantizedLatentKvCache {
    capacity_tokens: usize,
    dimension: usize,
    key_rank: usize,
    value_rank: usize,
    key_format: LatentStorageFormat,
    value_format: LatentStorageFormat,
    key_basis: Vec<f32>,
    value_basis: Vec<f32>,
    key_payload: Vec<u8>,
    value_payload: Vec<u8>,
    key_scales: Vec<f32>,
    value_scales: Vec<f32>,
    key_projection: Vec<f32>,
    value_projection: Vec<f32>,
    len: usize,
}

impl QuantizedLatentKvCache {
    /// Creates an empty cache with row-major bases shaped `[dimension, rank]`.
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
        non_zero("capacity_tokens", capacity_tokens)?;
        non_zero("dimension", dimension)?;
        non_zero("key_rank", key_rank)?;
        non_zero("value_rank", value_rank)?;
        require_rank("key", key_rank, dimension)?;
        require_rank("value", value_rank, dimension)?;

        let expected_key_basis = checked_product(dimension, key_rank)?;
        let expected_value_basis = checked_product(dimension, value_rank)?;
        require_length("key basis", key_basis.len(), expected_key_basis)?;
        require_length("value basis", value_basis.len(), expected_value_basis)?;
        require_finite("key basis", &key_basis)?;
        require_finite("value basis", &value_basis)?;

        let key_row_bytes = row_bytes(key_format, key_rank)?;
        let value_row_bytes = row_bytes(value_format, value_rank)?;
        let key_payload = vec![0_u8; checked_product(capacity_tokens, key_row_bytes)?];
        let value_payload = vec![0_u8; checked_product(capacity_tokens, value_row_bytes)?];
        let key_scales = scales_for(key_format, capacity_tokens);
        let value_scales = scales_for(value_format, capacity_tokens);

        Ok(Self {
            capacity_tokens,
            dimension,
            key_rank,
            value_rank,
            key_format,
            value_format,
            key_basis,
            value_basis,
            key_payload,
            value_payload,
            key_scales,
            value_scales,
            key_projection: vec![0.0; key_rank],
            value_projection: vec![0.0; value_rank],
            len: 0,
        })
    }

    /// Returns the dense width of one key or value vector.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the fixed token capacity.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity_tokens
    }

    /// Returns the number of resident tokens.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether no token has been appended.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the latent key rank.
    #[must_use]
    pub const fn key_rank(&self) -> usize {
        self.key_rank
    }

    /// Returns the latent value rank.
    #[must_use]
    pub const fn value_rank(&self) -> usize {
        self.value_rank
    }

    /// Returns the key coefficient storage format.
    #[must_use]
    pub const fn key_format(&self) -> LatentStorageFormat {
        self.key_format
    }

    /// Returns the value coefficient storage format.
    #[must_use]
    pub const fn value_format(&self) -> LatentStorageFormat {
        self.value_format
    }

    /// Returns the logical packed bytes used by bases and resident coefficients.
    #[must_use]
    pub fn used_bytes(&self) -> usize {
        let basis_bytes = (self.key_basis.len() + self.value_basis.len())
            .saturating_mul(core::mem::size_of::<f32>());
        let key_bytes = self.len.saturating_mul(row_bytes_unchecked(self.key_format, self.key_rank));
        let value_bytes = self
            .len
            .saturating_mul(row_bytes_unchecked(self.value_format, self.value_rank));
        let scale_count = scale_count(self.key_format, self.len)
            .saturating_add(scale_count(self.value_format, self.len));
        basis_bytes
            .saturating_add(key_bytes)
            .saturating_add(value_bytes)
            .saturating_add(scale_count.saturating_mul(core::mem::size_of::<f32>()))
    }

    /// Returns the actual fixed allocation owned by the cache in bytes.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        let float_capacity = self
            .key_basis
            .capacity()
            .saturating_add(self.value_basis.capacity())
            .saturating_add(self.key_scales.capacity())
            .saturating_add(self.value_scales.capacity())
            .saturating_add(self.key_projection.capacity())
            .saturating_add(self.value_projection.capacity());
        self.key_payload
            .capacity()
            .saturating_add(self.value_payload.capacity())
            .saturating_add(float_capacity.saturating_mul(core::mem::size_of::<f32>()))
    }

    /// Projects and appends one dense key/value pair without growing storage.
    pub fn append(&mut self, key: &[f32], value: &[f32]) -> Result<(), LatentCacheError> {
        require_length("key", key.len(), self.dimension)?;
        require_length("value", value.len(), self.dimension)?;
        require_finite("key", key)?;
        require_finite("value", value)?;

        if self.len == self.capacity_tokens {
            return Err(LatentCacheError::CapacityExceeded {
                capacity: self.capacity_tokens,
            });
        }

        project_transpose(
            &self.key_basis,
            self.dimension,
            self.key_rank,
            key,
            &mut self.key_projection,
        );
        project_transpose(
            &self.value_basis,
            self.dimension,
            self.value_rank,
            value,
            &mut self.value_projection,
        );

        encode_row(
            self.key_format,
            self.key_rank,
            self.len,
            &self.key_projection,
            &mut self.key_payload,
            &mut self.key_scales,
        );
        encode_row(
            self.value_format,
            self.value_rank,
            self.len,
            &self.value_projection,
            &mut self.value_payload,
            &mut self.value_scales,
        );
        self.len += 1;
        Ok(())
    }

    /// Computes attention directly from quantized coefficient rows.
    ///
    /// The query is projected once into key-latent space. Each score is a dot
    /// product against a coefficient row dequantized on the fly. Values are
    /// accumulated directly in value-latent space, then up-projected once.
    pub fn attention_into(
        &self,
        query: &[f32],
        output: &mut [f32],
        scratch: &mut LatentAttentionScratch,
    ) -> Result<(), LatentCacheError> {
        if self.is_empty() {
            return Err(LatentCacheError::EmptyCache);
        }
        require_length("query", query.len(), self.dimension)?;
        require_length("output", output.len(), self.dimension)?;
        require_finite("query", query)?;
        scratch.validate(self.len, self.key_rank, self.value_rank)?;

        project_transpose(
            &self.key_basis,
            self.dimension,
            self.key_rank,
            query,
            &mut scratch.query_latent[..self.key_rank],
        );

        let scale = 1.0 / (self.dimension as f32).sqrt();
        let query_latent = &scratch.query_latent[..self.key_rank];
        let scores = &mut scratch.scores[..self.len];
        for (row, score) in scores.iter_mut().enumerate() {
            *score = row_dot(
                self.key_format,
                self.key_rank,
                row,
                &self.key_payload,
                &self.key_scales,
                query_latent,
            ) * scale;
        }
        softmax_numerators_in_place(scores);

        let latent_value = &mut scratch.value_latent[..self.value_rank];
        latent_value.fill(0.0);
        let denominator: f32 = scores.iter().copied().sum();
        for (row, numerator) in scores.iter().copied().enumerate() {
            accumulate_row(
                self.value_format,
                self.value_rank,
                row,
                &self.value_payload,
                &self.value_scales,
                numerator / denominator,
                latent_value,
            );
        }

        up_project(
            &self.value_basis,
            self.dimension,
            self.value_rank,
            latent_value,
            output,
        );
        Ok(())
    }
}

fn non_zero(field: &'static str, value: usize) -> Result<(), LatentCacheError> {
    if value == 0 {
        return Err(LatentCacheError::ZeroDimension { field });
    }
    Ok(())
}

fn require_rank(name: &'static str, rank: usize, dimension: usize) -> Result<(), LatentCacheError> {
    if rank > dimension {
        return Err(LatentCacheError::RankTooLarge {
            name,
            rank,
            dimension,
        });
    }
    Ok(())
}

fn require_length(name: &'static str, actual: usize, expected: usize) -> Result<(), LatentCacheError> {
    if actual != expected {
        return Err(LatentCacheError::Length {
            name,
            expected,
            actual,
        });
    }
    Ok(())
}

fn require_finite(name: &'static str, values: &[f32]) -> Result<(), LatentCacheError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(LatentCacheError::NonFinite { name, index });
    }
    Ok(())
}

fn require_scratch(name: &'static str, available: usize, required: usize) -> Result<(), LatentCacheError> {
    if available < required {
        return Err(LatentCacheError::ScratchTooSmall {
            name,
            required,
            available,
        });
    }
    Ok(())
}

fn checked_product(left: usize, right: usize) -> Result<usize, LatentCacheError> {
    left.checked_mul(right).ok_or(LatentCacheError::Overflow)
}

fn scales_for(format: LatentStorageFormat, capacity: usize) -> Vec<f32> {
    match format {
        LatentStorageFormat::F32 => Vec::new(),
        LatentStorageFormat::Int8 | LatentStorageFormat::Int4 => vec![1.0; capacity],
    }
}

const fn scale_count(format: LatentStorageFormat, rows: usize) -> usize {
    match format {
        LatentStorageFormat::F32 => 0,
        LatentStorageFormat::Int8 | LatentStorageFormat::Int4 => rows,
    }
}

fn row_bytes(format: LatentStorageFormat, columns: usize) -> Result<usize, LatentCacheError> {
    match format {
        LatentStorageFormat::F32 => checked_product(columns, core::mem::size_of::<f32>()),
        LatentStorageFormat::Int8 => Ok(columns),
        LatentStorageFormat::Int4 => Ok(columns.div_ceil(2)),
    }
}

const fn row_bytes_unchecked(format: LatentStorageFormat, columns: usize) -> usize {
    match format {
        LatentStorageFormat::F32 => columns * core::mem::size_of::<f32>(),
        LatentStorageFormat::Int8 => columns,
        LatentStorageFormat::Int4 => columns.div_ceil(2),
    }
}

fn project_transpose(
    basis: &[f32],
    rows: usize,
    columns: usize,
    vector: &[f32],
    output: &mut [f32],
) {
    output.fill(0.0);
    for (row, scalar) in vector.iter().copied().enumerate().take(rows) {
        let offset = row * columns;
        for column in 0..columns {
            output[column] += basis[offset + column] * scalar;
        }
    }
}

fn up_project(
    basis: &[f32],
    rows: usize,
    columns: usize,
    latent: &[f32],
    output: &mut [f32],
) {
    for (row, output_scalar) in output.iter_mut().enumerate().take(rows) {
        let offset = row * columns;
        let mut sum = 0.0_f32;
        for column in 0..columns {
            sum += basis[offset + column] * latent[column];
        }
        *output_scalar = sum;
    }
}

fn encode_row(
    format: LatentStorageFormat,
    columns: usize,
    row: usize,
    source: &[f32],
    payload: &mut [u8],
    scales: &mut [f32],
) {
    let bytes_per_row = row_bytes_unchecked(format, columns);
    let offset = row * bytes_per_row;
    let target = &mut payload[offset..offset + bytes_per_row];
    match format {
        LatentStorageFormat::F32 => {
            for (value, bytes) in source.iter().zip(target.chunks_exact_mut(4)) {
                bytes.copy_from_slice(&value.to_le_bytes());
            }
        }
        LatentStorageFormat::Int8 | LatentStorageFormat::Int4 => {
            target.fill(0);
            let limit = format
                .quantization_limit()
                .expect("integer formats have a quantization limit");
            let maximum = source
                .iter()
                .copied()
                .map(f32::abs)
                .fold(0.0_f32, f32::max);
            let scale = if maximum == 0.0 {
                1.0
            } else {
                maximum / f32::from(limit)
            };
            scales[row] = scale;
            for (column, value) in source.iter().copied().enumerate() {
                let code = quantize(value, scale, limit);
                match format {
                    LatentStorageFormat::Int8 => {
                        target[column] = code.to_ne_bytes()[0];
                    }
                    LatentStorageFormat::Int4 => {
                        let nibble = code.to_ne_bytes()[0] & 0x0f;
                        if column.is_multiple_of(2) {
                            target[column / 2] = nibble;
                        } else {
                            target[column / 2] |= nibble << 4;
                        }
                    }
                    LatentStorageFormat::F32 => unreachable!(),
                }
            }
        }
    }
}

fn quantize(value: f32, scale: f32, limit: i8) -> i8 {
    (value / scale)
        .round()
        .clamp(-f32::from(limit), f32::from(limit)) as i8
}

fn row_dot(
    format: LatentStorageFormat,
    columns: usize,
    row: usize,
    payload: &[u8],
    scales: &[f32],
    vector: &[f32],
) -> f32 {
    let mut sum = 0.0_f32;
    for (column, vector_scalar) in vector.iter().copied().enumerate().take(columns) {
        sum += vector_scalar * coefficient(format, columns, row, column, payload, scales);
    }
    sum
}

fn accumulate_row(
    format: LatentStorageFormat,
    columns: usize,
    row: usize,
    payload: &[u8],
    scales: &[f32],
    weight: f32,
    output: &mut [f32],
) {
    for (column, output_scalar) in output.iter_mut().enumerate().take(columns) {
        *output_scalar += weight * coefficient(format, columns, row, column, payload, scales);
    }
}

fn coefficient(
    format: LatentStorageFormat,
    columns: usize,
    row: usize,
    column: usize,
    payload: &[u8],
    scales: &[f32],
) -> f32 {
    let bytes_per_row = row_bytes_unchecked(format, columns);
    let offset = row * bytes_per_row;
    match format {
        LatentStorageFormat::F32 => {
            let start = offset + column * 4;
            f32::from_le_bytes([
                payload[start],
                payload[start + 1],
                payload[start + 2],
                payload[start + 3],
            ])
        }
        LatentStorageFormat::Int8 => {
            f32::from(i8::from_ne_bytes([payload[offset + column]])) * scales[row]
        }
        LatentStorageFormat::Int4 => {
            let packed = payload[offset + column / 2];
            let nibble = if column.is_multiple_of(2) {
                packed & 0x0f
            } else {
                packed >> 4
            };
            let signed = if nibble < 8 {
                nibble as i8
            } else {
                (i16::from(nibble) - 16) as i8
            };
            f32::from(signed) * scales[row]
        }
    }
}

fn softmax_numerators_in_place(scores: &mut [f32]) {
    let mut maximum = scores[0];
    for score in &scores[1..] {
        if *score > maximum {
            maximum = *score;
        }
    }
    for score in scores {
        *score = (*score - maximum).exp();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        LatentAttentionScratch, LatentCacheError, LatentStorageFormat,
        QuantizedLatentKvCache,
    };
    use crate::nn::paged_attention::contiguous_attention;
    use crate::nn::rng::PcgEngine;

    fn identity(dimension: usize, rank: usize) -> Vec<f32> {
        let mut basis = vec![0.0; dimension * rank];
        for diagonal in 0..rank {
            basis[diagonal * rank + diagonal] = 1.0;
        }
        basis
    }

    fn seeded_vectors(tokens: usize, dimension: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut rng = PcgEngine::new(seed);
        let keys = (0..tokens * dimension)
            .map(|_| rng.float_signed())
            .collect();
        let values = (0..tokens * dimension)
            .map(|_| rng.float_signed())
            .collect();
        let query = (0..dimension).map(|_| rng.float_signed()).collect();
        (keys, values, query)
    }

    fn run_cache(
        format: LatentStorageFormat,
        tokens: usize,
        dimension: usize,
        keys: &[f32],
        values: &[f32],
        query: &[f32],
    ) -> Vec<f32> {
        let mut cache = QuantizedLatentKvCache::new(
            tokens,
            dimension,
            dimension,
            dimension,
            format,
            format,
            identity(dimension, dimension),
            identity(dimension, dimension),
        )
        .unwrap();
        for token in 0..tokens {
            let offset = token * dimension;
            cache
                .append(
                    &keys[offset..offset + dimension],
                    &values[offset..offset + dimension],
                )
                .unwrap();
        }
        let mut output = vec![0.0; dimension];
        let mut scratch = LatentAttentionScratch::new(tokens, dimension, dimension);
        cache
            .attention_into(query, &mut output, &mut scratch)
            .unwrap();
        output
    }

    #[test]
    fn full_rank_f32_matches_contiguous_attention_bit_for_bit() {
        let (tokens, dimension) = (11, 8);
        let (keys, values, query) = seeded_vectors(tokens, dimension, 7);
        let expected = contiguous_attention(&keys, &values, &query, dimension, tokens);
        let actual = run_cache(
            LatentStorageFormat::F32,
            tokens,
            dimension,
            &keys,
            &values,
            &query,
        );
        assert_eq!(
            expected.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            actual.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn quantized_formats_stay_close_to_f32() {
        let (tokens, dimension) = (32, 16);
        let (keys, values, query) = seeded_vectors(tokens, dimension, 11);
        let expected = run_cache(
            LatentStorageFormat::F32,
            tokens,
            dimension,
            &keys,
            &values,
            &query,
        );
        let int8 = run_cache(
            LatentStorageFormat::Int8,
            tokens,
            dimension,
            &keys,
            &values,
            &query,
        );
        let int4 = run_cache(
            LatentStorageFormat::Int4,
            tokens,
            dimension,
            &keys,
            &values,
            &query,
        );
        let int8_error = expected
            .iter()
            .zip(&int8)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        let int4_error = expected
            .iter()
            .zip(&int4)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(int8_error < 0.015, "INT8 max error {int8_error}");
        assert!(int4_error < 0.20, "INT4 max error {int4_error}");
    }

    #[test]
    fn append_and_attention_are_bit_deterministic() {
        let (tokens, dimension) = (13, 7);
        let (keys, values, query) = seeded_vectors(tokens, dimension, 29);
        let first = run_cache(
            LatentStorageFormat::Int4,
            tokens,
            dimension,
            &keys,
            &values,
            &query,
        );
        let second = run_cache(
            LatentStorageFormat::Int4,
            tokens,
            dimension,
            &keys,
            &values,
            &query,
        );
        assert_eq!(
            first.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            second.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn fixed_allocation_does_not_change_after_append() {
        let (tokens, dimension) = (8, 6);
        let (keys, values, _) = seeded_vectors(tokens, dimension, 41);
        let mut cache = QuantizedLatentKvCache::new(
            tokens,
            dimension,
            4,
            3,
            LatentStorageFormat::Int4,
            LatentStorageFormat::Int8,
            identity(dimension, 4),
            identity(dimension, 3),
        )
        .unwrap();
        let allocation = cache.allocated_bytes();
        for token in 0..tokens {
            let offset = token * dimension;
            cache
                .append(
                    &keys[offset..offset + dimension],
                    &values[offset..offset + dimension],
                )
                .unwrap();
            assert_eq!(cache.allocated_bytes(), allocation);
        }
        assert_eq!(cache.len(), tokens);
        assert_eq!(
            cache.append(&keys[..dimension], &values[..dimension]),
            Err(LatentCacheError::CapacityExceeded { capacity: tokens })
        );
    }

    #[test]
    fn reduced_rank_int4_is_smaller_than_dense_capacity() {
        let capacity = 128;
        let dimension = 64;
        let cache = QuantizedLatentKvCache::new(
            capacity,
            dimension,
            12,
            16,
            LatentStorageFormat::Int4,
            LatentStorageFormat::Int4,
            identity(dimension, 12),
            identity(dimension, 16),
        )
        .unwrap();
        let dense_capacity_bytes = capacity * dimension * 2 * core::mem::size_of::<f32>();
        assert!(cache.allocated_bytes() < dense_capacity_bytes);
    }

    #[test]
    fn invalid_shapes_and_non_finite_inputs_are_rejected() {
        assert_eq!(
            QuantizedLatentKvCache::new(
                2,
                4,
                5,
                2,
                LatentStorageFormat::F32,
                LatentStorageFormat::F32,
                vec![0.0; 20],
                vec![0.0; 8],
            ),
            Err(LatentCacheError::RankTooLarge {
                name: "key",
                rank: 5,
                dimension: 4,
            })
        );

        let mut cache = QuantizedLatentKvCache::new(
            2,
            4,
            2,
            2,
            LatentStorageFormat::Int8,
            LatentStorageFormat::Int8,
            identity(4, 2),
            identity(4, 2),
        )
        .unwrap();
        assert_eq!(
            cache.append(&[0.0, f32::NAN, 0.0, 0.0], &[0.0; 4]),
            Err(LatentCacheError::NonFinite {
                name: "key",
                index: 1,
            })
        );
    }

    #[test]
    fn empty_cache_and_small_scratch_are_rejected() {
        let mut cache = QuantizedLatentKvCache::new(
            4,
            4,
            2,
            2,
            LatentStorageFormat::Int4,
            LatentStorageFormat::Int4,
            identity(4, 2),
            identity(4, 2),
        )
        .unwrap();
        let mut output = [0.0; 4];
        let mut scratch = LatentAttentionScratch::new(4, 2, 2);
        assert_eq!(
            cache.attention_into(&[0.0; 4], &mut output, &mut scratch),
            Err(LatentCacheError::EmptyCache)
        );
        cache.append(&[0.1; 4], &[0.2; 4]).unwrap();
        let mut small = LatentAttentionScratch::new(0, 1, 1);
        assert_eq!(
            cache.attention_into(&[0.0; 4], &mut output, &mut small),
            Err(LatentCacheError::ScratchTooSmall {
                name: "score",
                required: 1,
                available: 0,
            })
        );
    }
}
