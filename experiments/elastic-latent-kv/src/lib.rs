//! Deterministic reference implementation for elastic latent KV research.
//!
//! The crate contains no production integration. Its purpose is to provide
//! dense and fixed-rank latent attention oracles for differential testing.

#![forbid(unsafe_code)]

use core::fmt;

/// Errors returned by the Phase 0 reference caches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CacheError {
    /// A dimension or capacity was zero.
    ZeroDimension,
    /// A basis matrix did not have the required number of elements.
    InvalidBasisLength {
        /// Human-readable basis name.
        name: &'static str,
        /// Required number of elements.
        expected: usize,
        /// Supplied number of elements.
        actual: usize,
    },
    /// An input vector did not have the required number of elements.
    InvalidVectorLength {
        /// Human-readable input name.
        name: &'static str,
        /// Required number of elements.
        expected: usize,
        /// Supplied number of elements.
        actual: usize,
    },
    /// The cache has reached its fixed token capacity.
    CapacityExceeded {
        /// Maximum number of tokens.
        capacity: usize,
    },
    /// Attention was requested before any token was appended.
    EmptyCache,
    /// Scratch storage is too small for the requested operation.
    ScratchTooSmall {
        /// Human-readable scratch region.
        name: &'static str,
        /// Required number of elements.
        required: usize,
        /// Available number of elements.
        available: usize,
    },
}

impl fmt::Display for CacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::ZeroDimension => write!(formatter, "dimensions and capacity must be non-zero"),
            Self::InvalidBasisLength {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} basis length mismatch: expected {expected}, received {actual}"
            ),
            Self::InvalidVectorLength {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} length mismatch: expected {expected}, received {actual}"
            ),
            Self::CapacityExceeded { capacity } =>
            {
                write!(formatter, "cache token capacity exceeded: {capacity}")
            },
            Self::EmptyCache => write!(formatter, "attention requires at least one cached token"),
            Self::ScratchTooSmall {
                name,
                required,
                available,
            } => write!(
                formatter,
                "{name} scratch is too small: required {required}, available {available}"
            ),
        }
    }
}

impl std::error::Error for CacheError {}

/// Preallocated temporary storage shared by dense and latent attention.
#[derive(Debug, Clone)]
pub struct AttentionScratch {
    scores: Vec<f32>,
    query_latent: Vec<f32>,
    value_latent: Vec<f32>,
    dense_vector: Vec<f32>,
}

impl AttentionScratch {
    /// Creates scratch storage for a maximum token count, head dimension and rank.
    #[must_use]
    pub fn new(max_tokens: usize, head_dimension: usize, max_rank: usize) -> Self {
        Self {
            scores: vec![0.0; max_tokens],
            query_latent: vec![0.0; max_rank],
            value_latent: vec![0.0; max_rank],
            dense_vector: vec![0.0; head_dimension],
        }
    }

    fn require_scores(&self, required: usize) -> Result<(), CacheError> {
        if self.scores.len() < required
        {
            return Err(CacheError::ScratchTooSmall {
                name: "score",
                required,
                available: self.scores.len(),
            });
        }
        Ok(())
    }

    fn require_query_latent(&self, required: usize) -> Result<(), CacheError> {
        if self.query_latent.len() < required
        {
            return Err(CacheError::ScratchTooSmall {
                name: "query-latent",
                required,
                available: self.query_latent.len(),
            });
        }
        Ok(())
    }

    fn require_value_latent(&self, required: usize) -> Result<(), CacheError> {
        if self.value_latent.len() < required
        {
            return Err(CacheError::ScratchTooSmall {
                name: "value-latent",
                required,
                available: self.value_latent.len(),
            });
        }
        Ok(())
    }

    fn require_dense_vector(&self, required: usize) -> Result<(), CacheError> {
        if self.dense_vector.len() < required
        {
            return Err(CacheError::ScratchTooSmall {
                name: "dense-vector",
                required,
                available: self.dense_vector.len(),
            });
        }
        Ok(())
    }
}

/// Deterministic dense KV-cache oracle.
#[derive(Debug, Clone)]
pub struct DenseKvCache {
    capacity_tokens: usize,
    head_dimension: usize,
    keys: Vec<f32>,
    values: Vec<f32>,
    len: usize,
}

impl DenseKvCache {
    /// Creates an empty dense cache with fixed token capacity.
    pub fn new(capacity_tokens: usize, head_dimension: usize) -> Result<Self, CacheError> {
        if capacity_tokens == 0 || head_dimension == 0
        {
            return Err(CacheError::ZeroDimension);
        }

        let element_capacity = capacity_tokens
            .checked_mul(head_dimension)
            .ok_or(CacheError::ZeroDimension)?;

        Ok(Self {
            capacity_tokens,
            head_dimension,
            keys: Vec::with_capacity(element_capacity),
            values: Vec::with_capacity(element_capacity),
            len: 0,
        })
    }

    /// Returns the number of cached tokens.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the cache contains no token.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Appends one dense key and one dense value.
    pub fn append(&mut self, key: &[f32], value: &[f32]) -> Result<(), CacheError> {
        require_vector_length("key", key, self.head_dimension)?;
        require_vector_length("value", value, self.head_dimension)?;

        if self.len == self.capacity_tokens
        {
            return Err(CacheError::CapacityExceeded {
                capacity: self.capacity_tokens,
            });
        }

        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
        self.len += 1;
        Ok(())
    }

    /// Computes scaled dot-product attention over all cached tokens.
    pub fn attention(
        &self,
        query: &[f32],
        scale: f32,
        output: &mut [f32],
        scratch: &mut AttentionScratch,
    ) -> Result<(), CacheError> {
        if self.is_empty()
        {
            return Err(CacheError::EmptyCache);
        }

        require_vector_length("query", query, self.head_dimension)?;
        require_vector_length("output", output, self.head_dimension)?;
        scratch.require_scores(self.len)?;

        let scores = &mut scratch.scores[..self.len];

        for (token_index, score) in scores.iter_mut().enumerate()
        {
            let offset = token_index * self.head_dimension;
            let key = &self.keys[offset..offset + self.head_dimension];
            *score = scale * dot(query, key);
        }

        softmax_in_place(scores);
        output.fill(0.0);

        for (token_index, probability) in scores.iter().copied().enumerate()
        {
            let offset = token_index * self.head_dimension;
            let value = &self.values[offset..offset + self.head_dimension];

            for (output_element, value_element) in output.iter_mut().zip(value)
            {
                *output_element += probability * value_element;
            }
        }

        Ok(())
    }
}

/// Fixed-rank latent KV-cache reference implementation.
#[derive(Debug, Clone)]
pub struct FixedRankLatentCache {
    capacity_tokens: usize,
    head_dimension: usize,
    key_rank: usize,
    value_rank: usize,
    key_basis: Vec<f32>,
    value_basis: Vec<f32>,
    key_coefficients: Vec<f32>,
    value_coefficients: Vec<f32>,
    len: usize,
}

impl FixedRankLatentCache {
    /// Creates an empty latent cache.
    ///
    /// Each basis is row-major with shape `[head_dimension, rank]`.
    pub fn new(
        capacity_tokens: usize,
        head_dimension: usize,
        key_rank: usize,
        value_rank: usize,
        key_basis: Vec<f32>,
        value_basis: Vec<f32>,
    ) -> Result<Self, CacheError> {
        if capacity_tokens == 0 || head_dimension == 0 || key_rank == 0 || value_rank == 0
        {
            return Err(CacheError::ZeroDimension);
        }

        let expected_key_basis = head_dimension
            .checked_mul(key_rank)
            .ok_or(CacheError::ZeroDimension)?;
        let expected_value_basis = head_dimension
            .checked_mul(value_rank)
            .ok_or(CacheError::ZeroDimension)?;

        require_basis_length("key", &key_basis, expected_key_basis)?;
        require_basis_length("value", &value_basis, expected_value_basis)?;

        let key_capacity = capacity_tokens
            .checked_mul(key_rank)
            .ok_or(CacheError::ZeroDimension)?;
        let value_capacity = capacity_tokens
            .checked_mul(value_rank)
            .ok_or(CacheError::ZeroDimension)?;

        Ok(Self {
            capacity_tokens,
            head_dimension,
            key_rank,
            value_rank,
            key_basis,
            value_basis,
            key_coefficients: Vec::with_capacity(key_capacity),
            value_coefficients: Vec::with_capacity(value_capacity),
            len: 0,
        })
    }

    /// Returns the number of cached tokens.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the cache contains no token.
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

    /// Appends one pair of key and value coefficient vectors.
    pub fn append_coefficients(
        &mut self,
        key_coefficients: &[f32],
        value_coefficients: &[f32],
    ) -> Result<(), CacheError> {
        require_vector_length("key coefficients", key_coefficients, self.key_rank)?;
        require_vector_length("value coefficients", value_coefficients, self.value_rank)?;

        if self.len == self.capacity_tokens
        {
            return Err(CacheError::CapacityExceeded {
                capacity: self.capacity_tokens,
            });
        }

        self.key_coefficients.extend_from_slice(key_coefficients);
        self.value_coefficients
            .extend_from_slice(value_coefficients);
        self.len += 1;
        Ok(())
    }

    /// Computes attention by explicitly reconstructing every latent key and value.
    ///
    /// This is the fixed-rank differential oracle, not the optimized path.
    pub fn attention_explicit(
        &self,
        query: &[f32],
        scale: f32,
        output: &mut [f32],
        scratch: &mut AttentionScratch,
    ) -> Result<(), CacheError> {
        self.validate_attention_inputs(query, output, scratch)?;
        scratch.require_dense_vector(self.head_dimension)?;

        let scores = &mut scratch.scores[..self.len];

        for (token_index, score) in scores.iter_mut().enumerate()
        {
            let coefficient_offset = token_index * self.key_rank;
            let coefficients =
                &self.key_coefficients[coefficient_offset..coefficient_offset + self.key_rank];

            matrix_vector(
                &self.key_basis,
                self.head_dimension,
                self.key_rank,
                coefficients,
                &mut scratch.dense_vector[..self.head_dimension],
            );

            *score = scale * dot(query, &scratch.dense_vector[..self.head_dimension]);
        }

        softmax_in_place(scores);
        output.fill(0.0);

        for (token_index, probability) in scores.iter().copied().enumerate()
        {
            let coefficient_offset = token_index * self.value_rank;
            let coefficients =
                &self.value_coefficients[coefficient_offset..coefficient_offset + self.value_rank];

            matrix_vector(
                &self.value_basis,
                self.head_dimension,
                self.value_rank,
                coefficients,
                &mut scratch.dense_vector[..self.head_dimension],
            );

            for (output_element, value_element) in output
                .iter_mut()
                .zip(&scratch.dense_vector[..self.head_dimension])
            {
                *output_element += probability * value_element;
            }
        }

        Ok(())
    }

    /// Computes attention without reconstructing keys or per-token values.
    ///
    /// The query is projected once into key-latent space. Values are accumulated
    /// in value-latent space and up-projected once after softmax aggregation.
    pub fn attention_reconstruction_free(
        &self,
        query: &[f32],
        scale: f32,
        output: &mut [f32],
        scratch: &mut AttentionScratch,
    ) -> Result<(), CacheError> {
        self.validate_attention_inputs(query, output, scratch)?;
        scratch.require_query_latent(self.key_rank)?;
        scratch.require_value_latent(self.value_rank)?;

        transpose_matrix_vector(
            &self.key_basis,
            self.head_dimension,
            self.key_rank,
            query,
            &mut scratch.query_latent[..self.key_rank],
        );

        let scores = &mut scratch.scores[..self.len];

        for (token_index, score) in scores.iter_mut().enumerate()
        {
            let coefficient_offset = token_index * self.key_rank;
            let coefficients =
                &self.key_coefficients[coefficient_offset..coefficient_offset + self.key_rank];

            *score = scale * dot(&scratch.query_latent[..self.key_rank], coefficients);
        }

        softmax_in_place(scores);

        let latent_value = &mut scratch.value_latent[..self.value_rank];
        latent_value.fill(0.0);

        for (token_index, probability) in scores.iter().copied().enumerate()
        {
            let coefficient_offset = token_index * self.value_rank;
            let coefficients =
                &self.value_coefficients[coefficient_offset..coefficient_offset + self.value_rank];

            for (accumulator, coefficient) in latent_value.iter_mut().zip(coefficients)
            {
                *accumulator += probability * coefficient;
            }
        }

        matrix_vector(
            &self.value_basis,
            self.head_dimension,
            self.value_rank,
            latent_value,
            output,
        );

        Ok(())
    }

    fn validate_attention_inputs(
        &self,
        query: &[f32],
        output: &[f32],
        scratch: &AttentionScratch,
    ) -> Result<(), CacheError> {
        if self.is_empty()
        {
            return Err(CacheError::EmptyCache);
        }

        require_vector_length("query", query, self.head_dimension)?;
        require_vector_length("output", output, self.head_dimension)?;
        scratch.require_scores(self.len)?;
        Ok(())
    }
}

fn require_basis_length(
    name: &'static str,
    basis: &[f32],
    expected: usize,
) -> Result<(), CacheError> {
    if basis.len() != expected
    {
        return Err(CacheError::InvalidBasisLength {
            name,
            expected,
            actual: basis.len(),
        });
    }
    Ok(())
}

fn require_vector_length(
    name: &'static str,
    vector: &[f32],
    expected: usize,
) -> Result<(), CacheError> {
    if vector.len() != expected
    {
        return Err(CacheError::InvalidVectorLength {
            name,
            expected,
            actual: vector.len(),
        });
    }
    Ok(())
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());

    left.iter()
        .zip(right)
        .map(|(left_element, right_element)| left_element * right_element)
        .sum()
}

fn matrix_vector(matrix: &[f32], rows: usize, columns: usize, vector: &[f32], output: &mut [f32]) {
    debug_assert_eq!(matrix.len(), rows * columns);
    debug_assert_eq!(vector.len(), columns);
    debug_assert_eq!(output.len(), rows);

    for (row_index, output_element) in output.iter_mut().enumerate()
    {
        let row_offset = row_index * columns;
        let row = &matrix[row_offset..row_offset + columns];
        *output_element = dot(row, vector);
    }
}

fn transpose_matrix_vector(
    matrix: &[f32],
    rows: usize,
    columns: usize,
    vector: &[f32],
    output: &mut [f32],
) {
    debug_assert_eq!(matrix.len(), rows * columns);
    debug_assert_eq!(vector.len(), rows);
    debug_assert_eq!(output.len(), columns);

    output.fill(0.0);

    for (row_index, vector_element) in vector.iter().copied().enumerate()
    {
        let row_offset = row_index * columns;

        for column_index in 0..columns
        {
            output[column_index] += matrix[row_offset + column_index] * vector_element;
        }
    }
}

fn softmax_in_place(values: &mut [f32]) {
    debug_assert!(!values.is_empty());

    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut sum = 0.0_f32;
    for value in values.iter_mut()
    {
        *value = (*value - maximum).exp();
        sum += *value;
    }

    for value in values
    {
        *value /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::{AttentionScratch, CacheError, DenseKvCache, FixedRankLatentCache};

    const TOLERANCE: f32 = 2.0e-6;

    fn assert_close(left: &[f32], right: &[f32]) {
        assert_eq!(left.len(), right.len());

        for (index, (left_element, right_element)) in left.iter().zip(right).enumerate()
        {
            let difference = (left_element - right_element).abs();
            assert!(
                difference <= TOLERANCE,
                "index {index}: left={left_element}, right={right_element}, \
                 difference={difference}, tolerance={TOLERANCE}"
            );
        }
    }

    fn identity(dimension: usize) -> Vec<f32> {
        let mut matrix = vec![0.0; dimension * dimension];

        for diagonal in 0..dimension
        {
            matrix[diagonal * dimension + diagonal] = 1.0;
        }

        matrix
    }

    #[test]
    fn identity_latent_basis_matches_dense_attention() {
        let dimension = 3;
        let capacity = 4;
        let scale = 1.0 / (dimension as f32).sqrt();

        let keys = [
            [0.25, -0.50, 1.00],
            [1.25, 0.75, -0.25],
            [-0.75, 0.50, 0.80],
        ];
        let values = [
            [2.00, -1.00, 0.50],
            [0.25, 1.50, -0.75],
            [-1.25, 0.40, 2.25],
        ];
        let query = [0.70, -0.20, 0.90];

        let mut dense = DenseKvCache::new(capacity, dimension).unwrap();
        let mut latent = FixedRankLatentCache::new(
            capacity,
            dimension,
            dimension,
            dimension,
            identity(dimension),
            identity(dimension),
        )
        .unwrap();

        for (key, value) in keys.iter().zip(values.iter())
        {
            dense.append(key, value).unwrap();
            latent.append_coefficients(key, value).unwrap();
        }

        let mut dense_output = [0.0; 3];
        let mut latent_output = [0.0; 3];
        let mut dense_scratch = AttentionScratch::new(capacity, dimension, dimension);
        let mut latent_scratch = AttentionScratch::new(capacity, dimension, dimension);

        dense
            .attention(&query, scale, &mut dense_output, &mut dense_scratch)
            .unwrap();
        latent
            .attention_reconstruction_free(&query, scale, &mut latent_output, &mut latent_scratch)
            .unwrap();

        assert_close(&dense_output, &latent_output);
    }

    #[test]
    fn reconstruction_free_matches_explicit_fixed_rank_oracle() {
        let dimension = 4;
        let key_rank = 2;
        let value_rank = 3;
        let capacity = 5;
        let scale = 0.5;

        let key_basis = vec![
            1.00, 0.20, //
            0.10, 0.90, //
            -0.40, 0.30, //
            0.70, -0.20,
        ];
        let value_basis = vec![
            1.00, 0.00, 0.25, //
            0.10, 0.80, -0.30, //
            -0.50, 0.20, 1.10, //
            0.70, -0.40, 0.15,
        ];

        let key_coefficients = [[0.50, -0.25], [1.20, 0.30], [-0.70, 0.90], [0.10, -1.10]];
        let value_coefficients = [
            [0.25, -0.50, 1.00],
            [1.10, 0.20, -0.40],
            [-0.75, 0.80, 0.30],
            [0.60, -0.10, 0.90],
        ];
        let query = [0.80, -0.30, 0.55, 1.10];

        let mut cache = FixedRankLatentCache::new(
            capacity,
            dimension,
            key_rank,
            value_rank,
            key_basis,
            value_basis,
        )
        .unwrap();

        for (key, value) in key_coefficients.iter().zip(value_coefficients.iter())
        {
            cache.append_coefficients(key, value).unwrap();
        }

        let mut explicit_output = [0.0; 4];
        let mut reconstruction_free_output = [0.0; 4];
        let mut explicit_scratch =
            AttentionScratch::new(capacity, dimension, value_rank.max(key_rank));
        let mut reconstruction_free_scratch =
            AttentionScratch::new(capacity, dimension, value_rank.max(key_rank));

        cache
            .attention_explicit(&query, scale, &mut explicit_output, &mut explicit_scratch)
            .unwrap();
        cache
            .attention_reconstruction_free(
                &query,
                scale,
                &mut reconstruction_free_output,
                &mut reconstruction_free_scratch,
            )
            .unwrap();

        assert_close(&explicit_output, &reconstruction_free_output);
    }

    #[test]
    fn repeated_reconstruction_free_execution_is_bit_deterministic() {
        let mut cache = FixedRankLatentCache::new(2, 2, 2, 2, identity(2), identity(2)).unwrap();

        cache
            .append_coefficients(&[0.25, 1.0], &[1.5, -0.25])
            .unwrap();
        cache
            .append_coefficients(&[-0.5, 0.75], &[0.2, 2.0])
            .unwrap();

        let query = [0.8, -0.3];
        let mut first = [0.0; 2];
        let mut second = [0.0; 2];
        let mut scratch = AttentionScratch::new(2, 2, 2);

        cache
            .attention_reconstruction_free(&query, 0.5, &mut first, &mut scratch)
            .unwrap();
        cache
            .attention_reconstruction_free(&query, 0.5, &mut second, &mut scratch)
            .unwrap();

        assert_eq!(first.map(f32::to_bits), second.map(f32::to_bits));
    }

    #[test]
    fn fixed_capacity_is_enforced() {
        let mut cache =
            FixedRankLatentCache::new(1, 2, 1, 1, vec![1.0, 0.0], vec![0.0, 1.0]).unwrap();

        cache.append_coefficients(&[1.0], &[2.0]).unwrap();

        assert_eq!(
            cache.append_coefficients(&[3.0], &[4.0]),
            Err(CacheError::CapacityExceeded { capacity: 1 })
        );
    }

    #[test]
    fn invalid_basis_shape_is_rejected() {
        let error = FixedRankLatentCache::new(2, 3, 2, 1, vec![0.0; 5], vec![0.0; 3]).unwrap_err();

        assert_eq!(
            error,
            CacheError::InvalidBasisLength {
                name: "key",
                expected: 6,
                actual: 5,
            }
        );
    }

    #[derive(Debug, Clone)]
    struct DeterministicRng {
        state: u64,
    }

    impl DeterministicRng {
        fn new(seed: u64) -> Self {
            Self { state: seed.max(1) }
        }

        fn next_u64(&mut self) -> u64 {
            let mut value = self.state;
            value ^= value >> 12;
            value ^= value << 25;
            value ^= value >> 27;
            self.state = value;
            value.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn next_symmetric_f32(&mut self, amplitude: f32) -> f32 {
            const MANTISSA_MASK: u64 = (1_u64 << 24) - 1;
            const DENOMINATOR: f32 = MANTISSA_MASK as f32;

            let mantissa = (self.next_u64() >> 40) & MANTISSA_MASK;
            let unit_interval = mantissa as f32 / DENOMINATOR;

            (2.0 * unit_interval - 1.0) * amplitude
        }
    }

    fn random_vector(rng: &mut DeterministicRng, length: usize, amplitude: f32) -> Vec<f32> {
        (0..length)
            .map(|_| rng.next_symmetric_f32(amplitude))
            .collect()
    }

    fn assert_close_with_tolerance(
        left: &[f32],
        right: &[f32],
        absolute_tolerance: f32,
        relative_tolerance: f32,
        family: &str,
        case_index: usize,
    ) {
        assert_eq!(left.len(), right.len());

        for (element_index, (left_element, right_element)) in left.iter().zip(right).enumerate()
        {
            let difference = (left_element - right_element).abs();
            let magnitude = left_element.abs().max(right_element.abs());
            let tolerance = absolute_tolerance + relative_tolerance * magnitude;

            assert!(
                difference <= tolerance,
                "{family} case {case_index}, element {element_index}: \
                 left={left_element}, right={right_element}, \
                 difference={difference}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn generated_identity_bases_match_dense_oracle() {
        let dimensions = [1_usize, 2, 3, 8, 16];
        let token_counts = [1_usize, 2, 7];
        let mut rng = DeterministicRng::new(0x5C1A_57E1_CAFE_BABE);
        let mut case_index = 0;

        for dimension in dimensions
        {
            for token_count in token_counts
            {
                let scale = (dimension as f32).sqrt().recip();
                let basis = identity(dimension);

                let mut dense = DenseKvCache::new(token_count, dimension).unwrap();
                let mut latent = FixedRankLatentCache::new(
                    token_count,
                    dimension,
                    dimension,
                    dimension,
                    basis.clone(),
                    basis.clone(),
                )
                .unwrap();

                for _ in 0..token_count
                {
                    let key = random_vector(&mut rng, dimension, 1.25);
                    let value = random_vector(&mut rng, dimension, 1.25);

                    dense.append(&key, &value).unwrap();
                    latent.append_coefficients(&key, &value).unwrap();
                }

                let query = random_vector(&mut rng, dimension, 1.25);
                let mut dense_output = vec![0.0; dimension];
                let mut latent_output = vec![0.0; dimension];
                let mut dense_scratch = AttentionScratch::new(token_count, dimension, dimension);
                let mut latent_scratch = AttentionScratch::new(token_count, dimension, dimension);

                dense
                    .attention(&query, scale, &mut dense_output, &mut dense_scratch)
                    .unwrap();

                latent
                    .attention_reconstruction_free(
                        &query,
                        scale,
                        &mut latent_output,
                        &mut latent_scratch,
                    )
                    .unwrap();

                assert_close_with_tolerance(
                    &dense_output,
                    &latent_output,
                    3.0e-6,
                    3.0e-6,
                    "identity",
                    case_index,
                );

                case_index += 1;
            }
        }

        assert_eq!(case_index, 15);
    }

    #[test]
    fn generated_fixed_rank_sweep_matches_explicit_oracle() {
        let dimensions = [1_usize, 2, 3, 4, 7, 8, 16, 32];
        let amplitudes = [1.0e-3_f32, 5.0e-2, 5.0e-1, 1.5];
        let mut rng = DeterministicRng::new(0xE1A5_71C0_5A11_D00D);

        for case_index in 0..64
        {
            let dimension = dimensions[case_index % dimensions.len()];
            let key_rank = 1 + ((case_index * 5 + 1) % dimension);
            let value_rank = 1 + ((case_index * 7 + 2) % dimension);
            let token_count = 1 + ((case_index * 3 + 1) % 13);
            let amplitude = amplitudes[(case_index * 11 + 3) % amplitudes.len()];
            let scale = (dimension as f32).sqrt().recip();

            let key_basis = random_vector(&mut rng, dimension * key_rank, amplitude);
            let value_basis = random_vector(&mut rng, dimension * value_rank, amplitude);

            let mut cache = FixedRankLatentCache::new(
                token_count,
                dimension,
                key_rank,
                value_rank,
                key_basis,
                value_basis,
            )
            .unwrap();

            for _ in 0..token_count
            {
                let key_coefficients = random_vector(&mut rng, key_rank, amplitude);
                let value_coefficients = random_vector(&mut rng, value_rank, amplitude);

                cache
                    .append_coefficients(&key_coefficients, &value_coefficients)
                    .unwrap();
            }

            let query = random_vector(&mut rng, dimension, amplitude);
            let max_rank = key_rank.max(value_rank);

            let mut explicit_output = vec![0.0; dimension];
            let mut reconstruction_free_output = vec![0.0; dimension];
            let mut explicit_scratch = AttentionScratch::new(token_count, dimension, max_rank);
            let mut reconstruction_free_scratch =
                AttentionScratch::new(token_count, dimension, max_rank);

            cache
                .attention_explicit(&query, scale, &mut explicit_output, &mut explicit_scratch)
                .unwrap();

            cache
                .attention_reconstruction_free(
                    &query,
                    scale,
                    &mut reconstruction_free_output,
                    &mut reconstruction_free_scratch,
                )
                .unwrap();

            assert_close_with_tolerance(
                &explicit_output,
                &reconstruction_free_output,
                5.0e-5,
                5.0e-5,
                "fixed-rank",
                case_index,
            );
        }
    }

    #[test]
    fn empty_cache_and_undersized_scratch_are_rejected() {
        let query = [0.25, -0.50, 0.75];
        let mut output = [0.0; 3];

        let dense = DenseKvCache::new(2, 3).unwrap();
        let mut dense_scratch = AttentionScratch::new(2, 3, 3);

        assert_eq!(
            dense.attention(&query, 1.0, &mut output, &mut dense_scratch,),
            Err(CacheError::EmptyCache)
        );

        let mut latent = FixedRankLatentCache::new(
            1,
            3,
            2,
            2,
            vec![
                1.0, 0.0, //
                0.0, 1.0, //
                0.5, 0.5,
            ],
            vec![
                1.0, 0.0, //
                0.0, 1.0, //
                0.5, -0.5,
            ],
        )
        .unwrap();

        latent
            .append_coefficients(&[0.25, 0.75], &[1.0, -0.5])
            .unwrap();

        let mut undersized = AttentionScratch::new(1, 3, 1);

        assert_eq!(
            latent.attention_reconstruction_free(&query, 1.0, &mut output, &mut undersized,),
            Err(CacheError::ScratchTooSmall {
                name: "query-latent",
                required: 2,
                available: 1,
            })
        );
    }
}

/// Deterministic Phase 1 measurement harness.
pub mod phase1;

/// Deterministic Phase 2 dense-to-latent projection.
pub mod phase2;
