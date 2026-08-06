//! Reconstruction-free quantized latent KV-cache with sparse residual channels.
//!
//! Phase 8 keeps the Phase 7 latent coefficient path intact and augments it with
//! independently configured fixed-slot key and value residuals. Residual indices
//! are stored as `u16`; residual values use FP32, symmetric INT8, or packed
//! symmetric INT4. Key residuals correct scores directly, while value residuals
//! are scattered into the dense output after exactly one latent up-projection.
//!
//! All persistent storage and append/attention scratch are allocated at
//! construction. Appending tokens and attention evaluation do not grow buffers.

use crate::nn::latent_kv_cache::{LatentCacheError, LatentStorageFormat};
use core::fmt;

const EMPTY_INDEX: u16 = u16::MAX;
const MAXIMUM_DIMENSION: usize = u16::MAX as usize;

/// Fixed-slot sparse residual configuration for one key or value channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseResidualConfig {
    slots_per_token: usize,
    format: LatentStorageFormat,
}

impl SparseResidualConfig {
    /// Creates a residual channel with the supplied fixed slots and value format.
    #[must_use]
    pub const fn new(slots_per_token: usize, format: LatentStorageFormat) -> Self {
        Self {
            slots_per_token,
            format,
        }
    }

    /// Disables sparse residual storage.
    #[must_use]
    pub const fn disabled() -> Self {
        Self::new(0, LatentStorageFormat::F32)
    }

    /// Returns reserved residual slots per token.
    #[must_use]
    pub const fn slots_per_token(self) -> usize {
        self.slots_per_token
    }

    /// Returns the residual value storage format.
    #[must_use]
    pub const fn format(self) -> LatentStorageFormat {
        self.format
    }
}

/// Errors returned by the sparse-residual latent cache.
#[derive(Debug, Clone, PartialEq)]
pub enum ResidualLatentCacheError {
    /// A shared latent-cache validation failed.
    Latent(LatentCacheError),
    /// A residual slot count exceeded the dense dimension.
    ResidualSlotsTooLarge {
        /// Human-readable channel name.
        name: &'static str,
        /// Supplied fixed slots per token.
        slots: usize,
        /// Dense head dimension.
        dimension: usize,
    },
    /// The dense dimension cannot be encoded while reserving `u16::MAX` as empty.
    DimensionTooLarge {
        /// Supplied dense dimension.
        dimension: usize,
        /// Largest accepted dense dimension.
        maximum: usize,
    },
}

impl fmt::Display for ResidualLatentCacheError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::Latent(error) => write!(output, "{error}"),
            Self::ResidualSlotsTooLarge {
                name,
                slots,
                dimension,
            } => write!(
                output,
                "{name} residual slots {slots} exceed head dimension {dimension}"
            ),
            Self::DimensionTooLarge { dimension, maximum } => write!(
                output,
                "head dimension {dimension} exceeds residual-index maximum {maximum}"
            ),
        }
    }
}

impl std::error::Error for ResidualLatentCacheError {}

impl From<LatentCacheError> for ResidualLatentCacheError {
    fn from(error: LatentCacheError) -> Self {
        Self::Latent(error)
    }
}

/// Reusable scratch for reconstruction-free sparse-residual attention.
#[derive(Debug, Clone)]
pub struct ResidualLatentAttentionScratch {
    scores: Vec<f32>,
    query_latent: Vec<f32>,
    value_latent: Vec<f32>,
}

impl ResidualLatentAttentionScratch {
    /// Allocates attention scratch for the configured capacity and ranks.
    #[must_use]
    pub fn new(max_tokens: usize, key_rank: usize, value_rank: usize) -> Self {
        Self {
            scores: vec![0.0; max_tokens],
            query_latent: vec![0.0; key_rank],
            value_latent: vec![0.0; value_rank],
        }
    }

    fn validate(
        &self,
        tokens: usize,
        key_rank: usize,
        value_rank: usize,
    ) -> Result<(), ResidualLatentCacheError> {
        require_scratch("score", self.scores.len(), tokens)?;
        require_scratch("query-latent", self.query_latent.len(), key_rank)?;
        require_scratch("value-latent", self.value_latent.len(), value_rank)
    }
}

/// Fixed-capacity latent KV-cache with deterministic sparse residual correction.
#[derive(Debug, Clone)]
pub struct ResidualQuantizedLatentKvCache {
    capacity_tokens: usize,
    dimension: usize,
    key_rank: usize,
    value_rank: usize,
    key_format: LatentStorageFormat,
    value_format: LatentStorageFormat,
    key_residual: SparseResidualConfig,
    value_residual: SparseResidualConfig,
    key_basis: Vec<f32>,
    value_basis: Vec<f32>,
    key_payload: Vec<u8>,
    value_payload: Vec<u8>,
    key_scales: Vec<f32>,
    value_scales: Vec<f32>,
    key_residual_indices: Vec<u16>,
    value_residual_indices: Vec<u16>,
    key_residual_payload: Vec<u8>,
    value_residual_payload: Vec<u8>,
    key_residual_scales: Vec<f32>,
    value_residual_scales: Vec<f32>,
    key_projection: Vec<f32>,
    value_projection: Vec<f32>,
    key_reconstruction: Vec<f32>,
    value_reconstruction: Vec<f32>,
    key_residual_values: Vec<f32>,
    value_residual_values: Vec<f32>,
    selected_coordinates: Vec<bool>,
    len: usize,
}

impl ResidualQuantizedLatentKvCache {
    /// Creates an empty cache with row-major bases shaped `[dimension, rank]`.
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
        non_zero("capacity_tokens", capacity_tokens)?;
        non_zero("dimension", dimension)?;
        non_zero("key_rank", key_rank)?;
        non_zero("value_rank", value_rank)?;
        require_rank("key", key_rank, dimension)?;
        require_rank("value", value_rank, dimension)?;
        require_dimension(dimension)?;
        require_slots("key", key_residual.slots_per_token, dimension)?;
        require_slots("value", value_residual.slots_per_token, dimension)?;

        let expected_key_basis = checked_product(dimension, key_rank)?;
        let expected_value_basis = checked_product(dimension, value_rank)?;
        require_length("key basis", key_basis.len(), expected_key_basis)?;
        require_length("value basis", value_basis.len(), expected_value_basis)?;
        require_finite("key basis", &key_basis)?;
        require_finite("value basis", &value_basis)?;

        let key_payload =
            vec![0_u8; checked_product(capacity_tokens, row_bytes(key_format, key_rank)?)?];
        let value_payload =
            vec![0_u8; checked_product(capacity_tokens, row_bytes(value_format, value_rank)?)?];
        let key_residual_indices =
            vec![EMPTY_INDEX; checked_product(capacity_tokens, key_residual.slots_per_token)?];
        let value_residual_indices =
            vec![EMPTY_INDEX; checked_product(capacity_tokens, value_residual.slots_per_token)?];
        let key_residual_payload = vec![
            0_u8;
            checked_product(
                capacity_tokens,
                row_bytes(key_residual.format, key_residual.slots_per_token)?,
            )?
        ];
        let value_residual_payload = vec![
            0_u8;
            checked_product(
                capacity_tokens,
                row_bytes(value_residual.format, value_residual.slots_per_token)?,
            )?
        ];

        Ok(Self {
            capacity_tokens,
            dimension,
            key_rank,
            value_rank,
            key_format,
            value_format,
            key_residual,
            value_residual,
            key_basis,
            value_basis,
            key_payload,
            value_payload,
            key_scales: scales_for(key_format, key_rank, capacity_tokens),
            value_scales: scales_for(value_format, value_rank, capacity_tokens),
            key_residual_indices,
            value_residual_indices,
            key_residual_payload,
            value_residual_payload,
            key_residual_scales: scales_for(
                key_residual.format,
                key_residual.slots_per_token,
                capacity_tokens,
            ),
            value_residual_scales: scales_for(
                value_residual.format,
                value_residual.slots_per_token,
                capacity_tokens,
            ),
            key_projection: vec![0.0; key_rank],
            value_projection: vec![0.0; value_rank],
            key_reconstruction: vec![0.0; dimension],
            value_reconstruction: vec![0.0; dimension],
            key_residual_values: vec![0.0; key_residual.slots_per_token],
            value_residual_values: vec![0.0; value_residual.slots_per_token],
            selected_coordinates: vec![false; dimension],
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

    /// Returns the key coefficient format.
    #[must_use]
    pub const fn key_format(&self) -> LatentStorageFormat {
        self.key_format
    }

    /// Returns the value coefficient format.
    #[must_use]
    pub const fn value_format(&self) -> LatentStorageFormat {
        self.value_format
    }

    /// Returns the key residual configuration.
    #[must_use]
    pub const fn key_residual_config(&self) -> SparseResidualConfig {
        self.key_residual
    }

    /// Returns the value residual configuration.
    #[must_use]
    pub const fn value_residual_config(&self) -> SparseResidualConfig {
        self.value_residual
    }

    /// Returns the key residual indices for one resident token.
    #[must_use]
    pub fn key_residual_indices_for(&self, token: usize) -> Option<&[u16]> {
        residual_indices_for(
            &self.key_residual_indices,
            self.key_residual.slots_per_token,
            self.len,
            token,
        )
    }

    /// Returns the value residual indices for one resident token.
    #[must_use]
    pub fn value_residual_indices_for(&self, token: usize) -> Option<&[u16]> {
        residual_indices_for(
            &self.value_residual_indices,
            self.value_residual.slots_per_token,
            self.len,
            token,
        )
    }

    /// Returns logical persistent bytes used by resident tokens and shared bases.
    #[must_use]
    pub fn used_bytes(&self) -> usize {
        let basis_bytes = (self.key_basis.len() + self.value_basis.len())
            .saturating_mul(core::mem::size_of::<f32>());
        let coefficient_bytes = self
            .len
            .saturating_mul(row_bytes_unchecked(self.key_format, self.key_rank))
            .saturating_add(
                self.len
                    .saturating_mul(row_bytes_unchecked(self.value_format, self.value_rank)),
            );
        let coefficient_scales = scale_count(self.key_format, self.key_rank, self.len)
            .saturating_add(scale_count(self.value_format, self.value_rank, self.len))
            .saturating_mul(core::mem::size_of::<f32>());
        let residual_slots = self
            .len
            .saturating_mul(self.key_residual.slots_per_token)
            .saturating_add(self.len.saturating_mul(self.value_residual.slots_per_token));
        let residual_indices = residual_slots.saturating_mul(core::mem::size_of::<u16>());
        let residual_payload = self
            .len
            .saturating_mul(row_bytes_unchecked(
                self.key_residual.format,
                self.key_residual.slots_per_token,
            ))
            .saturating_add(self.len.saturating_mul(row_bytes_unchecked(
                self.value_residual.format,
                self.value_residual.slots_per_token,
            )));
        let residual_scales = scale_count(
            self.key_residual.format,
            self.key_residual.slots_per_token,
            self.len,
        )
        .saturating_add(scale_count(
            self.value_residual.format,
            self.value_residual.slots_per_token,
            self.len,
        ))
        .saturating_mul(core::mem::size_of::<f32>());

        basis_bytes
            .saturating_add(coefficient_bytes)
            .saturating_add(coefficient_scales)
            .saturating_add(residual_indices)
            .saturating_add(residual_payload)
            .saturating_add(residual_scales)
    }

    /// Returns the actual fixed allocation owned by the cache in bytes.
    #[must_use]
    pub fn allocated_bytes(&self) -> usize {
        let byte_capacity = self
            .key_payload
            .capacity()
            .saturating_add(self.value_payload.capacity())
            .saturating_add(self.key_residual_payload.capacity())
            .saturating_add(self.value_residual_payload.capacity());
        let index_capacity = self
            .key_residual_indices
            .capacity()
            .saturating_add(self.value_residual_indices.capacity())
            .saturating_mul(core::mem::size_of::<u16>());
        let float_capacity = self
            .key_basis
            .capacity()
            .saturating_add(self.value_basis.capacity())
            .saturating_add(self.key_scales.capacity())
            .saturating_add(self.value_scales.capacity())
            .saturating_add(self.key_residual_scales.capacity())
            .saturating_add(self.value_residual_scales.capacity())
            .saturating_add(self.key_projection.capacity())
            .saturating_add(self.value_projection.capacity())
            .saturating_add(self.key_reconstruction.capacity())
            .saturating_add(self.value_reconstruction.capacity())
            .saturating_add(self.key_residual_values.capacity())
            .saturating_add(self.value_residual_values.capacity())
            .saturating_mul(core::mem::size_of::<f32>());
        let selection_capacity =
            self.selected_coordinates.capacity() * core::mem::size_of::<bool>();

        byte_capacity
            .saturating_add(index_capacity)
            .saturating_add(float_capacity)
            .saturating_add(selection_capacity)
    }

    /// Projects and appends one dense key/value pair without growing storage.
    pub fn append(&mut self, key: &[f32], value: &[f32]) -> Result<(), ResidualLatentCacheError> {
        require_length("key", key.len(), self.dimension)?;
        require_length("value", value.len(), self.dimension)?;
        require_finite("key", key)?;
        require_finite("value", value)?;
        if self.len == self.capacity_tokens
        {
            return Err(LatentCacheError::CapacityExceeded {
                capacity: self.capacity_tokens,
            }
            .into());
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

        select_and_encode_residuals(
            key,
            &self.key_basis,
            self.dimension,
            self.key_rank,
            &self.key_projection,
            self.key_residual,
            self.len,
            &mut self.key_reconstruction,
            &mut self.key_residual_values,
            &mut self.selected_coordinates,
            &mut self.key_residual_indices,
            &mut self.key_residual_payload,
            &mut self.key_residual_scales,
        );
        select_and_encode_residuals(
            value,
            &self.value_basis,
            self.dimension,
            self.value_rank,
            &self.value_projection,
            self.value_residual,
            self.len,
            &mut self.value_reconstruction,
            &mut self.value_residual_values,
            &mut self.selected_coordinates,
            &mut self.value_residual_indices,
            &mut self.value_residual_payload,
            &mut self.value_residual_scales,
        );

        self.len += 1;
        Ok(())
    }

    /// Computes attention from latent rows plus sparse residual corrections.
    pub fn attention_into(
        &self,
        query: &[f32],
        output: &mut [f32],
        scratch: &mut ResidualLatentAttentionScratch,
    ) -> Result<(), ResidualLatentCacheError> {
        if self.is_empty()
        {
            return Err(LatentCacheError::EmptyCache.into());
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

        let attention_scale = 1.0 / (self.dimension as f32).sqrt();
        let query_latent = &scratch.query_latent[..self.key_rank];
        let scores = &mut scratch.scores[..self.len];
        for (row, score) in scores.iter_mut().enumerate()
        {
            let latent_score = row_dot(
                self.key_format,
                self.key_rank,
                row,
                &self.key_payload,
                &self.key_scales,
                query_latent,
            );
            let residual_score = sparse_row_dot(
                self.key_residual,
                row,
                &self.key_residual_indices,
                &self.key_residual_payload,
                &self.key_residual_scales,
                query,
            );
            *score = (latent_score + residual_score) * attention_scale;
        }
        softmax_numerators_in_place(scores);

        let latent_value = &mut scratch.value_latent[..self.value_rank];
        latent_value.fill(0.0);
        let denominator: f32 = scores.iter().copied().sum();
        for (row, numerator) in scores.iter().copied().enumerate()
        {
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
        for (row, numerator) in scores.iter().copied().enumerate()
        {
            add_sparse_row(
                self.value_residual,
                row,
                &self.value_residual_indices,
                &self.value_residual_payload,
                &self.value_residual_scales,
                numerator / denominator,
                output,
            );
        }
        Ok(())
    }
}

fn residual_indices_for(
    indices: &[u16],
    slots_per_token: usize,
    resident_tokens: usize,
    token: usize,
) -> Option<&[u16]> {
    if token >= resident_tokens
    {
        return None;
    }
    let start = token * slots_per_token;
    Some(&indices[start..start + slots_per_token])
}

fn non_zero(field: &'static str, value: usize) -> Result<(), ResidualLatentCacheError> {
    if value == 0
    {
        return Err(LatentCacheError::ZeroDimension { field }.into());
    }
    Ok(())
}

fn require_rank(
    name: &'static str,
    rank: usize,
    dimension: usize,
) -> Result<(), ResidualLatentCacheError> {
    if rank > dimension
    {
        return Err(LatentCacheError::RankTooLarge {
            name,
            rank,
            dimension,
        }
        .into());
    }
    Ok(())
}

fn require_dimension(dimension: usize) -> Result<(), ResidualLatentCacheError> {
    if dimension > MAXIMUM_DIMENSION
    {
        return Err(ResidualLatentCacheError::DimensionTooLarge {
            dimension,
            maximum: MAXIMUM_DIMENSION,
        });
    }
    Ok(())
}

fn require_slots(
    name: &'static str,
    slots: usize,
    dimension: usize,
) -> Result<(), ResidualLatentCacheError> {
    if slots > dimension
    {
        return Err(ResidualLatentCacheError::ResidualSlotsTooLarge {
            name,
            slots,
            dimension,
        });
    }
    Ok(())
}

fn require_length(
    name: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), ResidualLatentCacheError> {
    if actual != expected
    {
        return Err(LatentCacheError::Length {
            name,
            expected,
            actual,
        }
        .into());
    }
    Ok(())
}

fn require_finite(name: &'static str, values: &[f32]) -> Result<(), ResidualLatentCacheError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite())
    {
        return Err(LatentCacheError::NonFinite { name, index }.into());
    }
    Ok(())
}

fn require_scratch(
    name: &'static str,
    available: usize,
    required: usize,
) -> Result<(), ResidualLatentCacheError> {
    if available < required
    {
        return Err(LatentCacheError::ScratchTooSmall {
            name,
            required,
            available,
        }
        .into());
    }
    Ok(())
}

fn checked_product(left: usize, right: usize) -> Result<usize, ResidualLatentCacheError> {
    left.checked_mul(right)
        .ok_or_else(|| LatentCacheError::Overflow.into())
}

fn scales_for(format: LatentStorageFormat, columns: usize, capacity: usize) -> Vec<f32> {
    if columns == 0 || format == LatentStorageFormat::F32
    {
        Vec::new()
    }
    else
    {
        vec![1.0; capacity]
    }
}

const fn scale_count(format: LatentStorageFormat, columns: usize, rows: usize) -> usize {
    if columns == 0 || matches!(format, LatentStorageFormat::F32)
    {
        0
    }
    else
    {
        rows
    }
}

fn row_bytes(
    format: LatentStorageFormat,
    columns: usize,
) -> Result<usize, ResidualLatentCacheError> {
    match format
    {
        LatentStorageFormat::F32 => checked_product(columns, core::mem::size_of::<f32>()),
        LatentStorageFormat::Int8 => Ok(columns),
        LatentStorageFormat::Int4 => Ok(columns.div_ceil(2)),
    }
}

const fn row_bytes_unchecked(format: LatentStorageFormat, columns: usize) -> usize {
    match format
    {
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
    for (row, scalar) in vector.iter().copied().enumerate().take(rows)
    {
        let offset = row * columns;
        for column in 0..columns
        {
            output[column] += basis[offset + column] * scalar;
        }
    }
}

fn up_project(basis: &[f32], rows: usize, columns: usize, latent: &[f32], output: &mut [f32]) {
    for (row, output_scalar) in output.iter_mut().enumerate().take(rows)
    {
        let offset = row * columns;
        let mut sum = 0.0_f32;
        for column in 0..columns
        {
            sum += basis[offset + column] * latent[column];
        }
        *output_scalar = sum;
    }
}

#[allow(clippy::too_many_arguments)]
fn select_and_encode_residuals(
    dense: &[f32],
    basis: &[f32],
    dimension: usize,
    rank: usize,
    projection: &[f32],
    config: SparseResidualConfig,
    row: usize,
    reconstruction: &mut [f32],
    residual_values: &mut [f32],
    selected: &mut [bool],
    indices: &mut [u16],
    payload: &mut [u8],
    scales: &mut [f32],
) {
    if config.slots_per_token == 0
    {
        return;
    }

    up_project(basis, dimension, rank, projection, reconstruction);
    selected.fill(false);
    residual_values.fill(0.0);
    let start = row * config.slots_per_token;
    let row_indices = &mut indices[start..start + config.slots_per_token];
    row_indices.fill(EMPTY_INDEX);

    for slot in 0..config.slots_per_token
    {
        let mut best_coordinate = None;
        let mut best_magnitude = 0.0_f32;
        let mut best_value = 0.0_f32;
        for coordinate in 0..dimension
        {
            if selected[coordinate]
            {
                continue;
            }
            let value = dense[coordinate] - reconstruction[coordinate];
            let magnitude = value.abs();
            if magnitude > best_magnitude
            {
                best_coordinate = Some(coordinate);
                best_magnitude = magnitude;
                best_value = value;
            }
        }
        let Some(coordinate) = best_coordinate
        else
        {
            break;
        };
        selected[coordinate] = true;
        row_indices[slot] =
            u16::try_from(coordinate).expect("validated residual coordinate must fit inside u16");
        residual_values[slot] = best_value;
    }

    encode_row(
        config.format,
        config.slots_per_token,
        row,
        residual_values,
        payload,
        scales,
    );
}

fn encode_row(
    format: LatentStorageFormat,
    columns: usize,
    row: usize,
    source: &[f32],
    payload: &mut [u8],
    scales: &mut [f32],
) {
    if columns == 0
    {
        return;
    }
    let bytes_per_row = row_bytes_unchecked(format, columns);
    let offset = row * bytes_per_row;
    let target = &mut payload[offset..offset + bytes_per_row];
    match format
    {
        LatentStorageFormat::F32 =>
        {
            for (value, bytes) in source.iter().zip(target.as_chunks_mut::<4>().0)
            {
                bytes.copy_from_slice(&value.to_le_bytes());
            }
        },
        LatentStorageFormat::Int8 | LatentStorageFormat::Int4 =>
        {
            target.fill(0);
            let limit = quantization_limit(format);
            let maximum = source.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
            let scale = if maximum == 0.0
            {
                1.0
            }
            else
            {
                maximum / f32::from(limit)
            };
            scales[row] = scale;
            for (column, value) in source.iter().copied().enumerate()
            {
                let code = quantize(value, scale, limit);
                match format
                {
                    LatentStorageFormat::Int8 =>
                    {
                        target[column] = code.to_ne_bytes()[0];
                    },
                    LatentStorageFormat::Int4 =>
                    {
                        let nibble = code.to_ne_bytes()[0] & 0x0f;
                        if column.is_multiple_of(2)
                        {
                            target[column / 2] = nibble;
                        }
                        else
                        {
                            target[column / 2] |= nibble << 4;
                        }
                    },
                    LatentStorageFormat::F32 => unreachable!(),
                }
            }
        },
    }
}

const fn quantization_limit(format: LatentStorageFormat) -> i8 {
    match format
    {
        LatentStorageFormat::F32 => 0,
        LatentStorageFormat::Int8 => 127,
        LatentStorageFormat::Int4 => 7,
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
    for (column, vector_scalar) in vector.iter().copied().enumerate().take(columns)
    {
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
    for (column, output_scalar) in output.iter_mut().enumerate().take(columns)
    {
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
    match format
    {
        LatentStorageFormat::F32 =>
        {
            let start = offset + column * 4;
            f32::from_le_bytes([
                payload[start],
                payload[start + 1],
                payload[start + 2],
                payload[start + 3],
            ])
        },
        LatentStorageFormat::Int8 =>
        {
            f32::from(i8::from_ne_bytes([payload[offset + column]])) * scales[row]
        },
        LatentStorageFormat::Int4 =>
        {
            let packed = payload[offset + column / 2];
            let nibble = if column.is_multiple_of(2)
            {
                packed & 0x0f
            }
            else
            {
                packed >> 4
            };
            let signed = if nibble < 8
            {
                nibble as i8
            }
            else
            {
                (i16::from(nibble) - 16) as i8
            };
            f32::from(signed) * scales[row]
        },
    }
}

fn sparse_row_dot(
    config: SparseResidualConfig,
    row: usize,
    indices: &[u16],
    payload: &[u8],
    scales: &[f32],
    dense: &[f32],
) -> f32 {
    let start = row * config.slots_per_token;
    let mut sum = 0.0_f32;
    for slot in 0..config.slots_per_token
    {
        let index = indices[start + slot];
        if index != EMPTY_INDEX
        {
            sum += dense[usize::from(index)]
                * coefficient(
                    config.format,
                    config.slots_per_token,
                    row,
                    slot,
                    payload,
                    scales,
                );
        }
    }
    sum
}

fn add_sparse_row(
    config: SparseResidualConfig,
    row: usize,
    indices: &[u16],
    payload: &[u8],
    scales: &[f32],
    weight: f32,
    output: &mut [f32],
) {
    let start = row * config.slots_per_token;
    for slot in 0..config.slots_per_token
    {
        let index = indices[start + slot];
        if index != EMPTY_INDEX
        {
            output[usize::from(index)] += weight
                * coefficient(
                    config.format,
                    config.slots_per_token,
                    row,
                    slot,
                    payload,
                    scales,
                );
        }
    }
}

fn softmax_numerators_in_place(scores: &mut [f32]) {
    let mut maximum = scores[0];
    for score in &scores[1..]
    {
        if *score > maximum
        {
            maximum = *score;
        }
    }
    for score in scores
    {
        *score = (*score - maximum).exp();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ResidualLatentAttentionScratch, ResidualLatentCacheError, ResidualQuantizedLatentKvCache,
        SparseResidualConfig,
    };
    use crate::nn::latent_kv_cache::LatentStorageFormat;
    use crate::nn::paged_attention::contiguous_attention;

    fn identity_prefix(dimension: usize, rank: usize) -> Vec<f32> {
        let mut basis = vec![0.0; dimension * rank];
        for diagonal in 0..rank
        {
            basis[diagonal * rank + diagonal] = 1.0;
        }
        basis
    }

    fn build_cache(
        capacity: usize,
        dimension: usize,
        rank: usize,
        coefficient_format: LatentStorageFormat,
        residual_slots: usize,
        residual_format: LatentStorageFormat,
    ) -> ResidualQuantizedLatentKvCache {
        let basis = identity_prefix(dimension, rank);
        ResidualQuantizedLatentKvCache::new(
            capacity,
            dimension,
            rank,
            rank,
            coefficient_format,
            coefficient_format,
            basis.clone(),
            basis,
            SparseResidualConfig::new(residual_slots, residual_format),
            SparseResidualConfig::new(residual_slots, residual_format),
        )
        .unwrap()
    }

    fn max_error(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max)
    }

    fn structured_data(tokens: usize, dimension: usize) -> (Vec<f32>, Vec<f32>) {
        let mut keys = vec![0.0; tokens * dimension];
        let mut values = vec![0.0; tokens * dimension];
        for token in 0..tokens
        {
            for coordinate in 0..4
            {
                keys[token * dimension + coordinate] =
                    ((token + 1) * (coordinate + 2)) as f32 * 0.03 - 0.2;
                values[token * dimension + coordinate] =
                    ((token + 3) * (coordinate + 1)) as f32 * -0.025 + 0.15;
            }
            keys[token * dimension + 4] = (token as f32 * 0.37).sin() * 0.8;
            keys[token * dimension + 5] = (token as f32 * 0.19).cos() * -0.6;
            values[token * dimension + 4] = (token as f32 * 0.23).cos() * 0.7;
            values[token * dimension + 5] = (token as f32 * 0.31).sin() * -0.5;
        }
        (keys, values)
    }

    fn evaluate(cache: &ResidualQuantizedLatentKvCache, query: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; cache.dimension()];
        let mut scratch = ResidualLatentAttentionScratch::new(
            cache.capacity(),
            cache.key_rank(),
            cache.value_rank(),
        );
        cache
            .attention_into(query, &mut output, &mut scratch)
            .unwrap();
        output
    }

    #[test]
    fn full_rank_f32_matches_contiguous_attention() {
        let (tokens, dimension) = (7, 8);
        let (keys, values) = structured_data(tokens, dimension);
        let query = vec![0.2, -0.1, 0.3, 0.4, -0.25, 0.5, 0.0, 0.1];
        let expected = contiguous_attention(&keys, &values, &query, dimension, tokens);
        let mut cache = build_cache(
            tokens,
            dimension,
            dimension,
            LatentStorageFormat::F32,
            3,
            LatentStorageFormat::F32,
        );
        for token in 0..tokens
        {
            let start = token * dimension;
            cache
                .append(
                    &keys[start..start + dimension],
                    &values[start..start + dimension],
                )
                .unwrap();
        }
        let actual = evaluate(&cache, &query);
        assert!(max_error(&expected, &actual) <= 2.0e-6);
    }

    #[test]
    fn sparse_residuals_restore_structured_reduced_rank_tail() {
        let (tokens, dimension, rank) = (9, 8, 4);
        let (keys, values) = structured_data(tokens, dimension);
        let query = vec![0.3, -0.2, 0.1, 0.4, 0.8, -0.7, 0.0, 0.0];
        let expected = contiguous_attention(&keys, &values, &query, dimension, tokens);
        let mut baseline = build_cache(
            tokens,
            dimension,
            rank,
            LatentStorageFormat::F32,
            0,
            LatentStorageFormat::F32,
        );
        let mut residual = build_cache(
            tokens,
            dimension,
            rank,
            LatentStorageFormat::F32,
            2,
            LatentStorageFormat::F32,
        );
        for token in 0..tokens
        {
            let start = token * dimension;
            let key = &keys[start..start + dimension];
            let value = &values[start..start + dimension];
            baseline.append(key, value).unwrap();
            residual.append(key, value).unwrap();
        }
        let baseline_error = max_error(&expected, &evaluate(&baseline, &query));
        let residual_error = max_error(&expected, &evaluate(&residual, &query));
        assert!(baseline_error > 0.05);
        assert!(residual_error <= 2.0e-6);
        assert!(residual_error < baseline_error);
    }

    #[test]
    fn quantized_residuals_bound_error_against_f32_residuals() {
        let (tokens, dimension, rank) = (11, 8, 4);
        let (keys, values) = structured_data(tokens, dimension);
        let query = vec![0.25, -0.15, 0.35, 0.05, 0.7, -0.9, 0.0, 0.0];
        let mut f32_cache = build_cache(
            tokens,
            dimension,
            rank,
            LatentStorageFormat::F32,
            2,
            LatentStorageFormat::F32,
        );
        let mut int8_cache = build_cache(
            tokens,
            dimension,
            rank,
            LatentStorageFormat::F32,
            2,
            LatentStorageFormat::Int8,
        );
        let mut int4_cache = build_cache(
            tokens,
            dimension,
            rank,
            LatentStorageFormat::F32,
            2,
            LatentStorageFormat::Int4,
        );
        for token in 0..tokens
        {
            let start = token * dimension;
            let key = &keys[start..start + dimension];
            let value = &values[start..start + dimension];
            f32_cache.append(key, value).unwrap();
            int8_cache.append(key, value).unwrap();
            int4_cache.append(key, value).unwrap();
        }
        let reference = evaluate(&f32_cache, &query);
        assert!(max_error(&reference, &evaluate(&int8_cache, &query)) <= 0.01);
        assert!(max_error(&reference, &evaluate(&int4_cache, &query)) <= 0.12);
    }

    #[test]
    fn equal_magnitude_ties_select_lowest_coordinate() {
        let dimension = 4;
        let mut cache = build_cache(
            1,
            dimension,
            1,
            LatentStorageFormat::F32,
            1,
            LatentStorageFormat::F32,
        );
        cache.append(&[0.0, 1.0, -1.0, 0.5], &[0.0; 4]).unwrap();
        assert_eq!(cache.key_residual_indices_for(0), Some(&[1_u16][..]));
    }

    #[test]
    fn allocation_is_fixed_across_appends() {
        let mut cache = build_cache(
            8,
            8,
            4,
            LatentStorageFormat::Int4,
            2,
            LatentStorageFormat::Int4,
        );
        let allocated = cache.allocated_bytes();
        for token in 0..8
        {
            let key = vec![token as f32 * 0.1; 8];
            let value = vec![token as f32 * -0.07; 8];
            cache.append(&key, &value).unwrap();
            assert_eq!(cache.allocated_bytes(), allocated);
        }
    }

    #[test]
    fn packed_residual_cache_reduces_fixed_allocation() {
        let capacity = 128;
        let dimension = 16;
        let cache = build_cache(
            capacity,
            dimension,
            4,
            LatentStorageFormat::Int4,
            2,
            LatentStorageFormat::Int4,
        );
        let dense_bytes = capacity * dimension * 2 * core::mem::size_of::<f32>();
        assert!(cache.allocated_bytes() < dense_bytes);
    }

    #[test]
    fn invalid_residual_slots_are_rejected() {
        let dimension = 4;
        let basis = identity_prefix(dimension, 2);
        let error = ResidualQuantizedLatentKvCache::new(
            2,
            dimension,
            2,
            2,
            LatentStorageFormat::F32,
            LatentStorageFormat::F32,
            basis.clone(),
            basis,
            SparseResidualConfig::new(5, LatentStorageFormat::Int8),
            SparseResidualConfig::disabled(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            ResidualLatentCacheError::ResidualSlotsTooLarge {
                name: "key",
                slots: 5,
                dimension: 4,
            }
        );
    }

    #[test]
    fn repeated_attention_is_bit_identical() {
        let (tokens, dimension, rank) = (6, 8, 4);
        let (keys, values) = structured_data(tokens, dimension);
        let query = vec![0.4, 0.1, -0.2, 0.3, 0.6, -0.8, 0.0, 0.0];
        let mut cache = build_cache(
            tokens,
            dimension,
            rank,
            LatentStorageFormat::Int4,
            2,
            LatentStorageFormat::Int4,
        );
        for token in 0..tokens
        {
            let start = token * dimension;
            cache
                .append(
                    &keys[start..start + dimension],
                    &values[start..start + dimension],
                )
                .unwrap();
        }
        let first = evaluate(&cache, &query);
        let second = evaluate(&cache, &query);
        assert_eq!(
            first
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            second
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}
