//! Device-resident incremental Transformer encoder over Elastic Latent KV state.
//!
//! Phase 17 extends the Phase 16 resident block boundary across every block in
//! one [`TransformerEncoder`] plus its final LayerNorm. One decode step uploads
//! one `d_model` row, executes all blocks and final normalisation in fixed order
//! on WGPU, then downloads one `d_model` row. Per-layer latent KV rings remain
//! resident between steps.

use core::fmt;

use crate::{
    WgpuComputeAdapter, WgpuComputeBuffer, WgpuComputeEvent, WgpuComputeKernel, WgpuComputeStream,
    WgpuContext, WgpuLatentHeadBasis,
};
use scirust_compute::{
    BufferAccess, BufferBinding, ComputeBackend, ComputeError, KernelFormat, KernelModule,
    LaunchConfig, MemorySpace,
};
use scirust_core::nn::transformer::{block::TransformerBlock, encoder::TransformerEncoder};

const F32_BYTES: usize = core::mem::size_of::<f32>();
const HEADER_WORDS: usize = 9;

const FUSED_ENCODER_WGSL: &str = r#"
struct ResidentState {
    d_model: u32,
    n_layers: u32,
    n_heads: u32,
    d_head: u32,
    rank: u32,
    capacity: u32,
    d_ff: u32,
    len: u32,
    next_slot: u32,
    data: array<f32>,
};

@group(0) @binding(0) var<storage, read_write> io: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> bases: array<f32>;
@group(0) @binding(3) var<storage, read_write> state: ResidentState;

fn physical_slot(logical: u32, oldest: u32, capacity: u32) -> u32 {
    return (oldest + logical) % capacity;
}

fn dense_from_state(
    input_offset: u32,
    input_width: u32,
    matrix_offset: u32,
    bias_offset: u32,
    output_column: u32,
    output_width: u32,
) -> f32 {
    var value = weights[bias_offset + output_column];
    for (var input_index: u32 = 0u; input_index < input_width; input_index = input_index + 1u) {
        value = value
            + state.data[input_offset + input_index]
                * weights[matrix_offset + input_index * output_width + output_column];
    }
    return value;
}

@compute @workgroup_size(1)
fn main() {
    let d_model = state.d_model;
    let n_layers = state.n_layers;
    let n_heads = state.n_heads;
    let d_head = state.d_head;
    let rank = state.rank;
    let capacity = state.capacity;
    let d_ff = state.d_ff;

    let model_matrix = d_model * d_model;
    let block_weight_elements =
        4u * model_matrix
        + 2u * d_model * d_ff
        + 9u * d_model
        + d_ff;

    let basis_head_elements = d_head * rank;
    let layer_basis_elements = 2u * n_heads * basis_head_elements;
    let coefficient_block = n_heads * capacity * rank;
    let all_key_cache_elements = n_layers * coefficient_block;
    let value_cache_offset = all_key_cache_elements;

    let query_latent_offset = 2u * all_key_cache_elements;
    let scores_offset = query_latent_offset + rank;
    let context_offset = scores_offset + capacity;
    let ln1_offset = context_offset + d_model;
    let q_scratch_offset = ln1_offset + d_model;
    let k_scratch_offset = q_scratch_offset + d_head;
    let v_scratch_offset = k_scratch_offset + d_head;
    let x1_offset = v_scratch_offset + d_head;
    let ln2_offset = x1_offset + d_model;
    let ff_offset = ln2_offset + d_model;

    let write_slot = state.next_slot;
    let new_next = (write_slot + 1u) % capacity;
    let new_len = min(state.len + 1u, capacity);
    let oldest = select(0u, new_next, new_len == capacity);
    let scale = 1.0 / sqrt(f32(d_head));

    for (var layer: u32 = 0u; layer < n_layers; layer = layer + 1u) {
        let weight_base = layer * block_weight_elements;
        let ln1_gamma_offset = weight_base;
        let ln1_beta_offset = ln1_gamma_offset + d_model;
        let wq_offset = ln1_beta_offset + d_model;
        let wk_offset = wq_offset + model_matrix;
        let wv_offset = wk_offset + model_matrix;
        let wo_offset = wv_offset + model_matrix;
        let bq_offset = wo_offset + model_matrix;
        let bk_offset = bq_offset + d_model;
        let bv_offset = bk_offset + d_model;
        let bo_offset = bv_offset + d_model;
        let ln2_gamma_offset = bo_offset + d_model;
        let ln2_beta_offset = ln2_gamma_offset + d_model;
        let ffn1_weight_offset = ln2_beta_offset + d_model;
        let ffn1_bias_offset = ffn1_weight_offset + d_model * d_ff;
        let ffn2_weight_offset = ffn1_bias_offset + d_ff;
        let ffn2_bias_offset = ffn2_weight_offset + d_ff * d_model;

        var ln1_mean = 0.0;
        for (var column: u32 = 0u; column < d_model; column = column + 1u) {
            ln1_mean = ln1_mean + io[column];
        }
        ln1_mean = ln1_mean / f32(d_model);
        var ln1_variance = 0.0;
        for (var column: u32 = 0u; column < d_model; column = column + 1u) {
            let delta = io[column] - ln1_mean;
            ln1_variance = ln1_variance + delta * delta;
        }
        ln1_variance = ln1_variance / f32(d_model);
        let ln1_inv_std = inverseSqrt(ln1_variance + 0.00001);
        for (var column: u32 = 0u; column < d_model; column = column + 1u) {
            state.data[ln1_offset + column] =
                (io[column] - ln1_mean) * ln1_inv_std * weights[ln1_gamma_offset + column]
                + weights[ln1_beta_offset + column];
        }

        let layer_basis_base = layer * layer_basis_elements;
        let layer_key_cache_base = layer * coefficient_block;

        for (var head: u32 = 0u; head < n_heads; head = head + 1u) {
            let column_base = head * d_head;
            let key_basis_head = layer_basis_base + head * basis_head_elements;
            let value_basis_head =
                layer_basis_base + n_heads * basis_head_elements + head * basis_head_elements;

            for (var local: u32 = 0u; local < d_head; local = local + 1u) {
                let column = column_base + local;
                state.data[q_scratch_offset + local] = dense_from_state(
                    ln1_offset, d_model, wq_offset, bq_offset, column, d_model);
                state.data[k_scratch_offset + local] = dense_from_state(
                    ln1_offset, d_model, wk_offset, bk_offset, column, d_model);
                state.data[v_scratch_offset + local] = dense_from_state(
                    ln1_offset, d_model, wv_offset, bv_offset, column, d_model);
            }

            for (var latent: u32 = 0u; latent < rank; latent = latent + 1u) {
                var key_coeff = 0.0;
                var value_coeff = 0.0;
                for (var local: u32 = 0u; local < d_head; local = local + 1u) {
                    key_coeff = key_coeff
                        + state.data[k_scratch_offset + local]
                            * bases[key_basis_head + local * rank + latent];
                    value_coeff = value_coeff
                        + state.data[v_scratch_offset + local]
                            * bases[value_basis_head + local * rank + latent];
                }
                let local_slot = (head * capacity + write_slot) * rank + latent;
                let slot_index = layer_key_cache_base + local_slot;
                state.data[slot_index] = key_coeff;
                state.data[value_cache_offset + slot_index] = value_coeff;
            }

            for (var latent: u32 = 0u; latent < rank; latent = latent + 1u) {
                var query_coeff = 0.0;
                for (var local: u32 = 0u; local < d_head; local = local + 1u) {
                    query_coeff = query_coeff
                        + state.data[q_scratch_offset + local]
                            * bases[key_basis_head + local * rank + latent];
                }
                state.data[query_latent_offset + latent] = query_coeff;
            }

            var maximum = -3.402823466e+38;
            for (var logical: u32 = 0u; logical < new_len; logical = logical + 1u) {
                let slot = physical_slot(logical, oldest, capacity);
                var score = 0.0;
                for (var latent: u32 = 0u; latent < rank; latent = latent + 1u) {
                    let local_slot = (head * capacity + slot) * rank + latent;
                    let slot_index = layer_key_cache_base + local_slot;
                    score = score
                        + state.data[query_latent_offset + latent] * state.data[slot_index];
                }
                score = score * scale;
                state.data[scores_offset + logical] = score;
                maximum = max(maximum, score);
            }

            var denominator = 0.0;
            for (var logical: u32 = 0u; logical < new_len; logical = logical + 1u) {
                let weight = exp(state.data[scores_offset + logical] - maximum);
                state.data[scores_offset + logical] = weight;
                denominator = denominator + weight;
            }
            for (var logical: u32 = 0u; logical < new_len; logical = logical + 1u) {
                state.data[scores_offset + logical] =
                    state.data[scores_offset + logical] / denominator;
            }

            for (var latent: u32 = 0u; latent < rank; latent = latent + 1u) {
                var coefficient = 0.0;
                for (var logical: u32 = 0u; logical < new_len; logical = logical + 1u) {
                    let slot = physical_slot(logical, oldest, capacity);
                    let local_slot = (head * capacity + slot) * rank + latent;
                    let slot_index = layer_key_cache_base + local_slot;
                    coefficient = coefficient
                        + state.data[scores_offset + logical]
                            * state.data[value_cache_offset + slot_index];
                }
                state.data[query_latent_offset + latent] = coefficient;
            }

            for (var local: u32 = 0u; local < d_head; local = local + 1u) {
                var reconstructed = 0.0;
                for (var latent: u32 = 0u; latent < rank; latent = latent + 1u) {
                    reconstructed = reconstructed
                        + state.data[query_latent_offset + latent]
                            * bases[value_basis_head + local * rank + latent];
                }
                state.data[context_offset + column_base + local] = reconstructed;
            }
        }

        for (var column: u32 = 0u; column < d_model; column = column + 1u) {
            var attention = weights[bo_offset + column];
            for (var input_index: u32 = 0u; input_index < d_model; input_index = input_index + 1u) {
                attention = attention
                    + state.data[context_offset + input_index]
                        * weights[wo_offset + input_index * d_model + column];
            }
            state.data[x1_offset + column] = io[column] + attention;
        }

        var ln2_mean = 0.0;
        for (var column: u32 = 0u; column < d_model; column = column + 1u) {
            ln2_mean = ln2_mean + state.data[x1_offset + column];
        }
        ln2_mean = ln2_mean / f32(d_model);
        var ln2_variance = 0.0;
        for (var column: u32 = 0u; column < d_model; column = column + 1u) {
            let delta = state.data[x1_offset + column] - ln2_mean;
            ln2_variance = ln2_variance + delta * delta;
        }
        ln2_variance = ln2_variance / f32(d_model);
        let ln2_inv_std = inverseSqrt(ln2_variance + 0.00001);
        for (var column: u32 = 0u; column < d_model; column = column + 1u) {
            state.data[ln2_offset + column] =
                (state.data[x1_offset + column] - ln2_mean) * ln2_inv_std
                    * weights[ln2_gamma_offset + column]
                + weights[ln2_beta_offset + column];
        }

        for (var column: u32 = 0u; column < d_ff; column = column + 1u) {
            let projected = dense_from_state(
                ln2_offset,
                d_model,
                ffn1_weight_offset,
                ffn1_bias_offset,
                column,
                d_ff,
            );
            state.data[ff_offset + column] = max(projected, 0.0);
        }

        for (var column: u32 = 0u; column < d_model; column = column + 1u) {
            let projected = dense_from_state(
                ff_offset,
                d_ff,
                ffn2_weight_offset,
                ffn2_bias_offset,
                column,
                d_model,
            );
            io[column] = state.data[x1_offset + column] + projected;
        }
    }

    let final_gamma_offset = n_layers * block_weight_elements;
    let final_beta_offset = final_gamma_offset + d_model;
    var final_mean = 0.0;
    for (var column: u32 = 0u; column < d_model; column = column + 1u) {
        final_mean = final_mean + io[column];
    }
    final_mean = final_mean / f32(d_model);
    var final_variance = 0.0;
    for (var column: u32 = 0u; column < d_model; column = column + 1u) {
        let delta = io[column] - final_mean;
        final_variance = final_variance + delta * delta;
    }
    final_variance = final_variance / f32(d_model);
    let final_inv_std = inverseSqrt(final_variance + 0.00001);
    for (var column: u32 = 0u; column < d_model; column = column + 1u) {
        io[column] =
            (io[column] - final_mean) * final_inv_std * weights[final_gamma_offset + column]
            + weights[final_beta_offset + column];
    }

    state.len = new_len;
    state.next_slot = new_next;
}
"#;

/// Borrowed per-layer head bases for [`WgpuResidentTransformerEncoder`].
#[derive(Clone, Copy)]
pub struct WgpuLatentLayerBasis<'a> {
    pub heads: &'a [WgpuLatentHeadBasis<'a>],
}

#[derive(Debug)]
pub enum WgpuResidentTransformerEncoderError {
    InvalidConfig(&'static str),
    LayerCount {
        expected: usize,
        actual: usize,
    },
    HeadCount {
        layer: usize,
        expected: usize,
        actual: usize,
    },
    BasisLength {
        layer: usize,
        head: usize,
        expected: usize,
        key: usize,
        value: usize,
    },
    VectorLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    PositionMismatch {
        expected: usize,
        actual: usize,
    },
    Compute(ComputeError),
}

impl fmt::Display for WgpuResidentTransformerEncoderError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidConfig(message) => write!(output, "{message}"),
            Self::LayerCount { expected, actual } => write!(
                output,
                "resident transformer encoder layer mismatch: expected {expected}, got {actual}"
            ),
            Self::HeadCount {
                layer,
                expected,
                actual,
            } => write!(
                output,
                "resident transformer encoder head mismatch at layer {layer}: expected {expected}, got {actual}"
            ),
            Self::BasisLength {
                layer,
                head,
                expected,
                key,
                value,
            } => write!(
                output,
                "resident transformer encoder basis mismatch at layer {layer}, head {head}: expected {expected}, key={key}, value={value}"
            ),
            Self::VectorLength {
                name,
                expected,
                actual,
            } => write!(
                output,
                "{name} length mismatch: expected {expected}, got {actual}"
            ),
            Self::PositionMismatch { expected, actual } => write!(
                output,
                "resident transformer encoder position mismatch: expected {expected}, got {actual}"
            ),
            Self::Compute(error) => write!(output, "{error}"),
        }
    }
}

impl std::error::Error for WgpuResidentTransformerEncoderError {}

impl From<ComputeError> for WgpuResidentTransformerEncoderError {
    fn from(error: ComputeError) -> Self {
        Self::Compute(error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuResidentTransformerEncoderTelemetry {
    pub capacity_tokens: usize,
    pub resident_tokens: usize,
    pub steps: usize,
    pub n_layers: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_head: usize,
    pub d_ff: usize,
    pub rank: usize,
    pub resident_bytes: usize,
    pub upload_bytes_per_step: usize,
    pub download_bytes_per_step: usize,
    pub next_write_slot: usize,
}

/// Persistent WGPU inference snapshot of one [`TransformerEncoder`].
pub struct WgpuResidentTransformerEncoder {
    adapter: WgpuComputeAdapter,
    weights: WgpuComputeBuffer,
    bases: WgpuComputeBuffer,
    state: WgpuComputeBuffer,
    io: WgpuComputeBuffer,
    kernel: WgpuComputeKernel,
    stream: WgpuComputeStream,
    capacity: usize,
    n_layers: usize,
    d_model: usize,
    n_heads: usize,
    d_head: usize,
    d_ff: usize,
    rank: usize,
    resident_tokens: usize,
    steps: usize,
    next_slot: usize,
    resident_bytes: usize,
}

impl WgpuResidentTransformerEncoder {
    pub fn new(
        encoder: &TransformerEncoder,
        capacity: usize,
        rank: usize,
        layers: &[WgpuLatentLayerBasis<'_>],
    ) -> Result<Self, WgpuResidentTransformerEncoderError> {
        let context = WgpuContext::new().map_err(|_| {
            WgpuResidentTransformerEncoderError::InvalidConfig("WGPU backend is unavailable")
        })?;
        Self::from_context(context, encoder, capacity, rank, layers)
    }

    pub fn from_context(
        context: WgpuContext,
        encoder: &TransformerEncoder,
        capacity: usize,
        rank: usize,
        layers: &[WgpuLatentLayerBasis<'_>],
    ) -> Result<Self, WgpuResidentTransformerEncoderError> {
        validate_topology(encoder, capacity, rank, layers)?;

        let first = &encoder.blocks[0];
        let n_layers = encoder.blocks.len();
        let d_model = encoder.d_model;
        let n_heads = first.n_heads;
        let d_head = first.mha.d_head;
        let d_ff = first.d_ff;
        let weights_data = pack_weights(encoder)?;
        let bases_data = pack_bases(layers, d_head, rank)?;

        let coefficient_elements = n_layers
            .checked_mul(n_heads)
            .and_then(|value| value.checked_mul(capacity))
            .and_then(|value| value.checked_mul(rank))
            .and_then(|value| value.checked_mul(2))
            .ok_or(WgpuResidentTransformerEncoderError::InvalidConfig(
                "resident transformer encoder coefficient storage overflows usize",
            ))?;
        let scratch_elements = rank
            .checked_add(capacity)
            .and_then(|value| value.checked_add(d_model.checked_mul(4)?))
            .and_then(|value| value.checked_add(d_head.checked_mul(3)?))
            .and_then(|value| value.checked_add(d_ff))
            .ok_or(WgpuResidentTransformerEncoderError::InvalidConfig(
                "resident transformer encoder scratch storage overflows usize",
            ))?;
        let state_data_elements = coefficient_elements.checked_add(scratch_elements).ok_or(
            WgpuResidentTransformerEncoderError::InvalidConfig(
                "resident transformer encoder state storage overflows usize",
            ),
        )?;

        ensure_wgpu_indexable(
            weights_data.len(),
            "packed encoder weights exceed WGPU u32 range",
        )?;
        ensure_wgpu_indexable(
            bases_data.len(),
            "packed encoder bases exceed WGPU u32 range",
        )?;
        ensure_wgpu_indexable(
            state_data_elements,
            "resident transformer encoder state exceeds WGPU u32 range",
        )?;

        let weights_bytes = bytes_for_f32(weights_data.len())?;
        let bases_bytes = bytes_for_f32(bases_data.len())?;
        let state_bytes = HEADER_WORDS
            .checked_mul(core::mem::size_of::<u32>())
            .and_then(|header| header.checked_add(bytes_for_f32(state_data_elements).ok()?))
            .ok_or(WgpuResidentTransformerEncoderError::InvalidConfig(
                "resident transformer encoder state bytes overflow usize",
            ))?;
        let io_bytes = bytes_for_f32(d_model)?;

        let adapter = WgpuComputeAdapter::from_context(context);
        let weights = adapter.allocate(weights_bytes, 4, MemorySpace::Device)?;
        let bases = adapter.allocate(bases_bytes, 4, MemorySpace::Device)?;
        let state = adapter.allocate(state_bytes, 4, MemorySpace::Device)?;
        let io = adapter.allocate(io_bytes, 4, MemorySpace::Device)?;

        adapter.write(&weights, 0, bytemuck::cast_slice(&weights_data))?;
        adapter.write(&bases, 0, bytemuck::cast_slice(&bases_data))?;
        let header = [
            usize_to_u32(d_model, "d_model exceeds WGPU u32 range")?,
            usize_to_u32(n_layers, "n_layers exceeds WGPU u32 range")?,
            usize_to_u32(n_heads, "n_heads exceeds WGPU u32 range")?,
            usize_to_u32(d_head, "d_head exceeds WGPU u32 range")?,
            usize_to_u32(rank, "rank exceeds WGPU u32 range")?,
            usize_to_u32(capacity, "capacity exceeds WGPU u32 range")?,
            usize_to_u32(d_ff, "d_ff exceeds WGPU u32 range")?,
            0,
            0,
        ];
        adapter.write(&state, 0, bytemuck::cast_slice(&header))?;

        let module = KernelModule::new(
            KernelFormat::Wgsl,
            "main",
            FUSED_ENCODER_WGSL.as_bytes().to_vec(),
        )?;
        let kernel = adapter.compile(&module)?;
        let stream = adapter.create_stream()?;

        Ok(Self {
            adapter,
            weights,
            bases,
            state,
            io,
            kernel,
            stream,
            capacity,
            n_layers,
            d_model,
            n_heads,
            d_head,
            d_ff,
            rank,
            resident_tokens: 0,
            steps: 0,
            next_slot: 0,
            resident_bytes: weights_bytes + bases_bytes + state_bytes + io_bytes,
        })
    }

    #[must_use]
    pub const fn telemetry(&self) -> WgpuResidentTransformerEncoderTelemetry {
        WgpuResidentTransformerEncoderTelemetry {
            capacity_tokens: self.capacity,
            resident_tokens: self.resident_tokens,
            steps: self.steps,
            n_layers: self.n_layers,
            d_model: self.d_model,
            n_heads: self.n_heads,
            d_head: self.d_head,
            d_ff: self.d_ff,
            rank: self.rank,
            resident_bytes: self.resident_bytes,
            upload_bytes_per_step: self.d_model * F32_BYTES,
            download_bytes_per_step: self.d_model * F32_BYTES,
            next_write_slot: self.next_slot,
        }
    }

    pub fn infer_step_at_into(
        &mut self,
        input: &[f32],
        pos: usize,
        output: &mut [f32],
    ) -> Result<(), WgpuResidentTransformerEncoderError> {
        if pos != self.steps
        {
            return Err(WgpuResidentTransformerEncoderError::PositionMismatch {
                expected: self.steps,
                actual: pos,
            });
        }
        self.require_vector("input", input)?;
        if output.len() != self.d_model
        {
            return Err(WgpuResidentTransformerEncoderError::VectorLength {
                name: "output",
                expected: self.d_model,
                actual: output.len(),
            });
        }

        self.adapter
            .write(&self.io, 0, bytemuck::cast_slice(input))?;
        let event = self.launch()?;
        self.adapter.wait(&event)?;

        self.steps = self.steps.saturating_add(1);
        self.resident_tokens = self.resident_tokens.saturating_add(1).min(self.capacity);
        self.next_slot = (self.next_slot + 1) % self.capacity;

        self.adapter
            .read(&self.io, 0, bytemuck::cast_slice_mut(output))?;
        Ok(())
    }

    pub fn infer_step_at(
        &mut self,
        input: &[f32],
        pos: usize,
    ) -> Result<Vec<f32>, WgpuResidentTransformerEncoderError> {
        let mut output = vec![0.0; self.d_model];
        self.infer_step_at_into(input, pos, &mut output)?;
        Ok(output)
    }

    pub fn infer_step(
        &mut self,
        input: &[f32],
    ) -> Result<Vec<f32>, WgpuResidentTransformerEncoderError> {
        self.infer_step_at(input, self.steps)
    }

    pub fn reload_weights(
        &mut self,
        encoder: &TransformerEncoder,
    ) -> Result<(), WgpuResidentTransformerEncoderError> {
        validate_reload_topology(
            encoder,
            self.n_layers,
            self.d_model,
            self.n_heads,
            self.d_head,
            self.d_ff,
        )?;
        let packed = pack_weights(encoder)?;
        self.adapter
            .write(&self.weights, 0, bytemuck::cast_slice(&packed))?;
        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), WgpuResidentTransformerEncoderError> {
        let cleared = [0u32, 0u32];
        self.adapter.write(
            &self.state,
            7 * core::mem::size_of::<u32>(),
            bytemuck::cast_slice(&cleared),
        )?;
        self.resident_tokens = 0;
        self.steps = 0;
        self.next_slot = 0;
        Ok(())
    }

    fn require_vector(
        &self,
        name: &'static str,
        vector: &[f32],
    ) -> Result<(), WgpuResidentTransformerEncoderError> {
        if vector.len() != self.d_model
        {
            return Err(WgpuResidentTransformerEncoderError::VectorLength {
                name,
                expected: self.d_model,
                actual: vector.len(),
            });
        }
        Ok(())
    }

    fn launch(&self) -> Result<WgpuComputeEvent, WgpuResidentTransformerEncoderError> {
        let bindings = [
            binding(0, &self.io, BufferAccess::ReadWrite),
            binding(1, &self.weights, BufferAccess::ReadOnly),
            binding(2, &self.bases, BufferAccess::ReadOnly),
            binding(3, &self.state, BufferAccess::ReadWrite),
        ];
        let config = LaunchConfig::new([1, 1, 1], [1, 1, 1], 0)?;
        Ok(self
            .adapter
            .launch(&self.kernel, &self.stream, config, &bindings)?)
    }
}

fn validate_topology(
    encoder: &TransformerEncoder,
    capacity: usize,
    rank: usize,
    layers: &[WgpuLatentLayerBasis<'_>],
) -> Result<(), WgpuResidentTransformerEncoderError> {
    if capacity == 0
    {
        return Err(WgpuResidentTransformerEncoderError::InvalidConfig(
            "resident transformer encoder capacity must be non-zero",
        ));
    }
    if encoder.blocks.is_empty()
    {
        return Err(WgpuResidentTransformerEncoderError::InvalidConfig(
            "resident transformer encoder requires at least one block",
        ));
    }
    let first = &encoder.blocks[0];
    if encoder.d_model == 0 || first.n_heads == 0 || first.mha.d_head == 0 || first.d_ff == 0
    {
        return Err(WgpuResidentTransformerEncoderError::InvalidConfig(
            "resident transformer encoder topology must be non-zero",
        ));
    }
    if rank == 0 || rank > first.mha.d_head
    {
        return Err(WgpuResidentTransformerEncoderError::InvalidConfig(
            "resident transformer encoder rank must be in 1..=d_head",
        ));
    }
    if layers.len() != encoder.blocks.len()
    {
        return Err(WgpuResidentTransformerEncoderError::LayerCount {
            expected: encoder.blocks.len(),
            actual: layers.len(),
        });
    }

    validate_reload_topology(
        encoder,
        encoder.blocks.len(),
        encoder.d_model,
        first.n_heads,
        first.mha.d_head,
        first.d_ff,
    )?;

    let expected = first.mha.d_head.checked_mul(rank).ok_or(
        WgpuResidentTransformerEncoderError::InvalidConfig(
            "resident transformer encoder basis shape overflows usize",
        ),
    )?;
    for (layer_index, layer) in layers.iter().enumerate()
    {
        if layer.heads.len() != first.n_heads
        {
            return Err(WgpuResidentTransformerEncoderError::HeadCount {
                layer: layer_index,
                expected: first.n_heads,
                actual: layer.heads.len(),
            });
        }
        for (head_index, basis) in layer.heads.iter().enumerate()
        {
            if basis.key.len() != expected || basis.value.len() != expected
            {
                return Err(WgpuResidentTransformerEncoderError::BasisLength {
                    layer: layer_index,
                    head: head_index,
                    expected,
                    key: basis.key.len(),
                    value: basis.value.len(),
                });
            }
        }
    }
    Ok(())
}

fn validate_reload_topology(
    encoder: &TransformerEncoder,
    n_layers: usize,
    d_model: usize,
    n_heads: usize,
    d_head: usize,
    d_ff: usize,
) -> Result<(), WgpuResidentTransformerEncoderError> {
    if encoder.blocks.len() != n_layers || encoder.d_model != d_model
    {
        return Err(WgpuResidentTransformerEncoderError::InvalidConfig(
            "cannot reload weights from a different Transformer encoder topology",
        ));
    }
    for block in &encoder.blocks
    {
        validate_block_shapes(block, d_model, n_heads, d_head, d_ff)?;
    }
    if encoder.final_ln.gamma.data.len() != d_model
        || encoder.final_ln.beta.data.len() != d_model
        || (encoder.final_ln.eps - 1e-5).abs() > f32::EPSILON
    {
        return Err(WgpuResidentTransformerEncoderError::InvalidConfig(
            "resident transformer encoder final LayerNorm shape or epsilon mismatch",
        ));
    }
    Ok(())
}

fn validate_block_shapes(
    block: &TransformerBlock,
    d_model: usize,
    n_heads: usize,
    d_head: usize,
    d_ff: usize,
) -> Result<(), WgpuResidentTransformerEncoderError> {
    if block.d_model != d_model
        || block.n_heads != n_heads
        || block.mha.d_model != d_model
        || block.mha.n_heads != n_heads
        || block.mha.d_head != d_head
        || block.d_ff != d_ff
        || d_head.checked_mul(n_heads) != Some(d_model)
    {
        return Err(WgpuResidentTransformerEncoderError::InvalidConfig(
            "resident transformer encoder requires uniform block topology",
        ));
    }
    let model_matrix =
        d_model
            .checked_mul(d_model)
            .ok_or(WgpuResidentTransformerEncoderError::InvalidConfig(
                "resident transformer encoder model matrix overflows usize",
            ))?;
    for linear in [
        &block.mha.w_q,
        &block.mha.w_k,
        &block.mha.w_v,
        &block.mha.w_o,
    ]
    {
        if linear.in_features != d_model
            || linear.out_features != d_model
            || linear.weight.data.len() != model_matrix
            || linear.bias.data.len() != d_model
        {
            return Err(WgpuResidentTransformerEncoderError::InvalidConfig(
                "resident transformer encoder requires square model-width Q/K/V/O projections",
            ));
        }
    }
    if block.ln1.gamma.data.len() != d_model
        || block.ln1.beta.data.len() != d_model
        || block.ln2.gamma.data.len() != d_model
        || block.ln2.beta.data.len() != d_model
        || (block.ln1.eps - 1e-5).abs() > f32::EPSILON
        || (block.ln2.eps - 1e-5).abs() > f32::EPSILON
    {
        return Err(WgpuResidentTransformerEncoderError::InvalidConfig(
            "resident transformer encoder LayerNorm shape or epsilon mismatch",
        ));
    }
    if block.ffn1.in_features != d_model
        || block.ffn1.out_features != d_ff
        || block.ffn1.weight.data.len() != d_model * d_ff
        || block.ffn1.bias.data.len() != d_ff
        || block.ffn2.in_features != d_ff
        || block.ffn2.out_features != d_model
        || block.ffn2.weight.data.len() != d_ff * d_model
        || block.ffn2.bias.data.len() != d_model
    {
        return Err(WgpuResidentTransformerEncoderError::InvalidConfig(
            "resident transformer encoder FFN topology mismatch",
        ));
    }
    Ok(())
}

fn pack_weights(
    encoder: &TransformerEncoder,
) -> Result<Vec<f32>, WgpuResidentTransformerEncoderError> {
    if encoder.blocks.is_empty()
    {
        return Err(WgpuResidentTransformerEncoderError::InvalidConfig(
            "resident transformer encoder requires at least one block",
        ));
    }
    let first = &encoder.blocks[0];
    validate_reload_topology(
        encoder,
        encoder.blocks.len(),
        encoder.d_model,
        first.n_heads,
        first.mha.d_head,
        first.d_ff,
    )?;

    let d_model = encoder.d_model;
    let d_ff = first.d_ff;
    let block_elements = d_model
        .checked_mul(d_model)
        .and_then(|value| value.checked_mul(4))
        .and_then(|value| value.checked_add(d_model.checked_mul(d_ff)?.checked_mul(2)?))
        .and_then(|value| value.checked_add(d_model.checked_mul(9)?))
        .and_then(|value| value.checked_add(d_ff))
        .ok_or(WgpuResidentTransformerEncoderError::InvalidConfig(
            "resident transformer encoder packed block weights overflow usize",
        ))?;
    let total = block_elements
        .checked_mul(encoder.blocks.len())
        .and_then(|value| value.checked_add(d_model.checked_mul(2)?))
        .ok_or(WgpuResidentTransformerEncoderError::InvalidConfig(
            "resident transformer encoder packed weights overflow usize",
        ))?;
    let mut packed = Vec::with_capacity(total);
    for block in &encoder.blocks
    {
        packed.extend_from_slice(&block.ln1.gamma.data);
        packed.extend_from_slice(&block.ln1.beta.data);
        packed.extend_from_slice(&block.mha.w_q.weight.data);
        packed.extend_from_slice(&block.mha.w_k.weight.data);
        packed.extend_from_slice(&block.mha.w_v.weight.data);
        packed.extend_from_slice(&block.mha.w_o.weight.data);
        packed.extend_from_slice(&block.mha.w_q.bias.data);
        packed.extend_from_slice(&block.mha.w_k.bias.data);
        packed.extend_from_slice(&block.mha.w_v.bias.data);
        packed.extend_from_slice(&block.mha.w_o.bias.data);
        packed.extend_from_slice(&block.ln2.gamma.data);
        packed.extend_from_slice(&block.ln2.beta.data);
        packed.extend_from_slice(&block.ffn1.weight.data);
        packed.extend_from_slice(&block.ffn1.bias.data);
        packed.extend_from_slice(&block.ffn2.weight.data);
        packed.extend_from_slice(&block.ffn2.bias.data);
    }
    packed.extend_from_slice(&encoder.final_ln.gamma.data);
    packed.extend_from_slice(&encoder.final_ln.beta.data);
    debug_assert_eq!(packed.len(), total);
    Ok(packed)
}

fn pack_bases(
    layers: &[WgpuLatentLayerBasis<'_>],
    d_head: usize,
    rank: usize,
) -> Result<Vec<f32>, WgpuResidentTransformerEncoderError> {
    let per_head =
        d_head
            .checked_mul(rank)
            .ok_or(WgpuResidentTransformerEncoderError::InvalidConfig(
                "resident transformer encoder basis shape overflows usize",
            ))?;
    let head_count = layers
        .iter()
        .try_fold(0usize, |count, layer| count.checked_add(layer.heads.len()))
        .ok_or(WgpuResidentTransformerEncoderError::InvalidConfig(
            "resident transformer encoder head count overflows usize",
        ))?;
    let total = head_count
        .checked_mul(per_head)
        .and_then(|value| value.checked_mul(2))
        .ok_or(WgpuResidentTransformerEncoderError::InvalidConfig(
            "resident transformer encoder packed bases overflow usize",
        ))?;
    let mut packed = Vec::with_capacity(total);
    for layer in layers
    {
        for head in layer.heads
        {
            packed.extend_from_slice(head.key);
        }
        for head in layer.heads
        {
            packed.extend_from_slice(head.value);
        }
    }
    Ok(packed)
}

fn binding<'a>(
    slot: u32,
    buffer: &'a WgpuComputeBuffer,
    access: BufferAccess,
) -> BufferBinding<'a, WgpuComputeBuffer> {
    BufferBinding {
        slot,
        buffer,
        offset_bytes: 0,
        length_bytes: buffer.len(),
        access,
    }
}

fn bytes_for_f32(elements: usize) -> Result<usize, WgpuResidentTransformerEncoderError> {
    elements
        .checked_mul(F32_BYTES)
        .ok_or(WgpuResidentTransformerEncoderError::InvalidConfig(
            "resident transformer encoder f32 buffer size overflows usize",
        ))
}

fn ensure_wgpu_indexable(
    elements: usize,
    message: &'static str,
) -> Result<(), WgpuResidentTransformerEncoderError> {
    usize_to_u32(elements, message).map(|_| ())
}

fn usize_to_u32(
    value: usize,
    message: &'static str,
) -> Result<u32, WgpuResidentTransformerEncoderError> {
    u32::try_from(value).map_err(|_| WgpuResidentTransformerEncoderError::InvalidConfig(message))
}
