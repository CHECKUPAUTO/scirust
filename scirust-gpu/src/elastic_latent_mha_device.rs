//! Fused device-resident Multi-Head Attention over Elastic Latent KV state.
//!
//! Phase 14 made each latent KV ring persistent on WGPU, but still accepted
//! dense Q/K/V vectors from the host. This module moves the Q/K/V and output
//! projections into the same persistent WGPU inference substrate.
//!
//! Per decode step the host uploads only the normalised `d_model` input and
//! reads only the final `d_model` attention output. Q/K/V projections, latent
//! append, score/softmax, latent value aggregation, reconstruction and the O
//! projection remain on device. The first implementation intentionally uses one
//! invocation with a fixed loop order: it is a deterministic correctness
//! baseline, not a throughput claim.

use core::fmt;

use crate::{
    WgpuComputeAdapter, WgpuComputeBuffer, WgpuComputeEvent, WgpuComputeKernel, WgpuComputeStream,
    WgpuContext,
};
use scirust_compute::{
    BufferAccess, BufferBinding, ComputeBackend, ComputeError, KernelFormat, KernelModule,
    LaunchConfig, MemorySpace,
};
use scirust_core::nn::transformer::attention::MultiHeadAttention;

const F32_BYTES: usize = core::mem::size_of::<f32>();
const HEADER_WORDS: usize = 8;

const FUSED_MHA_WGSL: &str = r#"
struct ResidentState {
    d_model: u32,
    n_heads: u32,
    d_head: u32,
    rank: u32,
    capacity: u32,
    len: u32,
    next_slot: u32,
    _pad: u32,
    data: array<f32>,
};

@group(0) @binding(0) var<storage, read_write> io: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> bases: array<f32>;
@group(0) @binding(3) var<storage, read_write> state: ResidentState;

fn dense_projection(matrix_offset: u32, bias_offset: u32, column: u32) -> f32 {
    var value = weights[bias_offset + column];
    for (var input_index: u32 = 0u; input_index < state.d_model; input_index = input_index + 1u) {
        value = value
            + io[input_index]
                * weights[matrix_offset + input_index * state.d_model + column];
    }
    return value;
}

fn physical_slot(logical: u32, oldest: u32, capacity: u32) -> u32 {
    return (oldest + logical) % capacity;
}

@compute @workgroup_size(1)
fn main() {
    let d_model = state.d_model;
    let n_heads = state.n_heads;
    let d_head = state.d_head;
    let rank = state.rank;
    let capacity = state.capacity;

    let matrix_elements = d_model * d_model;
    let wq_offset = 0u;
    let wk_offset = matrix_elements;
    let wv_offset = 2u * matrix_elements;
    let wo_offset = 3u * matrix_elements;
    let bq_offset = 4u * matrix_elements;
    let bk_offset = bq_offset + d_model;
    let bv_offset = bk_offset + d_model;
    let bo_offset = bv_offset + d_model;

    let basis_head_elements = d_head * rank;
    let value_basis_offset = n_heads * basis_head_elements;

    let coefficient_block = n_heads * capacity * rank;
    let value_cache_offset = coefficient_block;
    let query_latent_offset = 2u * coefficient_block;
    let scores_offset = query_latent_offset + rank;
    let context_offset = scores_offset + capacity;
    let q_scratch_offset = context_offset + d_model;
    let k_scratch_offset = q_scratch_offset + d_head;
    let v_scratch_offset = k_scratch_offset + d_head;

    let write_slot = state.next_slot;
    let new_next = (write_slot + 1u) % capacity;
    let new_len = min(state.len + 1u, capacity);
    let oldest = select(0u, new_next, new_len == capacity);
    let scale = 1.0 / sqrt(f32(d_head));

    for (var head: u32 = 0u; head < n_heads; head = head + 1u) {
        let column_base = head * d_head;
        let key_basis_head = head * basis_head_elements;
        let value_basis_head = value_basis_offset + head * basis_head_elements;

        // Dense Q/K/V projections for one head. Scratch is shared because heads
        // execute sequentially in this deterministic baseline.
        for (var local: u32 = 0u; local < d_head; local = local + 1u) {
            let column = column_base + local;
            state.data[q_scratch_offset + local] = dense_projection(wq_offset, bq_offset, column);
            state.data[k_scratch_offset + local] = dense_projection(wk_offset, bk_offset, column);
            state.data[v_scratch_offset + local] = dense_projection(wv_offset, bv_offset, column);
        }

        // Project current K/V into the committed per-head latent bases and append
        // directly into the persistent sliding ring.
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
            let slot_index = (head * capacity + write_slot) * rank + latent;
            state.data[slot_index] = key_coeff;
            state.data[value_cache_offset + slot_index] = value_coeff;
        }

        // Project Q into the key basis.
        for (var latent: u32 = 0u; latent < rank; latent = latent + 1u) {
            var query_coeff = 0.0;
            for (var local: u32 = 0u; local < d_head; local = local + 1u) {
                query_coeff = query_coeff
                    + state.data[q_scratch_offset + local]
                        * bases[key_basis_head + local * rank + latent];
            }
            state.data[query_latent_offset + latent] = query_coeff;
        }

        // Scores in logical oldest-to-newest order.
        var maximum = -3.402823466e+38;
        for (var logical: u32 = 0u; logical < new_len; logical = logical + 1u) {
            let slot = physical_slot(logical, oldest, capacity);
            var score = 0.0;
            for (var latent: u32 = 0u; latent < rank; latent = latent + 1u) {
                let slot_index = (head * capacity + slot) * rank + latent;
                score = score + state.data[query_latent_offset + latent] * state.data[slot_index];
            }
            score = score * scale;
            state.data[scores_offset + logical] = score;
            maximum = max(maximum, score);
        }

        // Stable fixed-order softmax. Scratch scores become normalised weights.
        var denominator = 0.0;
        for (var logical: u32 = 0u; logical < new_len; logical = logical + 1u) {
            let weight = exp(state.data[scores_offset + logical] - maximum);
            state.data[scores_offset + logical] = weight;
            denominator = denominator + weight;
        }
        for (var logical: u32 = 0u; logical < new_len; logical = logical + 1u) {
            state.data[scores_offset + logical] = state.data[scores_offset + logical] / denominator;
        }

        // Q-latent scratch is no longer needed, so reuse it for the aggregated V
        // coefficients before reconstructing this head's dense context.
        for (var latent: u32 = 0u; latent < rank; latent = latent + 1u) {
            var coefficient = 0.0;
            for (var logical: u32 = 0u; logical < new_len; logical = logical + 1u) {
                let slot = physical_slot(logical, oldest, capacity);
                let slot_index = (head * capacity + slot) * rank + latent;
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

    // Fused output projection from the concatenated resident head context.
    for (var column: u32 = 0u; column < d_model; column = column + 1u) {
        var value = weights[bo_offset + column];
        for (var input_index: u32 = 0u; input_index < d_model; input_index = input_index + 1u) {
            value = value
                + state.data[context_offset + input_index]
                    * weights[wo_offset + input_index * d_model + column];
        }
        io[d_model + column] = value;
    }

    state.len = new_len;
    state.next_slot = new_next;
}
"#;

/// Borrowed per-head basis matrices used to construct the resident MHA.
///
/// Both matrices are row-major `(d_head, rank)` arrays.
#[derive(Clone, Copy)]
pub struct WgpuLatentHeadBasis<'a> {
    pub key: &'a [f32],
    pub value: &'a [f32],
}

/// Construction or execution error for [`WgpuResidentLatentMha`].
#[derive(Debug)]
pub enum WgpuResidentLatentMhaError {
    InvalidConfig(&'static str),
    HeadCount {
        expected: usize,
        actual: usize,
    },
    BasisLength {
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

impl fmt::Display for WgpuResidentLatentMhaError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(output, "{message}"),
            Self::HeadCount { expected, actual } => {
                write!(output, "latent MHA head mismatch: expected {expected}, got {actual}")
            }
            Self::BasisLength {
                head,
                expected,
                key,
                value,
            } => write!(
                output,
                "latent MHA basis length mismatch for head {head}: expected {expected}, key={key}, value={value}"
            ),
            Self::VectorLength {
                name,
                expected,
                actual,
            } => write!(output, "{name} length mismatch: expected {expected}, got {actual}"),
            Self::PositionMismatch { expected, actual } => write!(
                output,
                "latent MHA position mismatch: expected {expected}, got {actual}"
            ),
            Self::Compute(error) => write!(output, "{error}"),
        }
    }
}

impl std::error::Error for WgpuResidentLatentMhaError {}

impl From<ComputeError> for WgpuResidentLatentMhaError {
    fn from(error: ComputeError) -> Self {
        Self::Compute(error)
    }
}

/// Persistent allocation and transfer telemetry for the fused MHA runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuResidentLatentMhaTelemetry {
    pub capacity_tokens: usize,
    pub resident_tokens: usize,
    pub steps: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_head: usize,
    pub rank: usize,
    pub resident_bytes: usize,
    pub upload_bytes_per_step: usize,
    pub download_bytes_per_step: usize,
    pub next_write_slot: usize,
}

/// Fused WGPU inference snapshot of one [`MultiHeadAttention`] layer.
///
/// Weights and bases are copied to device memory at construction. Training and
/// the legacy tape path are untouched. If model weights change later, call
/// [`Self::reload_weights`] before continuing inference.
pub struct WgpuResidentLatentMha {
    adapter: WgpuComputeAdapter,
    weights: WgpuComputeBuffer,
    bases: WgpuComputeBuffer,
    state: WgpuComputeBuffer,
    io: WgpuComputeBuffer,
    kernel: WgpuComputeKernel,
    stream: WgpuComputeStream,
    capacity: usize,
    d_model: usize,
    n_heads: usize,
    d_head: usize,
    rank: usize,
    resident_tokens: usize,
    steps: usize,
    next_slot: usize,
    resident_bytes: usize,
}

impl WgpuResidentLatentMha {
    /// Acquires one WGPU context and snapshots an MHA layer into resident state.
    pub fn new(
        mha: &MultiHeadAttention,
        capacity: usize,
        rank: usize,
        heads: &[WgpuLatentHeadBasis<'_>],
    ) -> Result<Self, WgpuResidentLatentMhaError> {
        let context = WgpuContext::new().map_err(|_| {
            WgpuResidentLatentMhaError::InvalidConfig("WGPU backend is unavailable")
        })?;
        Self::from_context(context, mha, capacity, rank, heads)
    }

    /// Constructs from a caller-owned shared context without acquiring another
    /// WGPU device. This is the preferred path for multi-layer sessions.
    pub fn from_context(
        context: WgpuContext,
        mha: &MultiHeadAttention,
        capacity: usize,
        rank: usize,
        heads: &[WgpuLatentHeadBasis<'_>],
    ) -> Result<Self, WgpuResidentLatentMhaError> {
        validate_topology(mha, capacity, rank, heads)?;

        let d_model = mha.d_model;
        let n_heads = mha.n_heads;
        let d_head = mha.d_head;
        let weights_data = pack_weights(mha)?;
        let bases_data = pack_bases(heads, d_head, rank)?;

        let coefficient_elements = n_heads
            .checked_mul(capacity)
            .and_then(|value| value.checked_mul(rank))
            .and_then(|value| value.checked_mul(2))
            .ok_or(WgpuResidentLatentMhaError::InvalidConfig(
                "resident MHA coefficient storage overflows usize",
            ))?;
        let scratch_elements = rank
            .checked_add(capacity)
            .and_then(|value| value.checked_add(d_model))
            .and_then(|value| value.checked_add(d_head.checked_mul(3)?))
            .ok_or(WgpuResidentLatentMhaError::InvalidConfig(
                "resident MHA scratch storage overflows usize",
            ))?;
        let state_data_elements = coefficient_elements.checked_add(scratch_elements).ok_or(
            WgpuResidentLatentMhaError::InvalidConfig("resident MHA storage overflows usize"),
        )?;

        let weights_bytes = bytes_for_f32(weights_data.len())?;
        let bases_bytes = bytes_for_f32(bases_data.len())?;
        let state_bytes = HEADER_WORDS
            .checked_mul(core::mem::size_of::<u32>())
            .and_then(|header| header.checked_add(bytes_for_f32(state_data_elements).ok()?))
            .ok_or(WgpuResidentLatentMhaError::InvalidConfig(
                "resident MHA state bytes overflow usize",
            ))?;
        let io_bytes = bytes_for_f32(d_model.checked_mul(2).ok_or(
            WgpuResidentLatentMhaError::InvalidConfig("resident MHA IO storage overflows usize"),
        )?)?;

        let adapter = WgpuComputeAdapter::from_context(context);
        let weights = adapter.allocate(weights_bytes, 4, MemorySpace::Device)?;
        let bases = adapter.allocate(bases_bytes, 4, MemorySpace::Device)?;
        let state = adapter.allocate(state_bytes, 4, MemorySpace::Device)?;
        let io = adapter.allocate(io_bytes, 4, MemorySpace::Device)?;

        adapter.write(&weights, 0, bytemuck::cast_slice(&weights_data))?;
        adapter.write(&bases, 0, bytemuck::cast_slice(&bases_data))?;
        let header = [
            usize_to_u32(d_model, "d_model exceeds WGPU u32 range")?,
            usize_to_u32(n_heads, "n_heads exceeds WGPU u32 range")?,
            usize_to_u32(d_head, "d_head exceeds WGPU u32 range")?,
            usize_to_u32(rank, "rank exceeds WGPU u32 range")?,
            usize_to_u32(capacity, "capacity exceeds WGPU u32 range")?,
            0,
            0,
            0,
        ];
        adapter.write(&state, 0, bytemuck::cast_slice(&header))?;

        let module = KernelModule::new(
            KernelFormat::Wgsl,
            "main",
            FUSED_MHA_WGSL.as_bytes().to_vec(),
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
            d_model,
            n_heads,
            d_head,
            rank,
            resident_tokens: 0,
            steps: 0,
            next_slot: 0,
            resident_bytes: weights_bytes + bases_bytes + state_bytes + io_bytes,
        })
    }

    #[must_use]
    pub const fn telemetry(&self) -> WgpuResidentLatentMhaTelemetry {
        WgpuResidentLatentMhaTelemetry {
            capacity_tokens: self.capacity,
            resident_tokens: self.resident_tokens,
            steps: self.steps,
            d_model: self.d_model,
            n_heads: self.n_heads,
            d_head: self.d_head,
            rank: self.rank,
            resident_bytes: self.resident_bytes,
            upload_bytes_per_step: self.d_model * F32_BYTES,
            download_bytes_per_step: self.d_model * F32_BYTES,
            next_write_slot: self.next_slot,
        }
    }

    /// Executes one incremental MHA token at the exact expected position and
    /// writes the final O-projected result into caller-owned memory.
    pub fn infer_step_at_into(
        &mut self,
        input: &[f32],
        pos: usize,
        output: &mut [f32],
    ) -> Result<(), WgpuResidentLatentMhaError> {
        if pos != self.steps {
            return Err(WgpuResidentLatentMhaError::PositionMismatch {
                expected: self.steps,
                actual: pos,
            });
        }
        self.require_vector("input", input)?;
        if output.len() != self.d_model {
            return Err(WgpuResidentLatentMhaError::VectorLength {
                name: "output",
                expected: self.d_model,
                actual: output.len(),
            });
        }

        self.adapter
            .write(&self.io, 0, bytemuck::cast_slice(input))?;
        let event = self.launch()?;
        self.adapter.wait(&event)?;

        // The kernel has committed the same deterministic ring transition.
        self.steps = self.steps.saturating_add(1);
        self.resident_tokens = self.resident_tokens.saturating_add(1).min(self.capacity);
        self.next_slot = (self.next_slot + 1) % self.capacity;

        self.adapter.read(
            &self.io,
            self.d_model * F32_BYTES,
            bytemuck::cast_slice_mut(output),
        )?;
        Ok(())
    }

    /// Convenience wrapper allocating only the returned dense model-width row.
    pub fn infer_step_at(
        &mut self,
        input: &[f32],
        pos: usize,
    ) -> Result<Vec<f32>, WgpuResidentLatentMhaError> {
        let mut output = vec![0.0; self.d_model];
        self.infer_step_at_into(input, pos, &mut output)?;
        Ok(output)
    }

    /// Executes at the runtime's next expected position.
    pub fn infer_step(
        &mut self,
        input: &[f32],
    ) -> Result<Vec<f32>, WgpuResidentLatentMhaError> {
        self.infer_step_at(input, self.steps)
    }

    /// Replaces the persistent Q/K/V/O snapshot without reallocating device
    /// storage. Topology must remain identical.
    pub fn reload_weights(
        &mut self,
        mha: &MultiHeadAttention,
    ) -> Result<(), WgpuResidentLatentMhaError> {
        if mha.d_model != self.d_model
            || mha.n_heads != self.n_heads
            || mha.d_head != self.d_head
        {
            return Err(WgpuResidentLatentMhaError::InvalidConfig(
                "cannot reload weights from a different MHA topology",
            ));
        }
        let packed = pack_weights(mha)?;
        self.adapter
            .write(&self.weights, 0, bytemuck::cast_slice(&packed))?;
        Ok(())
    }

    /// Clears the logical ring while keeping all persistent WGPU allocations.
    pub fn reset(&mut self) -> Result<(), WgpuResidentLatentMhaError> {
        let cleared = [0u32, 0u32];
        self.adapter.write(
            &self.state,
            5 * core::mem::size_of::<u32>(),
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
    ) -> Result<(), WgpuResidentLatentMhaError> {
        if vector.len() != self.d_model {
            return Err(WgpuResidentLatentMhaError::VectorLength {
                name,
                expected: self.d_model,
                actual: vector.len(),
            });
        }
        Ok(())
    }

    fn launch(&self) -> Result<WgpuComputeEvent, WgpuResidentLatentMhaError> {
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
    mha: &MultiHeadAttention,
    capacity: usize,
    rank: usize,
    heads: &[WgpuLatentHeadBasis<'_>],
) -> Result<(), WgpuResidentLatentMhaError> {
    if capacity == 0 {
        return Err(WgpuResidentLatentMhaError::InvalidConfig(
            "resident MHA capacity must be non-zero",
        ));
    }
    if mha.d_model == 0 || mha.n_heads == 0 || mha.d_head == 0 {
        return Err(WgpuResidentLatentMhaError::InvalidConfig(
            "resident MHA topology must be non-zero",
        ));
    }
    if mha.d_head.checked_mul(mha.n_heads) != Some(mha.d_model) {
        return Err(WgpuResidentLatentMhaError::InvalidConfig(
            "resident MHA requires d_model == n_heads * d_head",
        ));
    }
    if rank == 0 || rank > mha.d_head {
        return Err(WgpuResidentLatentMhaError::InvalidConfig(
            "resident MHA rank must be in 1..=d_head",
        ));
    }
    if heads.len() != mha.n_heads {
        return Err(WgpuResidentLatentMhaError::HeadCount {
            expected: mha.n_heads,
            actual: heads.len(),
        });
    }
    let expected = mha.d_head.checked_mul(rank).ok_or(
        WgpuResidentLatentMhaError::InvalidConfig("resident MHA basis shape overflows usize"),
    )?;
    for (head, basis) in heads.iter().enumerate() {
        if basis.key.len() != expected || basis.value.len() != expected {
            return Err(WgpuResidentLatentMhaError::BasisLength {
                head,
                expected,
                key: basis.key.len(),
                value: basis.value.len(),
            });
        }
    }
    validate_weight_shapes(mha)
}

fn validate_weight_shapes(mha: &MultiHeadAttention) -> Result<(), WgpuResidentLatentMhaError> {
    let matrix = mha.d_model.checked_mul(mha.d_model).ok_or(
        WgpuResidentLatentMhaError::InvalidConfig("resident MHA matrix shape overflows usize"),
    )?;
    let linears = [&mha.w_q, &mha.w_k, &mha.w_v, &mha.w_o];
    for linear in linears {
        if linear.in_features != mha.d_model
            || linear.out_features != mha.d_model
            || linear.weight.data.len() != matrix
            || linear.bias.data.len() != mha.d_model
        {
            return Err(WgpuResidentLatentMhaError::InvalidConfig(
                "resident MHA requires square model-width Q/K/V/O projections",
            ));
        }
    }
    Ok(())
}

fn pack_weights(
    mha: &MultiHeadAttention,
) -> Result<Vec<f32>, WgpuResidentLatentMhaError> {
    validate_weight_shapes(mha)?;
    let matrix = mha.d_model * mha.d_model;
    let total = matrix
        .checked_mul(4)
        .and_then(|value| value.checked_add(mha.d_model.checked_mul(4)?))
        .ok_or(WgpuResidentLatentMhaError::InvalidConfig(
            "resident MHA packed weights overflow usize",
        ))?;
    let mut packed = Vec::with_capacity(total);
    packed.extend_from_slice(&mha.w_q.weight.data);
    packed.extend_from_slice(&mha.w_k.weight.data);
    packed.extend_from_slice(&mha.w_v.weight.data);
    packed.extend_from_slice(&mha.w_o.weight.data);
    packed.extend_from_slice(&mha.w_q.bias.data);
    packed.extend_from_slice(&mha.w_k.bias.data);
    packed.extend_from_slice(&mha.w_v.bias.data);
    packed.extend_from_slice(&mha.w_o.bias.data);
    Ok(packed)
}

fn pack_bases(
    heads: &[WgpuLatentHeadBasis<'_>],
    d_head: usize,
    rank: usize,
) -> Result<Vec<f32>, WgpuResidentLatentMhaError> {
    let per_head = d_head.checked_mul(rank).ok_or(
        WgpuResidentLatentMhaError::InvalidConfig("resident MHA basis shape overflows usize"),
    )?;
    let total = heads
        .len()
        .checked_mul(per_head)
        .and_then(|value| value.checked_mul(2))
        .ok_or(WgpuResidentLatentMhaError::InvalidConfig(
            "resident MHA packed bases overflow usize",
        ))?;
    let mut packed = Vec::with_capacity(total);
    for head in heads {
        packed.extend_from_slice(head.key);
    }
    for head in heads {
        packed.extend_from_slice(head.value);
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

fn bytes_for_f32(elements: usize) -> Result<usize, WgpuResidentLatentMhaError> {
    elements
        .checked_mul(F32_BYTES)
        .ok_or(WgpuResidentLatentMhaError::InvalidConfig(
            "resident MHA f32 buffer size overflows usize",
        ))
}

fn usize_to_u32(
    value: usize,
    message: &'static str,
) -> Result<u32, WgpuResidentLatentMhaError> {
    u32::try_from(value).map_err(|_| WgpuResidentLatentMhaError::InvalidConfig(message))
}
