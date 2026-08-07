//! Device-resident Elastic Latent KV cache for the WGPU backend.
//!
//! This module is intentionally scoped to the persistent latent-state substrate:
//! key/value latent coefficients, attention scratch and the reconstructed context
//! remain in device memory across decode steps. Per-step dense q/k/v vectors are
//! uploaded and only the final dense context is read back.
//!
//! The kernels use a single invocation and fixed loop order. This is not yet a
//! throughput-optimised attention kernel; it is the deterministic correctness
//! baseline on which wider parallel kernels can be validated.

use core::fmt;

use crate::{
    WgpuComputeAdapter, WgpuComputeBuffer, WgpuComputeEvent, WgpuComputeKernel,
    WgpuComputeStream, WgpuContext,
};
use scirust_compute::{
    BufferAccess, BufferBinding, ComputeBackend, ComputeError, KernelFormat, KernelModule,
    LaunchConfig, MemorySpace,
};

const PARAM_WORDS: usize = 8;
const F32_BYTES: usize = core::mem::size_of::<f32>();

const APPEND_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> io: array<f32>;
@group(0) @binding(1) var<storage, read> basis: array<f32>;
@group(0) @binding(2) var<storage, read_write> cache: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

@compute @workgroup_size(1)
fn main() {
    let dim = params[0];
    let rank = params[1];
    let capacity = params[2];
    let slot = params[3];
    let value_basis_offset = dim * rank;
    let value_cache_offset = capacity * rank;

    for (var j: u32 = 0u; j < rank; j = j + 1u) {
        var key_coeff: f32 = 0.0;
        var value_coeff: f32 = 0.0;
        for (var i: u32 = 0u; i < dim; i = i + 1u) {
            key_coeff = key_coeff + io[i] * basis[i * rank + j];
            value_coeff = value_coeff
                + io[dim + i] * basis[value_basis_offset + i * rank + j];
        }
        cache[slot * rank + j] = key_coeff;
        cache[value_cache_offset + slot * rank + j] = value_coeff;
    }
}
"#;

const ATTEND_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> io: array<f32>;
@group(0) @binding(1) var<storage, read> basis: array<f32>;
@group(0) @binding(2) var<storage, read_write> cache: array<f32>;
@group(0) @binding(3) var<storage, read> params: array<u32>;

fn physical_slot(logical: u32, oldest: u32, capacity: u32) -> u32 {
    return (oldest + logical) % capacity;
}

@compute @workgroup_size(1)
fn main() {
    let dim = params[0];
    let rank = params[1];
    let capacity = params[2];
    let len = params[3];
    let oldest = params[4];

    let value_basis_offset = dim * rank;
    let value_cache_offset = capacity * rank;
    let query_offset = 2u * capacity * rank;
    let scores_offset = query_offset + rank;
    let context_offset = scores_offset + capacity;

    // Project the dense query into the key latent basis.
    for (var j: u32 = 0u; j < rank; j = j + 1u) {
        var coeff: f32 = 0.0;
        for (var i: u32 = 0u; i < dim; i = i + 1u) {
            coeff = coeff + io[i] * basis[i * rank + j];
        }
        cache[query_offset + j] = coeff;
    }

    // Fixed-order score computation in logical (oldest -> newest) order.
    let scale = 1.0 / sqrt(f32(dim));
    var maximum: f32 = -3.402823466e+38;
    for (var logical: u32 = 0u; logical < len; logical = logical + 1u) {
        let slot = physical_slot(logical, oldest, capacity);
        var score: f32 = 0.0;
        for (var j: u32 = 0u; j < rank; j = j + 1u) {
            score = score + cache[query_offset + j] * cache[slot * rank + j];
        }
        score = score * scale;
        cache[scores_offset + logical] = score;
        maximum = max(maximum, score);
    }

    // Stable softmax. The score scratch is overwritten with normalised weights.
    var denominator: f32 = 0.0;
    for (var logical: u32 = 0u; logical < len; logical = logical + 1u) {
        let weight = exp(cache[scores_offset + logical] - maximum);
        cache[scores_offset + logical] = weight;
        denominator = denominator + weight;
    }
    for (var logical: u32 = 0u; logical < len; logical = logical + 1u) {
        cache[scores_offset + logical] = cache[scores_offset + logical] / denominator;
    }

    // Weighted value aggregation remains latent.
    for (var j: u32 = 0u; j < rank; j = j + 1u) {
        var coeff: f32 = 0.0;
        for (var logical: u32 = 0u; logical < len; logical = logical + 1u) {
            let slot = physical_slot(logical, oldest, capacity);
            coeff = coeff
                + cache[scores_offset + logical]
                    * cache[value_cache_offset + slot * rank + j];
        }
        cache[context_offset + j] = coeff;
    }

    // Reconstruct only the final dense context. It is written to the second half
    // of the IO buffer so the query remains intact for the whole dispatch.
    for (var i: u32 = 0u; i < dim; i = i + 1u) {
        var value: f32 = 0.0;
        for (var j: u32 = 0u; j < rank; j = j + 1u) {
            value = value
                + cache[context_offset + j]
                    * basis[value_basis_offset + i * rank + j];
        }
        io[dim + i] = value;
    }
}
"#;

/// Construction or execution error for [`WgpuResidentLatentKvCache`].
#[derive(Debug)]
pub enum WgpuResidentLatentKvError {
    InvalidConfig(&'static str),
    BasisLength {
        expected: usize,
        key: usize,
        value: usize,
    },
    VectorLength {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    EmptyCache,
    Compute(ComputeError),
}

impl fmt::Display for WgpuResidentLatentKvError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(formatter, "{message}"),
            Self::BasisLength {
                expected,
                key,
                value,
            } => write!(
                formatter,
                "latent basis length mismatch: expected {expected}, key={key}, value={value}"
            ),
            Self::VectorLength {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} length mismatch: expected {expected}, got {actual}"
            ),
            Self::EmptyCache => write!(formatter, "cannot attend over an empty latent KV cache"),
            Self::Compute(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for WgpuResidentLatentKvError {}

impl From<ComputeError> for WgpuResidentLatentKvError {
    fn from(error: ComputeError) -> Self {
        Self::Compute(error)
    }
}

/// Snapshot of the persistent device allocation and ring state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuResidentLatentKvTelemetry {
    pub capacity_tokens: usize,
    pub resident_tokens: usize,
    pub dimension: usize,
    pub rank: usize,
    pub resident_bytes: usize,
    pub next_write_slot: usize,
}

/// Persistent WGPU latent KV ring.
///
/// The basis and cache buffers are allocated once. `append` and `attention_into`
/// reuse them without growing persistent device storage. The implementation uses
/// the repository's public `ComputeBackend` contract rather than reaching into
/// private `wgpu` handles.
pub struct WgpuResidentLatentKvCache {
    adapter: WgpuComputeAdapter,
    basis: WgpuComputeBuffer,
    cache: WgpuComputeBuffer,
    io: WgpuComputeBuffer,
    params: WgpuComputeBuffer,
    append_kernel: WgpuComputeKernel,
    attend_kernel: WgpuComputeKernel,
    stream: WgpuComputeStream,
    capacity: usize,
    dimension: usize,
    rank: usize,
    len: usize,
    next_slot: usize,
    resident_bytes: usize,
}

impl WgpuResidentLatentKvCache {
    /// Acquires a WGPU context and constructs one persistent latent cache.
    pub fn new(
        capacity: usize,
        dimension: usize,
        rank: usize,
        key_basis: &[f32],
        value_basis: &[f32],
    ) -> Result<Self, WgpuResidentLatentKvError> {
        let context = WgpuContext::new().map_err(|_| {
            WgpuResidentLatentKvError::InvalidConfig("WGPU backend is unavailable")
        })?;
        Self::from_context(context, capacity, dimension, rank, key_basis, value_basis)
    }

    /// Constructs a cache from an existing context so multiple heads can share
    /// one WGPU device and queue.
    pub fn from_context(
        context: WgpuContext,
        capacity: usize,
        dimension: usize,
        rank: usize,
        key_basis: &[f32],
        value_basis: &[f32],
    ) -> Result<Self, WgpuResidentLatentKvError> {
        if capacity == 0 {
            return Err(WgpuResidentLatentKvError::InvalidConfig(
                "latent cache capacity must be non-zero",
            ));
        }
        if dimension == 0 {
            return Err(WgpuResidentLatentKvError::InvalidConfig(
                "latent cache dimension must be non-zero",
            ));
        }
        if rank == 0 || rank > dimension {
            return Err(WgpuResidentLatentKvError::InvalidConfig(
                "latent cache rank must be in 1..=dimension",
            ));
        }

        let basis_elements = dimension
            .checked_mul(rank)
            .ok_or(WgpuResidentLatentKvError::InvalidConfig(
                "latent basis shape overflows usize",
            ))?;
        if key_basis.len() != basis_elements || value_basis.len() != basis_elements {
            return Err(WgpuResidentLatentKvError::BasisLength {
                expected: basis_elements,
                key: key_basis.len(),
                value: value_basis.len(),
            });
        }

        let cache_coefficients = capacity
            .checked_mul(rank)
            .and_then(|value| value.checked_mul(2))
            .ok_or(WgpuResidentLatentKvError::InvalidConfig(
                "latent cache coefficient storage overflows usize",
            ))?;
        let scratch_elements = rank
            .checked_mul(2)
            .and_then(|value| value.checked_add(capacity))
            .ok_or(WgpuResidentLatentKvError::InvalidConfig(
                "latent cache scratch storage overflows usize",
            ))?;
        let cache_elements = cache_coefficients
            .checked_add(scratch_elements)
            .ok_or(WgpuResidentLatentKvError::InvalidConfig(
                "latent cache storage overflows usize",
            ))?;
        let io_elements = dimension
            .checked_mul(2)
            .ok_or(WgpuResidentLatentKvError::InvalidConfig(
                "latent IO storage overflows usize",
            ))?;
        let packed_basis_elements = basis_elements
            .checked_mul(2)
            .ok_or(WgpuResidentLatentKvError::InvalidConfig(
                "packed latent basis storage overflows usize",
            ))?;

        let basis_bytes = bytes_for_f32(packed_basis_elements)?;
        let cache_bytes = bytes_for_f32(cache_elements)?;
        let io_bytes = bytes_for_f32(io_elements)?;
        let params_bytes = PARAM_WORDS * core::mem::size_of::<u32>();

        let adapter = WgpuComputeAdapter::from_context(context);
        let basis = adapter.allocate(basis_bytes, 4, MemorySpace::Device)?;
        let cache = adapter.allocate(cache_bytes, 4, MemorySpace::Device)?;
        let io = adapter.allocate(io_bytes, 4, MemorySpace::Device)?;
        let params = adapter.allocate(params_bytes, 4, MemorySpace::Device)?;

        let mut packed_basis = Vec::with_capacity(packed_basis_elements);
        packed_basis.extend_from_slice(key_basis);
        packed_basis.extend_from_slice(value_basis);
        adapter.write(&basis, 0, bytemuck::cast_slice(&packed_basis))?;

        let append_module = KernelModule::new(
            KernelFormat::Wgsl,
            "main",
            APPEND_WGSL.as_bytes().to_vec(),
        )?;
        let attend_module = KernelModule::new(
            KernelFormat::Wgsl,
            "main",
            ATTEND_WGSL.as_bytes().to_vec(),
        )?;
        let append_kernel = adapter.compile(&append_module)?;
        let attend_kernel = adapter.compile(&attend_module)?;
        let stream = adapter.create_stream()?;

        Ok(Self {
            adapter,
            basis,
            cache,
            io,
            params,
            append_kernel,
            attend_kernel,
            stream,
            capacity,
            dimension,
            rank,
            len: 0,
            next_slot: 0,
            resident_bytes: basis_bytes + cache_bytes + io_bytes + params_bytes,
        })
    }

    #[must_use]
    pub const fn telemetry(&self) -> WgpuResidentLatentKvTelemetry {
        WgpuResidentLatentKvTelemetry {
            capacity_tokens: self.capacity,
            resident_tokens: self.len,
            dimension: self.dimension,
            rank: self.rank,
            resident_bytes: self.resident_bytes,
            next_write_slot: self.next_slot,
        }
    }

    /// Projects and appends one dense key/value pair into the resident ring.
    pub fn append(
        &mut self,
        key: &[f32],
        value: &[f32],
    ) -> Result<(), WgpuResidentLatentKvError> {
        self.require_vector("key", key)?;
        self.require_vector("value", value)?;

        self.adapter.write(&self.io, 0, bytemuck::cast_slice(key))?;
        self.adapter.write(
            &self.io,
            self.dimension * F32_BYTES,
            bytemuck::cast_slice(value),
        )?;

        let params = [
            usize_to_u32(self.dimension, "dimension exceeds WGPU u32 range")?,
            usize_to_u32(self.rank, "rank exceeds WGPU u32 range")?,
            usize_to_u32(self.capacity, "capacity exceeds WGPU u32 range")?,
            usize_to_u32(self.next_slot, "write slot exceeds WGPU u32 range")?,
            0,
            0,
            0,
            0,
        ];
        self.adapter
            .write(&self.params, 0, bytemuck::cast_slice(&params))?;

        let event = self.launch_append()?;
        self.adapter.wait(&event)?;

        self.next_slot = (self.next_slot + 1) % self.capacity;
        self.len = self.len.saturating_add(1).min(self.capacity);
        Ok(())
    }

    /// Computes attention against the resident ring and writes the reconstructed
    /// dense context into caller-owned memory. No output `Vec` is allocated.
    pub fn attention_into(
        &mut self,
        query: &[f32],
        output: &mut [f32],
    ) -> Result<(), WgpuResidentLatentKvError> {
        self.require_vector("query", query)?;
        if output.len() != self.dimension {
            return Err(WgpuResidentLatentKvError::VectorLength {
                name: "output",
                expected: self.dimension,
                actual: output.len(),
            });
        }
        if self.len == 0 {
            return Err(WgpuResidentLatentKvError::EmptyCache);
        }

        self.adapter
            .write(&self.io, 0, bytemuck::cast_slice(query))?;
        let oldest = if self.len < self.capacity {
            0
        } else {
            self.next_slot
        };
        let params = [
            usize_to_u32(self.dimension, "dimension exceeds WGPU u32 range")?,
            usize_to_u32(self.rank, "rank exceeds WGPU u32 range")?,
            usize_to_u32(self.capacity, "capacity exceeds WGPU u32 range")?,
            usize_to_u32(self.len, "resident length exceeds WGPU u32 range")?,
            usize_to_u32(oldest, "oldest slot exceeds WGPU u32 range")?,
            0,
            0,
            0,
        ];
        self.adapter
            .write(&self.params, 0, bytemuck::cast_slice(&params))?;

        let event = self.launch_attention()?;
        self.adapter.wait(&event)?;
        self.adapter.read(
            &self.io,
            self.dimension * F32_BYTES,
            bytemuck::cast_slice_mut(output),
        )?;
        Ok(())
    }

    /// Convenience wrapper that allocates only the returned dense context.
    pub fn attention(
        &mut self,
        query: &[f32],
    ) -> Result<Vec<f32>, WgpuResidentLatentKvError> {
        let mut output = vec![0.0; self.dimension];
        self.attention_into(query, &mut output)?;
        Ok(output)
    }

    /// Append the current token then attend over the updated sliding window.
    pub fn append_and_attention_into(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        output: &mut [f32],
    ) -> Result<(), WgpuResidentLatentKvError> {
        self.append(key, value)?;
        self.attention_into(query, output)
    }

    fn require_vector(
        &self,
        name: &'static str,
        vector: &[f32],
    ) -> Result<(), WgpuResidentLatentKvError> {
        if vector.len() != self.dimension {
            return Err(WgpuResidentLatentKvError::VectorLength {
                name,
                expected: self.dimension,
                actual: vector.len(),
            });
        }
        Ok(())
    }

    fn launch_append(&self) -> Result<WgpuComputeEvent, WgpuResidentLatentKvError> {
        let bindings = [
            binding(0, &self.io, BufferAccess::ReadOnly),
            binding(1, &self.basis, BufferAccess::ReadOnly),
            binding(2, &self.cache, BufferAccess::ReadWrite),
            binding(3, &self.params, BufferAccess::ReadOnly),
        ];
        let config = LaunchConfig::new([1, 1, 1], [1, 1, 1], 0)?;
        Ok(self
            .adapter
            .launch(&self.append_kernel, &self.stream, config, &bindings)?)
    }

    fn launch_attention(&self) -> Result<WgpuComputeEvent, WgpuResidentLatentKvError> {
        let bindings = [
            binding(0, &self.io, BufferAccess::ReadWrite),
            binding(1, &self.basis, BufferAccess::ReadOnly),
            binding(2, &self.cache, BufferAccess::ReadWrite),
            binding(3, &self.params, BufferAccess::ReadOnly),
        ];
        let config = LaunchConfig::new([1, 1, 1], [1, 1, 1], 0)?;
        Ok(self
            .adapter
            .launch(&self.attend_kernel, &self.stream, config, &bindings)?)
    }
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

fn bytes_for_f32(elements: usize) -> Result<usize, WgpuResidentLatentKvError> {
    elements
        .checked_mul(F32_BYTES)
        .ok_or(WgpuResidentLatentKvError::InvalidConfig(
            "resident f32 buffer size overflows usize",
        ))
}

fn usize_to_u32(
    value: usize,
    message: &'static str,
) -> Result<u32, WgpuResidentLatentKvError> {
    u32::try_from(value).map_err(|_| WgpuResidentLatentKvError::InvalidConfig(message))
}
