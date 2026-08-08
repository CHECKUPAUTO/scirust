//! Exact bounded top-k sampling with one 64-lane WGPU workgroup.
//!
//! This is the Phase 24 parallel candidate-selection primitive. It is additive:
//! [`crate::WgpuDeterministicSampler`] remains the production/oracle WGPU path
//! until benchmark evidence justifies integration.
//!
//! Only probability comparisons are parallelized. Top-p normalization, final
//! probability summation, categorical scanning and PCG state mutation remain on
//! lane 0 in the historical token order so floating-point accumulation order is
//! unchanged.

use core::fmt;

use crate::{
    WgpuComputeAdapter, WgpuComputeBuffer, WgpuComputeEvent, WgpuComputeKernel, WgpuComputeStream,
    WgpuContext, WgpuDeterministicSamplerError, WgpuDeterministicSamplerTelemetry,
};
use scirust_compute::{
    BufferAccess, BufferBinding, ComputeBackend, ComputeError, KernelFormat, KernelModule,
    LaunchConfig, MemorySpace,
};
use scirust_core::nn::sampling::SamplingConfig;

pub const PARALLEL_TOP_K_LANES: usize = 64;
pub const PARALLEL_TOP_K_MAX: usize = 256;

const STATE_WORDS: usize = 11;
const F32_BYTES: usize = core::mem::size_of::<f32>();
const U32_BYTES: usize = core::mem::size_of::<u32>();
const MAX_EXACT_F32_INDEX: usize = 1 << 24;
const PCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
const TOP_K_PLACEHOLDER: &str = "__TOP_K__";

const PARALLEL_TOP_K_WGSL: &str = r#"
struct SamplerState {
    vocab_size: u32,
    temperature_bits: u32,
    top_k: u32,
    top_p_bits: u32,
    state_lo: u32,
    state_hi: u32,
    inc_lo: u32,
    inc_hi: u32,
    output_id: u32,
    draws: u32,
    enabled: u32,
};

struct WideU32 {
    lo: u32,
    hi: u32,
};

const TOP_K: u32 = __TOP_K__u;
const LANES: u32 = 64u;

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(2) var<storage, read_write> sampling_state: SamplerState;

var<workgroup> reduction_values: array<f32, 64>;
var<workgroup> candidate_ids: array<u32, 64>;
var<workgroup> candidate_positions: array<u32, 64>;

fn mul_u32_wide(a: u32, b: u32) -> WideU32 {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;

    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;

    let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
    let lo = (p00 & 0xffffu) | ((mid & 0xffffu) << 16u);
    let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
    return WideU32(lo, hi);
}

fn pcg_next_u32() -> u32 {
    let old_lo = sampling_state.state_lo;
    let old_hi = sampling_state.state_hi;
    let product_lo = mul_u32_wide(old_lo, 0x4c957f2du);
    let cross_lo_hi = old_lo * 0x5851f42du;
    let cross_hi_lo = old_hi * 0x4c957f2du;

    let added_lo = product_lo.lo + sampling_state.inc_lo;
    let carry = select(0u, 1u, added_lo < product_lo.lo);
    let added_hi = product_lo.hi
        + cross_lo_hi
        + cross_hi_lo
        + sampling_state.inc_hi
        + carry;
    sampling_state.state_lo = added_lo;
    sampling_state.state_hi = added_hi;

    let shifted18_lo = (old_lo >> 18u) | (old_hi << 14u);
    let shifted18_hi = old_hi >> 18u;
    let xor_lo = shifted18_lo ^ old_lo;
    let xor_hi = shifted18_hi ^ old_hi;
    let xorshifted = (xor_lo >> 27u) | (xor_hi << 5u);
    let rot = old_hi >> 27u;
    return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
}

fn candidate_is_better(candidate: u32, incumbent: u32, vocab_size: u32) -> bool {
    if (candidate >= vocab_size) {
        return false;
    }
    if (incumbent >= vocab_size) {
        return true;
    }
    let candidate_probability = scratch[candidate];
    let incumbent_probability = scratch[incumbent];
    return candidate_probability > incumbent_probability
        || (candidate_probability == incumbent_probability && candidate < incumbent);
}

fn greedy_argmax(vocab_size: u32) -> u32 {
    var best = 0u;
    var best_value = -3.402823466e+38;
    for (var token: u32 = 0u; token < vocab_size; token = token + 1u) {
        let value = logits[token];
        if (value > best_value) {
            best_value = value;
            best = token;
        }
    }
    return best;
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane: u32) {
    let vocab_size = sampling_state.vocab_size;
    let running = sampling_state.enabled != 0u;
    let temperature = bitcast<f32>(sampling_state.temperature_bits);
    let top_p = bitcast<f32>(sampling_state.top_p_bits);
    let order_offset = vocab_size;

    // Max reduction is comparison-only, so parallel evaluation cannot change
    // floating-point accumulation order. All barriers are unconditional.
    var local_max = -3.402823466e+38;
    if (running) {
        for (var token = lane; token < vocab_size; token = token + LANES) {
            local_max = max(local_max, logits[token]);
        }
    }
    reduction_values[lane] = local_max;
    workgroupBarrier();

    if (lane < 32u) { reduction_values[lane] = max(reduction_values[lane], reduction_values[lane + 32u]); }
    workgroupBarrier();
    if (lane < 16u) { reduction_values[lane] = max(reduction_values[lane], reduction_values[lane + 16u]); }
    workgroupBarrier();
    if (lane < 8u) { reduction_values[lane] = max(reduction_values[lane], reduction_values[lane + 8u]); }
    workgroupBarrier();
    if (lane < 4u) { reduction_values[lane] = max(reduction_values[lane], reduction_values[lane + 4u]); }
    workgroupBarrier();
    if (lane < 2u) { reduction_values[lane] = max(reduction_values[lane], reduction_values[lane + 2u]); }
    workgroupBarrier();
    if (lane < 1u) { reduction_values[lane] = max(reduction_values[lane], reduction_values[lane + 1u]); }
    workgroupBarrier();

    if (running) {
        let maximum = reduction_values[0];
        let inv_temperature = 1.0 / temperature;
        for (var token = lane; token < vocab_size; token = token + LANES) {
            scratch[token] = exp((logits[token] - maximum) * inv_temperature);
            scratch[order_offset + token] = f32(token);
        }
    }
    workgroupBarrier();

    // TOP_K is specialized into the shader source by the Rust constructor, so
    // every lane executes the exact same fixed number of barriers.
    for (var rank: u32 = 0u; rank < TOP_K; rank = rank + 1u) {
        var local_position = vocab_size;
        var local_id = vocab_size;
        if (running) {
            for (var position = rank + lane; position < vocab_size; position = position + LANES) {
                let token = u32(scratch[order_offset + position]);
                if (candidate_is_better(token, local_id, vocab_size)) {
                    local_id = token;
                    local_position = position;
                }
            }
        }
        candidate_ids[lane] = local_id;
        candidate_positions[lane] = local_position;
        workgroupBarrier();

        if (lane < 32u && candidate_is_better(candidate_ids[lane + 32u], candidate_ids[lane], vocab_size)) {
            candidate_ids[lane] = candidate_ids[lane + 32u];
            candidate_positions[lane] = candidate_positions[lane + 32u];
        }
        workgroupBarrier();
        if (lane < 16u && candidate_is_better(candidate_ids[lane + 16u], candidate_ids[lane], vocab_size)) {
            candidate_ids[lane] = candidate_ids[lane + 16u];
            candidate_positions[lane] = candidate_positions[lane + 16u];
        }
        workgroupBarrier();
        if (lane < 8u && candidate_is_better(candidate_ids[lane + 8u], candidate_ids[lane], vocab_size)) {
            candidate_ids[lane] = candidate_ids[lane + 8u];
            candidate_positions[lane] = candidate_positions[lane + 8u];
        }
        workgroupBarrier();
        if (lane < 4u && candidate_is_better(candidate_ids[lane + 4u], candidate_ids[lane], vocab_size)) {
            candidate_ids[lane] = candidate_ids[lane + 4u];
            candidate_positions[lane] = candidate_positions[lane + 4u];
        }
        workgroupBarrier();
        if (lane < 2u && candidate_is_better(candidate_ids[lane + 2u], candidate_ids[lane], vocab_size)) {
            candidate_ids[lane] = candidate_ids[lane + 2u];
            candidate_positions[lane] = candidate_positions[lane + 2u];
        }
        workgroupBarrier();
        if (lane < 1u && candidate_is_better(candidate_ids[lane + 1u], candidate_ids[lane], vocab_size)) {
            candidate_ids[lane] = candidate_ids[lane + 1u];
            candidate_positions[lane] = candidate_positions[lane + 1u];
        }
        workgroupBarrier();

        if (lane == 0u && running) {
            let best_position = candidate_positions[0];
            if (best_position != rank) {
                let temporary = scratch[order_offset + rank];
                scratch[order_offset + rank] = scratch[order_offset + best_position];
                scratch[order_offset + best_position] = temporary;
            }
        }
        workgroupBarrier();
    }

    if (running) {
        for (var rank = TOP_K + lane; rank < vocab_size; rank = rank + LANES) {
            let token = u32(scratch[order_offset + rank]);
            scratch[token] = 0.0;
        }
    }
    workgroupBarrier();

    // Preserve the scalar accumulation and PCG order exactly.
    if (lane == 0u && running) {
        if (top_p < 1.0) {
            var total = 0.0;
            for (var token: u32 = 0u; token < vocab_size; token = token + 1u) {
                total = total + scratch[token];
            }
            if (total > 0.0) {
                var cumulative = 0.0;
                var cutoff = TOP_K;
                for (var rank: u32 = 0u; rank < TOP_K; rank = rank + 1u) {
                    let token = u32(scratch[order_offset + rank]);
                    cumulative = cumulative + scratch[token] / total;
                    if (cumulative >= top_p) {
                        cutoff = rank + 1u;
                        break;
                    }
                }
                for (var rank = cutoff; rank < TOP_K; rank = rank + 1u) {
                    let token = u32(scratch[order_offset + rank]);
                    scratch[token] = 0.0;
                }
            }
        }

        var sum = 0.0;
        for (var token: u32 = 0u; token < vocab_size; token = token + 1u) {
            sum = sum + scratch[token];
        }
        if (sum <= 0.0) {
            sampling_state.output_id = greedy_argmax(vocab_size);
            return;
        }

        let random_word = pcg_next_u32();
        sampling_state.draws = sampling_state.draws + 1u;
        let random_unit = f32(random_word) / 4294967296.0;
        let threshold = random_unit * sum;
        var cumulative = 0.0;
        for (var token: u32 = 0u; token < vocab_size; token = token + 1u) {
            cumulative = cumulative + scratch[token];
            if (threshold < cumulative) {
                sampling_state.output_id = token;
                return;
            }
        }
        sampling_state.output_id = u32(scratch[order_offset]);
    }
}
"#;

/// Additive Phase 24 exact parallel bounded-top-k sampler.
pub struct WgpuParallelTopKSampler {
    adapter: WgpuComputeAdapter,
    logits: WgpuComputeBuffer,
    scratch: WgpuComputeBuffer,
    state: WgpuComputeBuffer,
    kernel: WgpuComputeKernel,
    stream: WgpuComputeStream,
    vocab_size: usize,
    top_k: usize,
    draws: usize,
    resident_bytes: usize,
    initial_state: u64,
    increment: u64,
}

impl fmt::Debug for WgpuParallelTopKSampler {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WgpuParallelTopKSampler")
            .field("vocab_size", &self.vocab_size)
            .field("top_k", &self.top_k)
            .field("draws", &self.draws)
            .finish_non_exhaustive()
    }
}

impl WgpuParallelTopKSampler {
    pub fn new(
        vocab_size: usize,
        config: SamplingConfig,
        seed: u64,
    ) -> Result<Self, WgpuDeterministicSamplerError> {
        let context = WgpuContext::new().map_err(|_| {
            WgpuDeterministicSamplerError::InvalidConfig("WGPU backend is unavailable")
        })?;
        Self::from_context(context, vocab_size, config, seed)
    }

    pub fn from_context(
        context: WgpuContext,
        vocab_size: usize,
        config: SamplingConfig,
        seed: u64,
    ) -> Result<Self, WgpuDeterministicSamplerError> {
        validate_parallel_config(vocab_size, &config)?;
        let (state_word, increment) = pcg_seed_state(seed);
        let state_words = [
            usize_to_u32(vocab_size, "sampler vocab size exceeds WGPU u32 range")?,
            config.temperature.to_bits(),
            usize_to_u32(config.top_k, "sampler top_k exceeds WGPU u32 range")?,
            config.top_p.to_bits(),
            state_word as u32,
            (state_word >> 32) as u32,
            increment as u32,
            (increment >> 32) as u32,
            0,
            0,
            1,
        ];

        let logits_bytes = bytes_for_f32(vocab_size)?;
        let scratch_elements = vocab_size.checked_mul(2).ok_or(
            WgpuDeterministicSamplerError::InvalidConfig("sampler scratch size overflows usize"),
        )?;
        let scratch_bytes = bytes_for_f32(scratch_elements)?;
        let state_bytes = STATE_WORDS.checked_mul(U32_BYTES).ok_or(
            WgpuDeterministicSamplerError::InvalidConfig("sampler state size overflows usize"),
        )?;

        let adapter = WgpuComputeAdapter::from_context(context);
        if adapter.capabilities().max_workgroup_size[0] < PARALLEL_TOP_K_LANES as u32
        {
            return Err(WgpuDeterministicSamplerError::InvalidConfig(
                "WGPU adapter does not support 64-lane workgroups",
            ));
        }
        let logits = adapter.allocate(logits_bytes, 4, MemorySpace::Device)?;
        let scratch = adapter.allocate(scratch_bytes, 4, MemorySpace::Device)?;
        let state = adapter.allocate(state_bytes, 4, MemorySpace::Device)?;
        adapter.write(&state, 0, bytemuck::cast_slice(&state_words))?;

        let source = PARALLEL_TOP_K_WGSL.replace(TOP_K_PLACEHOLDER, &config.top_k.to_string());
        let module = KernelModule::new(KernelFormat::Wgsl, "main", source.into_bytes())?;
        let kernel = adapter.compile(&module)?;
        let stream = adapter.create_stream()?;

        Ok(Self {
            adapter,
            logits,
            scratch,
            state,
            kernel,
            stream,
            vocab_size,
            top_k: config.top_k,
            draws: 0,
            resident_bytes: logits_bytes + scratch_bytes + state_bytes,
            initial_state: state_word,
            increment,
        })
    }

    #[must_use]
    pub const fn telemetry(&self) -> WgpuDeterministicSamplerTelemetry {
        WgpuDeterministicSamplerTelemetry {
            vocab_size: self.vocab_size,
            draws: self.draws,
            resident_bytes: self.resident_bytes,
            upload_bytes_per_sample: self.vocab_size * F32_BYTES,
            download_bytes_per_sample: U32_BYTES,
        }
    }

    #[must_use]
    pub const fn top_k(&self) -> usize {
        self.top_k
    }

    #[must_use]
    pub const fn ranking_lanes_per_sample(&self) -> usize {
        PARALLEL_TOP_K_LANES
    }

    #[must_use]
    pub const fn ranking_passes_per_sample(&self) -> usize {
        self.top_k
    }

    pub fn sample(&mut self, logits: &[f32]) -> Result<usize, WgpuDeterministicSamplerError> {
        if logits.len() != self.vocab_size
        {
            return Err(WgpuDeterministicSamplerError::LogitLength {
                expected: self.vocab_size,
                actual: logits.len(),
            });
        }
        if let Some((index, _)) = logits
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(WgpuDeterministicSamplerError::NonFiniteLogit { index });
        }

        self.adapter
            .write(&self.logits, 0, bytemuck::cast_slice(logits))?;
        let event = self.launch()?;
        self.adapter.wait(&event)?;

        let mut output_id = [0u32; 1];
        self.adapter.read(
            &self.state,
            8 * U32_BYTES,
            bytemuck::cast_slice_mut(&mut output_id),
        )?;
        self.draws = self.draws.saturating_add(1);
        Ok(output_id[0] as usize)
    }

    pub fn reset(&mut self) -> Result<(), WgpuDeterministicSamplerError> {
        let words = [
            self.initial_state as u32,
            (self.initial_state >> 32) as u32,
            self.increment as u32,
            (self.increment >> 32) as u32,
            0,
            0,
            1,
        ];
        self.adapter
            .write(&self.state, 4 * U32_BYTES, bytemuck::cast_slice(&words))?;
        self.draws = 0;
        Ok(())
    }

    fn launch(&self) -> Result<WgpuComputeEvent, WgpuDeterministicSamplerError> {
        let bindings = [
            binding(0, &self.logits, BufferAccess::ReadOnly),
            binding(1, &self.scratch, BufferAccess::ReadWrite),
            binding(2, &self.state, BufferAccess::ReadWrite),
        ];
        // The shader encodes @workgroup_size(64); one dispatch group therefore
        // launches exactly one 64-lane workgroup.
        let config = LaunchConfig::new([1, 1, 1], [1, 1, 1], 0)?;
        Ok(self
            .adapter
            .launch(&self.kernel, &self.stream, config, &bindings)?)
    }
}

fn validate_parallel_config(
    vocab_size: usize,
    config: &SamplingConfig,
) -> Result<(), WgpuDeterministicSamplerError> {
    if vocab_size == 0
    {
        return Err(WgpuDeterministicSamplerError::InvalidConfig(
            "sampler vocab size must be non-zero",
        ));
    }
    if vocab_size >= MAX_EXACT_F32_INDEX
    {
        return Err(WgpuDeterministicSamplerError::InvalidConfig(
            "sampler vocab size must be below 2^24 for exact scratch index encoding",
        ));
    }
    if !config.temperature.is_finite() || config.temperature <= 0.0
    {
        return Err(WgpuDeterministicSamplerError::InvalidConfig(
            "parallel top-k sampler requires a finite positive temperature",
        ));
    }
    if !config.top_p.is_finite() || !(0.0..=1.0).contains(&config.top_p)
    {
        return Err(WgpuDeterministicSamplerError::InvalidConfig(
            "sampler top_p must be finite and in 0..=1",
        ));
    }
    if config.top_k < 2 || config.top_k >= vocab_size || config.top_k > PARALLEL_TOP_K_MAX
    {
        return Err(WgpuDeterministicSamplerError::InvalidConfig(
            "parallel top-k sampler requires 2 <= top_k < vocab_size and top_k <= 256",
        ));
    }
    usize_to_u32(vocab_size, "sampler vocab size exceeds WGPU u32 range")?;
    usize_to_u32(config.top_k, "sampler top_k exceeds WGPU u32 range")?;
    Ok(())
}

fn pcg_seed_state(seed: u64) -> (u64, u64) {
    let increment = (seed << 1) | 1;
    let mut state = 0u64;
    state = state.wrapping_mul(PCG_MULTIPLIER).wrapping_add(increment);
    state = state.wrapping_add(seed);
    state = state.wrapping_mul(PCG_MULTIPLIER).wrapping_add(increment);
    (state, increment)
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

fn bytes_for_f32(elements: usize) -> Result<usize, WgpuDeterministicSamplerError> {
    elements
        .checked_mul(F32_BYTES)
        .ok_or(WgpuDeterministicSamplerError::InvalidConfig(
            "sampler f32 buffer size overflows usize",
        ))
}

fn usize_to_u32(
    value: usize,
    message: &'static str,
) -> Result<u32, WgpuDeterministicSamplerError> {
    u32::try_from(value).map_err(|_| WgpuDeterministicSamplerError::InvalidConfig(message))
}
