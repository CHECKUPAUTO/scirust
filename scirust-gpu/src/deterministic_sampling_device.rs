//! Deterministic WGPU token sampling compatible with SciRust's seeded CPU sampler.
//!
//! WGSL has no portable native `u64`, while [`PcgEngine`] uses a 64-bit PCG
//! state. This module emulates the state transition with two `u32` limbs and a
//! fixed-order 32x32->64 multiply, then applies the same temperature, top-k,
//! top-p and categorical-draw semantics as `scirust_core::nn::sampling`.
//!
//! The first kernel deliberately uses one invocation and selection-sort style
//! ranking. It is a reproducibility baseline, not a throughput claim.

use core::fmt;

use crate::{
    WgpuComputeAdapter, WgpuComputeBuffer, WgpuComputeEvent, WgpuComputeKernel, WgpuComputeStream,
    WgpuContext,
};
use scirust_compute::{
    BufferAccess, BufferBinding, ComputeBackend, ComputeError, KernelFormat, KernelModule,
    LaunchConfig, MemorySpace,
};
use scirust_core::nn::sampling::SamplingConfig;

const STATE_WORDS: usize = 10;
const F32_BYTES: usize = core::mem::size_of::<f32>();
const U32_BYTES: usize = core::mem::size_of::<u32>();
const MAX_EXACT_F32_INDEX: usize = 1 << 24;
const PCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;

const DETERMINISTIC_SAMPLER_WGSL: &str = r#"
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
};

struct WideU32 {
    lo: u32,
    hi: u32,
};

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(2) var<storage, read_write> sampler: SamplerState;

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
    let old_lo = sampler.state_lo;
    let old_hi = sampler.state_hi;

    // 0x5851f42d4c957f2d, reduced modulo 2^64. For a two-limb
    // multiplication only the low words of the two cross products contribute
    // to the high output limb.
    let product_lo = mul_u32_wide(old_lo, 0x4c957f2du);
    let cross_lo_hi = old_lo * 0x5851f42du;
    let cross_hi_lo = old_hi * 0x4c957f2du;

    let added_lo = product_lo.lo + sampler.inc_lo;
    let carry = select(0u, 1u, added_lo < product_lo.lo);
    let added_hi = product_lo.hi
        + cross_lo_hi
        + cross_hi_lo
        + sampler.inc_hi
        + carry;
    sampler.state_lo = added_lo;
    sampler.state_hi = added_hi;

    // xorshifted = (((oldstate >> 18) ^ oldstate) >> 27) as u32
    let shifted18_lo = (old_lo >> 18u) | (old_hi << 14u);
    let shifted18_hi = old_hi >> 18u;
    let xor_lo = shifted18_lo ^ old_lo;
    let xor_hi = shifted18_hi ^ old_hi;
    let xorshifted = (xor_lo >> 27u) | (xor_hi << 5u);
    let rot = old_hi >> 27u;
    return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
}

fn greedy_argmax(vocab_size: u32) -> u32 {
    var best = 0u;
    var best_value = -3.402823466e+38;
    for (var token: u32 = 0u; token < vocab_size; token = token + 1u) {
        let value = logits[token];
        // CPU sampling::argmax uses `>` and therefore keeps the lowest id on
        // exact ties.
        if (value > best_value) {
            best_value = value;
            best = token;
        }
    }
    return best;
}

@compute @workgroup_size(1)
fn main() {
    let vocab_size = sampler.vocab_size;
    let temperature = bitcast<f32>(sampler.temperature_bits);
    let top_p = bitcast<f32>(sampler.top_p_bits);
    let order_offset = vocab_size;

    if (temperature <= 0.0 || sampler.top_k == 1u) {
        sampler.output_id = greedy_argmax(vocab_size);
        return;
    }

    let inv_temperature = 1.0 / temperature;
    var maximum = -3.402823466e+38;
    for (var token: u32 = 0u; token < vocab_size; token = token + 1u) {
        maximum = max(maximum, logits[token]);
    }

    for (var token: u32 = 0u; token < vocab_size; token = token + 1u) {
        scratch[token] = exp((logits[token] - maximum) * inv_temperature);
        scratch[order_offset + token] = f32(token);
    }

    // CPU sorts probability descending and breaks ties by the lower token id.
    // Selection sort is intentionally O(V^2): fixed order and no workgroup
    // races make this a useful oracle-grade baseline.
    for (var rank: u32 = 0u; rank < vocab_size; rank = rank + 1u) {
        var best_pos = rank;
        for (var pos: u32 = rank + 1u; pos < vocab_size; pos = pos + 1u) {
            let best_id = u32(scratch[order_offset + best_pos]);
            let candidate_id = u32(scratch[order_offset + pos]);
            let best_probability = scratch[best_id];
            let candidate_probability = scratch[candidate_id];
            if (candidate_probability > best_probability
                || (candidate_probability == best_probability && candidate_id < best_id)) {
                best_pos = pos;
            }
        }
        if (best_pos != rank) {
            let temporary = scratch[order_offset + rank];
            scratch[order_offset + rank] = scratch[order_offset + best_pos];
            scratch[order_offset + best_pos] = temporary;
        }
    }

    if (sampler.top_k > 0u && sampler.top_k < vocab_size) {
        for (var rank: u32 = sampler.top_k; rank < vocab_size; rank = rank + 1u) {
            let token = u32(scratch[order_offset + rank]);
            scratch[token] = 0.0;
        }
    }

    if (top_p < 1.0) {
        var total = 0.0;
        for (var token: u32 = 0u; token < vocab_size; token = token + 1u) {
            total = total + scratch[token];
        }
        if (total > 0.0) {
            var cumulative = 0.0;
            var cutoff = vocab_size;
            for (var rank: u32 = 0u; rank < vocab_size; rank = rank + 1u) {
                let token = u32(scratch[order_offset + rank]);
                cumulative = cumulative + scratch[token] / total;
                if (cumulative >= top_p) {
                    cutoff = rank + 1u;
                    break;
                }
            }
            for (var rank: u32 = cutoff; rank < vocab_size; rank = rank + 1u) {
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
        sampler.output_id = greedy_argmax(vocab_size);
        return;
    }

    let random_word = pcg_next_u32();
    sampler.draws = sampler.draws + 1u;
    let random_unit = f32(random_word) / 4294967296.0;
    let threshold = random_unit * sum;
    var cumulative = 0.0;
    for (var token: u32 = 0u; token < vocab_size; token = token + 1u) {
        cumulative = cumulative + scratch[token];
        if (threshold < cumulative) {
            sampler.output_id = token;
            return;
        }
    }

    sampler.output_id = u32(scratch[order_offset]);
}
"#;

#[derive(Debug)]
pub enum WgpuDeterministicSamplerError {
    InvalidConfig(&'static str),
    LogitLength {
        expected: usize,
        actual: usize,
    },
    NonFiniteLogit {
        index: usize,
    },
    Compute(ComputeError),
}

impl fmt::Display for WgpuDeterministicSamplerError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidConfig(message) => write!(output, "{message}"),
            Self::LogitLength { expected, actual } => write!(
                output,
                "sampler logits length mismatch: expected {expected}, got {actual}"
            ),
            Self::NonFiniteLogit { index } => {
                write!(output, "sampler logit at index {index} is not finite")
            },
            Self::Compute(error) => write!(output, "{error}"),
        }
    }
}

impl std::error::Error for WgpuDeterministicSamplerError {}

impl From<ComputeError> for WgpuDeterministicSamplerError {
    fn from(error: ComputeError) -> Self {
        Self::Compute(error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuDeterministicSamplerTelemetry {
    pub vocab_size: usize,
    pub draws: usize,
    pub resident_bytes: usize,
    pub upload_bytes_per_sample: usize,
    pub download_bytes_per_sample: usize,
}

/// Persistent one-invocation WGPU sampler with an emulated 64-bit PCG state.
pub struct WgpuDeterministicSampler {
    adapter: WgpuComputeAdapter,
    logits: WgpuComputeBuffer,
    scratch: WgpuComputeBuffer,
    state: WgpuComputeBuffer,
    kernel: WgpuComputeKernel,
    stream: WgpuComputeStream,
    vocab_size: usize,
    draws: usize,
    resident_bytes: usize,
}

impl WgpuDeterministicSampler {
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
        validate_config(vocab_size, &config)?;
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
        let logits = adapter.allocate(logits_bytes, 4, MemorySpace::Device)?;
        let scratch = adapter.allocate(scratch_bytes, 4, MemorySpace::Device)?;
        let state = adapter.allocate(state_bytes, 4, MemorySpace::Device)?;
        adapter.write(&state, 0, bytemuck::cast_slice(&state_words))?;

        let module = KernelModule::new(
            KernelFormat::Wgsl,
            "main",
            DETERMINISTIC_SAMPLER_WGSL.as_bytes().to_vec(),
        )?;
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
            draws: 0,
            resident_bytes: logits_bytes + scratch_bytes + state_bytes,
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

    pub fn sample(
        &mut self,
        logits: &[f32],
    ) -> Result<usize, WgpuDeterministicSamplerError> {
        if logits.len() != self.vocab_size
        {
            return Err(WgpuDeterministicSamplerError::LogitLength {
                expected: self.vocab_size,
                actual: logits.len(),
            });
        }
        if let Some((index, _)) = logits.iter().enumerate().find(|(_, value)| !value.is_finite())
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
        if self.config_consumes_rng()?
        {
            self.draws = self.draws.saturating_add(1);
        }
        Ok(output_id[0] as usize)
    }

    fn config_consumes_rng(&self) -> Result<bool, WgpuDeterministicSamplerError> {
        let mut words = [0u32; 3];
        self.adapter.read(
            &self.state,
            U32_BYTES,
            bytemuck::cast_slice_mut(&mut words),
        )?;
        let temperature = f32::from_bits(words[0]);
        let top_k = words[1];
        Ok(temperature > 0.0 && top_k != 1)
    }

    fn launch(&self) -> Result<WgpuComputeEvent, WgpuDeterministicSamplerError> {
        let bindings = [
            binding(0, &self.logits, BufferAccess::ReadOnly),
            binding(1, &self.scratch, BufferAccess::ReadWrite),
            binding(2, &self.state, BufferAccess::ReadWrite),
        ];
        let config = LaunchConfig::new([1, 1, 1], [1, 1, 1], 0)?;
        Ok(self
            .adapter
            .launch(&self.kernel, &self.stream, config, &bindings)?)
    }
}

fn validate_config(
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
    if !config.temperature.is_finite()
    {
        return Err(WgpuDeterministicSamplerError::InvalidConfig(
            "sampler temperature must be finite",
        ));
    }
    if !config.top_p.is_finite() || !(0.0..=1.0).contains(&config.top_p)
    {
        return Err(WgpuDeterministicSamplerError::InvalidConfig(
            "sampler top_p must be finite and in 0..=1",
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
