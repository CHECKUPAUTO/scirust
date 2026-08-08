//! Phase 21 device-feedback generation for the resident sampled MiniLLM.
//!
//! Generated token ids remain on WGPU between autoregressive steps. The host
//! submits a fixed burst of dispatches, then reads one compact token buffer.
//! EOS disables the sampler and encoder on device so later scheduled dispatches
//! are no-ops and cannot advance PCG or latent-KV state.

use core::fmt;

use super::{WgpuResidentSampledMiniLlm, WgpuResidentSampledMiniLlmError};
use crate::{
    WgpuComputeBuffer, WgpuComputeEvent, WgpuComputeKernel, WgpuLatentLayerBasis,
};
use scirust_compute::{
    BufferAccess, BufferBinding, ComputeBackend, ComputeError, KernelFormat, KernelModule,
    LaunchConfig, MemorySpace,
};
use scirust_core::nn::sampling::SamplingConfig;
use scirust_core::nn::transformer::mini_llm::MiniLlmInferenceSnapshot;

const U32_BYTES: usize = core::mem::size_of::<u32>();
const FEEDBACK_HEADER_WORDS: usize = 4;
const MINI_PHASE_MODE_OFFSET_WORDS: usize = 6;

const DEVICE_FEEDBACK_WGSL: &str = r#"
struct MiniState {
    token_id: u32,
    position: u32,
    output_id: u32,
    d_model: u32,
    vocab_size: u32,
    max_seq_len: u32,
    phase_mode: u32,
    _pad1: u32,
};

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

struct EncoderState {
    d_model: u32,
    n_layers: u32,
    n_heads: u32,
    d_head: u32,
    rank: u32,
    capacity: u32,
    d_ff: u32,
    len: u32,
    next_slot: u32,
    enabled: u32,
    data: array<f32>,
};

struct FeedbackState {
    generated_count: u32,
    active: u32,
    sampler_draws: u32,
    max_tokens: u32,
    tokens: array<u32>,
};

@group(0) @binding(0) var<storage, read_write> mini: MiniState;
@group(0) @binding(1) var<storage, read_write> sampling_state: SamplerState;
@group(0) @binding(2) var<storage, read_write> feedback: FeedbackState;
@group(0) @binding(3) var<storage, read_write> encoder: EncoderState;

@compute @workgroup_size(1)
fn main() {
    if (feedback.active == 0u) {
        return;
    }

    let index = feedback.generated_count;
    if (index >= feedback.max_tokens) {
        feedback.active = 0u;
        return;
    }

    let token = sampling_state.output_id;
    feedback.tokens[index] = token;
    feedback.generated_count = index + 1u;
    feedback.sampler_draws = sampling_state.draws;
    mini.token_id = token;

    if (token == 0u) {
        feedback.active = 0u;
        sampling_state.enabled = 0u;
        encoder.enabled = 0u;
        mini.phase_mode = 2u;
    }
}
"#;

#[derive(Debug)]
pub enum WgpuResidentDeviceFeedbackMiniLlmError {
    Sampled(WgpuResidentSampledMiniLlmError),
    InvalidConfig(&'static str),
    CorruptDeviceCount { count: usize, limit: usize },
    Compute(ComputeError),
}

impl fmt::Display for WgpuResidentDeviceFeedbackMiniLlmError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sampled(error) => write!(output, "{error}"),
            Self::InvalidConfig(message) => write!(output, "{message}"),
            Self::CorruptDeviceCount { count, limit } => write!(
                output,
                "device-feedback generated-count {count} exceeds burst limit {limit}"
            ),
            Self::Compute(error) => write!(output, "{error}"),
        }
    }
}

impl std::error::Error for WgpuResidentDeviceFeedbackMiniLlmError {}

impl From<WgpuResidentSampledMiniLlmError> for WgpuResidentDeviceFeedbackMiniLlmError {
    fn from(error: WgpuResidentSampledMiniLlmError) -> Self {
        Self::Sampled(error)
    }
}

impl From<ComputeError> for WgpuResidentDeviceFeedbackMiniLlmError {
    fn from(error: ComputeError) -> Self {
        Self::Compute(error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuResidentDeviceFeedbackMiniLlmTelemetry {
    pub capacity_tokens: usize,
    pub resident_tokens: usize,
    pub ingested_tokens: usize,
    pub sampling_draws: usize,
    pub d_model: usize,
    pub vocab_size: usize,
    pub rank: usize,
    pub resident_bytes: usize,
    pub prompt_upload_bytes_per_token: usize,
    pub generated_upload_bytes_per_token: usize,
    pub generated_download_bytes_per_token: usize,
    pub last_burst_readback_bytes: usize,
}

/// Resident sampled MiniLLM with generated-token feedback entirely on WGPU.
pub struct WgpuResidentDeviceFeedbackMiniLlm {
    inner: WgpuResidentSampledMiniLlm,
    feedback: WgpuComputeBuffer,
    feedback_kernel: WgpuComputeKernel,
    feedback_capacity: usize,
    resident_bytes: usize,
    last_burst_readback_bytes: usize,
}

impl WgpuResidentDeviceFeedbackMiniLlm {
    pub fn new(
        snapshot: MiniLlmInferenceSnapshot<'_>,
        capacity: usize,
        rank: usize,
        layers: &[WgpuLatentLayerBasis<'_>],
        sampling: SamplingConfig,
        seed: u64,
    ) -> Result<Self, WgpuResidentDeviceFeedbackMiniLlmError> {
        let feedback_capacity = snapshot.config.max_seq_len;
        let inner = WgpuResidentSampledMiniLlm::new(
            snapshot, capacity, rank, layers, sampling, seed,
        )?;
        let words = FEEDBACK_HEADER_WORDS
            .checked_add(feedback_capacity)
            .ok_or(Self::invalid("device-feedback buffer size overflows usize"))?;
        let feedback_bytes = words
            .checked_mul(U32_BYTES)
            .ok_or(Self::invalid("device-feedback byte size overflows usize"))?;
        let feedback = inner
            .inner
            .encoder
            .adapter
            .allocate(feedback_bytes, 4, MemorySpace::Device)?;
        let module = KernelModule::new(
            KernelFormat::Wgsl,
            "main",
            DEVICE_FEEDBACK_WGSL.as_bytes().to_vec(),
        )?;
        let feedback_kernel = inner.inner.encoder.adapter.compile(&module)?;
        let resident_bytes = inner
            .resident_bytes
            .checked_add(feedback_bytes)
            .ok_or(Self::invalid("device-feedback resident bytes overflow usize"))?;

        Ok(Self {
            inner,
            feedback,
            feedback_kernel,
            feedback_capacity,
            resident_bytes,
            last_burst_readback_bytes: 0,
        })
    }

    #[must_use]
    pub fn telemetry(&self) -> WgpuResidentDeviceFeedbackMiniLlmTelemetry {
        let sampled = self.inner.telemetry();
        WgpuResidentDeviceFeedbackMiniLlmTelemetry {
            capacity_tokens: sampled.capacity_tokens,
            resident_tokens: sampled.resident_tokens,
            ingested_tokens: sampled.ingested_tokens,
            sampling_draws: sampled.sampling_draws,
            d_model: sampled.d_model,
            vocab_size: sampled.vocab_size,
            rank: sampled.rank,
            resident_bytes: self.resident_bytes,
            prompt_upload_bytes_per_token: U32_BYTES,
            generated_upload_bytes_per_token: 0,
            generated_download_bytes_per_token: 0,
            last_burst_readback_bytes: self.last_burst_readback_bytes,
        }
    }

    /// Generate a sampled suffix while keeping every generated-token feedback
    /// dependency on WGPU. The host primes the prompt, submits a fixed dispatch
    /// burst, then performs one bounded readback for the generated suffix.
    pub fn generate_ids_resident(
        &mut self,
        prompt_ids: &[usize],
        max_tokens: usize,
    ) -> Result<Vec<usize>, WgpuResidentDeviceFeedbackMiniLlmError> {
        self.inner.reset()?;
        self.last_burst_readback_bytes = 0;

        if prompt_ids.is_empty() {
            return Ok(Vec::new());
        }
        if prompt_ids.len() > self.inner.inner.max_seq_len {
            return Err(Self::invalid("device-feedback prompt exceeds max_seq_len"));
        }
        for &token_id in prompt_ids {
            if token_id >= self.inner.inner.vocab_size {
                return Err(WgpuResidentSampledMiniLlmError::MiniLlm(
                    super::super::WgpuResidentMiniLlmError::TokenOutOfRange {
                        token_id,
                        vocab_size: self.inner.inner.vocab_size,
                    },
                )
                .into());
            }
        }

        for (pos, &token_id) in prompt_ids.iter().enumerate() {
            self.inner.ingest_at(token_id, pos)?;
        }

        let limit = max_tokens.min(
            self.inner
                .inner
                .max_seq_len
                .saturating_sub(prompt_ids.len()),
        );
        if limit == 0 {
            return Ok(prompt_ids.to_vec());
        }
        if limit > self.feedback_capacity {
            return Err(Self::invalid("device-feedback burst exceeds resident capacity"));
        }

        self.set_phase_mode(1)?;
        self.inner.sampler.set_enabled(true)?;
        self.inner.inner.encoder.set_enabled(true)?;
        let limit_u32 = u32::try_from(limit)
            .map_err(|_| Self::invalid("device-feedback limit exceeds WGPU u32 range"))?;
        let header = [0u32, 1u32, 0u32, limit_u32];
        self.inner.inner.encoder.adapter.write(
            &self.feedback,
            0,
            bytemuck::cast_slice(&header),
        )?;

        let burst_result = self.run_burst(limit);
        let restore_result = self.restore_enabled_state();
        if let Err(error) = burst_result {
            restore_result?;
            return Err(error);
        }
        restore_result?;

        let words_to_read = FEEDBACK_HEADER_WORDS
            .checked_add(limit)
            .ok_or(Self::invalid("device-feedback readback size overflows usize"))?;
        let mut words = vec![0u32; words_to_read];
        self.inner.inner.encoder.adapter.read(
            &self.feedback,
            0,
            bytemuck::cast_slice_mut(&mut words),
        )?;
        self.last_burst_readback_bytes = words_to_read * U32_BYTES;

        let generated_count = words[0] as usize;
        if generated_count > limit {
            return Err(WgpuResidentDeviceFeedbackMiniLlmError::CorruptDeviceCount {
                count: generated_count,
                limit,
            });
        }
        let sampler_draws = words[2] as usize;
        self.inner.sampler.sync_host_draws(sampler_draws);

        // The final sampled token is intentionally not ingested: exactly like
        // the CPU cached generator, logits for a token are only computed when a
        // later token is requested. EOS is likewise never ingested.
        let generated_ingests = generated_count.saturating_sub(1);
        let total_ingested = prompt_ids.len().saturating_add(generated_ingests);
        self.inner.inner.encoder.steps = total_ingested;
        self.inner.inner.encoder.resident_tokens = total_ingested.min(self.inner.inner.encoder.capacity);
        self.inner.inner.encoder.next_slot = total_ingested % self.inner.inner.encoder.capacity;
        self.inner.sample_ready = false;

        let mut ids = prompt_ids.to_vec();
        ids.extend(
            words[FEEDBACK_HEADER_WORDS..FEEDBACK_HEADER_WORDS + generated_count]
                .iter()
                .map(|&token| token as usize),
        );
        Ok(ids)
    }

    pub fn reset(&mut self) -> Result<(), WgpuResidentDeviceFeedbackMiniLlmError> {
        self.inner.reset()?;
        self.last_burst_readback_bytes = 0;
        self.restore_enabled_state()?;
        Ok(())
    }

    fn run_burst(
        &mut self,
        limit: usize,
    ) -> Result<(), WgpuResidentDeviceFeedbackMiniLlmError> {
        for generated_index in 0..limit {
            self.inner.sampler.launch_resident_without_readback()?;
            let feedback = self.launch_feedback()?;
            self.inner.inner.encoder.adapter.wait(&feedback)?;

            if generated_index + 1 < limit {
                let preprocess = self.inner.inner.launch_preprocess()?;
                self.inner.inner.encoder.adapter.wait(&preprocess)?;

                let encoder = self.inner.inner.encoder.launch()?;
                self.inner.inner.encoder.adapter.wait(&encoder)?;

                let logits = self.inner.launch_logits()?;
                self.inner.inner.encoder.adapter.wait(&logits)?;
            }
        }
        Ok(())
    }

    fn launch_feedback(
        &self,
    ) -> Result<WgpuComputeEvent, WgpuResidentDeviceFeedbackMiniLlmError> {
        let bindings = [
            binding(0, &self.inner.inner.state, BufferAccess::ReadWrite),
            binding(1, self.inner.sampler.state_buffer(), BufferAccess::ReadWrite),
            binding(2, &self.feedback, BufferAccess::ReadWrite),
            binding(
                3,
                &self.inner.inner.encoder.state,
                BufferAccess::ReadWrite,
            ),
        ];
        let config = LaunchConfig::new([1, 1, 1], [1, 1, 1], 0)?;
        Ok(self.inner.inner.encoder.adapter.launch(
            &self.feedback_kernel,
            &self.inner.inner.encoder.stream,
            config,
            &bindings,
        )?)
    }

    fn set_phase_mode(
        &self,
        mode: u32,
    ) -> Result<(), WgpuResidentDeviceFeedbackMiniLlmError> {
        let word = [mode];
        self.inner.inner.encoder.adapter.write(
            &self.inner.inner.state,
            MINI_PHASE_MODE_OFFSET_WORDS * U32_BYTES,
            bytemuck::cast_slice(&word),
        )?;
        Ok(())
    }

    fn restore_enabled_state(&self) -> Result<(), WgpuResidentDeviceFeedbackMiniLlmError> {
        self.set_phase_mode(0)?;
        self.inner.sampler.set_enabled(true)?;
        self.inner.inner.encoder.set_enabled(true)?;
        Ok(())
    }

    fn invalid(message: &'static str) -> WgpuResidentDeviceFeedbackMiniLlmError {
        WgpuResidentDeviceFeedbackMiniLlmError::InvalidConfig(message)
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
