//! Seeded sampled MiniLLM inference without host-visible logits.
//!
//! This wrapper reuses the resident MiniLLM and deterministic WGPU samplers on
//! the same context. Phase 26 promotes the exact 64-lane bounded-top-k sampler
//! for eligible configurations while retaining the historical sequential
//! sampler as the deterministic fallback.

mod device_feedback;
pub use device_feedback::{
    WgpuResidentDeviceFeedbackMiniLlm, WgpuResidentDeviceFeedbackMiniLlmError,
    WgpuResidentDeviceFeedbackMiniLlmTelemetry,
};

use core::fmt;

use super::{U32_BYTES, WgpuResidentMiniLlm, WgpuResidentMiniLlmError};
use crate::{
    PARALLEL_TOP_K_LANES, PARALLEL_TOP_K_MAX, WgpuComputeBuffer, WgpuComputeEvent,
    WgpuComputeKernel, WgpuDeterministicSampler, WgpuDeterministicSamplerError,
    WgpuDeterministicSamplerTelemetry, WgpuLatentLayerBasis, WgpuParallelTopKSampler,
};
use scirust_compute::{
    BufferAccess, BufferBinding, ComputeBackend, ComputeError, KernelFormat, KernelModule,
    LaunchConfig,
};
use scirust_core::nn::sampling::SamplingConfig;
use scirust_core::nn::transformer::mini_llm::MiniLlmInferenceSnapshot;

const LOGITS_WGSL: &str = r#"
struct MiniState {
    token_id: u32,
    position: u32,
    output_id: u32,
    d_model: u32,
    vocab_size: u32,
    max_seq_len: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> mini: MiniState;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> encoder_io: array<f32>;
@group(0) @binding(3) var<storage, read_write> logits: array<f32>;

@compute @workgroup_size(1)
fn main() {
    if (mini._pad0 == 2u) {
        return;
    }

    let d_model = mini.d_model;
    let vocab_size = mini.vocab_size;
    let embedding_elements = vocab_size * d_model;
    let gamma_offset = embedding_elements;
    let beta_offset = gamma_offset + d_model;
    let head_weight_offset = beta_offset + d_model;
    let head_bias_offset = head_weight_offset + d_model * vocab_size;

    var mean = 0.0;
    for (var column: u32 = 0u; column < d_model; column = column + 1u) {
        mean = mean + encoder_io[column];
    }
    mean = mean / f32(d_model);

    var variance = 0.0;
    for (var column: u32 = 0u; column < d_model; column = column + 1u) {
        let delta = encoder_io[column] - mean;
        variance = variance + delta * delta;
    }
    variance = variance / f32(d_model);
    let inv_std = inverseSqrt(variance + 0.00001);

    for (var column: u32 = 0u; column < d_model; column = column + 1u) {
        encoder_io[column] =
            (encoder_io[column] - mean) * inv_std * weights[gamma_offset + column]
            + weights[beta_offset + column];
    }

    for (var token: u32 = 0u; token < vocab_size; token = token + 1u) {
        var logit = weights[head_bias_offset + token];
        for (var column: u32 = 0u; column < d_model; column = column + 1u) {
            logit = logit
                + encoder_io[column]
                    * weights[head_weight_offset + column * vocab_size + token];
        }
        logits[token] = logit;
    }

    mini.position = mini.position + 1u;
}
"#;

#[derive(Debug)]
pub enum WgpuResidentSampledMiniLlmError {
    MiniLlm(WgpuResidentMiniLlmError),
    Sampling(WgpuDeterministicSamplerError),
    NoPendingLogits,
    Compute(ComputeError),
}

impl fmt::Display for WgpuResidentSampledMiniLlmError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::MiniLlm(error) => write!(output, "{error}"),
            Self::Sampling(error) => write!(output, "{error}"),
            Self::NoPendingLogits => write!(
                output,
                "resident sampled MiniLLM has no unsampled logits; ingest a token first"
            ),
            Self::Compute(error) => write!(output, "{error}"),
        }
    }
}

impl std::error::Error for WgpuResidentSampledMiniLlmError {}

impl From<WgpuResidentMiniLlmError> for WgpuResidentSampledMiniLlmError {
    fn from(error: WgpuResidentMiniLlmError) -> Self {
        Self::MiniLlm(error)
    }
}

impl From<super::super::WgpuResidentTransformerEncoderError> for WgpuResidentSampledMiniLlmError {
    fn from(error: super::super::WgpuResidentTransformerEncoderError) -> Self {
        Self::MiniLlm(error.into())
    }
}

impl From<WgpuDeterministicSamplerError> for WgpuResidentSampledMiniLlmError {
    fn from(error: WgpuDeterministicSamplerError) -> Self {
        Self::Sampling(error)
    }
}

impl From<ComputeError> for WgpuResidentSampledMiniLlmError {
    fn from(error: ComputeError) -> Self {
        Self::Compute(error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuResidentSampledMiniLlmTelemetry {
    pub capacity_tokens: usize,
    pub resident_tokens: usize,
    pub ingested_tokens: usize,
    pub sampling_draws: usize,
    pub d_model: usize,
    pub vocab_size: usize,
    pub rank: usize,
    pub resident_bytes: usize,
    pub upload_bytes_per_ingest: usize,
    pub download_bytes_per_sample: usize,
    pub sample_ready: bool,
}

enum ResidentSampler {
    Sequential(WgpuDeterministicSampler),
    Parallel(WgpuParallelTopKSampler),
}

impl ResidentSampler {
    fn select(
        context: crate::WgpuContext,
        vocab_size: usize,
        sampling: SamplingConfig,
        seed: u64,
        max_workgroup_size_x: u32,
    ) -> Result<Self, WgpuDeterministicSamplerError> {
        let parallel_eligible = sampling.temperature > 0.0
            && sampling.top_k >= 2
            && sampling.top_k < vocab_size
            && sampling.top_k <= PARALLEL_TOP_K_MAX
            && max_workgroup_size_x >= PARALLEL_TOP_K_LANES as u32;

        if parallel_eligible
        {
            Ok(Self::Parallel(WgpuParallelTopKSampler::from_context(
                context, vocab_size, sampling, seed,
            )?))
        }
        else
        {
            Ok(Self::Sequential(WgpuDeterministicSampler::from_context(
                context, vocab_size, sampling, seed,
            )?))
        }
    }

    fn telemetry(&self) -> WgpuDeterministicSamplerTelemetry {
        match self
        {
            Self::Sequential(sampler) => sampler.telemetry(),
            Self::Parallel(sampler) => sampler.telemetry(),
        }
    }

    fn is_parallel(&self) -> bool {
        matches!(self, Self::Parallel(_))
    }

    fn logits_buffer(&self) -> &WgpuComputeBuffer {
        match self
        {
            Self::Sequential(sampler) => sampler.logits_buffer(),
            Self::Parallel(sampler) => sampler.logits_buffer(),
        }
    }

    fn state_buffer(&self) -> &WgpuComputeBuffer {
        match self
        {
            Self::Sequential(sampler) => sampler.state_buffer(),
            Self::Parallel(sampler) => sampler.state_buffer(),
        }
    }

    fn launch_resident_without_readback(&self) -> Result<(), WgpuDeterministicSamplerError> {
        match self
        {
            Self::Sequential(sampler) => sampler.launch_resident_without_readback(),
            Self::Parallel(sampler) => sampler.launch_resident_without_readback(),
        }
    }

    fn set_enabled(&self, enabled: bool) -> Result<(), WgpuDeterministicSamplerError> {
        match self
        {
            Self::Sequential(sampler) => sampler.set_enabled(enabled),
            Self::Parallel(sampler) => sampler.set_enabled(enabled),
        }
    }

    fn sync_host_draws(&mut self, draws: usize) {
        match self
        {
            Self::Sequential(sampler) => sampler.sync_host_draws(draws),
            Self::Parallel(sampler) => sampler.sync_host_draws(draws),
        }
    }

    fn sample_resident(&mut self) -> Result<usize, WgpuDeterministicSamplerError> {
        match self
        {
            Self::Sequential(sampler) => sampler.sample_resident(),
            Self::Parallel(sampler) => sampler.sample_resident(),
        }
    }

    fn reset(&mut self) -> Result<(), WgpuDeterministicSamplerError> {
        match self
        {
            Self::Sequential(sampler) => sampler.reset(),
            Self::Parallel(sampler) => sampler.reset(),
        }
    }
}

/// Seeded sampled MiniLLM runtime with logits and RNG state resident on WGPU.
pub struct WgpuResidentSampledMiniLlm {
    inner: WgpuResidentMiniLlm,
    sampler: ResidentSampler,
    logits_kernel: WgpuComputeKernel,
    sample_ready: bool,
    resident_bytes: usize,
}

impl WgpuResidentSampledMiniLlm {
    pub fn new(
        snapshot: MiniLlmInferenceSnapshot<'_>,
        capacity: usize,
        rank: usize,
        layers: &[WgpuLatentLayerBasis<'_>],
        sampling: SamplingConfig,
        seed: u64,
    ) -> Result<Self, WgpuResidentSampledMiniLlmError> {
        let vocab_size = snapshot.config.vocab_size;
        let inner = WgpuResidentMiniLlm::new(snapshot, capacity, rank, layers)?;
        let max_workgroup_size_x = inner.encoder.adapter.capabilities().max_workgroup_size[0];
        let context = inner.encoder.adapter.context().clone();
        let sampler =
            ResidentSampler::select(context, vocab_size, sampling, seed, max_workgroup_size_x)?;
        let module =
            KernelModule::new(KernelFormat::Wgsl, "main", LOGITS_WGSL.as_bytes().to_vec())?;
        let logits_kernel = inner.encoder.adapter.compile(&module)?;
        let resident_bytes = inner
            .resident_bytes
            .checked_add(sampler.telemetry().resident_bytes)
            .ok_or(WgpuResidentMiniLlmError::InvalidConfig(
                "sampled MiniLLM resident byte count overflows usize",
            ))?;

        Ok(Self {
            inner,
            sampler,
            logits_kernel,
            sample_ready: false,
            resident_bytes,
        })
    }

    #[must_use]
    pub fn telemetry(&self) -> WgpuResidentSampledMiniLlmTelemetry {
        let mini = self.inner.telemetry();
        let sampling = self.sampler.telemetry();
        WgpuResidentSampledMiniLlmTelemetry {
            capacity_tokens: mini.capacity_tokens,
            resident_tokens: mini.resident_tokens,
            ingested_tokens: mini.steps,
            sampling_draws: sampling.draws,
            d_model: mini.d_model,
            vocab_size: mini.vocab_size,
            rank: mini.rank,
            resident_bytes: self.resident_bytes,
            upload_bytes_per_ingest: U32_BYTES,
            download_bytes_per_sample: U32_BYTES,
            sample_ready: self.sample_ready,
        }
    }

    /// Whether Phase 26 selected the exact 64-lane bounded-top-k sampler.
    #[must_use]
    pub fn uses_parallel_sampler(&self) -> bool {
        self.sampler.is_parallel()
    }

    pub fn ingest_at(
        &mut self,
        token_id: usize,
        pos: usize,
    ) -> Result<(), WgpuResidentSampledMiniLlmError> {
        self.validate_input(token_id, pos)?;

        let token = [u32::try_from(token_id).map_err(|_| {
            WgpuResidentMiniLlmError::InvalidConfig("MiniLLM token id exceeds WGPU u32 range")
        })?];
        self.inner
            .encoder
            .adapter
            .write(&self.inner.state, 0, bytemuck::cast_slice(&token))?;

        let preprocess = self.inner.launch_preprocess()?;
        self.inner.encoder.adapter.wait(&preprocess)?;

        let encoder = self.inner.encoder.launch()?;
        self.inner.encoder.adapter.wait(&encoder)?;

        let logits = self.launch_logits()?;
        self.inner.encoder.adapter.wait(&logits)?;

        self.inner.encoder.steps = self.inner.encoder.steps.saturating_add(1);
        self.inner.encoder.resident_tokens = self
            .inner
            .encoder
            .resident_tokens
            .saturating_add(1)
            .min(self.inner.encoder.capacity);
        self.inner.encoder.next_slot =
            (self.inner.encoder.next_slot + 1) % self.inner.encoder.capacity;
        self.sample_ready = true;
        Ok(())
    }

    pub fn sample_next(&mut self) -> Result<usize, WgpuResidentSampledMiniLlmError> {
        if !self.sample_ready
        {
            return Err(WgpuResidentSampledMiniLlmError::NoPendingLogits);
        }
        let token = self.sampler.sample_resident()?;
        self.sample_ready = false;
        Ok(token)
    }

    pub fn step_sample_at(
        &mut self,
        token_id: usize,
        pos: usize,
    ) -> Result<usize, WgpuResidentSampledMiniLlmError> {
        self.ingest_at(token_id, pos)?;
        self.sample_next()
    }

    pub fn step_sample(
        &mut self,
        token_id: usize,
    ) -> Result<usize, WgpuResidentSampledMiniLlmError> {
        self.step_sample_at(token_id, self.inner.encoder.steps)
    }

    pub fn reload_weights(
        &mut self,
        snapshot: MiniLlmInferenceSnapshot<'_>,
    ) -> Result<(), WgpuResidentSampledMiniLlmError> {
        self.inner.reload_weights(snapshot)?;
        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), WgpuResidentSampledMiniLlmError> {
        self.inner.reset()?;
        self.sampler.reset()?;
        self.sample_ready = false;
        Ok(())
    }

    fn validate_input(
        &self,
        token_id: usize,
        pos: usize,
    ) -> Result<(), WgpuResidentSampledMiniLlmError> {
        if pos != self.inner.encoder.steps
        {
            return Err(WgpuResidentMiniLlmError::PositionMismatch {
                expected: self.inner.encoder.steps,
                actual: pos,
            }
            .into());
        }
        if pos >= self.inner.max_seq_len
        {
            return Err(WgpuResidentMiniLlmError::InvalidConfig(
                "MiniLLM position exceeds max_seq_len",
            )
            .into());
        }
        if token_id >= self.inner.vocab_size
        {
            return Err(WgpuResidentMiniLlmError::TokenOutOfRange {
                token_id,
                vocab_size: self.inner.vocab_size,
            }
            .into());
        }
        Ok(())
    }

    fn launch_logits(&self) -> Result<WgpuComputeEvent, WgpuResidentSampledMiniLlmError> {
        let bindings = [
            binding(0, &self.inner.state, BufferAccess::ReadWrite),
            binding(1, &self.inner.weights, BufferAccess::ReadOnly),
            binding(2, &self.inner.encoder.io, BufferAccess::ReadWrite),
            binding(3, self.sampler.logits_buffer(), BufferAccess::ReadWrite),
        ];
        let config = LaunchConfig::new([1, 1, 1], [1, 1, 1], 0)?;
        Ok(self.inner.encoder.adapter.launch(
            &self.logits_kernel,
            &self.inner.encoder.stream,
            config,
            &bindings,
        )?)
    }
}

fn binding<'a>(
    slot: u32,
    buffer: &'a crate::WgpuComputeBuffer,
    access: BufferAccess,
) -> BufferBinding<'a, crate::WgpuComputeBuffer> {
    BufferBinding {
        slot,
        buffer,
        offset_bytes: 0,
        length_bytes: buffer.len(),
        access,
    }
}
