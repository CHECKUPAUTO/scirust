//! Device-resident greedy MiniLLM step built on the Phase 17 resident encoder.
//!
//! The host writes one token id. WGPU performs embedding lookup, sinusoidal
//! positional encoding, the complete resident encoder, MiniLLM's final
//! LayerNorm, LM-head projection and greedy argmax. The host then reads one
//! token id. Hidden states and logits never cross the host boundary.

use core::fmt;

use super::{
    WgpuLatentLayerBasis, WgpuResidentTransformerEncoder, WgpuResidentTransformerEncoderError,
    binding,
};
use crate::{WgpuComputeBuffer, WgpuComputeEvent, WgpuComputeKernel};
use scirust_compute::{
    BufferAccess, ComputeBackend, ComputeError, KernelFormat, KernelModule, LaunchConfig,
    MemorySpace,
};
use scirust_core::nn::transformer::mini_llm::MiniLlmInferenceSnapshot;

const U32_BYTES: usize = core::mem::size_of::<u32>();
const MINI_STATE_WORDS: usize = 8;

const PREPROCESS_WGSL: &str = r#"
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

@compute @workgroup_size(1)
fn main() {
    let d_model = mini.d_model;
    let token_base = mini.token_id * d_model;
    let position = f32(mini.position);

    for (var column: u32 = 0u; column < d_model; column = column + 1u) {
        let pair = column / 2u;
        let exponent = 2.0 * f32(pair) / f32(d_model);
        let divisor = pow(10000.0, exponent);
        let angle = position / divisor;
        let positional = select(cos(angle), sin(angle), (column % 2u) == 0u);
        encoder_io[column] = weights[token_base + column] + positional;
    }
}
"#;

const POSTPROCESS_WGSL: &str = r#"
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

@compute @workgroup_size(1)
fn main() {
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

    // The next preprocess overwrites encoder_io completely, so reuse it for
    // MiniLLM's final normalised hidden row instead of allocating more scratch.
    for (var column: u32 = 0u; column < d_model; column = column + 1u) {
        encoder_io[column] =
            (encoder_io[column] - mean) * inv_std * weights[gamma_offset + column]
            + weights[beta_offset + column];
    }

    var best_value = -3.402823466e+38;
    var best_id = 0u;
    for (var token: u32 = 0u; token < vocab_size; token = token + 1u) {
        var logit = weights[head_bias_offset + token];
        for (var column: u32 = 0u; column < d_model; column = column + 1u) {
            logit = logit
                + encoder_io[column]
                    * weights[head_weight_offset + column * vocab_size + token];
        }
        // Rust Iterator::max_by keeps the later element on Equal. Scanning ids
        // upward with >= reproduces that highest-id tie break for finite logits.
        if (logit >= best_value) {
            best_value = logit;
            best_id = token;
        }
    }

    mini.output_id = best_id;
    mini.position = mini.position + 1u;
}
"#;

#[derive(Debug)]
pub enum WgpuResidentMiniLlmError {
    InvalidConfig(&'static str),
    TokenOutOfRange { token_id: usize, vocab_size: usize },
    PositionMismatch { expected: usize, actual: usize },
    Encoder(WgpuResidentTransformerEncoderError),
    Compute(ComputeError),
}

impl fmt::Display for WgpuResidentMiniLlmError {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidConfig(message) => write!(output, "{message}"),
            Self::TokenOutOfRange {
                token_id,
                vocab_size,
            } => write!(
                output,
                "MiniLLM token id {token_id} is outside vocab size {vocab_size}"
            ),
            Self::PositionMismatch { expected, actual } => write!(
                output,
                "resident MiniLLM position mismatch: expected {expected}, got {actual}"
            ),
            Self::Encoder(error) => write!(output, "{error}"),
            Self::Compute(error) => write!(output, "{error}"),
        }
    }
}

impl std::error::Error for WgpuResidentMiniLlmError {}

impl From<WgpuResidentTransformerEncoderError> for WgpuResidentMiniLlmError {
    fn from(error: WgpuResidentTransformerEncoderError) -> Self {
        Self::Encoder(error)
    }
}

impl From<ComputeError> for WgpuResidentMiniLlmError {
    fn from(error: ComputeError) -> Self {
        Self::Compute(error)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WgpuResidentMiniLlmTelemetry {
    pub capacity_tokens: usize,
    pub resident_tokens: usize,
    pub steps: usize,
    pub d_model: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rank: usize,
    pub resident_bytes: usize,
    pub upload_bytes_per_step: usize,
    pub download_bytes_per_step: usize,
}

/// Persistent greedy MiniLLM inference runtime using a resident Phase 17 encoder.
pub struct WgpuResidentMiniLlm {
    encoder: WgpuResidentTransformerEncoder,
    weights: WgpuComputeBuffer,
    state: WgpuComputeBuffer,
    preprocess_kernel: WgpuComputeKernel,
    postprocess_kernel: WgpuComputeKernel,
    vocab_size: usize,
    max_seq_len: usize,
    resident_bytes: usize,
}

impl WgpuResidentMiniLlm {
    pub fn new(
        snapshot: MiniLlmInferenceSnapshot<'_>,
        capacity: usize,
        rank: usize,
        layers: &[WgpuLatentLayerBasis<'_>],
    ) -> Result<Self, WgpuResidentMiniLlmError> {
        validate_snapshot(&snapshot)?;
        let weights_data = pack_weights(&snapshot)?;
        let vocab_size = snapshot.config.vocab_size;
        let max_seq_len = snapshot.config.max_seq_len;
        let d_model = snapshot.config.d_model;

        let encoder =
            WgpuResidentTransformerEncoder::new(snapshot.encoder, capacity, rank, layers)?;
        if encoder.d_model != d_model
        {
            return Err(WgpuResidentMiniLlmError::InvalidConfig(
                "MiniLLM and resident encoder model widths differ",
            ));
        }

        let weights_bytes = bytes_for_f32(weights_data.len())?;
        let state_bytes = MINI_STATE_WORDS.checked_mul(U32_BYTES).ok_or(
            WgpuResidentMiniLlmError::InvalidConfig("MiniLLM state bytes overflow usize"),
        )?;
        let weights = encoder
            .adapter
            .allocate(weights_bytes, 4, MemorySpace::Device)?;
        let state = encoder
            .adapter
            .allocate(state_bytes, 4, MemorySpace::Device)?;
        encoder
            .adapter
            .write(&weights, 0, bytemuck::cast_slice(&weights_data))?;
        let state_words = [
            0u32,
            0,
            0,
            usize_to_u32(d_model, "MiniLLM d_model exceeds WGPU u32 range")?,
            usize_to_u32(vocab_size, "MiniLLM vocab size exceeds WGPU u32 range")?,
            usize_to_u32(max_seq_len, "MiniLLM max_seq_len exceeds WGPU u32 range")?,
            0,
            0,
        ];
        encoder
            .adapter
            .write(&state, 0, bytemuck::cast_slice(&state_words))?;

        let preprocess_module = KernelModule::new(
            KernelFormat::Wgsl,
            "main",
            PREPROCESS_WGSL.as_bytes().to_vec(),
        )?;
        let postprocess_module = KernelModule::new(
            KernelFormat::Wgsl,
            "main",
            POSTPROCESS_WGSL.as_bytes().to_vec(),
        )?;
        let preprocess_kernel = encoder.adapter.compile(&preprocess_module)?;
        let postprocess_kernel = encoder.adapter.compile(&postprocess_module)?;
        let encoder_bytes = encoder.resident_bytes;

        Ok(Self {
            encoder,
            weights,
            state,
            preprocess_kernel,
            postprocess_kernel,
            vocab_size,
            max_seq_len,
            resident_bytes: encoder_bytes + weights_bytes + state_bytes,
        })
    }

    #[must_use]
    pub const fn telemetry(&self) -> WgpuResidentMiniLlmTelemetry {
        WgpuResidentMiniLlmTelemetry {
            capacity_tokens: self.encoder.capacity,
            resident_tokens: self.encoder.resident_tokens,
            steps: self.encoder.steps,
            d_model: self.encoder.d_model,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_seq_len,
            rank: self.encoder.rank,
            resident_bytes: self.resident_bytes,
            upload_bytes_per_step: U32_BYTES,
            download_bytes_per_step: U32_BYTES,
        }
    }

    /// Execute one greedy cached token step at the exact absolute position.
    ///
    /// Only the input token id is uploaded and only the greedy next token id is
    /// downloaded. The device-resident position is advanced by the postprocess
    /// kernel after successful encoder execution.
    pub fn step_argmax_at(
        &mut self,
        token_id: usize,
        pos: usize,
    ) -> Result<usize, WgpuResidentMiniLlmError> {
        if pos != self.encoder.steps
        {
            return Err(WgpuResidentMiniLlmError::PositionMismatch {
                expected: self.encoder.steps,
                actual: pos,
            });
        }
        if pos >= self.max_seq_len
        {
            return Err(WgpuResidentMiniLlmError::InvalidConfig(
                "MiniLLM position exceeds max_seq_len",
            ));
        }
        if token_id >= self.vocab_size
        {
            return Err(WgpuResidentMiniLlmError::TokenOutOfRange {
                token_id,
                vocab_size: self.vocab_size,
            });
        }

        let token = [usize_to_u32(
            token_id,
            "MiniLLM token id exceeds WGPU u32 range",
        )?];
        self.encoder
            .adapter
            .write(&self.state, 0, bytemuck::cast_slice(&token))?;

        let pre_event = self.launch_preprocess()?;
        self.encoder.adapter.wait(&pre_event)?;

        let encoder_event = self.encoder.launch()?;
        self.encoder.adapter.wait(&encoder_event)?;

        let post_event = self.launch_postprocess()?;
        self.encoder.adapter.wait(&post_event)?;

        self.encoder.steps = self.encoder.steps.saturating_add(1);
        self.encoder.resident_tokens = self
            .encoder
            .resident_tokens
            .saturating_add(1)
            .min(self.encoder.capacity);
        self.encoder.next_slot = (self.encoder.next_slot + 1) % self.encoder.capacity;

        let mut output_id = [0u32; 1];
        self.encoder.adapter.read(
            &self.state,
            2 * U32_BYTES,
            bytemuck::cast_slice_mut(&mut output_id),
        )?;
        Ok(output_id[0] as usize)
    }

    pub fn step_argmax(&mut self, token_id: usize) -> Result<usize, WgpuResidentMiniLlmError> {
        self.step_argmax_at(token_id, self.encoder.steps)
    }

    pub fn reload_weights(
        &mut self,
        snapshot: MiniLlmInferenceSnapshot<'_>,
    ) -> Result<(), WgpuResidentMiniLlmError> {
        validate_snapshot(&snapshot)?;
        if snapshot.config.d_model != self.encoder.d_model
            || snapshot.config.vocab_size != self.vocab_size
            || snapshot.config.max_seq_len != self.max_seq_len
        {
            return Err(WgpuResidentMiniLlmError::InvalidConfig(
                "cannot reload weights from a different MiniLLM topology",
            ));
        }
        self.encoder.reload_weights(snapshot.encoder)?;
        let packed = pack_weights(&snapshot)?;
        self.encoder
            .adapter
            .write(&self.weights, 0, bytemuck::cast_slice(&packed))?;
        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), WgpuResidentMiniLlmError> {
        self.encoder.reset()?;
        let cleared = [0u32, 0, 0];
        self.encoder
            .adapter
            .write(&self.state, 0, bytemuck::cast_slice(&cleared))?;
        Ok(())
    }

    fn launch_preprocess(&self) -> Result<WgpuComputeEvent, WgpuResidentMiniLlmError> {
        let bindings = [
            binding(0, &self.state, BufferAccess::ReadWrite),
            binding(1, &self.weights, BufferAccess::ReadOnly),
            binding(2, &self.encoder.io, BufferAccess::ReadWrite),
        ];
        let config = LaunchConfig::new([1, 1, 1], [1, 1, 1], 0)?;
        Ok(self.encoder.adapter.launch(
            &self.preprocess_kernel,
            &self.encoder.stream,
            config,
            &bindings,
        )?)
    }

    fn launch_postprocess(&self) -> Result<WgpuComputeEvent, WgpuResidentMiniLlmError> {
        let bindings = [
            binding(0, &self.state, BufferAccess::ReadWrite),
            binding(1, &self.weights, BufferAccess::ReadOnly),
            binding(2, &self.encoder.io, BufferAccess::ReadWrite),
        ];
        let config = LaunchConfig::new([1, 1, 1], [1, 1, 1], 0)?;
        Ok(self.encoder.adapter.launch(
            &self.postprocess_kernel,
            &self.encoder.stream,
            config,
            &bindings,
        )?)
    }
}

fn validate_snapshot(
    snapshot: &MiniLlmInferenceSnapshot<'_>,
) -> Result<(), WgpuResidentMiniLlmError> {
    let config = snapshot.config;
    if config.vocab_size == 0
        || config.d_model == 0
        || config.n_heads == 0
        || config.n_layers == 0
        || config.d_ff == 0
        || config.max_seq_len == 0
    {
        return Err(WgpuResidentMiniLlmError::InvalidConfig(
            "resident MiniLLM topology must be non-zero",
        ));
    }
    if snapshot.embedding.vocab_size != config.vocab_size
        || snapshot.embedding.embedding_dim != config.d_model
        || snapshot.embedding.weight.data.len() != config.vocab_size * config.d_model
    {
        return Err(WgpuResidentMiniLlmError::InvalidConfig(
            "MiniLLM embedding topology mismatch",
        ));
    }
    if snapshot.positional_encoding.d_model != config.d_model
        || snapshot.positional_encoding.max_seq_len != config.max_seq_len
    {
        return Err(WgpuResidentMiniLlmError::InvalidConfig(
            "MiniLLM positional encoding topology mismatch",
        ));
    }
    if snapshot.encoder.d_model != config.d_model
        || snapshot.encoder.blocks.len() != config.n_layers
        || snapshot
            .encoder
            .blocks
            .iter()
            .any(|block| block.n_heads != config.n_heads || block.d_ff != config.d_ff)
    {
        return Err(WgpuResidentMiniLlmError::InvalidConfig(
            "MiniLLM encoder topology mismatch",
        ));
    }
    if snapshot.final_norm.gamma.data.len() != config.d_model
        || snapshot.final_norm.beta.data.len() != config.d_model
        || (snapshot.final_norm.eps - 1e-5).abs() > f32::EPSILON
    {
        return Err(WgpuResidentMiniLlmError::InvalidConfig(
            "MiniLLM final LayerNorm topology or epsilon mismatch",
        ));
    }
    if snapshot.lm_head.in_features != config.d_model
        || snapshot.lm_head.out_features != config.vocab_size
        || snapshot.lm_head.weight.data.len() != config.d_model * config.vocab_size
        || snapshot.lm_head.bias.data.len() != config.vocab_size
    {
        return Err(WgpuResidentMiniLlmError::InvalidConfig(
            "MiniLLM LM-head topology mismatch",
        ));
    }
    ensure_u32(config.d_model, "MiniLLM d_model exceeds WGPU u32 range")?;
    ensure_u32(
        config.vocab_size,
        "MiniLLM vocab size exceeds WGPU u32 range",
    )?;
    ensure_u32(
        config.max_seq_len,
        "MiniLLM max_seq_len exceeds WGPU u32 range",
    )?;
    Ok(())
}

fn pack_weights(
    snapshot: &MiniLlmInferenceSnapshot<'_>,
) -> Result<Vec<f32>, WgpuResidentMiniLlmError> {
    validate_snapshot(snapshot)?;
    let config = snapshot.config;
    let total = config
        .vocab_size
        .checked_mul(config.d_model)
        .and_then(|value| value.checked_add(config.d_model.checked_mul(2)?))
        .and_then(|value| value.checked_add(config.d_model.checked_mul(config.vocab_size)?))
        .and_then(|value| value.checked_add(config.vocab_size))
        .ok_or(WgpuResidentMiniLlmError::InvalidConfig(
            "MiniLLM packed inference weights overflow usize",
        ))?;
    ensure_u32(total, "MiniLLM packed weights exceed WGPU u32 range")?;
    let mut packed = Vec::with_capacity(total);
    packed.extend_from_slice(&snapshot.embedding.weight.data);
    packed.extend_from_slice(&snapshot.final_norm.gamma.data);
    packed.extend_from_slice(&snapshot.final_norm.beta.data);
    packed.extend_from_slice(&snapshot.lm_head.weight.data);
    packed.extend_from_slice(&snapshot.lm_head.bias.data);
    debug_assert_eq!(packed.len(), total);
    Ok(packed)
}

fn bytes_for_f32(elements: usize) -> Result<usize, WgpuResidentMiniLlmError> {
    elements.checked_mul(core::mem::size_of::<f32>()).ok_or(
        WgpuResidentMiniLlmError::InvalidConfig("MiniLLM f32 buffer size overflows usize"),
    )
}

fn ensure_u32(value: usize, message: &'static str) -> Result<(), WgpuResidentMiniLlmError> {
    usize_to_u32(value, message).map(|_| ())
}

fn usize_to_u32(value: usize, message: &'static str) -> Result<u32, WgpuResidentMiniLlmError> {
    u32::try_from(value).map_err(|_| WgpuResidentMiniLlmError::InvalidConfig(message))
}
