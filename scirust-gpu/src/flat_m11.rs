//! Resident SciRust ↔ FLAT M11/M15 bridge.
//!
//! This module is intentionally separate from [`crate::GpuChain`] while FLAT is
//! being qualified for SciAgent decode. It proves that SciRust-owned resident
//! `GpuMatrix` buffers can be consumed directly by FLAT's rectangular
//! projection-layout pipeline and that the resulting O buffer remains resident.
//!
//! The bridge never maps or copies Q/K/V through the host. [`record`] and
//! [`record_pre_rotated_k`] also leave command submission and synchronization
//! entirely to the caller, so they can be inserted into an existing SciRust
//! command stream.
//!
//! [`record`]: WgpuFlatM11Bridge::record
//! [`record_pre_rotated_k`]: WgpuFlatM11Bridge::record_pre_rotated_k

use crate::wgpu_backend::{GpuMatrix, WgpuContext};
use crate::{BackendError, BackendResult};
use flat_attention::{
    AsymmetricGroupedAttentionShape, AsymmetricRotaryEmbeddingConfig,
    ExternalAsymmetricProjectionPass, ExternalAsymmetricProjectionRotaryGroupedPipeline,
    FlatAttentionConfig,
};

/// Shape, masking and rotary-position contract for one resident M11 dispatch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlatM11ResidentConfig {
    pub batch: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub query_len: usize,
    pub kv_len: usize,
    pub head_dim: usize,
    pub causal: bool,
    pub softmax_scale: Option<f32>,
    /// Logical query position used by the causal mask.
    pub query_position_offset: usize,
    pub theta: f32,
    /// RoPE origin for local Q row zero.
    pub query_rope_position_offset: usize,
    /// RoPE origin for resident KV row zero.
    pub kv_rope_position_offset: usize,
}

impl FlatM11ResidentConfig {
    fn shape(self) -> AsymmetricGroupedAttentionShape {
        AsymmetricGroupedAttentionShape {
            batch: self.batch,
            q_heads: self.q_heads,
            kv_heads: self.kv_heads,
            query_len: self.query_len,
            kv_len: self.kv_len,
            head_dim: self.head_dim,
            query_position_offset: self.query_position_offset,
        }
    }

    fn attention(self) -> FlatAttentionConfig {
        FlatAttentionConfig {
            causal: self.causal,
            softmax_scale: self.softmax_scale,
        }
    }

    fn rotary(self) -> AsymmetricRotaryEmbeddingConfig {
        AsymmetricRotaryEmbeddingConfig {
            theta: self.theta,
            query_position_offset: self.query_rope_position_offset,
            kv_position_offset: self.kv_rope_position_offset,
        }
    }
}

/// Reusable M11/M15 pipeline bound to one SciRust WGPU context.
///
/// `WgpuContext` is a cheap clone over the same underlying device, so
/// [`from_context`](Self::from_context) can share a device with other resident
/// SciRust operations without creating a second adapter/device.
pub struct WgpuFlatM11Bridge {
    ctx: WgpuContext,
    pipeline: ExternalAsymmetricProjectionRotaryGroupedPipeline,
}

impl core::fmt::Debug for WgpuFlatM11Bridge {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WgpuFlatM11Bridge")
            .field("adapter", &self.ctx.adapter_name())
            .finish_non_exhaustive()
    }
}

impl WgpuFlatM11Bridge {
    /// Acquire a fresh SciRust WGPU context and compile the FLAT pipeline.
    pub fn new() -> BackendResult<Self> {
        Self::from_context(WgpuContext::new()?)
    }

    /// Compile FLAT on an existing SciRust context. The context clone shares the
    /// same device/queue and therefore the same resident-buffer ownership domain.
    pub fn from_context(ctx: WgpuContext) -> BackendResult<Self> {
        let pipeline = ExternalAsymmetricProjectionRotaryGroupedPipeline::new(ctx.device())
            .map_err(|error| BackendError::Execution(format!("FLAT M11 pipeline: {error}")))?;
        Ok(Self { ctx, pipeline })
    }

    /// Underlying adapter name, useful for benchmark provenance.
    #[must_use]
    pub fn adapter_name(&self) -> &str {
        self.ctx.adapter_name()
    }

    /// Borrow the shared context for adjacent resident SciRust operations.
    #[must_use]
    pub fn context(&self) -> &WgpuContext {
        &self.ctx
    }

    /// Allocate the combined FLAT O|LSE backing buffer while exposing only O as
    /// a logical `GpuMatrix`. The hidden trailing LSE region remains available
    /// to FLAT in the same backing allocation.
    pub fn create_output(&self, config: FlatM11ResidentConfig) -> BackendResult<GpuMatrix> {
        let shape = config.shape();
        let rows = checked_mul(config.batch, config.query_len, "M11 batch*query_len")?;
        let cols = checked_mul(config.q_heads, config.head_dim, "M11 q_heads*head_dim")?;
        let output = self
            .pipeline
            .create_output_buffer(self.ctx.device(), shape)
            .map_err(|error| {
                BackendError::Execution(format!("FLAT M11 output allocation: {error}"))
            })?;
        GpuMatrix::from_external_buffer(output, rows, cols)
    }

    /// Record one zero-copy rectangular attention pass with raw projected K.
    /// Q and K RoPE are both fused by FLAT. This method does not submit, poll,
    /// map or synchronize.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
        output: &GpuMatrix,
        config: FlatM11ResidentConfig,
    ) -> BackendResult<()> {
        let pass = self.external_pass(q, k, v, output, config)?;
        self.record_impl(encoder, pass, false)
    }

    /// Record one zero-copy decode-compatible pass where K is already
    /// RoPE-rotated by the resident cache owner. Q RoPE remains fused in FLAT;
    /// K is consumed as-is and V remains raw. No submission, polling, mapping or
    /// synchronization occurs.
    pub fn record_pre_rotated_k(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
        output: &GpuMatrix,
        config: FlatM11ResidentConfig,
    ) -> BackendResult<()> {
        let pass = self.external_pass(q, k, v, output, config)?;
        self.record_impl(encoder, pass, true)
    }

    /// Convenience resident forward using raw projected K.
    pub fn forward(
        &self,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
        config: FlatM11ResidentConfig,
    ) -> BackendResult<GpuMatrix> {
        self.forward_impl(q, k, v, config, false)
    }

    /// Convenience resident forward for a cache whose K rows are already
    /// RoPE-rotated. Q/K/V never leave VRAM and O remains resident.
    pub fn forward_pre_rotated_k(
        &self,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
        config: FlatM11ResidentConfig,
    ) -> BackendResult<GpuMatrix> {
        self.forward_impl(q, k, v, config, true)
    }

    fn external_pass<'a>(
        &self,
        q: &'a GpuMatrix,
        k: &'a GpuMatrix,
        v: &'a GpuMatrix,
        output: &'a GpuMatrix,
        config: FlatM11ResidentConfig,
    ) -> BackendResult<ExternalAsymmetricProjectionPass<'a>> {
        self.validate_matrices(q, k, v, output, config)?;
        Ok(ExternalAsymmetricProjectionPass {
            q: q.buffer(),
            k: k.buffer(),
            v: v.buffer(),
            out_and_lse: output.buffer(),
            shape: config.shape(),
            config: config.attention(),
            rotary: config.rotary(),
        })
    }

    fn record_impl(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pass: ExternalAsymmetricProjectionPass<'_>,
        pre_rotated_k: bool,
    ) -> BackendResult<()> {
        let result = if pre_rotated_k
        {
            self.pipeline
                .encode_pre_rotated_k(self.ctx.device(), encoder, pass)
        }
        else
        {
            self.pipeline.encode(self.ctx.device(), encoder, pass)
        };
        result.map_err(|error| BackendError::Execution(format!("FLAT M11 encode: {error}")))?;
        Ok(())
    }

    fn forward_impl(
        &self,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
        config: FlatM11ResidentConfig,
        pre_rotated_k: bool,
    ) -> BackendResult<GpuMatrix> {
        let output = self.create_output(config)?;
        let mut encoder =
            self.ctx
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("scirust-flat-m11-resident"),
                });
        if pre_rotated_k
        {
            self.record_pre_rotated_k(&mut encoder, q, k, v, &output, config)?;
        }
        else
        {
            self.record(&mut encoder, q, k, v, &output, config)?;
        }
        self.ctx.queue().submit(Some(encoder.finish()));
        Ok(output)
    }

    fn validate_matrices(
        &self,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
        output: &GpuMatrix,
        config: FlatM11ResidentConfig,
    ) -> BackendResult<()> {
        let q_rows = checked_mul(config.batch, config.query_len, "M11 batch*query_len")?;
        let kv_rows = checked_mul(config.batch, config.kv_len, "M11 batch*kv_len")?;
        let q_cols = checked_mul(config.q_heads, config.head_dim, "M11 q_heads*head_dim")?;
        let kv_cols = checked_mul(config.kv_heads, config.head_dim, "M11 kv_heads*head_dim")?;

        expect_shape("Q", q, q_rows, q_cols)?;
        expect_shape("K", k, kv_rows, kv_cols)?;
        expect_shape("V", v, kv_rows, kv_cols)?;
        expect_shape("O", output, q_rows, q_cols)?;
        Ok(())
    }
}

fn checked_mul(a: usize, b: usize, what: &'static str) -> BackendResult<usize> {
    a.checked_mul(b)
        .ok_or_else(|| BackendError::ShapeMismatch(format!("{what} overflow")))
}

fn expect_shape(
    name: &'static str,
    matrix: &GpuMatrix,
    rows: usize,
    cols: usize,
) -> BackendResult<()> {
    if matrix.rows() != rows || matrix.cols() != cols
    {
        return Err(BackendError::ShapeMismatch(format!(
            "FLAT M11 {name} is {}x{}, expected {rows}x{cols}",
            matrix.rows(),
            matrix.cols()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flat_attention::forward_reference_projection_grouped_rope_asymmetric;

    const ATOL: f32 = 1.5e-4;
    const RTOL: f32 = 1.0e-3;

    fn fixture(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let x = index as f32 * 0.023 + phase;
                x.sin() * 1.875 + (x * 0.41).cos() * 0.28125
            })
            .collect()
    }

    fn rotate_k_projection(
        raw: &[f32],
        kv_len: usize,
        kv_heads: usize,
        head_dim: usize,
        theta: f32,
        position_offset: usize,
    ) -> Vec<f32> {
        let mut rotated = raw.to_vec();
        let width = kv_heads * head_dim;
        for position in 0..kv_len
        {
            let absolute_position = position_offset + position;
            for head in 0..kv_heads
            {
                let head_base = position * width + head * head_dim;
                for pair in 0..head_dim / 2
                {
                    let dim = 2 * pair;
                    let exponent = -2.0 * pair as f32 / head_dim as f32;
                    let frequency = theta.powf(exponent);
                    let angle = absolute_position as f32 * frequency;
                    let (sin, cos) = angle.sin_cos();
                    let even = raw[head_base + dim];
                    let odd = raw[head_base + dim + 1];
                    rotated[head_base + dim] = even * cos - odd * sin;
                    rotated[head_base + dim + 1] = even * sin + odd * cos;
                }
            }
        }
        rotated
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate()
        {
            let tolerance = ATOL + RTOL * expected.abs();
            let error = (actual - expected).abs();
            assert!(
                error <= tolerance,
                "index {index}: actual={actual}, expected={expected}, abs_error={error}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn resident_decode_matches_flat_reference_without_host_qkv_roundtrip() {
        let Ok(bridge) = WgpuFlatM11Bridge::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping FLAT M11 resident bridge test");
            return;
        };
        let config = FlatM11ResidentConfig {
            batch: 1,
            q_heads: 8,
            kv_heads: 2,
            query_len: 1,
            kv_len: 17,
            head_dim: 64,
            causal: true,
            softmax_scale: None,
            query_position_offset: 16,
            theta: 10_000.0,
            query_rope_position_offset: 16,
            kv_rope_position_offset: 0,
        };
        let shape = config.shape();
        let q = fixture(shape.q_tensor_len().unwrap(), 0.2);
        let k = fixture(shape.kv_tensor_len().unwrap(), 0.8);
        let v = fixture(shape.kv_tensor_len().unwrap(), 1.4);
        let q_gpu = bridge.context().upload(&q, 1, 8 * 64);
        let k_gpu = bridge.context().upload(&k, 17, 2 * 64);
        let v_gpu = bridge.context().upload(&v, 17, 2 * 64);

        let output = bridge.forward(&q_gpu, &k_gpu, &v_gpu, config).unwrap();
        let actual = bridge.context().download(&output).unwrap();
        let expected = forward_reference_projection_grouped_rope_asymmetric(
            &q,
            &k,
            &v,
            shape,
            config.attention(),
            config.rotary(),
        )
        .unwrap();
        assert_close(&actual, &expected.output);
    }

    #[test]
    fn resident_pre_rotated_k_decode_matches_raw_k_reference() {
        let Ok(bridge) = WgpuFlatM11Bridge::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping FLAT M15 resident bridge test");
            return;
        };
        let config = FlatM11ResidentConfig {
            batch: 1,
            q_heads: 8,
            kv_heads: 2,
            query_len: 1,
            kv_len: 17,
            head_dim: 64,
            causal: true,
            softmax_scale: None,
            query_position_offset: 16,
            theta: 10_000.0,
            query_rope_position_offset: 16,
            kv_rope_position_offset: 0,
        };
        let shape = config.shape();
        let q = fixture(shape.q_tensor_len().unwrap(), 0.25);
        let raw_k = fixture(shape.kv_tensor_len().unwrap(), 0.85);
        let v = fixture(shape.kv_tensor_len().unwrap(), 1.45);
        let rotated_k = rotate_k_projection(
            &raw_k,
            config.kv_len,
            config.kv_heads,
            config.head_dim,
            config.theta,
            config.kv_rope_position_offset,
        );
        let q_gpu = bridge
            .context()
            .upload(&q, 1, config.q_heads * config.head_dim);
        let k_gpu =
            bridge
                .context()
                .upload(&rotated_k, config.kv_len, config.kv_heads * config.head_dim);
        let v_gpu = bridge
            .context()
            .upload(&v, config.kv_len, config.kv_heads * config.head_dim);

        let output = bridge
            .forward_pre_rotated_k(&q_gpu, &k_gpu, &v_gpu, config)
            .unwrap();
        let actual = bridge.context().download(&output).unwrap();
        let expected = forward_reference_projection_grouped_rope_asymmetric(
            &q,
            &raw_k,
            &v,
            shape,
            config.attention(),
            config.rotary(),
        )
        .unwrap();
        assert_close(&actual, &expected.output);
    }

    #[test]
    fn resident_bridge_rejects_logically_short_k_even_with_valid_gpu_buffer() {
        let Ok(bridge) = WgpuFlatM11Bridge::new()
        else
        {
            eprintln!("wgpu: no adapter, skipping FLAT M11 shape test");
            return;
        };
        let config = FlatM11ResidentConfig {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            query_len: 1,
            kv_len: 8,
            head_dim: 32,
            causal: true,
            softmax_scale: None,
            query_position_offset: 7,
            theta: 10_000.0,
            query_rope_position_offset: 7,
            kv_rope_position_offset: 0,
        };
        let q = bridge.context().upload(&vec![0.1; 4 * 32], 1, 4 * 32);
        let short_k = bridge.context().upload(&vec![0.2; 7 * 2 * 32], 7, 2 * 32);
        let v = bridge.context().upload(&vec![0.3; 8 * 2 * 32], 8, 2 * 32);
        let output = bridge.create_output(config).unwrap();
        let mut encoder =
            bridge
                .context()
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("scirust-flat-m11-short-k"),
                });
        let error = bridge
            .record(&mut encoder, &q, &short_k, &v, &output, config)
            .unwrap_err();
        assert!(matches!(error, BackendError::ShapeMismatch(_)));
    }
}
