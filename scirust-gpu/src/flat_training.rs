//! Resident SciRust ↔ FLAT grouped forward/backward training bridge.
//!
//! This bridge wires FLAT M24's caller-owned native GQA/MQA forward pipeline to
//! the qualified grouped backward-recomputation pipeline on one SciRust-owned
//! WGPU context. Q/K/V, O/LSE, dO and dQ/dK/dV remain device-resident throughout
//! the chained dispatch. The bridge uses only device-to-device copies to build
//! FLAT's packed backward input; it never maps Q/K/V or forward activations
//! through the host.
//!
//! The tensor contract in this module is FLAT's native **head-major** layout:
//! Q/O/dO/dQ are `[batch, q_heads, seq_len, head_dim]`, K/V/dK/dV are
//! `[batch, kv_heads, seq_len, head_dim]`, and LSE is
//! `[batch, q_heads, seq_len]`. Each tensor is exposed as a [`GpuMatrix`] whose
//! rows flatten every axis except `head_dim` (LSE uses `seq_len` as columns).
//! This deliberately does not replace the projection-layout M11/M15 decode
//! bridge; layout/RoPE integration into a full training block is a later slice.

use crate::wgpu_backend::{GpuMatrix, WgpuContext};
use crate::{BackendError, BackendResult};
use flat_attention::{
    FlatAttentionConfig, GroupedAttentionShape, GroupedBackwardRecomputePass, GroupedForwardPass,
    WgpuGroupedBackwardRecomputePipeline, WgpuGroupedForwardPipeline,
};

/// Native grouped-attention shape and masking contract for one training pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlatGroupedTrainingConfig {
    pub batch: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub seq_len: usize,
    pub head_dim: usize,
    pub causal: bool,
    pub softmax_scale: Option<f32>,
}

impl FlatGroupedTrainingConfig {
    fn shape(self) -> GroupedAttentionShape {
        GroupedAttentionShape {
            batch: self.batch,
            q_heads: self.q_heads,
            kv_heads: self.kv_heads,
            seq_len: self.seq_len,
            head_dim: self.head_dim,
        }
    }

    fn attention(self) -> FlatAttentionConfig {
        FlatAttentionConfig {
            causal: self.causal,
            softmax_scale: self.softmax_scale,
        }
    }

    fn q_rows(self) -> BackendResult<usize> {
        checked_product(
            &[self.batch, self.q_heads, self.seq_len],
            "FLAT grouped Q rows",
        )
    }

    fn kv_rows(self) -> BackendResult<usize> {
        checked_product(
            &[self.batch, self.kv_heads, self.seq_len],
            "FLAT grouped KV rows",
        )
    }

    fn lse_rows(self) -> BackendResult<usize> {
        checked_product(&[self.batch, self.q_heads], "FLAT grouped LSE rows")
    }
}

/// Resident outputs from one grouped forward→backward training chain.
///
/// `output`, `lse`, `dq`, `dk` and `dv` are ordinary SciRust [`GpuMatrix`]
/// handles and may feed subsequent resident operations. Private buffers retain
/// the packed backward state until the result is dropped. When using
/// [`WgpuFlatGroupedTrainingBridge::record_forward_backward`], keep this value
/// alive at least until the containing command buffer has been submitted.
pub struct FlatGroupedTrainingResult {
    pub output: GpuMatrix,
    pub lse: GpuMatrix,
    pub dq: GpuMatrix,
    pub dk: GpuMatrix,
    pub dv: GpuMatrix,
    _packed_forward: wgpu::Buffer,
    _packed_grads: wgpu::Buffer,
}

/// Reusable grouped forward/backward pipelines bound to one SciRust WGPU context.
pub struct WgpuFlatGroupedTrainingBridge {
    ctx: WgpuContext,
    forward: WgpuGroupedForwardPipeline,
    backward: WgpuGroupedBackwardRecomputePipeline,
}

impl core::fmt::Debug for WgpuFlatGroupedTrainingBridge {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("WgpuFlatGroupedTrainingBridge")
            .field("adapter", &self.ctx.adapter_name())
            .finish_non_exhaustive()
    }
}

impl WgpuFlatGroupedTrainingBridge {
    /// Acquire a fresh SciRust WGPU context and compile both FLAT pipelines.
    pub fn new() -> BackendResult<Self> {
        Self::from_context(WgpuContext::new()?)
    }

    /// Compile both FLAT pipelines on an existing SciRust context.
    ///
    /// `WgpuContext` clones share the same device and queue, so callers can
    /// upload or produce Q/K/V/dO with adjacent SciRust operations and pass the
    /// resident buffers here without opening a second adapter/device.
    pub fn from_context(ctx: WgpuContext) -> BackendResult<Self> {
        let forward = WgpuGroupedForwardPipeline::new(ctx.device()).map_err(|error| {
            BackendError::Execution(format!("FLAT M24 grouped forward pipeline: {error}"))
        })?;
        let backward =
            WgpuGroupedBackwardRecomputePipeline::new(ctx.device()).map_err(|error| {
                BackendError::Execution(format!("FLAT grouped backward pipeline: {error}"))
            })?;
        Ok(Self {
            ctx,
            forward,
            backward,
        })
    }

    /// Underlying adapter name for correctness/benchmark provenance.
    #[must_use]
    pub fn adapter_name(&self) -> &str {
        self.ctx.adapter_name()
    }

    /// Borrow the shared SciRust context for adjacent resident operations.
    #[must_use]
    pub fn context(&self) -> &WgpuContext {
        &self.ctx
    }

    /// Record one native grouped forward→backward chain into `encoder`.
    ///
    /// This method performs no queue submission, polling, mapping or host
    /// readback. Forward output/LSE are copied into the backward packed buffer
    /// with `copy_buffer_to_buffer`, and packed dQ/dK/dV are split into resident
    /// SciRust matrices with the same GPU-only mechanism.
    pub fn record_forward_backward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
        d_out: &GpuMatrix,
        config: FlatGroupedTrainingConfig,
    ) -> BackendResult<FlatGroupedTrainingResult> {
        let shape = config.shape();
        let attention = config.attention();
        let forward_layout = WgpuGroupedForwardPipeline::layout(shape).map_err(|error| {
            BackendError::ShapeMismatch(format!("FLAT grouped forward layout: {error}"))
        })?;
        let backward_layout =
            WgpuGroupedBackwardRecomputePipeline::layout(shape).map_err(|error| {
                BackendError::ShapeMismatch(format!("FLAT grouped backward layout: {error}"))
            })?;

        let q_rows = config.q_rows()?;
        let kv_rows = config.kv_rows()?;
        let lse_rows = config.lse_rows()?;
        expect_shape("Q", q, q_rows, config.head_dim)?;
        expect_shape("K", k, kv_rows, config.head_dim)?;
        expect_shape("V", v, kv_rows, config.head_dim)?;
        expect_shape("dO", d_out, q_rows, config.head_dim)?;

        let forward_buffer = self
            .forward
            .create_output_buffer(self.ctx.device(), shape)
            .map_err(|error| {
                BackendError::Execution(format!("FLAT grouped forward output allocation: {error}"))
            })?;
        let output = GpuMatrix::from_external_buffer(forward_buffer, q_rows, config.head_dim)?;
        let lse = create_matrix(
            &self.ctx,
            lse_rows,
            config.seq_len,
            "scirust-flat-grouped-lse",
        )?;
        let dq = create_matrix(
            &self.ctx,
            q_rows,
            config.head_dim,
            "scirust-flat-grouped-dq",
        )?;
        let dk = create_matrix(
            &self.ctx,
            kv_rows,
            config.head_dim,
            "scirust-flat-grouped-dk",
        )?;
        let dv = create_matrix(
            &self.ctx,
            kv_rows,
            config.head_dim,
            "scirust-flat-grouped-dv",
        )?;

        let packed_forward = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("scirust-flat-grouped-packed-forward"),
            size: backward_layout.packed_forward_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let packed_grads = self
            .backward
            .create_gradient_buffer(self.ctx.device(), shape)
            .map_err(|error| {
                BackendError::Execution(format!("FLAT grouped gradient allocation: {error}"))
            })?;

        // Keep backward validation fail-closed before mutating the caller's
        // encoder. The concrete prepared type stays internal to FLAT and is
        // intentionally not exposed through the SciRust public API.
        let prepared_backward = self
            .backward
            .prepare(
                self.ctx.device(),
                GroupedBackwardRecomputePass {
                    packed_forward: &packed_forward,
                    packed_grads: &packed_grads,
                    shape,
                    config: attention,
                },
            )
            .map_err(|error| {
                BackendError::Execution(format!("FLAT grouped backward prepare: {error}"))
            })?;

        self.forward
            .encode(
                self.ctx.device(),
                encoder,
                GroupedForwardPass {
                    q: q.buffer(),
                    k: k.buffer(),
                    v: v.buffer(),
                    output: output.buffer(),
                    shape,
                    config: attention,
                },
            )
            .map_err(|error| {
                BackendError::Execution(format!("FLAT M24 grouped forward encode: {error}"))
            })?;

        let q_bytes = checked_bytes(forward_layout.q_elements, "FLAT grouped Q bytes")?;
        let kv_bytes = checked_bytes(forward_layout.kv_elements, "FLAT grouped KV bytes")?;
        let lse_bytes = checked_bytes(forward_layout.lse_elements, "FLAT grouped LSE bytes")?;

        // Expose LSE as its own resident SciRust matrix while the combined
        // forward backing buffer remains the direct O source for backward.
        encoder.copy_buffer_to_buffer(output.buffer(), q_bytes, lse.buffer(), 0, lse_bytes);

        encoder.copy_buffer_to_buffer(q.buffer(), 0, &packed_forward, 0, q_bytes);
        encoder.copy_buffer_to_buffer(
            k.buffer(),
            0,
            &packed_forward,
            checked_bytes(backward_layout.k_offset(), "FLAT grouped K offset")?,
            kv_bytes,
        );
        encoder.copy_buffer_to_buffer(
            v.buffer(),
            0,
            &packed_forward,
            checked_bytes(backward_layout.v_offset(), "FLAT grouped V offset")?,
            kv_bytes,
        );
        encoder.copy_buffer_to_buffer(
            d_out.buffer(),
            0,
            &packed_forward,
            checked_bytes(backward_layout.d_out_offset(), "FLAT grouped dO offset")?,
            q_bytes,
        );
        encoder.copy_buffer_to_buffer(
            output.buffer(),
            0,
            &packed_forward,
            checked_bytes(backward_layout.output_offset(), "FLAT grouped O offset")?,
            forward_layout.output_bytes,
        );

        self.backward.encode_prepared(encoder, &prepared_backward);

        encoder.copy_buffer_to_buffer(&packed_grads, 0, dq.buffer(), 0, q_bytes);
        encoder.copy_buffer_to_buffer(
            &packed_grads,
            checked_bytes(backward_layout.dk_offset(), "FLAT grouped dK offset")?,
            dk.buffer(),
            0,
            kv_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &packed_grads,
            checked_bytes(backward_layout.dv_offset(), "FLAT grouped dV offset")?,
            dv.buffer(),
            0,
            kv_bytes,
        );

        Ok(FlatGroupedTrainingResult {
            output,
            lse,
            dq,
            dk,
            dv,
            _packed_forward: packed_forward,
            _packed_grads: packed_grads,
        })
    }

    /// Convenience asynchronous submission of one resident training chain.
    ///
    /// Q/K/V/dO are never read back to the host and the returned O/LSE/gradients
    /// remain resident. The method submits once and deliberately does not poll
    /// or wait; synchronization remains explicit at the consumer boundary.
    pub fn forward_backward(
        &self,
        q: &GpuMatrix,
        k: &GpuMatrix,
        v: &GpuMatrix,
        d_out: &GpuMatrix,
        config: FlatGroupedTrainingConfig,
    ) -> BackendResult<FlatGroupedTrainingResult> {
        let mut encoder =
            self.ctx
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("scirust-flat-grouped-forward-backward"),
                });
        let result = self.record_forward_backward(&mut encoder, q, k, v, d_out, config)?;
        self.ctx.queue().submit(Some(encoder.finish()));
        Ok(result)
    }
}

fn create_matrix(
    ctx: &WgpuContext,
    rows: usize,
    cols: usize,
    label: &'static str,
) -> BackendResult<GpuMatrix> {
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| BackendError::ShapeMismatch(format!("{label} shape overflow")))?;
    let bytes = checked_bytes(elements, label)?;
    let buffer = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(core::mem::size_of::<f32>() as u64),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    GpuMatrix::from_external_buffer(buffer, rows, cols)
}

fn checked_product(values: &[usize], what: &'static str) -> BackendResult<usize> {
    values.iter().copied().try_fold(1usize, |acc, value| {
        acc.checked_mul(value)
            .ok_or_else(|| BackendError::ShapeMismatch(format!("{what} overflow")))
    })
}

fn checked_bytes(elements: usize, what: &'static str) -> BackendResult<u64> {
    let bytes = elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| BackendError::ShapeMismatch(format!("{what} overflow")))?;
    u64::try_from(bytes)
        .map_err(|_| BackendError::ShapeMismatch(format!("{what} exceeds u64 byte space")))
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
            "FLAT grouped {name} is {}x{}, expected {rows}x{cols}",
            matrix.rows(),
            matrix.cols()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flat_attention::{backward_reference_grouped, forward_reference_grouped};

    const ATOL: f32 = 6.0e-4;
    const RTOL: f32 = 2.5e-3;

    fn fixture(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|index| {
                let x = index as f32 * 0.043 + phase;
                x.sin() * 0.7 + (x * 0.37).cos() * 0.3
            })
            .collect()
    }

    fn assert_close(name: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate()
        {
            let tolerance = ATOL + RTOL * expected.abs();
            let error = (actual - expected).abs();
            assert!(
                error <= tolerance,
                "{name}[{index}]: actual={actual}, expected={expected}, abs_error={error}, tolerance={tolerance}"
            );
        }
    }

    fn bridge_or_skip(label: &str) -> Option<WgpuFlatGroupedTrainingBridge> {
        match WgpuFlatGroupedTrainingBridge::new()
        {
            Ok(bridge) => Some(bridge),
            Err(error) =>
            {
                if std::env::var_os("SCIRUST_REQUIRE_FLAT_GROUPED_WGPU").is_some()
                {
                    panic!("{label}: WGPU FLAT grouped training bridge is required: {error}");
                }
                eprintln!("{label}: no WGPU adapter, skipping: {error}");
                None
            },
        }
    }

    fn run_case(config: FlatGroupedTrainingConfig) {
        let Some(bridge) = bridge_or_skip("FLAT grouped training parity")
        else
        {
            return;
        };
        let shape = config.shape();
        let q_len = shape.q_tensor_len().unwrap();
        let kv_len = shape.kv_tensor_len().unwrap();
        let q = fixture(q_len, 0.1);
        let k = fixture(kv_len, 0.7);
        let v = fixture(kv_len, 1.3);
        let d_out = fixture(q_len, 1.9);
        let expected_forward =
            forward_reference_grouped(&q, &k, &v, shape, config.attention()).unwrap();
        let expected_backward = backward_reference_grouped(
            &q,
            &k,
            &v,
            &d_out,
            shape,
            config.attention(),
            &expected_forward,
        )
        .unwrap();

        let q_gpu = bridge
            .context()
            .upload(&q, config.q_rows().unwrap(), config.head_dim);
        let k_gpu = bridge
            .context()
            .upload(&k, config.kv_rows().unwrap(), config.head_dim);
        let v_gpu = bridge
            .context()
            .upload(&v, config.kv_rows().unwrap(), config.head_dim);
        let d_out_gpu = bridge
            .context()
            .upload(&d_out, config.q_rows().unwrap(), config.head_dim);

        let mut encoder =
            bridge
                .context()
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("scirust-flat-grouped-training-test"),
                });
        let result = bridge
            .record_forward_backward(&mut encoder, &q_gpu, &k_gpu, &v_gpu, &d_out_gpu, config)
            .unwrap();
        bridge.context().queue().submit(Some(encoder.finish()));

        let output = bridge.context().download(&result.output).unwrap();
        let lse = bridge.context().download(&result.lse).unwrap();
        let dq = bridge.context().download(&result.dq).unwrap();
        let dk = bridge.context().download(&result.dk).unwrap();
        let dv = bridge.context().download(&result.dv).unwrap();

        assert_close("O", &output, &expected_forward.output);
        assert_close("LSE", &lse, &expected_forward.lse);
        assert_close("dQ", &dq, &expected_backward.dq);
        assert_close("dK", &dk, &expected_backward.dk);
        assert_close("dV", &dv, &expected_backward.dv);
    }

    #[test]
    fn resident_gqa_causal_training_chain_matches_oracle() {
        run_case(FlatGroupedTrainingConfig {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            seq_len: 7,
            head_dim: 16,
            causal: true,
            softmax_scale: None,
        });
    }

    #[test]
    fn resident_mqa_noncausal_training_chain_matches_oracle() {
        run_case(FlatGroupedTrainingConfig {
            batch: 2,
            q_heads: 4,
            kv_heads: 1,
            seq_len: 5,
            head_dim: 8,
            causal: false,
            softmax_scale: Some(0.31),
        });
    }

    #[test]
    fn rejects_non_native_head_major_input_shape_before_encoding() {
        let Some(bridge) = bridge_or_skip("FLAT grouped shape rejection")
        else
        {
            return;
        };
        let config = FlatGroupedTrainingConfig {
            batch: 1,
            q_heads: 4,
            kv_heads: 2,
            seq_len: 3,
            head_dim: 8,
            causal: true,
            softmax_scale: None,
        };
        let q = bridge.context().upload(&[0.0; 4 * 3 * 8], 3, 4 * 8);
        let k = bridge.context().upload(&[0.0; 2 * 3 * 8], 2 * 3, 8);
        let v = bridge.context().upload(&[0.0; 2 * 3 * 8], 2 * 3, 8);
        let d_out = bridge.context().upload(&[0.0; 4 * 3 * 8], 4 * 3, 8);
        let mut encoder =
            bridge
                .context()
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("scirust-flat-grouped-shape-reject"),
                });
        let error = bridge
            .record_forward_backward(&mut encoder, &q, &k, &v, &d_out, config)
            .err()
            .expect("sequence-major Q shape must be rejected");
        assert!(matches!(error, BackendError::ShapeMismatch(_)));
    }
}
