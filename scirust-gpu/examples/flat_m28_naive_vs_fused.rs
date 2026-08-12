//! Paired M28 resident-attention benchmark: SciRust's existing naive multi-dispatch
//! WGPU composition versus FLAT's public fused grouped-forward pipeline.
//!
//! The two paths share one SciRust-owned WGPU context and use separate resident
//! Q/K/V buffers populated from identical bytes. Shape, causal flag and scalar
//! oracle are identical. Upload and readback are outside the timed region. The
//! benchmark reports measurements only; it does not promote a generic speedup claim.

use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use flat_attention::{
    FlatAttentionConfig, GroupedAttentionShape, GroupedForwardPass, WgpuGroupedForwardPipeline,
    forward_reference_grouped,
};
use scirust_gpu::{GpuMatrix, WgpuContext};

const DEFAULT_WARMUPS: usize = 3;
const DEFAULT_REPEATS: usize = 9;
const DEFAULT_SEQ_LEN: usize = 128;
const DEFAULT_HEAD_DIM: usize = 64;
const ATOL: f32 = 8.0e-4;
const RTOL: f32 = 3.0e-3;

#[derive(Clone, Copy)]
struct FlatResidentInputs<'a> {
    q: &'a wgpu::Buffer,
    k: &'a wgpu::Buffer,
    v: &'a wgpu::Buffer,
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn fixture(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let x = index as f32 * 0.037 + phase;
            x.sin() * 0.65 + (x * 0.41).cos() * 0.35
        })
        .collect()
}

fn bytes_f32(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for &value in values
    {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn flat_input_buffer(ctx: &WgpuContext, values: &[f32], label: &'static str) -> wgpu::Buffer {
    let bytes = bytes_f32(values);
    let buffer = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !bytes.is_empty()
    {
        ctx.queue().write_buffer(&buffer, 0, &bytes);
    }
    buffer
}

fn percentile_ns(samples: &[Duration], percentile: usize) -> u128 {
    let mut values: Vec<u128> = samples.iter().map(Duration::as_nanos).collect();
    values.sort_unstable();
    let rank = percentile.saturating_mul(values.len()).div_ceil(100).max(1);
    values[rank - 1]
}

fn median_ns(samples: &[Duration]) -> u128 {
    percentile_ns(samples, 50)
}

fn max_abs_error(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f32::max)
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32]) -> Result<f32, Box<dyn Error>> {
    if actual.len() != expected.len()
    {
        return Err(format!("{name} length {} != {}", actual.len(), expected.len()).into());
    }
    let mut worst = 0.0f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate()
    {
        let abs = (actual - expected).abs();
        let limit = ATOL + RTOL * actual.abs().max(expected.abs());
        if !actual.is_finite() || abs > limit
        {
            return Err(format!(
                "{name}[{index}] actual={actual} expected={expected} abs={abs} limit={limit}"
            )
            .into());
        }
        worst = worst.max(abs);
    }
    Ok(worst)
}

fn naive_attention(
    ctx: &WgpuContext,
    q: &GpuMatrix,
    k: &GpuMatrix,
    v: &GpuMatrix,
    causal: bool,
) -> Result<GpuMatrix, Box<dyn Error>> {
    let scores = ctx.gemm_resident(q, k, false, true)?;
    let scale = 1.0 / (q.cols() as f32).sqrt();
    let masked = ctx.scale_causal_mask_resident(&scores, scale, causal)?;
    let probs = ctx.softmax_resident(&masked)?;
    Ok(ctx.gemm_resident(&probs, v, false, false)?)
}

fn run_flat(
    ctx: &WgpuContext,
    pipeline: &WgpuGroupedForwardPipeline,
    inputs: FlatResidentInputs<'_>,
    output: &wgpu::Buffer,
    shape: GroupedAttentionShape,
    config: FlatAttentionConfig,
) -> Result<(), Box<dyn Error>> {
    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scirust-m28-flat-forward"),
        });
    pipeline.encode(
        ctx.device(),
        &mut encoder,
        GroupedForwardPass {
            q: inputs.q,
            k: inputs.k,
            v: inputs.v,
            output,
            shape,
            config,
        },
    )?;
    ctx.queue().submit(Some(encoder.finish()));
    let _ = ctx.device().poll(wgpu::Maintain::Wait);
    Ok(())
}

fn time_naive(
    ctx: &WgpuContext,
    q: &GpuMatrix,
    k: &GpuMatrix,
    v: &GpuMatrix,
    causal: bool,
) -> Result<Duration, Box<dyn Error>> {
    let start = Instant::now();
    let output = naive_attention(ctx, q, k, v, causal)?;
    let _ = ctx.device().poll(wgpu::Maintain::Wait);
    black_box(output);
    Ok(start.elapsed())
}

fn time_flat_reused(
    ctx: &WgpuContext,
    pipeline: &WgpuGroupedForwardPipeline,
    inputs: FlatResidentInputs<'_>,
    output: &wgpu::Buffer,
    shape: GroupedAttentionShape,
    config: FlatAttentionConfig,
) -> Result<Duration, Box<dyn Error>> {
    let start = Instant::now();
    run_flat(ctx, pipeline, inputs, output, shape, config)?;
    Ok(start.elapsed())
}

fn time_flat_fresh_output(
    ctx: &WgpuContext,
    pipeline: &WgpuGroupedForwardPipeline,
    inputs: FlatResidentInputs<'_>,
    shape: GroupedAttentionShape,
    config: FlatAttentionConfig,
) -> Result<Duration, Box<dyn Error>> {
    let start = Instant::now();
    let output = pipeline.create_output_buffer(ctx.device(), shape)?;
    run_flat(ctx, pipeline, inputs, &output, shape, config)?;
    black_box(output);
    Ok(start.elapsed())
}

fn main() -> Result<(), Box<dyn Error>> {
    let seq_len = env_usize("SCIRUST_M28_SEQ_LEN", DEFAULT_SEQ_LEN);
    let head_dim = env_usize("SCIRUST_M28_HEAD_DIM", DEFAULT_HEAD_DIM);
    let warmups = env_usize("SCIRUST_M28_WARMUPS", DEFAULT_WARMUPS);
    let repeats = env_usize("SCIRUST_M28_REPEATS", DEFAULT_REPEATS);
    if seq_len == 0 || head_dim == 0 || warmups == 0 || repeats == 0
    {
        return Err("seq_len, head_dim, warmups and repeats must be non-zero".into());
    }

    let ctx = WgpuContext::new()?;
    let pipeline = WgpuGroupedForwardPipeline::new(ctx.device())?;
    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads: 1,
        kv_heads: 1,
        seq_len,
        head_dim,
    };
    let layout = WgpuGroupedForwardPipeline::layout(shape)?;
    let elements = seq_len
        .checked_mul(head_dim)
        .ok_or("fixture length overflow")?;
    let q = fixture(elements, 0.2);
    let k = fixture(elements, 0.8);
    let v = fixture(elements, 1.4);

    let q_naive = ctx.upload(&q, seq_len, head_dim);
    let k_naive = ctx.upload(&k, seq_len, head_dim);
    let v_naive = ctx.upload(&v, seq_len, head_dim);
    let q_flat = flat_input_buffer(&ctx, &q, "scirust-m28-flat-q");
    let k_flat = flat_input_buffer(&ctx, &k, "scirust-m28-flat-k");
    let v_flat = flat_input_buffer(&ctx, &v, "scirust-m28-flat-v");
    let flat_inputs = FlatResidentInputs {
        q: &q_flat,
        k: &k_flat,
        v: &v_flat,
    };

    println!(
        "adapter,backend,causal,seq_len,head_dim,warmups,repeats,naive_median_us,naive_p95_us,flat_fresh_median_us,flat_fresh_p95_us,flat_reused_median_us,flat_reused_p95_us,naive_over_flat_fresh,naive_over_flat_reused,naive_parity_max_abs,flat_parity_max_abs,performance_claim"
    );

    for causal in [false, true]
    {
        let config = FlatAttentionConfig {
            causal,
            softmax_scale: None,
        };
        let expected = forward_reference_grouped(&q, &k, &v, shape, config)?;

        let naive = naive_attention(&ctx, &q_naive, &k_naive, &v_naive, causal)?;
        let naive_host = ctx.download(&naive)?;
        let naive_parity = assert_close("SciRust naive output", &naive_host, &expected.output)?;

        let reused_output = pipeline.create_output_buffer(ctx.device(), shape)?;
        run_flat(&ctx, &pipeline, flat_inputs, &reused_output, shape, config)?;
        let flat_host =
            ctx.download_buffer(&reused_output, layout.output_elements, layout.output_bytes)?;
        let flat_parity = assert_close(
            "FLAT output",
            &flat_host[..layout.q_elements],
            &expected.output,
        )?;
        assert_close("FLAT LSE", &flat_host[layout.lse_offset()..], &expected.lse)?;

        for _ in 0..warmups
        {
            let _ = time_naive(&ctx, &q_naive, &k_naive, &v_naive, causal)?;
            let _ = time_flat_fresh_output(&ctx, &pipeline, flat_inputs, shape, config)?;
            let _ = time_flat_reused(&ctx, &pipeline, flat_inputs, &reused_output, shape, config)?;
        }

        let mut naive_samples = Vec::with_capacity(repeats);
        let mut flat_fresh_samples = Vec::with_capacity(repeats);
        let mut flat_reused_samples = Vec::with_capacity(repeats);
        for iteration in 0..repeats
        {
            // Rotate all three measured paths through first/middle/last position. Over each
            // complete three-iteration cycle every path occupies every position once, so the
            // reused-output ratio is not structurally coupled to always running third.
            match iteration % 3
            {
                0 =>
                {
                    naive_samples.push(time_naive(&ctx, &q_naive, &k_naive, &v_naive, causal)?);
                    flat_fresh_samples.push(time_flat_fresh_output(
                        &ctx,
                        &pipeline,
                        flat_inputs,
                        shape,
                        config,
                    )?);
                    flat_reused_samples.push(time_flat_reused(
                        &ctx,
                        &pipeline,
                        flat_inputs,
                        &reused_output,
                        shape,
                        config,
                    )?);
                },
                1 =>
                {
                    flat_fresh_samples.push(time_flat_fresh_output(
                        &ctx,
                        &pipeline,
                        flat_inputs,
                        shape,
                        config,
                    )?);
                    flat_reused_samples.push(time_flat_reused(
                        &ctx,
                        &pipeline,
                        flat_inputs,
                        &reused_output,
                        shape,
                        config,
                    )?);
                    naive_samples.push(time_naive(&ctx, &q_naive, &k_naive, &v_naive, causal)?);
                },
                _ =>
                {
                    flat_reused_samples.push(time_flat_reused(
                        &ctx,
                        &pipeline,
                        flat_inputs,
                        &reused_output,
                        shape,
                        config,
                    )?);
                    naive_samples.push(time_naive(&ctx, &q_naive, &k_naive, &v_naive, causal)?);
                    flat_fresh_samples.push(time_flat_fresh_output(
                        &ctx,
                        &pipeline,
                        flat_inputs,
                        shape,
                        config,
                    )?);
                },
            }
        }

        let naive_median = median_ns(&naive_samples);
        let flat_fresh_median = median_ns(&flat_fresh_samples);
        let flat_reused_median = median_ns(&flat_reused_samples);
        let naive_over_flat_fresh = naive_median as f64 / flat_fresh_median.max(1) as f64;
        let naive_over_flat_reused = naive_median as f64 / flat_reused_median.max(1) as f64;

        println!(
            "{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.6},{:.6},{:.8},{:.8},none",
            ctx.adapter_name().replace(',', ";"),
            ctx.adapter_backend().replace(',', ";"),
            causal,
            seq_len,
            head_dim,
            warmups,
            repeats,
            naive_median as f64 / 1_000.0,
            percentile_ns(&naive_samples, 95) as f64 / 1_000.0,
            flat_fresh_median as f64 / 1_000.0,
            percentile_ns(&flat_fresh_samples, 95) as f64 / 1_000.0,
            flat_reused_median as f64 / 1_000.0,
            percentile_ns(&flat_reused_samples, 95) as f64 / 1_000.0,
            naive_over_flat_fresh,
            naive_over_flat_reused,
            max_abs_error(&naive_host, &expected.output).max(naive_parity),
            flat_parity,
        );
    }

    Ok(())
}
