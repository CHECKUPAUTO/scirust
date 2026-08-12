//! Evidence-only resident M28 candidate benchmark.
//!
//! Compiled by the Thor workflow in a temporary Cargo project against an exact
//! FLAT commit. The SciRust product dependency pin is intentionally untouched.

use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use flat_attention::api::wgpu::PreparedGroupedForward;
use flat_attention::{
    FlatAttentionConfig, GroupedAttentionShape, GroupedForwardPass, WgpuGroupedForwardPipeline,
    forward_reference_grouped,
};
use scirust_gpu::{GpuMatrix, WgpuContext};

const ATOL: f32 = 8.0e-4;
const RTOL: f32 = 3.0e-3;

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

fn storage(ctx: &WgpuContext, values: &[f32], label: &'static str) -> wgpu::Buffer {
    let bytes = bytes_f32(values);
    let buffer = ctx.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    ctx.queue().write_buffer(&buffer, 0, &bytes);
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

fn assert_close(name: &str, actual: &[f32], expected: &[f32]) -> Result<f32, Box<dyn Error>> {
    if actual.len() != expected.len()
    {
        return Err(format!("{name}: length {} != {}", actual.len(), expected.len()).into());
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

fn time_prepared(
    ctx: &WgpuContext,
    pipeline: &WgpuGroupedForwardPipeline,
    prepared: &PreparedGroupedForward,
) -> Duration {
    let start = Instant::now();
    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scirust-m28-candidate-prepared"),
        });
    let _ = pipeline.encode_prepared(&mut encoder, prepared);
    ctx.queue().submit(Some(encoder.finish()));
    let _ = ctx.device().poll(wgpu::Maintain::Wait);
    start.elapsed()
}

fn main() -> Result<(), Box<dyn Error>> {
    let seq_len = env_usize("SCIRUST_M28_SEQ_LEN", 128);
    let head_dim = env_usize("SCIRUST_M28_HEAD_DIM", 64);
    let warmups = env_usize("SCIRUST_M28_WARMUPS", 3);
    let repeats = env_usize("SCIRUST_M28_REPEATS", 12);
    if seq_len == 0 || !matches!(head_dim, 64 | 128) || warmups == 0 || repeats < 3
    {
        return Err("candidate requires seq_len>0, head_dim=64|128, warmups>0, repeats>=3".into());
    }

    let ctx = WgpuContext::new()?;
    let portable = WgpuGroupedForwardPipeline::new(ctx.device())?;
    let vec4 = WgpuGroupedForwardPipeline::with_vectorization(ctx.device(), true)?;
    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads: 1,
        kv_heads: 1,
        seq_len,
        head_dim,
    };
    let portable_variant = format!("{:?}", portable.kernel_variant_for_shape(shape));
    let vec4_variant = format!("{:?}", vec4.kernel_variant_for_shape(shape));
    if portable_variant != "Q4PortableGrouped"
    {
        return Err(
            format!("portable grouped kernel selection drifted: {portable_variant}").into(),
        );
    }
    if vec4_variant != "Q4Vec4Mha"
    {
        return Err(format!(
            "vec4 candidate was not selected for qualified MHA geometry: {vec4_variant}"
        )
        .into());
    }

    let layout = WgpuGroupedForwardPipeline::layout(shape)?;
    let elements = seq_len.checked_mul(head_dim).ok_or("fixture overflow")?;
    let q = fixture(elements, 0.2);
    let k = fixture(elements, 0.8);
    let v = fixture(elements, 1.4);
    let q_naive = ctx.upload(&q, seq_len, head_dim);
    let k_naive = ctx.upload(&k, seq_len, head_dim);
    let v_naive = ctx.upload(&v, seq_len, head_dim);
    let q_flat = storage(&ctx, &q, "scirust-m28-candidate-q");
    let k_flat = storage(&ctx, &k, "scirust-m28-candidate-k");
    let v_flat = storage(&ctx, &v, "scirust-m28-candidate-v");

    println!(
        "adapter,backend,causal,seq_len,head_dim,warmups,repeats,naive_median_us,naive_p95_us,portable_median_us,portable_p95_us,vec4_median_us,vec4_p95_us,naive_over_portable,naive_over_vec4,portable_over_vec4,naive_parity_max_abs,portable_parity_max_abs,vec4_parity_max_abs,candidate_variant,performance_claim"
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
        let naive_parity = assert_close("naive", &naive_host, &expected.output)?;

        let portable_output = portable.create_output_buffer(ctx.device(), shape)?;
        let vec4_output = vec4.create_output_buffer(ctx.device(), shape)?;
        let portable_prepared = portable.prepare(
            ctx.device(),
            GroupedForwardPass {
                q: &q_flat,
                k: &k_flat,
                v: &v_flat,
                output: &portable_output,
                shape,
                config,
            },
        )?;
        let vec4_prepared = vec4.prepare(
            ctx.device(),
            GroupedForwardPass {
                q: &q_flat,
                k: &k_flat,
                v: &v_flat,
                output: &vec4_output,
                shape,
                config,
            },
        )?;
        let prepared_variant = format!("{:?}", vec4_prepared.kernel_variant());
        if prepared_variant != "Q4Vec4Mha"
        {
            return Err(format!("prepared vec4 request selected {prepared_variant}").into());
        }

        let _ = time_prepared(&ctx, &portable, &portable_prepared);
        let _ = time_prepared(&ctx, &vec4, &vec4_prepared);
        let portable_host = ctx.download_buffer(
            &portable_output,
            layout.output_elements,
            layout.output_bytes,
        )?;
        let vec4_host =
            ctx.download_buffer(&vec4_output, layout.output_elements, layout.output_bytes)?;
        let portable_parity = assert_close(
            "portable O",
            &portable_host[..layout.q_elements],
            &expected.output,
        )?;
        assert_close(
            "portable LSE",
            &portable_host[layout.lse_offset()..],
            &expected.lse,
        )?;
        let vec4_parity =
            assert_close("vec4 O", &vec4_host[..layout.q_elements], &expected.output)?;
        assert_close("vec4 LSE", &vec4_host[layout.lse_offset()..], &expected.lse)?;

        for _ in 0..warmups
        {
            let _ = time_naive(&ctx, &q_naive, &k_naive, &v_naive, causal)?;
            black_box(time_prepared(&ctx, &portable, &portable_prepared));
            black_box(time_prepared(&ctx, &vec4, &vec4_prepared));
        }

        let mut naive_samples = Vec::with_capacity(repeats);
        let mut portable_samples = Vec::with_capacity(repeats);
        let mut vec4_samples = Vec::with_capacity(repeats);
        for iteration in 0..repeats
        {
            match iteration % 3
            {
                0 =>
                {
                    naive_samples.push(time_naive(&ctx, &q_naive, &k_naive, &v_naive, causal)?);
                    portable_samples.push(time_prepared(&ctx, &portable, &portable_prepared));
                    vec4_samples.push(time_prepared(&ctx, &vec4, &vec4_prepared));
                },
                1 =>
                {
                    portable_samples.push(time_prepared(&ctx, &portable, &portable_prepared));
                    vec4_samples.push(time_prepared(&ctx, &vec4, &vec4_prepared));
                    naive_samples.push(time_naive(&ctx, &q_naive, &k_naive, &v_naive, causal)?);
                },
                _ =>
                {
                    vec4_samples.push(time_prepared(&ctx, &vec4, &vec4_prepared));
                    naive_samples.push(time_naive(&ctx, &q_naive, &k_naive, &v_naive, causal)?);
                    portable_samples.push(time_prepared(&ctx, &portable, &portable_prepared));
                },
            }
        }

        let naive_median = median_ns(&naive_samples);
        let portable_median = median_ns(&portable_samples);
        let vec4_median = median_ns(&vec4_samples);
        println!(
            "{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.6},{:.6},{:.6},{:.8},{:.8},{:.8},{},none",
            ctx.adapter_name().replace(',', ";"),
            ctx.adapter_backend().replace(',', ";"),
            causal,
            seq_len,
            head_dim,
            warmups,
            repeats,
            naive_median as f64 / 1_000.0,
            percentile_ns(&naive_samples, 95) as f64 / 1_000.0,
            portable_median as f64 / 1_000.0,
            percentile_ns(&portable_samples, 95) as f64 / 1_000.0,
            vec4_median as f64 / 1_000.0,
            percentile_ns(&vec4_samples, 95) as f64 / 1_000.0,
            naive_median as f64 / portable_median.max(1) as f64,
            naive_median as f64 / vec4_median.max(1) as f64,
            portable_median as f64 / vec4_median.max(1) as f64,
            naive_parity,
            portable_parity,
            vec4_parity,
            prepared_variant,
        );
    }

    Ok(())
}
