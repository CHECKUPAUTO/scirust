//! Evidence-only resident native-GQA/MQA candidate benchmark.
//!
//! This is compiled by a physical-Thor workflow in a temporary Cargo project
//! against one exact FLAT revision. It does not change SciRust's product FLAT
//! dependency pin.

use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use flat_attention::api::wgpu::PreparedGroupedForward;
use flat_attention::{
    FlatAttentionConfig, GroupedAttentionShape, GroupedForwardPass, WgpuGroupedForwardPipeline,
    forward_reference_grouped,
};
use scirust_gpu::WgpuContext;

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

fn time_prepared(
    ctx: &WgpuContext,
    pipeline: &WgpuGroupedForwardPipeline,
    prepared: &PreparedGroupedForward,
) -> Duration {
    let start = Instant::now();
    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scirust-m45-grouped-vec4-candidate"),
        });
    let _ = pipeline.encode_prepared(&mut encoder, prepared);
    ctx.queue().submit(Some(encoder.finish()));
    let _ = ctx.device().poll(wgpu::Maintain::Wait);
    start.elapsed()
}

fn main() -> Result<(), Box<dyn Error>> {
    let seq_len = env_usize("SCIRUST_M45_SEQ_LEN", 128);
    let head_dim = env_usize("SCIRUST_M45_HEAD_DIM", 64);
    let q_heads = env_usize("SCIRUST_M45_Q_HEADS", 8);
    let kv_heads = env_usize("SCIRUST_M45_KV_HEADS", 2);
    let warmups = env_usize("SCIRUST_M45_WARMUPS", 3);
    let repeats = env_usize("SCIRUST_M45_REPEATS", 12);
    if seq_len == 0
        || !matches!(head_dim, 64 | 128)
        || q_heads == 0
        || kv_heads == 0
        || q_heads == kv_heads
        || q_heads % kv_heads != 0
        || warmups == 0
        || repeats < 4
    {
        return Err(
            "candidate requires seq_len>0, D64|D128, native GQA/MQA grouping, warmups>0, repeats>=4"
                .into(),
        );
    }

    let ctx = WgpuContext::new()?;
    let portable = WgpuGroupedForwardPipeline::new(ctx.device())?;
    let grouped_vec4 = WgpuGroupedForwardPipeline::with_grouped_vectorization(ctx.device(), true)?;
    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads,
        kv_heads,
        seq_len,
        head_dim,
    };
    let portable_variant = format!("{:?}", portable.kernel_variant_for_shape(shape));
    let grouped_variant = format!("{:?}", grouped_vec4.kernel_variant_for_shape(shape));
    if portable_variant != "Q4PortableGrouped"
    {
        return Err(format!("portable grouped selection drifted: {portable_variant}").into());
    }
    if grouped_variant != "Q4Vec4Grouped"
    {
        return Err(format!("native grouped vec4 selection drifted: {grouped_variant}").into());
    }

    let q_len = shape.q_tensor_len()?;
    let kv_len = shape.kv_tensor_len()?;
    if kv_len >= q_len
    {
        return Err("native grouped candidate unexpectedly expands K/V cardinality".into());
    }
    let q = fixture(q_len, 0.2);
    let k = fixture(kv_len, 0.8);
    let v = fixture(kv_len, 1.4);
    let q_gpu = storage(&ctx, &q, "scirust-m45-q");
    let k_gpu = storage(&ctx, &k, "scirust-m45-k");
    let v_gpu = storage(&ctx, &v, "scirust-m45-v");
    let layout = WgpuGroupedForwardPipeline::layout(shape)?;

    println!(
        "adapter,backend,causal,q_heads,kv_heads,seq_len,head_dim,warmups,repeats,portable_median_us,portable_p95_us,grouped_vec4_median_us,grouped_vec4_p95_us,portable_over_grouped_vec4,portable_parity_max_abs,grouped_vec4_parity_max_abs,candidate_variant,kv_elements,q_elements,performance_claim"
    );

    for causal in [false, true]
    {
        let config = FlatAttentionConfig {
            causal,
            softmax_scale: None,
        };
        let expected = forward_reference_grouped(&q, &k, &v, shape, config)?;
        let portable_output = portable.create_output_buffer(ctx.device(), shape)?;
        let grouped_output = grouped_vec4.create_output_buffer(ctx.device(), shape)?;
        let portable_prepared = portable.prepare(
            ctx.device(),
            GroupedForwardPass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                output: &portable_output,
                shape,
                config,
            },
        )?;
        let grouped_prepared = grouped_vec4.prepare(
            ctx.device(),
            GroupedForwardPass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                output: &grouped_output,
                shape,
                config,
            },
        )?;
        if format!("{:?}", grouped_prepared.kernel_variant()) != "Q4Vec4Grouped"
        {
            return Err("prepared native grouped vec4 request selected another kernel".into());
        }

        let _ = time_prepared(&ctx, &portable, &portable_prepared);
        let _ = time_prepared(&ctx, &grouped_vec4, &grouped_prepared);
        let portable_host = ctx.download_buffer(
            &portable_output,
            layout.output_elements,
            layout.output_bytes,
        )?;
        let grouped_host =
            ctx.download_buffer(&grouped_output, layout.output_elements, layout.output_bytes)?;
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
        let grouped_parity = assert_close(
            "grouped vec4 O",
            &grouped_host[..layout.q_elements],
            &expected.output,
        )?;
        assert_close(
            "grouped vec4 LSE",
            &grouped_host[layout.lse_offset()..],
            &expected.lse,
        )?;

        for _ in 0..warmups
        {
            black_box(time_prepared(&ctx, &portable, &portable_prepared));
            black_box(time_prepared(&ctx, &grouped_vec4, &grouped_prepared));
        }

        let mut portable_samples = Vec::with_capacity(repeats);
        let mut grouped_samples = Vec::with_capacity(repeats);
        for iteration in 0..repeats
        {
            if iteration % 2 == 0
            {
                portable_samples.push(time_prepared(&ctx, &portable, &portable_prepared));
                grouped_samples.push(time_prepared(&ctx, &grouped_vec4, &grouped_prepared));
            }
            else
            {
                grouped_samples.push(time_prepared(&ctx, &grouped_vec4, &grouped_prepared));
                portable_samples.push(time_prepared(&ctx, &portable, &portable_prepared));
            }
        }

        let portable_median = median_ns(&portable_samples);
        let grouped_median = median_ns(&grouped_samples);
        println!(
            "{},{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.6},{:.8},{:.8},{},{},{},none",
            ctx.adapter_name().replace(',', ";"),
            ctx.adapter_backend().replace(',', ";"),
            causal,
            q_heads,
            kv_heads,
            seq_len,
            head_dim,
            warmups,
            repeats,
            portable_median as f64 / 1_000.0,
            percentile_ns(&portable_samples, 95) as f64 / 1_000.0,
            grouped_median as f64 / 1_000.0,
            percentile_ns(&grouped_samples, 95) as f64 / 1_000.0,
            portable_median as f64 / grouped_median.max(1) as f64,
            portable_parity,
            grouped_parity,
            grouped_variant,
            layout.kv_elements,
            layout.q_elements,
        );
    }

    Ok(())
}
