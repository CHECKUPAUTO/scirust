//! Evidence-only resident GQA/MQA K/V-reuse candidate benchmark.
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
            label: Some("scirust-m47-gqa-kv-reuse-candidate"),
        });
    let _ = pipeline.encode_prepared(&mut encoder, prepared);
    ctx.queue().submit(Some(encoder.finish()));
    let _ = ctx.device().poll(wgpu::Maintain::Wait);
    start.elapsed()
}

fn main() -> Result<(), Box<dyn Error>> {
    let seq_len = env_usize("SCIRUST_M47_SEQ_LEN", 128);
    let head_dim = env_usize("SCIRUST_M47_HEAD_DIM", 64);
    let q_heads = env_usize("SCIRUST_M47_Q_HEADS", 8);
    let kv_heads = env_usize("SCIRUST_M47_KV_HEADS", 2);
    let warmups = env_usize("SCIRUST_M47_WARMUPS", 3);
    let repeats = env_usize("SCIRUST_M47_REPEATS", 12);
    if seq_len == 0
        || !matches!(head_dim, 64 | 128)
        || q_heads == 0
        || kv_heads == 0
        || q_heads == kv_heads
        || q_heads % kv_heads != 0
        || q_heads / kv_heads < 2
        || warmups == 0
        || repeats < 4
    {
        return Err(
            "candidate requires seq_len>0, D64|D128, native GQA/MQA grouping>=2, warmups>0, repeats>=4"
                .into(),
        );
    }

    let ctx = WgpuContext::new()?;
    let grouped_vec4 = WgpuGroupedForwardPipeline::with_grouped_vectorization(ctx.device(), true)?;
    let grouped_kv_reuse = WgpuGroupedForwardPipeline::with_grouped_kv_reuse(ctx.device(), true)?;
    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads,
        kv_heads,
        seq_len,
        head_dim,
    };
    let baseline_variant = format!("{:?}", grouped_vec4.kernel_variant_for_shape(shape));
    let candidate_variant = format!("{:?}", grouped_kv_reuse.kernel_variant_for_shape(shape));
    if baseline_variant != "Q4Vec4Grouped"
    {
        return Err(format!("M45 grouped vec4 selection drifted: {baseline_variant}").into());
    }
    if candidate_variant != "Q4Vec4GroupedKvReuse"
    {
        return Err(format!("M47 grouped K/V-reuse selection drifted: {candidate_variant}").into());
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
    let q_gpu = storage(&ctx, &q, "scirust-m47-q");
    let k_gpu = storage(&ctx, &k, "scirust-m47-k");
    let v_gpu = storage(&ctx, &v, "scirust-m47-v");
    let layout = WgpuGroupedForwardPipeline::layout(shape)?;

    println!(
        "adapter,backend,causal,q_heads,kv_heads,seq_len,head_dim,warmups,repeats,grouped_vec4_median_us,grouped_vec4_p95_us,kv_reuse_median_us,kv_reuse_p95_us,grouped_vec4_over_kv_reuse,grouped_vec4_parity_max_abs,kv_reuse_parity_max_abs,baseline_variant,candidate_variant,kv_elements,q_elements,performance_claim"
    );

    for causal in [false, true]
    {
        let config = FlatAttentionConfig {
            causal,
            softmax_scale: None,
        };
        let expected = forward_reference_grouped(&q, &k, &v, shape, config)?;
        let baseline_output = grouped_vec4.create_output_buffer(ctx.device(), shape)?;
        let candidate_output = grouped_kv_reuse.create_output_buffer(ctx.device(), shape)?;
        let baseline_prepared = grouped_vec4.prepare(
            ctx.device(),
            GroupedForwardPass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                output: &baseline_output,
                shape,
                config,
            },
        )?;
        let candidate_prepared = grouped_kv_reuse.prepare(
            ctx.device(),
            GroupedForwardPass {
                q: &q_gpu,
                k: &k_gpu,
                v: &v_gpu,
                output: &candidate_output,
                shape,
                config,
            },
        )?;
        if format!("{:?}", baseline_prepared.kernel_variant()) != "Q4Vec4Grouped"
        {
            return Err("prepared M45 baseline selected another kernel".into());
        }
        if format!("{:?}", candidate_prepared.kernel_variant()) != "Q4Vec4GroupedKvReuse"
        {
            return Err("prepared M47 candidate selected another kernel".into());
        }

        let _ = time_prepared(&ctx, &grouped_vec4, &baseline_prepared);
        let _ = time_prepared(&ctx, &grouped_kv_reuse, &candidate_prepared);
        let baseline_host = ctx.download_buffer(
            &baseline_output,
            layout.output_elements,
            layout.output_bytes,
        )?;
        let candidate_host = ctx.download_buffer(
            &candidate_output,
            layout.output_elements,
            layout.output_bytes,
        )?;
        let baseline_parity = assert_close(
            "M45 grouped vec4 O",
            &baseline_host[..layout.q_elements],
            &expected.output,
        )?;
        assert_close(
            "M45 grouped vec4 LSE",
            &baseline_host[layout.lse_offset()..],
            &expected.lse,
        )?;
        let candidate_parity = assert_close(
            "M47 grouped K/V-reuse O",
            &candidate_host[..layout.q_elements],
            &expected.output,
        )?;
        assert_close(
            "M47 grouped K/V-reuse LSE",
            &candidate_host[layout.lse_offset()..],
            &expected.lse,
        )?;

        for iteration in 0..warmups
        {
            if iteration % 2 == 0
            {
                black_box(time_prepared(&ctx, &grouped_vec4, &baseline_prepared));
                black_box(time_prepared(&ctx, &grouped_kv_reuse, &candidate_prepared));
            }
            else
            {
                black_box(time_prepared(&ctx, &grouped_kv_reuse, &candidate_prepared));
                black_box(time_prepared(&ctx, &grouped_vec4, &baseline_prepared));
            }
        }

        let mut baseline_samples = Vec::with_capacity(repeats);
        let mut candidate_samples = Vec::with_capacity(repeats);
        for iteration in 0..repeats
        {
            if iteration % 2 == 0
            {
                baseline_samples.push(time_prepared(&ctx, &grouped_vec4, &baseline_prepared));
                candidate_samples.push(time_prepared(&ctx, &grouped_kv_reuse, &candidate_prepared));
            }
            else
            {
                candidate_samples.push(time_prepared(&ctx, &grouped_kv_reuse, &candidate_prepared));
                baseline_samples.push(time_prepared(&ctx, &grouped_vec4, &baseline_prepared));
            }
        }

        let baseline_median = median_ns(&baseline_samples);
        let candidate_median = median_ns(&candidate_samples);
        println!(
            "{},{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.6},{:.8},{:.8},{},{},{},{},none",
            ctx.adapter_name().replace(',', ";"),
            ctx.adapter_backend().replace(',', ";"),
            causal,
            q_heads,
            kv_heads,
            seq_len,
            head_dim,
            warmups,
            repeats,
            baseline_median as f64 / 1_000.0,
            percentile_ns(&baseline_samples, 95) as f64 / 1_000.0,
            candidate_median as f64 / 1_000.0,
            percentile_ns(&candidate_samples, 95) as f64 / 1_000.0,
            baseline_median as f64 / candidate_median.max(1) as f64,
            baseline_parity,
            candidate_parity,
            baseline_variant,
            candidate_variant,
            layout.kv_elements,
            layout.q_elements,
        );
    }

    Ok(())
}
