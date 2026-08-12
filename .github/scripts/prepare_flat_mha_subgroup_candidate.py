#!/usr/bin/env python3
"""Materialize the FLAT grouped-MHA subgroup candidate for physical Thor evidence.

The committed SciRust product pin and WgpuContext policy remain unchanged. This
script is executed only inside the evidence workflow. It pins the exact FLAT PR
head, opts the temporary SciRust WGPU device into SUBGROUP when the adapter
supports it, and writes a temporary benchmark example comparing:

1. SciRust's previous resident multi-dispatch attention;
2. FLAT grouped Q4 with subgroup explicitly disabled;
3. FLAT grouped Q4 with subgroup explicitly required.
"""

from pathlib import Path

OLD_FLAT_REV = "24d3340edeb059e40e0fe0c400e814685701d855"
CANDIDATE_FLAT_REV = "a49ef227e3e90a60c4680400b97c6870f0b8e07f"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, got {count}")
    return text.replace(old, new, 1)


def patch_flat_pin() -> None:
    path = Path("scirust-gpu/Cargo.toml")
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        f'rev = "{OLD_FLAT_REV}"',
        f'rev = "{CANDIDATE_FLAT_REV}"',
        "FLAT git revision",
    )
    path.write_text(text, encoding="utf-8")


def patch_wgpu_features() -> None:
    path = Path("scirust-gpu/src/wgpu_backend.rs")
    text = path.read_text(encoding="utf-8")
    old = '''        let adapter_info = adapter.get_info();
        let adapter_name = adapter_info.name;
        let adapter_backend = format!("{:?}", adapter_info.backend);
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("scirust-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
'''
    new = '''        let adapter_info = adapter.get_info();
        let adapter_name = adapter_info.name;
        let adapter_backend = format!("{:?}", adapter_info.backend);
        let adapter_features = adapter.features();
        let required_features = if adapter_features.contains(wgpu::Features::SUBGROUP) {
            wgpu::Features::SUBGROUP
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("scirust-gpu"),
                required_features,
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
'''
    text = replace_once(text, old, new, "WGPU subgroup feature request")
    path.write_text(text, encoding="utf-8")


def write_benchmark() -> None:
    path = Path("scirust-gpu/examples/flat_mha_subgroup_candidate.rs")
    path.write_text(r'''//! Physical qualification benchmark for the FLAT grouped-MHA subgroup route.

use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use flat_attention::{
    api::wgpu::GroupedForwardKernelVariant, forward_reference_grouped, FlatAttentionConfig,
    GroupedAttentionShape, GroupedForwardPass, WgpuGroupedForwardPipeline, WgpuSubgroupPolicy,
};
use scirust_gpu::{GpuMatrix, WgpuContext};

const DEFAULT_WARMUPS: usize = 3;
const DEFAULT_REPEATS: usize = 12;
const DEFAULT_SEQ_LEN: usize = 128;
const DEFAULT_HEAD_DIM: usize = 64;
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
    for &value in values {
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
    if !bytes.is_empty() {
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

fn assert_close(name: &str, actual: &[f32], expected: &[f32]) -> Result<f32, Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err(format!("{name} length {} != {}", actual.len(), expected.len()).into());
    }
    let mut worst = 0.0f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let abs = (actual - expected).abs();
        let limit = ATOL + RTOL * actual.abs().max(expected.abs());
        if !actual.is_finite() || abs > limit {
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

fn main() -> Result<(), Box<dyn Error>> {
    let seq_len = env_usize("SCIRUST_M28_SEQ_LEN", DEFAULT_SEQ_LEN);
    let head_dim = env_usize("SCIRUST_M28_HEAD_DIM", DEFAULT_HEAD_DIM);
    let warmups = env_usize("SCIRUST_M28_WARMUPS", DEFAULT_WARMUPS);
    let repeats = env_usize("SCIRUST_M28_REPEATS", DEFAULT_REPEATS);
    if seq_len == 0 || head_dim == 0 || warmups == 0 || repeats == 0 {
        return Err("seq_len, head_dim, warmups and repeats must be non-zero".into());
    }

    let ctx = WgpuContext::new()?;
    if !ctx.device().features().contains(wgpu::Features::SUBGROUP) {
        return Err("physical candidate requires SUBGROUP enabled on the SciRust device".into());
    }

    let portable_pipeline = WgpuGroupedForwardPipeline::with_subgroup_policy(
        ctx.device(),
        WgpuSubgroupPolicy::Disable,
    )?;
    let subgroup_pipeline = WgpuGroupedForwardPipeline::with_subgroup_policy(
        ctx.device(),
        WgpuSubgroupPolicy::Require,
    )?;

    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads: 1,
        kv_heads: 1,
        seq_len,
        head_dim,
    };
    let layout = WgpuGroupedForwardPipeline::layout(shape)?;
    let elements = seq_len.checked_mul(head_dim).ok_or("fixture length overflow")?;
    let q = fixture(elements, 0.2);
    let k = fixture(elements, 0.8);
    let v = fixture(elements, 1.4);

    let q_naive = ctx.upload(&q, seq_len, head_dim);
    let k_naive = ctx.upload(&k, seq_len, head_dim);
    let v_naive = ctx.upload(&v, seq_len, head_dim);
    let q_flat = flat_input_buffer(&ctx, &q, "mha-subgroup-q");
    let k_flat = flat_input_buffer(&ctx, &k, "mha-subgroup-k");
    let v_flat = flat_input_buffer(&ctx, &v, "mha-subgroup-v");

    println!(
        "adapter,backend,causal,seq_len,head_dim,warmups,repeats,portable_variant,subgroup_variant,naive_median_us,naive_p95_us,portable_median_us,portable_p95_us,subgroup_median_us,subgroup_p95_us,naive_over_portable,naive_over_subgroup,portable_over_subgroup,naive_parity_max_abs,portable_parity_max_abs,subgroup_parity_max_abs,performance_claim"
    );

    for causal in [false, true] {
        let config = FlatAttentionConfig {
            causal,
            softmax_scale: None,
        };
        let expected = forward_reference_grouped(&q, &k, &v, shape, config)?;

        let naive = naive_attention(&ctx, &q_naive, &k_naive, &v_naive, causal)?;
        let naive_host = ctx.download(&naive)?;
        let naive_parity = assert_close("SciRust naive", &naive_host, &expected.output)?;

        let portable_output = portable_pipeline.create_output_buffer(ctx.device(), shape)?;
        let subgroup_output = subgroup_pipeline.create_output_buffer(ctx.device(), shape)?;
        let portable = portable_pipeline.prepare(
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
        let subgroup = subgroup_pipeline.prepare(
            ctx.device(),
            GroupedForwardPass {
                q: &q_flat,
                k: &k_flat,
                v: &v_flat,
                output: &subgroup_output,
                shape,
                config,
            },
        )?;
        if portable.kernel_variant() != GroupedForwardKernelVariant::Q4PortableGrouped {
            return Err("portable pipeline selected an unexpected kernel".into());
        }
        if subgroup.kernel_variant() != GroupedForwardKernelVariant::Q4SubgroupMha {
            return Err("required subgroup pipeline did not select Q4SubgroupMha".into());
        }

        let time_prepared = |pipeline: &WgpuGroupedForwardPipeline,
                             prepared: &flat_attention::api::wgpu::PreparedGroupedForward|
         -> Result<Duration, Box<dyn Error>> {
            let start = Instant::now();
            let mut encoder = ctx
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("flat-mha-subgroup-candidate"),
                });
            let _ = pipeline.encode_prepared(&mut encoder, prepared);
            ctx.queue().submit(Some(encoder.finish()));
            let _ = ctx.device().poll(wgpu::Maintain::Wait);
            Ok(start.elapsed())
        };

        let _ = time_prepared(&portable_pipeline, &portable)?;
        let portable_host = ctx.download_buffer(
            &portable_output,
            layout.output_elements,
            layout.output_bytes,
        )?;
        let portable_parity = assert_close(
            "FLAT portable output",
            &portable_host[..layout.q_elements],
            &expected.output,
        )?;
        assert_close(
            "FLAT portable LSE",
            &portable_host[layout.lse_offset()..],
            &expected.lse,
        )?;

        let _ = time_prepared(&subgroup_pipeline, &subgroup)?;
        let subgroup_host = ctx.download_buffer(
            &subgroup_output,
            layout.output_elements,
            layout.output_bytes,
        )?;
        let subgroup_parity = assert_close(
            "FLAT subgroup output",
            &subgroup_host[..layout.q_elements],
            &expected.output,
        )?;
        assert_close(
            "FLAT subgroup LSE",
            &subgroup_host[layout.lse_offset()..],
            &expected.lse,
        )?;

        for _ in 0..warmups {
            let _ = time_naive(&ctx, &q_naive, &k_naive, &v_naive, causal)?;
            let _ = time_prepared(&portable_pipeline, &portable)?;
            let _ = time_prepared(&subgroup_pipeline, &subgroup)?;
        }

        let mut naive_samples = Vec::with_capacity(repeats);
        let mut portable_samples = Vec::with_capacity(repeats);
        let mut subgroup_samples = Vec::with_capacity(repeats);
        for iteration in 0..repeats {
            match iteration % 3 {
                0 => {
                    naive_samples.push(time_naive(&ctx, &q_naive, &k_naive, &v_naive, causal)?);
                    portable_samples.push(time_prepared(&portable_pipeline, &portable)?);
                    subgroup_samples.push(time_prepared(&subgroup_pipeline, &subgroup)?);
                }
                1 => {
                    portable_samples.push(time_prepared(&portable_pipeline, &portable)?);
                    subgroup_samples.push(time_prepared(&subgroup_pipeline, &subgroup)?);
                    naive_samples.push(time_naive(&ctx, &q_naive, &k_naive, &v_naive, causal)?);
                }
                _ => {
                    subgroup_samples.push(time_prepared(&subgroup_pipeline, &subgroup)?);
                    naive_samples.push(time_naive(&ctx, &q_naive, &k_naive, &v_naive, causal)?);
                    portable_samples.push(time_prepared(&portable_pipeline, &portable)?);
                }
            }
        }

        let naive_median = median_ns(&naive_samples);
        let portable_median = median_ns(&portable_samples);
        let subgroup_median = median_ns(&subgroup_samples);
        println!(
            "{},{},{},{},{},{},{},{:?},{:?},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.6},{:.6},{:.6},{:.8},{:.8},{:.8},none",
            ctx.adapter_name().replace(',', ";"),
            ctx.adapter_backend().replace(',', ";"),
            causal,
            seq_len,
            head_dim,
            warmups,
            repeats,
            portable.kernel_variant(),
            subgroup.kernel_variant(),
            naive_median as f64 / 1_000.0,
            percentile_ns(&naive_samples, 95) as f64 / 1_000.0,
            portable_median as f64 / 1_000.0,
            percentile_ns(&portable_samples, 95) as f64 / 1_000.0,
            subgroup_median as f64 / 1_000.0,
            percentile_ns(&subgroup_samples, 95) as f64 / 1_000.0,
            naive_median as f64 / portable_median.max(1) as f64,
            naive_median as f64 / subgroup_median.max(1) as f64,
            portable_median as f64 / subgroup_median.max(1) as f64,
            naive_parity,
            portable_parity,
            subgroup_parity,
        );
    }

    Ok(())
}
''', encoding="utf-8")


if __name__ == "__main__":
    patch_flat_pin()
    patch_wgpu_features()
    write_benchmark()
    print(f"candidate_flat_revision={CANDIDATE_FLAT_REV}")
