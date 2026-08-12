#!/usr/bin/env python3
"""Materialize FLAT's existing MHA kernel generations for physical Thor ranking.

The committed SciRust FLAT pin remains unchanged. The physical evidence workflow
uses this script to pin the exact FLAT default-branch revision and create an
uncommitted benchmark example comparing Q4 portable, vec4, double-buffered and
Auto generations through FLAT's public API.
"""

from pathlib import Path

OLD_FLAT_REV = "24d3340edeb059e40e0fe0c400e814685701d855"
CANDIDATE_FLAT_REV = "f8e1be6a073596b446d298910eb04c122c7e29b0"


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


def write_benchmark() -> None:
    Path("scirust-gpu/examples/flat_kernel_generations_candidate.rs").write_text(
        r'''use std::{cmp::Ordering, error::Error, hint::black_box, time::Instant};

use flat_attention::{
    forward_reference_grouped, AttentionShape, FlatAttentionConfig, GroupedAttentionShape,
    WgpuFlatAttention, WgpuKernelVariant, WgpuSubgroupPolicy,
};

const DEFAULT_WARMUP: usize = 3;
const DEFAULT_ITERATIONS: usize = 12;
const DEFAULT_SEQ_LEN: usize = 128;
const DEFAULT_HEAD_DIM: usize = 64;
const ATOL: f32 = 7.0e-4;
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
            let x = index as f32 * 0.041 + phase;
            x.sin() * 0.68 + (x * 0.33).cos() * 0.32
        })
        .collect()
}

fn summarize(mut samples_us: Vec<f64>) -> (f64, f64) {
    samples_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median = if samples_us.len() % 2 == 0 {
        let upper = samples_us.len() / 2;
        (samples_us[upper - 1] + samples_us[upper]) * 0.5
    } else {
        samples_us[samples_us.len() / 2]
    };
    let p95_index = ((samples_us.len() * 95).div_ceil(100)).saturating_sub(1);
    (median, samples_us[p95_index])
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32]) -> Result<f32, Box<dyn Error>> {
    if actual.len() != expected.len() {
        return Err(format!("{name}: length mismatch").into());
    }
    let mut worst = 0.0f32;
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        let tolerance = ATOL + RTOL * actual.abs().max(expected.abs());
        if !actual.is_finite() || error > tolerance {
            return Err(format!(
                "{name}[{index}] actual={actual} expected={expected} abs={error} tolerance={tolerance}"
            )
            .into());
        }
        worst = worst.max(error);
    }
    Ok(worst)
}

fn candidate(
    name: &'static str,
) -> Result<WgpuFlatAttention, flat_attention::WgpuFlatAttentionError> {
    match name {
        "q4_portable" => WgpuFlatAttention::with_subgroup_policy_and_vectorization(
            WgpuSubgroupPolicy::Disable,
            false,
        ),
        "q4_vec4_portable" => WgpuFlatAttention::with_subgroup_policy_and_vectorization(
            WgpuSubgroupPolicy::Disable,
            true,
        ),
        "q4_vec4_double_buffered" => {
            WgpuFlatAttention::with_subgroup_vectorization_and_double_buffering(
                WgpuSubgroupPolicy::Disable,
                true,
                true,
            )
        }
        "auto" => WgpuFlatAttention::new(),
        _ => unreachable!(),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let warmup = env_usize("FLAT_M28_GENERATIONS_WARMUP", DEFAULT_WARMUP);
    let iterations = env_usize("FLAT_M28_GENERATIONS_ITERATIONS", DEFAULT_ITERATIONS);
    let seq_len = env_usize("FLAT_M28_GENERATIONS_SEQ_LEN", DEFAULT_SEQ_LEN);
    let head_dim = env_usize("FLAT_M28_GENERATIONS_HEAD_DIM", DEFAULT_HEAD_DIM);
    if warmup == 0 || iterations == 0 || seq_len == 0 || head_dim == 0 {
        return Err("benchmark dimensions and sample counts must be non-zero".into());
    }

    let shape = AttentionShape {
        batch: 1,
        heads: 1,
        seq_len,
        head_dim,
    };
    let grouped_shape = GroupedAttentionShape {
        batch: 1,
        q_heads: 1,
        kv_heads: 1,
        seq_len,
        head_dim,
    };
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let len = shape.tensor_len()?;
    let q = fixture(len, 0.2);
    let k = fixture(len, 0.8);
    let v = fixture(len, 1.4);
    let expected = forward_reference_grouped(&q, &k, &v, grouped_shape, config)?;

    println!("candidate,selected_variant,adapter,backend,driver,driver_info,seq_len,head_dim,warmup,iterations,median_us,p95_us,parity_o_max_abs,parity_lse_max_abs,performance_claim");

    for name in [
        "q4_portable",
        "q4_vec4_portable",
        "q4_vec4_double_buffered",
        "auto",
    ] {
        let attention = candidate(name)?;
        let selected = attention.kernel_variant_for_head_dim(head_dim);
        match name {
            "q4_portable" => assert_eq!(selected, WgpuKernelVariant::Q4Portable),
            "q4_vec4_portable" => assert_eq!(selected, WgpuKernelVariant::Q4Vec4Portable),
            "q4_vec4_double_buffered" => {
                assert_eq!(selected, WgpuKernelVariant::Q4Vec4DoubleBuffered)
            }
            _ => {}
        }

        let telemetry = attention.runtime_telemetry(shape)?;
        if telemetry.device.backend != "Vulkan" {
            return Err(format!(
                "{name}: expected Vulkan backend on Thor, got {}",
                telemetry.device.backend
            )
            .into());
        }
        if !telemetry.device.name.contains("NVIDIA Tegra NVIDIA Thor") {
            return Err(format!(
                "{name}: expected physical Thor adapter, got {}",
                telemetry.device.name
            )
            .into());
        }

        let actual = attention.forward(&q, &k, &v, shape, config)?;
        let parity_o = assert_close("O", &actual.output, &expected.output)?;
        let parity_lse = assert_close("LSE", &actual.lse, &expected.lse)?;

        for _ in 0..warmup {
            black_box(attention.forward(&q, &k, &v, shape, config)?);
        }
        let mut samples = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            black_box(attention.forward(&q, &k, &v, shape, config)?);
            samples.push(start.elapsed().as_secs_f64() * 1.0e6);
        }
        let (median_us, p95_us) = summarize(samples);
        println!(
            "{name},{selected:?},{},{},{},{},{seq_len},{head_dim},{warmup},{iterations},{median_us:.3},{p95_us:.3},{parity_o:.8},{parity_lse:.8},none",
            telemetry.device.name.replace(',', ";"),
            telemetry.device.backend.replace(',', ";"),
            telemetry.device.driver.replace(',', ";"),
            telemetry.device.driver_info.replace(',', ";"),
        );
    }

    Ok(())
}
''',
        encoding="utf-8",
    )


if __name__ == "__main__":
    patch_flat_pin()
    write_benchmark()
    print(f"candidate_flat_revision={CANDIDATE_FLAT_REV}")
