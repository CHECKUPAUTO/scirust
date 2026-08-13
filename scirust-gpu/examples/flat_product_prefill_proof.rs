//! M50 product proof for SciAgent's resident GQA prefill attention.
//!
//! This compares the exact feature-off production composition
//! (`GpuChain::gqa_attention`) with the exact feature-on FLAT M32 bridge on one
//! WGPU context. Both paths include Q/K RoPE, use native GQA K/V cardinality,
//! keep upload and readback outside timing, and are correctness-gated before
//! any measurement is emitted.

use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use scirust_gpu::{FlatM11ResidentConfig, GpuChain, GpuMatrix, WgpuFlatM11Bridge};

const DEFAULT_Q_HEADS: usize = 8;
const DEFAULT_KV_HEADS: usize = 2;
const DEFAULT_SEQ_LEN: usize = 128;
const DEFAULT_HEAD_DIM: usize = 64;
const DEFAULT_WARMUPS: usize = 3;
const DEFAULT_REPEATS: usize = 12;
const THETA: f32 = 10_000.0;
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

#[allow(clippy::too_many_arguments)]
fn legacy_forward(
    chain: &GpuChain,
    q: &GpuMatrix,
    k: &GpuMatrix,
    v: &GpuMatrix,
    q_heads: usize,
    kv_heads: usize,
    seq_len: usize,
    causal: bool,
) -> Result<GpuMatrix, Box<dyn Error>> {
    Ok(chain.gqa_attention(q, k, v, q_heads, kv_heads, seq_len, THETA, causal)?)
}

fn flat_forward(
    bridge: &WgpuFlatM11Bridge,
    q: &GpuMatrix,
    k: &GpuMatrix,
    v: &GpuMatrix,
    config: FlatM11ResidentConfig,
) -> Result<GpuMatrix, Box<dyn Error>> {
    Ok(bridge.forward(q, k, v, config)?)
}

#[allow(clippy::too_many_arguments)]
fn time_legacy(
    chain: &GpuChain,
    bridge: &WgpuFlatM11Bridge,
    q: &GpuMatrix,
    k: &GpuMatrix,
    v: &GpuMatrix,
    q_heads: usize,
    kv_heads: usize,
    seq_len: usize,
    causal: bool,
) -> Result<Duration, Box<dyn Error>> {
    let start = Instant::now();
    let output = legacy_forward(chain, q, k, v, q_heads, kv_heads, seq_len, causal)?;
    let _ = bridge.context().device().poll(wgpu::Maintain::Wait);
    black_box(output);
    Ok(start.elapsed())
}

fn time_flat(
    bridge: &WgpuFlatM11Bridge,
    q: &GpuMatrix,
    k: &GpuMatrix,
    v: &GpuMatrix,
    config: FlatM11ResidentConfig,
) -> Result<Duration, Box<dyn Error>> {
    let start = Instant::now();
    let output = flat_forward(bridge, q, k, v, config)?;
    let _ = bridge.context().device().poll(wgpu::Maintain::Wait);
    black_box(output);
    Ok(start.elapsed())
}

fn main() -> Result<(), Box<dyn Error>> {
    let q_heads = env_usize("SCIRUST_M50_Q_HEADS", DEFAULT_Q_HEADS);
    let kv_heads = env_usize("SCIRUST_M50_KV_HEADS", DEFAULT_KV_HEADS);
    let seq_len = env_usize("SCIRUST_M50_SEQ_LEN", DEFAULT_SEQ_LEN);
    let head_dim = env_usize("SCIRUST_M50_HEAD_DIM", DEFAULT_HEAD_DIM);
    let warmups = env_usize("SCIRUST_M50_WARMUPS", DEFAULT_WARMUPS);
    let repeats = env_usize("SCIRUST_M50_REPEATS", DEFAULT_REPEATS);
    if q_heads == 0
        || kv_heads == 0
        || !q_heads.is_multiple_of(kv_heads)
        || seq_len == 0
        || head_dim == 0
        || !head_dim.is_multiple_of(2)
        || warmups == 0
        || repeats < 4
    {
        return Err("M50 requires divisible non-zero heads, seq_len>0, even head_dim, warmups>0 and repeats>=4".into());
    }

    let Some(chain) = GpuChain::new()
    else
    {
        return Err("WGPU adapter unavailable".into());
    };
    let bridge = chain.flat_m11_bridge()?;
    if chain.adapter_name() != bridge.adapter_name()
    {
        return Err("legacy and FLAT routes selected different adapters".into());
    }

    let q_width = q_heads.checked_mul(head_dim).ok_or("Q width overflow")?;
    let kv_width = kv_heads.checked_mul(head_dim).ok_or("K/V width overflow")?;
    let q = fixture(
        seq_len.checked_mul(q_width).ok_or("Q length overflow")?,
        0.2,
    );
    let k = fixture(
        seq_len.checked_mul(kv_width).ok_or("K length overflow")?,
        0.8,
    );
    let v = fixture(
        seq_len.checked_mul(kv_width).ok_or("V length overflow")?,
        1.4,
    );
    let q_gpu = chain.upload(&q, seq_len, q_width);
    let k_gpu = chain.upload(&k, seq_len, kv_width);
    let v_gpu = chain.upload(&v, seq_len, kv_width);

    // Fence setup before parity and timing; upload is outside the measured scope.
    let _ = bridge.context().device().poll(wgpu::Maintain::Wait);
    println!(
        "adapter,backend,causal,q_heads,kv_heads,seq_len,head_dim,warmups,repeats,legacy_median_us,legacy_p95_us,flat_median_us,flat_p95_us,legacy_over_flat,parity_max_abs,performance_claim"
    );

    for causal in [false, true]
    {
        let config = FlatM11ResidentConfig {
            batch: 1,
            q_heads,
            kv_heads,
            query_len: seq_len,
            kv_len: seq_len,
            head_dim,
            causal,
            softmax_scale: None,
            query_position_offset: 0,
            theta: THETA,
            query_rope_position_offset: 0,
            kv_rope_position_offset: 0,
        };
        let legacy = legacy_forward(
            &chain, &q_gpu, &k_gpu, &v_gpu, q_heads, kv_heads, seq_len, causal,
        )?;
        let flat = flat_forward(&bridge, &q_gpu, &k_gpu, &v_gpu, config)?;
        let legacy_host = chain.download(&legacy)?;
        let flat_host = chain.download(&flat)?;
        let parity = assert_close("FLAT versus legacy GQA prefill", &flat_host, &legacy_host)?;

        for iteration in 0..warmups
        {
            if iteration % 2 == 0
            {
                black_box(time_legacy(
                    &chain, &bridge, &q_gpu, &k_gpu, &v_gpu, q_heads, kv_heads, seq_len, causal,
                )?);
                black_box(time_flat(&bridge, &q_gpu, &k_gpu, &v_gpu, config)?);
            }
            else
            {
                black_box(time_flat(&bridge, &q_gpu, &k_gpu, &v_gpu, config)?);
                black_box(time_legacy(
                    &chain, &bridge, &q_gpu, &k_gpu, &v_gpu, q_heads, kv_heads, seq_len, causal,
                )?);
            }
        }

        let mut legacy_samples = Vec::with_capacity(repeats);
        let mut flat_samples = Vec::with_capacity(repeats);
        for iteration in 0..repeats
        {
            if iteration % 2 == 0
            {
                legacy_samples.push(time_legacy(
                    &chain, &bridge, &q_gpu, &k_gpu, &v_gpu, q_heads, kv_heads, seq_len, causal,
                )?);
                flat_samples.push(time_flat(&bridge, &q_gpu, &k_gpu, &v_gpu, config)?);
            }
            else
            {
                flat_samples.push(time_flat(&bridge, &q_gpu, &k_gpu, &v_gpu, config)?);
                legacy_samples.push(time_legacy(
                    &chain, &bridge, &q_gpu, &k_gpu, &v_gpu, q_heads, kv_heads, seq_len, causal,
                )?);
            }
        }

        let legacy_median = median_ns(&legacy_samples);
        let flat_median = median_ns(&flat_samples);
        println!(
            "{},{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.6},{:.8},none",
            bridge.adapter_name().replace(',', ";"),
            bridge.context().adapter_backend().replace(',', ";"),
            causal,
            q_heads,
            kv_heads,
            seq_len,
            head_dim,
            warmups,
            repeats,
            legacy_median as f64 / 1_000.0,
            percentile_ns(&legacy_samples, 95) as f64 / 1_000.0,
            flat_median as f64 / 1_000.0,
            percentile_ns(&flat_samples, 95) as f64 / 1_000.0,
            legacy_median as f64 / flat_median.max(1) as f64,
            parity,
        );
    }
    Ok(())
}
