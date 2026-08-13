//! M53 product-attention qualification for the asymmetric vec4 candidate.
//!
//! The legacy resident GQA composition, current portable FLAT bridge, and M53
//! vec4 bridge share one WGPU context and the same resident Q/K/V buffers.
//! Upload and readback are excluded from timing. Every row is parity-gated.

use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use scirust_gpu::{FlatM11ResidentConfig, GpuChain, GpuMatrix, WgpuFlatM11Bridge};

const THETA: f32 = 10_000.0;
const ATOL: f32 = 8.0e-4;
const RTOL: f32 = 3.0e-3;

#[derive(Clone, Copy)]
enum Route {
    Legacy,
    Portable,
    Vec4,
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

fn percentile_ns(samples: &[Duration], percentile: usize) -> u128 {
    let mut values: Vec<u128> = samples.iter().map(Duration::as_nanos).collect();
    values.sort_unstable();
    let rank = percentile.saturating_mul(values.len()).div_ceil(100).max(1);
    values[rank - 1]
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
fn forward(
    route: Route,
    chain: &GpuChain,
    portable: &WgpuFlatM11Bridge,
    vec4: &WgpuFlatM11Bridge,
    q: &GpuMatrix,
    k: &GpuMatrix,
    v: &GpuMatrix,
    config: FlatM11ResidentConfig,
) -> Result<GpuMatrix, Box<dyn Error>> {
    match route
    {
        Route::Legacy => Ok(chain.gqa_attention(
            q,
            k,
            v,
            config.q_heads,
            config.kv_heads,
            config.query_len,
            config.theta,
            config.causal,
        )?),
        Route::Portable => Ok(portable.forward(q, k, v, config)?),
        Route::Vec4 => Ok(vec4.forward(q, k, v, config)?),
    }
}

#[allow(clippy::too_many_arguments)]
fn timed(
    route: Route,
    chain: &GpuChain,
    portable: &WgpuFlatM11Bridge,
    vec4: &WgpuFlatM11Bridge,
    q: &GpuMatrix,
    k: &GpuMatrix,
    v: &GpuMatrix,
    config: FlatM11ResidentConfig,
) -> Result<Duration, Box<dyn Error>> {
    let started = Instant::now();
    let output = forward(route, chain, portable, vec4, q, k, v, config)?;
    let _ = portable.context().device().poll(wgpu::Maintain::Wait);
    black_box(output);
    Ok(started.elapsed())
}

fn main() -> Result<(), Box<dyn Error>> {
    let q_heads = env_usize("SCIRUST_M53_Q_HEADS", 8);
    let kv_heads = env_usize("SCIRUST_M53_KV_HEADS", 2);
    let seq_len = env_usize("SCIRUST_M53_SEQ_LEN", 128);
    let head_dim = env_usize("SCIRUST_M53_HEAD_DIM", 64);
    let warmups = env_usize("SCIRUST_M53_WARMUPS", 3);
    let repeats = env_usize("SCIRUST_M53_REPEATS", 12);
    if q_heads == 0
        || kv_heads == 0
        || !q_heads.is_multiple_of(kv_heads)
        || seq_len == 0
        || !matches!(head_dim, 64 | 128)
        || warmups == 0
        || repeats < 4
    {
        return Err(
            "M53 requires divisible non-zero heads, seq_len>0, D64/D128, warmups>0 and repeats>=4"
                .into(),
        );
    }

    let Some(chain) = GpuChain::new()
    else
    {
        return Err("WGPU adapter unavailable".into());
    };
    let portable = chain.flat_m11_bridge()?;
    let vec4 = chain.flat_m11_bridge_with_vectorization(true)?;
    if portable.vectorization_enabled() || !vec4.vectorization_enabled()
    {
        return Err("M53 bridge selection contract failed".into());
    }
    if chain.adapter_name() != portable.adapter_name()
        || chain.adapter_name() != vec4.adapter_name()
    {
        return Err("M53 routes selected different adapters".into());
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
    let _ = portable.context().device().poll(wgpu::Maintain::Wait);

    println!(
        "adapter,backend,causal,q_heads,kv_heads,seq_len,head_dim,warmups,repeats,legacy_median_us,portable_median_us,vec4_median_us,portable_p95_us,vec4_p95_us,legacy_over_vec4,portable_over_vec4,portable_legacy_max_abs,vec4_portable_max_abs,performance_claim"
    );
    let routes = [Route::Legacy, Route::Portable, Route::Vec4];
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

        let legacy = forward(
            Route::Legacy,
            &chain,
            &portable,
            &vec4,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            config,
        )?;
        let portable_output = forward(
            Route::Portable,
            &chain,
            &portable,
            &vec4,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            config,
        )?;
        let vec4_output = forward(
            Route::Vec4,
            &chain,
            &portable,
            &vec4,
            &q_gpu,
            &k_gpu,
            &v_gpu,
            config,
        )?;
        let legacy_host = chain.download(&legacy)?;
        let portable_host = chain.download(&portable_output)?;
        let vec4_host = chain.download(&vec4_output)?;
        let portable_legacy = assert_close("portable versus legacy", &portable_host, &legacy_host)?;
        let vec4_portable = assert_close("vec4 versus portable", &vec4_host, &portable_host)?;

        for iteration in 0..warmups
        {
            for offset in 0..routes.len()
            {
                let route = routes[(iteration + offset) % routes.len()];
                black_box(timed(
                    route, &chain, &portable, &vec4, &q_gpu, &k_gpu, &v_gpu, config,
                )?);
            }
        }

        let mut samples: [Vec<Duration>; 3] = std::array::from_fn(|_| Vec::with_capacity(repeats));
        for iteration in 0..repeats
        {
            for offset in 0..routes.len()
            {
                let index = (iteration + offset) % routes.len();
                samples[index].push(timed(
                    routes[index],
                    &chain,
                    &portable,
                    &vec4,
                    &q_gpu,
                    &k_gpu,
                    &v_gpu,
                    config,
                )?);
            }
        }
        let medians = samples.each_ref().map(|values| percentile_ns(values, 50));
        println!(
            "{},{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.6},{:.6},{:.8},{:.8},none",
            portable.adapter_name().replace(',', ";"),
            portable.context().adapter_backend().replace(',', ";"),
            causal,
            q_heads,
            kv_heads,
            seq_len,
            head_dim,
            warmups,
            repeats,
            medians[0] as f64 / 1_000.0,
            medians[1] as f64 / 1_000.0,
            medians[2] as f64 / 1_000.0,
            percentile_ns(&samples[1], 95) as f64 / 1_000.0,
            percentile_ns(&samples[2], 95) as f64 / 1_000.0,
            medians[0] as f64 / medians[2].max(1) as f64,
            medians[1] as f64 / medians[2].max(1) as f64,
            portable_legacy,
            vec4_portable,
        );
    }
    Ok(())
}
