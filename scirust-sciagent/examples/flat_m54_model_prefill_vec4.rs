//! M54 full-model SCIAGENT prefill qualification for the M53 vec4 candidate.
//!
//! The portable and vec4 FLAT attention bridges share one `GpuChain`, WGPU
//! device/queue, resident weights, prompt and benchmark protocol. Each timed run
//! receives fresh fixed-capacity K/V caches allocated before the timing window.
//! The timing then mirrors `ResidentModel::prefill`: embedding, all 24 GQA+MLP
//! blocks, K/V cache seeding, final RMSNorm, tied LM head and the product-visible
//! logits readback. No production route is changed by this harness.

use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use scirust_core::autodiff::reverse::Tensor;
use scirust_gpu::{
    FlatM11ResidentConfig, GpuChain, GpuMatrix, WgpuDenseKvCache, WgpuFlatM11Bridge,
};
use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::model::SciAgentModel;

#[derive(Clone, Copy)]
enum Route {
    Portable,
    Vec4,
}

struct UploadedBlock {
    norm1: GpuMatrix,
    wq: GpuMatrix,
    wk: GpuMatrix,
    wv: GpuMatrix,
    wo: GpuMatrix,
    norm2: GpuMatrix,
    wg: GpuMatrix,
    wu: GpuMatrix,
    wd: GpuMatrix,
}

struct UploadedModel {
    embedding: GpuMatrix,
    final_norm: GpuMatrix,
    blocks: Vec<UploadedBlock>,
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn synthetic_tokens(len: usize, vocab: usize) -> Vec<u32> {
    (0..len)
        .map(|index| {
            let mixed = (index as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (mixed % vocab as u64) as u32
        })
        .collect()
}

fn upload_tensor(chain: &GpuChain, tensor: &Tensor) -> GpuMatrix {
    chain.upload(&tensor.data, tensor.rows, tensor.cols)
}

fn upload_model(chain: &GpuChain, model: &SciAgentModel) -> UploadedModel {
    let embedding = upload_tensor(chain, &model.embed.weight);
    let final_norm = upload_tensor(chain, &model.rms_final.weight);
    let blocks = model
        .layers
        .iter()
        .map(|layer| UploadedBlock {
            norm1: upload_tensor(chain, &layer.rms_attn.weight),
            wq: upload_tensor(chain, &layer.attn.w_q.weight),
            wk: upload_tensor(chain, &layer.attn.w_k.weight),
            wv: upload_tensor(chain, &layer.attn.w_v.weight),
            wo: upload_tensor(chain, &layer.attn.w_o.weight),
            norm2: upload_tensor(chain, &layer.rms_ffn.weight),
            wg: upload_tensor(chain, &layer.ffn.gate.weight),
            wu: upload_tensor(chain, &layer.ffn.up.weight),
            wd: upload_tensor(chain, &layer.ffn.down.weight),
        })
        .collect();
    UploadedModel {
        embedding,
        final_norm,
        blocks,
    }
}

fn fresh_caches(
    chain: &GpuChain,
    layers: usize,
    capacity: usize,
    kv_dim: usize,
) -> Result<(Vec<WgpuDenseKvCache>, Vec<WgpuDenseKvCache>), Box<dyn Error>> {
    let mut kcache = Vec::with_capacity(layers);
    let mut vcache = Vec::with_capacity(layers);
    for _ in 0..layers
    {
        kcache.push(chain.dense_kv_cache(capacity, kv_dim)?);
        vcache.push(chain.dense_kv_cache(capacity, kv_dim)?);
    }
    Ok((kcache, vcache))
}

#[allow(clippy::too_many_arguments)]
fn prefill(
    route: Route,
    chain: &GpuChain,
    portable: &WgpuFlatM11Bridge,
    vec4: &WgpuFlatM11Bridge,
    weights: &UploadedModel,
    config: &SciAgentConfig,
    prompt: &[u32],
    kcache: &mut [WgpuDenseKvCache],
    vcache: &mut [WgpuDenseKvCache],
) -> Result<Vec<f32>, Box<dyn Error>> {
    if prompt.is_empty()
    {
        return Err("M54 requires a non-empty prompt".into());
    }
    if kcache.len() != weights.blocks.len() || vcache.len() != weights.blocks.len()
    {
        return Err("M54 cache/layer count mismatch".into());
    }

    let p = prompt.len();
    let d_head = config.d_model / config.n_heads;
    let mut x = chain.embed(prompt, &weights.embedding)?;

    for (layer_index, block) in weights.blocks.iter().enumerate()
    {
        let xn = chain.rms_norm(&x, &block.norm1, config.eps)?;
        let q = chain.matmul(&xn, &block.wq)?;
        let k = chain.matmul(&xn, &block.wk)?;
        let v = chain.matmul(&xn, &block.wv)?;
        let flat_config = FlatM11ResidentConfig {
            batch: 1,
            q_heads: config.n_heads,
            kv_heads: config.n_kv_heads,
            query_len: p,
            kv_len: p,
            head_dim: d_head,
            causal: true,
            softmax_scale: None,
            query_position_offset: 0,
            theta: config.rope_theta,
            query_rope_position_offset: 0,
            kv_rope_position_offset: 0,
        };
        let ctx = match route
        {
            Route::Portable => portable.forward(&q, &k, &v, flat_config)?,
            Route::Vec4 => vec4.forward(&q, &k, &v, flat_config)?,
        };

        // Match ResidentModel::prefill cache seeding exactly: K is stored after
        // head-local RoPE, while V remains raw.
        let kr = chain.rope_heads(&k, config.n_kv_heads, p, 0, config.rope_theta)?;
        chain.dense_kv_append(&mut kcache[layer_index], &kr)?;
        chain.dense_kv_append(&mut vcache[layer_index], &v)?;

        let attn_out = chain.matmul(&ctx, &block.wo)?;
        x = chain.add(&x, &attn_out)?;
        let hn = chain.rms_norm(&x, &block.norm2, config.eps)?;
        let mlp = chain.swiglu_mlp(&hn, &block.wg, &block.wu, &block.wd)?;
        x = chain.add(&x, &mlp)?;
    }

    let normed = chain.rms_norm(&x, &weights.final_norm, config.eps)?;
    let logits = chain.matmul_t(&normed, &weights.embedding, false, true)?;
    let all = chain.download(&logits)?;
    Ok(all[(p - 1) * config.vocab_size..p * config.vocab_size].to_vec())
}

#[allow(clippy::too_many_arguments)]
fn timed(
    route: Route,
    chain: &GpuChain,
    portable: &WgpuFlatM11Bridge,
    vec4: &WgpuFlatM11Bridge,
    weights: &UploadedModel,
    config: &SciAgentConfig,
    prompt: &[u32],
) -> Result<(Duration, Vec<f32>), Box<dyn Error>> {
    let d_head = config.d_model / config.n_heads;
    let kv_dim = config.n_kv_heads * d_head;
    // Product allocation happens before ResidentModel::prefill, so keep fixed
    // cache allocation outside the M54 timing window as well.
    let (mut kcache, mut vcache) = fresh_caches(chain, weights.blocks.len(), prompt.len(), kv_dim)?;
    let started = Instant::now();
    let logits = prefill(
        route,
        chain,
        portable,
        vec4,
        weights,
        config,
        prompt,
        &mut kcache,
        &mut vcache,
    )?;
    let elapsed = started.elapsed();
    black_box(&logits);
    Ok((elapsed, logits))
}

fn percentile_ns(samples: &[Duration], percentile: usize) -> u128 {
    let mut values: Vec<u128> = samples.iter().map(Duration::as_nanos).collect();
    values.sort_unstable();
    let rank = percentile.saturating_mul(values.len()).div_ceil(100).max(1);
    values[rank - 1]
}

fn diff_stats(left: &[f32], right: &[f32]) -> (usize, f32) {
    assert_eq!(left.len(), right.len());
    let mut different = 0usize;
    let mut max_abs = 0.0f32;
    for (&left, &right) in left.iter().zip(right)
    {
        if left.to_bits() != right.to_bits()
        {
            different += 1;
        }
        max_abs = max_abs.max((left - right).abs());
    }
    (different, max_abs)
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    for index in 1..values.len()
    {
        if values[index] > values[best]
        {
            best = index;
        }
    }
    best
}

fn main() -> Result<(), Box<dyn Error>> {
    let prompt_len = env_usize("SCIAGENT_M54_PROMPT", 128);
    let warmups = env_usize("SCIAGENT_M54_WARMUPS", 1);
    let repeats = env_usize("SCIAGENT_M54_REPEATS", 5);
    if prompt_len == 0 || warmups == 0 || repeats < 3
    {
        return Err("M54 requires prompt>0, warmups>0 and repeats>=3".into());
    }

    let config = SciAgentConfig::sciagent_350m();
    if prompt_len > config.max_seq_len
        || config.n_heads == 0
        || config.n_kv_heads == 0
        || !config.n_heads.is_multiple_of(config.n_kv_heads)
        || !config.d_model.is_multiple_of(config.n_heads)
    {
        return Err("M54 SCIAGENT geometry is invalid for this prompt".into());
    }
    let d_head = config.d_model / config.n_heads;
    if d_head != 64 || config.n_heads != 16 || config.n_kv_heads != 4
    {
        return Err(format!(
            "M54 expected SCIAGENT 16Q/4KV D64, got {}/{} D{}",
            config.n_heads, config.n_kv_heads, d_head
        )
        .into());
    }

    let model = SciAgentModel::new(&config);
    let Some(chain) = GpuChain::new()
    else
    {
        return Err("WGPU adapter unavailable".into());
    };
    let portable = chain.flat_m11_bridge()?;
    let vec4 = chain.flat_m11_bridge_with_vectorization(true)?;
    if portable.vectorization_enabled() || !vec4.vectorization_enabled()
    {
        return Err("M54 bridge selection contract failed".into());
    }
    if chain.adapter_name() != portable.adapter_name()
        || chain.adapter_name() != vec4.adapter_name()
    {
        return Err("M54 routes selected different adapters".into());
    }

    let weights = upload_model(&chain, &model);
    let prompt = synthetic_tokens(prompt_len, config.vocab_size);

    // Correctness gate before any timing claim. M53 was exactly equal to the
    // portable attention output on its qualified Thor matrix; M54 requires that
    // exact equality to survive the complete SCIAGENT prefill and LM head too.
    let (_, portable_logits) = timed(
        Route::Portable,
        &chain,
        &portable,
        &vec4,
        &weights,
        &config,
        &prompt,
    )?;
    let (_, vec4_logits) = timed(
        Route::Vec4,
        &chain,
        &portable,
        &vec4,
        &weights,
        &config,
        &prompt,
    )?;
    let (different_logits, max_abs) = diff_stats(&portable_logits, &vec4_logits);
    let portable_token = argmax(&portable_logits);
    let vec4_token = argmax(&vec4_logits);
    if different_logits != 0 || portable_token != vec4_token
    {
        return Err(format!(
            "M54 parity failed: different_logits={different_logits} max_abs={max_abs:.9e} portable_token={portable_token} vec4_token={vec4_token}"
        )
        .into());
    }

    let routes = [Route::Portable, Route::Vec4];
    for iteration in 0..warmups
    {
        for offset in 0..routes.len()
        {
            let route = routes[(iteration + offset) % routes.len()];
            black_box(timed(
                route, &chain, &portable, &vec4, &weights, &config, &prompt,
            )?);
        }
    }

    let mut samples: [Vec<Duration>; 2] = std::array::from_fn(|_| Vec::with_capacity(repeats));
    for iteration in 0..repeats
    {
        for offset in 0..routes.len()
        {
            let index = (iteration + offset) % routes.len();
            let (elapsed, logits) = timed(
                routes[index],
                &chain,
                &portable,
                &vec4,
                &weights,
                &config,
                &prompt,
            )?;
            black_box(logits);
            samples[index].push(elapsed);
        }
    }

    let portable_median_ns = percentile_ns(&samples[0], 50);
    let vec4_median_ns = percentile_ns(&samples[1], 50);
    println!(
        "adapter,backend,params,d_model,n_layers,q_heads,kv_heads,head_dim,d_ff,vocab,prompt,warmups,repeats,portable_median_ms,vec4_median_ms,portable_p95_ms,vec4_p95_ms,portable_over_vec4,different_logits,max_abs,portable_token,vec4_token,performance_claim"
    );
    println!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.6},{},{:.9e},{},{},none",
        portable.adapter_name().replace(',', ";"),
        portable.context().adapter_backend().replace(',', ";"),
        config.total_parameters(),
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        d_head,
        config.d_ff,
        config.vocab_size,
        prompt_len,
        warmups,
        repeats,
        portable_median_ns as f64 / 1_000_000.0,
        vec4_median_ns as f64 / 1_000_000.0,
        percentile_ns(&samples[0], 95) as f64 / 1_000_000.0,
        percentile_ns(&samples[1], 95) as f64 / 1_000_000.0,
        portable_median_ns as f64 / vec4_median_ns.max(1) as f64,
        different_logits,
        max_abs,
        portable_token,
        vec4_token,
    );
    Ok(())
}
