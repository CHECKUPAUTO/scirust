use std::time::{Duration, Instant};

#[cfg(feature = "flat-attention")]
use scirust_sciagent::config::SciAgentConfig;
#[cfg(feature = "flat-attention")]
use scirust_sciagent::gpu::ResidentModel;
#[cfg(feature = "flat-attention")]
use scirust_sciagent::model::SciAgentModel;

#[cfg(feature = "flat-attention")]
fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "flat-attention")]
fn prompt(len: usize, vocab: usize) -> Vec<u32> {
    (0..len)
        .map(|index| ((index * 17 + 3) % vocab) as u32)
        .collect()
}

#[cfg(feature = "flat-attention")]
fn elapsed(mut operation: impl FnMut()) -> Duration {
    let started = Instant::now();
    operation();
    started.elapsed()
}

#[cfg(feature = "flat-attention")]
fn percentile(values: &mut [f64], percentile: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values[index]
}

#[cfg(feature = "flat-attention")]
fn main() {
    let vocab = 256usize;
    let prompt_len = env_usize("SCIAGENT_FLAT_M33_PROMPT", 128);
    let new_tokens = env_usize("SCIAGENT_FLAT_M33_NEW", 32).max(2);
    let warmups = env_usize("SCIAGENT_FLAT_M33_WARMUPS", 2);
    let repeats = env_usize("SCIAGENT_FLAT_M33_REPEATS", 7).max(1);
    let config = SciAgentConfig {
        vocab_size: vocab,
        d_model: env_usize("SCIAGENT_FLAT_M33_D_MODEL", 512),
        n_layers: env_usize("SCIAGENT_FLAT_M33_LAYERS", 8),
        n_heads: env_usize("SCIAGENT_FLAT_M33_Q_HEADS", 8),
        n_kv_heads: env_usize("SCIAGENT_FLAT_M33_KV_HEADS", 2),
        d_ff: env_usize("SCIAGENT_FLAT_M33_D_FF", 1408),
        max_seq_len: prompt_len + new_tokens + 8,
        rope_theta: 10_000.0,
        tie_embeddings: true,
        use_bias: false,
        eps: 1e-5,
    };
    let model = SciAgentModel::new(&config);
    let Some(resident) = ResidentModel::from_model(&model)
    else {
        eprintln!("FLAT M33 requires a real WGPU adapter; none was available");
        std::process::exit(2);
    };
    let prompt = prompt(prompt_len, vocab);

    for _ in 0..warmups {
        let _ = resident.generate_cached(&prompt, 1);
        let _ = resident.generate_cached(&prompt, new_tokens);
    }

    let mut prefill_ms = Vec::with_capacity(repeats);
    let mut decode_ms_per_token = Vec::with_capacity(repeats);
    for index in 0..repeats {
        let (prefill, total) = if index.is_multiple_of(2) {
            let prefill = elapsed(|| {
                let _ = resident.generate_cached(&prompt, 1);
            });
            let total = elapsed(|| {
                let _ = resident.generate_cached(&prompt, new_tokens);
            });
            (prefill, total)
        } else {
            let total = elapsed(|| {
                let _ = resident.generate_cached(&prompt, new_tokens);
            });
            let prefill = elapsed(|| {
                let _ = resident.generate_cached(&prompt, 1);
            });
            (prefill, total)
        };
        let prefill = prefill.as_secs_f64() * 1e3;
        let total = total.as_secs_f64() * 1e3;
        let decode_steps = (new_tokens - 1) as f64;
        let decode = (total - prefill).max(f64::MIN_POSITIVE) / decode_steps;
        prefill_ms.push(prefill);
        decode_ms_per_token.push(decode);
    }

    let mut prefill_for_median = prefill_ms.clone();
    let mut decode_for_median = decode_ms_per_token.clone();
    let mut decode_for_p95 = decode_ms_per_token;
    let prefill_median_ms = percentile(&mut prefill_for_median, 0.5);
    let decode_median_ms = percentile(&mut decode_for_median, 0.5);
    let decode_p95_ms = percentile(&mut decode_for_p95, 0.95);
    let tokens_per_second = 1_000.0 / decode_median_ms;
    let revision = std::env::var("SCIRUST_SOURCE_REVISION")
        .or_else(|_| std::env::var("GITHUB_SHA"))
        .unwrap_or_else(|_| "unknown".to_owned());

    println!(
        "revision,adapter,d_model,layers,q_heads,kv_heads,prompt_len,new_tokens,warmups,repeats,prefill_median_ms,decode_median_ms_per_token,decode_p95_ms_per_token,decode_tokens_per_second,performance_claim"
    );
    println!(
        "{revision},\"{}\",{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},none",
        resident.adapter_name().replace('"', "\"\""),
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        prompt_len,
        new_tokens,
        warmups,
        repeats,
        prefill_median_ms,
        decode_median_ms,
        decode_p95_ms,
        tokens_per_second,
    );
}

#[cfg(not(feature = "flat-attention"))]
fn main() {
    eprintln!("enable the flat-attention feature to run the FLAT M33 benchmark");
    std::process::exit(2);
}
