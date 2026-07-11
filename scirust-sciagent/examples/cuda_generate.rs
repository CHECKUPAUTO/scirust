//! **Route B — sample from a trained checkpoint on the CUDA backend**
//! (`CudaModel::generate`, feature `cuda`). Loads a checkpoint written by
//! `cuda_pretrain` (or any `save_checkpoint`), mirrors it into VRAM as bf16, and
//! generates autoregressively on Blackwell Tensor cores using the shared
//! deterministic sampler — the Route-B counterpart of `resident_generate`.
//!
//! Non-cached (re-runs the forward each step) — fine for eyeballing quality on
//! short samples; a KV-cached decode is the follow-up.
//!
//! - `SCIAGENT_CKPT` — a checkpoint dir (`step_N/`, or a parent holding `step_*`).
//! - `SCIAGENT_TOKENIZER` — a BPE tokenizer json (from `train-tokenizer`). Set it
//!   for a BPE model (e.g. the 350m config); omit for a byte-level model, where the
//!   prompt is UTF-8 bytes and output is decoded back to text.
//! - `SCIAGENT_PROMPT` (default `fn `) — the prompt text.
//! - `SCIAGENT_MAX_NEW` (128) — tokens to generate.
//! - `SCIAGENT_TEMP` (0.0 = greedy), `SCIAGENT_TOP_K` (0), `SCIAGENT_TOP_P` (1.0),
//!   `SCIAGENT_REP_PENALTY` (1.0), `SCIAGENT_REP_WINDOW` (64), `SCIAGENT_SEED` (0).
//!
//! ```text
//! # byte-level model:
//! SCIAGENT_CKPT=checkpoints/code350m SCIAGENT_PROMPT='fn main() {' \
//!   SCIAGENT_TEMP=0.8 SCIAGENT_TOP_P=0.95 SCIAGENT_MAX_NEW=200 \
//!   cargo run -p scirust-sciagent --features cuda --release --example cuda_generate
//!
//! # BPE 350M model:
//! SCIAGENT_CKPT=checkpoints/bpe350m SCIAGENT_TOKENIZER=tokenizer.json \
//!   SCIAGENT_PROMPT='fn main() {' SCIAGENT_TEMP=0.8 \
//!   cargo run -p scirust-sciagent --features cuda --release --example cuda_generate
//! ```
//!
//! Exit code 2 means no CUDA device — run on the Jetson Thor.

use std::path::Path;

use scirust_sciagent::bpe::BpeTokenizer;
use scirust_sciagent::cuda_model::CudaModel;
use scirust_sciagent::generate::SamplingParams;
use scirust_sciagent::model::SciAgentModel;
use scirust_sciagent::train::checkpoint::{latest_checkpoint, load_checkpoint, read_meta};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    // Load the checkpoint (its own config wins).
    let ckpt = match std::env::var("SCIAGENT_CKPT")
    {
        Ok(c) => c,
        Err(_) =>
        {
            eprintln!("set SCIAGENT_CKPT to a checkpoint dir (e.g. checkpoints/code350m)");
            std::process::exit(1);
        },
    };
    let dir = latest_checkpoint(Path::new(&ckpt)).unwrap_or_else(|| ckpt.clone().into());
    let meta = read_meta(&dir).unwrap_or_else(|e| panic!("read meta {}: {e}", dir.display()));
    let mut model = SciAgentModel::new(&meta.config);
    load_checkpoint(&mut model, &dir).unwrap_or_else(|e| panic!("load {}: {e}", dir.display()));
    let vocab = meta.config.vocab_size;
    println!(
        "loaded {} (step {}, loss {:.4}) — vocab {vocab}, d {}, {} layers",
        dir.display(),
        meta.step,
        meta.loss,
        meta.config.d_model,
        meta.config.n_layers
    );

    // Optional BPE tokenizer (required for a non-byte vocab).
    let tokenizer = std::env::var("SCIAGENT_TOKENIZER")
        .ok()
        .map(|p| BpeTokenizer::load_json(&p).unwrap_or_else(|e| panic!("load tokenizer {p}: {e}")));
    if tokenizer.is_none() && vocab > 256
    {
        eprintln!(
            "model vocab is {vocab} (BPE) but no SCIAGENT_TOKENIZER set — the prompt can't be\n\
             encoded and output can't be decoded. Point SCIAGENT_TOKENIZER at the tokenizer json."
        );
        std::process::exit(1);
    }

    let Some(cm) = CudaModel::from_model(&model)
    else
    {
        eprintln!("no CUDA device available. Run on the Jetson Thor.");
        std::process::exit(2);
    };

    let prompt_str = std::env::var("SCIAGENT_PROMPT").unwrap_or_else(|_| "fn ".into());
    let prompt: Vec<u32> = match &tokenizer
    {
        Some(tok) => tok
            .encode_with_special(&prompt_str, true, false)
            .iter()
            .map(|&i| i as u32)
            .collect(),
        None => prompt_str.bytes().map(u32::from).collect(),
    };
    if prompt.is_empty()
    {
        eprintln!("empty prompt");
        std::process::exit(1);
    }

    let params = SamplingParams {
        temperature: env_f32("SCIAGENT_TEMP", 0.0),
        top_k: env_usize("SCIAGENT_TOP_K", 0),
        top_p: env_f32("SCIAGENT_TOP_P", 1.0),
        repetition_penalty: env_f32("SCIAGENT_REP_PENALTY", 1.0),
        repetition_window: env_usize("SCIAGENT_REP_WINDOW", 64),
    };
    let max_new = env_usize("SCIAGENT_MAX_NEW", 128)
        .min(meta.config.max_seq_len.saturating_sub(prompt.len()).max(1));
    let seed: u64 = std::env::var("SCIAGENT_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    println!(
        "prompt {:?} ({} tokens) | max_new {max_new} | temp {} top_k {} top_p {} rep {}\n",
        prompt_str,
        prompt.len(),
        params.temperature,
        params.top_k,
        params.top_p,
        params.repetition_penalty
    );

    let out = cm.generate(&prompt, max_new, &params, seed);
    let generated = &out[prompt.len()..];

    // Raw ids first — so an empty/garbled decode is still diagnosable (e.g. an
    // under-trained model looping on one token, or generating special/placeholder
    // ids that `decode` skips).
    println!("prompt ids:    {:?}", prompt);
    let head: Vec<u32> = generated.iter().take(32).copied().collect();
    println!(
        "generated ids: {head:?}{}",
        if generated.len() > 32 { " …" } else { "" }
    );
    let mut distinct = generated.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    println!(
        "distinct generated ids: {} of {}",
        distinct.len(),
        generated.len()
    );

    let text =
        match &tokenizer
        {
            Some(tok) => tok.decode(&out.iter().map(|&t| t as usize).collect::<Vec<_>>()),
            None => String::from_utf8_lossy(&out.iter().map(|&t| t as u8).collect::<Vec<_>>())
                .into_owned(),
        };
    println!("\n=== generation ({} new tokens) ===", generated.len());
    println!("{text}");
}
