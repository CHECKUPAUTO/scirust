//! **Resident generation** — load a (pretrained / fine-tuned) `SciAgentModel`,
//! mirror it into VRAM, and decode a continuation on the fully-resident GPU path
//! with the `O(n)`-per-token KV cache and the shared deterministic sampler. This
//! is the runnable end of the on-device loop: **pretrain / fine-tune → merge →
//! generate**, all on the Jetson Thor.
//!
//! Environment:
//! - `SCIAGENT_CKPT` — a checkpoint dir (`step_N/`, as written by
//!   `resident_pretrain`, or the `merged/` dir from `resident_lora_finetune`) to
//!   load. Without it a fresh `byte` model is used (output is noise — set this
//!   for a real generation).
//! - `SCIAGENT_PROMPT` — the prompt. On a byte-level model (vocab ≥ 256) it is
//!   ingested **byte-level** and the continuation is decoded back to (lossy) UTF-8;
//!   otherwise pass comma-separated token ids and ids are printed.
//! - `SCIAGENT_MAX_NEW` (128) — tokens to generate (capped so prompt+new ≤
//!   `max_seq_len`, the trained RoPE range).
//! - `SCIAGENT_TEMP` (0.0 = greedy), `SCIAGENT_TOP_K` (0 = off),
//!   `SCIAGENT_TOP_P` (1.0 = off), `SCIAGENT_REP_PENALTY` (1.0 = off),
//!   `SCIAGENT_SEED` (0) — decoding knobs, forwarded to the shared sampler.
//!
//! ```text
//! SCIAGENT_CKPT=checkpoints/cervo-bytes/step_2000 SCIAGENT_PROMPT='fn main() {' \
//!   SCIAGENT_TEMP=0.8 SCIAGENT_TOP_P=0.95 SCIAGENT_MAX_NEW=200 \
//!   cargo run -p scirust-sciagent --features gpu --release --example resident_generate
//! ```
//!
//! Exit code 2 means no GPU adapter was found — run on the Thor or install a
//! Vulkan ICD.

use std::io::{self, Write};
use std::path::Path;

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::generate::SamplingParams;
use scirust_sciagent::gpu::ResidentModel;
use scirust_sciagent::model::SciAgentModel;
use scirust_sciagent::train::checkpoint::{latest_checkpoint, load_checkpoint, read_meta};

fn byte_config() -> SciAgentConfig {
    SciAgentConfig {
        vocab_size: 256,
        d_model: 256,
        n_layers: 6,
        n_heads: 8,
        n_kv_heads: 2,
        d_ff: 512,
        max_seq_len: 512,
        rope_theta: 10_000.0,
        tie_embeddings: true,
        use_bias: false,
        eps: 1e-5,
    }
}

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
    // Base model: a checkpoint (real generation) or a fresh byte model (demo).
    let mut model;
    if let Ok(ckpt) = std::env::var("SCIAGENT_CKPT")
    {
        let dir = latest_checkpoint(Path::new(&ckpt)).unwrap_or_else(|| ckpt.clone().into());
        let meta = read_meta(&dir).unwrap_or_else(|e| panic!("read meta {}: {e}", dir.display()));
        model = SciAgentModel::new(&meta.config);
        load_checkpoint(&mut model, &dir).unwrap_or_else(|e| panic!("load {}: {e}", dir.display()));
        println!(
            "model loaded from {} (vocab {}, {} layers)",
            dir.display(),
            meta.config.vocab_size,
            meta.config.n_layers
        );
    }
    else
    {
        model = SciAgentModel::new(&byte_config());
        println!("no SCIAGENT_CKPT — fresh byte model (demo; set it for a real generation)");
    }
    let config = model.config.clone();
    assert!(
        config.tie_embeddings,
        "resident generation requires a tied-embedding model"
    );

    let Some(rm) = ResidentModel::from_model(&model)
    else
    {
        eprintln!("no GPU adapter available. Install a Vulkan ICD or run on the Jetson Thor.");
        std::process::exit(2);
    };
    println!("resident generation on: {}", rm.adapter_name());

    // Prompt → token ids. Byte-level models ingest the raw UTF-8 bytes; otherwise
    // the prompt must be given as comma-separated ids.
    let byte_level = config.vocab_size >= 256;
    let prompt_str = std::env::var("SCIAGENT_PROMPT").unwrap_or_else(|_| "The ".into());
    let prompt: Vec<u32> = if byte_level
    {
        prompt_str.bytes().map(u32::from).collect()
    }
    else
    {
        prompt_str
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .filter(|&t| (t as usize) < config.vocab_size)
            .collect()
    };
    assert!(
        !prompt.is_empty(),
        "empty prompt — set SCIAGENT_PROMPT (byte text, or comma-separated ids for a non-byte vocab)"
    );

    // Cap the run to the trained RoPE range (prompt + new ≤ max_seq_len).
    let want_new = env_usize("SCIAGENT_MAX_NEW", 128);
    let budget = config.max_seq_len.saturating_sub(prompt.len());
    let max_new = want_new.min(budget);
    if max_new < want_new
    {
        eprintln!(
            "note: capping to {max_new} new tokens (prompt {} + new ≤ max_seq_len {})",
            prompt.len(),
            config.max_seq_len
        );
    }

    let params = SamplingParams {
        temperature: env_f32("SCIAGENT_TEMP", 0.0),
        top_k: env_usize("SCIAGENT_TOP_K", 0),
        top_p: env_f32("SCIAGENT_TOP_P", 1.0),
        repetition_penalty: env_f32("SCIAGENT_REP_PENALTY", 1.0),
        repetition_window: env_usize("SCIAGENT_REP_WINDOW", 64),
    };
    let seed: u64 = std::env::var("SCIAGENT_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    println!(
        "prompt {} tokens · max_new {max_new} · T {} top_k {} top_p {} rep {} · seed {seed}\n",
        prompt.len(),
        params.temperature,
        params.top_k,
        params.top_p,
        params.repetition_penalty
    );

    // Echo the prompt, then stream the continuation token-by-token as it decodes.
    if byte_level
    {
        let prompt_bytes: Vec<u8> = prompt.iter().map(|&t| t as u8).collect();
        println!("=== prompt ===\n{}", String::from_utf8_lossy(&prompt_bytes));
        println!("=== continuation (streaming) ===");
    }
    else
    {
        println!("prompt ids:       {prompt:?}");
        print!("continuation ids: ");
    }
    let mut so = io::stdout();
    let out = rm.generate_streaming(&prompt, max_new, &params, seed, |tok| {
        if byte_level
        {
            let _ = so.write_all(&[tok as u8]);
        }
        else
        {
            let _ = write!(so, "{tok} ");
        }
        let _ = so.flush();
    });
    let _ = writeln!(so);
    eprintln!("[{} tokens generated]", out.len() - prompt.len());
}
