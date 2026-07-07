//! **Production-scale resident pretraining** — the full run harness on the
//! fully-resident GPU path (`ResidentModel::pretrain`): real corpus ingestion, a
//! warmup + cosine LR schedule, throughput logging, and periodic safetensors
//! checkpointing, all in VRAM (the path that beats the per-op tape ~4× on the
//! Jetson Thor).
//!
//! Everything is driven by environment variables so one binary covers a smoke
//! test and a real run:
//!
//! - `SCIAGENT_CONFIG` — model preset: `350m`, `small`, `byte` (vocab-256
//!   byte-level), or `demo` (default). When resuming, the checkpoint's own
//!   config wins (so the reloaded weights always match).
//! - `SCIAGENT_TEXT` — a file or directory of source; ingested **byte-level**
//!   (each byte = one token, vocab 256), so a real run needs **no tokenizer**.
//!   Auto-selects the `byte` config when `SCIAGENT_CONFIG` is unset.
//! - `SCIAGENT_SHARDS` — a directory of little-endian `u32` `.bin` token shards
//!   (as written by the `collect-data` binary, which BPE-tokenises → packs
//!   `u32`). The run **aborts** if a token id ≥ the config's `vocab_size`, i.e.
//!   the shards were tokenised for a different vocab (rather than silently
//!   mapping most of the corpus to `<unk>`).
//! - `SCIAGENT_CKPT` (default `checkpoints/resident`), `SCIAGENT_STEPS`
//!   (default 300), `SCIAGENT_SEQ` (default 128), `SCIAGENT_LR` — run knobs.
//!
//! ```text
//! # self-contained smoke run (synthetic corpus, demo config):
//! cargo run -p scirust-sciagent --features gpu --release --example resident_pretrain
//!
//! # real run on a code tree, no tokenizer needed (byte-level):
//! SCIAGENT_TEXT=$HOME/CERVO SCIAGENT_STEPS=2000 \
//!   cargo run -p scirust-sciagent --features gpu --release --example resident_pretrain
//!
//! # real run on BPE shards at the 350M config:
//! SCIAGENT_CONFIG=350m SCIAGENT_SHARDS=$HOME/data/shards \
//!   cargo run -p scirust-sciagent --features gpu --release --example resident_pretrain
//! ```
//!
//! On start-up the newest `step_N/` in `SCIAGENT_CKPT` is loaded and training
//! resumes from it (the LR schedule continues from `meta.step`; the AdamW moments
//! restart from zero, which the warmup re-absorbs). Exit code 2 means no GPU
//! adapter was found — run on the Thor or install a Vulkan ICD.

use std::path::Path;

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::gpu::{ResidentModel, ResidentPretrainConfig};
use scirust_sciagent::model::SciAgentModel;
use scirust_sciagent::train::checkpoint::{latest_checkpoint, load_checkpoint, read_meta};
use scirust_sciagent::train::dataset::ShardLoader;

/// A tied, vocab-256 byte-level config — small enough to iterate fast, real
/// enough to train on an actual code tree with no tokenizer.
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

/// The default self-contained demo config (tied, vocab 512).
fn demo_config() -> SciAgentConfig {
    SciAgentConfig {
        vocab_size: 512,
        d_model: 256,
        n_layers: 6,
        n_heads: 8,
        n_kv_heads: 2,
        d_ff: 512,
        max_seq_len: 256,
        rope_theta: 10_000.0,
        tie_embeddings: true,
        use_bias: false,
        eps: 1e-5,
    }
}

fn preset_by_name(name: &str) -> (SciAgentConfig, String) {
    match name.to_ascii_lowercase().as_str()
    {
        "350m" => (SciAgentConfig::sciagent_350m(), "350m".into()),
        "small" => (SciAgentConfig::small(), "small".into()),
        "byte" => (byte_config(), "byte".into()),
        "demo" => (demo_config(), "demo".into()),
        other =>
        {
            eprintln!("unknown SCIAGENT_CONFIG='{other}', falling back to demo");
            (demo_config(), "demo".into())
        },
    }
}

/// Recursively read raw file bytes under `root` (deterministic order), up to
/// `cap` bytes. Used for byte-level ingestion.
fn read_bytes_recursive(root: &Path, out: &mut Vec<u8>, cap: usize) {
    if out.len() >= cap
    {
        return;
    }
    if root.is_file()
    {
        if let Ok(b) = std::fs::read(root)
        {
            let take = (cap - out.len()).min(b.len());
            out.extend_from_slice(&b[..take]);
        }
        return;
    }
    if let Ok(entries) = std::fs::read_dir(root)
    {
        let mut paths: Vec<_> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        paths.sort();
        for p in paths
        {
            if out.len() >= cap
            {
                break;
            }
            read_bytes_recursive(&p, out, cap);
        }
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let ckpt_dir = std::env::var("SCIAGENT_CKPT").unwrap_or_else(|_| "checkpoints/resident".into());

    // Config: a resumed checkpoint's own config wins (the reloaded weights must
    // match); else an explicit SCIAGENT_CONFIG preset; else the byte config when
    // ingesting raw text; else the demo config.
    let resume =
        latest_checkpoint(Path::new(&ckpt_dir)).and_then(|p| read_meta(&p).ok().map(|m| (p, m)));
    let (config, config_src) = if let Some((_, meta)) = &resume
    {
        (meta.config.clone(), "checkpoint".to_string())
    }
    else if let Ok(name) = std::env::var("SCIAGENT_CONFIG")
    {
        preset_by_name(&name)
    }
    else if std::env::var("SCIAGENT_TEXT").is_ok()
    {
        (byte_config(), "byte (auto for SCIAGENT_TEXT)".into())
    }
    else
    {
        (demo_config(), "demo".into())
    };

    let mut model = SciAgentModel::new(&config);
    let mut start_step = 0usize;
    if let Some((path, meta)) = &resume
    {
        match load_checkpoint(&mut model, path)
        {
            Ok(_) =>
            {
                start_step = meta.step;
                println!(
                    "resuming from {} (step {}, loss {:.4})",
                    path.display(),
                    meta.step,
                    meta.loss
                );
            },
            Err(e) => eprintln!("could not load {}: {e}; starting fresh", path.display()),
        }
    }

    let Some(mut rm) = ResidentModel::from_model(&model)
    else
    {
        eprintln!("no GPU adapter available. Install a Vulkan ICD or run on the Jetson Thor.");
        std::process::exit(2);
    };
    rm.reset_step(); // fresh AdamW moments; the LR schedule continues via start_step

    let params = config.total_parameters();
    let weight_mb = params as f64 * 4.0 / 1e6;
    let opt_mb = params as f64 * 8.0 / 1e6; // AdamW m + v, f32
    println!("resident pretraining on: {}\n", rm.adapter_name());
    println!(
        "config [{config_src}]: d {}, {} layers, {}h/{}kv, d_ff {}, vocab {} | {:.1}M params",
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.d_ff,
        config.vocab_size,
        params as f64 / 1e6
    );
    println!(
        "resident VRAM estimate: weights ~{weight_mb:.0} MB + AdamW state ~{opt_mb:.0} MB (f32; activations extra)\n"
    );

    let seq_len = env_usize("SCIAGENT_SEQ", 128).min(config.max_seq_len);
    let max_tokens = env_usize("SCIAGENT_MAX_TOKENS", 16_000_000);

    // Token stream: BPE shards, byte-level text, or a synthetic corpus.
    let tokens: Vec<u32> = if let Ok(dir) = std::env::var("SCIAGENT_SHARDS")
    {
        let mut loader = ShardLoader::new();
        loader
            .load_dir(&dir)
            .unwrap_or_else(|e| panic!("failed to load shards from {dir}: {e}"));
        let raw = loader.tokens();
        let maxid = raw.iter().copied().max().unwrap_or(0) as usize;
        if maxid >= config.vocab_size
        {
            eprintln!(
                "shard token id {maxid} >= config vocab_size {}: these shards were tokenised for a\n\
                 different vocab. Set SCIAGENT_CONFIG to the preset matching your tokenizer (e.g.\n\
                 SCIAGENT_CONFIG=350m for a 32768-vocab BPE corpus), or re-tokenise with collect-data.",
                config.vocab_size
            );
            std::process::exit(1);
        }
        println!("streaming {} tokens from BPE shards in {dir}", raw.len());
        raw.iter().take(max_tokens).copied().collect()
    }
    else if let Ok(text) = std::env::var("SCIAGENT_TEXT")
    {
        assert!(
            config.vocab_size >= 256,
            "byte-level ingestion needs vocab_size >= 256 (got {}); use SCIAGENT_CONFIG=byte",
            config.vocab_size
        );
        let mut bytes = Vec::new();
        read_bytes_recursive(Path::new(&text), &mut bytes, max_tokens);
        if bytes.is_empty()
        {
            eprintln!("SCIAGENT_TEXT={text} yielded no bytes (empty or unreadable)");
            std::process::exit(1);
        }
        println!("byte-level: {} tokens from {text}", bytes.len());
        bytes.into_iter().map(u32::from).collect()
    }
    else
    {
        let pattern: Vec<u32> = (0..48u32)
            .map(|i| (i * 11 + 5) % config.vocab_size as u32)
            .collect();
        let toks: Vec<u32> = (0..seq_len * 400)
            .map(|i| pattern[i % pattern.len()])
            .collect();
        println!(
            "no SCIAGENT_TEXT / SCIAGENT_SHARDS set — synthetic corpus of {} tokens",
            toks.len()
        );
        toks
    };

    let total_steps = start_step + env_usize("SCIAGENT_STEPS", 300);
    // A conservative default LR: higher for the tiny demo/byte configs, the usual
    // 3e-4 for the real presets. Override with SCIAGENT_LR.
    let default_lr = if config.vocab_size <= 512 { 3e-3 } else { 3e-4 };
    let base_lr = std::env::var("SCIAGENT_LR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default_lr);
    let cfg = ResidentPretrainConfig {
        base_lr,
        min_lr: base_lr * 0.1,
        warmup_steps: start_step + (env_usize("SCIAGENT_STEPS", 300) / 10).max(1),
        total_steps,
        start_step,
        seq_len,
        weight_decay: 0.0,
        log_interval: 25,
        save_interval: 100,
        checkpoint_dir: ckpt_dir.clone(),
        ..Default::default()
    };
    println!(
        "seq_len {seq_len} | steps {start_step}..{total_steps} | base_lr {base_lr:.1e} | ckpt → {ckpt_dir}\n"
    );

    let losses = rm.pretrain(&tokens, &mut model, &config, &cfg);
    if losses.is_empty()
    {
        eprintln!("no steps ran (corpus too short for one seq_len={seq_len} window?)");
        std::process::exit(1);
    }

    let n = losses.len().clamp(1, 5);
    let first: f32 = losses[..n].iter().sum::<f32>() / n as f32;
    let last: f32 = losses[losses.len() - n..].iter().sum::<f32>() / n as f32;
    println!(
        "\n{} resident steps: loss {first:.4} -> {last:.4}  ({:.1}% reduction)",
        losses.len(),
        (1.0 - last / first) * 100.0
    );

    // Final sync + checkpoint so the last weights are always persisted.
    rm.sync_to_model(&mut model);
    println!("trained weights synced back into the SciAgentModel; resume from {ckpt_dir}.");
}
