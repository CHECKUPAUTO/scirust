//! **Route B — production-scale bf16 pretraining on Tensor cores**
//! (`CudaTrainer::pretrain`, feature `cuda`). The Route-B counterpart of
//! `resident_pretrain`: same real-corpus ingestion, warmup+cosine LR schedule,
//! throughput logging, and periodic safetensors checkpointing — but the whole
//! forward+backward+AdamW runs in bf16 on Blackwell Tensor cores with fp32 master
//! weights (the ~4.7× training path, see `cuda_train_bench` / `JETSON_THOR.md`).
//!
//! Same environment interface as `resident_pretrain` so the two are drop-in:
//!
//! - `SCIAGENT_CONFIG` — model preset: `350m` (BPE, needs shards), `code350m`
//!   (byte-level ~270M — a real large run with no tokenizer), `small`, `byte`, or
//!   `demo` (default). On resume the checkpoint's own config wins.
//! - `SCIAGENT_TEXT` — a file/dir ingested **byte-level** (vocab 256, no tokenizer).
//!   Auto-selects the `byte` config when `SCIAGENT_CONFIG` is unset.
//! - `SCIAGENT_SHARDS` — a dir of little-endian `u32` `.bin` token shards; aborts if
//!   a token id ≥ the config's `vocab_size` (shards tokenised for another vocab).
//! - `SCIAGENT_CKPT` (default `checkpoints/cuda`), `SCIAGENT_STEPS` (300),
//!   `SCIAGENT_SEQ` (128), `SCIAGENT_LR` — run knobs.
//!
//! ```text
//! # self-contained smoke run (synthetic corpus, demo config):
//! cargo run -p scirust-sciagent --features cuda --release --example cuda_pretrain
//!
//! # real ~270M byte-level run on a code tree — no tokenizer, turnkey:
//! SCIAGENT_CONFIG=code350m SCIAGENT_TEXT=$HOME/corpus SCIAGENT_SEQ=512 \
//!   SCIAGENT_STEPS=20000 \
//!   cargo run -p scirust-sciagent --features cuda --release --example cuda_pretrain
//!
//! # full 350M bf16 run on BPE shards (needs the collect-data → tokenizer pipeline):
//! SCIAGENT_CONFIG=350m SCIAGENT_SHARDS=$HOME/data/shards SCIAGENT_STEPS=2000 \
//!   cargo run -p scirust-sciagent --features cuda --release --example cuda_pretrain
//! ```
//!
//! On start-up the newest `step_N/` in `SCIAGENT_CKPT` is loaded and training
//! resumes from it (the LR schedule continues from `meta.step`; the AdamW moments
//! restart from zero, which the warmup re-absorbs). Exit code 2 means no CUDA
//! device was found — run on the Jetson Thor.

use std::path::Path;

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::cuda_model::{CudaPretrainConfig, CudaTrainer};
use scirust_sciagent::model::SciAgentModel;
use scirust_sciagent::train::checkpoint::{latest_checkpoint, load_checkpoint, read_meta};
use scirust_sciagent::train::dataset::{ShardLoader, source_quality};

/// A tied, vocab-256 byte-level config — small enough to iterate fast, real enough
/// to train on an actual code tree with no tokenizer.
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

/// A **byte-level large** config — the `sciagent_350m` trunk shape (d1024, 24
/// layers, 16h/4kv, d_ff 2816) at vocab 256, so a genuine ~270M-parameter model
/// trains from scratch on raw bytes with **no tokenizer or shard pipeline**. This
/// is the turnkey large run: point `SCIAGENT_TEXT` at a code tree and go.
fn code_large_config() -> SciAgentConfig {
    SciAgentConfig {
        vocab_size: 256,
        d_model: 1024,
        n_layers: 24,
        n_heads: 16,
        n_kv_heads: 4,
        d_ff: 2816,
        max_seq_len: 2048,
        rope_theta: 1_000_000.0,
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
        "code350m" | "large" | "byte-large" =>
        {
            (code_large_config(), "code350m (byte-level ~270M)".into())
        },
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

/// Directories never worth ingesting for byte-level *source* pretraining — VCS
/// internals, build artifacts, vendored deps, caches. Skipping them matters: the
/// sorted walk reads `.git` first (dot sorts before letters), so its packed binary
/// objects would otherwise dominate the head of the corpus — a real run collapsed
/// deterministically on exactly that garbage (see `ROUTE_B.md`).
fn skip_dir(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | ".hg"
            | ".svn"
            | "target"
            | "node_modules"
            | ".cargo"
            | "dist"
            | "build"
            | ".venv"
            | "venv"
            | "__pycache__"
            | ".mypy_cache"
            | ".pytest_cache"
            | ".idea"
            | ".vscode"
    )
}

/// Whether `bytes` look like source text: valid UTF-8 with no NUL byte. Binary
/// files (compiled artifacts, images, `.git` objects, archives) fail this and are
/// skipped — byte-level pretraining should see text, not binary blobs.
fn is_probably_text(bytes: &[u8]) -> bool {
    !bytes.contains(&0) && std::str::from_utf8(bytes).is_ok()
}

/// Recursively read raw file bytes under `root` (deterministic order), up to `cap`
/// bytes — **source text only**: non-source directories ([`skip_dir`]) and non-text
/// files ([`is_probably_text`]) are skipped.
fn read_bytes_recursive(root: &Path, out: &mut Vec<u8>, cap: usize) {
    if out.len() >= cap
    {
        return;
    }
    if root.is_file()
    {
        if let Ok(b) = std::fs::read(root)
        {
            if is_probably_text(&b)
            {
                // Same corpus-quality gate as collect-data: skip generated/minified/
                // data-table files (valid UTF-8 is already established, so the str
                // conversion is safe).
                let name = root.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if let Ok(text) = std::str::from_utf8(&b)
                {
                    if source_quality(name, text).is_err()
                    {
                        return;
                    }
                }
                let take = (cap - out.len()).min(b.len());
                out.extend_from_slice(&b[..take]);
            }
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
            if p.is_dir()
            {
                if let Some(name) = p.file_name().and_then(|n| n.to_str())
                {
                    if skip_dir(name)
                    {
                        continue;
                    }
                }
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
    let ckpt_dir = std::env::var("SCIAGENT_CKPT").unwrap_or_else(|_| "checkpoints/cuda".into());

    // Config: a resumed checkpoint's own config wins; else SCIAGENT_CONFIG; else the
    // byte config when ingesting raw text; else demo.
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

    let Some(mut trainer) = CudaTrainer::from_model(&model)
    else
    {
        eprintln!("no CUDA device available. Run on the Jetson Thor (needs the CUDA toolkit).");
        std::process::exit(2);
    };
    trainer.reset_step(); // fresh AdamW moments; the LR schedule continues via start_step

    let params = config.total_parameters();
    let weight_mb = params as f64 * 4.0 / 1e6; // fp32 master
    let bf16_mb = params as f64 * 2.0 / 1e6; // bf16 forward view
    let opt_mb = params as f64 * 8.0 / 1e6; // AdamW m + v, fp32
    println!("Route B bf16 pretraining on: {}\n", trainer.adapter_name());
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
        "resident VRAM estimate: fp32 master ~{weight_mb:.0} MB + bf16 view ~{bf16_mb:.0} MB + \
         AdamW state ~{opt_mb:.0} MB (activations extra)\n"
    );

    let seq_len = env_usize("SCIAGENT_SEQ", 128).min(config.max_seq_len);
    let max_tokens = env_usize("SCIAGENT_MAX_TOKENS", 16_000_000);

    // Token stream: BPE shards, byte-level text, or a synthetic corpus.
    let tokens: Vec<u32> = if let Ok(dir) = std::env::var("SCIAGENT_SHARDS")
    {
        let mut loader = ShardLoader::new();
        if let Err(e) = loader.load_dir(&dir)
        {
            eprintln!(
                "failed to load shards from {dir}: {e}\n\
                 (SCIAGENT_SHARDS must point at a directory of little-endian u32 .bin token\n\
                 shards, as written by the collect-data binary. For a tokenizer-free run,\n\
                 use SCIAGENT_TEXT=<file|dir> instead for byte-level ingestion.)"
            );
            std::process::exit(1);
        }
        let raw = loader.tokens();
        let maxid = raw.iter().copied().max().unwrap_or(0) as usize;
        if maxid >= config.vocab_size
        {
            eprintln!(
                "shard token id {maxid} >= config vocab_size {}: these shards were tokenised for a\n\
                 different vocab. Set SCIAGENT_CONFIG to the matching preset (e.g. 350m), or\n\
                 re-tokenise with collect-data.",
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
    // LR must key off model *size*, not vocab: a 270M byte-level model (vocab 256)
    // diverges at the 3e-3 that suits a tiny demo. Small trunks (d_model ≤ 256) can
    // take the hot 3e-3; anything larger gets the standard 3e-4 (with warmup+cosine).
    let default_lr = if config.d_model <= 256 { 3e-3 } else { 3e-4 };
    let base_lr = std::env::var("SCIAGENT_LR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default_lr);
    // Global grad-norm clip (default 1.0; SCIAGENT_CLIP overrides, <= 0 disables).
    let max_grad_norm = std::env::var("SCIAGENT_CLIP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0f32);
    // AdamW epsilon (default 1e-5, bf16-appropriate; SCIAGENT_EPS overrides).
    let adam_eps = std::env::var("SCIAGENT_EPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-5f32);
    // Held-out validation fraction (tail; default 2%; SCIAGENT_VAL_FRAC overrides, 0 disables).
    let val_frac = std::env::var("SCIAGENT_VAL_FRAC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.02f32);
    let cfg = CudaPretrainConfig {
        base_lr,
        min_lr: base_lr * 0.1,
        warmup_steps: start_step + (env_usize("SCIAGENT_STEPS", 300) / 10).max(1),
        total_steps,
        start_step,
        seq_len,
        weight_decay: 0.0,
        adam_eps,
        log_interval: 25,
        save_interval: 100,
        checkpoint_dir: ckpt_dir.clone(),
        max_grad_norm,
        val_frac,
        eval_interval: 100,
        ..Default::default()
    };
    println!(
        "seq_len {seq_len} | steps {start_step}..{total_steps} | base_lr {base_lr:.1e} | \
         eps {adam_eps:.0e} | clip {max_grad_norm} | ckpt → {ckpt_dir}\n"
    );

    let losses = trainer.pretrain(&tokens, &mut model, &config, &cfg);
    if losses.is_empty()
    {
        eprintln!("no steps ran (corpus too short for one seq_len={seq_len} window?)");
        std::process::exit(1);
    }

    let n = losses.len().clamp(1, 5);
    let first: f32 = losses[..n].iter().sum::<f32>() / n as f32;
    let last: f32 = losses[losses.len() - n..].iter().sum::<f32>() / n as f32;
    println!(
        "\n{} bf16 steps: loss {first:.4} -> {last:.4}  ({:.1}% reduction)",
        losses.len(),
        (1.0 - last / first) * 100.0
    );

    // Final sync + checkpoint so the last weights are always persisted.
    trainer.sync_to_model(&mut model);
    println!("trained fp32 masters synced back into the SciAgentModel; resume from {ckpt_dir}.");
}
