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
//!   `SCIAGENT_SEQ` (128), `SCIAGENT_BATCH` (1), `SCIAGENT_TELEMETRY` (25),
//!   `SCIAGENT_LR` — run knobs.
//! - `SCIAGENT_MAX_TOKENS` — cap the corpus (default: **no cap**). Truncation keeps a
//!   *prefix*, not a sample — the shard walk is alphabetical — so a cap on a crates.io
//!   corpus trains only on the alphabetically-first crates. Always logged when it bites.
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

use std::collections::HashSet;
use std::path::Path;

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::cuda_model::{CudaPretrainConfig, CudaTrainer};
use scirust_sciagent::model::SciAgentModel;
use scirust_sciagent::train::checkpoint::{latest_checkpoint, load_checkpoint, read_meta};
use scirust_sciagent::train::dataset::{
    ShardLoader, WINDOW_SPLIT_VERSION, content_hash, source_quality, token_stream_hash,
};

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
/// bytes — **source text only**: non-source directories ([`skip_dir`]), non-text
/// files ([`is_probably_text`]), low-quality files ([`source_quality`]), and
/// byte-identical duplicates (`seen` content hashes) are skipped.
fn read_bytes_recursive(root: &Path, out: &mut Vec<u8>, cap: usize, seen: &mut HashSet<u64>) {
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
                // Same corpus-quality + dedup gate as collect-data (valid UTF-8 is
                // already established, so the str conversion is safe).
                let name = root.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if let Ok(text) = std::str::from_utf8(&b)
                {
                    if source_quality(name, text).is_err()
                    {
                        return;
                    }
                    if !seen.insert(content_hash(text))
                    {
                        return; // duplicate content
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
            read_bytes_recursive(&p, out, cap, seen);
        }
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn allow_nonexact_resume() -> bool {
    matches!(
        std::env::var("SCIAGENT_ALLOW_NONEXACT_RESUME").as_deref(),
        Ok("1" | "true")
    )
}

fn same_f32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits()
}

fn enforce_exact_resume(mismatches: &[String]) {
    if mismatches.is_empty()
    {
        return;
    }
    if !allow_nonexact_resume()
    {
        eprintln!(
            "exact resume refused: {}. Set SCIAGENT_ALLOW_NONEXACT_RESUME=1 only for an intentional branch experiment.",
            mismatches.join(", ")
        );
        std::process::exit(1);
    }
    eprintln!(
        "WARNING: non-exact resume explicitly allowed: {}",
        mismatches.join(", ")
    );
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
    let mut loaded_resume_path: Option<std::path::PathBuf> = None;
    if let Some((path, meta)) = &resume
    {
        match load_checkpoint(&mut model, path)
        {
            Ok(_) =>
            {
                start_step = meta.step;
                loaded_resume_path = Some(path.clone());
                println!(
                    "resuming model from {} (step {}, loss {:.4})",
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
    let optimizer_resume = if let Some(path) = loaded_resume_path.as_deref()
    {
        match trainer.load_optimizer_state(path)
        {
            Ok(Some(state)) =>
            {
                if state.step != start_step
                {
                    eprintln!(
                        "optimizer checkpoint step {} != model step {}; refusing mismatched resume",
                        state.step, start_step
                    );
                    std::process::exit(1);
                }
                println!(
                    "optimizer state restored exactly at step {} (AdamW m/v + bias correction)",
                    state.step
                );
                Some(state)
            },
            Ok(None) =>
            {
                eprintln!(
                    "legacy checkpoint has no optimizer state; AdamW moments restart once. \
                     The next B32 checkpoint will be exactly resumable."
                );
                trainer.reset_step();
                None
            },
            Err(e) =>
            {
                eprintln!("optimizer checkpoint is present but invalid: {e}");
                std::process::exit(1);
            },
        }
    }
    else
    {
        trainer.reset_step();
        None
    };

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

    // Large production runs default to 512: the historical 128-token default
    // rarely held a complete Rust function and contributed to the syntax wall. Exact
    // B34 resumes inherit their saved B/T unless the operator explicitly overrides.
    let default_seq = if config.d_model > 256 { 512 } else { 128 };
    let explicit_seq = std::env::var("SCIAGENT_SEQ")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let seq_len = explicit_seq
        .or_else(|| optimizer_resume.as_ref().and_then(|s| s.seq_len))
        .unwrap_or(default_seq)
        .min(config.max_seq_len);
    let explicit_batch = std::env::var("SCIAGENT_BATCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let batch_size = explicit_batch
        .or_else(|| optimizer_resume.as_ref().and_then(|s| s.batch_size))
        .unwrap_or(1)
        .max(1);
    let telemetry_interval = env_usize("SCIAGENT_TELEMETRY", 25).max(1);
    // No cap by default. This used to default to 16M, which silently trained on the
    // first 16M tokens of the corpus and threw the rest away — and because the shard
    // walk is alphabetical, that meant a crates.io corpus was really just the crates
    // starting with "a". A 1.03B-token corpus looked like it was in use (the
    // "streaming N tokens" line prints the full count) while the model overfit 1.6% of
    // it. Truncation is now opt-in and always announced.
    let max_tokens = env_usize("SCIAGENT_MAX_TOKENS", usize::MAX);

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
        let mut raw = loader.into_tokens();
        let original_len = raw.len();
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
        if max_tokens < original_len
        {
            println!(
                "streaming {} of {} tokens from BPE shards in {dir} \
                 (TRUNCATED to {:.1}% by SCIAGENT_MAX_TOKENS={max_tokens} — the shard walk is\n\
                 alphabetical, so a truncated corpus is a *prefix*, not a sample)",
                max_tokens,
                original_len,
                100.0 * max_tokens as f64 / original_len as f64
            );
            raw.truncate(max_tokens);
        }
        else
        {
            println!("streaming {} tokens from BPE shards in {dir}", original_len);
        }
        raw
    }
    else if let Ok(text) = std::env::var("SCIAGENT_TEXT")
    {
        assert!(
            config.vocab_size >= 256,
            "byte-level ingestion needs vocab_size >= 256 (got {}); use SCIAGENT_CONFIG=byte",
            config.vocab_size
        );
        let mut bytes = Vec::new();
        let mut seen = HashSet::new();
        read_bytes_recursive(Path::new(&text), &mut bytes, max_tokens, &mut seen);
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

    let corpus_tokens = tokens.len();
    let corpus_hash = token_stream_hash(&tokens);
    println!("corpus identity: {corpus_tokens} tokens | fnv64 {corpus_hash:016x}");
    if let Some(saved) = optimizer_resume.as_ref()
    {
        let mut mismatches = Vec::new();
        if let Some(v) = saved.seq_len
        {
            if v != seq_len
            {
                mismatches.push(format!("seq_len saved={v} current={seq_len}"));
            }
        }
        if let Some(v) = saved.batch_size
        {
            if v != batch_size
            {
                mismatches.push(format!("batch saved={v} current={batch_size}"));
            }
        }
        if let Some(v) = saved.corpus_tokens
        {
            if v != corpus_tokens
            {
                mismatches.push(format!("corpus_tokens saved={v} current={corpus_tokens}"));
            }
        }
        if let Some(v) = saved.corpus_hash
        {
            if v != corpus_hash
            {
                mismatches.push(format!(
                    "corpus_hash saved={v:016x} current={corpus_hash:016x}"
                ));
            }
        }
        match saved.split_version
        {
            Some(v) if v == WINDOW_SPLIT_VERSION =>
            {},
            Some(v) => mismatches.push(format!(
                "split_version saved={v} current={WINDOW_SPLIT_VERSION}"
            )),
            None if saved.step > 0 => mismatches.push(format!(
                "split_version saved=legacy-tail current={WINDOW_SPLIT_VERSION}"
            )),
            None =>
            {},
        }
        enforce_exact_resume(&mismatches);
    }

    // Exact resumes inherit the saved optimizer/LR trajectory. Explicit changes
    // are treated as branch experiments and require SCIAGENT_ALLOW_NONEXACT_RESUME=1. Legacy checkpoints (no AdamW sidecar)
    // keep the historical one-time re-warm that cushions their zero-moment restart.
    let steps_env = env_usize("SCIAGENT_STEPS", 300);
    let explicit_total_steps = std::env::var("SCIAGENT_TOTAL_STEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    if explicit_total_steps.is_none()
    {
        if let Some(saved) = optimizer_resume.as_ref()
        {
            if saved.total_steps <= start_step
            {
                eprintln!(
                    "checkpoint step {start_step} already reached saved target {}; set \
                     SCIAGENT_TOTAL_STEPS to a larger value plus \
                     SCIAGENT_ALLOW_NONEXACT_RESUME=1 to intentionally extend the run",
                    saved.total_steps
                );
                std::process::exit(0);
            }
        }
    }
    let total_steps = explicit_total_steps
        .filter(|&t| t > start_step)
        .or_else(|| {
            optimizer_resume
                .as_ref()
                .map(|s| s.total_steps)
                .filter(|&t| t > start_step)
        })
        .unwrap_or(start_step + steps_env);
    let run_len = total_steps.saturating_sub(start_step).max(1);
    let explicit_warmup = std::env::var("SCIAGENT_WARMUP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let warmup_steps = if let Some(extra) = explicit_warmup
    {
        start_step + extra
    }
    else if let Some(saved) = optimizer_resume.as_ref()
    {
        saved.warmup_steps
    }
    else
    {
        start_step + (run_len / 10).max(1)
    };
    // LR must key off model size for fresh/legacy runs. Exact resumes inherit the
    // saved schedule unless SCIAGENT_LR explicitly requests a new peak LR.
    let size_default_lr = if config.d_model <= 256 { 3e-3 } else { 3e-4 };
    let explicit_lr = std::env::var("SCIAGENT_LR")
        .ok()
        .and_then(|v| v.parse::<f32>().ok());
    let base_lr = explicit_lr
        .or_else(|| optimizer_resume.as_ref().map(|s| s.base_lr))
        .unwrap_or(size_default_lr);
    let min_lr = if explicit_lr.is_some()
    {
        base_lr * 0.1
    }
    else
    {
        optimizer_resume
            .as_ref()
            .map(|s| s.min_lr)
            .unwrap_or(base_lr * 0.1)
    };
    // Trajectory-changing settings inherit the saved run contract by default.
    let explicit_clip = std::env::var("SCIAGENT_CLIP")
        .ok()
        .and_then(|v| v.parse::<f32>().ok());
    let max_grad_norm = explicit_clip
        .or_else(|| optimizer_resume.as_ref().and_then(|s| s.max_grad_norm))
        .unwrap_or(1.0f32);
    let explicit_eps = std::env::var("SCIAGENT_EPS")
        .ok()
        .and_then(|v| v.parse::<f32>().ok());
    let adam_eps = explicit_eps
        .or_else(|| optimizer_resume.as_ref().map(|s| s.adam_eps))
        .unwrap_or(1e-5f32);
    let explicit_val_frac = std::env::var("SCIAGENT_VAL_FRAC")
        .ok()
        .and_then(|v| v.parse::<f32>().ok());
    let val_frac = explicit_val_frac
        .or_else(|| optimizer_resume.as_ref().and_then(|s| s.val_frac))
        .unwrap_or(0.02f32);
    // Checkpoint/telemetry cadence does not alter model math.
    let save_interval = env_usize("SCIAGENT_SAVE", 500);
    let keep_last = env_usize("SCIAGENT_KEEP", 3);
    let explicit_shuffle = std::env::var("SCIAGENT_SHUFFLE")
        .ok()
        .map(|v| !matches!(v.as_str(), "0" | "false"));
    let shuffle = explicit_shuffle
        .or_else(|| optimizer_resume.as_ref().and_then(|s| s.shuffle))
        .unwrap_or(true);
    if let Some(saved) = optimizer_resume.as_ref()
    {
        let mut mismatches = Vec::new();
        if total_steps != saved.total_steps
        {
            mismatches.push(format!(
                "total_steps saved={} current={total_steps}",
                saved.total_steps
            ));
        }
        if warmup_steps != saved.warmup_steps
        {
            mismatches.push(format!(
                "warmup_steps saved={} current={warmup_steps}",
                saved.warmup_steps
            ));
        }
        if !same_f32(base_lr, saved.base_lr)
        {
            mismatches.push(format!("base_lr saved={} current={base_lr}", saved.base_lr));
        }
        if !same_f32(min_lr, saved.min_lr)
        {
            mismatches.push(format!("min_lr saved={} current={min_lr}", saved.min_lr));
        }
        if !same_f32(adam_eps, saved.adam_eps)
        {
            mismatches.push(format!(
                "adam_eps saved={} current={adam_eps}",
                saved.adam_eps
            ));
        }
        if let Some(v) = saved.max_grad_norm
        {
            if !same_f32(max_grad_norm, v)
            {
                mismatches.push(format!("clip saved={v} current={max_grad_norm}"));
            }
        }
        if let Some(v) = saved.val_frac
        {
            if !same_f32(val_frac, v)
            {
                mismatches.push(format!("val_frac saved={v} current={val_frac}"));
            }
        }
        if let Some(v) = saved.shuffle
        {
            if shuffle != v
            {
                mismatches.push(format!("shuffle saved={v} current={shuffle}"));
            }
        }
        enforce_exact_resume(&mismatches);
    }

    let resume_betas = optimizer_resume.as_ref().map(|s| s.betas);
    let resume_weight_decay = optimizer_resume.as_ref().map(|s| s.weight_decay);
    let cfg = CudaPretrainConfig {
        base_lr,
        min_lr,
        warmup_steps,
        total_steps,
        start_step,
        seq_len,
        batch_size,
        telemetry_interval,
        betas: resume_betas.unwrap_or(CudaPretrainConfig::default().betas),
        weight_decay: resume_weight_decay.unwrap_or(0.0),
        adam_eps,
        log_interval: 25,
        save_interval,
        checkpoint_dir: ckpt_dir.clone(),
        max_grad_norm,
        val_frac,
        eval_interval: 100,
        keep_last,
        corpus_tokens,
        corpus_hash,
        shuffle,
        ..Default::default()
    };
    println!(
        "batch {batch_size} × seq_len {seq_len} | telemetry/{telemetry_interval} | steps {start_step}..{total_steps} | base_lr {base_lr:.1e} | \
         eps {adam_eps:.0e} | clip {max_grad_norm} | save/{save_interval} keep {keep_last} | \
         shuffle {shuffle} | ckpt → {ckpt_dir}\n"
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
