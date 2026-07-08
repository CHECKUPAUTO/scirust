//! **Resident LoRA / DoRA fine-tuning** — adapt a frozen base `SciAgentModel` on
//! the fully-resident GPU path with small q/k/v/o adapters, then merge them back
//! into a plain fine-tuned model. The base never moves; only the adapters train
//! (tiny optimizer footprint), which is the resident path's natural strength now
//! that full-weight training is throughput-bound at scale.
//!
//! Environment:
//! - `SCIAGENT_ADAPTER` — `lora` (default) or `dora`. DoRA additionally learns a
//!   per-row magnitude (direction/magnitude decomposition), often higher quality
//!   at the same rank.
//! - `SCIAGENT_BASE_CKPT` — a checkpoint dir (`step_N/`, as written by
//!   `resident_pretrain`) to load the **base** model from. Without it a fresh
//!   `byte` model is used (the loss still drops, but there's nothing meaningful
//!   to adapt — set this for a real fine-tune).
//! - `SCIAGENT_TEXT` — a file/dir ingested **byte-level** (vocab 256) as the
//!   fine-tuning corpus; otherwise a small synthetic pattern.
//! - `SCIAGENT_CKPT` (default `checkpoints/lora-merged`) — where the **merged**
//!   fine-tuned model is written.
//! - `SCIAGENT_RANK` (8), `SCIAGENT_ALPHA` (16), `SCIAGENT_STEPS` (200),
//!   `SCIAGENT_SEQ` (64), `SCIAGENT_LR` (1e-3) — run knobs.
//!
//! ```text
//! # adapt a pretrained base on your code, no tokenizer:
//! SCIAGENT_BASE_CKPT=checkpoints/cervo-bytes/step_2000 SCIAGENT_TEXT=$HOME/CERVO \
//!   cargo run -p scirust-sciagent --features gpu --release --example resident_lora_finetune
//! ```
//!
//! Exit code 2 means no GPU adapter was found — run on the Thor or install a
//! Vulkan ICD.

use std::path::Path;

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::gpu::{DoraConfig, LoraConfig, ResidentDoraModel, ResidentLoraModel};
use scirust_sciagent::model::SciAgentModel;
use scirust_sciagent::train::checkpoint::{
    CheckpointMeta, latest_checkpoint, load_checkpoint, read_meta, save_checkpoint,
};

/// A LoRA or DoRA resident fine-tuner, chosen by `SCIAGENT_ADAPTER`. Both expose
/// the same train/sync/forward surface, so the loop below is adapter-agnostic.
enum Finetuner {
    Lora(ResidentLoraModel),
    Dora(ResidentDoraModel),
}

impl Finetuner {
    fn kind(&self) -> &'static str {
        match self
        {
            Self::Lora(_) => "LoRA",
            Self::Dora(_) => "DoRA",
        }
    }
    fn adapter_name(&self) -> &str {
        match self
        {
            Self::Lora(m) => m.adapter_name(),
            Self::Dora(m) => m.adapter_name(),
        }
    }
    fn train_step(&mut self, i: &[u32], t: &[u32], lr: f32, b: (f32, f32), e: f32, wd: f32) -> f32 {
        match self
        {
            Self::Lora(m) => m.train_step(i, t, lr, b, e, wd),
            Self::Dora(m) => m.train_step(i, t, lr, b, e, wd),
        }
    }
    fn sync_to_model(&self, model: &mut SciAgentModel) {
        match self
        {
            Self::Lora(m) => m.sync_to_model(model),
            Self::Dora(m) => m.sync_to_model(model),
        }
    }
}

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

fn main() {
    // Base model: from a checkpoint (real fine-tune) or a fresh byte model (demo).
    let mut model;
    if let Ok(base) = std::env::var("SCIAGENT_BASE_CKPT")
    {
        let dir = latest_checkpoint(Path::new(&base)).unwrap_or_else(|| base.clone().into());
        let cfg =
            read_meta(&dir).unwrap_or_else(|e| panic!("read base meta {}: {e}", dir.display()));
        model = SciAgentModel::new(&cfg.config);
        load_checkpoint(&mut model, &dir)
            .unwrap_or_else(|e| panic!("load base {}: {e}", dir.display()));
        println!(
            "base model loaded from {} (vocab {})",
            dir.display(),
            cfg.config.vocab_size
        );
        assert!(
            cfg.config.tie_embeddings,
            "resident LoRA requires a tied-embedding base model"
        );
    }
    else
    {
        model = SciAgentModel::new(&byte_config());
        println!("no SCIAGENT_BASE_CKPT — fresh byte model (demo; set it for a real fine-tune)");
    }
    let config = model.config.clone();

    let rank = env_usize("SCIAGENT_RANK", 8);
    let alpha = env_f32("SCIAGENT_ALPHA", 16.0);
    let kind = std::env::var("SCIAGENT_ADAPTER").unwrap_or_else(|_| "lora".into());
    let built = match kind.to_ascii_lowercase().as_str()
    {
        "dora" => ResidentDoraModel::from_model(&model, DoraConfig { rank }).map(Finetuner::Dora),
        _ => ResidentLoraModel::from_model(&model, LoraConfig { rank, alpha }).map(Finetuner::Lora),
    };
    let Some(mut rm) = built
    else
    {
        eprintln!("no GPU adapter available. Install a Vulkan ICD or run on the Jetson Thor.");
        std::process::exit(2);
    };
    println!(
        "resident {} fine-tuning on: {}",
        rm.kind(),
        rm.adapter_name()
    );
    println!("rank {rank} on q/k/v/o (base frozen)\n");

    // Fine-tuning corpus: byte-level text or a synthetic pattern.
    let seq = env_usize("SCIAGENT_SEQ", 64).min(config.max_seq_len);
    let tokens: Vec<u32> = match std::env::var("SCIAGENT_TEXT")
    {
        Ok(text) =>
        {
            assert!(config.vocab_size >= 256, "byte-level needs vocab >= 256");
            let mut bytes = Vec::new();
            read_bytes_recursive(Path::new(&text), &mut bytes, 2_000_000);
            assert!(!bytes.is_empty(), "SCIAGENT_TEXT={text} yielded no bytes");
            println!("byte-level corpus: {} tokens from {text}", bytes.len());
            bytes.into_iter().map(u32::from).collect()
        },
        Err(_) =>
        {
            let pattern: Vec<u32> = (0..24u32)
                .map(|i| (i * 9 + 3) % config.vocab_size as u32)
                .collect();
            let t: Vec<u32> = (0..seq * 300).map(|i| pattern[i % pattern.len()]).collect();
            println!("no SCIAGENT_TEXT — synthetic corpus of {} tokens", t.len());
            t
        },
    };

    let steps = env_usize("SCIAGENT_STEPS", 200);
    let lr = env_f32("SCIAGENT_LR", 1e-3);
    let betas = (0.9, 0.999);
    let mut losses = Vec::new();
    let mut cursor = 0usize;
    for step in 0..steps
    {
        if cursor + seq + 1 > tokens.len()
        {
            cursor = 0;
        }
        let inputs = &tokens[cursor..cursor + seq];
        let targets = &tokens[cursor + 1..cursor + seq + 1];
        let loss = rm.train_step(inputs, targets, lr, betas, 1e-8, 0.0);
        losses.push(loss);
        cursor += seq;
        if step % 20 == 0 || step + 1 == steps
        {
            println!("[lora step {step:>4}] loss {loss:>9.4}");
        }
    }

    let n = losses.len().clamp(1, 5);
    let first: f32 = losses[..n].iter().sum::<f32>() / n as f32;
    let last: f32 = losses[losses.len() - n..].iter().sum::<f32>() / n as f32;
    println!(
        "\n{} LoRA steps: loss {first:.4} -> {last:.4}  ({:.1}% reduction)",
        losses.len(),
        (1.0 - last / first) * 100.0
    );

    // Merge the adapters into the base and save a plain fine-tuned model.
    rm.sync_to_model(&mut model);
    let out = std::env::var("SCIAGENT_CKPT").unwrap_or_else(|_| "checkpoints/lora-merged".into());
    let meta = CheckpointMeta {
        step: steps,
        loss: last,
        lr,
        config: config.clone(),
    };
    let dir = Path::new(&out).join("merged");
    match save_checkpoint(&model, &meta, &dir)
    {
        Ok(()) => println!("merged fine-tuned model saved → {}", dir.display()),
        Err(e) => eprintln!("save failed: {e}"),
    }
}
