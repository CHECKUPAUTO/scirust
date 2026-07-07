//! **Production-scale resident pretraining** — the full run harness on the
//! fully-resident GPU path (`ResidentModel::pretrain`): real token-shard
//! streaming, a warmup + cosine LR schedule, and periodic safetensors
//! checkpointing, all in VRAM (the path that beats the per-op tape ~4× on the
//! Jetson Thor).
//!
//!   # synthetic corpus (self-contained smoke run):
//!   cargo run -p scirust-sciagent --features gpu --release --example resident_pretrain
//!
//!   # real shards (a directory of little-endian u32 `.bin` token files, e.g.
//!   # produced by the `byte-shard` / tokenizer tooling):
//!   SCIAGENT_SHARDS=/path/to/shards \
//!     cargo run -p scirust-sciagent --features gpu --release --example resident_pretrain
//!
//! Checkpoints land in `SCIAGENT_CKPT` (default `checkpoints/resident`); on
//! start-up the newest `step_N/` there is loaded and training resumes from it
//! (the LR schedule continues from `meta.step`; the AdamW moments restart from
//! zero, which the warmup re-absorbs). Exit code 2 means no GPU adapter was
//! found — run on the Thor or install a Vulkan ICD.

use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::gpu::{ResidentModel, ResidentPretrainConfig};
use scirust_sciagent::model::SciAgentModel;
use scirust_sciagent::train::checkpoint::{latest_checkpoint, load_checkpoint};
use scirust_sciagent::train::dataset::ShardLoader;

fn main() {
    // A tied-embedding config (the resident path uses E as the LM head). Small
    // enough to converge fast in a demo; scale d_model / n_layers / vocab up for
    // the real 350M run — the harness is identical.
    let config = SciAgentConfig {
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
    };
    let mut model = SciAgentModel::new(&config);

    // Resume from the newest checkpoint, if one exists.
    let ckpt_dir = std::env::var("SCIAGENT_CKPT").unwrap_or_else(|_| "checkpoints/resident".into());
    let mut start_step = 0usize;
    if let Some(latest) = latest_checkpoint(std::path::Path::new(&ckpt_dir))
    {
        match load_checkpoint(&mut model, &latest)
        {
            Ok(meta) =>
            {
                start_step = meta.step;
                println!(
                    "resuming from {} (step {}, loss {:.4})",
                    latest.display(),
                    meta.step,
                    meta.loss
                );
            },
            Err(e) => eprintln!("could not load {}: {e}; starting fresh", latest.display()),
        }
    }

    let Some(mut rm) = ResidentModel::from_model(&model)
    else
    {
        eprintln!("no GPU adapter available. Install a Vulkan ICD or run on the Jetson Thor.");
        std::process::exit(2);
    };
    rm.reset_step(); // fresh AdamW moments; the LR schedule continues via start_step
    println!("resident pretraining on: {}\n", rm.adapter_name());

    // Token stream: real shards if SCIAGENT_SHARDS points at a dir of `.bin`
    // files, otherwise a learnable synthetic corpus so the demo is self-contained.
    let seq_len = 128usize;
    let tokens: Vec<u32> = match std::env::var("SCIAGENT_SHARDS")
    {
        Ok(dir) =>
        {
            let mut loader = ShardLoader::new();
            loader
                .load_dir(&dir)
                .unwrap_or_else(|e| panic!("failed to load shards from {dir}: {e}"));
            println!(
                "streaming {} tokens from shards in {dir}",
                loader.total_tokens()
            );
            let mut buf = loader.into_dataset(seq_len, config.vocab_size);
            // Drain the whole (sanitised) corpus into a flat stream.
            let mut all = Vec::new();
            while let Some((inputs, _)) = buf.next_batch(1)
            {
                if all.len() > 4_000_000
                {
                    break;
                }
                all.extend(inputs.into_iter().map(|t| t as u32));
            }
            all
        },
        Err(_) =>
        {
            let pattern: Vec<u32> = (0..48u32)
                .map(|i| (i * 11 + 5) % config.vocab_size as u32)
                .collect();
            let toks: Vec<u32> = (0..seq_len * 400)
                .map(|i| pattern[i % pattern.len()])
                .collect();
            println!(
                "no SCIAGENT_SHARDS set — using a synthetic corpus of {} tokens",
                toks.len()
            );
            toks
        },
    };

    let total_steps = start_step + 300;
    let cfg = ResidentPretrainConfig {
        base_lr: 3e-3,
        min_lr: 3e-4,
        warmup_steps: start_step + 30,
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
        "model: d {}, {} layers, {}h/{}kv, d_ff {}, vocab {} | steps {}..{} | ckpt → {}\n",
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.n_kv_heads,
        config.d_ff,
        config.vocab_size,
        start_step,
        total_steps,
        ckpt_dir
    );

    let losses = rm.pretrain(&tokens, &mut model, &config, &cfg);
    if losses.is_empty()
    {
        eprintln!("no steps ran (corpus too short?)");
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
