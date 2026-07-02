use std::path::PathBuf;

use clap::Parser;
use scirust_core::autodiff::reverse::Tape;
use scirust_core::autodiff::scheduler::LrSchedule;
use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::model::SciAgentModel;
use scirust_sciagent::train::TrainerConfig;
use scirust_sciagent::train::checkpoint::{
    CheckpointMeta, latest_checkpoint, load_checkpoint, save_checkpoint,
};
use scirust_sciagent::train::dataset::{PretrainDataset, ShardLoader};
use scirust_sciagent::train::optimizer::TrainOptimizer;
use scirust_sciagent::train::scheduler::WarmupCosineSchedule;

#[derive(Parser)]
#[command(name = "sciagent-train", about = "SCIAGENT SLM training binary")]
struct Args {
    #[arg(long, default_value = "350m")]
    model: String,

    #[arg(long, default_value_t = 32768)]
    vocab_size: usize,

    #[arg(long, default_value_t = 1024)]
    d_model: usize,

    #[arg(long, default_value_t = 24)]
    n_layers: usize,

    #[arg(long, default_value_t = 16)]
    n_heads: usize,

    #[arg(long, default_value_t = 4)]
    n_kv_heads: usize,

    #[arg(long, default_value_t = 2816)]
    d_ff: usize,

    #[arg(long, default_value_t = 8192)]
    max_seq_len: usize,

    #[arg(long, default_value_t = 1_000_000.0)]
    rope_theta: f32,

    #[arg(long, default_value_t = 3e-4)]
    lr: f32,

    #[arg(long, default_value_t = 3e-5)]
    min_lr: f32,

    #[arg(long, default_value_t = 2000)]
    warmup_steps: usize,

    #[arg(long, default_value_t = 50000)]
    total_steps: usize,

    #[arg(long, default_value_t = 8)]
    micro_batch_size: usize,

    #[arg(long, default_value_t = 8)]
    grad_accum_steps: usize,

    #[arg(long, default_value_t = 1.0)]
    max_grad_norm: f32,

    #[arg(long, default_value_t = 100)]
    log_interval: usize,

    #[arg(long, default_value_t = 500)]
    save_interval: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long, default_value = "./checkpoints")]
    checkpoint_dir: PathBuf,

    #[arg(long)]
    resume: bool,

    #[arg(long)]
    data: Option<PathBuf>,

    #[arg(long)]
    data_dir: Option<PathBuf>,

    #[arg(long, default_value_t = 64)]
    effective_batch_size: usize,

    #[arg(long, default_value_t = 0)]
    start_step: usize,
}

fn main() {
    let args = Args::parse();

    let config = match args.model.as_str()
    {
        "debug" => SciAgentConfig::debug(),
        "small" | "Small" => SciAgentConfig::small(),
        "350m" | "350M" => SciAgentConfig::sciagent_350m(),
        "7b" | "7B" => SciAgentConfig::sciagent_7b(),
        _ => SciAgentConfig {
            vocab_size: args.vocab_size,
            d_model: args.d_model,
            n_layers: args.n_layers,
            n_heads: args.n_heads,
            n_kv_heads: args.n_kv_heads,
            d_ff: args.d_ff,
            max_seq_len: args.max_seq_len,
            rope_theta: args.rope_theta,
            tie_embeddings: true,
            use_bias: false,
            eps: 1e-5,
        },
    };

    let trainer_cfg = TrainerConfig {
        lr: args.lr,
        min_lr: args.min_lr,
        warmup_steps: args.warmup_steps,
        total_steps: args.total_steps,
        batch_size: args.effective_batch_size,
        seq_len: args.max_seq_len,
        micro_batch_size: args.micro_batch_size,
        grad_accum_steps: args.grad_accum_steps,
        max_grad_norm: args.max_grad_norm,
        log_interval: args.log_interval,
        save_interval: args.save_interval,
        seed: args.seed,
        checkpoint_dir: args.checkpoint_dir.to_string_lossy().to_string(),
    };

    let total_params = config.total_parameters();
    println!("=== SCIAGENT Training ===");
    println!("Model: {} params", format_params(total_params));
    println!("Vocab size: {}", config.vocab_size);
    println!(
        "d_model={}, n_layers={}, n_heads={}, n_kv_heads={}",
        config.d_model, config.n_layers, config.n_heads, config.n_kv_heads
    );
    println!("d_ff={}, max_seq_len={}", config.d_ff, config.max_seq_len);
    println!(
        "LR: {:.2e} → {:.2e}, warmup {} steps, total {} steps",
        trainer_cfg.lr, trainer_cfg.min_lr, trainer_cfg.warmup_steps, trainer_cfg.total_steps
    );
    println!(
        "Effective batch: {}, micro-batch: {}, grad accum: {}",
        trainer_cfg.batch_size, trainer_cfg.micro_batch_size, trainer_cfg.grad_accum_steps
    );
    println!("Checkpoint dir: {}", trainer_cfg.checkpoint_dir);
    println!();

    let mut model = SciAgentModel::new(&config);
    let mut start_step = 0;

    if args.resume
    {
        if let Some(latest) = latest_checkpoint(&args.checkpoint_dir)
        {
            match load_checkpoint(&mut model, &latest)
            {
                Ok(meta) =>
                {
                    println!("Resumed from step {} (loss: {:.4})", meta.step, meta.loss);
                    start_step = meta.step;
                },
                Err(e) =>
                {
                    eprintln!("Failed to load checkpoint: {e}");
                    std::process::exit(1);
                },
            }
        }
        else
        {
            eprintln!("No checkpoint found in {:?}", args.checkpoint_dir);
            std::process::exit(1);
        }
    }
    else if args.start_step > 0
    {
        start_step = args.start_step;
    }

    let dataset = if let Some(dir) = &args.data_dir
    {
        let mut loader = ShardLoader::new();
        loader.load_dir(dir).expect("Cannot load shard directory");
        eprintln!(
            "Loaded {} tokens from shard dir {:?}",
            loader.total_tokens(),
            dir
        );
        let mut ds = loader.into_dataset(config.max_seq_len, config.vocab_size);
        ds.shuffle(args.seed);
        ds
    }
    else if let Some(path) = &args.data
    {
        let raw = std::fs::read(path).expect("Cannot read data file");
        let tokens: Vec<u32> = raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mut ds = PretrainDataset::from_slice(&tokens, config.max_seq_len, config.vocab_size);
        ds.shuffle(args.seed);
        ds
    }
    else
    {
        eprintln!("No --data or --data-dir provided, using synthetic data for smoke test");
        let synthetic: Vec<u32> = (0u32..config.max_seq_len as u32 * 100)
            .map(|i| i % config.vocab_size as u32)
            .collect();
        PretrainDataset::from_slice(&synthetic, config.max_seq_len, config.vocab_size)
    };

    println!("Dataset ready ({} sequences)", dataset.len());

    let mut opt = TrainOptimizer::new_muon(trainer_cfg.lr);
    let scheduler = WarmupCosineSchedule::new(
        trainer_cfg.lr,
        trainer_cfg.min_lr,
        trainer_cfg.warmup_steps,
        trainer_cfg.total_steps,
    );

    let mut dataset = dataset;
    let mut total_loss = 0.0f64;

    for step in start_step..trainer_cfg.total_steps
    {
        let tape = Tape::new();

        let mut step_loss = 0.0f32;
        for _ in 0..trainer_cfg.grad_accum_steps
        {
            let (inputs, targets) = dataset
                .next_batch(trainer_cfg.micro_batch_size)
                .unwrap_or_else(|| {
                    dataset.shuffle(args.seed);
                    dataset
                        .next_batch(trainer_cfg.micro_batch_size)
                        .expect("Dataset too small for a single batch")
                });

            let logits = model.forward(&tape, &inputs, trainer_cfg.seq_len);
            let loss = scirust_sciagent::train::cross_entropy_loss(&tape, logits, &targets);
            loss.backward();
            step_loss += tape.value(loss.idx()).data[0];
        }
        step_loss /= trainer_cfg.grad_accum_steps as f32;

        let lr = scheduler.lr_at(step);
        opt.set_lr(lr);
        if trainer_cfg.max_grad_norm > 0.0
        {
            opt.clip_grad_norm(&tape, trainer_cfg.max_grad_norm);
        }

        let params = model.parameter_indices();
        opt.step(&params, &tape);
        model.sync(&tape);

        total_loss += step_loss as f64;

        if step % trainer_cfg.log_interval == 0
        {
            let avg = total_loss / ((step - start_step + 1) as f64);
            println!("[Step {step}] loss: {step_loss:.6} | avg: {avg:.6} | lr: {lr:.8}");
        }

        if step > 0 && step % trainer_cfg.save_interval == 0
        {
            let ckpt_dir = args.checkpoint_dir.join(format!("step_{step}"));
            let meta = CheckpointMeta {
                step,
                loss: step_loss,
                lr,
                config: config.clone(),
            };
            if let Err(e) = save_checkpoint(&model, &meta, &ckpt_dir)
            {
                eprintln!("Failed to save checkpoint: {e}");
            }
            else
            {
                println!("Checkpoint saved to {:?}", ckpt_dir);
            }
        }
    }

    // Save final checkpoint
    let final_dir = args.checkpoint_dir.join("final");
    let meta = CheckpointMeta {
        step: trainer_cfg.total_steps,
        loss: total_loss as f32 / (trainer_cfg.total_steps - start_step).max(1) as f32,
        lr: trainer_cfg.min_lr,
        config: config.clone(),
    };
    if let Err(e) = save_checkpoint(&model, &meta, &final_dir)
    {
        eprintln!("Failed to save final checkpoint: {e}");
    }
    else
    {
        println!("Final checkpoint saved to {:?}", final_dir);
    }

    println!("Training complete.");
}

fn format_params(n: usize) -> String {
    if n >= 1_000_000_000
    {
        format!("{:.1}B", n as f64 / 1e9)
    }
    else if n >= 1_000_000
    {
        format!("{:.1}M", n as f64 / 1e6)
    }
    else
    {
        format!("{n}")
    }
}
