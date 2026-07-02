//! Held-out evaluation for SCIAGENT checkpoints.
//!
//! Reports mean next-token cross-entropy and perplexity over a directory of
//! token shards, with NO gradient step and NO shuffling — a deterministic
//! measurement. Point `--data-dir` at shards the model was NOT trained on to
//! measure generalization (a small train/held-out gap means the model learned
//! structure rather than memorizing).

use std::path::PathBuf;

use clap::Parser;
use scirust_core::autodiff::reverse::Tape;
use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::model::SciAgentModel;
use scirust_sciagent::train::checkpoint::load_checkpoint;
use scirust_sciagent::train::cross_entropy_loss;
use scirust_sciagent::train::dataset::ShardLoader;

#[derive(Parser)]
#[command(name = "sciagent-eval", about = "Held-out perplexity for a checkpoint")]
struct Args {
    /// Checkpoint directory (contains meta.json + model.safetensors).
    #[arg(long)]
    checkpoint: PathBuf,

    /// Directory of `*.bin` token shards to evaluate on.
    #[arg(long)]
    data_dir: PathBuf,

    /// Sequences per forward pass.
    #[arg(long, default_value_t = 8)]
    batch_size: usize,

    /// Stop after this many batches (0 = all). Bounds wall-clock on big sets.
    #[arg(long, default_value_t = 0)]
    max_batches: usize,
}

fn config_from_meta(path: &std::path::Path) -> Option<SciAgentConfig> {
    let meta_str = std::fs::read_to_string(path.join("meta.json")).ok()?;
    let meta: serde_json::Value = serde_json::from_str(&meta_str).ok()?;
    let cfg = &meta["config"];
    Some(SciAgentConfig {
        vocab_size: cfg["vocab_size"].as_u64()? as usize,
        d_model: cfg["d_model"].as_u64()? as usize,
        n_layers: cfg["n_layers"].as_u64()? as usize,
        n_heads: cfg["n_heads"].as_u64()? as usize,
        n_kv_heads: cfg["n_kv_heads"].as_u64()? as usize,
        d_ff: cfg["d_ff"].as_u64()? as usize,
        max_seq_len: cfg["max_seq_len"].as_u64()? as usize,
        rope_theta: cfg["rope_theta"].as_f64()? as f32,
        tie_embeddings: cfg["tie_embeddings"].as_bool()?,
        use_bias: cfg["use_bias"].as_bool().unwrap_or(false),
        eps: cfg["eps"].as_f64().unwrap_or(1e-5) as f32,
    })
}

fn main() {
    let args = Args::parse();

    let config = config_from_meta(&args.checkpoint).unwrap_or_else(|| {
        eprintln!("Cannot read config from {:?}/meta.json", args.checkpoint);
        std::process::exit(1);
    });

    let mut model = SciAgentModel::new(&config);
    let meta = match load_checkpoint(&mut model, &args.checkpoint)
    {
        Ok(m) => m,
        Err(e) =>
        {
            eprintln!("Failed to load checkpoint: {e}");
            std::process::exit(1);
        },
    };

    // Deterministic eval: load shards in filename order, no shuffle.
    let mut loader = ShardLoader::new();
    loader
        .load_dir(&args.data_dir)
        .expect("Cannot load eval shard directory");
    let total_tokens = loader.total_tokens();
    let seq_len = config.max_seq_len;
    let mut dataset = loader.into_dataset(seq_len, config.vocab_size);

    println!("=== SCIAGENT eval ===");
    println!(
        "Checkpoint: step {} (train loss {:.4})",
        meta.step, meta.loss
    );
    println!(
        "Eval set: {} tokens, {} sequences of len {}",
        total_tokens,
        dataset.len(),
        seq_len
    );

    // `next_batch` wraps around forever, so bound the loop to exactly one
    // pass over the set (or `--max-batches`, whichever is smaller).
    let full_batches = dataset.len() / args.batch_size.max(1);
    let target_batches = if args.max_batches > 0
    {
        full_batches.min(args.max_batches)
    }
    else
    {
        full_batches
    };

    // Token-weighted mean of the per-batch cross-entropy. cross_entropy_loss
    // already averages over the batch's rows; weighting by row count keeps the
    // corpus mean exact even if some batch is short.
    let mut sum_loss = 0.0f64;
    let mut n_rows = 0usize;
    let mut batches = 0usize;

    while batches < target_batches
    {
        let Some((inputs, targets)) = dataset.next_batch(args.batch_size)
        else
        {
            break;
        };
        let tape = Tape::new();
        let logits = model.forward(&tape, &inputs, seq_len);
        let loss = cross_entropy_loss(&tape, logits, &targets);
        let rows = targets.len();
        sum_loss += tape.value(loss.idx()).data[0] as f64 * rows as f64;
        n_rows += rows;
        batches += 1;

        if batches % 20 == 0
        {
            let m = sum_loss / n_rows as f64;
            println!(
                "  [{batches}/{target_batches} batches] running loss {:.4} | ppl {:.2}",
                m,
                m.exp()
            );
        }
    }

    if n_rows == 0
    {
        eprintln!("No eval batches produced (set too small for batch_size?)");
        std::process::exit(1);
    }

    let mean_loss = sum_loss / n_rows as f64;
    println!("---");
    println!(
        "held-out loss: {:.4}  |  perplexity: {:.2}  ({} tokens over {} batches)",
        mean_loss,
        mean_loss.exp(),
        n_rows,
        batches
    );
}
