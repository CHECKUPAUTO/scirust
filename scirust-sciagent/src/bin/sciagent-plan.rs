//! Size a training run before writing a single kernel.
//!
//! Prints the first-order training-memory budget for a SCIAGENT config and
//! tells you whether it fits a memory ceiling (default: a 128 GB Jetson Thor),
//! across the flash-attention / activation-checkpointing / precision matrix.

use clap::Parser;
use scirust_sciagent::config::SciAgentConfig;
use scirust_sciagent::planning::{Precision, estimate, gib, max_seq_len_that_fits};

#[derive(Parser)]
#[command(
    name = "sciagent-plan",
    about = "Training memory planner (Jetson Thor by default)"
)]
struct Args {
    /// Config: debug | small | 350m | 7b.
    #[arg(long, default_value = "350m")]
    model: String,

    /// Sequence length to plan for.
    #[arg(long, default_value_t = 8192)]
    seq_len: usize,

    /// Micro-batch size (sequences per forward).
    #[arg(long, default_value_t = 1)]
    batch: usize,

    /// Memory ceiling in GB (Jetson Thor dev kit = 128).
    #[arg(long, default_value_t = 128)]
    ceiling_gb: u64,

    /// Precision: fp32 | bf16.
    #[arg(long, default_value = "bf16")]
    precision: String,
}

fn config(name: &str) -> SciAgentConfig {
    match name
    {
        "debug" => SciAgentConfig::debug(),
        "small" => SciAgentConfig::small(),
        "350m" | "350M" => SciAgentConfig::sciagent_350m(),
        "7b" | "7B" => SciAgentConfig::sciagent_7b(),
        _ =>
        {
            eprintln!("Unknown model '{name}', using 350m");
            SciAgentConfig::sciagent_350m()
        },
    }
}

fn main() {
    let args = Args::parse();
    let cfg = config(&args.model);
    let prec = match args.precision.as_str()
    {
        "fp32" => Precision::fp32(),
        _ => Precision::mixed_bf16(),
    };
    let ceiling = args.ceiling_gb * (1u64 << 30);
    let params = cfg.total_parameters();

    println!("=== SCIAGENT training memory plan ===");
    println!(
        "model {} : {:.1}M params | seq {} | batch {} | {} | ceiling {} GB",
        args.model,
        params as f64 / 1e6,
        args.seq_len,
        args.batch,
        args.precision,
        args.ceiling_gb
    );
    println!();
    println!(
        "{:<28} {:>10} {:>10} {:>8}",
        "configuration", "activ. GB", "total GB", "fits?"
    );

    let modes = [
        ("naive (no flash, no ckpt)", false, false),
        ("flash-attention", true, false),
        ("flash + checkpointing", true, true),
    ];
    for (label, flash, ckpt) in modes
    {
        let b = estimate(&cfg, args.seq_len, args.batch, prec, flash, ckpt);
        println!(
            "{:<28} {:>10.1} {:>10.1} {:>8}",
            label,
            gib(b.activations),
            b.total_gib(),
            if b.fits(ceiling) { "yes" } else { "NO" }
        );
    }

    println!();
    let best = max_seq_len_that_fits(&cfg, args.batch, prec, true, true, ceiling);
    match best
    {
        Some(s) => println!(
            "Longest sequence that fits {ceiling_gb} GB (flash + ckpt, {precision}): {s} tokens",
            ceiling_gb = args.ceiling_gb,
            precision = args.precision
        ),
        None => println!(
            "Even seq 256 does not fit {} GB for this config.",
            args.ceiling_gb
        ),
    }
    println!("(First-order estimate: exact asymptotics, approximate linear constants.)");
}
