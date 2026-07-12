use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use clap::Parser;
use scirust_sciagent::bpe::BpeTrainer;
use scirust_sciagent::train::dataset::{
    matches_extension, parse_extensions, skip_source_dir, source_quality,
};

#[derive(Parser)]
#[command(
    name = "train-tokenizer",
    about = "Train BPE tokenizer on Rust source code"
)]
struct Args {
    #[arg(short, long)]
    input: Vec<String>,

    #[arg(short, long, default_value = "32768")]
    vocab_size: usize,

    #[arg(short, long)]
    output: String,

    #[arg(long, default_value_t = 2)]
    min_frequency: u32,

    #[arg(long)]
    recursive: bool,

    /// Comma-separated source extensions to train on (e.g. `rs,md,toml,py`).
    #[arg(long, default_value = "rs")]
    extension: String,

    /// Disable the corpus-quality filter (keep generated/minified/data-table files).
    /// Off by default. Must match the `collect-data` setting so the tokenizer sees
    /// the same corpus the shards are built from.
    #[arg(long)]
    no_quality_filter: bool,
}

fn main() {
    let args = Args::parse();
    let exts = parse_extensions(&args.extension);
    eprintln!("Training on extensions: {exts:?}");
    let filter = !args.no_quality_filter;
    eprintln!(
        "corpus-quality filter: {}",
        if filter
        {
            "on"
        }
        else
        {
            "OFF (--no-quality-filter)"
        }
    );
    let mut all_texts: Vec<String> = Vec::new();
    let mut skipped: BTreeMap<&'static str, usize> = BTreeMap::new();

    for path in &args.input
    {
        let p = Path::new(path);
        if p.is_file()
        {
            if let Ok(content) = fs::read_to_string(p)
            {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                match (filter, source_quality(name, &content))
                {
                    (true, Err(reason)) =>
                    {
                        *skipped.entry(reason).or_insert(0) += 1;
                    },
                    _ => all_texts.push(content),
                }
            }
        }
        else if p.is_dir() && args.recursive
        {
            collect_dir(p, &exts, filter, &mut all_texts, &mut skipped);
        }
    }

    eprintln!(
        "Collected {} files, {} chars | skipped {} | reasons {:?}",
        all_texts.len(),
        all_texts.iter().map(|s| s.len()).sum::<usize>(),
        skipped.values().sum::<usize>(),
        skipped
    );

    let trainer = BpeTrainer::new(args.vocab_size).min_frequency(args.min_frequency);
    let tokenizer = trainer.train(&all_texts);

    tokenizer
        .save_json(&args.output)
        .expect("Failed to save tokenizer");
    eprintln!(
        "Tokenizer saved to {} (vocab size: {})",
        args.output,
        tokenizer.vocab_size()
    );
}

fn collect_dir(
    dir: &Path,
    exts: &[String],
    filter: bool,
    texts: &mut Vec<String>,
    skipped: &mut BTreeMap<&'static str, usize>,
) {
    if let Ok(entries) = fs::read_dir(dir)
    {
        // Deterministic order (see collect-data): `read_dir` is OS-arbitrary, which
        // would make the trained tokenizer irreproducible across machines.
        let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        paths.sort();
        for path in paths
        {
            if path.is_dir()
            {
                if let Some(name) = path.file_name().and_then(|n| n.to_str())
                {
                    if skip_source_dir(name)
                    {
                        continue;
                    }
                }
                collect_dir(&path, exts, filter, texts, skipped);
            }
            else if matches_extension(&path, exts)
            {
                if let Ok(content) = fs::read_to_string(&path)
                {
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    match (filter, source_quality(name, &content))
                    {
                        (true, Err(reason)) =>
                        {
                            *skipped.entry(reason).or_insert(0) += 1;
                        },
                        _ => texts.push(content),
                    }
                }
            }
        }
    }
}
