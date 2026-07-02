use std::fs;
use std::path::Path;

use clap::Parser;
use scirust_sciagent::bpe::BpeTrainer;

#[derive(Parser)]
#[command(name = "train-tokenizer", about = "Train BPE tokenizer on Rust source code")]
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

    #[arg(long, default_value = "rs")]
    extension: String,
}

fn main() {
    let args = Args::parse();
    let mut all_texts: Vec<String> = Vec::new();

    for path in &args.input {
        let p = Path::new(path);
        if p.is_file() {
            if let Ok(content) = fs::read_to_string(p) {
                all_texts.push(content);
            }
        } else if p.is_dir() && args.recursive {
            collect_dir(p, &args.extension, &mut all_texts);
        }
    }

    eprintln!("Collected {} files, {} chars", all_texts.len(),
              all_texts.iter().map(|s| s.len()).sum::<usize>());

    let trainer = BpeTrainer::new(args.vocab_size).min_frequency(args.min_frequency);
    let tokenizer = trainer.train(&all_texts);

    tokenizer.save_json(&args.output).expect("Failed to save tokenizer");
    eprintln!("Tokenizer saved to {} (vocab size: {})", args.output, tokenizer.vocab_size());
}

fn collect_dir(dir: &Path, ext: &str, texts: &mut Vec<String>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_dir(&path, ext, texts);
            } else if path.extension().and_then(|e| e.to_str()) == Some(ext) {
                if let Ok(content) = fs::read_to_string(&path) {
                    texts.push(content);
                }
            }
        }
    }
}
