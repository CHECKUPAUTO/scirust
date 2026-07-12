use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use std::collections::BTreeMap;

use clap::Parser;
use scirust_sciagent::bpe::BpeTokenizer;
use scirust_sciagent::train::dataset::{
    matches_extension, parse_extensions, skip_source_dir, source_quality,
};

#[derive(Parser)]
#[command(
    name = "collect-data",
    about = "Tokenize source code into packed training shards"
)]
struct Args {
    #[arg(short, long)]
    input: Vec<String>,

    #[arg(short, long, default_value = "32768")]
    vocab_size: usize,

    #[arg(short, long)]
    tokenizer: String,

    #[arg(short, long, default_value = "./data/shards")]
    output: PathBuf,

    /// Comma-separated source extensions to ingest (e.g. `rs,md,toml,py`).
    #[arg(long, default_value = "rs")]
    extension: String,

    #[arg(long)]
    recursive: bool,

    #[arg(long, default_value_t = 8192)]
    seq_len: usize,

    #[arg(long, default_value_t = 100_000)]
    tokens_per_shard: usize,

    /// Disable the corpus-quality filter (keep generated/minified/data-table files).
    /// Off by default: the filter drops low-value bulk that dilutes the code model.
    #[arg(long)]
    no_quality_filter: bool,
}

/// Kept/skipped tallies for the quality filter, so nothing is dropped silently.
#[derive(Default)]
struct CollectStats {
    kept: usize,
    skipped: usize,
    reasons: BTreeMap<&'static str, usize>,
}

impl CollectStats {
    fn skip(&mut self, reason: &'static str) {
        self.skipped += 1;
        *self.reasons.entry(reason).or_insert(0) += 1;
    }
}

fn main() {
    let args = Args::parse();
    fs::create_dir_all(&args.output).expect("Cannot create output dir");

    let tok = BpeTokenizer::load_json(&args.tokenizer).expect("Failed to load tokenizer");
    eprintln!("Tokenizer loaded: vocab_size={}", tok.vocab_size());

    let exts = parse_extensions(&args.extension);
    eprintln!("Ingesting extensions: {exts:?}");

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

    let mut all_tokens: Vec<u32> = Vec::new();
    let mut stats = CollectStats::default();
    for path in &args.input
    {
        let p = Path::new(path);
        if p.is_file()
        {
            if let Ok(content) = fs::read_to_string(p)
            {
                ingest_file(p, &content, filter, &tok, &mut all_tokens, &mut stats);
            }
        }
        else if p.is_dir() && args.recursive
        {
            collect_dir(p, &exts, filter, &tok, &mut all_tokens, &mut stats);
        }
    }

    eprintln!(
        "files kept {} | skipped {} | reasons {:?}",
        stats.kept, stats.skipped, stats.reasons
    );
    eprintln!("Total tokens: {}", all_tokens.len());
    eprintln!("Packing into shards of {} tokens...", args.tokens_per_shard);

    let shard_size = args.tokens_per_shard;
    let num_shards = all_tokens.len().div_ceil(shard_size);

    for shard_idx in 0..num_shards
    {
        let start = shard_idx * shard_size;
        let end = std::cmp::min(start + shard_size, all_tokens.len());
        let shard_data = &all_tokens[start..end];

        let shard_path = args.output.join(format!("shard_{:04}.bin", shard_idx));
        let mut f = fs::File::create(&shard_path).expect("Cannot create shard file");

        for &token in shard_data
        {
            f.write_all(&token.to_le_bytes()).expect("Write error");
        }

        eprintln!(
            "Shard {:04}: {} tokens -> {:?}",
            shard_idx,
            shard_data.len(),
            shard_path
        );
    }

    eprintln!("Done: {} shards written to {:?}", num_shards, args.output);
}

/// Tokenize one file into `tokens`, applying the quality filter and updating stats.
fn ingest_file(
    path: &Path,
    content: &str,
    filter: bool,
    tok: &BpeTokenizer,
    tokens: &mut Vec<u32>,
    stats: &mut CollectStats,
) {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if filter
    {
        if let Err(reason) = source_quality(name, content)
        {
            stats.skip(reason);
            return;
        }
    }
    let ids = tok.encode_with_special(content, true, true);
    tokens.extend(ids.iter().map(|&i| i as u32));
    stats.kept += 1;
}

fn collect_dir(
    dir: &Path,
    exts: &[String],
    filter: bool,
    tok: &BpeTokenizer,
    tokens: &mut Vec<u32>,
    stats: &mut CollectStats,
) {
    if let Ok(entries) = fs::read_dir(dir)
    {
        // Sort by path so the corpus order is deterministic across machines/runs —
        // `read_dir` yields OS-arbitrary order, which would make the shards (and the
        // tokenizer) irreproducible.
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
                collect_dir(&path, exts, filter, tok, tokens, stats);
            }
            else if matches_extension(&path, exts)
            {
                if let Ok(content) = fs::read_to_string(&path)
                {
                    ingest_file(&path, &content, filter, tok, tokens, stats);
                }
            }
        }
    }
}
