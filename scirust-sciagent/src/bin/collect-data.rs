use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use std::collections::{BTreeMap, HashSet};

use clap::Parser;
use scirust_sciagent::bpe::BpeTokenizer;
use scirust_sciagent::train::dataset::{
    content_hash, matches_extension, parse_extensions, skip_source_dir, source_quality,
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
    /// Content hashes already ingested — for deduplication.
    seen: HashSet<u64>,
}

impl CollectStats {
    fn skip(&mut self, reason: &'static str) {
        self.skipped += 1;
        *self.reasons.entry(reason).or_insert(0) += 1;
    }
}

/// Streams tokens to fixed-size little-endian `u32` `.bin` shards, flushing a
/// shard as soon as the buffer fills — so peak memory is O(tokens_per_shard),
/// **not** O(corpus). This is what lets the collector tokenize a billion-token
/// corpus on a memory-shared Jetson without holding the whole token stream in
/// RAM (4 GB at 1B tokens), and it lands partial progress on disk instead of
/// losing everything if the pass is interrupted late. Writes go through a
/// `BufWriter`, replacing the old per-token `write_all` syscall storm.
///
/// Filenames are zero-padded to 6 digits (`shard_000000.bin`): the loader sorts
/// shards *lexically* by name, so a uniform width keeps the concatenation order
/// correct at any shard count (`shard_{:04}` silently misordered past 9999).
struct ShardWriter {
    out: PathBuf,
    shard_size: usize,
    buf: Vec<u32>,
    shard_idx: usize,
    total_tokens: usize,
}

impl ShardWriter {
    fn new(out: PathBuf, shard_size: usize) -> Self {
        let shard_size = shard_size.max(1);
        Self {
            out,
            shard_size,
            buf: Vec::with_capacity(shard_size),
            shard_idx: 0,
            total_tokens: 0,
        }
    }

    /// Append token ids, flushing whenever a shard's worth has accumulated.
    fn extend(&mut self, ids: impl IntoIterator<Item = u32>) {
        for id in ids
        {
            self.buf.push(id);
            self.total_tokens += 1;
            if self.buf.len() >= self.shard_size
            {
                self.flush();
            }
        }
    }

    /// Write the buffered tokens as the next shard and clear the buffer.
    fn flush(&mut self) {
        if self.buf.is_empty()
        {
            return;
        }
        let shard_path = self.out.join(format!("shard_{:06}.bin", self.shard_idx));
        let f = fs::File::create(&shard_path).expect("Cannot create shard file");
        let mut w = BufWriter::new(f);
        for &token in &self.buf
        {
            w.write_all(&token.to_le_bytes()).expect("Write error");
        }
        w.flush().expect("Flush error");
        eprintln!(
            "Shard {:06}: {} tokens -> {:?}",
            self.shard_idx,
            self.buf.len(),
            shard_path
        );
        self.shard_idx += 1;
        self.buf.clear();
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

    eprintln!("Packing into shards of {} tokens...", args.tokens_per_shard);
    let mut writer = ShardWriter::new(args.output.clone(), args.tokens_per_shard);
    let mut stats = CollectStats::default();
    for path in &args.input
    {
        let p = Path::new(path);
        if p.is_file()
        {
            if let Ok(content) = fs::read_to_string(p)
            {
                ingest_file(p, &content, filter, &tok, &mut writer, &mut stats);
            }
        }
        else if p.is_dir() && args.recursive
        {
            collect_dir(p, &exts, filter, &tok, &mut writer, &mut stats);
        }
    }
    writer.flush(); // write the final partial shard

    eprintln!(
        "files kept {} | skipped {} | reasons {:?}",
        stats.kept, stats.skipped, stats.reasons
    );
    eprintln!("Total tokens: {}", writer.total_tokens);
    eprintln!(
        "Done: {} shards written to {:?}",
        writer.shard_idx, args.output
    );
}

/// Tokenize one file into `tokens`, applying the quality filter and updating stats.
fn ingest_file(
    path: &Path,
    content: &str,
    filter: bool,
    tok: &BpeTokenizer,
    writer: &mut ShardWriter,
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
    // Deduplicate by content: crates.io is full of duplicated files, and this also
    // cancels the fetch-crates `all/`-symlink double-count. First-seen wins (the walk
    // is sorted, so it's deterministic).
    if !stats.seen.insert(content_hash(content))
    {
        stats.skip("duplicate");
        return;
    }
    let ids = tok.encode_with_special(content, true, true);
    writer.extend(ids.iter().map(|&i| i as u32));
    stats.kept += 1;
}

fn collect_dir(
    dir: &Path,
    exts: &[String],
    filter: bool,
    tok: &BpeTokenizer,
    writer: &mut ShardWriter,
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
                collect_dir(&path, exts, filter, tok, writer, stats);
            }
            else if matches_extension(&path, exts)
            {
                if let Ok(content) = fs::read_to_string(&path)
                {
                    ingest_file(&path, &content, filter, tok, writer, stats);
                }
            }
        }
    }
}
