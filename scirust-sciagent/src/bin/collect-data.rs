use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use clap::Parser;
use scirust_sciagent::corpus_paths;
use scirust_sciagent::train::dataset::{
    content_hash, matches_extension, parse_extensions, skip_source_dir, source_quality,
};
use scirust_sciagent::{ElasticHardwareIdentity, StoredElasticProfile, VersionedBpeTokenizer};

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

    /// Optional persisted ElasticTokenizer execution profile. The profile is
    /// accepted only for canonical BPE semantics and only when both its ordered
    /// merge fingerprint and hardware identity match this run.
    #[arg(long, value_name = "FILE")]
    elastic_profile: Option<PathBuf>,

    /// Stable deployment-local hardware discriminator used to bind a persisted
    /// ElasticTokenizer profile. Architecture and OS are added automatically.
    #[arg(long, default_value = "generic")]
    elastic_device: String,

    /// External directory for generated shards. Defaults to the platform data
    /// directory; paths inside the SciRust checkout are rejected.
    #[arg(short, long, value_name = "DIR")]
    output: Option<PathBuf>,

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

#[derive(Default)]
struct CollectStats {
    kept: usize,
    skipped: usize,
    reasons: BTreeMap<&'static str, usize>,
    seen: HashSet<u64>,
}

impl CollectStats {
    fn skip(&mut self, reason: &'static str) {
        self.skipped += 1;
        *self.reasons.entry(reason).or_insert(0) += 1;
    }
}

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

    fn flush(&mut self) {
        if self.buf.is_empty()
        {
            return;
        }
        let shard_path = self.out.join(format!("shard_{:06}.bin", self.shard_idx));
        let file = fs::File::create(&shard_path).expect("Cannot create shard file");
        let mut writer = BufWriter::new(file);
        for &token in &self.buf
        {
            writer
                .write_all(&token.to_le_bytes())
                .expect("Write error");
        }
        writer.flush().expect("Flush error");
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
    let output = corpus_paths::resolve_external_shards_dir(args.output).unwrap_or_else(|error| {
        eprintln!("Cannot use shard output directory: {error}");
        std::process::exit(2);
    });
    fs::create_dir_all(&output).expect("Cannot create output dir");

    let mut tok =
        VersionedBpeTokenizer::load_json(&args.tokenizer).expect("Failed to load tokenizer");
    if let Some(profile_path) = &args.elastic_profile
    {
        let stored = StoredElasticProfile::load(profile_path)
            .expect("Failed to load ElasticTokenizer execution profile");
        let hardware = ElasticHardwareIdentity::new(
            std::env::consts::ARCH,
            std::env::consts::OS,
            args.elastic_device.clone(),
        );
        tok.apply_stored_profile(&stored, &hardware)
            .expect("ElasticTokenizer execution profile rejected");
        eprintln!(
            "Elastic profile applied: hardware={} profile={:?}",
            hardware.fingerprint(),
            tok.elastic_profile()
        );
    }
    eprintln!(
        "Tokenizer loaded: vocab_size={} merge_semantics={}",
        tok.vocab_size(),
        tok.merge_semantics().as_str()
    );

    if tok.vocab_size() > args.vocab_size
    {
        panic!(
            "tokenizer vocab_size={} exceeds configured shard vocab_size={}",
            tok.vocab_size(),
            args.vocab_size
        );
    }

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
    let mut writer = ShardWriter::new(output.clone(), args.tokens_per_shard);
    let mut stats = CollectStats::default();
    for path in &args.input
    {
        let path = Path::new(path);
        if path.is_file()
        {
            if let Ok(content) = fs::read_to_string(path)
            {
                ingest_file(path, &content, filter, &tok, &mut writer, &mut stats);
            }
        }
        else if path.is_dir() && args.recursive
        {
            collect_dir(path, &exts, filter, &tok, &mut writer, &mut stats);
        }
    }
    writer.flush();

    eprintln!(
        "files kept {} | skipped {} | reasons {:?}",
        stats.kept, stats.skipped, stats.reasons
    );
    eprintln!("Total tokens: {}", writer.total_tokens);
    eprintln!("Done: {} shards written to {:?}", writer.shard_idx, output);
}

fn ingest_file(
    path: &Path,
    content: &str,
    filter: bool,
    tok: &VersionedBpeTokenizer,
    writer: &mut ShardWriter,
    stats: &mut CollectStats,
) {
    let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("");
    if filter
    {
        if let Err(reason) = source_quality(name, content)
        {
            stats.skip(reason);
            return;
        }
    }
    if !stats.seen.insert(content_hash(content))
    {
        stats.skip("duplicate");
        return;
    }

    let ids = tok.encode_with_special(content, true, true);
    writer.extend(ids.into_iter().map(|id| {
        u32::try_from(id).expect("token id exceeds the u32 shard format")
    }));
    stats.kept += 1;
}

fn collect_dir(
    dir: &Path,
    exts: &[String],
    filter: bool,
    tok: &VersionedBpeTokenizer,
    writer: &mut ShardWriter,
    stats: &mut CollectStats,
) {
    if let Ok(entries) = fs::read_dir(dir)
    {
        let mut paths: Vec<PathBuf> = entries.flatten().map(|entry| entry.path()).collect();
        paths.sort();
        for path in paths
        {
            if path.is_dir()
            {
                if let Some(name) = path.file_name().and_then(|name| name.to_str())
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
