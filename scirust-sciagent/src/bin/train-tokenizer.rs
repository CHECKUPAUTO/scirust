use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use scirust_sciagent::bpe::BpeTrainer;
use scirust_sciagent::train::dataset::{
    content_hash, matches_extension, parse_extensions, skip_source_dir, source_quality,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum MergeSemanticsArg {
    /// Historical repeated left-to-right bulk merges. Keeps compatibility with
    /// existing SciAgent shards/checkpoints.
    LegacyParallelV1,
    /// Canonical global rank-priority merges for ElasticTokenizer execution.
    CanonicalRankV1,
}

impl MergeSemanticsArg {
    const fn as_artifact_tag(self) -> &'static str {
        match self
        {
            Self::LegacyParallelV1 => "legacy-parallel-v1",
            Self::CanonicalRankV1 => "canonical-rank-v1",
        }
    }
}

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

    /// Merge execution semantics to record in the new tokenizer artifact.
    ///
    /// This does not retroactively migrate existing shards. `legacy-parallel-v1`
    /// remains the default so existing SciAgent workflows retain their token ids.
    #[arg(long, value_enum, default_value_t = MergeSemanticsArg::LegacyParallelV1)]
    merge_semantics: MergeSemanticsArg,

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
    eprintln!(
        "merge semantics: {}",
        args.merge_semantics.as_artifact_tag()
    );
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
    let mut seen: HashSet<u64> = HashSet::new();

    for path in &args.input
    {
        let p = Path::new(path);
        if p.is_file()
        {
            if let Ok(content) = fs::read_to_string(p)
            {
                let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                ingest_text(
                    name,
                    content,
                    filter,
                    &mut seen,
                    &mut skipped,
                    &mut all_texts,
                );
            }
        }
        else if p.is_dir() && args.recursive
        {
            collect_dir(p, &exts, filter, &mut seen, &mut all_texts, &mut skipped);
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
    write_merge_semantics_tag(&args.output, args.merge_semantics)
        .expect("Failed to write tokenizer merge semantics");
    eprintln!(
        "Tokenizer saved to {} (vocab size: {}, merge semantics: {})",
        args.output,
        tokenizer.vocab_size(),
        args.merge_semantics.as_artifact_tag()
    );
}

fn write_merge_semantics_tag(
    path: &str,
    semantics: MergeSemanticsArg,
) -> Result<(), Box<dyn std::error::Error>> {
    let input = fs::read_to_string(path)?;
    let mut value: serde_json::Value = serde_json::from_str(&input)?;
    let object = value
        .as_object_mut()
        .ok_or("tokenizer JSON root must be an object")?;
    object.insert(
        "merge_semantics".to_string(),
        serde_json::Value::String(semantics.as_artifact_tag().to_string()),
    );
    fs::write(path, serde_json::to_string_pretty(&value)?)?;
    Ok(())
}

/// Quality-filter, deduplicate, and (if kept) push one file's text — the shared
/// gate for train-tokenizer, mirroring collect-data so both see the same corpus.
fn ingest_text(
    name: &str,
    content: String,
    filter: bool,
    seen: &mut HashSet<u64>,
    skipped: &mut BTreeMap<&'static str, usize>,
    texts: &mut Vec<String>,
) {
    if filter
    {
        if let Err(reason) = source_quality(name, &content)
        {
            *skipped.entry(reason).or_insert(0) += 1;
            return;
        }
    }
    if !seen.insert(content_hash(&content))
    {
        *skipped.entry("duplicate").or_insert(0) += 1;
        return;
    }
    texts.push(content);
}

fn collect_dir(
    dir: &Path,
    exts: &[String],
    filter: bool,
    seen: &mut HashSet<u64>,
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
                collect_dir(&path, exts, filter, seen, texts, skipped);
            }
            else if matches_extension(&path, exts)
            {
                if let Ok(content) = fs::read_to_string(&path)
                {
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    ingest_text(name, content, filter, seen, skipped, texts);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_semantics_tags_are_stable() {
        assert_eq!(
            MergeSemanticsArg::LegacyParallelV1.as_artifact_tag(),
            "legacy-parallel-v1"
        );
        assert_eq!(
            MergeSemanticsArg::CanonicalRankV1.as_artifact_tag(),
            "canonical-rank-v1"
        );
    }

    #[test]
    fn merge_semantics_tag_is_written_without_losing_existing_fields() {
        let path = std::env::temp_dir().join("scirust_tokenizer_semantics_tag.json");
        fs::write(&path, r#"{"version":"byte_level_v2","vocab":{},"merges":[]}"#).unwrap();
        write_merge_semantics_tag(path.to_str().unwrap(), MergeSemanticsArg::CanonicalRankV1)
            .unwrap();
        let value: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(value["version"], "byte_level_v2");
        assert_eq!(value["merge_semantics"], "canonical-rank-v1");
        assert!(value.get("vocab").is_some());
        assert!(value.get("merges").is_some());
        let _ = fs::remove_file(path);
    }
}
