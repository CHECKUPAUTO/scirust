use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use scirust_sciagent::BpeTokenizer;

#[derive(Parser)]
#[command(
    name = "legacy-bpe-cache-probe",
    about = "Verify and measure a cached implementation of historical SciAgent BPE semantics"
)]
struct Args {
    /// Historical byte_level_v2 tokenizer artifact.
    #[arg(short, long)]
    tokenizer: PathBuf,

    /// UTF-8 corpus files measured in deterministic path order.
    #[arg(short, long)]
    input: Vec<PathBuf>,

    #[arg(long, default_value_t = 2)]
    warmup_rounds: usize,

    #[arg(long, default_value_t = 9)]
    measured_rounds: usize,
}

#[derive(Clone, Debug)]
struct CachedLegacyBpe {
    byte_ids: [usize; 256],
    merges: HashMap<(usize, usize), usize>,
}

impl CachedLegacyBpe {
    fn load(path: &Path) -> Result<Self, String> {
        let input = fs::read_to_string(path).map_err(|error| error.to_string())?;
        let value: serde_json::Value =
            serde_json::from_str(&input).map_err(|error| error.to_string())?;
        if value.get("version").and_then(serde_json::Value::as_str) != Some("byte_level_v2")
        {
            return Err("legacy cache probe requires version=byte_level_v2".to_string());
        }
        if let Some(semantics) = value
            .get("merge_semantics")
            .and_then(serde_json::Value::as_str)
        {
            if semantics != "legacy-parallel-v1"
            {
                return Err(format!(
                    "legacy cache probe rejects merge_semantics={semantics}"
                ));
            }
        }

        let vocab: BTreeMap<String, usize> = serde_json::from_value(
            value
                .get("vocab")
                .cloned()
                .ok_or_else(|| "tokenizer missing vocab".to_string())?,
        )
        .map_err(|error| error.to_string())?;
        let mut byte_ids = [0usize; 256];
        for byte in 0u8..=255
        {
            let token = byte_to_unit(byte).to_string();
            byte_ids[usize::from(byte)] = vocab
                .get(&token)
                .copied()
                .ok_or_else(|| format!("tokenizer missing byte token {byte}"))?;
        }

        let raw_merges = value
            .get("merges")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "tokenizer missing merges array".to_string())?;
        let mut merges = HashMap::with_capacity(raw_merges.len());
        for (index, raw) in raw_merges.iter().enumerate()
        {
            let rule = raw
                .as_str()
                .ok_or_else(|| format!("merge rule {index} is not a string"))?;
            let mut parts = rule.split_whitespace();
            let left = parse_usize(parts.next(), index, "left")?;
            let right = parse_usize(parts.next(), index, "right")?;
            let output = parse_usize(parts.next(), index, "output")?;
            if parts.next().is_some()
            {
                return Err(format!("merge rule {index} has extra fields"));
            }
            if merges.insert((left, right), output).is_some()
            {
                return Err(format!("duplicate legacy merge pair ({left}, {right})"));
            }
        }

        Ok(Self { byte_ids, merges })
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = text
            .bytes()
            .map(|byte| self.byte_ids[usize::from(byte)])
            .collect::<Vec<_>>();

        loop
        {
            let mut changed = false;
            let mut next = Vec::with_capacity(ids.len());
            let mut index = 0usize;
            while index < ids.len()
            {
                if index + 1 < ids.len()
                {
                    if let Some(&output) = self.merges.get(&(ids[index], ids[index + 1]))
                    {
                        next.push(output);
                        index += 2;
                        changed = true;
                        continue;
                    }
                }
                next.push(ids[index]);
                index += 1;
            }
            ids = next;
            if !changed
            {
                return ids;
            }
        }
    }
}

fn main() {
    let args = Args::parse();
    if args.measured_rounds == 0
    {
        panic!("--measured-rounds must be greater than zero");
    }

    let baseline = BpeTokenizer::load_json(
        args.tokenizer
            .to_str()
            .expect("tokenizer path must be valid UTF-8"),
    )
    .expect("failed to load historical BpeTokenizer");
    let cached = CachedLegacyBpe::load(&args.tokenizer).expect("failed to load cached legacy path");

    let mut inputs = args.input;
    inputs.sort();
    inputs.dedup();
    let texts = inputs
        .iter()
        .map(|path| fs::read_to_string(path).unwrap_or_else(|error| {
            panic!("failed to read input {}: {error}", path.display())
        }))
        .collect::<Vec<_>>();
    if texts.is_empty()
    {
        panic!("at least one --input file is required");
    }

    verify_parity(&baseline, &cached, &texts).expect("cached legacy BPE changed token IDs");

    for _ in 0..args.warmup_rounds
    {
        black_box(run_baseline(&baseline, &texts));
        black_box(run_cached(&cached, &texts));
    }

    let mut baseline_samples = Vec::with_capacity(args.measured_rounds);
    let mut cached_samples = Vec::with_capacity(args.measured_rounds);
    for round in 0..args.measured_rounds
    {
        if round.is_multiple_of(2)
        {
            baseline_samples.push(time_nanos(|| run_baseline(&baseline, &texts)));
            cached_samples.push(time_nanos(|| run_cached(&cached, &texts)));
        }
        else
        {
            cached_samples.push(time_nanos(|| run_cached(&cached, &texts)));
            baseline_samples.push(time_nanos(|| run_baseline(&baseline, &texts)));
        }
    }

    let baseline_median = median(&baseline_samples).expect("baseline samples are non-empty");
    let cached_median = median(&cached_samples).expect("cached samples are non-empty");
    let bytes = texts.iter().map(String::len).sum::<usize>();

    println!("files={}", texts.len());
    println!("utf8_bytes={bytes}");
    println!("baseline_median_ns={baseline_median}");
    println!("cached_median_ns={cached_median}");
    if cached_median != 0
    {
        println!(
            "cached_speedup_x={:.6}",
            baseline_median as f64 / cached_median as f64
        );
    }
}

fn verify_parity(
    baseline: &BpeTokenizer,
    cached: &CachedLegacyBpe,
    texts: &[String],
) -> Result<(), String> {
    for (index, text) in texts.iter().enumerate()
    {
        let expected = baseline.encode(text);
        let actual = cached.encode(text);
        if actual != expected
        {
            return Err(format!("token-ID mismatch for input index {index}"));
        }
    }
    Ok(())
}

fn run_baseline(tokenizer: &BpeTokenizer, texts: &[String]) -> usize {
    texts
        .iter()
        .map(|text| black_box(tokenizer.encode(black_box(text))).len())
        .sum()
}

fn run_cached(tokenizer: &CachedLegacyBpe, texts: &[String]) -> usize {
    texts
        .iter()
        .map(|text| black_box(tokenizer.encode(black_box(text))).len())
        .sum()
}

fn time_nanos(f: impl FnOnce() -> usize) -> u64 {
    let start = Instant::now();
    black_box(f());
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn median(samples: &[u64]) -> Option<u64> {
    if samples.is_empty()
    {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let middle = sorted.len() / 2;
    if !sorted.len().is_multiple_of(2)
    {
        Some(sorted[middle])
    }
    else
    {
        let low = sorted[middle - 1];
        let high = sorted[middle];
        Some(low + (high - low) / 2)
    }
}

fn parse_usize(value: Option<&str>, index: usize, field: &str) -> Result<usize, String> {
    value
        .ok_or_else(|| format!("merge rule {index} missing {field}"))?
        .parse::<usize>()
        .map_err(|_| format!("merge rule {index} has invalid {field}"))
}

fn byte_to_unit(byte: u8) -> char {
    let codepoint = if byte < 128
    {
        u32::from(byte)
    }
    else
    {
        256 + (u32::from(byte) - 128)
    };
    char::from_u32(codepoint).expect("byte-unit codepoint is always valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer_json() -> String {
        let mut vocab = serde_json::Map::new();
        vocab.insert("<pad>".to_string(), serde_json::json!(0));
        vocab.insert("<bos>".to_string(), serde_json::json!(1));
        vocab.insert("<eos>".to_string(), serde_json::json!(2));
        vocab.insert("<unk>".to_string(), serde_json::json!(3));
        for byte in 0u8..=255
        {
            vocab.insert(
                byte_to_unit(byte).to_string(),
                serde_json::json!(usize::from(byte) + 4),
            );
        }
        vocab.insert("ab".to_string(), serde_json::json!(260));
        vocab.insert("bc".to_string(), serde_json::json!(261));
        serde_json::json!({
            "version": "byte_level_v2",
            "merge_semantics": "legacy-parallel-v1",
            "vocab": vocab,
            "merges": ["101 102 260", "102 103 261"],
        })
        .to_string()
    }

    #[test]
    fn cached_path_matches_historical_parallel_semantics() {
        let path = std::env::temp_dir().join("scirust_legacy_bpe_cache_probe.json");
        fs::write(&path, tokenizer_json()).unwrap();
        let baseline = BpeTokenizer::load_json(path.to_str().unwrap()).unwrap();
        let cached = CachedLegacyBpe::load(&path).unwrap();
        for text in ["abc", "ababc", "cab", "", "αβ Rust 🚀"]
        {
            assert_eq!(cached.encode(text), baseline.encode(text), "input={text:?}");
        }
        let _ = fs::remove_file(path);
    }

    #[test]
    fn canonical_semantics_are_rejected() {
        let path = std::env::temp_dir().join("scirust_legacy_bpe_cache_probe_canonical.json");
        let input = tokenizer_json().replace("legacy-parallel-v1", "canonical-rank-v1");
        fs::write(&path, input).unwrap();
        assert!(CachedLegacyBpe::load(&path).is_err());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn median_is_overflow_safe() {
        assert_eq!(median(&[u64::MAX - 1, u64::MAX]), Some(u64::MAX - 1));
    }
}
