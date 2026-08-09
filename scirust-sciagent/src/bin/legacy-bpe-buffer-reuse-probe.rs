use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;

#[derive(Parser)]
#[command(
    name = "legacy-bpe-buffer-reuse-probe",
    about = "Measure per-pass allocation versus double-buffer reuse in cached legacy BPE"
)]
struct Args {
    #[arg(short, long)]
    tokenizer: PathBuf,

    #[arg(short, long)]
    input: Vec<PathBuf>,

    #[arg(long, default_value_t = 3)]
    warmup_rounds: usize,

    #[arg(long, default_value_t = 11)]
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
            return Err("buffer-reuse probe requires version=byte_level_v2".to_string());
        }
        if let Some(semantics) = value
            .get("merge_semantics")
            .and_then(serde_json::Value::as_str)
        {
            if semantics != "legacy-parallel-v1"
            {
                return Err(format!(
                    "buffer-reuse probe rejects merge_semantics={semantics}"
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
            let key = byte_to_unit(byte).to_string();
            byte_ids[usize::from(byte)] = vocab
                .get(&key)
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
                return Err(format!("duplicate merge pair ({left}, {right})"));
            }
        }
        Ok(Self { byte_ids, merges })
    }

    fn base_ids(&self, text: &str) -> Vec<usize> {
        text.bytes()
            .map(|byte| self.byte_ids[usize::from(byte)])
            .collect()
    }

    fn encode_allocating(&self, text: &str) -> Vec<usize> {
        let mut ids = self.base_ids(text);
        loop
        {
            let mut next = Vec::with_capacity(ids.len());
            let changed = merge_pass(&ids, &mut next, &self.merges);
            if !changed
            {
                return ids;
            }
            ids = next;
        }
    }

    fn encode_reusing(&self, text: &str) -> Vec<usize> {
        let mut ids = self.base_ids(text);
        let mut next = Vec::with_capacity(ids.len());
        loop
        {
            next.clear();
            let changed = merge_pass(&ids, &mut next, &self.merges);
            if !changed
            {
                return ids;
            }
            std::mem::swap(&mut ids, &mut next);
        }
    }
}

fn merge_pass(
    ids: &[usize],
    output: &mut Vec<usize>,
    merges: &HashMap<(usize, usize), usize>,
) -> bool {
    let mut changed = false;
    let mut index = 0usize;
    while index < ids.len()
    {
        if index + 1 < ids.len() && ids[index] != 0 && ids[index + 1] != 0
        {
            if let Some(&merged) = merges.get(&(ids[index], ids[index + 1]))
            {
                output.push(merged);
                index += 2;
                changed = true;
                continue;
            }
        }
        output.push(ids[index]);
        index += 1;
    }
    changed
}

fn main() {
    let args = Args::parse();
    if args.measured_rounds == 0
    {
        panic!("--measured-rounds must be greater than zero");
    }
    let tokenizer = CachedLegacyBpe::load(&args.tokenizer).expect("failed to load probe tokenizer");
    let mut paths = args.input;
    paths.sort();
    paths.dedup();
    let texts = paths
        .iter()
        .map(|path| {
            fs::read_to_string(path)
                .unwrap_or_else(|error| panic!("failed to read input {}: {error}", path.display()))
        })
        .collect::<Vec<_>>();
    if texts.is_empty()
    {
        panic!("at least one --input file is required");
    }

    for (index, text) in texts.iter().enumerate()
    {
        assert_eq!(
            tokenizer.encode_allocating(text),
            tokenizer.encode_reusing(text),
            "buffer reuse changed token IDs for input {index}"
        );
    }

    for _ in 0..args.warmup_rounds
    {
        black_box(run_allocating(&tokenizer, &texts));
        black_box(run_reusing(&tokenizer, &texts));
    }

    let mut allocating = Vec::with_capacity(args.measured_rounds);
    let mut reusing = Vec::with_capacity(args.measured_rounds);
    for round in 0..args.measured_rounds
    {
        if round.is_multiple_of(2)
        {
            allocating.push(time_nanos(|| run_allocating(&tokenizer, &texts)));
            reusing.push(time_nanos(|| run_reusing(&tokenizer, &texts)));
        }
        else
        {
            reusing.push(time_nanos(|| run_reusing(&tokenizer, &texts)));
            allocating.push(time_nanos(|| run_allocating(&tokenizer, &texts)));
        }
    }

    let allocating_median = median(&allocating).expect("allocating samples are non-empty");
    let reusing_median = median(&reusing).expect("reusing samples are non-empty");
    println!("files={}", texts.len());
    println!("utf8_bytes={}", texts.iter().map(String::len).sum::<usize>());
    println!("allocating_median_ns={allocating_median}");
    println!("reusing_median_ns={reusing_median}");
    if reusing_median != 0
    {
        println!(
            "buffer_reuse_speedup_x={:.6}",
            allocating_median as f64 / reusing_median as f64
        );
    }
}

fn run_allocating(tokenizer: &CachedLegacyBpe, texts: &[String]) -> usize {
    texts
        .iter()
        .map(|text| black_box(tokenizer.encode_allocating(black_box(text))).len())
        .sum()
}

fn run_reusing(tokenizer: &CachedLegacyBpe, texts: &[String]) -> usize {
    texts
        .iter()
        .map(|text| black_box(tokenizer.encode_reusing(black_box(text))).len())
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

    #[test]
    fn merge_pass_preserves_parallel_semantics() {
        let merges = HashMap::from([((1, 2), 10), ((10, 3), 11)]);
        let mut output = Vec::new();
        assert!(merge_pass(&[1, 2, 3], &mut output, &merges));
        assert_eq!(output, vec![10, 3]);
        let mut second = Vec::new();
        assert!(merge_pass(&output, &mut second, &merges));
        assert_eq!(second, vec![11]);
    }

    #[test]
    fn unchanged_pass_is_reported_without_mutating_input() {
        let merges = HashMap::new();
        let input = vec![1, 2, 3];
        let mut output = Vec::new();
        assert!(!merge_pass(&input, &mut output, &merges));
        assert_eq!(input, vec![1, 2, 3]);
        assert_eq!(output, input);
    }

    #[test]
    fn median_is_overflow_safe() {
        assert_eq!(median(&[u64::MAX - 1, u64::MAX]), Some(u64::MAX - 1));
    }
}
