use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use scirust_sciagent::PairKey;

#[derive(Parser)]
#[command(
    name = "legacy-bpe-packed-map-probe",
    about = "Compare wide tuple and PairKey(u64) legacy merge maps with RandomState"
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
struct ProbeTokenizer {
    byte_ids: [usize; 256],
    wide_merges: HashMap<(usize, usize), usize>,
    packed_merges: HashMap<PairKey, u32>,
}

impl ProbeTokenizer {
    fn load(path: &Path) -> Result<Self, String> {
        let input = fs::read_to_string(path).map_err(|error| error.to_string())?;
        let value: serde_json::Value =
            serde_json::from_str(&input).map_err(|error| error.to_string())?;
        if value.get("version").and_then(serde_json::Value::as_str) != Some("byte_level_v2")
        {
            return Err("packed-map probe requires version=byte_level_v2".to_string());
        }
        if let Some(semantics) = value
            .get("merge_semantics")
            .and_then(serde_json::Value::as_str)
        {
            if semantics != "legacy-parallel-v1"
            {
                return Err(format!("packed-map probe rejects merge_semantics={semantics}"));
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
            let id = vocab
                .get(&key)
                .copied()
                .ok_or_else(|| format!("tokenizer missing byte token {byte}"))?;
            u32::try_from(id).map_err(|_| format!("byte token {byte} id {id} exceeds u32"))?;
            byte_ids[usize::from(byte)] = id;
        }

        let raw_merges = value
            .get("merges")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "tokenizer missing merges array".to_string())?;
        let mut wide_merges = HashMap::with_capacity(raw_merges.len());
        let mut packed_merges = HashMap::with_capacity(raw_merges.len());
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
            let key = PairKey::try_from_usize(left, right)
                .map_err(|error| format!("merge rule {index}: {error}"))?;
            let packed_output = u32::try_from(output)
                .map_err(|_| format!("merge rule {index} output {output} exceeds u32"))?;
            if wide_merges.insert((left, right), output).is_some()
                || packed_merges.insert(key, packed_output).is_some()
            {
                return Err(format!("duplicate merge pair ({left}, {right})"));
            }
        }
        Ok(Self {
            byte_ids,
            wide_merges,
            packed_merges,
        })
    }

    fn base_ids(&self, text: &str) -> Vec<usize> {
        text.bytes()
            .map(|byte| self.byte_ids[usize::from(byte)])
            .collect()
    }

    fn encode_wide(&self, text: &str) -> Vec<usize> {
        let ids = self.base_ids(text);
        apply_wide(ids, &self.wide_merges)
    }

    fn encode_packed(&self, text: &str) -> Vec<usize> {
        let ids = self.base_ids(text);
        apply_packed(ids, &self.packed_merges)
    }
}

fn apply_wide(
    mut ids: Vec<usize>,
    merges: &HashMap<(usize, usize), usize>,
) -> Vec<usize> {
    loop
    {
        let mut changed = false;
        let mut next = Vec::with_capacity(ids.len());
        let mut index = 0usize;
        while index < ids.len()
        {
            if index + 1 < ids.len() && ids[index] != 0 && ids[index + 1] != 0
            {
                if let Some(&output) = merges.get(&(ids[index], ids[index + 1]))
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
        if !changed
        {
            return ids;
        }
        ids = next;
    }
}

fn apply_packed(mut ids: Vec<usize>, merges: &HashMap<PairKey, u32>) -> Vec<usize> {
    loop
    {
        let mut changed = false;
        let mut next = Vec::with_capacity(ids.len());
        let mut index = 0usize;
        while index < ids.len()
        {
            if index + 1 < ids.len() && ids[index] != 0 && ids[index + 1] != 0
            {
                let left = u32::try_from(ids[index]).expect("tokenizer preflight checked u32 ids");
                let right =
                    u32::try_from(ids[index + 1]).expect("tokenizer preflight checked u32 ids");
                if let Some(&output) = merges.get(&PairKey::new(left, right))
                {
                    next.push(usize::try_from(output).expect("u32 token id fits usize"));
                    index += 2;
                    changed = true;
                    continue;
                }
            }
            next.push(ids[index]);
            index += 1;
        }
        if !changed
        {
            return ids;
        }
        ids = next;
    }
}

fn main() {
    let args = Args::parse();
    if args.measured_rounds == 0
    {
        panic!("--measured-rounds must be greater than zero");
    }
    let tokenizer = ProbeTokenizer::load(&args.tokenizer).expect("failed to load probe tokenizer");
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
            tokenizer.encode_wide(text),
            tokenizer.encode_packed(text),
            "packed merge map changed token IDs for input {index}"
        );
    }

    for _ in 0..args.warmup_rounds
    {
        black_box(run_wide(&tokenizer, &texts));
        black_box(run_packed(&tokenizer, &texts));
    }

    let mut wide_samples = Vec::with_capacity(args.measured_rounds);
    let mut packed_samples = Vec::with_capacity(args.measured_rounds);
    for round in 0..args.measured_rounds
    {
        if round.is_multiple_of(2)
        {
            wide_samples.push(time_nanos(|| run_wide(&tokenizer, &texts)));
            packed_samples.push(time_nanos(|| run_packed(&tokenizer, &texts)));
        }
        else
        {
            packed_samples.push(time_nanos(|| run_packed(&tokenizer, &texts)));
            wide_samples.push(time_nanos(|| run_wide(&tokenizer, &texts)));
        }
    }

    let wide_median = median(&wide_samples).expect("wide samples are non-empty");
    let packed_median = median(&packed_samples).expect("packed samples are non-empty");
    println!("files={}", texts.len());
    println!("utf8_bytes={}", texts.iter().map(String::len).sum::<usize>());
    println!("merge_rules={}", tokenizer.wide_merges.len());
    println!(
        "wide_entry_payload_bytes={}",
        std::mem::size_of::<((usize, usize), usize)>()
    );
    println!(
        "packed_entry_payload_bytes={}",
        std::mem::size_of::<(PairKey, u32)>()
    );
    println!("wide_median_ns={wide_median}");
    println!("packed_median_ns={packed_median}");
    if packed_median != 0
    {
        println!(
            "packed_speedup_x={:.6}",
            wide_median as f64 / packed_median as f64
        );
    }
}

fn run_wide(tokenizer: &ProbeTokenizer, texts: &[String]) -> usize {
    texts
        .iter()
        .map(|text| black_box(tokenizer.encode_wide(black_box(text))).len())
        .sum()
}

fn run_packed(tokenizer: &ProbeTokenizer, texts: &[String]) -> usize {
    texts
        .iter()
        .map(|text| black_box(tokenizer.encode_packed(black_box(text))).len())
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
    fn packed_payload_is_smaller_on_64_bit_hosts() {
        if usize::BITS == 64
        {
            assert!(
                std::mem::size_of::<(PairKey, u32)>()
                    < std::mem::size_of::<((usize, usize), usize)>()
            );
        }
    }

    #[test]
    fn packed_and_wide_parallel_passes_match() {
        let wide = HashMap::from([((1usize, 2usize), 10usize), ((10, 3), 11)]);
        let packed = HashMap::from([
            (PairKey::new(1, 2), 10u32),
            (PairKey::new(10, 3), 11u32),
        ]);
        assert_eq!(apply_wide(vec![1, 2, 3], &wide), vec![11]);
        assert_eq!(apply_packed(vec![1, 2, 3], &packed), vec![11]);
    }

    #[test]
    fn median_is_overflow_safe() {
        assert_eq!(median(&[u64::MAX - 1, u64::MAX]), Some(u64::MAX - 1));
    }
}
