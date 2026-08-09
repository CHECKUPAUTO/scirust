use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;

#[derive(Parser)]
#[command(
    name = "legacy-bpe-lut-width-probe",
    about = "Measure usize versus u32 byte-ID LUTs under identical cached legacy BPE semantics"
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
    byte_ids_usize: [usize; 256],
    byte_ids_u32: [u32; 256],
    merges: HashMap<(usize, usize), usize>,
}

impl ProbeTokenizer {
    fn load(path: &Path) -> Result<Self, String> {
        let input = fs::read_to_string(path).map_err(|error| error.to_string())?;
        let value: serde_json::Value =
            serde_json::from_str(&input).map_err(|error| error.to_string())?;
        if value.get("version").and_then(serde_json::Value::as_str) != Some("byte_level_v2")
        {
            return Err("LUT-width probe requires version=byte_level_v2".to_string());
        }
        if let Some(semantics) = value
            .get("merge_semantics")
            .and_then(serde_json::Value::as_str)
        {
            if semantics != "legacy-parallel-v1"
            {
                return Err(format!(
                    "LUT-width probe rejects merge_semantics={semantics}"
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

        let mut byte_ids_usize = [0usize; 256];
        let mut byte_ids_u32 = [0u32; 256];
        for byte in 0u8..=255
        {
            let token = byte_to_unit(byte).to_string();
            let id = vocab
                .get(&token)
                .copied()
                .ok_or_else(|| format!("tokenizer missing byte token {byte}"))?;
            let compact = u32::try_from(id)
                .map_err(|_| format!("byte token {byte} id {id} exceeds u32"))?;
            byte_ids_usize[usize::from(byte)] = id;
            byte_ids_u32[usize::from(byte)] = compact;
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

        Ok(Self {
            byte_ids_usize,
            byte_ids_u32,
            merges,
        })
    }

    fn encode_usize(&self, text: &str) -> Vec<usize> {
        let ids = text
            .bytes()
            .map(|byte| self.byte_ids_usize[usize::from(byte)])
            .collect();
        apply_parallel_merges(ids, &self.merges)
    }

    fn encode_u32(&self, text: &str) -> Vec<usize> {
        let ids = text
            .bytes()
            .map(|byte| {
                usize::try_from(self.byte_ids_u32[usize::from(byte)])
                    .expect("u32 token id fits usize on supported targets")
            })
            .collect();
        apply_parallel_merges(ids, &self.merges)
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
        let wide = tokenizer.encode_usize(text);
        let compact = tokenizer.encode_u32(text);
        assert_eq!(wide, compact, "LUT width changed token IDs for input {index}");
    }

    for _ in 0..args.warmup_rounds
    {
        black_box(run_usize(&tokenizer, &texts));
        black_box(run_u32(&tokenizer, &texts));
    }

    let mut usize_samples = Vec::with_capacity(args.measured_rounds);
    let mut u32_samples = Vec::with_capacity(args.measured_rounds);
    for round in 0..args.measured_rounds
    {
        if round.is_multiple_of(2)
        {
            usize_samples.push(time_nanos(|| run_usize(&tokenizer, &texts)));
            u32_samples.push(time_nanos(|| run_u32(&tokenizer, &texts)));
        }
        else
        {
            u32_samples.push(time_nanos(|| run_u32(&tokenizer, &texts)));
            usize_samples.push(time_nanos(|| run_usize(&tokenizer, &texts)));
        }
    }

    let usize_median = median(&usize_samples).expect("usize samples are non-empty");
    let u32_median = median(&u32_samples).expect("u32 samples are non-empty");
    let bytes = texts.iter().map(String::len).sum::<usize>();

    println!("files={}", texts.len());
    println!("utf8_bytes={bytes}");
    println!(
        "usize_lut_bytes={}",
        std::mem::size_of::<[usize; 256]>()
    );
    println!("u32_lut_bytes={}", std::mem::size_of::<[u32; 256]>());
    println!("usize_lut_median_ns={usize_median}");
    println!("u32_lut_median_ns={u32_median}");
    if u32_median != 0
    {
        println!(
            "u32_lut_speedup_x={:.6}",
            usize_median as f64 / u32_median as f64
        );
    }
}

fn apply_parallel_merges(
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
            if index + 1 < ids.len()
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

fn run_usize(tokenizer: &ProbeTokenizer, texts: &[String]) -> usize {
    texts
        .iter()
        .map(|text| black_box(tokenizer.encode_usize(black_box(text))).len())
        .sum()
}

fn run_u32(tokenizer: &ProbeTokenizer, texts: &[String]) -> usize {
    texts
        .iter()
        .map(|text| black_box(tokenizer.encode_u32(black_box(text))).len())
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
    fn merge_loop_preserves_historical_parallel_passes() {
        let merges = HashMap::from([((1, 2), 10), ((10, 3), 11)]);
        assert_eq!(apply_parallel_merges(vec![1, 2, 3], &merges), vec![11]);
    }

    #[test]
    fn median_is_overflow_safe() {
        assert_eq!(median(&[u64::MAX - 1, u64::MAX]), Some(u64::MAX - 1));
    }

    #[test]
    fn compact_lut_is_half_width_on_64_bit_hosts() {
        assert_eq!(std::mem::size_of::<[u32; 256]>(), 1024);
        if usize::BITS == 64
        {
            assert_eq!(std::mem::size_of::<[usize; 256]>(), 2048);
        }
    }
}
