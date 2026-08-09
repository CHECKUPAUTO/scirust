use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use scirust_sciagent::{PackedRule, PairKey};

#[derive(Parser)]
#[command(
    name = "tokenizer-csr-soa-probe",
    about = "Compare AoS and SoA left-indexed CSR canonical BPE rule lookup"
)]
struct Args {
    #[arg(short, long)]
    tokenizer: PathBuf,

    #[arg(long, default_value_t = 100)]
    sweeps: usize,

    #[arg(long, default_value_t = 3)]
    warmup_rounds: usize,

    #[arg(long, default_value_t = 11)]
    measured_rounds: usize,
}

#[derive(Clone, Debug)]
struct CsrAos {
    offsets: Vec<u32>,
    entries: Vec<(u32, PackedRule)>,
}

#[derive(Clone, Debug)]
struct CsrSoa {
    offsets: Vec<u32>,
    rights: Vec<u32>,
    rules: Vec<PackedRule>,
}

impl CsrAos {
    fn build(entries: &[(PairKey, PackedRule)], vocab_size: usize) -> Result<Self, String> {
        let (offsets, sorted) = build_sorted(entries, vocab_size)?;
        let entries = sorted
            .into_iter()
            .map(|(key, rule)| (key.right(), rule))
            .collect();
        Ok(Self { offsets, entries })
    }

    #[inline]
    fn get(&self, key: PairKey) -> Option<PackedRule> {
        let (start, end) = range(&self.offsets, key.left())?;
        let slice = &self.entries[start..end];
        slice
            .binary_search_by_key(&key.right(), |(right, _)| *right)
            .ok()
            .map(|index| slice[index].1)
    }
}

impl CsrSoa {
    fn build(entries: &[(PairKey, PackedRule)], vocab_size: usize) -> Result<Self, String> {
        let (offsets, sorted) = build_sorted(entries, vocab_size)?;
        let mut rights = Vec::with_capacity(sorted.len());
        let mut rules = Vec::with_capacity(sorted.len());
        for (key, rule) in sorted
        {
            rights.push(key.right());
            rules.push(rule);
        }
        Ok(Self {
            offsets,
            rights,
            rules,
        })
    }

    #[inline]
    fn get(&self, key: PairKey) -> Option<PackedRule> {
        let (start, end) = range(&self.offsets, key.left())?;
        self.rights[start..end]
            .binary_search(&key.right())
            .ok()
            .map(|local| self.rules[start + local])
    }
}

fn build_sorted(
    entries: &[(PairKey, PackedRule)],
    vocab_size: usize,
) -> Result<(Vec<u32>, Vec<(PairKey, PackedRule)>), String> {
    let mut sorted = entries.to_vec();
    sorted.sort_unstable_by_key(|(key, _)| *key);
    for pair in sorted.windows(2)
    {
        if pair[0].0 == pair[1].0
        {
            return Err(format!("duplicate merge pair key {}", pair[0].0.raw()));
        }
    }

    let mut offsets = vec![0u32; vocab_size + 1];
    for &(key, _) in &sorted
    {
        let left = usize::try_from(key.left()).map_err(|_| "left id exceeds usize")?;
        if left >= vocab_size
        {
            return Err(format!("left id {left} exceeds vocab size {vocab_size}"));
        }
        offsets[left + 1] = offsets[left + 1]
            .checked_add(1)
            .ok_or_else(|| "CSR degree exceeds u32".to_string())?;
    }
    for index in 1..offsets.len()
    {
        offsets[index] = offsets[index]
            .checked_add(offsets[index - 1])
            .ok_or_else(|| "CSR rule count exceeds u32".to_string())?;
    }
    Ok((offsets, sorted))
}

fn range(offsets: &[u32], left: u32) -> Option<(usize, usize)> {
    let left = usize::try_from(left).ok()?;
    if left + 1 >= offsets.len()
    {
        return None;
    }
    Some((
        usize::try_from(offsets[left]).ok()?,
        usize::try_from(offsets[left + 1]).ok()?,
    ))
}

fn main() {
    let args = Args::parse();
    if args.sweeps == 0 || args.measured_rounds == 0
    {
        panic!("--sweeps and --measured-rounds must be greater than zero");
    }

    let (vocab_size, entries) =
        load_canonical_rules(&args.tokenizer).expect("failed to load canonical rules");
    let aos = CsrAos::build(&entries, vocab_size).expect("failed to build AoS CSR");
    let soa = CsrSoa::build(&entries, vocab_size).expect("failed to build SoA CSR");
    let queries = build_queries(&entries);
    for &query in &queries
    {
        assert_eq!(aos.get(query), soa.get(query), "CSR layout changed lookup result");
    }

    for _ in 0..args.warmup_rounds
    {
        black_box(run_aos(&aos, &queries, args.sweeps));
        black_box(run_soa(&soa, &queries, args.sweeps));
    }

    let mut aos_samples = Vec::with_capacity(args.measured_rounds);
    let mut soa_samples = Vec::with_capacity(args.measured_rounds);
    for round in 0..args.measured_rounds
    {
        if round.is_multiple_of(2)
        {
            aos_samples.push(time_nanos(|| run_aos(&aos, &queries, args.sweeps)));
            soa_samples.push(time_nanos(|| run_soa(&soa, &queries, args.sweeps)));
        }
        else
        {
            soa_samples.push(time_nanos(|| run_soa(&soa, &queries, args.sweeps)));
            aos_samples.push(time_nanos(|| run_aos(&aos, &queries, args.sweeps)));
        }
    }

    let aos_median = median(&aos_samples).expect("AoS samples are non-empty");
    let soa_median = median(&soa_samples).expect("SoA samples are non-empty");
    let aos_payload = aos.offsets.len() * std::mem::size_of::<u32>()
        + aos.entries.len() * std::mem::size_of::<(u32, PackedRule)>();
    let soa_payload = soa.offsets.len() * std::mem::size_of::<u32>()
        + soa.rights.len() * std::mem::size_of::<u32>()
        + soa.rules.len() * std::mem::size_of::<PackedRule>();

    println!("vocab_size={vocab_size}");
    println!("rules={}", entries.len());
    println!("queries_per_sweep={}", queries.len());
    println!("sweeps={}", args.sweeps);
    println!("aos_payload_bytes={aos_payload}");
    println!("soa_payload_bytes={soa_payload}");
    println!("aos_median_ns={aos_median}");
    println!("soa_median_ns={soa_median}");
    if soa_median != 0
    {
        println!("soa_speedup_x={:.6}", aos_median as f64 / soa_median as f64);
    }
}

fn load_canonical_rules(path: &PathBuf) -> Result<(usize, Vec<(PairKey, PackedRule)>), String> {
    let input = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let value: serde_json::Value =
        serde_json::from_str(&input).map_err(|error| error.to_string())?;
    if value
        .get("merge_semantics")
        .and_then(serde_json::Value::as_str)
        != Some("canonical-rank-v1")
    {
        return Err("SoA CSR probe requires merge_semantics=canonical-rank-v1".to_string());
    }
    let vocab_size = value
        .get("vocab")
        .and_then(serde_json::Value::as_object)
        .map(serde_json::Map::len)
        .ok_or_else(|| "tokenizer missing vocab object".to_string())?;
    let merges = value
        .get("merges")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "tokenizer missing merges array".to_string())?;

    let mut output = Vec::with_capacity(merges.len());
    for (rank, merge) in merges.iter().enumerate()
    {
        let text = merge
            .as_str()
            .ok_or_else(|| format!("merge rule {rank} is not a string"))?;
        let mut parts = text.split_whitespace();
        let left = parse_u32(parts.next(), rank, "left")?;
        let right = parse_u32(parts.next(), rank, "right")?;
        let token = parse_u32(parts.next(), rank, "output")?;
        if parts.next().is_some()
        {
            return Err(format!("merge rule {rank} has extra fields"));
        }
        let rank = u32::try_from(rank).map_err(|_| "merge rank exceeds u32".to_string())?;
        output.push((PairKey::new(left, right), PackedRule::new(rank, token)));
    }
    Ok((vocab_size, output))
}

fn parse_u32(value: Option<&str>, rank: usize, field: &str) -> Result<u32, String> {
    value
        .ok_or_else(|| format!("merge rule {rank} missing {field}"))?
        .parse::<u32>()
        .map_err(|_| format!("merge rule {rank} has invalid {field}"))
}

fn build_queries(entries: &[(PairKey, PackedRule)]) -> Vec<PairKey> {
    let mut queries = Vec::with_capacity(entries.len().saturating_mul(2));
    for &(key, _) in entries
    {
        queries.push(key);
        queries.push(PairKey::new(
            key.left(),
            key.right().wrapping_add(0x9e37_79b9),
        ));
    }
    queries
}

fn run_aos(table: &CsrAos, queries: &[PairKey], sweeps: usize) -> u64 {
    let mut checksum = 0u64;
    for _ in 0..sweeps
    {
        for &query in queries
        {
            if let Some(rule) = table.get(black_box(query))
            {
                checksum ^= rule.raw();
            }
        }
    }
    checksum
}

fn run_soa(table: &CsrSoa, queries: &[PairKey], sweeps: usize) -> u64 {
    let mut checksum = 0u64;
    for _ in 0..sweeps
    {
        for &query in queries
        {
            if let Some(rule) = table.get(black_box(query))
            {
                checksum ^= rule.raw();
            }
        }
    }
    checksum
}

fn time_nanos(f: impl FnOnce() -> u64) -> u64 {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn entries() -> Vec<(PairKey, PackedRule)> {
        vec![
            (PairKey::new(7, 3), PackedRule::new(2, 12)),
            (PairKey::new(1, 9), PackedRule::new(0, 10)),
            (PairKey::new(1, 2), PackedRule::new(1, 11)),
        ]
    }

    #[test]
    fn soa_matches_aos_hits_and_misses() {
        let entries = entries();
        let aos = CsrAos::build(&entries, 16).unwrap();
        let soa = CsrSoa::build(&entries, 16).unwrap();
        for query in build_queries(&entries)
        {
            assert_eq!(aos.get(query), soa.get(query));
        }
    }

    #[test]
    fn soa_payload_removes_aos_padding() {
        if std::mem::size_of::<(u32, PackedRule)>() == 16
        {
            assert_eq!(std::mem::size_of::<u32>() + std::mem::size_of::<PackedRule>(), 12);
        }
    }

    #[test]
    fn median_is_overflow_safe() {
        assert_eq!(median(&[u64::MAX - 1, u64::MAX]), Some(u64::MAX - 1));
    }
}
