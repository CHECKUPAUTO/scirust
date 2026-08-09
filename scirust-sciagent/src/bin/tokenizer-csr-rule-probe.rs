use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use scirust_sciagent::{PackedRule, PairKey};

#[derive(Parser)]
#[command(
    name = "tokenizer-csr-rule-probe",
    about = "Compare flat global and left-indexed CSR canonical BPE rule lookup"
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
struct FlatRuleTable {
    entries: Vec<(PairKey, PackedRule)>,
}

impl FlatRuleTable {
    fn from_entries(mut entries: Vec<(PairKey, PackedRule)>) -> Result<Self, String> {
        entries.sort_unstable_by_key(|(key, _)| *key);
        reject_duplicates(&entries)?;
        Ok(Self { entries })
    }

    #[inline]
    fn get(&self, key: PairKey) -> Option<PackedRule> {
        self.entries
            .binary_search_by_key(&key, |(entry_key, _)| *entry_key)
            .ok()
            .map(|index| self.entries[index].1)
    }
}

#[derive(Clone, Debug)]
struct CsrRuleTable {
    offsets: Vec<u32>,
    entries: Vec<(u32, PackedRule)>,
}

impl CsrRuleTable {
    fn from_entries(
        mut entries: Vec<(PairKey, PackedRule)>,
        vocab_size: usize,
    ) -> Result<Self, String> {
        if vocab_size == 0
        {
            return Err("vocab must not be empty".to_string());
        }
        entries.sort_unstable_by_key(|(key, _)| *key);
        reject_duplicates(&entries)?;

        let mut offsets = vec![0u32; vocab_size + 1];
        for &(key, _) in &entries
        {
            let left = usize::try_from(key.left()).map_err(|_| "left id exceeds usize")?;
            if left >= vocab_size
            {
                return Err(format!(
                    "left token id {left} exceeds vocab size {vocab_size}"
                ));
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

        let entries = entries
            .into_iter()
            .map(|(key, rule)| (key.right(), rule))
            .collect();
        Ok(Self { offsets, entries })
    }

    #[inline]
    fn slice(&self, left: u32) -> &[(u32, PackedRule)] {
        let Ok(left) = usize::try_from(left)
        else
        {
            return &[];
        };
        if left + 1 >= self.offsets.len()
        {
            return &[];
        }
        let start = usize::try_from(self.offsets[left]).expect("u32 offset fits usize");
        let end = usize::try_from(self.offsets[left + 1]).expect("u32 offset fits usize");
        &self.entries[start..end]
    }

    #[inline]
    fn get_binary(&self, key: PairKey) -> Option<PackedRule> {
        let entries = self.slice(key.left());
        entries
            .binary_search_by_key(&key.right(), |(right, _)| *right)
            .ok()
            .map(|index| entries[index].1)
    }

    #[inline]
    fn get_linear(&self, key: PairKey) -> Option<PackedRule> {
        let right = key.right();
        self.slice(key.left())
            .iter()
            .find_map(|&(candidate, rule)| (candidate == right).then_some(rule))
    }
}

fn main() {
    let args = Args::parse();
    if args.sweeps == 0 || args.measured_rounds == 0
    {
        panic!("--sweeps and --measured-rounds must be greater than zero");
    }

    let (vocab_size, entries) =
        load_canonical_rules(&args.tokenizer).expect("failed to load canonical rules");
    if entries.is_empty()
    {
        panic!("canonical tokenizer has no merge rules");
    }

    let flat = FlatRuleTable::from_entries(entries.clone()).expect("invalid flat table");
    let csr = CsrRuleTable::from_entries(entries.clone(), vocab_size).expect("invalid CSR table");
    let queries = build_queries(&entries);
    verify_parity(&flat, &csr, &queries).expect("CSR lookup parity failure");

    for _ in 0..args.warmup_rounds
    {
        black_box(run_flat(&flat, &queries, args.sweeps));
        black_box(run_csr_binary(&csr, &queries, args.sweeps));
        black_box(run_csr_linear(&csr, &queries, args.sweeps));
    }

    let mut flat_samples = Vec::with_capacity(args.measured_rounds);
    let mut binary_samples = Vec::with_capacity(args.measured_rounds);
    let mut linear_samples = Vec::with_capacity(args.measured_rounds);
    for round in 0..args.measured_rounds
    {
        match round % 3
        {
            0 =>
            {
                flat_samples.push(time_nanos(|| run_flat(&flat, &queries, args.sweeps)));
                binary_samples.push(time_nanos(|| run_csr_binary(&csr, &queries, args.sweeps)));
                linear_samples.push(time_nanos(|| run_csr_linear(&csr, &queries, args.sweeps)));
            },
            1 =>
            {
                binary_samples.push(time_nanos(|| run_csr_binary(&csr, &queries, args.sweeps)));
                linear_samples.push(time_nanos(|| run_csr_linear(&csr, &queries, args.sweeps)));
                flat_samples.push(time_nanos(|| run_flat(&flat, &queries, args.sweeps)));
            },
            _ =>
            {
                linear_samples.push(time_nanos(|| run_csr_linear(&csr, &queries, args.sweeps)));
                flat_samples.push(time_nanos(|| run_flat(&flat, &queries, args.sweeps)));
                binary_samples.push(time_nanos(|| run_csr_binary(&csr, &queries, args.sweeps)));
            },
        }
    }

    let flat_median = median(&flat_samples).expect("flat samples are non-empty");
    let binary_median = median(&binary_samples).expect("binary samples are non-empty");
    let linear_median = median(&linear_samples).expect("linear samples are non-empty");
    let flat_payload = flat.entries.len() * std::mem::size_of::<(PairKey, PackedRule)>();
    let csr_payload = csr.offsets.len() * std::mem::size_of::<u32>()
        + csr.entries.len() * std::mem::size_of::<(u32, PackedRule)>();

    println!("vocab_size={vocab_size}");
    println!("rules={}", entries.len());
    println!("queries_per_sweep={}", queries.len());
    println!("sweeps={}", args.sweeps);
    println!("flat_payload_bytes={flat_payload}");
    println!("csr_payload_bytes={csr_payload}");
    println!("flat_median_ns={flat_median}");
    println!("csr_binary_median_ns={binary_median}");
    println!("csr_linear_median_ns={linear_median}");
    if binary_median != 0
    {
        println!(
            "csr_binary_speedup_x={:.6}",
            flat_median as f64 / binary_median as f64
        );
    }
    if linear_median != 0
    {
        println!(
            "csr_linear_speedup_x={:.6}",
            flat_median as f64 / linear_median as f64
        );
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
        return Err("CSR rule probe requires merge_semantics=canonical-rank-v1".to_string());
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

fn reject_duplicates(entries: &[(PairKey, PackedRule)]) -> Result<(), String> {
    for pair in entries.windows(2)
    {
        if pair[0].0 == pair[1].0
        {
            return Err(format!("duplicate merge pair key {}", pair[0].0.raw()));
        }
    }
    Ok(())
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

fn verify_parity(
    flat: &FlatRuleTable,
    csr: &CsrRuleTable,
    queries: &[PairKey],
) -> Result<(), String> {
    for &query in queries
    {
        let expected = flat.get(query);
        if csr.get_binary(query) != expected || csr.get_linear(query) != expected
        {
            return Err(format!("lookup mismatch for pair key {}", query.raw()));
        }
    }
    Ok(())
}

fn run_flat(flat: &FlatRuleTable, queries: &[PairKey], sweeps: usize) -> u64 {
    let mut checksum = 0u64;
    for _ in 0..sweeps
    {
        for &query in queries
        {
            if let Some(rule) = flat.get(black_box(query))
            {
                checksum ^= rule.raw();
            }
        }
    }
    checksum
}

fn run_csr_binary(csr: &CsrRuleTable, queries: &[PairKey], sweeps: usize) -> u64 {
    let mut checksum = 0u64;
    for _ in 0..sweeps
    {
        for &query in queries
        {
            if let Some(rule) = csr.get_binary(black_box(query))
            {
                checksum ^= rule.raw();
            }
        }
    }
    checksum
}

fn run_csr_linear(csr: &CsrRuleTable, queries: &[PairKey], sweeps: usize) -> u64 {
    let mut checksum = 0u64;
    for _ in 0..sweeps
    {
        for &query in queries
        {
            if let Some(rule) = csr.get_linear(black_box(query))
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
    fn csr_matches_flat_for_hits_and_misses() {
        let entries = entries();
        let flat = FlatRuleTable::from_entries(entries.clone()).unwrap();
        let csr = CsrRuleTable::from_entries(entries.clone(), 16).unwrap();
        verify_parity(&flat, &csr, &build_queries(&entries)).unwrap();
    }

    #[test]
    fn csr_offsets_bound_each_left_id() {
        let csr = CsrRuleTable::from_entries(entries(), 16).unwrap();
        assert_eq!(csr.slice(1).len(), 2);
        assert_eq!(csr.slice(7).len(), 1);
        assert!(csr.slice(15).is_empty());
        assert!(csr.slice(16).is_empty());
    }

    #[test]
    fn duplicate_pair_keys_are_rejected() {
        let key = PairKey::new(1, 2);
        assert!(
            CsrRuleTable::from_entries(
                vec![
                    (key, PackedRule::new(0, 3)),
                    (key, PackedRule::new(1, 4)),
                ],
                8,
            )
            .is_err()
        );
    }

    #[test]
    fn median_is_overflow_safe() {
        assert_eq!(median(&[u64::MAX - 1, u64::MAX]), Some(u64::MAX - 1));
    }
}
