use std::collections::BTreeMap;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use scirust_sciagent::{PackedRule, PairKey};

#[derive(Parser)]
#[command(
    name = "tokenizer-rule-table-probe",
    about = "Compare deterministic BTreeMap and flat sorted lookup for canonical BPE rules"
)]
struct Args {
    /// Canonical tokenizer JSON containing ordered merge rules.
    #[arg(short, long)]
    tokenizer: PathBuf,

    /// Number of full deterministic lookup sweeps per measured round.
    #[arg(long, default_value_t = 100)]
    sweeps: usize,

    /// Warm-up rounds per backend.
    #[arg(long, default_value_t = 2)]
    warmup_rounds: usize,

    /// Measured rounds per backend.
    #[arg(long, default_value_t = 9)]
    measured_rounds: usize,
}

#[derive(Clone, Debug)]
struct FlatRuleTable {
    entries: Vec<(PairKey, PackedRule)>,
}

impl FlatRuleTable {
    fn from_entries(mut entries: Vec<(PairKey, PackedRule)>) -> Result<Self, String> {
        entries.sort_unstable_by_key(|(key, _)| *key);
        for pair in entries.windows(2)
        {
            if pair[0].0 == pair[1].0
            {
                return Err(format!("duplicate merge pair key {}", pair[0].0.raw()));
            }
        }
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

fn main() {
    let args = Args::parse();
    if args.sweeps == 0 || args.measured_rounds == 0
    {
        panic!("--sweeps and --measured-rounds must be greater than zero");
    }

    let entries = load_canonical_rules(&args.tokenizer).expect("failed to load canonical rules");
    if entries.is_empty()
    {
        panic!("canonical tokenizer has no merge rules");
    }

    let tree = entries.iter().copied().collect::<BTreeMap<_, _>>();
    let flat = FlatRuleTable::from_entries(entries.clone()).expect("invalid flat rule table");
    let queries = build_queries(&entries);
    verify_lookup_parity(&tree, &flat, &queries).expect("rule-table lookup mismatch");

    for _ in 0..args.warmup_rounds
    {
        black_box(run_tree(&tree, &queries, args.sweeps));
        black_box(run_flat(&flat, &queries, args.sweeps));
    }

    let mut tree_samples = Vec::with_capacity(args.measured_rounds);
    let mut flat_samples = Vec::with_capacity(args.measured_rounds);
    for round in 0..args.measured_rounds
    {
        if round.is_multiple_of(2)
        {
            tree_samples.push(time_nanos(|| run_tree(&tree, &queries, args.sweeps)));
            flat_samples.push(time_nanos(|| run_flat(&flat, &queries, args.sweeps)));
        }
        else
        {
            flat_samples.push(time_nanos(|| run_flat(&flat, &queries, args.sweeps)));
            tree_samples.push(time_nanos(|| run_tree(&tree, &queries, args.sweeps)));
        }
    }

    let tree_median = median(&tree_samples).expect("tree samples are non-empty");
    let flat_median = median(&flat_samples).expect("flat samples are non-empty");
    let speedup = if flat_median == 0
    {
        None
    }
    else
    {
        Some(tree_median as f64 / flat_median as f64)
    };

    println!("rules={}", entries.len());
    println!("queries_per_sweep={}", queries.len());
    println!("sweeps={}", args.sweeps);
    println!("tree_median_ns={tree_median}");
    println!("flat_median_ns={flat_median}");
    if let Some(speedup) = speedup
    {
        println!("flat_speedup_x={speedup:.6}");
    }
}

fn load_canonical_rules(path: &PathBuf) -> Result<Vec<(PairKey, PackedRule)>, String> {
    let input = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let value: serde_json::Value = serde_json::from_str(&input).map_err(|error| error.to_string())?;
    if value
        .get("merge_semantics")
        .and_then(serde_json::Value::as_str)
        != Some("canonical-rank-v1")
    {
        return Err("rule-table probe requires merge_semantics=canonical-rank-v1".to_string());
    }
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
    Ok(output)
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
        queries.push(PairKey::new(key.left(), key.right().wrapping_add(0x9e37_79b9)));
    }
    queries
}

fn verify_lookup_parity(
    tree: &BTreeMap<PairKey, PackedRule>,
    flat: &FlatRuleTable,
    queries: &[PairKey],
) -> Result<(), String> {
    for &query in queries
    {
        if tree.get(&query).copied() != flat.get(query)
        {
            return Err(format!("lookup mismatch for pair key {}", query.raw()));
        }
    }
    Ok(())
}

fn run_tree(
    tree: &BTreeMap<PairKey, PackedRule>,
    queries: &[PairKey],
    sweeps: usize,
) -> u64 {
    let mut checksum = 0u64;
    for _ in 0..sweeps
    {
        for &query in queries
        {
            if let Some(rule) = tree.get(black_box(&query)).copied()
            {
                checksum ^= rule.raw();
            }
        }
    }
    checksum
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
    fn flat_lookup_matches_tree_for_hits_and_misses() {
        let entries = entries();
        let tree = entries.iter().copied().collect::<BTreeMap<_, _>>();
        let flat = FlatRuleTable::from_entries(entries.clone()).unwrap();
        let queries = build_queries(&entries);
        verify_lookup_parity(&tree, &flat, &queries).unwrap();
    }

    #[test]
    fn duplicate_pair_keys_are_rejected() {
        let key = PairKey::new(1, 2);
        assert!(FlatRuleTable::from_entries(vec![
            (key, PackedRule::new(0, 3)),
            (key, PackedRule::new(1, 4)),
        ])
        .is_err());
    }

    #[test]
    fn median_is_overflow_safe() {
        assert_eq!(median(&[u64::MAX - 1, u64::MAX]), Some(u64::MAX - 1));
        assert_eq!(median(&[3, 1, 2]), Some(2));
    }
}
