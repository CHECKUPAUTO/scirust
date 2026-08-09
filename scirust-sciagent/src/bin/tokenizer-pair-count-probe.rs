use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::fs;
use std::hash::{BuildHasherDefault, Hasher};
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use scirust_sciagent::PairKey;

#[derive(Parser)]
#[command(
    name = "tokenizer-pair-count-probe",
    about = "Compare RandomState and identity-u64 hashing for canonical BPE pair counting"
)]
struct Args {
    /// UTF-8 or arbitrary-byte corpus files. Pair boundaries never cross files.
    #[arg(short, long)]
    input: Vec<PathBuf>,

    #[arg(long, default_value_t = 3)]
    warmup_rounds: usize,

    #[arg(long, default_value_t = 11)]
    measured_rounds: usize,
}

#[derive(Clone, Default)]
struct U64IdentityHasher(u64);

impl Hasher for U64IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        // PairKey's derived `Hash` uses `write_u64`; keep a deterministic
        // fallback for trait completeness rather than relying on a panic.
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for &byte in bytes
        {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.0 = hash;
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

type IdentityState = BuildHasherDefault<U64IdentityHasher>;
type RandomMap = HashMap<PairKey, u64, RandomState>;
type IdentityMap = HashMap<PairKey, u64, IdentityState>;

fn main() {
    let args = Args::parse();
    if args.measured_rounds == 0
    {
        panic!("--measured-rounds must be greater than zero");
    }
    let pairs = load_pairs(&args.input).expect("failed to build corpus pair stream");
    if pairs.is_empty()
    {
        panic!("pair-count probe requires at least one adjacent byte pair");
    }

    verify_count_parity(&pairs).expect("identity hashing changed pair counts");

    let mut random = RandomMap::with_capacity_and_hasher(pairs.len(), RandomState::new());
    let mut identity = IdentityMap::with_capacity_and_hasher(pairs.len(), IdentityState::default());
    for _ in 0..args.warmup_rounds
    {
        black_box(count_random(&mut random, &pairs));
        black_box(count_identity(&mut identity, &pairs));
    }

    let mut random_samples = Vec::with_capacity(args.measured_rounds);
    let mut identity_samples = Vec::with_capacity(args.measured_rounds);
    for round in 0..args.measured_rounds
    {
        if round.is_multiple_of(2)
        {
            random_samples.push(time_nanos(|| count_random(&mut random, &pairs)));
            identity_samples.push(time_nanos(|| count_identity(&mut identity, &pairs)));
        }
        else
        {
            identity_samples.push(time_nanos(|| count_identity(&mut identity, &pairs)));
            random_samples.push(time_nanos(|| count_random(&mut random, &pairs)));
        }
    }

    let random_median = median(&random_samples).expect("random samples are non-empty");
    let identity_median = median(&identity_samples).expect("identity samples are non-empty");
    println!("pair_observations={}", pairs.len());
    println!("distinct_pairs={}", identity.len());
    println!("random_state_median_ns={random_median}");
    println!("identity_u64_median_ns={identity_median}");
    if identity_median != 0
    {
        println!(
            "identity_speedup_x={:.6}",
            random_median as f64 / identity_median as f64
        );
    }
}

fn load_pairs(paths: &[PathBuf]) -> Result<Vec<PairKey>, String> {
    let mut paths = paths.to_vec();
    paths.sort();
    paths.dedup();
    let mut pairs = Vec::new();
    for path in paths
    {
        let bytes = fs::read(&path)
            .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        for window in bytes.windows(2)
        {
            let left = u32::from(window[0]) + 4;
            let right = u32::from(window[1]) + 4;
            pairs.push(PairKey::new(left, right));
        }
    }
    Ok(pairs)
}

fn count_random(map: &mut RandomMap, pairs: &[PairKey]) -> u64 {
    map.clear();
    for &pair in pairs
    {
        *map.entry(black_box(pair)).or_insert(0) += 1;
    }
    checksum(map.iter().map(|(key, count)| (key.raw(), *count)))
}

fn count_identity(map: &mut IdentityMap, pairs: &[PairKey]) -> u64 {
    map.clear();
    for &pair in pairs
    {
        *map.entry(black_box(pair)).or_insert(0) += 1;
    }
    checksum(map.iter().map(|(key, count)| (key.raw(), *count)))
}

fn checksum(entries: impl Iterator<Item = (u64, u64)>) -> u64 {
    entries.fold(0u64, |acc, (key, count)| {
        acc.wrapping_add(key.rotate_left(17) ^ count)
    })
}

fn verify_count_parity(pairs: &[PairKey]) -> Result<(), String> {
    let mut random = RandomMap::with_capacity_and_hasher(pairs.len(), RandomState::new());
    let mut identity = IdentityMap::with_capacity_and_hasher(pairs.len(), IdentityState::default());
    count_random(&mut random, pairs);
    count_identity(&mut identity, pairs);
    if random.len() != identity.len()
    {
        return Err("distinct pair counts differ".to_string());
    }
    for (key, count) in random
    {
        if identity.get(&key).copied() != Some(count)
        {
            return Err(format!("count mismatch for pair key {}", key.raw()));
        }
    }
    Ok(())
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
    use std::hash::{Hash, Hasher as _};

    #[test]
    fn pair_key_hash_is_exact_u64_identity() {
        let key = PairKey::new(0x1234_5678, 0x9abc_def0);
        let mut hasher = U64IdentityHasher::default();
        key.hash(&mut hasher);
        assert_eq!(hasher.finish(), key.raw());
    }

    #[test]
    fn identity_and_random_maps_count_identically() {
        let pairs = vec![
            PairKey::new(1, 2),
            PairKey::new(1, 2),
            PairKey::new(2, 3),
            PairKey::new(7, 9),
            PairKey::new(2, 3),
        ];
        verify_count_parity(&pairs).unwrap();
    }

    #[test]
    fn median_is_overflow_safe() {
        assert_eq!(median(&[u64::MAX - 1, u64::MAX]), Some(u64::MAX - 1));
        assert_eq!(median(&[3, 1, 2]), Some(2));
    }
}
