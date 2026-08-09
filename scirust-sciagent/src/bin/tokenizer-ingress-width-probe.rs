use std::collections::BTreeMap;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

const TINY_CAPACITY: usize = 128;
const DEFAULT_LENGTHS: &str = "8,16,32,64,96,128";

#[derive(Parser)]
#[command(
    name = "tokenizer-ingress-width-probe",
    about = "Measure Vec<usize> plus compaction versus direct u32 TinyScan ingress"
)]
struct Args {
    #[arg(short, long)]
    tokenizer: PathBuf,

    #[arg(short, long)]
    input: Vec<PathBuf>,

    #[arg(long, default_value = DEFAULT_LENGTHS)]
    lengths: String,

    #[arg(long, default_value_t = 1_000)]
    sweeps: usize,

    #[arg(long, default_value_t = 3)]
    warmup_rounds: usize,

    #[arg(long, default_value_t = 11)]
    measured_rounds: usize,
}

#[derive(Clone, Debug)]
struct ByteLuts {
    wide: [usize; 256],
    compact: [u32; 256],
}

fn main() {
    let args = Args::parse();
    if args.sweeps == 0 || args.measured_rounds == 0
    {
        panic!("--sweeps and --measured-rounds must be greater than zero");
    }
    let lengths = parse_lengths(&args.lengths).expect("invalid --lengths");
    let luts = load_byte_luts(&args.tokenizer).expect("failed to load canonical byte vocabulary");
    let cases = collect_cases(&args.input, &lengths).expect("failed to build ingress cases");

    println!("cases={}", cases.len());
    println!("sweeps={}", args.sweeps);
    println!("wide_lut_bytes={}", std::mem::size_of_val(&luts.wide));
    println!("compact_lut_bytes={}", std::mem::size_of_val(&luts.compact));
    println!("piece_len,current_median_ns,direct_u32_median_ns,speedup_x");

    for &length in &lengths
    {
        let length_cases = cases
            .iter()
            .filter(|case| case.len() == length)
            .collect::<Vec<_>>();
        if length_cases.is_empty()
        {
            panic!("no ingress case for length {length}");
        }

        for case in &length_cases
        {
            assert_same_ids(&luts, case);
        }

        for _ in 0..args.warmup_rounds
        {
            black_box(run_current(&luts, &length_cases, args.sweeps));
            black_box(run_direct(&luts, &length_cases, args.sweeps));
        }

        let mut current = Vec::with_capacity(args.measured_rounds);
        let mut direct = Vec::with_capacity(args.measured_rounds);
        for round in 0..args.measured_rounds
        {
            if round.is_multiple_of(2)
            {
                current.push(time_nanos(|| run_current(&luts, &length_cases, args.sweeps)));
                direct.push(time_nanos(|| run_direct(&luts, &length_cases, args.sweeps)));
            }
            else
            {
                direct.push(time_nanos(|| run_direct(&luts, &length_cases, args.sweeps)));
                current.push(time_nanos(|| run_current(&luts, &length_cases, args.sweeps)));
            }
        }

        let current_median = median(&current).expect("current samples are non-empty");
        let direct_median = median(&direct).expect("direct samples are non-empty");
        let speedup = if direct_median == 0
        {
            0.0
        }
        else
        {
            current_median as f64 / direct_median as f64
        };
        println!("{length},{current_median},{direct_median},{speedup:.6}");
    }
}

fn load_byte_luts(path: &PathBuf) -> Result<ByteLuts, String> {
    let input = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let value: serde_json::Value =
        serde_json::from_str(&input).map_err(|error| error.to_string())?;
    if value
        .get("merge_semantics")
        .and_then(serde_json::Value::as_str)
        != Some("canonical-rank-v1")
    {
        return Err("ingress probe requires merge_semantics=canonical-rank-v1".to_string());
    }
    if value.get("version").and_then(serde_json::Value::as_str) != Some("byte_level_v2")
    {
        return Err("ingress probe requires version=byte_level_v2".to_string());
    }
    let vocab: BTreeMap<String, usize> = serde_json::from_value(
        value
            .get("vocab")
            .cloned()
            .ok_or_else(|| "tokenizer missing vocab".to_string())?,
    )
    .map_err(|error| error.to_string())?;

    let mut wide = [0usize; 256];
    let mut compact = [0u32; 256];
    for byte in 0u8..=255
    {
        let token = byte_to_unit(byte).to_string();
        let id = vocab
            .get(&token)
            .copied()
            .ok_or_else(|| format!("tokenizer missing byte token {byte}"))?;
        wide[usize::from(byte)] = id;
        compact[usize::from(byte)] =
            u32::try_from(id).map_err(|_| format!("byte token {byte} id {id} exceeds u32"))?;
    }
    Ok(ByteLuts { wide, compact })
}

fn collect_cases(inputs: &[PathBuf], lengths: &[usize]) -> Result<Vec<Vec<u8>>, String> {
    let mut paths = inputs.to_vec();
    paths.sort();
    paths.dedup();
    if paths.is_empty()
    {
        return Err("at least one --input is required".to_string());
    }

    let mut cases = Vec::new();
    for path in paths
    {
        let bytes = fs::read(&path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        for &length in lengths
        {
            if length > TINY_CAPACITY
            {
                return Err(format!("length {length} exceeds TinyScan capacity"));
            }
            if bytes.len() >= length
            {
                cases.push(bytes[..length].to_vec());
            }
        }
    }
    Ok(cases)
}

fn parse_lengths(input: &str) -> Result<Vec<usize>, String> {
    let mut lengths = input
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| format!("invalid length `{value}`"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if lengths.iter().any(|&length| length == 0 || length > TINY_CAPACITY)
    {
        return Err(format!("lengths must be in 1..={TINY_CAPACITY}"));
    }
    lengths.sort_unstable();
    lengths.dedup();
    Ok(lengths)
}

fn current_ingress(luts: &ByteLuts, bytes: &[u8]) -> ([u32; TINY_CAPACITY], usize) {
    let wide = bytes
        .iter()
        .copied()
        .map(|byte| luts.wide[usize::from(byte)])
        .collect::<Vec<_>>();
    assert!(wide.iter().all(|&id| u32::try_from(id).is_ok()));
    let mut work = [0u32; TINY_CAPACITY];
    for (slot, id) in work.iter_mut().zip(wide)
    {
        *slot = u32::try_from(id).expect("preflight checked byte IDs");
    }
    (work, bytes.len())
}

fn direct_ingress(luts: &ByteLuts, bytes: &[u8]) -> ([u32; TINY_CAPACITY], usize) {
    let mut work = [0u32; TINY_CAPACITY];
    for (slot, &byte) in work.iter_mut().zip(bytes)
    {
        *slot = luts.compact[usize::from(byte)];
    }
    (work, bytes.len())
}

fn assert_same_ids(luts: &ByteLuts, bytes: &[u8]) {
    let (current, current_len) = current_ingress(luts, bytes);
    let (direct, direct_len) = direct_ingress(luts, bytes);
    assert_eq!(current_len, direct_len);
    assert_eq!(&current[..current_len], &direct[..direct_len]);
}

fn run_current(luts: &ByteLuts, cases: &[&Vec<u8>], sweeps: usize) -> u64 {
    let mut checksum = 0u64;
    for _ in 0..sweeps
    {
        for case in cases
        {
            let (work, len) = current_ingress(black_box(luts), black_box(case));
            checksum ^= u64::from(work[len - 1]);
        }
    }
    checksum
}

fn run_direct(luts: &ByteLuts, cases: &[&Vec<u8>], sweeps: usize) -> u64 {
    let mut checksum = 0u64;
    for _ in 0..sweeps
    {
        for case in cases
        {
            let (work, len) = direct_ingress(black_box(luts), black_box(case));
            checksum ^= u64::from(work[len - 1]);
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

    fn luts() -> ByteLuts {
        let mut wide = [0usize; 256];
        let mut compact = [0u32; 256];
        for index in 0..256
        {
            wide[index] = index + 4;
            compact[index] = u32::try_from(index + 4).unwrap();
        }
        ByteLuts { wide, compact }
    }

    #[test]
    fn direct_ingress_matches_current_ids() {
        let luts = luts();
        for len in [1usize, 8, 16, 64, 128]
        {
            let bytes = (0..len)
                .map(|index| u8::try_from(index).unwrap())
                .collect::<Vec<_>>();
            assert_same_ids(&luts, &bytes);
        }
    }

    #[test]
    fn lengths_are_sorted_and_bounded() {
        assert_eq!(parse_lengths("128,8,16,16").unwrap(), vec![8, 16, 128]);
        assert!(parse_lengths("129").is_err());
        assert!(parse_lengths("0").is_err());
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
