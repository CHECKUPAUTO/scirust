use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use scirust_sciagent::{
    BpeKernel, ElasticProfile, ElasticTextTokenizer, ElasticThresholds,
};

const DEFAULT_LENGTHS: &str = "8,16,32,64,96,128";

#[derive(Parser)]
#[command(
    name = "tokenizer-text-encode-probe",
    about = "Measure canonical TinyScan from UTF-8 text bytes through final Token IDs"
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

fn main() {
    let args = Args::parse();
    if args.sweeps == 0 || args.measured_rounds == 0
    {
        panic!("--sweeps and --measured-rounds must be greater than zero");
    }

    let lengths = parse_lengths(&args.lengths).expect("invalid --lengths");
    let tokenizer_json =
        fs::read_to_string(&args.tokenizer).expect("failed to read canonical tokenizer");
    let thresholds = ElasticThresholds::new(16, 32, 64, 96, 128)
        .expect("fixed text-probe thresholds must be valid");
    let reference_profile = ElasticProfile::reference_only(thresholds);
    let tiny_profile = ElasticProfile::new(thresholds, [BpeKernel::TinyScan; 6]);
    let reference = ElasticTextTokenizer::from_json_str(&tokenizer_json, reference_profile)
        .expect("failed to load reference tokenizer");
    let tiny = ElasticTextTokenizer::from_json_str(&tokenizer_json, tiny_profile)
        .expect("failed to load TinyScan tokenizer");
    let cases = collect_ascii_cases(&args.input, &lengths).expect("failed to build text cases");

    for (length, text) in &cases
    {
        let expected = reference.encode(text);
        let candidate = tiny.encode(text);
        assert_eq!(
            candidate.ids, expected.ids,
            "TinyScan changed Token IDs for text length {length}"
        );
        assert_eq!(candidate.requested_kernel, BpeKernel::TinyScan);
        assert_eq!(candidate.executed_kernel, BpeKernel::TinyScan);
    }

    println!("cases={}", cases.len());
    println!("sweeps={}", args.sweeps);
    println!("piece_len,median_ns");

    for &length in &lengths
    {
        let length_cases = cases
            .iter()
            .filter(|(case_len, _)| *case_len == length)
            .map(|(_, text)| text.as_str())
            .collect::<Vec<_>>();
        if length_cases.is_empty()
        {
            panic!("no text case for length {length}");
        }

        for _ in 0..args.warmup_rounds
        {
            black_box(run(&tiny, &length_cases, args.sweeps));
        }

        let mut samples = Vec::with_capacity(args.measured_rounds);
        for _ in 0..args.measured_rounds
        {
            samples.push(time_nanos(|| run(&tiny, &length_cases, args.sweeps)));
        }
        let median = median(&samples).expect("measured samples are non-empty");
        println!("{length},{median}");
    }
}

fn collect_ascii_cases(
    inputs: &[PathBuf],
    lengths: &[usize],
) -> Result<Vec<(usize, String)>, String> {
    let mut paths = inputs.to_vec();
    paths.sort();
    paths.dedup();
    if paths.is_empty()
    {
        return Err("at least one --input file is required".to_string());
    }

    let mut cases = Vec::new();
    for path in paths
    {
        let content = fs::read_to_string(&path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let ascii = content
            .chars()
            .filter(char::is_ascii)
            .collect::<String>();
        for &length in lengths
        {
            if ascii.len() >= length
            {
                cases.push((length, ascii[..length].to_string()));
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
    if lengths.contains(&0)
    {
        return Err("lengths must be greater than zero".to_string());
    }
    lengths.sort_unstable();
    lengths.dedup();
    Ok(lengths)
}

fn run(tokenizer: &ElasticTextTokenizer, cases: &[&str], sweeps: usize) -> u64 {
    let mut checksum = 0u64;
    for _ in 0..sweeps
    {
        for &text in cases
        {
            let encoded = tokenizer.encode(black_box(text));
            checksum ^= u64::try_from(encoded.ids.len()).unwrap_or(u64::MAX);
            if let Some(&last) = encoded.ids.last()
            {
                checksum ^= u64::try_from(last).unwrap_or(u64::MAX);
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

    #[test]
    fn lengths_are_sorted_and_deduplicated() {
        assert_eq!(parse_lengths("128,8,16,16").unwrap(), vec![8, 16, 128]);
        assert!(parse_lengths("0").is_err());
    }

    #[test]
    fn median_is_overflow_safe() {
        assert_eq!(median(&[u64::MAX - 1, u64::MAX]), Some(u64::MAX - 1));
    }
}
