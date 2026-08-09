use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use clap::Parser;
use scirust_sciagent::train::dataset::{matches_extension, parse_extensions, skip_source_dir};
use scirust_sciagent::{
    AutotuneConfig, AutotuneResult, BpeKernel, BpeMergeSemantics, CalibrationCase,
    ElasticAutotuner, ElasticHardwareIdentity, ElasticProfile, StoredElasticProfile,
    VersionedBpeTokenizer, ordered_merges_fingerprint,
};

const DEFAULT_PROBE_LENGTHS: &str = "8,16,32,64,128,256,512,1024,2048,4096";
const AUTOTUNE_REPORT_SCHEMA_V1: u32 = 1;

#[derive(Parser)]
#[command(
    name = "tokenizer-autotune",
    about = "Calibrate ElasticTokenizer kernels and persist a hardware-local execution profile"
)]
struct Args {
    /// Explicitly canonical tokenizer (`merge_semantics=canonical-rank-v1`).
    #[arg(short, long)]
    tokenizer: PathBuf,

    /// Corpus files/directories used to construct representative byte pieces.
    #[arg(short, long)]
    input: Vec<PathBuf>,

    /// Output profile JSON.
    #[arg(short, long)]
    output: PathBuf,

    /// Optional raw timing report for reproducible before/after comparisons.
    #[arg(long, value_name = "FILE")]
    report: Option<PathBuf>,

    /// Comma-separated byte lengths to probe.
    #[arg(long, default_value = DEFAULT_PROBE_LENGTHS)]
    probe_lengths: String,

    /// Maximum number of distinct corpus pieces measured per probe length.
    #[arg(long, default_value_t = 3)]
    cases_per_length: usize,

    /// Warm-up executions per compatible kernel and case.
    #[arg(long, default_value_t = 1)]
    warmup_runs: usize,

    /// Timed executions per compatible kernel and case.
    #[arg(long, default_value_t = 5)]
    measured_runs: usize,

    /// Comma-separated source extensions used when walking directories.
    #[arg(long, default_value = "rs,md,toml")]
    extension: String,

    #[arg(long)]
    recursive: bool,

    /// Stable deployment-local discriminator included in the hardware identity.
    #[arg(long, default_value = "generic")]
    device: String,
}

struct RawReportContext<'a> {
    result: &'a AutotuneResult,
    profile: ElasticProfile,
    hardware: &'a ElasticHardwareIdentity,
    tokenizer_fingerprint: &'a str,
    probe_lengths: &'a [usize],
    cases_per_length: usize,
    warmup_runs: usize,
    measured_runs: usize,
}

fn main() {
    let args = Args::parse();
    let probe_lengths = parse_probe_lengths(&args.probe_lengths)
        .expect("invalid --probe-lengths; expected six or more strictly positive integers");
    if probe_lengths.len() < 6
    {
        panic!("ElasticTokenizer profile fitting requires at least six probe lengths");
    }
    if args.cases_per_length == 0
    {
        panic!("--cases-per-length must be greater than zero");
    }

    let tokenizer = VersionedBpeTokenizer::load_json(&args.tokenizer)
        .expect("failed to load tokenizer for auto-calibration");
    if tokenizer.merge_semantics() != BpeMergeSemantics::CanonicalRankV1
    {
        panic!("tokenizer-autotune requires merge_semantics=canonical-rank-v1");
    }
    let canonical = match tokenizer
    {
        VersionedBpeTokenizer::Canonical(tokenizer) => tokenizer,
        VersionedBpeTokenizer::Legacy(_) => unreachable!("semantic guard above rejected legacy"),
    };

    let vocab = load_reversible_byte_vocab(&args.tokenizer)
        .expect("failed to load byte vocabulary for calibration cases");
    let extensions = parse_extensions(&args.extension);
    let paths = collect_input_paths(&args.input, &extensions, args.recursive);
    let cases = collect_calibration_cases(&paths, &probe_lengths, args.cases_per_length, &vocab)
        .expect("failed to build calibration cases");

    let config = AutotuneConfig::new(args.warmup_runs, args.measured_runs)
        .expect("invalid auto-calibration repetition counts");
    let tuner = ElasticAutotuner::from_ordered_merges(canonical.ordered_merges(), config)
        .expect("invalid canonical merge table");

    eprintln!(
        "ElasticTokenizer autotune: {} cases across {} probe lengths, warmup={}, measured={}",
        cases.len(),
        probe_lengths.len(),
        args.warmup_runs,
        args.measured_runs
    );
    let result = tuner
        .calibrate(&cases)
        .expect("ElasticTokenizer calibration failed");
    let profile = result
        .fit_profile()
        .expect("failed to fit six-class ElasticTokenizer profile");

    let hardware = ElasticHardwareIdentity::new(
        std::env::consts::ARCH,
        std::env::consts::OS,
        args.device.clone(),
    );
    let stored = StoredElasticProfile::new(canonical.ordered_merges(), hardware.clone(), profile);
    stored
        .save(&args.output)
        .expect("failed to save ElasticTokenizer profile");

    if let Some(report_path) = &args.report
    {
        let tokenizer_fingerprint = ordered_merges_fingerprint(canonical.ordered_merges());
        let context = RawReportContext {
            result: &result,
            profile,
            hardware: &hardware,
            tokenizer_fingerprint: &tokenizer_fingerprint,
            probe_lengths: &probe_lengths,
            cases_per_length: args.cases_per_length,
            warmup_runs: args.warmup_runs,
            measured_runs: args.measured_runs,
        };
        write_raw_report(report_path, &context)
            .expect("failed to save ElasticTokenizer raw timing report");
        eprintln!("raw timing report saved to {report_path:?}");
    }

    eprintln!("hardware fingerprint: {}", hardware.fingerprint());
    eprintln!("fitted thresholds: {:?}", profile.thresholds());
    eprintln!("fitted kernels: {:?}", profile.kernels());
    eprintln!(
        "semantic mismatches rejected: {}",
        result.report().rejected_semantic_measurements()
    );
    eprintln!("profile saved to {:?}", args.output);
}

fn write_raw_report(
    path: &Path,
    context: &RawReportContext<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let measurements = context
        .result
        .measurements()
        .iter()
        .map(|measurement| {
            serde_json::json!({
                "piece_len": measurement.piece_len,
                "kernel": kernel_name(measurement.kernel),
                "elapsed_nanos": measurement.elapsed_nanos,
                "semantic_match": measurement.semantic_match,
            })
        })
        .collect::<Vec<_>>();
    let thresholds = context.profile.thresholds();
    let kernels = context
        .profile
        .kernels()
        .into_iter()
        .map(kernel_name)
        .collect::<Vec<_>>();
    let value = serde_json::json!({
        "schema_version": AUTOTUNE_REPORT_SCHEMA_V1,
        "tokenizer_fingerprint": context.tokenizer_fingerprint,
        "hardware": {
            "arch": context.hardware.arch.as_str(),
            "os": context.hardware.os.as_str(),
            "device": context.hardware.device.as_str(),
            "fingerprint": context.hardware.fingerprint(),
        },
        "calibration": {
            "probe_lengths": context.probe_lengths,
            "cases_per_length": context.cases_per_length,
            "warmup_runs": context.warmup_runs,
            "measured_runs": context.measured_runs,
        },
        "measurements": measurements,
        "rejected_semantic_measurements": context.result.report().rejected_semantic_measurements(),
        "fitted_profile": {
            "thresholds": {
                "s_max": thresholds.s_max,
                "m_max": thresholds.m_max,
                "l_max": thresholds.l_max,
                "xl_max": thresholds.xl_max,
                "xxl_max": thresholds.xxl_max,
            },
            "kernels": kernels,
        },
    });
    fs::write(path, serde_json::to_string_pretty(&value)?)?;
    Ok(())
}

const fn kernel_name(kernel: BpeKernel) -> &'static str {
    match kernel
    {
        BpeKernel::Reference => "reference",
        BpeKernel::TinyScan => "tiny_scan",
        BpeKernel::Indexed => "indexed",
        BpeKernel::Heap => "heap",
    }
}

fn parse_probe_lengths(input: &str) -> Result<Vec<usize>, String> {
    let mut lengths = input
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .map_err(|_| format!("invalid probe length `{part}`"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if lengths.contains(&0)
    {
        return Err("probe lengths must be greater than zero".to_string());
    }
    lengths.sort_unstable();
    lengths.dedup();
    Ok(lengths)
}

fn collect_input_paths(input: &[PathBuf], extensions: &[String], recursive: bool) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for path in input
    {
        collect_path(path, extensions, recursive, &mut paths);
    }
    paths.sort();
    paths.dedup();
    paths
}

fn collect_path(path: &Path, extensions: &[String], recursive: bool, output: &mut Vec<PathBuf>) {
    if path.is_file()
    {
        if matches_extension(path, extensions)
        {
            output.push(path.to_path_buf());
        }
        return;
    }
    if !path.is_dir() || !recursive
    {
        return;
    }
    let Ok(entries) = fs::read_dir(path)
    else
    {
        return;
    };
    let mut children = entries
        .flatten()
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    children.sort();
    for child in children
    {
        if child.is_dir()
        {
            if child
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(skip_source_dir)
            {
                continue;
            }
            collect_path(&child, extensions, true, output);
        }
        else if matches_extension(&child, extensions)
        {
            output.push(child);
        }
    }
}

fn load_reversible_byte_vocab(path: &Path) -> Result<BTreeMap<String, usize>, String> {
    let input = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let value: serde_json::Value =
        serde_json::from_str(&input).map_err(|error| error.to_string())?;
    if value.get("version").and_then(serde_json::Value::as_str) != Some("byte_level_v2")
    {
        return Err("tokenizer-autotune currently requires byte_level_v2".to_string());
    }
    serde_json::from_value(
        value
            .get("vocab")
            .cloned()
            .ok_or_else(|| "tokenizer missing vocab".to_string())?,
    )
    .map_err(|error| error.to_string())
}

fn collect_calibration_cases(
    paths: &[PathBuf],
    probe_lengths: &[usize],
    cases_per_length: usize,
    vocab: &BTreeMap<String, usize>,
) -> Result<Vec<CalibrationCase>, String> {
    let mut counts = vec![0usize; probe_lengths.len()];
    let mut cases = Vec::new();

    for path in paths
    {
        if counts.iter().all(|&count| count >= cases_per_length)
        {
            break;
        }
        let bytes = match fs::read(path)
        {
            Ok(bytes) => bytes,
            Err(_) => continue,
        };
        for (probe_index, &piece_len) in probe_lengths.iter().enumerate()
        {
            if counts[probe_index] >= cases_per_length || bytes.len() < piece_len
            {
                continue;
            }
            let ids = bytes[..piece_len]
                .iter()
                .copied()
                .map(|byte| {
                    vocab
                        .get(&byte_to_unit(byte).to_string())
                        .copied()
                        .ok_or_else(|| format!("tokenizer missing byte token {byte}"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            cases.push(CalibrationCase::new(piece_len, ids));
            counts[probe_index] += 1;
        }
    }

    if let Some((index, _)) = counts.iter().enumerate().find(|(_, count)| **count == 0)
    {
        return Err(format!(
            "no corpus file was large enough for probe length {}",
            probe_lengths[index]
        ));
    }
    Ok(cases)
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
    fn probe_lengths_are_sorted_and_deduplicated() {
        assert_eq!(
            parse_probe_lengths("64,8,32,32,16").unwrap(),
            vec![8, 16, 32, 64]
        );
    }

    #[test]
    fn zero_probe_length_is_rejected() {
        assert!(parse_probe_lengths("8,0,16").is_err());
    }

    #[test]
    fn byte_unit_map_is_injective_for_all_bytes() {
        let mut chars = (0u8..=255).map(byte_to_unit).collect::<Vec<_>>();
        chars.sort_unstable();
        chars.dedup();
        assert_eq!(chars.len(), 256);
    }

    #[test]
    fn raw_report_kernel_names_are_stable() {
        assert_eq!(kernel_name(BpeKernel::Reference), "reference");
        assert_eq!(kernel_name(BpeKernel::TinyScan), "tiny_scan");
        assert_eq!(kernel_name(BpeKernel::Indexed), "indexed");
        assert_eq!(kernel_name(BpeKernel::Heap), "heap");
    }
}
