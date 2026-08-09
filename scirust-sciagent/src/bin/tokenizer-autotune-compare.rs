use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use clap::Parser;

const AUTOTUNE_REPORT_SCHEMA_V1: u32 = 1;

type MeasurementKey = (usize, String);

#[derive(Parser)]
#[command(
    name = "tokenizer-autotune-compare",
    about = "Compare two raw ElasticTokenizer autotune reports on identical semantics and hardware"
)]
struct Args {
    /// Baseline raw report produced by `tokenizer-autotune --report`.
    #[arg(long)]
    baseline: PathBuf,

    /// Candidate raw report produced on the same tokenizer and hardware.
    #[arg(long)]
    candidate: PathBuf,
}

#[derive(Debug)]
struct RawReport {
    tokenizer_fingerprint: String,
    hardware_fingerprint: String,
    measurements: BTreeMap<MeasurementKey, Vec<u64>>,
    disqualified: BTreeSet<MeasurementKey>,
}

fn main() {
    let args = Args::parse();
    let baseline = load_report(&args.baseline).expect("failed to load baseline autotune report");
    let candidate = load_report(&args.candidate).expect("failed to load candidate autotune report");
    verify_comparable(&baseline, &candidate).expect("autotune reports are not comparable");

    println!("piece_len,kernel,baseline_median_ns,candidate_median_ns,speedup_x,status");
    for key in baseline
        .measurements
        .keys()
        .chain(candidate.measurements.keys())
        .cloned()
        .collect::<BTreeSet<_>>()
    {
        let status = if baseline.disqualified.contains(&key)
        {
            "baseline-semantic-mismatch"
        }
        else if candidate.disqualified.contains(&key)
        {
            "candidate-semantic-mismatch"
        }
        else
        {
            "ok"
        };

        let baseline_median = baseline
            .measurements
            .get(&key)
            .filter(|_| !baseline.disqualified.contains(&key))
            .and_then(|samples| median(samples));
        let candidate_median = candidate
            .measurements
            .get(&key)
            .filter(|_| !candidate.disqualified.contains(&key))
            .and_then(|samples| median(samples));

        let speedup = match (baseline_median, candidate_median)
        {
            (Some(base), Some(candidate)) if candidate != 0 => {
                format!("{:.6}", base as f64 / candidate as f64)
            },
            _ => String::new(),
        };

        println!(
            "{},{},{},{},{},{}",
            key.0,
            key.1,
            baseline_median.map_or_else(String::new, |value| value.to_string()),
            candidate_median.map_or_else(String::new, |value| value.to_string()),
            speedup,
            status
        );
    }
}

fn load_report(path: &Path) -> Result<RawReport, String> {
    let input = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let value: serde_json::Value = serde_json::from_str(&input).map_err(|error| error.to_string())?;

    let schema = required_u64(&value, "schema_version")?;
    if schema != u64::from(AUTOTUNE_REPORT_SCHEMA_V1)
    {
        return Err(format!("unsupported autotune report schema {schema}"));
    }

    let tokenizer_fingerprint = required_str(&value, "tokenizer_fingerprint")?.to_string();
    let hardware = value
        .get("hardware")
        .ok_or_else(|| "autotune report missing `hardware`".to_string())?;
    let hardware_fingerprint = required_str(hardware, "fingerprint")?.to_string();
    let raw_measurements = value
        .get("measurements")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "autotune report missing `measurements` array".to_string())?;

    let mut measurements: BTreeMap<MeasurementKey, Vec<u64>> = BTreeMap::new();
    let mut disqualified = BTreeSet::new();
    for measurement in raw_measurements
    {
        let piece_len = usize::try_from(required_u64(measurement, "piece_len")?)
            .map_err(|_| "piece_len exceeds this platform".to_string())?;
        let kernel = required_str(measurement, "kernel")?.to_string();
        let elapsed_nanos = required_u64(measurement, "elapsed_nanos")?;
        let semantic_match = required_bool(measurement, "semantic_match")?;
        let key = (piece_len, kernel);
        if semantic_match
        {
            measurements.entry(key).or_default().push(elapsed_nanos);
        }
        else
        {
            disqualified.insert(key);
        }
    }

    Ok(RawReport {
        tokenizer_fingerprint,
        hardware_fingerprint,
        measurements,
        disqualified,
    })
}

fn verify_comparable(baseline: &RawReport, candidate: &RawReport) -> Result<(), String> {
    if baseline.tokenizer_fingerprint != candidate.tokenizer_fingerprint
    {
        return Err("tokenizer fingerprints differ".to_string());
    }
    if baseline.hardware_fingerprint != candidate.hardware_fingerprint
    {
        return Err("hardware fingerprints differ".to_string());
    }
    Ok(())
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

fn required_str<'a>(value: &'a serde_json::Value, field: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| format!("autotune report missing string `{field}`"))
}

fn required_u64(value: &serde_json::Value, field: &str) -> Result<u64, String> {
    value
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| format!("autotune report missing integer `{field}`"))
}

fn required_bool(value: &serde_json::Value, field: &str) -> Result<bool, String> {
    value
        .get(field)
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| format!("autotune report missing boolean `{field}`"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(tokenizer: &str, hardware: &str, elapsed: &[u64]) -> String {
        let measurements = elapsed
            .iter()
            .map(|&elapsed_nanos| {
                serde_json::json!({
                    "piece_len": 64,
                    "kernel": "indexed",
                    "elapsed_nanos": elapsed_nanos,
                    "semantic_match": true,
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "schema_version": 1,
            "tokenizer_fingerprint": tokenizer,
            "hardware": { "fingerprint": hardware },
            "measurements": measurements,
        })
        .to_string()
    }

    #[test]
    fn integer_median_is_overflow_safe() {
        assert_eq!(median(&[u64::MAX - 1, u64::MAX]), Some(u64::MAX - 1));
        assert_eq!(median(&[3, 1, 2]), Some(2));
    }

    #[test]
    fn comparability_requires_same_tokenizer_and_hardware() {
        let path_a = std::env::temp_dir().join("scirust_autotune_compare_a.json");
        let path_b = std::env::temp_dir().join("scirust_autotune_compare_b.json");
        fs::write(&path_a, report("tok-a", "hw-a", &[10, 20, 30])).unwrap();
        fs::write(&path_b, report("tok-b", "hw-a", &[5, 10, 15])).unwrap();
        let a = load_report(&path_a).unwrap();
        let b = load_report(&path_b).unwrap();
        assert!(verify_comparable(&a, &b).is_err());
        let _ = fs::remove_file(path_a);
        let _ = fs::remove_file(path_b);
    }

    #[test]
    fn semantic_mismatch_disqualifies_kernel_length_pair() {
        let path = std::env::temp_dir().join("scirust_autotune_compare_mismatch.json");
        let input = serde_json::json!({
            "schema_version": 1,
            "tokenizer_fingerprint": "tok",
            "hardware": { "fingerprint": "hw" },
            "measurements": [
                {"piece_len": 64, "kernel": "heap", "elapsed_nanos": 10, "semantic_match": true},
                {"piece_len": 64, "kernel": "heap", "elapsed_nanos": 1, "semantic_match": false}
            ],
        });
        fs::write(&path, serde_json::to_string(&input).unwrap()).unwrap();
        let loaded = load_report(&path).unwrap();
        assert!(loaded.disqualified.contains(&(64, "heap".to_string())));
        let _ = fs::remove_file(path);
    }
}
