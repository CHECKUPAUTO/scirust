use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use clap::Parser;
use scirust_metrology::allan_deviation;
use scirust_sciagent::{
    BpeKernel, CalibrationMeasurement, ElasticModelSelectionReport, SelectionConfidence,
};

#[derive(Parser)]
#[command(
    name = "tokenizer-model-select",
    about = "Select the robust ElasticTokenizer kernel winner from an autotune raw report"
)]
struct Args {
    /// Raw schema-v2 report emitted by tokenizer-autotune --report.
    #[arg(short, long)]
    report: PathBuf,

    /// Optional JSON output path. Without it, the report is printed to stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Exit nonzero if any selected winner is only provisional.
    #[arg(long)]
    require_significant: bool,
}

#[derive(Clone, Copy, Debug)]
struct StabilityMetrics {
    allan_m1_nanos: f64,
    allan_m2_nanos: Option<f64>,
    allan_m4_nanos: Option<f64>,
}

fn main() {
    let args = Args::parse();
    let input = fs::read_to_string(&args.report).expect("failed to read autotune report");
    let value: serde_json::Value =
        serde_json::from_str(&input).expect("invalid autotune report JSON");
    if value
        .get("schema_version")
        .and_then(serde_json::Value::as_u64)
        != Some(2)
    {
        panic!("tokenizer-model-select requires autotune report schema_version=2");
    }

    let measurements = value
        .get("measurements")
        .and_then(serde_json::Value::as_array)
        .expect("autotune report missing measurements")
        .iter()
        .map(parse_measurement)
        .collect::<Result<Vec<_>, _>>()
        .expect("invalid autotune measurement");

    let report = ElasticModelSelectionReport::from_measurements(&measurements)
        .expect("robust ElasticTokenizer model selection failed");
    let cases_per_length = value
        .get("calibration")
        .and_then(|calibration| calibration.get("cases_per_length"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|count| usize::try_from(count).ok());
    let stability = if cases_per_length == Some(1)
    {
        stability_metrics(&measurements)
    }
    else
    {
        BTreeMap::new()
    };

    let selections = report
        .selections()
        .iter()
        .map(|selection| {
            let winner_stability = stability.get(&(
                selection.piece_len,
                kernel_order(selection.winner.kernel),
            ));
            serde_json::json!({
                "piece_len": selection.piece_len,
                "winner": kernel_name(selection.winner.kernel),
                "winner_mean_nanos": selection.winner.mean_nanos,
                "winner_std_dev_nanos": selection.winner.std_dev_nanos,
                "winner_coefficient_of_variation": selection.winner.coefficient_of_variation,
                "winner_median_nanos": selection.winner.median_nanos,
                "winner_p95_nanos": selection.winner.p95_nanos,
                "winner_q1_nanos": selection.winner.q1_nanos,
                "winner_q3_nanos": selection.winner.q3_nanos,
                "winner_raw_samples": selection.winner.raw_samples,
                "winner_clean_samples": selection.winner.clean_samples,
                "winner_dropped_outliers": selection.winner.dropped_outliers,
                "winner_allan_m1_nanos": winner_stability.map(|metrics| metrics.allan_m1_nanos),
                "winner_allan_m2_nanos": winner_stability.and_then(|metrics| metrics.allan_m2_nanos),
                "winner_allan_m4_nanos": winner_stability.and_then(|metrics| metrics.allan_m4_nanos),
                "winner_allan_m1_over_median": winner_stability.map(|metrics| metrics.allan_m1_nanos / selection.winner.median_nanos.max(1.0)),
                "runner_up": selection.runner_up.as_ref().map(|summary| kernel_name(summary.kernel)),
                "runner_up_median_nanos": selection.runner_up.as_ref().map(|summary| summary.median_nanos),
                "runner_up_coefficient_of_variation": selection.runner_up.as_ref().map(|summary| summary.coefficient_of_variation),
                "median_speedup": selection.median_speedup,
                "welch_p_value": selection.welch_p_value,
                "confidence": confidence_name(selection.confidence),
            })
        })
        .collect::<Vec<_>>();

    let output = serde_json::json!({
        "schema_version": 1,
        "source_report": args.report,
        "source_tokenizer_fingerprint": value.get("tokenizer_fingerprint"),
        "source_case_fingerprint": value.get("case_fingerprint"),
        "source_hardware": value.get("hardware"),
        "source_calibration": value.get("calibration"),
        "stability_protocol_valid": cases_per_length == Some(1),
        "rejected_semantic_measurements": report.rejected_semantic_measurements(),
        "selections": selections,
    });
    let encoded =
        serde_json::to_string_pretty(&output).expect("model-selection report serialization");

    if let Some(path) = args.output
    {
        fs::write(&path, encoded).expect("failed to write model-selection report");
        eprintln!("robust model-selection report saved to {path:?}");
    }
    else
    {
        println!("{encoded}");
    }

    if args.require_significant
        && report
            .selections()
            .iter()
            .any(|selection| selection.confidence == SelectionConfidence::Provisional)
    {
        eprintln!("ERROR: at least one ElasticTokenizer winner is statistically provisional");
        std::process::exit(2);
    }
}

fn stability_metrics(
    measurements: &[CalibrationMeasurement],
) -> BTreeMap<(usize, u8), StabilityMetrics> {
    let mut grouped = BTreeMap::<(usize, u8), Vec<f64>>::new();
    for measurement in measurements
    {
        if measurement.semantic_match
        {
            grouped
                .entry((measurement.piece_len, kernel_order(measurement.kernel)))
                .or_default()
                .push(measurement.elapsed_nanos as f64);
        }
    }
    grouped
        .into_iter()
        .filter_map(|(key, samples)| {
            allan_deviation(&samples, 1).map(|allan_m1_nanos| {
                (
                    key,
                    StabilityMetrics {
                        allan_m1_nanos,
                        allan_m2_nanos: allan_deviation(&samples, 2),
                        allan_m4_nanos: allan_deviation(&samples, 4),
                    },
                )
            })
        })
        .collect()
}

fn parse_measurement(value: &serde_json::Value) -> Result<CalibrationMeasurement, String> {
    let piece_len = value
        .get("piece_len")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| "measurement missing piece_len".to_string())?;
    let piece_len =
        usize::try_from(piece_len).map_err(|_| "piece_len exceeds usize".to_string())?;
    let kernel = value
        .get("kernel")
        .and_then(serde_json::Value::as_str)
        .and_then(parse_kernel)
        .ok_or_else(|| "measurement has unknown kernel".to_string())?;
    let elapsed_nanos = value
        .get("elapsed_nanos")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| "measurement missing elapsed_nanos".to_string())?;
    let semantic_match = value
        .get("semantic_match")
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| "measurement missing semantic_match".to_string())?;
    Ok(CalibrationMeasurement {
        piece_len,
        kernel,
        elapsed_nanos,
        semantic_match,
    })
}

const fn parse_kernel(name: &str) -> Option<BpeKernel> {
    match name.as_bytes()
    {
        b"reference" => Some(BpeKernel::Reference),
        b"tiny_scan" => Some(BpeKernel::TinyScan),
        b"indexed" => Some(BpeKernel::Indexed),
        b"heap" => Some(BpeKernel::Heap),
        _ => None,
    }
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

const fn kernel_order(kernel: BpeKernel) -> u8 {
    match kernel
    {
        BpeKernel::Reference => 0,
        BpeKernel::TinyScan => 1,
        BpeKernel::Indexed => 2,
        BpeKernel::Heap => 3,
    }
}

const fn confidence_name(confidence: SelectionConfidence) -> &'static str {
    match confidence
    {
        SelectionConfidence::Strong => "strong",
        SelectionConfidence::Significant => "significant",
        SelectionConfidence::Provisional => "provisional",
        SelectionConfidence::Uncontested => "uncontested",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_kernel_names_roundtrip() {
        for kernel in [
            BpeKernel::Reference,
            BpeKernel::TinyScan,
            BpeKernel::Indexed,
            BpeKernel::Heap,
        ]
        {
            assert_eq!(parse_kernel(kernel_name(kernel)), Some(kernel));
        }
    }

    #[test]
    fn allan_metrics_require_repeated_samples_and_are_deterministic() {
        let measurements = [10, 12, 9, 11, 10, 12, 9, 11]
            .into_iter()
            .map(|elapsed_nanos| CalibrationMeasurement {
                piece_len: 64,
                kernel: BpeKernel::Indexed,
                elapsed_nanos,
                semantic_match: true,
            })
            .collect::<Vec<_>>();
        let metrics = stability_metrics(&measurements);
        let indexed = metrics
            .get(&(64, kernel_order(BpeKernel::Indexed)))
            .unwrap();
        assert!(indexed.allan_m1_nanos > 0.0);
        assert!(indexed.allan_m2_nanos.is_some());
        assert!(indexed.allan_m4_nanos.is_some());
    }
}
