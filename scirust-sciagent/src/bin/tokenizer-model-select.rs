use std::fs;
use std::path::PathBuf;

use clap::Parser;
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

    let selections = report
        .selections()
        .iter()
        .map(|selection| {
            serde_json::json!({
                "piece_len": selection.piece_len,
                "winner": kernel_name(selection.winner.kernel),
                "winner_median_nanos": selection.winner.median_nanos,
                "winner_p95_nanos": selection.winner.p95_nanos,
                "winner_q1_nanos": selection.winner.q1_nanos,
                "winner_q3_nanos": selection.winner.q3_nanos,
                "winner_raw_samples": selection.winner.raw_samples,
                "winner_clean_samples": selection.winner.clean_samples,
                "winner_dropped_outliers": selection.winner.dropped_outliers,
                "runner_up": selection.runner_up.as_ref().map(|summary| kernel_name(summary.kernel)),
                "runner_up_median_nanos": selection.runner_up.as_ref().map(|summary| summary.median_nanos),
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
}
