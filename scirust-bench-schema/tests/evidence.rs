use scirust_bench_schema::{
    BenchRecord, EvidenceDisposition, ScientificEvidence, ScientificEvidenceKind, parse_jsonl,
    to_jsonl,
};

fn classified_negative_record() -> BenchRecord {
    let evidence = ScientificEvidence::new(
        ScientificEvidenceKind::NumericalApproximation,
        EvidenceDisposition::Rejects,
        "bounded history improves endpoint error over complete history",
    )
    .unwrap();

    BenchRecord::new(
        "history_retention",
        "nonlocal-reference-case",
        "window-32",
        0,
        "endpoint_relative_error_delta",
        0.03125,
    )
    .with_evidence(evidence)
}

#[test]
fn negative_result_round_trips_as_evidence() {
    let record = classified_negative_record();
    assert!(record.evidence.as_ref().unwrap().is_negative_result());

    let jsonl = to_jsonl(std::slice::from_ref(&record));
    assert!(jsonl.contains("\"kind\":\"numerical_approximation\""));
    assert!(jsonl.contains("\"disposition\":\"rejects\""));
    assert_eq!(parse_jsonl(&jsonl).unwrap(), vec![record]);
}

#[test]
fn legacy_row_without_evidence_still_parses() {
    let legacy =
        r#"{"kernel":"k","dataset":"d","method":"m","seed":7,"metric":"snr_db","value":1.5}"#;
    let parsed = parse_jsonl(legacy).expect("legacy row must remain valid");
    assert_eq!(parsed.len(), 1);
    assert!(parsed[0].evidence.is_none());
}

#[test]
fn absent_evidence_does_not_change_minimal_json_shape() {
    let row = BenchRecord::new("k", "d", "m", 1, "metric", 2.0).to_json_row();
    assert!(!row.contains("\"evidence\""));
}
