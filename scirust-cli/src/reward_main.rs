//! Versioned JSONL process contract for deterministic SciRust reward/verifier calls.
//!
//! This binary is intentionally separate from the interactive `scirust` CLI so
//! post-training systems can invoke a narrow, stable machine-readable surface.

use std::io::{self, BufRead, Write};

use scirust_symbolic::{parse, prove_equal};
use serde_json::{json, Value};

const SCHEMA_VERSION: u64 = 1;
const KIND: &str = "symbolic_equivalence";
const MAX_LINE_BYTES: usize = 1024 * 1024;
const MAX_RECORDS: usize = 4096;

fn required_string<'a>(object: &'a serde_json::Map<String, Value>, key: &str) -> Result<&'a str, String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("field `{key}` must be a non-empty string"))
}

fn evaluate_record(value: Value) -> Result<Value, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "record must be a JSON object".to_owned())?;
    if object.get("schema_version").and_then(Value::as_u64) != Some(SCHEMA_VERSION) {
        return Err(format!("unsupported schema_version; expected {SCHEMA_VERSION}"));
    }
    if object.get("kind").and_then(Value::as_str) != Some(KIND) {
        return Err(format!("unsupported kind; expected `{KIND}`"));
    }
    let candidate = required_string(object, "candidate")?;
    let reference = required_string(object, "reference")?;
    let id = object.get("id").cloned().unwrap_or(Value::Null);
    if !id.is_null() && !id.is_string() && !id.is_number() {
        return Err("field `id`, when present, must be a string or number".to_owned());
    }

    let reference_expr = parse(reference)
        .map_err(|error| format!("trusted reference expression failed to parse: {error}"))?;
    let candidate_expr = match parse(candidate) {
        Ok(expr) => expr,
        Err(error) => {
            return Ok(json!({
                "schema_version": SCHEMA_VERSION,
                "kind": KIND,
                "id": id,
                "score": 0.0,
                "proven_equal": false,
                "verdict": "candidate_parse_error",
                "detail": error.to_string(),
            }));
        }
    };

    let equal = prove_equal(&candidate_expr, &reference_expr);
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "id": id,
        "score": if equal { 1.0 } else { 0.0 },
        "proven_equal": equal,
        "verdict": if equal { "proven_equal" } else { "not_proven" },
        "detail": if equal {
            "SciRust symbolic prover established equivalence"
        } else {
            "equivalence was not proven; the prover is sound but incomplete"
        },
    }))
}

fn run<R: BufRead, W: Write>(mut input: R, mut output: W) -> Result<(), String> {
    let mut line = String::new();
    let mut records = 0usize;
    loop {
        line.clear();
        let bytes = input
            .read_line(&mut line)
            .map_err(|error| format!("reading JSONL input: {error}"))?;
        if bytes == 0 {
            break;
        }
        if bytes > MAX_LINE_BYTES {
            return Err(format!("input line exceeds {MAX_LINE_BYTES} bytes"));
        }
        if line.trim().is_empty() {
            continue;
        }
        records += 1;
        if records > MAX_RECORDS {
            return Err(format!("batch exceeds {MAX_RECORDS} records"));
        }
        let value: Value = serde_json::from_str(&line)
            .map_err(|error| format!("invalid JSONL record {records}: {error}"))?;
        let result = evaluate_record(value)
            .map_err(|error| format!("invalid reward record {records}: {error}"))?;
        serde_json::to_writer(&mut output, &result)
            .map_err(|error| format!("serializing reward record {records}: {error}"))?;
        output
            .write_all(b"\n")
            .map_err(|error| format!("writing reward record {records}: {error}"))?;
    }
    output
        .flush()
        .map_err(|error| format!("flushing reward output: {error}"))?;
    Ok(())
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    if let Err(error) = run(stdin.lock(), stdout.lock()) {
        eprintln!("scirust-reward: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn proven_equivalence_scores_one() {
        let result = evaluate_record(json!({
            "schema_version": 1,
            "kind": "symbolic_equivalence",
            "id": "case-1",
            "candidate": "x + x",
            "reference": "2*x"
        }))
        .expect("valid record");
        assert_eq!(result["score"], 1.0);
        assert_eq!(result["verdict"], "proven_equal");
        assert_eq!(result["id"], "case-1");
    }

    #[test]
    fn non_proof_is_zero_without_claiming_inequality() {
        let result = evaluate_record(json!({
            "schema_version": 1,
            "kind": "symbolic_equivalence",
            "candidate": "x + 1",
            "reference": "x + 2"
        }))
        .expect("valid record");
        assert_eq!(result["score"], 0.0);
        assert_eq!(result["verdict"], "not_proven");
        assert_eq!(result["proven_equal"], false);
    }

    #[test]
    fn malformed_candidate_is_a_deterministic_zero_reward() {
        let result = evaluate_record(json!({
            "schema_version": 1,
            "kind": "symbolic_equivalence",
            "candidate": "x + )",
            "reference": "x"
        }))
        .expect("candidate failure is a scored result");
        assert_eq!(result["score"], 0.0);
        assert_eq!(result["verdict"], "candidate_parse_error");
    }

    #[test]
    fn malformed_reference_fails_closed() {
        let error = evaluate_record(json!({
            "schema_version": 1,
            "kind": "symbolic_equivalence",
            "candidate": "x",
            "reference": "x + )"
        }))
        .expect_err("trusted reference must parse");
        assert!(error.contains("trusted reference"));
    }

    #[test]
    fn jsonl_batch_preserves_record_order() {
        let input = concat!(
            "{\"schema_version\":1,\"kind\":\"symbolic_equivalence\",\"id\":1,\"candidate\":\"x+x\",\"reference\":\"2*x\"}\n",
            "{\"schema_version\":1,\"kind\":\"symbolic_equivalence\",\"id\":2,\"candidate\":\"x\",\"reference\":\"x+1\"}\n"
        );
        let mut output = Vec::new();
        run(Cursor::new(input.as_bytes()), &mut output).expect("batch");
        let rows: Vec<Value> = String::from_utf8(output)
            .expect("utf8")
            .lines()
            .map(|line| serde_json::from_str(line).expect("json"))
            .collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["id"], 1);
        assert_eq!(rows[0]["score"], 1.0);
        assert_eq!(rows[1]["id"], 2);
        assert_eq!(rows[1]["score"], 0.0);
    }

    #[test]
    fn unknown_schema_and_kind_fail_closed() {
        for record in [
            json!({"schema_version": 2, "kind": KIND, "candidate": "x", "reference": "x"}),
            json!({"schema_version": 1, "kind": "other", "candidate": "x", "reference": "x"}),
        ] {
            assert!(evaluate_record(record).is_err());
        }
    }
}
