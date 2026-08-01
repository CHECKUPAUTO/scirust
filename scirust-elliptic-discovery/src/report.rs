//! Stable, replay-oriented experiment reports.

use core::fmt::Write;

use crate::canonical::{hex, sha256};
use crate::{Corpus, Counterexample};

/// Deterministic report containing coverage and an optional first counterexample.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExperimentReport {
    manifest_fingerprint: [u8; 32],
    corpus_fingerprint: [u8; 32],
    corpus_name: &'static str,
    seed: u64,
    relation_id: String,
    curve_count: u64,
    point_count: u64,
    evaluated_tuples: u64,
    counterexample: Option<Counterexample>,
}

impl ExperimentReport {
    /// Captures an immutable result from a local corpus run.
    pub fn new(
        corpus: &Corpus,
        relation_id: impl Into<String>,
        evaluated_tuples: u64,
        counterexample: Option<Counterexample>,
    ) -> Self {
        let research_case = corpus.manifest().research_case();
        Self {
            manifest_fingerprint: corpus.manifest().fingerprint(),
            corpus_fingerprint: corpus.fingerprint(),
            corpus_name: research_case.corpus().name(),
            seed: research_case.seed(),
            relation_id: relation_id.into(),
            curve_count: u64::try_from(corpus.curves().len()).expect("curve count fits in u64"),
            point_count: corpus.total_points(),
            evaluated_tuples,
            counterexample,
        }
    }

    /// Stable UTF-8 JSON bytes with an explicit field order.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut output = String::new();
        writeln!(output, "{{").expect("writing to String cannot fail");
        writeln!(
            output,
            "  \"schema\": \"scirust-elliptic-discovery-report-v1\","
        )
        .expect("writing to String cannot fail");
        writeln!(
            output,
            "  \"manifest_sha256\": \"{}\",",
            hex(&self.manifest_fingerprint)
        )
        .expect("writing to String cannot fail");
        writeln!(
            output,
            "  \"corpus_sha256\": \"{}\",",
            hex(&self.corpus_fingerprint)
        )
        .expect("writing to String cannot fail");
        writeln!(output, "  \"corpus\": \"{}\",", self.corpus_name)
            .expect("writing to String cannot fail");
        writeln!(output, "  \"seed\": {},", self.seed).expect("writing to String cannot fail");
        writeln!(
            output,
            "  \"relation_id\": \"{}\",",
            escape_json(&self.relation_id)
        )
        .expect("writing to String cannot fail");
        writeln!(output, "  \"curve_count\": {},", self.curve_count)
            .expect("writing to String cannot fail");
        writeln!(output, "  \"point_count\": {},", self.point_count)
            .expect("writing to String cannot fail");
        writeln!(output, "  \"evaluated_tuples\": {},", self.evaluated_tuples)
            .expect("writing to String cannot fail");
        match &self.counterexample
        {
            Some(counterexample) =>
            {
                let (prime, a, b) = counterexample.curve_key();
                let point = match counterexample.point().affine_coordinates()
                {
                    Some((x, y)) => format!("[{}, {}]", x, y),
                    None => "null".to_string(),
                };
                writeln!(output, "  \"counterexample\": {{")
                    .expect("writing to String cannot fail");
                writeln!(output, "    \"prime\": {prime},").expect("writing to String cannot fail");
                writeln!(output, "    \"a\": {a},").expect("writing to String cannot fail");
                writeln!(output, "    \"b\": {b},").expect("writing to String cannot fail");
                writeln!(
                    output,
                    "    \"point_index\": {},",
                    counterexample.point_index()
                )
                .expect("writing to String cannot fail");
                writeln!(output, "    \"point\": {point}").expect("writing to String cannot fail");
                writeln!(output, "  }}").expect("writing to String cannot fail");
            },
            None =>
            {
                writeln!(output, "  \"counterexample\": null")
                    .expect("writing to String cannot fail");
            },
        }
        writeln!(output, "}}").expect("writing to String cannot fail");
        output.into_bytes()
    }

    /// SHA-256 integrity fingerprint of the complete report bytes.
    pub fn fingerprint(&self) -> [u8; 32] {
        sha256(&self.canonical_bytes())
    }
}

fn escape_json(input: &str) -> String {
    let mut output = String::new();
    for character in input.chars()
    {
        match character
        {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            value if value.is_control() =>
            {
                write!(output, "\\u{:04x}", u32::from(value))
                    .expect("writing to String cannot fail");
            },
            value => output.push(value),
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CorpusKind, ExperimentManifest, LocalResearchCase, first_point_counterexample};

    fn corpus() -> Corpus {
        Corpus::generate(ExperimentManifest::new(
            LocalResearchCase::new(17, CorpusKind::IndependentHoldout, 1, 100)
                .expect("valid local case"),
        ))
    }

    #[test]
    fn repeated_reports_are_byte_identical() {
        let corpus = corpus();
        let counterexample = first_point_counterexample(&corpus, "not-infinity", |_, point| {
            point.is_infinity()
        });
        let left = ExperimentReport::new(&corpus, "not-infinity", 2, counterexample.clone());
        let right = ExperimentReport::new(&corpus, "not-infinity", 2, counterexample);
        assert_eq!(left.canonical_bytes(), right.canonical_bytes());
        assert_eq!(left.fingerprint(), right.fingerprint());
    }

    #[test]
    fn first_counterexample_is_stable() {
        let corpus = corpus();
        let counterexample = first_point_counterexample(&corpus, "false", |_, _| false)
            .expect("false relation must be refuted");
        assert_eq!(counterexample.point_index(), 0);
        assert!(counterexample.point().is_infinity());
    }
}
