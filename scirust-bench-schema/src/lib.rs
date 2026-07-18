//! **Shared benchmark-result schema — the CANR §9 record, enforced as a type.**
//!
//! [CANR] (`docs/research/CANR_CERTIFIED_ADAPTIVE_REPRESENTATIONS_2026-07-16.md`
//! §9) proposed one machine-readable row shape for every benchmark in the
//! workspace: `{kernel, dataset, method, seed, metric, value, ci, cert}`.
//! [ANEE §3] (`docs/research/ANEE_ADAPTIVE_NUMERICAL_EXECUTION_ENGINE_2026-07-17.md`)
//! then verified, one phase later, that the proposal was **never adopted**:
//! every real harness (tdi-bench, `vst_bench`'s ad hoc `BenchRow`, ~16
//! criterion targets) rolled its own incompatible output shape. The program's
//! closing synthesis (`ANEE_PROGRAM_SYNTHESIS_2026-07-18.md` §7) therefore
//! recommends re-attempting the schema **as a compile-time-enforced crate
//! rather than a design document** — this crate is that re-attempt.
//!
//! ## How the enforcement works
//!
//! [`BenchRecord`]'s six mandatory fields (`kernel`, `dataset`, `method`,
//! `seed`, `metric`, `value`) are **constructor arguments** of
//! [`BenchRecord::new`] — omitting one is a compile error, not a documentation
//! lapse. The two optional fields ([`ConfidenceInterval`], [`Certificate`])
//! attach via [`BenchRecord::with_ci`] / [`BenchRecord::with_cert`] and are
//! omitted from the JSON when absent, so minimal rows stay minimal.
//!
//! `seed` is mandatory **on purpose**: a benchmark row whose randomness
//! cannot be reproduced is not a reproducible artifact, and making the field
//! required is precisely the forcing function the design-document version of
//! this schema lacked. Harnesses that "don't have a seed handy" are being
//! told something by the compiler.
//!
//! ## Interchange format
//!
//! JSON Lines (one [`BenchRecord`] per line, stable field order — the struct
//! declaration order below), via [`to_jsonl`] / [`write_jsonl`] /
//! [`parse_jsonl`]. Adopters at introduction time:
//! `scirust-signal::denoise::vst_bench` (`BenchTable::to_bench_records`) and
//! `scirust-core`'s `anee_phase_c_dose_response` example.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;

/// A two-sided confidence interval for [`BenchRecord::value`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Lower bound.
    pub lo: f64,
    /// Upper bound.
    pub hi: f64,
    /// Confidence level in (0, 1), e.g. `0.95`.
    pub level: f64,
}

/// A machine-checkable certificate attached to a measurement — e.g. the
/// `κ_rt`-based round-trip bound of `scirust-core::certified_numerics`
/// ([CANR §3.2]), or a determinism-level declaration ([CANR §6.1]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Certificate {
    /// What the certificate asserts (human-readable, stable).
    pub description: String,
    /// Certified bound in ulps, when the certificate is an error bound.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub bound_ulps: Option<f64>,
    /// Declared determinism level (`"D0"`..`"D3"`, [CANR §6.1]), when the
    /// certificate is a reproducibility claim.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub determinism: Option<String>,
}

/// One benchmark measurement — the CANR §9 row
/// `{kernel, dataset, method, seed, metric, value, ci, cert}`.
///
/// Field order here is the serialized field order; do not reorder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchRecord {
    /// What computation was benchmarked (e.g. `"vst_denoise/wiener_global"`).
    pub kernel: String,
    /// Which data/workload family (e.g. `"poisson_like"`, `"wide-range/L=64"`).
    pub dataset: String,
    /// Which candidate/method produced this value (e.g. `"anscombe"`,
    /// `"power+PairwiseF32"`, `"direct"`).
    pub method: String,
    /// The seed that generated the randomness behind this measurement.
    /// Mandatory — see the crate docs for why.
    pub seed: u64,
    /// What was measured (e.g. `"snr_db"`, `"held_out_relative_error"`).
    pub metric: String,
    /// The measured value.
    pub value: f64,
    /// Optional confidence interval for `value`.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub ci: Option<ConfidenceInterval>,
    /// Optional machine-checkable certificate backing the measurement.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cert: Option<Certificate>,
}

impl BenchRecord {
    /// Build a record from the six **mandatory** fields. There is no way to
    /// construct a [`BenchRecord`] without all six — that compile-time
    /// requirement is this crate's entire reason to exist.
    pub fn new(
        kernel: impl Into<String>,
        dataset: impl Into<String>,
        method: impl Into<String>,
        seed: u64,
        metric: impl Into<String>,
        value: f64,
    ) -> Self {
        Self {
            kernel: kernel.into(),
            dataset: dataset.into(),
            method: method.into(),
            seed,
            metric: metric.into(),
            value,
            ci: None,
            cert: None,
        }
    }

    /// Attach a confidence interval.
    pub fn with_ci(mut self, ci: ConfidenceInterval) -> Self {
        self.ci = Some(ci);
        self
    }

    /// Attach a certificate.
    pub fn with_cert(mut self, cert: Certificate) -> Self {
        self.cert = Some(cert);
        self
    }

    /// One JSON object, no trailing newline.
    pub fn to_json_row(&self) -> String {
        serde_json::to_string(self).expect("BenchRecord serialization is infallible")
    }
}

/// Serialize records as JSON Lines (one object per line, trailing newline).
pub fn to_jsonl(records: &[BenchRecord]) -> String {
    let mut out = String::new();
    for r in records
    {
        out.push_str(&r.to_json_row());
        out.push('\n');
    }
    out
}

/// Write records as a JSON Lines file.
pub fn write_jsonl(path: impl AsRef<Path>, records: &[BenchRecord]) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    f.write_all(to_jsonl(records).as_bytes())
}

/// Parse JSON Lines produced by [`to_jsonl`] (blank lines ignored).
pub fn parse_jsonl(s: &str) -> Result<Vec<BenchRecord>, serde_json::Error> {
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> BenchRecord {
        BenchRecord::new(
            "vst_denoise/wiener_global",
            "poisson_like",
            "anscombe",
            2,
            "snr_db",
            21.37,
        )
    }

    #[test]
    fn mandatory_fields_are_constructor_arguments() {
        // The enforcement itself is compile-time (BenchRecord::new's arity);
        // this test documents the runtime consequence: every mandatory field
        // is populated, optional ones absent.
        let r = sample();
        assert_eq!(r.kernel, "vst_denoise/wiener_global");
        assert_eq!(r.seed, 2);
        assert!(r.ci.is_none() && r.cert.is_none());
    }

    #[test]
    fn optional_fields_are_omitted_from_json_when_absent() {
        let row = sample().to_json_row();
        assert!(
            !row.contains("\"ci\""),
            "absent ci must not serialize: {row}"
        );
        assert!(
            !row.contains("\"cert\""),
            "absent cert must not serialize: {row}"
        );
        // Stable field order = declaration order.
        let k = row.find("\"kernel\"").unwrap();
        let d = row.find("\"dataset\"").unwrap();
        let s = row.find("\"seed\"").unwrap();
        assert!(
            k < d && d < s,
            "field order must be declaration order: {row}"
        );
    }

    #[test]
    fn jsonl_round_trips_with_ci_and_cert() {
        let records = vec![
            sample(),
            sample()
                .with_ci(ConfidenceInterval {
                    lo: 20.9,
                    hi: 21.8,
                    level: 0.95,
                })
                .with_cert(Certificate {
                    description: "kappa_rt round-trip bound (CANR §3.2)".into(),
                    bound_ulps: Some(12.0),
                    determinism: Some("D0".into()),
                }),
        ];
        let text = to_jsonl(&records);
        assert_eq!(text.lines().count(), 2);
        let back = parse_jsonl(&text).expect("round trip must parse");
        assert_eq!(back, records);
    }

    #[test]
    fn parse_tolerates_blank_lines_and_rejects_garbage() {
        let text = format!("{}\n\n{}\n", sample().to_json_row(), sample().to_json_row());
        assert_eq!(parse_jsonl(&text).unwrap().len(), 2);
        assert!(parse_jsonl("not json\n").is_err());
    }
}
