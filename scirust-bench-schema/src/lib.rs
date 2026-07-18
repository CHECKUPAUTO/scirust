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
//! `scirust-core`'s `anee_phase_c_dose_response` example. Phase D added
//! `scirust-tdi-bench`'s `tdi-holdout` (holdout metrics + paired-bootstrap
//! gains, exercising the `ci` field) and the Phase D experiment binaries.
//!
//! ## Migrating criterion targets
//!
//! Criterion owns its own timing loop, so timing rows are converted **after**
//! a run rather than emitted during it: `cargo bench`, then feed each
//! `target/criterion/<group>/<bench>/new/estimates.json` through
//! [`criterion_estimate_to_record`]. Criterion knows nothing about the
//! *data* behind a benchmark — every criterion target in this workspace pins
//! its inputs to a deterministic seeded generator (e.g. `vi_cfar_bench`'s
//! LCG) precisely so timings are comparable, and that pinned seed is what
//! the converter's mandatory `seed` argument is for.

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

/// Convert one criterion `estimates.json` (the layout criterion 0.5 writes
/// under `target/criterion/<group>/<bench>/new/estimates.json`) into a
/// [`BenchRecord`] carrying the mean wall time in nanoseconds
/// (metric `"mean_wall_time_ns"`), with criterion's confidence interval
/// attached when present.
///
/// `kernel`/`dataset`/`method` name the benchmark the way the adopting
/// harness wants it keyed; `seed` is the **data-generation** seed the bench
/// pins (criterion cannot know it — see the crate docs, "Migrating
/// criterion targets").
pub fn criterion_estimate_to_record(
    estimates_json: &str,
    kernel: impl Into<String>,
    dataset: impl Into<String>,
    method: impl Into<String>,
    seed: u64,
) -> Result<BenchRecord, String> {
    let v: serde_json::Value =
        serde_json::from_str(estimates_json).map_err(|e| format!("estimates.json: {e}"))?;
    let mean = v.get("mean").ok_or("estimates.json has no `mean` block")?;
    let point = mean
        .get("point_estimate")
        .and_then(serde_json::Value::as_f64)
        .ok_or("mean.point_estimate missing or not a number")?;
    let mut record = BenchRecord::new(kernel, dataset, method, seed, "mean_wall_time_ns", point);
    if let Some(interval) = mean.get("confidence_interval")
        && let (Some(lo), Some(hi), Some(level)) = (
            interval
                .get("lower_bound")
                .and_then(serde_json::Value::as_f64),
            interval
                .get("upper_bound")
                .and_then(serde_json::Value::as_f64),
            interval
                .get("confidence_level")
                .and_then(serde_json::Value::as_f64),
        )
    {
        record = record.with_ci(ConfidenceInterval { lo, hi, level });
    }
    Ok(record)
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

    /// Fixture mirroring the estimates.json layout criterion 0.5 writes
    /// (extra blocks present, `slope` null — both must be tolerated).
    const CRITERION_ESTIMATES_FIXTURE: &str = r#"{
        "mean": {
            "confidence_interval": {
                "confidence_level": 0.95,
                "lower_bound": 118.21,
                "upper_bound": 121.93
            },
            "point_estimate": 120.03,
            "standard_error": 0.94
        },
        "median": {
            "confidence_interval": {
                "confidence_level": 0.95,
                "lower_bound": 117.4,
                "upper_bound": 119.9
            },
            "point_estimate": 118.6,
            "standard_error": 0.6
        },
        "median_abs_dev": null,
        "slope": null,
        "std_dev": null
    }"#;

    #[test]
    fn criterion_estimates_convert_with_mean_and_ci() {
        let r = criterion_estimate_to_record(
            CRITERION_ESTIMATES_FIXTURE,
            "vi_cfar/classical",
            "homogeneous/window=32",
            "CfarDetector::detect",
            0x5EED,
        )
        .expect("fixture must convert");
        assert_eq!(r.metric, "mean_wall_time_ns");
        assert_eq!(r.value, 120.03);
        assert_eq!(r.seed, 0x5EED);
        let ci = r.ci.expect("criterion CI must carry over");
        assert_eq!((ci.lo, ci.hi, ci.level), (118.21, 121.93, 0.95));

        assert!(criterion_estimate_to_record("{}", "k", "d", "m", 0).is_err());
        assert!(criterion_estimate_to_record("not json", "k", "d", "m", 0).is_err());
    }
}
