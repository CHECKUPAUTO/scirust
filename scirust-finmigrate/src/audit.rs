//! Phase 3 — consolidated auditability.
//!
//! Makes the harness's tamper-evidence and audit-trail principles *executable*:
//! recomputes each committed Golden Baseline's SHA-256 and checks it against the
//! recorded digest, then renders a single consolidated audit report across every
//! migrated unit. Driven by the `finaudit` binary; unit-tested here.
//!
//! The baselines and their `.sha256` sidecars are embedded at compile time, so
//! the tool is self-contained and the check is meaningful: editing a committed
//! baseline without regenerating its digest makes [`audit_units`] fail.

use sha2::{Digest, Sha256};

/// One unit's audit status: baseline identity, tamper check, and gap count.
#[derive(Debug, Clone)]
pub struct UnitAudit {
    /// COBOL program / unit name.
    pub unit: &'static str,
    /// Committed baseline filename (under `tests/sandbox/`).
    pub baseline: &'static str,
    /// SHA-256 recorded in the committed `.sha256` sidecar.
    pub digest_expected: String,
    /// SHA-256 recomputed from the embedded baseline bytes.
    pub digest_actual: String,
    /// Number of data rows in the baseline (excluding the header).
    pub rows: usize,
    /// Documented hidden-dependency gaps for this unit (audit_report.md).
    pub gaps: &'static [&'static str],
}

impl UnitAudit {
    /// True when the recomputed digest matches the committed one (no tamper).
    pub fn digest_ok(&self) -> bool {
        self.digest_expected == self.digest_actual
    }
}

struct Embedded {
    unit: &'static str,
    baseline: &'static str,
    csv: &'static str,
    sha256_sidecar: &'static str,
    gaps: &'static [&'static str],
}

const UNITS: &[Embedded] = &[
    Embedded {
        unit: "INTACCR",
        baseline: "golden_baseline.csv",
        csv: include_str!("../tests/sandbox/golden_baseline.csv"),
        sha256_sidecar: include_str!("../tests/sandbox/golden_baseline.sha256"),
        gaps: &[
            "Gap-1 packed-decimal-is-decimal",
            "Gap-2 implied-scale-on-store",
            "Gap-3 NEAREST-AWAY-FROM-ZERO",
            "Gap-4 one-rounding-event",
            "Gap-5 size-error-not-silent-truncation",
            "Gap-6 intermediate-precision (GATE)",
        ],
    },
    Embedded {
        unit: "AMORTSCH",
        baseline: "amort_baseline.csv",
        csv: include_str!("../tests/sandbox/amort_baseline.csv"),
        sha256_sidecar: include_str!("../tests/sandbox/amort_baseline.sha256"),
        gaps: &[
            "Gap-A accumulated-rounding-drift",
            "Gap-B final-payment-reconciliation",
            "Gap-C negative-amortization/row-count",
            "Gap-D size-error",
        ],
    },
    Embedded {
        unit: "PAYCALC",
        baseline: "pay_baseline.csv",
        csv: include_str!("../tests/sandbox/pay_baseline.csv"),
        sha256_sidecar: include_str!("../tests/sandbox/pay_baseline.sha256"),
        gaps: &[
            "Gap-E exponentiation-float-dispatch (rewrite)",
            "Gap-F multiply-chain-intermediate-precision",
            "Gap-G zero-rate-divide-by-zero",
        ],
    },
    Embedded {
        unit: "DAYCOUNT",
        baseline: "day_baseline.csv",
        csv: include_str!("../tests/sandbox/day_baseline.csv"),
        sha256_sidecar: include_str!("../tests/sandbox/day_baseline.sha256"),
        gaps: &[
            "Gap-H NASD-vs-Excel-ambiguity",
            "Gap-I rule-ordering",
            "Gap-J leap-year-definition",
            "Gap-K one-rounding-event",
        ],
    },
];

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// The digest recorded in a `<hex>  <filename>` sidecar (first whitespace token).
fn parse_sidecar_digest(sidecar: &str) -> String {
    sidecar.split_whitespace().next().unwrap_or("").to_string()
}

fn data_rows(csv: &str) -> usize {
    // Rows minus the header, ignoring trailing blank lines.
    csv.lines()
        .filter(|l| !l.trim().is_empty())
        .count()
        .saturating_sub(1)
}

/// Audit every migrated unit: recompute and verify its baseline digest.
pub fn audit_units() -> Vec<UnitAudit> {
    UNITS
        .iter()
        .map(|e| UnitAudit {
            unit: e.unit,
            baseline: e.baseline,
            digest_expected: parse_sidecar_digest(e.sha256_sidecar),
            digest_actual: sha256_hex(e.csv.as_bytes()),
            rows: data_rows(e.csv),
            gaps: e.gaps,
        })
        .collect()
}

/// True when every unit's baseline digest verifies.
pub fn all_ok(units: &[UnitAudit]) -> bool {
    units.iter().all(UnitAudit::digest_ok)
}

/// Render a consolidated, human-readable audit report.
pub fn render_report(units: &[UnitAudit]) -> String {
    let mut out = String::new();
    out.push_str("scirust-finmigrate — consolidated audit report\n");
    out.push_str("==============================================\n\n");
    out.push_str("BASELINE TAMPER-EVIDENCE (SHA-256 of committed Golden Baselines)\n");

    let mut total_gaps = 0usize;
    let mut total_rows = 0usize;
    for u in units
    {
        let status = if u.digest_ok() { "OK  " } else { "FAIL" };
        out.push_str(&format!(
            "  [{}] {:<9} {:<20} rows={:<4} gaps={} sha256={}…\n",
            status,
            u.unit,
            u.baseline,
            u.rows,
            u.gaps.len(),
            &u.digest_actual[..u.digest_actual.len().min(12)],
        ));
        if !u.digest_ok()
        {
            out.push_str(&format!(
                "         expected {}\n         actual   {}\n",
                u.digest_expected, u.digest_actual
            ));
        }
        total_gaps += u.gaps.len();
        total_rows += u.rows;
    }

    out.push_str(&format!(
        "\n{} units · {} baseline rows · {} documented gaps\n",
        units.len(),
        total_rows,
        total_gaps
    ));

    out.push_str("\nDOCUMENTED HIDDEN-DEPENDENCY GAPS\n");
    for u in units
    {
        out.push_str(&format!("  {}:\n", u.unit));
        for g in u.gaps
        {
            out.push_str(&format!("    - {g}\n"));
        }
    }

    out.push_str("\nEQUIVALENCE\n");
    out.push_str(
        "  Per-baseline exact-parity proofs run under `cargo test -p scirust-finmigrate`\n",
    );
    out.push_str(
        "  (tests/*_equivalence.rs). This tool verifies baseline integrity, not parity.\n",
    );

    out.push_str("\n⚠️  PRODUCTION GATE (BLOCKING)\n");
    out.push_str(
        "  All baselines are MODEL-DERIVED (no cobc in the sandbox). Before production,\n",
    );
    out.push_str(
        "  regenerate each from a live target-compiler run of its cobol/*.cbl under the\n",
    );
    out.push_str("  production ARITH option and re-diff at exact parity. See audit_report.md.\n");

    let verdict = if all_ok(units)
    {
        "PASS — all committed baselines verify against their recorded digests"
    }
    else
    {
        "FAIL — a committed baseline does not match its recorded digest (tamper/regeneration)"
    };
    out.push_str(&format!("\nVERDICT: {verdict}\n"));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_baseline_digest_verifies() {
        let units = audit_units();
        assert_eq!(units.len(), 4, "expected four migrated units");
        for u in &units
        {
            assert!(
                u.digest_ok(),
                "{} baseline {} digest mismatch:\n expected {}\n actual   {}",
                u.unit,
                u.baseline,
                u.digest_expected,
                u.digest_actual
            );
            assert!(u.rows > 0, "{} baseline has no data rows", u.unit);
            assert!(!u.gaps.is_empty(), "{} has no documented gaps", u.unit);
        }
        assert!(all_ok(&units));
    }

    #[test]
    fn report_mentions_gate_and_verdict() {
        let report = render_report(&audit_units());
        assert!(report.contains("PRODUCTION GATE"));
        assert!(report.contains("VERDICT: PASS"));
    }

    #[test]
    fn tamper_is_detected() {
        // A doctored unit with a digest that cannot match its bytes must FAIL.
        let doctored = UnitAudit {
            unit: "TEST",
            baseline: "x.csv",
            digest_expected: "0".repeat(64),
            digest_actual: sha256_hex(b"different bytes"),
            rows: 1,
            gaps: &["g"],
        };
        assert!(!doctored.digest_ok());
        assert!(!all_ok(&[doctored]));
    }
}
