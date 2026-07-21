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
//!
//! ## Money-field certificate (second, independent dimension)
//!
//! The SHA-256 check above proves a baseline is *unmodified* — it says nothing
//! about whether the numbers inside are actually well-formed. Every unit except
//! CURRCVT stores its money fields through [`crate::store_money_rounded`] /
//! `store_money_trunc`, i.e. onto the same `PIC S9(9)V99` grid
//! [`crate::certified::money_field`] certifies (a proven ½-ULP / <1-ULP
//! round-trip bound — see that module). [`audit_units`] now also parses each
//! baseline's declared money columns and checks every value against that
//! field's *exactness domain* — orthogonal evidence from the tamper check:
//! a baseline could be byte-for-byte exactly what it claims to be and still
//! contain a value with a stray third decimal digit or an out-of-range
//! magnitude (a generator bug, a bad hand-edit before the digest was taken).
//! CURRCVT is explicitly declared with **no** money columns (empty
//! `money_columns`) rather than silently skipped, because its target amount's
//! minor-unit scale varies by destination currency (Gap-R) — this crate's
//! fixed 2-dp field does not describe it, and the report says so instead of
//! implying a certificate that was never checked.

use crate::certified::money_field;
use rust_decimal::Decimal;
use sha2::{Digest, Sha256};
use std::str::FromStr;

/// One unit's audit status: baseline identity, tamper check, gap count, and
/// money-field certificate.
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
    /// This unit's `PIC S9(9)V99` money-column names, checked against
    /// [`crate::certified::money_field`]'s exactness domain. Empty means the
    /// unit's money-like fields use a different or variable scale and are
    /// explicitly not covered (see the module docs) — never silently assumed
    /// certified.
    pub money_columns: &'static [&'static str],
    /// One entry per value that failed the money-field certificate. Empty
    /// when every declared column verified (including, vacuously, when
    /// `money_columns` is empty).
    pub money_grid_violations: Vec<String>,
}

impl UnitAudit {
    /// True when the recomputed digest matches the committed one (no tamper).
    pub fn digest_ok(&self) -> bool {
        self.digest_expected == self.digest_actual
    }

    /// True when every declared money column's values are exactly
    /// representable on the field's grid (trivially true if none are
    /// declared — see [`Self::money_columns`]).
    pub fn money_certified(&self) -> bool {
        self.money_grid_violations.is_empty()
    }
}

/// Parse a `header\nrow...` CSV (the simple, no-quoting convention already
/// used by `tests/equivalence.rs`) and check every named column's values
/// against the `PIC S9(9)V99` money field's exactness domain. A declared
/// column absent from the header, or a cell that fails to parse as a
/// [`Decimal`], is itself reported as a violation — schema drift is exactly
/// the kind of silent failure this check exists to catch.
fn money_grid_violations(csv: &str, money_columns: &[&str]) -> Vec<String> {
    let mut violations = Vec::new();
    if money_columns.is_empty()
    {
        return violations;
    }
    let field = money_field();
    let mut lines = csv.lines().filter(|l| !l.trim().is_empty());
    let Some(header) = lines.next()
    else
    {
        violations.push("baseline has no header row".to_string());
        return violations;
    };
    let headers: Vec<&str> = header.split(',').collect();
    let mut indices: Vec<(usize, &str)> = Vec::with_capacity(money_columns.len());
    for &name in money_columns
    {
        match headers.iter().position(|h| *h == name)
        {
            Some(i) => indices.push((i, name)),
            None => violations.push(format!(
                "declared money column `{name}` not found in baseline header — schema drift?"
            )),
        }
    }
    for row in lines
    {
        let cells: Vec<&str> = row.split(',').collect();
        let case_id = cells.first().copied().unwrap_or("<unknown row>");
        for &(idx, name) in &indices
        {
            let Some(raw) = cells.get(idx)
            else
            {
                violations.push(format!("{case_id}: missing cell for column `{name}`"));
                continue;
            };
            match Decimal::from_str(raw.trim())
            {
                Ok(value) if field.is_exactly_representable(value) =>
                {},
                Ok(value) => violations.push(format!(
                    "{case_id}: column `{name}` = {value} is not on the PIC S9(9)V99 money \
                     grid (out of domain, or finer than 2 dp)"
                )),
                Err(e) => violations.push(format!(
                    "{case_id}: column `{name}` = {raw:?} did not parse as a Decimal ({e})"
                )),
            }
        }
    }
    violations
}

struct Embedded {
    unit: &'static str,
    baseline: &'static str,
    csv: &'static str,
    sha256_sidecar: &'static str,
    gaps: &'static [&'static str],
    /// This unit's `PIC S9(9)V99` money columns — see [`UnitAudit::money_columns`].
    money_columns: &'static [&'static str],
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
        // principal/monthly_int/monthly_trunc/new_balance all pass through
        // store_money_rounded / store_money_trunc (lib.rs::accrue).
        money_columns: &["principal", "monthly_int", "monthly_trunc", "new_balance"],
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
        // interest is store_money_rounded; principal/payment/balance are
        // exact sums/differences of already-2dp values (amort.rs::amortize).
        money_columns: &["interest", "principal", "payment", "balance"],
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
        // monthly_rate (7dp) and factor (9dp) are NOT money-scale fields —
        // only principal/payment are (paycalc.rs::payment).
        money_columns: &["principal", "payment"],
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
        // nasd_days/excel_days are integer day counts, not money.
        money_columns: &["interest"],
    },
    Embedded {
        unit: "BRKTCALC",
        baseline: "brkt_baseline.csv",
        csv: include_str!("../tests/sandbox/brkt_baseline.csv"),
        sha256_sidecar: include_str!("../tests/sandbox/brkt_baseline.sha256"),
        gaps: &[
            "Gap-L marginal-not-flat",
            "Gap-M boundary-inclusivity",
            "Gap-N single-rounding-event",
            "Gap-O empty/partial-brackets",
        ],
        // effective_pct is a derived display ratio, not a PIC money field.
        money_columns: &["base", "tax", "flat_tax"],
    },
    Embedded {
        unit: "CURRCVT",
        baseline: "curr_baseline.csv",
        csv: include_str!("../tests/sandbox/curr_baseline.csv"),
        sha256_sidecar: include_str!("../tests/sandbox/curr_baseline.sha256"),
        gaps: &[
            "Gap-P triangulation-mandatory",
            "Gap-Q euro-intermediate-3dp (GATE)",
            "Gap-R variable-target-minor-unit",
            "Gap-S rates-6-sig-figs",
        ],
        // Deliberately empty: `result`/`direct` vary by destination
        // currency's minor unit (Gap-R, e.g. ITL/ESP have 0 decimals) and
        // `euro` is 3dp (EURO_SCALE), not this crate's 2dp money grid — the
        // module docs explain why this is declared, not just omitted.
        money_columns: &[],
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

/// Audit every migrated unit: recompute and verify its baseline digest, and
/// check its declared money columns against the `PIC S9(9)V99` certificate.
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
            money_columns: e.money_columns,
            money_grid_violations: money_grid_violations(e.csv, e.money_columns),
        })
        .collect()
}

/// True when every unit's baseline digest verifies AND every declared money
/// column satisfies its `PIC S9(9)V99` certificate — both dimensions gate
/// the overall verdict, not just tamper-evidence.
pub fn all_ok(units: &[UnitAudit]) -> bool {
    units.iter().all(|u| u.digest_ok() && u.money_certified())
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

    out.push_str(
        "\nMONEY-FIELD CERTIFICATE (PIC S9(9)V99 exactness domain, crate::certified::money_field)\n",
    );
    for u in units
    {
        if u.money_columns.is_empty()
        {
            out.push_str(&format!(
                "  {:<9} not covered — variable/other-scale money fields (see documented gaps)\n",
                u.unit
            ));
        }
        else if u.money_certified()
        {
            out.push_str(&format!(
                "  {:<9} certified — {} column(s): {}\n",
                u.unit,
                u.money_columns.len(),
                u.money_columns.join(", ")
            ));
        }
        else
        {
            out.push_str(&format!(
                "  {:<9} FAILED — {} violation(s):\n",
                u.unit,
                u.money_grid_violations.len()
            ));
            for v in &u.money_grid_violations
            {
                out.push_str(&format!("      - {v}\n"));
            }
        }
    }

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
        "PASS — every committed baseline verifies against its recorded digest, and every \
         declared money column satisfies the PIC S9(9)V99 certificate"
    }
    else
    {
        "FAIL — a committed baseline does not match its recorded digest (tamper/regeneration), \
         or a money-field value violates its declared PIC-field grid (see above)"
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
        assert_eq!(units.len(), 6, "expected six migrated units");
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
            assert!(
                u.money_certified(),
                "{} has money-grid violations: {:?}",
                u.unit,
                u.money_grid_violations
            );
        }
        // Every unit except CURRCVT (variable minor unit, Gap-R) declares
        // real money columns — the certificate must be load-bearing, not
        // vacuous everywhere.
        let with_columns = units.iter().filter(|u| !u.money_columns.is_empty()).count();
        assert_eq!(
            with_columns, 5,
            "expected 5/6 units to declare money columns"
        );
        let currcvt = units.iter().find(|u| u.unit == "CURRCVT").unwrap();
        assert!(currcvt.money_columns.is_empty());
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
            money_columns: &[],
            money_grid_violations: Vec::new(),
        };
        assert!(!doctored.digest_ok());
        assert!(!all_ok(&[doctored]));
    }

    #[test]
    fn money_grid_violation_is_detected_and_reported() {
        let csv = "case_id,amount\nrow1,100.00\nrow2,100.001\nrow3,1000000000.00\n";
        let violations = money_grid_violations(csv, &["amount"]);
        assert_eq!(violations.len(), 2, "{violations:?}");
        assert!(violations[0].contains("row2"), "{violations:?}");
        assert!(violations[1].contains("row3"), "{violations:?}");
    }

    #[test]
    fn money_grid_check_is_vacuous_when_no_columns_declared() {
        // CURRCVT's variable minor unit means it declares no money columns;
        // the checker must not silently invent a certificate for it.
        assert!(money_grid_violations("case_id,x\nr,1.234\n", &[]).is_empty());
    }

    #[test]
    fn schema_drift_on_a_declared_column_is_itself_a_violation() {
        let violations = money_grid_violations("case_id,amount\nr,1.00\n", &["missing_column"]);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("missing_column"));
    }

    #[test]
    fn money_grid_violation_flips_the_overall_verdict() {
        let doctored = UnitAudit {
            unit: "TEST",
            baseline: "x.csv",
            digest_expected: "0".repeat(64),
            digest_actual: "0".repeat(64),
            rows: 1,
            gaps: &["g"],
            money_columns: &["amount"],
            money_grid_violations: vec![
                "row1: column `amount` = 1.234 is not on the grid".to_string(),
            ],
        };
        assert!(doctored.digest_ok(), "digest matches in this scenario");
        assert!(!doctored.money_certified());
        assert!(
            !all_ok(&[doctored]),
            "a money-grid violation must fail the overall verdict"
        );
    }

    #[test]
    fn report_shows_the_money_field_certificate_section() {
        let report = render_report(&audit_units());
        assert!(report.contains("MONEY-FIELD CERTIFICATE"));
        let intaccr_line = report
            .lines()
            .find(|l| l.contains("INTACCR") && l.contains("certified"))
            .unwrap_or_else(|| panic!("no INTACCR-certified line in report:\n{report}"));
        assert!(intaccr_line.contains("principal"));
        assert!(
            report
                .lines()
                .any(|l| l.contains("CURRCVT") && l.contains("not covered")),
            "CURRCVT must be explicitly declared out of scope in the money-field \
             certificate section, not silently omitted:\n{report}"
        );
    }
}
