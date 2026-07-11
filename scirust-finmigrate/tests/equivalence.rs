//! Phase 2 — Equivalence proof.
//!
//! Drives the Rust port against every row of the committed Golden Baseline and
//! asserts EXACT decimal parity. Written before the implementation existed
//! (Test-Driven Migration): with the stub `accrue` it fails on the very first
//! discriminating row; only the real decimal port turns it green. See
//! `audit_trail.log` for the recorded red→green transition.
//!
//! Tolerance: the persona sets the failure threshold at a deviation > 1e-10.
//! For cent-scale packed decimals the port is *exactly* equal to the oracle, so
//! we assert exact `Decimal` equality AND check the |Δ| < 1e-10 gate explicitly,
//! comparing parsed values (not strings) so a `-0.00` vs `0.00` signed-zero
//! display difference is not a false failure (audit_report.md §4).

use rust_decimal::Decimal;
use scirust_finmigrate::{Accrual, accrue};
use std::path::PathBuf;
use std::str::FromStr;

const TOLERANCE: &str = "0.0000000001"; // 1e-10

struct BaselineRow {
    case_id: String,
    principal: Decimal,
    annual_rate: Decimal,
    monthly_int: Decimal,
    monthly_trunc: Decimal,
    new_balance: Decimal,
}

fn load_baseline() -> Vec<BaselineRow> {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "sandbox",
        "golden_baseline.csv",
    ]
    .iter()
    .collect();
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read golden baseline at {}: {e}", path.display()));

    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate()
    {
        if i == 0
        {
            assert_eq!(
                line, "case_id,principal,annual_rate,monthly_int,monthly_trunc,new_balance",
                "baseline header changed — regenerate and re-audit"
            );
            continue;
        }
        if line.trim().is_empty()
        {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert_eq!(f.len(), 6, "malformed baseline row {i}: {line:?}");
        let dec =
            |s: &str| Decimal::from_str(s).unwrap_or_else(|e| panic!("bad decimal {s:?}: {e}"));
        rows.push(BaselineRow {
            case_id: f[0].to_string(),
            principal: dec(f[1]),
            annual_rate: dec(f[2]),
            monthly_int: dec(f[3]),
            monthly_trunc: dec(f[4]),
            new_balance: dec(f[5]),
        });
    }
    assert!(!rows.is_empty(), "empty baseline");
    rows
}

/// Numeric-equality gate: exact match required, |Δ| < 1e-10 double-checked.
fn assert_parity(field: &str, case: &str, got: Decimal, want: Decimal) {
    let tol = Decimal::from_str(TOLERANCE).unwrap();
    let delta = (got - want).abs();
    assert!(
        got == want && delta < tol,
        "DIVERGENCE in `{field}` for case `{case}`: got {got}, want {want}, |Δ|={delta} \
         (> tolerance {TOLERANCE}). Write a Failure Case in tests/sandbox/ to reproduce.",
    );
}

#[test]
fn equivalence_against_golden_baseline() {
    let rows = load_baseline();
    let mut checked = 0usize;
    for row in &rows
    {
        let Accrual {
            monthly_int,
            monthly_trunc,
            new_balance,
        } = accrue(row.principal, row.annual_rate)
            .unwrap_or_else(|e| panic!("case `{}`: port returned error {e:?}", row.case_id));
        assert_parity("monthly_int", &row.case_id, monthly_int, row.monthly_int);
        assert_parity(
            "monthly_trunc",
            &row.case_id,
            monthly_trunc,
            row.monthly_trunc,
        );
        assert_parity("new_balance", &row.case_id, new_balance, row.new_balance);
        checked += 1;
    }
    assert_eq!(checked, rows.len(), "not every baseline row was checked");
    eprintln!("equivalence: {checked}/{} rows at 100% parity", rows.len());
}

/// The half-cent discriminators are the ones that separate NEAREST-AWAY-FROM-ZERO
/// from banker's rounding and from truncation. Pin them explicitly so a
/// regression is unmissable even if the baseline file is swapped.
#[test]
fn half_cent_tie_rounds_away_from_zero() {
    let d = |s: &str| Decimal::from_str(s).unwrap();
    // +0.005 -> ROUNDED 0.01 (up), TRUNC 0.00
    let a = accrue(d("100.00"), d("0.00060")).unwrap();
    assert_eq!(a.monthly_int, d("0.01"), "positive half-cent must round up");
    assert_eq!(
        a.monthly_trunc,
        d("0.00"),
        "positive half-cent must truncate down"
    );
    // -0.005 -> ROUNDED -0.01 (away from zero), TRUNC 0.00
    let b = accrue(d("-100.00"), d("0.00060")).unwrap();
    assert_eq!(
        b.monthly_int,
        d("-0.01"),
        "negative half-cent must round away from zero"
    );
    // 0.025 -> ROUNDED 0.03 (away), banker's would give 0.02
    let c = accrue(d("100.00"), d("0.00300")).unwrap();
    assert_eq!(
        c.monthly_int,
        d("0.03"),
        "tie must NOT use banker's rounding"
    );
}
