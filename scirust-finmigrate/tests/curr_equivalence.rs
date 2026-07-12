//! Phase 2 — Equivalence proof for the CURRCVT unit.
//!
//! Drives euro triangulation (EC 1103/97) against the committed Golden Baseline
//! and asserts EXACT parity on the result, the 3-dp euro intermediate, and the
//! (unlawful) direct cross-rate. It also pins, at the Rust level, that the
//! triangulated result differs from the direct one on exactly the rows the audit
//! says it should. Written before `currcvt::convert` existed (TDM): red, green.

use rust_decimal::Decimal;
use scirust_finmigrate::currcvt::{convert, direct_convert};
use std::path::PathBuf;
use std::str::FromStr;

fn dec(s: &str) -> Decimal {
    Decimal::from_str(s).unwrap_or_else(|e| panic!("bad decimal {s:?}: {e}"))
}

#[test]
fn curr_equivalence_against_golden_baseline() {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "sandbox",
        "curr_baseline.csv",
    ]
    .iter()
    .collect();
    let text = std::fs::read_to_string(&path).unwrap();
    let tol = dec("0.0000000001");
    let mut checked = 0usize;
    let mut divergences = 0usize;

    for (i, line) in text.lines().enumerate()
    {
        if i == 0
        {
            assert_eq!(
                line, "case_id,amount,from_ccy,to_ccy,result,direct,euro",
                "baseline header changed — regenerate and re-audit"
            );
            continue;
        }
        if line.trim().is_empty()
        {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert_eq!(f.len(), 7, "malformed baseline row {i}: {line:?}");
        let case_id = f[0];
        let amount = dec(f[1]);
        let (from, to) = (f[2], f[3]);
        let want_result = dec(f[4]);
        let want_direct = dec(f[5]);
        let want_euro = dec(f[6]);

        let got =
            convert(amount, from, to).unwrap_or_else(|e| panic!("`{case_id}`: port errored {e:?}"));
        assert!(
            got.result == want_result && (got.result - want_result).abs() < tol,
            "`{case_id}`: result got {}, want {want_result}",
            got.result
        );
        assert_eq!(got.euro, want_euro, "`{case_id}`: euro intermediate");

        let got_direct = direct_convert(amount, from, to)
            .unwrap_or_else(|e| panic!("`{case_id}`: direct errored {e:?}"));
        assert_eq!(got_direct, want_direct, "`{case_id}`: direct cross-rate");

        if got.result != got_direct
        {
            divergences += 1;
        }
        checked += 1;
    }
    assert_eq!(checked, 10, "expected 10 scenarios");
    // The whole point of the unit: at least one lawful-vs-direct divergence.
    assert!(
        divergences >= 2,
        "expected the recorded triangulation divergences"
    );
    eprintln!("curr equivalence: {checked} scenarios ({divergences} lawful≠direct) at 100% parity");
}

#[test]
fn unknown_currency_is_reported() {
    let e = convert(dec("100.00"), "XXX", "DEM").unwrap_err();
    assert!(matches!(
        e,
        scirust_finmigrate::AccrualError::UnknownCurrency { .. }
    ));
}
