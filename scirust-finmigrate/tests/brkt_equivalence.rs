//! Phase 2 — Equivalence proof for the BRKTCALC unit.
//!
//! Drives the progressive marginal tax against the committed Golden Baseline and
//! asserts EXACT parity. The baseline also carries the WRONG flat-top-rate tax;
//! this test pins, at the Rust level, that the marginal result differs from the
//! flat one on every scenario with a taxable base (so the marginal-vs-flat
//! divergence is a checked fact). Written before `brktcalc::bracket_tax` existed
//! (TDM): red against the stub, green with the real port.

use rust_decimal::Decimal;
use scirust_finmigrate::brktcalc::{bracket_tax, flat_top_tax};
use std::path::PathBuf;
use std::str::FromStr;

fn dec(s: &str) -> Decimal {
    Decimal::from_str(s).unwrap_or_else(|e| panic!("bad decimal {s:?}: {e}"))
}

#[test]
fn brkt_equivalence_against_golden_baseline() {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "sandbox",
        "brkt_baseline.csv",
    ]
    .iter()
    .collect();
    let text = std::fs::read_to_string(&path).unwrap();
    let tol = dec("0.0000000001");
    let mut checked = 0usize;

    for (i, line) in text.lines().enumerate()
    {
        if i == 0
        {
            assert_eq!(
                line, "case_id,base,tax,flat_tax,effective_pct",
                "baseline header changed — regenerate and re-audit"
            );
            continue;
        }
        if line.trim().is_empty()
        {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert_eq!(f.len(), 5, "malformed baseline row {i}: {line:?}");
        let case_id = f[0];
        let base = dec(f[1]);
        let want_tax = dec(f[2]);
        let want_flat = dec(f[3]);

        let got_tax =
            bracket_tax(base).unwrap_or_else(|e| panic!("`{case_id}`: port errored {e:?}"));
        assert!(
            got_tax == want_tax && (got_tax - want_tax).abs() < tol,
            "`{case_id}`: tax got {got_tax}, want {want_tax}"
        );

        let got_flat = flat_top_tax(base);
        assert_eq!(
            got_flat, want_flat,
            "`{case_id}`: flat_tax got {got_flat}, want {want_flat}"
        );

        // Marginal must be strictly less than flat once any base spills past the
        // first (0%) bracket — the whole point of graduated taxation.
        if base > dec("10000.00")
        {
            assert!(
                got_tax < got_flat,
                "`{case_id}`: marginal {got_tax} should be < flat {got_flat}"
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 9, "expected 9 scenarios");
    eprintln!("brkt equivalence: {checked} scenarios at 100% parity");
}
