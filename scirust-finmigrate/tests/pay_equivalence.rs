//! Phase 2 — Equivalence proof for the PAYCALC unit.
//!
//! Drives the decimal-native annuity-payment port against the committed Golden
//! Baseline and asserts EXACT parity on the 9-dp factor and the 2-dp payment.
//! Written before `paycalc::payment` existed (TDM): red against the stub, green
//! with the real port. The baseline itself carries the audit guarantee that the
//! decimal payment equals the legacy-float payment at the cent (see
//! gen_pay_baseline.py), so this test proves the shipped decimal path matches
//! that reconciled oracle.

use rust_decimal::Decimal;
use scirust_finmigrate::paycalc::{PayInput, payment};
use std::path::PathBuf;
use std::str::FromStr;

fn dec(s: &str) -> Decimal {
    Decimal::from_str(s).unwrap_or_else(|e| panic!("bad decimal {s:?}: {e}"))
}

#[test]
fn pay_equivalence_against_golden_baseline() {
    let path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "sandbox",
        "pay_baseline.csv",
    ]
    .iter()
    .collect();
    let text = std::fs::read_to_string(&path).unwrap();
    let tol = dec("0.0000000001"); // 1e-10 gate
    let mut checked = 0usize;

    for (i, line) in text.lines().enumerate()
    {
        if i == 0
        {
            assert_eq!(
                line, "case_id,principal,monthly_rate,num_periods,factor,payment",
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
        let case_id = f[0];
        let got = payment(&PayInput {
            principal: dec(f[1]),
            monthly_rate: dec(f[2]),
            num_periods: f[3].parse().unwrap(),
        })
        .unwrap_or_else(|e| panic!("`{case_id}`: port errored {e:?}"));

        let want_factor = dec(f[4]);
        let want_payment = dec(f[5]);
        assert!(
            got.factor == want_factor && (got.factor - want_factor).abs() < tol,
            "`{case_id}`: factor got {}, want {want_factor}",
            got.factor
        );
        assert!(
            got.payment == want_payment && (got.payment - want_payment).abs() < tol,
            "`{case_id}`: payment got {}, want {want_payment}",
            got.payment
        );
        checked += 1;
    }
    assert!(checked > 0, "empty baseline");
    eprintln!("pay equivalence: {checked} scenarios at 100% parity");
}
