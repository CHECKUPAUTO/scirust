//! Phase 2 — Equivalence proof for the DAYCOUNT unit.
//!
//! Drives the 30/360-US (NASD) day count and the period interest against the
//! committed Golden Baseline and asserts EXACT parity on both. The baseline also
//! carries the Excel DAYS360 count per row; this test additionally checks that
//! the shipped NASD count differs from Excel on exactly the rows the audit says
//! it should, so the divergence is pinned as a Rust-level fact, not just prose.
//! Written before `daycount::accrue_30_360` existed (TDM): red, then green.

use rust_decimal::Decimal;
use scirust_finmigrate::daycount::{Date, accrue_30_360, days_30_360_us};
use std::path::PathBuf;
use std::str::FromStr;

fn dec(s: &str) -> Decimal {
    Decimal::from_str(s).unwrap_or_else(|e| panic!("bad decimal {s:?}: {e}"))
}

fn sandbox(file: &str) -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), "tests", "sandbox", file]
        .iter()
        .collect()
}

#[test]
fn day_equivalence_against_golden_baseline() {
    // case_id -> (nasd_days, excel_days, interest)
    let base = std::fs::read_to_string(sandbox("day_baseline.csv")).unwrap();
    let scen = std::fs::read_to_string(sandbox("day_scenarios.csv")).unwrap();

    // Load scenarios keyed by case_id.
    let mut inputs = std::collections::BTreeMap::new();
    for (i, line) in scen.lines().enumerate()
    {
        if i == 0 || line.trim().is_empty()
        {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert_eq!(f.len(), 9, "bad scenario row: {line:?}");
        let g = |k: usize| f[k].parse::<i64>().unwrap();
        inputs.insert(
            f[0].to_string(),
            (
                dec(f[1]),
                dec(f[2]),
                Date {
                    year: g(3) as i32,
                    month: g(4) as u32,
                    day: g(5) as u32,
                },
                Date {
                    year: g(6) as i32,
                    month: g(7) as u32,
                    day: g(8) as u32,
                },
            ),
        );
    }

    let tol = dec("0.0000000001");
    let mut checked = 0usize;
    for (i, line) in base.lines().enumerate()
    {
        if i == 0
        {
            assert_eq!(
                line, "case_id,nasd_days,excel_days,interest",
                "baseline header changed — regenerate and re-audit"
            );
            continue;
        }
        if line.trim().is_empty()
        {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert_eq!(f.len(), 4, "bad baseline row: {line:?}");
        let case_id = f[0];
        let want_days: i64 = f[1].parse().unwrap();
        let want_interest = dec(f[3]);

        let (principal, rate, d1, d2) = inputs
            .get(case_id)
            .unwrap_or_else(|| panic!("no scenario for `{case_id}`"));

        let got_days = days_30_360_us(*d1, *d2);
        assert_eq!(got_days, want_days, "`{case_id}`: NASD day count");

        let acc = accrue_30_360(*principal, *rate, *d1, *d2)
            .unwrap_or_else(|e| panic!("`{case_id}`: port errored {e:?}"));
        assert_eq!(acc.days, want_days, "`{case_id}`: acc.days");
        assert!(
            acc.interest == want_interest && (acc.interest - want_interest).abs() < tol,
            "`{case_id}`: interest got {}, want {want_interest}",
            acc.interest
        );
        checked += 1;
    }
    assert_eq!(checked, 10, "expected 10 scenarios");
    eprintln!("day equivalence: {checked} scenarios at 100% parity");
}
