//! Phase 2 — Equivalence proof for the AMORTSCH unit.
//!
//! Drives the Rust amortization port against every period of every scenario in
//! the committed Golden Baseline and asserts EXACT decimal parity, row-for-row,
//! including the row COUNT (early payoff / reconciliation must match). Written
//! before `amort::amortize` existed (TDM); it fails to compile / fails at
//! runtime against the stub, then turns green with the real port.

use rust_decimal::Decimal;
use scirust_finmigrate::amort::{AmortInput, amortize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::str::FromStr;

fn sandbox(file: &str) -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), "tests", "sandbox", file]
        .iter()
        .collect()
}

fn dec(s: &str) -> Decimal {
    Decimal::from_str(s).unwrap_or_else(|e| panic!("bad decimal {s:?}: {e}"))
}

struct Scenario {
    principal: Decimal,
    monthly_rate: Decimal,
    payment: Decimal,
    num_periods: u32,
}

fn load_scenarios() -> BTreeMap<String, Scenario> {
    let text = std::fs::read_to_string(sandbox("amort_scenarios.csv")).unwrap();
    let mut out = BTreeMap::new();
    for (i, line) in text.lines().enumerate()
    {
        if i == 0 || line.trim().is_empty()
        {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert_eq!(f.len(), 5, "bad scenario row: {line:?}");
        out.insert(
            f[0].to_string(),
            Scenario {
                principal: dec(f[1]),
                monthly_rate: dec(f[2]),
                payment: dec(f[3]),
                num_periods: f[4].parse().unwrap(),
            },
        );
    }
    out
}

#[derive(Debug, PartialEq, Eq)]
struct ExpectedRow {
    period: u32,
    interest: Decimal,
    principal: Decimal,
    payment: Decimal,
    balance: Decimal,
}

fn load_baseline() -> BTreeMap<String, Vec<ExpectedRow>> {
    let text = std::fs::read_to_string(sandbox("amort_baseline.csv")).unwrap();
    let mut out: BTreeMap<String, Vec<ExpectedRow>> = BTreeMap::new();
    for (i, line) in text.lines().enumerate()
    {
        if i == 0
        {
            assert_eq!(
                line, "case_id,period,interest,principal,payment,balance",
                "baseline header changed — regenerate and re-audit"
            );
            continue;
        }
        if line.trim().is_empty()
        {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        assert_eq!(f.len(), 6, "bad baseline row: {line:?}");
        out.entry(f[0].to_string()).or_default().push(ExpectedRow {
            period: f[1].parse().unwrap(),
            interest: dec(f[2]),
            principal: dec(f[3]),
            payment: dec(f[4]),
            balance: dec(f[5]),
        });
    }
    out
}

#[test]
fn amort_equivalence_against_golden_baseline() {
    let scenarios = load_scenarios();
    let baseline = load_baseline();
    let tol = dec("0.0000000001"); // 1e-10 gate
    let mut rows_checked = 0usize;

    for (case_id, scen) in &scenarios
    {
        let expected = baseline
            .get(case_id)
            .unwrap_or_else(|| panic!("no baseline rows for `{case_id}`"));
        let schedule = amortize(&AmortInput {
            principal: scen.principal,
            monthly_rate: scen.monthly_rate,
            payment: scen.payment,
            num_periods: scen.num_periods,
        })
        .unwrap_or_else(|e| panic!("`{case_id}`: port errored {e:?}"));

        assert_eq!(
            schedule.len(),
            expected.len(),
            "`{case_id}`: row count {} != baseline {} (early-payoff / reconciliation mismatch)",
            schedule.len(),
            expected.len()
        );

        for (got, want) in schedule.iter().zip(expected.iter())
        {
            let check = |field: &str, g: Decimal, w: Decimal| {
                assert!(
                    g == w && (g - w).abs() < tol,
                    "`{case_id}` period {}: `{field}` got {g}, want {w}",
                    want.period
                );
            };
            assert_eq!(
                got.period, want.period,
                "`{case_id}`: period index mismatch"
            );
            check("interest", got.interest, want.interest);
            check("principal", got.principal, want.principal);
            check("payment", got.payment, want.payment);
            check("balance", got.balance, want.balance);
            rows_checked += 1;
        }

        // The defining invariant: a completed schedule closes to exactly 0.00.
        if let Some(last) = schedule.last()
        {
            if schedule.len() < scen.num_periods as usize || last.balance == Decimal::ZERO
            {
                assert_eq!(
                    last.balance,
                    Decimal::ZERO,
                    "`{case_id}`: final balance must reconcile to exactly 0.00"
                );
            }
        }
    }
    assert!(rows_checked > 0, "no rows checked");
    eprintln!("amort equivalence: {rows_checked} period-rows at 100% parity");
}
