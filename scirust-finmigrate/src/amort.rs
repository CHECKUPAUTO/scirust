//! AMORTSCH migration unit — fixed-payment loan amortization schedule.
//!
//! Port of `cobol/AMORTSCH.cbl`; contract in `cobol/SEMANTICS_AMORT.md`. Where
//! [`crate::accrue`] is a single store, this carries a running balance across
//! periods, so it reproduces two extra COBOL behaviours exactly:
//! **accumulated per-period rounding drift** and **final-payment reconciliation**
//! (the schedule closes to exactly `0.00`). All arithmetic is decimal; there is
//! no floating point and no exponentiation in the money path.

use crate::{check_size, store_money_rounded};
use rust_decimal::{Decimal, RoundingStrategy};

/// Fractional digits of the `PIC SV9(7)` monthly-rate field.
pub const AMORT_RATE_SCALE: u32 = 7;

/// Inputs to a schedule, mirroring AMORTSCH's WORKING-STORAGE inputs.
pub struct AmortInput {
    /// `WS-ORIG-PRINCIPAL` — starting balance (2 dp).
    pub principal: Decimal,
    /// `WS-MONTHLY-RATE` — monthly rate (7 dp).
    pub monthly_rate: Decimal,
    /// `WS-PAYMENT` — fixed scheduled payment (2 dp).
    pub payment: Decimal,
    /// `WS-NUM-PERIODS` — maximum number of periods.
    pub num_periods: u32,
}

/// One emitted schedule row (the per-period DISPLAY of AMORTSCH).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeriodRow {
    pub period: u32,
    /// `WS-INTEREST` — interest on the opening balance, ROUNDED to 2 dp.
    pub interest: Decimal,
    /// `WS-PRINC-PORTION` — principal repaid this period (may be negative).
    pub principal: Decimal,
    /// `WS-ACTUAL-PAYMENT` — payment actually taken (scheduled, or reconciled).
    pub payment: Decimal,
    /// `WS-BALANCE` — balance after posting this period.
    pub balance: Decimal,
}

/// Build the amortization schedule. Decimal-exact, COBOL-faithful.
///
/// Stops early when the balance reaches `0.00` (early payoff); the returned
/// length is the number of periods actually executed (≤ `num_periods`).
///
/// ```
/// use rust_decimal::Decimal;
/// use std::str::FromStr;
/// use scirust_finmigrate::amort::{amortize, AmortInput};
/// let d = |s: &str| Decimal::from_str(s).unwrap();
/// // Zero-rate 12-month loan: principal repaid in equal slices, closes at 0.00.
/// let sched = amortize(&AmortInput {
///     principal: d("1200.00"), monthly_rate: d("0.0000000"),
///     payment: d("100.00"), num_periods: 12,
/// }).unwrap();
/// assert_eq!(sched.len(), 12);
/// assert_eq!(sched.last().unwrap().balance, Decimal::ZERO);
/// ```
pub fn amortize(input: &AmortInput) -> Result<Vec<PeriodRow>, crate::AccrualError> {
    // Fields carry their declared fixed scale (enforced on store).
    let rate = input
        .monthly_rate
        .round_dp_with_strategy(AMORT_RATE_SCALE, RoundingStrategy::MidpointAwayFromZero);
    let payment = store_money_rounded(input.payment);
    let mut balance = check_size("principal", store_money_rounded(input.principal))?;

    let mut schedule = Vec::new();
    let mut period = 1u32;
    while period <= input.num_periods && !balance.is_zero()
    {
        // One rounding event: interest on the CURRENT balance (Gap: drift source).
        let interest = check_size("interest", store_money_rounded(balance * rate))?;
        // Principal portion of the scheduled payment; exact, may be negative
        // (negative amortization when payment < interest).
        let mut princ = check_size("principal", payment - interest)?;

        // Final-payment reconciliation: at the last period, or as soon as the
        // principal portion would meet-or-exceed the balance, pay off exactly.
        let actual_payment = if period == input.num_periods || princ >= balance
        {
            princ = balance;
            check_size("actual_payment", interest + princ)?
        }
        else
        {
            payment
        };

        balance = check_size("balance", balance - princ)?;
        schedule.push(PeriodRow {
            period,
            interest,
            principal: princ,
            payment: actual_payment,
            balance,
        });
        period += 1;
    }
    Ok(schedule)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn d(s: &str) -> Decimal {
        Decimal::from_str(s).unwrap()
    }

    fn input(p: &str, r: &str, pay: &str, n: u32) -> AmortInput {
        AmortInput {
            principal: d(p),
            monthly_rate: d(r),
            payment: d(pay),
            num_periods: n,
        }
    }

    #[test]
    fn closes_to_exactly_zero_after_drift() {
        let s = amortize(&input("10000.00", "0.0041667", "200.00", 60)).unwrap();
        assert_eq!(s.last().unwrap().balance, Decimal::ZERO);
        // Every balance is monotonically non-increasing for a normal loan.
        for w in s.windows(2)
        {
            assert!(w[1].balance <= w[0].balance);
        }
    }

    #[test]
    fn early_payoff_stops_before_num_periods() {
        let s = amortize(&input("500.00", "0.0041667", "300.00", 12)).unwrap();
        assert_eq!(s.len(), 2, "loan should clear in 2 periods, not 12");
        assert_eq!(s.last().unwrap().balance, Decimal::ZERO);
    }

    #[test]
    fn negative_amortization_grows_then_final_payoff() {
        let s = amortize(&input("10000.00", "0.0100000", "50.00", 6)).unwrap();
        assert_eq!(s.len(), 6);
        // Balance grows while payment < interest ...
        assert!(s[0].principal < Decimal::ZERO);
        assert!(s[1].balance > s[0].balance);
        // ... but the last period forces payoff to exactly zero.
        assert_eq!(s.last().unwrap().balance, Decimal::ZERO);
    }

    #[test]
    fn single_period_repays_whole_balance() {
        let s = amortize(&input("1000.00", "0.0050000", "250.00", 1)).unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].principal, d("1000.00"));
        assert_eq!(s[0].payment, d("1005.00"));
        assert_eq!(s[0].balance, Decimal::ZERO);
    }

    #[test]
    fn size_error_on_runaway_negative_amortization() {
        // Huge balance + high rate + tiny payment: interest overflows the field.
        let e = amortize(&input("999999999.99", "0.5000000", "1.00", 3)).unwrap_err();
        assert!(matches!(e, crate::AccrualError::SizeError { .. }));
    }
}
