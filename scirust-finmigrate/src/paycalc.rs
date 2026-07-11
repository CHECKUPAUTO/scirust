//! PAYCALC migration unit — level (annuity) payment for a fully-amortizing loan.
//!
//! Port of `cobol/PAYCALC.cbl`; contract in `cobol/SEMANTICS_PAY.md`. Computes
//! the fixed monthly payment that [`crate::amort::amortize`] consumes as an
//! input, closing the loop between the units.
//!
//! ## The float question, resolved
//! The textbook annuity formula `P·i / (1 − (1+i)^−n)` has a **negative**
//! exponent, which in COBOL forces the whole expression into IEEE-754 double
//! (fractional/negative exponents evaluate in long floating point). This port
//! uses the algebraically-identical **positive-integer** form
//! `payment = P·i·f / (f − 1)` with `f = (1+i)^n`. A nonzero integer power is a
//! *succession of fixed-point multiplications* in COBOL, so the whole
//! computation stays decimal — the no-float mandate holds. The baseline proves
//! the decimal payment equals the legacy-float payment to the cent.

use crate::{MONEY_SCALE, check_size, store_money_rounded};
use rust_decimal::{Decimal, RoundingStrategy};

/// Fractional digits of the `PIC SV9(7)` rate field.
pub const PAY_RATE_SCALE: u32 = 7;
/// Fractional digits of the `PIC S9(5)V9(9)` factor field.
pub const FACTOR_SCALE: u32 = 9;

/// Inputs mirroring PAYCALC's WORKING-STORAGE.
pub struct PayInput {
    /// `WS-PRINCIPAL` — loan amount (2 dp).
    pub principal: Decimal,
    /// `WS-RATE` — monthly rate (7 dp).
    pub monthly_rate: Decimal,
    /// `WS-NUM-PERIODS` — integer term.
    pub num_periods: u32,
}

/// Outputs: the compounding factor and the level payment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayCalc {
    /// `WS-FACTOR` — `(1+i)^n`, stored at 9 dp.
    pub factor: Decimal,
    /// `WS-PAYMENT` — level payment, ROUNDED to 2 dp.
    pub payment: Decimal,
}

fn round_to(value: Decimal, scale: u32) -> Decimal {
    value.round_dp_with_strategy(scale, RoundingStrategy::MidpointAwayFromZero)
}

/// Compute the level payment. Decimal-exact, COBOL-faithful.
///
/// A zero `num_periods` is outside the legacy `9(3)` term domain (≥ 1) and would
/// divide by zero, so it is rejected as [`crate::AccrualError::SizeError`] rather
/// than panicking.
///
/// ```
/// use rust_decimal::Decimal;
/// use std::str::FromStr;
/// use scirust_finmigrate::paycalc::{payment, PayInput};
/// let d = |s: &str| Decimal::from_str(s).unwrap();
/// // $10,000 at 0.41667%/month for 60 months -> $188.71/month.
/// let p = payment(&PayInput {
///     principal: d("10000.00"), monthly_rate: d("0.0041667"), num_periods: 60,
/// }).unwrap();
/// assert_eq!(p.payment, d("188.71"));
/// ```
pub fn payment(input: &PayInput) -> Result<PayCalc, crate::AccrualError> {
    if input.num_periods == 0
    {
        return Err(crate::AccrualError::SizeError {
            field: "num_periods",
            value: Decimal::ZERO,
        });
    }
    let principal = check_size("principal", store_money_rounded(input.principal))?;
    let rate = round_to(input.monthly_rate, PAY_RATE_SCALE);
    let n = Decimal::from(input.num_periods);

    // Zero-rate loan: straight-line principal / periods (the annuity form would
    // divide by (f - 1) = 0). Matches the legacy IF WS-RATE = ZERO branch.
    if rate.is_zero()
    {
        let factor = round_to(Decimal::ONE, FACTOR_SCALE);
        let payment = check_size("payment", store_money_rounded(principal / n))?;
        return Ok(PayCalc { factor, payment });
    }

    // f = (1+i)^n as a succession of fixed-point multiplications (integer power),
    // then ROUNDED once into the 9-dp factor field.
    let one_plus = Decimal::ONE + rate;
    let mut f = Decimal::ONE;
    for _ in 0..input.num_periods
    {
        f *= one_plus;
    }
    let factor = round_to(f, FACTOR_SCALE);

    // payment = (P * i * f) / (f - 1), evaluated at full precision, ROUNDED once.
    let numerator = principal * rate * factor;
    let payment = check_size(
        "payment",
        (numerator / (factor - Decimal::ONE))
            .round_dp_with_strategy(MONEY_SCALE, RoundingStrategy::MidpointAwayFromZero),
    )?;
    Ok(PayCalc { factor, payment })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn d(s: &str) -> Decimal {
        Decimal::from_str(s).unwrap()
    }

    fn inp(p: &str, r: &str, n: u32) -> PayInput {
        PayInput {
            principal: d(p),
            monthly_rate: d(r),
            num_periods: n,
        }
    }

    #[test]
    fn standard_mortgage_payment() {
        let p = payment(&inp("10000.00", "0.0041667", 60)).unwrap();
        assert_eq!(p.factor, d("1.283361235"));
        assert_eq!(p.payment, d("188.71"));
    }

    #[test]
    fn zero_rate_is_straight_line() {
        let p = payment(&inp("1200.00", "0.0000000", 10)).unwrap();
        assert_eq!(p.factor, d("1.000000000"));
        assert_eq!(p.payment, d("120.00"));
    }

    #[test]
    fn single_period_is_principal_plus_one_month() {
        let p = payment(&inp("5000.00", "0.0041667", 1)).unwrap();
        assert_eq!(p.payment, d("5020.83"));
    }

    /// The payment PAYCALC produces must actually amortize the loan to 0.00 when
    /// fed back into AMORTSCH — the two units agree at the boundary.
    #[test]
    fn payment_amortizes_to_zero_in_amort() {
        use crate::amort::{AmortInput, amortize};
        let pc = payment(&inp("10000.00", "0.0041667", 60)).unwrap();
        let sched = amortize(&AmortInput {
            principal: d("10000.00"),
            monthly_rate: d("0.0041667"),
            payment: pc.payment,
            num_periods: 60,
        })
        .unwrap();
        assert_eq!(sched.len(), 60);
        assert_eq!(sched.last().unwrap().balance, Decimal::ZERO);
    }

    #[test]
    fn zero_term_is_size_error_not_div_by_zero() {
        let e = payment(&inp("1000.00", "0.0041667", 0)).unwrap_err();
        assert!(matches!(
            e,
            crate::AccrualError::SizeError {
                field: "num_periods",
                ..
            }
        ));
    }
}
