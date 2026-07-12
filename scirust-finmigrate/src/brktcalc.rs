//! BRKTCALC migration unit — progressive (marginal) bracketed tax.
//!
//! Port of `cobol/BRKTCALC.cbl`; contract in `cobol/SEMANTICS_BRKT.md`. Unlike
//! the earlier single-`COMPUTE` units, this walks a bracket TABLE (the COBOL
//! `OCCURS` + `PERFORM VARYING` pattern) applying each rate MARGINALLY — only to
//! the slice of the base within that bracket — and rounds the accumulated total
//! ONCE. A flat `base × top_rate` port over-taxes wildly; the sandbox records
//! that flat figure alongside as audit evidence.

use crate::{MONEY_SCALE, RATE_SCALE, check_size, store_money_rounded};
use rust_decimal::{Decimal, RoundingStrategy};

/// A `(lower_inclusive_floor, marginal_rate)` bracket.
type Bracket = (Decimal, Decimal);

/// The graduated schedule, matching `BRKTCALC.cbl` `1000-LOAD-TABLE`.
/// Bracket `i` covers `(lower(i) .. lower(i+1)]`; the last is unbounded.
fn brackets() -> [Bracket; 5] {
    let money = |cents: i64| Decimal::new(cents, MONEY_SCALE);
    let rate = |units: i64| Decimal::new(units, RATE_SCALE);
    [
        (money(0), rate(0)),               //        0 ..  10 000  @  0%
        (money(1_000_000), rate(10_000)),  //  10 000 ..  40 000  @ 10%
        (money(4_000_000), rate(22_000)),  //  40 000 ..  85 000  @ 22%
        (money(8_500_000), rate(24_000)),  //  85 000 .. 165 000  @ 24%
        (money(16_500_000), rate(32_000)), // 165 000 ..          @ 32%
    ]
}

/// The top bracket's marginal rate (used for the wrong-flat cross-check).
fn top_rate() -> Decimal {
    Decimal::new(32_000, RATE_SCALE)
}

/// Progressive marginal tax on `base`. Decimal-exact, COBOL-faithful.
///
/// A negative base is out of contract (a tax base is ≥ 0) and is rejected as a
/// [`crate::AccrualError::SizeError`] rather than yielding a nonsensical result.
///
/// ```
/// use rust_decimal::Decimal;
/// use std::str::FromStr;
/// use scirust_finmigrate::brktcalc::bracket_tax;
/// let d = |s: &str| Decimal::from_str(s).unwrap();
/// // 100_000 -> 3_000 + 9_900 + 3_600 = 16_500 (marginal), NOT 32_000 (flat).
/// assert_eq!(bracket_tax(d("100000.00")).unwrap(), d("16500.00"));
/// ```
pub fn bracket_tax(base: Decimal) -> Result<Decimal, crate::AccrualError> {
    let base = store_money_rounded(base);
    if base.is_sign_negative() && !base.is_zero()
    {
        return Err(crate::AccrualError::SizeError {
            field: "base",
            value: base,
        });
    }
    check_size("base", base)?;

    let table = brackets();
    let n = table.len();
    // Accumulate the marginal amounts at FULL precision; round ONCE at the end.
    let mut accum = Decimal::ZERO;
    for i in 0..n
    {
        let (lower, rate) = table[i];
        let upper = if i + 1 < n { table[i + 1].0 } else { base };
        let upper = upper.min(base);
        let portion = if upper > lower
        {
            upper - lower
        }
        else
        {
            Decimal::ZERO
        };
        accum += portion * rate;
    }
    let tax = check_size(
        "tax",
        accum.round_dp_with_strategy(MONEY_SCALE, RoundingStrategy::MidpointAwayFromZero),
    )?;
    Ok(tax)
}

/// The WRONG flat computation — top rate on the whole base. Provided only so the
/// equivalence test can pin the marginal-vs-flat divergence; never the tax.
pub fn flat_top_tax(base: Decimal) -> Decimal {
    (store_money_rounded(base) * top_rate())
        .round_dp_with_strategy(MONEY_SCALE, RoundingStrategy::MidpointAwayFromZero)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn d(s: &str) -> Decimal {
        Decimal::from_str(s).unwrap()
    }

    #[test]
    fn zero_and_first_bracket_are_untaxed() {
        assert_eq!(bracket_tax(d("0.00")).unwrap(), d("0.00"));
        assert_eq!(bracket_tax(d("5000.00")).unwrap(), d("0.00"));
        assert_eq!(bracket_tax(d("10000.00")).unwrap(), d("0.00"));
    }

    #[test]
    fn boundary_fills_lower_bracket_only() {
        // Exactly 40_000: bracket 2 fully (30_000 @ 10% = 3_000), bracket 3 empty.
        assert_eq!(bracket_tax(d("40000.00")).unwrap(), d("3000.00"));
    }

    #[test]
    fn marginal_slices_accumulate() {
        assert_eq!(bracket_tax(d("60000.00")).unwrap(), d("7400.00")); // 3000 + 4400
        assert_eq!(bracket_tax(d("200000.00")).unwrap(), d("43300.00"));
    }

    #[test]
    fn flat_would_be_wrong() {
        // The classic porting bug: top rate on the whole base.
        assert_eq!(flat_top_tax(d("100000.00")), d("32000.00"));
        assert!(bracket_tax(d("100000.00")).unwrap() < flat_top_tax(d("100000.00")));
    }

    #[test]
    fn single_rounding_event() {
        // (12345.67 - 10000) * 0.10 = 234.567 -> ROUNDED 234.57.
        assert_eq!(bracket_tax(d("12345.67")).unwrap(), d("234.57"));
    }

    #[test]
    fn negative_base_is_size_error() {
        let e = bracket_tax(d("-1.00")).unwrap_err();
        assert!(matches!(
            e,
            crate::AccrualError::SizeError { field: "base", .. }
        ));
    }
}
