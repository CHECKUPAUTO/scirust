//! DAYCOUNT migration unit — accrued interest on the 30/360 US (NASD) convention.
//!
//! Port of `cobol/DAYCOUNT.cbl`; contract in `cobol/SEMANTICS_DAY.md`. The
//! arithmetic is trivial; the risk is the day count. "30/360 US" is ambiguous:
//! the SIFMA/NASD **bond basis** applies February end-of-month rules, Excel
//! `DAYS360` does not, and the two disagree by up to several days around
//! Feb/31st month-ends. This port implements the **NASD bond basis**; the
//! divergence from Excel is documented and cross-checked in the sandbox.

use crate::{check_size, store_money_rounded};
use rust_decimal::{Decimal, RoundingStrategy};

/// Fractional digits of the `PIC SV9(7)` annual-rate field.
pub const DAY_RATE_SCALE: u32 = 7;

/// A calendar date. Assumed valid; the port does not validate the Gregorian
/// calendar (the legacy `9(4)/9(2)/9(2)` fields carry whatever was stored).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Date {
    pub year: i32,
    pub month: u32,
    pub day: u32,
}

/// Day count plus the interest accrued over it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DayInterest {
    pub days: i64,
    pub interest: Decimal,
}

/// Gregorian leap year: divisible by 4, except centuries unless divisible by 400.
fn is_leap(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

/// The last calendar day of February for `year` (29 in a leap year, else 28).
fn last_day_of_feb(year: i32) -> u32 {
    if is_leap(year) { 29 } else { 28 }
}

fn is_last_of_feb(d: Date) -> bool {
    d.month == 2 && d.day == last_day_of_feb(d.year)
}

/// 30/360 US (NASD bond basis) day count. Rules applied in the exact order of
/// `cobol/SEMANTICS_DAY.md`, with the February flags read from the original
/// dates (rule 3 tests `30 or 31` to catch a not-yet-reduced `D1 = 31`).
pub fn days_30_360_us(d1: Date, d2: Date) -> i64 {
    let (mut ad1, mut ad2) = (d1.day, d2.day);
    let feb1 = is_last_of_feb(d1);
    let feb2 = is_last_of_feb(d2);

    if feb1 && feb2
    {
        ad2 = 30; // rule 1
    }
    if feb1
    {
        ad1 = 30; // rule 2
    }
    if ad2 == 31 && (ad1 == 30 || ad1 == 31)
    {
        ad2 = 30; // rule 3
    }
    if ad1 == 31
    {
        ad1 = 30; // rule 4
    }

    360 * (d2.year as i64 - d1.year as i64)
        + 30 * (d2.month as i64 - d1.month as i64)
        + (ad2 as i64 - ad1 as i64)
}

/// Accrue interest over `[d1, d2)` on the 30/360 US convention:
/// `principal · rate · days / 360`, ROUNDED once (NEAREST-AWAY-FROM-ZERO).
///
/// ```
/// use rust_decimal::Decimal;
/// use std::str::FromStr;
/// use scirust_finmigrate::daycount::{accrue_30_360, Date};
/// let d = |s: &str| Decimal::from_str(s).unwrap();
/// // 28-Feb-2023 -> 31-Aug-2023 is 180 days on the NASD basis (183 in Excel).
/// let a = accrue_30_360(d("100000.00"), d("0.0500000"),
///     Date { year: 2023, month: 2, day: 28 },
///     Date { year: 2023, month: 8, day: 31 }).unwrap();
/// assert_eq!(a.days, 180);
/// assert_eq!(a.interest, d("2500.00"));
/// ```
pub fn accrue_30_360(
    principal: Decimal,
    annual_rate: Decimal,
    d1: Date,
    d2: Date,
) -> Result<DayInterest, crate::AccrualError> {
    let principal = check_size("principal", store_money_rounded(principal))?;
    let rate =
        annual_rate.round_dp_with_strategy(DAY_RATE_SCALE, RoundingStrategy::MidpointAwayFromZero);
    let days = days_30_360_us(d1, d2);

    // principal * rate * days is exact; / 360 carried at full precision; ONE
    // rounding event into the 2 dp field (INTACCR Gap-4 discipline).
    let raw = (principal * rate * Decimal::from(days)) / Decimal::from(360);
    let interest = check_size(
        "interest",
        raw.round_dp_with_strategy(crate::MONEY_SCALE, RoundingStrategy::MidpointAwayFromZero),
    )?;
    Ok(DayInterest { days, interest })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn d(s: &str) -> Decimal {
        Decimal::from_str(s).unwrap()
    }

    fn date(y: i32, m: u32, dd: u32) -> Date {
        Date {
            year: y,
            month: m,
            day: dd,
        }
    }

    #[test]
    fn leap_year_rule() {
        assert!(is_leap(2024)); // div by 4
        assert!(!is_leap(2023));
        assert!(!is_leap(1900)); // century, not div by 400
        assert!(is_leap(2000)); // div by 400
    }

    #[test]
    fn feb_eom_differs_from_excel() {
        // NASD applies the Feb rule -> 180; Excel DAYS360 would give 183.
        assert_eq!(days_30_360_us(date(2023, 2, 28), date(2023, 8, 31)), 180);
        // Leap-year last day of Feb -> still 180 (29 -> 30).
        assert_eq!(days_30_360_us(date(2024, 2, 29), date(2024, 8, 31)), 180);
        // Both last-day-of-Feb across a year -> exactly 360.
        assert_eq!(days_30_360_us(date(2024, 2, 29), date(2025, 2, 28)), 360);
    }

    #[test]
    fn thirty_first_rules() {
        assert_eq!(days_30_360_us(date(2023, 1, 31), date(2023, 3, 31)), 60); // both 31
        assert_eq!(days_30_360_us(date(2023, 1, 31), date(2023, 4, 30)), 90); // d1 only
        assert_eq!(days_30_360_us(date(2023, 1, 15), date(2023, 3, 31)), 76); // no reduction
    }

    #[test]
    fn boundaries() {
        assert_eq!(days_30_360_us(date(2023, 6, 15), date(2023, 6, 15)), 0);
        assert_eq!(days_30_360_us(date(2023, 1, 1), date(2024, 1, 1)), 360);
    }

    #[test]
    fn interest_single_rounding() {
        // 100000 * 0.05 * 76 / 360 = 1055.5555... -> ROUNDED 1055.56.
        let a = accrue_30_360(
            d("100000.00"),
            d("0.0500000"),
            date(2023, 1, 15),
            date(2023, 3, 31),
        )
        .unwrap();
        assert_eq!(a.days, 76);
        assert_eq!(a.interest, d("1055.56"));
    }
}
