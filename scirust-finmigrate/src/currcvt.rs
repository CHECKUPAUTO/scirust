//! CURRCVT migration unit — national→national euro conversion by triangulation.
//!
//! Port of `cobol/CURRCVT.cbl`; contract in `cobol/SEMANTICS_CURR.md`. Reproduces
//! the legally-defined algorithm of Council Regulation (EC) No 1103/97:
//! route through the euro (never a direct cross-rate), round the intermediate
//! euro to ≥ 3 dp, then round to the **target currency's minor unit** (2 dp for
//! most, 0 dp for ITL/ESP). A naive `amount × rate_to / rate_from` port is
//! unlawful and can differ by a minor unit — the sandbox records that divergence.

use crate::AccrualError;
use rust_decimal::{Decimal, RoundingStrategy};

const AWAY: RoundingStrategy = RoundingStrategy::MidpointAwayFromZero;
/// Intermediate euro precision — the legal minimum of 3 decimals (Gap-Q).
const EURO_SCALE: u32 = 3;

/// Result of a conversion: the target amount and the 3-dp euro intermediate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrResult {
    pub result: Decimal,
    pub euro: Decimal,
}

/// `(rate: 1 EUR = X national, six significant figures; minor-unit dp)`.
/// Rates are fixed/irrevocable and must never be rounded or truncated further.
fn currency(code: &str) -> Option<(Decimal, u32)> {
    let d = |s: &str| Decimal::from_str_exact(s).unwrap();
    Some(match code
    {
        "DEM" => (d("1.95583"), 2),
        "FRF" => (d("6.55957"), 2),
        "ITL" => (d("1936.27"), 0),
        "ESP" => (d("166.386"), 0),
        "IEP" => (d("0.787564"), 2),
        _ => return None,
    })
}

fn lookup(code: &str) -> Result<(Decimal, u32), AccrualError> {
    currency(code).ok_or_else(|| AccrualError::UnknownCurrency {
        code: code.to_string(),
    })
}

/// Convert `amount` from one national currency to another via the euro.
///
/// ```
/// use rust_decimal::Decimal;
/// use std::str::FromStr;
/// use scirust_finmigrate::currcvt::convert;
/// let d = |s: &str| Decimal::from_str(s).unwrap();
/// // 100 DEM -> 51.129 EUR -> 335.38 FRF (lawful); a direct cross-rate gives 335.39.
/// let c = convert(d("100.00"), "DEM", "FRF").unwrap();
/// assert_eq!(c.euro, d("51.129"));
/// assert_eq!(c.result, d("335.38"));
/// ```
pub fn convert(amount: Decimal, from: &str, to: &str) -> Result<CurrResult, AccrualError> {
    let (rate_from, _minor_from) = lookup(from)?;
    let (rate_to, minor_to) = lookup(to)?;

    // Step 1: source -> euro, ROUNDED to the >=3 dp intermediate (once).
    let euro = (amount / rate_from).round_dp_with_strategy(EURO_SCALE, AWAY);
    // Step 2: euro -> target, ROUNDED to the target's minor unit (once).
    let result = (euro * rate_to).round_dp_with_strategy(minor_to, AWAY);
    Ok(CurrResult { result, euro })
}

/// The UNLAWFUL direct cross-rate `amount × rate_to / rate_from`, rounded once to
/// the target minor unit. Provided only so the equivalence test can pin the
/// divergence from lawful triangulation; never use it for a real conversion.
pub fn direct_convert(amount: Decimal, from: &str, to: &str) -> Result<Decimal, AccrualError> {
    let (rate_from, _) = lookup(from)?;
    let (rate_to, minor_to) = lookup(to)?;
    Ok((amount * rate_to / rate_from).round_dp_with_strategy(minor_to, AWAY))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn d(s: &str) -> Decimal {
        Decimal::from_str(s).unwrap()
    }

    #[test]
    fn triangulation_differs_from_direct() {
        // Lawful 335.38 vs unlawful direct 335.39 — a one-centime legal divergence.
        let c = convert(d("100.00"), "DEM", "FRF").unwrap();
        assert_eq!(c.euro, d("51.129"));
        assert_eq!(c.result, d("335.38"));
        assert_eq!(
            direct_convert(d("100.00"), "DEM", "FRF").unwrap(),
            d("335.39")
        );
    }

    #[test]
    fn target_minor_unit_zero_for_lira() {
        // FRF -> ITL rounds to whole lira (0 dp).
        let c = convert(d("1000.00"), "FRF", "ITL").unwrap();
        assert_eq!(c.result, d("295182"));
        assert_eq!(c.result.scale(), 0);
    }

    #[test]
    fn rate_below_one_source() {
        let c = convert(d("100.00"), "IEP", "DEM").unwrap();
        assert_eq!(c.euro, d("126.974"));
        assert_eq!(c.result, d("248.34"));
    }

    #[test]
    fn unknown_currency_errors() {
        assert!(matches!(
            convert(d("1.00"), "USD", "DEM").unwrap_err(),
            AccrualError::UnknownCurrency { .. }
        ));
    }
}
