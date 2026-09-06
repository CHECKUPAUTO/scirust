//! Deterministic crypto-derivatives analytics.
//!
//! This module models the market-state primitives that are specific to
//! perpetual futures and other crypto derivatives: spot/perpetual basis,
//! funding carry, open-interest change, and liquidation imbalance.
//!
//! The layer is deliberately read-only and simulation-first. It does not place
//! orders, connect to an exchange, or infer profitability. Its outputs are
//! auditable scalars that strategies, scanners, backtests, and MCP agents can
//! consume alongside the existing order-book and regime analytics.

use serde::{Deserialize, Serialize};

/// Which side transfers funding when the exchange funding rate is non-zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FundingBias {
    /// Positive funding: longs pay shorts.
    LongsPayShorts,
    /// Negative funding: shorts pay longs.
    ShortsPayLongs,
    /// Exactly zero funding.
    Neutral,
}

impl FundingBias {
    pub fn from_rate(rate: f32) -> Self {
        if rate > 0.0 {
            Self::LongsPayShorts
        } else if rate < 0.0 {
            Self::ShortsPayLongs
        } else {
            Self::Neutral
        }
    }
}

/// Position side used only for deterministic funding-cash-flow accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerpSide {
    Long,
    Short,
}

/// One aligned derivatives snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DerivativesSnapshot {
    pub symbol: String,
    pub ts_ms: i64,
    /// Spot reference price for the same underlying.
    pub spot_price: f32,
    /// Perpetual mark price.
    pub mark_price: f32,
    /// Exchange index price, when available. Use the spot reference if an
    /// exchange does not expose a distinct index.
    pub index_price: f32,
    /// Funding rate as a decimal fraction per funding interval (0.0001 = 1 bp).
    pub funding_rate: f32,
    /// Funding interval in hours (for example 8.0). Must be positive for
    /// annualization.
    pub funding_interval_hours: f32,
    /// Current open interest in a caller-defined, but internally consistent,
    /// unit (contracts, base units, or notional).
    pub open_interest: f32,
    /// Long-side liquidations observed over the caller's aggregation window.
    pub long_liquidations: f32,
    /// Short-side liquidations observed over the caller's aggregation window.
    pub short_liquidations: f32,
}

/// Open-interest change between two aligned snapshots.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct OpenInterestChange {
    pub absolute: f32,
    pub percent: f32,
}

/// Flat, serializable derivatives-state report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DerivativesReport {
    pub symbol: String,
    pub ts_ms: i64,
    /// `(mark - spot) / spot * 10_000`.
    pub spot_perp_basis_bps: f32,
    /// `(mark - index) / index * 10_000`.
    pub mark_index_basis_bps: f32,
    /// Funding rate for one interval in basis points.
    pub funding_interval_bps: f32,
    /// Simple annualization of the current funding rate, in percent per year.
    /// This is a state normalization, not a forecast and not compounded.
    pub annualized_funding_pct: f32,
    pub funding_bias: FundingBias,
    /// `(short_liq - long_liq) / (short_liq + long_liq)` in `[-1, 1]`.
    /// Positive means short liquidations dominate; negative means long
    /// liquidations dominate.
    pub liquidation_imbalance: f32,
    /// Change from the previous snapshot, if one was provided with positive
    /// previous open interest.
    pub open_interest_change: Option<OpenInterestChange>,
}

/// Basis in basis points. Returns `None` for a zero/non-finite reference price
/// or non-finite derivative price.
pub fn basis_bps(derivative_price: f32, reference_price: f32) -> Option<f32> {
    if !derivative_price.is_finite()
        || !reference_price.is_finite()
        || reference_price.abs() <= f32::EPSILON
    {
        return None;
    }
    Some((derivative_price - reference_price) / reference_price * 10_000.0)
}

/// Funding rate for one interval, expressed in basis points.
pub fn funding_interval_bps(funding_rate: f32) -> f32 {
    if funding_rate.is_finite() {
        funding_rate * 10_000.0
    } else {
        0.0
    }
}

/// Simple annualization of a funding rate, in percentage points per year.
///
/// `funding_rate` is a decimal fraction per interval. For example, 0.0001 every
/// 8 hours annualizes to `10.95%` with the simple-rate convention:
/// `rate * (24 / interval_hours) * 365 * 100`.
///
/// Returns `None` for a non-positive/non-finite interval or non-finite rate.
pub fn annualized_funding_pct(funding_rate: f32, interval_hours: f32) -> Option<f32> {
    if !funding_rate.is_finite()
        || !interval_hours.is_finite()
        || interval_hours <= 0.0
    {
        return None;
    }
    Some(funding_rate * (24.0 / interval_hours) * 365.0 * 100.0)
}

/// Signed funding cash flow to a position for one funding event.
///
/// Exchange convention used here: positive funding means longs pay shorts;
/// negative funding means shorts pay longs. `notional` is treated as an
/// absolute amount, so callers cannot accidentally invert the sign twice.
pub fn funding_payment(notional: f32, funding_rate: f32, side: PerpSide) -> f32 {
    if !notional.is_finite() || !funding_rate.is_finite() {
        return 0.0;
    }
    let payment_to_short = notional.abs() * funding_rate;
    match side {
        PerpSide::Long => -payment_to_short,
        PerpSide::Short => payment_to_short,
    }
}

/// Open-interest change from `previous` to `current`.
///
/// Returns `None` when either value is non-finite or the previous value is not
/// strictly positive, because a percentage change would be undefined or not
/// economically interpretable.
pub fn open_interest_change(previous: f32, current: f32) -> Option<OpenInterestChange> {
    if !previous.is_finite() || !current.is_finite() || previous <= 0.0 {
        return None;
    }
    let absolute = current - previous;
    Some(OpenInterestChange {
        absolute,
        percent: absolute / previous * 100.0,
    })
}

/// Liquidation imbalance in `[-1, 1]`.
///
/// Positive values mean short liquidations dominate the aggregation window;
/// negative values mean long liquidations dominate. Invalid/negative inputs are
/// clamped to zero because liquidation amounts cannot be negative quantities.
pub fn liquidation_imbalance(long_liquidations: f32, short_liquidations: f32) -> f32 {
    let long = if long_liquidations.is_finite() {
        long_liquidations.max(0.0)
    } else {
        0.0
    };
    let short = if short_liquidations.is_finite() {
        short_liquidations.max(0.0)
    } else {
        0.0
    };
    let total = long + short;
    if total <= f32::EPSILON {
        0.0
    } else {
        ((short - long) / total).clamp(-1.0, 1.0)
    }
}

/// Build a compact deterministic report for a derivatives snapshot.
///
/// Invalid basis references are represented as `0.0` in the flat report; the
/// lower-level [`basis_bps`] function remains available when callers need to
/// distinguish missing/invalid references explicitly.
pub fn analyze(
    current: &DerivativesSnapshot,
    previous: Option<&DerivativesSnapshot>,
) -> DerivativesReport {
    let spot_perp_basis_bps = basis_bps(current.mark_price, current.spot_price).unwrap_or(0.0);
    let mark_index_basis_bps = basis_bps(current.mark_price, current.index_price).unwrap_or(0.0);
    let annualized_funding_pct =
        annualized_funding_pct(current.funding_rate, current.funding_interval_hours).unwrap_or(0.0);
    let open_interest_change = previous.and_then(|prev| {
        open_interest_change(prev.open_interest, current.open_interest)
    });

    DerivativesReport {
        symbol: current.symbol.clone(),
        ts_ms: current.ts_ms,
        spot_perp_basis_bps,
        mark_index_basis_bps,
        funding_interval_bps: funding_interval_bps(current.funding_rate),
        annualized_funding_pct,
        funding_bias: FundingBias::from_rate(current.funding_rate),
        liquidation_imbalance: liquidation_imbalance(
            current.long_liquidations,
            current.short_liquidations,
        ),
        open_interest_change,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot() -> DerivativesSnapshot {
        DerivativesSnapshot {
            symbol: "BTCUSDT".to_string(),
            ts_ms: 1_000,
            spot_price: 100.0,
            mark_price: 101.0,
            index_price: 100.5,
            funding_rate: 0.0001,
            funding_interval_hours: 8.0,
            open_interest: 120.0,
            long_liquidations: 200.0,
            short_liquidations: 600.0,
        }
    }

    #[test]
    fn basis_reports_premium_in_bps() {
        let b = basis_bps(101.0, 100.0).unwrap();
        assert!((b - 100.0).abs() < 1e-4);
        assert!(basis_bps(101.0, 0.0).is_none());
    }

    #[test]
    fn funding_direction_and_cash_flow_follow_exchange_convention() {
        assert_eq!(FundingBias::from_rate(0.0001), FundingBias::LongsPayShorts);
        assert_eq!(FundingBias::from_rate(-0.0001), FundingBias::ShortsPayLongs);
        assert!((funding_payment(10_000.0, 0.001, PerpSide::Long) + 10.0).abs() < 1e-5);
        assert!((funding_payment(10_000.0, 0.001, PerpSide::Short) - 10.0).abs() < 1e-5);
    }

    #[test]
    fn funding_simple_annualization_is_explicit() {
        let pct = annualized_funding_pct(0.0001, 8.0).unwrap();
        assert!((pct - 10.95).abs() < 1e-4);
        assert!(annualized_funding_pct(0.0001, 0.0).is_none());
    }

    #[test]
    fn open_interest_change_tracks_level_and_percent() {
        let change = open_interest_change(100.0, 120.0).unwrap();
        assert!((change.absolute - 20.0).abs() < 1e-6);
        assert!((change.percent - 20.0).abs() < 1e-6);
        assert!(open_interest_change(0.0, 120.0).is_none());
    }

    #[test]
    fn liquidation_imbalance_is_signed_and_bounded() {
        assert!((liquidation_imbalance(200.0, 600.0) - 0.5).abs() < 1e-6);
        assert!((liquidation_imbalance(600.0, 200.0) + 0.5).abs() < 1e-6);
        assert_eq!(liquidation_imbalance(0.0, 0.0), 0.0);
    }

    #[test]
    fn analyze_combines_derivatives_state_deterministically() {
        let current = snapshot();
        let mut previous = current.clone();
        previous.ts_ms = 0;
        previous.open_interest = 100.0;

        let report = analyze(&current, Some(&previous));
        assert_eq!(report.symbol, "BTCUSDT");
        assert!((report.spot_perp_basis_bps - 100.0).abs() < 1e-4);
        assert!((report.funding_interval_bps - 1.0).abs() < 1e-6);
        assert!((report.annualized_funding_pct - 10.95).abs() < 1e-4);
        assert_eq!(report.funding_bias, FundingBias::LongsPayShorts);
        assert!((report.liquidation_imbalance - 0.5).abs() < 1e-6);
        assert!((report.open_interest_change.unwrap().percent - 20.0).abs() < 1e-6);
    }
}
