//! Spot/perpetual basis scenario analysis.
//!
//! Basis convergence is never assumed. The scanner builds a delta-hedged entry
//! from executable depth, charges explicit entry/exit fees and optional borrow
//! costs, applies a declared funding-rate scenario for an explicit number of
//! events, and values the unwind at caller-supplied terminal spot/perp prices.

use serde::{Deserialize, Serialize};

use crate::derivatives::{PerpSide, basis_bps, funding_payment};
use crate::orderbook::OrderBook;
use crate::orders::Side;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BasisDirection {
    /// Buy spot and short the perpetual.
    CashAndCarry,
    /// Short spot and buy the perpetual.
    ReverseCashAndCarry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisScenarioConfig {
    pub symbol: String,
    pub base_size: f32,
    pub direction: BasisDirection,
    pub spot_fee_bps: f32,
    pub perp_fee_bps: f32,
    /// Funding rate per event as a decimal fraction. This is an explicit
    /// scenario, not a forecast.
    pub funding_rate_per_event: f32,
    pub funding_events: u32,
    /// Borrow cost for the complete holding horizon, in bps of entry spot
    /// notional. Used only by reverse cash-and-carry.
    pub spot_borrow_cost_bps_for_horizon: f32,
    pub additional_cost_quote: f32,
    /// Holding horizon used only to report a simple annualized entry-basis
    /// normalization. It does not imply convergence within this horizon.
    pub holding_hours: f32,
    /// Explicit terminal scenario used to value the unwind.
    pub terminal_spot_price: f32,
    pub terminal_perp_price: f32,
    /// Declared perpetual leverage assumption for collateral reporting.
    pub perp_leverage: f32,
    /// Optional available collateral for the perpetual leg.
    pub available_perp_collateral_quote: Option<f32>,
    /// Minimum scenario net PnL required to mark the scenario economically
    /// positive. Must be non-negative.
    pub min_net_profit_quote: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BasisConstraint {
    SpotDepthInsufficient,
    PerpDepthInsufficient,
    PerpCollateralUnverified,
    InsufficientPerpCollateral,
    BelowScenarioProfitThreshold,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BasisScenarioReport {
    pub symbol: String,
    pub direction: BasisDirection,
    pub base_size: f32,
    pub spot_entry_vwap: f32,
    pub perp_entry_vwap: f32,
    pub spot_entry_slippage_bps: f32,
    pub perp_entry_slippage_bps: f32,
    pub entry_basis_bps: f32,
    /// Simple normalization: entry basis divided by the declared holding
    /// horizon and scaled to 365 days. This is not a return forecast.
    pub simple_entry_basis_annualized_pct: f32,
    pub terminal_spot_price: f32,
    pub terminal_perp_price: f32,
    pub terminal_basis_bps: f32,
    pub spot_leg_gross_pnl_quote: f32,
    pub perp_leg_gross_pnl_quote: f32,
    pub funding_pnl_quote: f32,
    pub entry_fees_quote: f32,
    pub exit_fees_quote: f32,
    pub borrow_cost_quote: f32,
    pub additional_cost_quote: f32,
    pub net_pnl_quote: f32,
    pub perp_leverage: f32,
    pub required_perp_collateral_quote: f32,
    pub scenario_positive: bool,
    pub constraints: Vec<BasisConstraint>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BasisError {
    InvalidBaseSize,
    InvalidSpotFee,
    InvalidPerpFee,
    InvalidFundingRate,
    InvalidBorrowCost,
    InvalidAdditionalCost,
    InvalidHoldingHours,
    InvalidTerminalSpotPrice,
    InvalidTerminalPerpPrice,
    InvalidLeverage,
    InvalidProfitThreshold,
}

/// Simple annualization of a basis observed over a declared horizon.
///
/// This is only a normalization convention. It must not be interpreted as a
/// forecast that the basis will converge over `holding_hours`.
pub fn annualize_basis_pct(entry_basis_bps: f32, holding_hours: f32) -> Option<f32> {
    if !entry_basis_bps.is_finite() || !holding_hours.is_finite() || holding_hours <= 0.0
    {
        return None;
    }
    Some(entry_basis_bps / 10_000.0 * (24.0 * 365.0 / holding_hours) * 100.0)
}

fn valid_non_negative(value: f32) -> bool {
    value.is_finite() && value >= 0.0
}

pub fn analyze_basis_scenario(
    spot_book: &OrderBook,
    perp_book: &OrderBook,
    cfg: &BasisScenarioConfig,
) -> Result<BasisScenarioReport, BasisError> {
    if !cfg.base_size.is_finite() || cfg.base_size <= 0.0
    {
        return Err(BasisError::InvalidBaseSize);
    }
    if !valid_non_negative(cfg.spot_fee_bps)
    {
        return Err(BasisError::InvalidSpotFee);
    }
    if !valid_non_negative(cfg.perp_fee_bps)
    {
        return Err(BasisError::InvalidPerpFee);
    }
    if !cfg.funding_rate_per_event.is_finite()
    {
        return Err(BasisError::InvalidFundingRate);
    }
    if !valid_non_negative(cfg.spot_borrow_cost_bps_for_horizon)
    {
        return Err(BasisError::InvalidBorrowCost);
    }
    if !valid_non_negative(cfg.additional_cost_quote)
    {
        return Err(BasisError::InvalidAdditionalCost);
    }
    if !cfg.holding_hours.is_finite() || cfg.holding_hours <= 0.0
    {
        return Err(BasisError::InvalidHoldingHours);
    }
    if !cfg.terminal_spot_price.is_finite() || cfg.terminal_spot_price <= 0.0
    {
        return Err(BasisError::InvalidTerminalSpotPrice);
    }
    if !cfg.terminal_perp_price.is_finite() || cfg.terminal_perp_price <= 0.0
    {
        return Err(BasisError::InvalidTerminalPerpPrice);
    }
    if !cfg.perp_leverage.is_finite() || cfg.perp_leverage <= 0.0
    {
        return Err(BasisError::InvalidLeverage);
    }
    if !valid_non_negative(cfg.min_net_profit_quote)
    {
        return Err(BasisError::InvalidProfitThreshold);
    }

    let (spot_side, perp_side, funding_side) = match cfg.direction
    {
        BasisDirection::CashAndCarry => (Side::Buy, Side::Sell, PerpSide::Short),
        BasisDirection::ReverseCashAndCarry => (Side::Sell, Side::Buy, PerpSide::Long),
    };
    let spot_fill = spot_book.vwap_to_fill(spot_side, cfg.base_size);
    let perp_fill = perp_book.vwap_to_fill(perp_side, cfg.base_size);
    let mut constraints = Vec::new();
    if !spot_fill.fully_filled
    {
        constraints.push(BasisConstraint::SpotDepthInsufficient);
    }
    if !perp_fill.fully_filled
    {
        constraints.push(BasisConstraint::PerpDepthInsufficient);
    }

    let entry_basis_bps = basis_bps(perp_fill.vwap, spot_fill.vwap).unwrap_or(0.0);
    let simple_entry_basis_annualized_pct =
        annualize_basis_pct(entry_basis_bps, cfg.holding_hours).unwrap_or(0.0);
    let terminal_basis_bps =
        basis_bps(cfg.terminal_perp_price, cfg.terminal_spot_price).unwrap_or(0.0);

    let spot_leg_gross_pnl_quote = match cfg.direction
    {
        BasisDirection::CashAndCarry =>
        {
            (cfg.terminal_spot_price - spot_fill.vwap) * cfg.base_size
        },
        BasisDirection::ReverseCashAndCarry =>
        {
            (spot_fill.vwap - cfg.terminal_spot_price) * cfg.base_size
        },
    };
    let perp_leg_gross_pnl_quote = match cfg.direction
    {
        BasisDirection::CashAndCarry =>
        {
            (perp_fill.vwap - cfg.terminal_perp_price) * cfg.base_size
        },
        BasisDirection::ReverseCashAndCarry =>
        {
            (cfg.terminal_perp_price - perp_fill.vwap) * cfg.base_size
        },
    };

    let spot_entry_notional = spot_fill.vwap * cfg.base_size;
    let perp_entry_notional = perp_fill.vwap * cfg.base_size;
    let terminal_spot_notional = cfg.terminal_spot_price * cfg.base_size;
    let terminal_perp_notional = cfg.terminal_perp_price * cfg.base_size;
    let entry_fees_quote = spot_entry_notional * cfg.spot_fee_bps / 10_000.0
        + perp_entry_notional * cfg.perp_fee_bps / 10_000.0;
    let exit_fees_quote = terminal_spot_notional * cfg.spot_fee_bps / 10_000.0
        + terminal_perp_notional * cfg.perp_fee_bps / 10_000.0;
    let funding_pnl_quote = funding_payment(
        perp_entry_notional,
        cfg.funding_rate_per_event,
        funding_side,
    ) * cfg.funding_events as f32;
    let borrow_cost_quote = if cfg.direction == BasisDirection::ReverseCashAndCarry
    {
        spot_entry_notional * cfg.spot_borrow_cost_bps_for_horizon / 10_000.0
    }
    else
    {
        0.0
    };
    let net_pnl_quote = spot_leg_gross_pnl_quote
        + perp_leg_gross_pnl_quote
        + funding_pnl_quote
        - entry_fees_quote
        - exit_fees_quote
        - borrow_cost_quote
        - cfg.additional_cost_quote;

    let required_perp_collateral_quote = perp_entry_notional / cfg.perp_leverage;
    match cfg.available_perp_collateral_quote
    {
        None => constraints.push(BasisConstraint::PerpCollateralUnverified),
        Some(v) if !v.is_finite() || v < required_perp_collateral_quote =>
        {
            constraints.push(BasisConstraint::InsufficientPerpCollateral);
        },
        Some(_) => {},
    }
    if net_pnl_quote <= cfg.min_net_profit_quote
    {
        constraints.push(BasisConstraint::BelowScenarioProfitThreshold);
    }

    let blocked = constraints.iter().any(|constraint| {
        matches!(
            constraint,
            BasisConstraint::SpotDepthInsufficient
                | BasisConstraint::PerpDepthInsufficient
                | BasisConstraint::InsufficientPerpCollateral
                | BasisConstraint::BelowScenarioProfitThreshold
        )
    });

    Ok(BasisScenarioReport {
        symbol: cfg.symbol.clone(),
        direction: cfg.direction,
        base_size: cfg.base_size,
        spot_entry_vwap: spot_fill.vwap,
        perp_entry_vwap: perp_fill.vwap,
        spot_entry_slippage_bps: spot_fill.slippage_bps,
        perp_entry_slippage_bps: perp_fill.slippage_bps,
        entry_basis_bps,
        simple_entry_basis_annualized_pct,
        terminal_spot_price: cfg.terminal_spot_price,
        terminal_perp_price: cfg.terminal_perp_price,
        terminal_basis_bps,
        spot_leg_gross_pnl_quote,
        perp_leg_gross_pnl_quote,
        funding_pnl_quote,
        entry_fees_quote,
        exit_fees_quote,
        borrow_cost_quote,
        additional_cost_quote: cfg.additional_cost_quote,
        net_pnl_quote,
        perp_leverage: cfg.perp_leverage,
        required_perp_collateral_quote,
        scenario_positive: !blocked,
        constraints,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orderbook::Level;

    fn book(ts: i64, bid: f32, ask: f32) -> OrderBook {
        OrderBook::new(
            "BTCUSDT",
            ts,
            vec![Level::new(bid, 10.0)],
            vec![Level::new(ask, 10.0)],
        )
    }

    fn cfg(direction: BasisDirection) -> BasisScenarioConfig {
        BasisScenarioConfig {
            symbol: "BTCUSDT".to_string(),
            base_size: 1.0,
            direction,
            spot_fee_bps: 5.0,
            perp_fee_bps: 5.0,
            funding_rate_per_event: 0.001,
            funding_events: 2,
            spot_borrow_cost_bps_for_horizon: 20.0,
            additional_cost_quote: 0.0,
            holding_hours: 24.0,
            terminal_spot_price: 101.0,
            terminal_perp_price: 101.0,
            perp_leverage: 2.0,
            available_perp_collateral_quote: Some(1_000.0),
            min_net_profit_quote: 0.0,
        }
    }

    #[test]
    fn cash_and_carry_values_explicit_convergence_scenario() {
        let spot = book(1, 99.0, 100.0);
        let perp = book(1, 103.0, 104.0);
        let report = analyze_basis_scenario(&spot, &perp, &cfg(BasisDirection::CashAndCarry)).unwrap();
        assert!((report.entry_basis_bps - 300.0).abs() < 1e-4);
        assert_eq!(report.terminal_basis_bps, 0.0);
        assert!(report.funding_pnl_quote > 0.0);
        assert_eq!(report.borrow_cost_quote, 0.0);
        assert!(report.net_pnl_quote > 0.0);
        assert!(report.scenario_positive);
    }

    #[test]
    fn reverse_carry_charges_spot_borrow_and_long_funding() {
        let spot = book(1, 100.0, 101.0);
        let perp = book(1, 97.0, 98.0);
        let report =
            analyze_basis_scenario(&spot, &perp, &cfg(BasisDirection::ReverseCashAndCarry)).unwrap();
        assert!(report.entry_basis_bps < 0.0);
        assert!(report.funding_pnl_quote < 0.0);
        assert!(report.borrow_cost_quote > 0.0);
    }

    #[test]
    fn terminal_basis_is_not_forced_to_zero() {
        let spot = book(1, 99.0, 100.0);
        let perp = book(1, 103.0, 104.0);
        let mut config = cfg(BasisDirection::CashAndCarry);
        config.terminal_spot_price = 100.0;
        config.terminal_perp_price = 102.0;
        let report = analyze_basis_scenario(&spot, &perp, &config).unwrap();
        assert!((report.terminal_basis_bps - 200.0).abs() < 1e-4);
        assert!(report.net_pnl_quote < 3.0);
    }

    #[test]
    fn insufficient_depth_blocks_positive_scenario() {
        let spot = OrderBook::new(
            "BTCUSDT",
            1,
            vec![Level::new(99.0, 10.0)],
            vec![Level::new(100.0, 0.1)],
        );
        let perp = book(1, 103.0, 104.0);
        let report = analyze_basis_scenario(&spot, &perp, &cfg(BasisDirection::CashAndCarry)).unwrap();
        assert!(!report.scenario_positive);
        assert!(report.constraints.contains(&BasisConstraint::SpotDepthInsufficient));
    }

    #[test]
    fn missing_collateral_is_reported_but_not_falsely_called_insufficient() {
        let spot = book(1, 99.0, 100.0);
        let perp = book(1, 103.0, 104.0);
        let mut config = cfg(BasisDirection::CashAndCarry);
        config.available_perp_collateral_quote = None;
        let report = analyze_basis_scenario(&spot, &perp, &config).unwrap();
        assert!(report.constraints.contains(&BasisConstraint::PerpCollateralUnverified));
        assert!(report.scenario_positive);
    }

    #[test]
    fn insufficient_collateral_blocks_scenario() {
        let spot = book(1, 99.0, 100.0);
        let perp = book(1, 103.0, 104.0);
        let mut config = cfg(BasisDirection::CashAndCarry);
        config.available_perp_collateral_quote = Some(1.0);
        let report = analyze_basis_scenario(&spot, &perp, &config).unwrap();
        assert!(!report.scenario_positive);
        assert!(report.constraints.contains(&BasisConstraint::InsufficientPerpCollateral));
    }

    #[test]
    fn annualization_is_only_a_normalization() {
        let annualized = annualize_basis_pct(100.0, 24.0).unwrap();
        assert!((annualized - 365.0).abs() < 1e-4);
    }
}
