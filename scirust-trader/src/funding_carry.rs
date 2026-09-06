//! Delta-neutral funding-carry scenario and stress analysis.
//!
//! A current funding rate is not a profit forecast. This module values a
//! declared spot/perpetual hedge across multiple caller-supplied funding and
//! terminal-basis scenarios. Entry depth, fees, borrow cost, re-hedging cost,
//! collateral, quote freshness and leg alignment are explicit.

use serde::{Deserialize, Serialize};

use crate::derivatives::{basis_bps, funding_payment, PerpSide};
use crate::orderbook::OrderBook;
use crate::orders::Side;

/// Delta-neutral hedge orientation used to collect a particular funding sign.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FundingCarryDirection {
    /// Buy spot and short the perpetual. Positive funding credits the perp leg.
    LongSpotShortPerp,
    /// Short spot and buy the perpetual. Negative funding credits the perp leg.
    ShortSpotLongPerp,
}

/// One explicit funding/basis stress scenario.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FundingCarryScenario {
    pub name: String,
    /// Funding rate per event as a decimal fraction. This may be positive,
    /// negative or zero and is a scenario input, not a forecast.
    pub funding_rate_per_event: f32,
    pub funding_events: u32,
    /// Explicit terminal prices used to value the hedge unwind.
    pub terminal_spot_price: f32,
    pub terminal_perp_price: f32,
    /// Re-hedging cost in bps of the combined terminal spot+perp notional.
    pub rehedge_cost_bps: f32,
}

/// Common assumptions shared by every scenario in one carry analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FundingCarryConfig {
    pub symbol: String,
    pub base_size: f32,
    pub direction: FundingCarryDirection,
    pub spot_fee_bps: f32,
    pub perp_fee_bps: f32,
    /// Borrow cost for the complete holding horizon, in bps of entry spot
    /// notional. Applied only when the spot hedge is short.
    pub spot_borrow_cost_bps_for_horizon: f32,
    pub additional_cost_quote: f32,
    /// Declared perpetual leverage used only to report required collateral.
    pub perp_leverage: f32,
    pub available_perp_collateral_quote: Option<f32>,
    /// Minimum strictly-exceeded net scenario PnL needed for a positive label.
    /// Must be non-negative.
    pub min_net_profit_quote: f32,
    pub now_ts_ms: i64,
    pub max_quote_age_ms: u64,
    pub max_leg_skew_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FundingCarryConstraint {
    SpotDepthInsufficient,
    PerpDepthInsufficient,
    SpotQuoteFromFuture,
    PerpQuoteFromFuture,
    SpotQuoteStale,
    PerpQuoteStale,
    LegTimestampSkew,
    PerpCollateralUnverified,
    InsufficientPerpCollateral,
}

/// Result for one funding/basis stress scenario.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FundingCarryScenarioResult {
    pub name: String,
    pub funding_rate_per_event: f32,
    pub funding_events: u32,
    pub terminal_spot_price: f32,
    pub terminal_perp_price: f32,
    pub terminal_basis_bps: f32,
    /// Terminal basis minus entry basis.
    pub basis_drift_bps: f32,
    pub spot_leg_gross_pnl_quote: f32,
    pub perp_leg_gross_pnl_quote: f32,
    pub funding_pnl_quote: f32,
    pub entry_fees_quote: f32,
    pub exit_fees_quote: f32,
    pub borrow_cost_quote: f32,
    pub rehedge_cost_quote: f32,
    pub additional_cost_quote: f32,
    pub net_pnl_quote: f32,
    /// True only when the common entry constraints are not blocking and this
    /// scenario strictly exceeds the configured minimum net PnL.
    pub scenario_positive: bool,
}

/// Multi-scenario funding-carry report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FundingCarryReport {
    pub symbol: String,
    pub direction: FundingCarryDirection,
    pub base_size: f32,
    pub spot_ts_ms: i64,
    pub perp_ts_ms: i64,
    pub timestamp_skew_ms: u64,
    pub spot_filled_base: f32,
    pub perp_filled_base: f32,
    pub spot_entry_vwap: f32,
    pub perp_entry_vwap: f32,
    pub spot_entry_slippage_bps: f32,
    pub perp_entry_slippage_bps: f32,
    pub entry_basis_bps: f32,
    pub perp_leverage: f32,
    pub required_perp_collateral_quote: f32,
    pub constraints: Vec<FundingCarryConstraint>,
    pub scenarios: Vec<FundingCarryScenarioResult>,
    pub positive_scenarios: usize,
    pub all_scenarios_positive: bool,
    pub worst_net_pnl_quote: f32,
    pub best_net_pnl_quote: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FundingCarryError {
    InvalidBaseSize,
    InvalidSpotFee,
    InvalidPerpFee,
    InvalidBorrowCost,
    InvalidAdditionalCost,
    InvalidLeverage,
    InvalidProfitThreshold,
    EmptyScenarios,
    InvalidScenarioFundingRate,
    InvalidScenarioTerminalSpotPrice,
    InvalidScenarioTerminalPerpPrice,
    InvalidScenarioRehedgeCost,
}

fn valid_non_negative(value: f32) -> bool {
    value.is_finite() && value >= 0.0
}

fn timestamp_constraints(
    ts_ms: i64,
    now_ts_ms: i64,
    max_age_ms: u64,
    from_future: FundingCarryConstraint,
    stale: FundingCarryConstraint,
    constraints: &mut Vec<FundingCarryConstraint>,
) {
    if ts_ms > now_ts_ms
    {
        constraints.push(from_future);
    }
    else if now_ts_ms.abs_diff(ts_ms) > max_age_ms
    {
        constraints.push(stale);
    }
}

fn validate_scenario(scenario: &FundingCarryScenario) -> Result<(), FundingCarryError> {
    if !scenario.funding_rate_per_event.is_finite()
    {
        return Err(FundingCarryError::InvalidScenarioFundingRate);
    }
    if !scenario.terminal_spot_price.is_finite() || scenario.terminal_spot_price <= 0.0
    {
        return Err(FundingCarryError::InvalidScenarioTerminalSpotPrice);
    }
    if !scenario.terminal_perp_price.is_finite() || scenario.terminal_perp_price <= 0.0
    {
        return Err(FundingCarryError::InvalidScenarioTerminalPerpPrice);
    }
    if !valid_non_negative(scenario.rehedge_cost_bps)
    {
        return Err(FundingCarryError::InvalidScenarioRehedgeCost);
    }
    Ok(())
}

/// Analyze a funding-carry hedge over several explicit stress scenarios.
///
/// No scenario is inferred from the current market and no funding or basis
/// convergence is assumed. The function is pure and never submits orders.
pub fn analyze_funding_carry(
    spot_book: &OrderBook,
    perp_book: &OrderBook,
    cfg: &FundingCarryConfig,
    scenarios: &[FundingCarryScenario],
) -> Result<FundingCarryReport, FundingCarryError> {
    if !cfg.base_size.is_finite() || cfg.base_size <= 0.0
    {
        return Err(FundingCarryError::InvalidBaseSize);
    }
    if !valid_non_negative(cfg.spot_fee_bps)
    {
        return Err(FundingCarryError::InvalidSpotFee);
    }
    if !valid_non_negative(cfg.perp_fee_bps)
    {
        return Err(FundingCarryError::InvalidPerpFee);
    }
    if !valid_non_negative(cfg.spot_borrow_cost_bps_for_horizon)
    {
        return Err(FundingCarryError::InvalidBorrowCost);
    }
    if !valid_non_negative(cfg.additional_cost_quote)
    {
        return Err(FundingCarryError::InvalidAdditionalCost);
    }
    if !cfg.perp_leverage.is_finite() || cfg.perp_leverage <= 0.0
    {
        return Err(FundingCarryError::InvalidLeverage);
    }
    if !valid_non_negative(cfg.min_net_profit_quote)
    {
        return Err(FundingCarryError::InvalidProfitThreshold);
    }
    if scenarios.is_empty()
    {
        return Err(FundingCarryError::EmptyScenarios);
    }
    for scenario in scenarios
    {
        validate_scenario(scenario)?;
    }

    let (spot_side, perp_side, funding_side) = match cfg.direction
    {
        FundingCarryDirection::LongSpotShortPerp => (Side::Buy, Side::Sell, PerpSide::Short),
        FundingCarryDirection::ShortSpotLongPerp => (Side::Sell, Side::Buy, PerpSide::Long),
    };

    let spot_fill = spot_book.vwap_to_fill(spot_side, cfg.base_size);
    let perp_fill = perp_book.vwap_to_fill(perp_side, cfg.base_size);
    let mut constraints = Vec::new();
    if !spot_fill.fully_filled
    {
        constraints.push(FundingCarryConstraint::SpotDepthInsufficient);
    }
    if !perp_fill.fully_filled
    {
        constraints.push(FundingCarryConstraint::PerpDepthInsufficient);
    }

    timestamp_constraints(
        spot_book.ts_ms,
        cfg.now_ts_ms,
        cfg.max_quote_age_ms,
        FundingCarryConstraint::SpotQuoteFromFuture,
        FundingCarryConstraint::SpotQuoteStale,
        &mut constraints,
    );
    timestamp_constraints(
        perp_book.ts_ms,
        cfg.now_ts_ms,
        cfg.max_quote_age_ms,
        FundingCarryConstraint::PerpQuoteFromFuture,
        FundingCarryConstraint::PerpQuoteStale,
        &mut constraints,
    );
    let timestamp_skew_ms = spot_book.ts_ms.abs_diff(perp_book.ts_ms);
    if timestamp_skew_ms > cfg.max_leg_skew_ms
    {
        constraints.push(FundingCarryConstraint::LegTimestampSkew);
    }

    let spot_entry_notional = spot_fill.vwap * cfg.base_size;
    let perp_entry_notional = perp_fill.vwap * cfg.base_size;
    let entry_basis_bps = basis_bps(perp_fill.vwap, spot_fill.vwap).unwrap_or(0.0);
    let required_perp_collateral_quote = perp_entry_notional / cfg.perp_leverage;
    match cfg.available_perp_collateral_quote
    {
        None => constraints.push(FundingCarryConstraint::PerpCollateralUnverified),
        Some(v) if !v.is_finite() || v < required_perp_collateral_quote =>
        {
            constraints.push(FundingCarryConstraint::InsufficientPerpCollateral);
        },
        Some(_) =>
        {},
    }

    let common_blocked = constraints.iter().any(|constraint| {
        matches!(
            constraint,
            FundingCarryConstraint::SpotDepthInsufficient
                | FundingCarryConstraint::PerpDepthInsufficient
                | FundingCarryConstraint::SpotQuoteFromFuture
                | FundingCarryConstraint::PerpQuoteFromFuture
                | FundingCarryConstraint::SpotQuoteStale
                | FundingCarryConstraint::PerpQuoteStale
                | FundingCarryConstraint::LegTimestampSkew
                | FundingCarryConstraint::InsufficientPerpCollateral
        )
    });

    let entry_fees_quote = spot_entry_notional * cfg.spot_fee_bps / 10_000.0
        + perp_entry_notional * cfg.perp_fee_bps / 10_000.0;
    let borrow_cost_quote = if cfg.direction == FundingCarryDirection::ShortSpotLongPerp
    {
        spot_entry_notional * cfg.spot_borrow_cost_bps_for_horizon / 10_000.0
    }
    else
    {
        0.0
    };

    let mut results = Vec::with_capacity(scenarios.len());
    for scenario in scenarios
    {
        let terminal_spot_notional = scenario.terminal_spot_price * cfg.base_size;
        let terminal_perp_notional = scenario.terminal_perp_price * cfg.base_size;
        let terminal_basis_bps =
            basis_bps(scenario.terminal_perp_price, scenario.terminal_spot_price).unwrap_or(0.0);
        let basis_drift_bps = terminal_basis_bps - entry_basis_bps;

        let spot_leg_gross_pnl_quote = match cfg.direction
        {
            FundingCarryDirection::LongSpotShortPerp =>
            {
                (scenario.terminal_spot_price - spot_fill.vwap) * cfg.base_size
            },
            FundingCarryDirection::ShortSpotLongPerp =>
            {
                (spot_fill.vwap - scenario.terminal_spot_price) * cfg.base_size
            },
        };
        let perp_leg_gross_pnl_quote = match cfg.direction
        {
            FundingCarryDirection::LongSpotShortPerp =>
            {
                (perp_fill.vwap - scenario.terminal_perp_price) * cfg.base_size
            },
            FundingCarryDirection::ShortSpotLongPerp =>
            {
                (scenario.terminal_perp_price - perp_fill.vwap) * cfg.base_size
            },
        };
        let funding_pnl_quote = funding_payment(
            perp_entry_notional,
            scenario.funding_rate_per_event,
            funding_side,
        ) * scenario.funding_events as f32;
        let exit_fees_quote = terminal_spot_notional * cfg.spot_fee_bps / 10_000.0
            + terminal_perp_notional * cfg.perp_fee_bps / 10_000.0;
        let rehedge_cost_quote = (terminal_spot_notional + terminal_perp_notional)
            * scenario.rehedge_cost_bps
            / 10_000.0;
        let net_pnl_quote = spot_leg_gross_pnl_quote
            + perp_leg_gross_pnl_quote
            + funding_pnl_quote
            - entry_fees_quote
            - exit_fees_quote
            - borrow_cost_quote
            - rehedge_cost_quote
            - cfg.additional_cost_quote;
        let scenario_positive = !common_blocked && net_pnl_quote > cfg.min_net_profit_quote;

        results.push(FundingCarryScenarioResult {
            name: scenario.name.clone(),
            funding_rate_per_event: scenario.funding_rate_per_event,
            funding_events: scenario.funding_events,
            terminal_spot_price: scenario.terminal_spot_price,
            terminal_perp_price: scenario.terminal_perp_price,
            terminal_basis_bps,
            basis_drift_bps,
            spot_leg_gross_pnl_quote,
            perp_leg_gross_pnl_quote,
            funding_pnl_quote,
            entry_fees_quote,
            exit_fees_quote,
            borrow_cost_quote,
            rehedge_cost_quote,
            additional_cost_quote: cfg.additional_cost_quote,
            net_pnl_quote,
            scenario_positive,
        });
    }

    let positive_scenarios = results.iter().filter(|result| result.scenario_positive).count();
    let all_scenarios_positive = positive_scenarios == results.len();
    let worst_net_pnl_quote = results
        .iter()
        .map(|result| result.net_pnl_quote)
        .fold(f32::INFINITY, f32::min);
    let best_net_pnl_quote = results
        .iter()
        .map(|result| result.net_pnl_quote)
        .fold(f32::NEG_INFINITY, f32::max);

    Ok(FundingCarryReport {
        symbol: cfg.symbol.clone(),
        direction: cfg.direction,
        base_size: cfg.base_size,
        spot_ts_ms: spot_book.ts_ms,
        perp_ts_ms: perp_book.ts_ms,
        timestamp_skew_ms,
        spot_filled_base: spot_fill.filled,
        perp_filled_base: perp_fill.filled,
        spot_entry_vwap: spot_fill.vwap,
        perp_entry_vwap: perp_fill.vwap,
        spot_entry_slippage_bps: spot_fill.slippage_bps,
        perp_entry_slippage_bps: perp_fill.slippage_bps,
        entry_basis_bps,
        perp_leverage: cfg.perp_leverage,
        required_perp_collateral_quote,
        constraints,
        scenarios: results,
        positive_scenarios,
        all_scenarios_positive,
        worst_net_pnl_quote,
        best_net_pnl_quote,
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

    fn cfg(direction: FundingCarryDirection) -> FundingCarryConfig {
        FundingCarryConfig {
            symbol: "BTCUSDT".to_string(),
            base_size: 1.0,
            direction,
            spot_fee_bps: 5.0,
            perp_fee_bps: 5.0,
            spot_borrow_cost_bps_for_horizon: 20.0,
            additional_cost_quote: 0.0,
            perp_leverage: 2.0,
            available_perp_collateral_quote: Some(1_000.0),
            min_net_profit_quote: 0.0,
            now_ts_ms: 1_000,
            max_quote_age_ms: 100,
            max_leg_skew_ms: 25,
        }
    }

    fn scenario(name: &str, funding: f32, spot: f32, perp: f32) -> FundingCarryScenario {
        FundingCarryScenario {
            name: name.to_string(),
            funding_rate_per_event: funding,
            funding_events: 2,
            terminal_spot_price: spot,
            terminal_perp_price: perp,
            rehedge_cost_bps: 0.0,
        }
    }

    #[test]
    fn positive_funding_credits_long_spot_short_perp() {
        let spot = book(990, 99.0, 100.0);
        let perp = book(995, 101.0, 102.0);
        let scenarios = [scenario("base", 0.001, 100.5, 100.5)];
        let report = analyze_funding_carry(
            &spot,
            &perp,
            &cfg(FundingCarryDirection::LongSpotShortPerp),
            &scenarios,
        )
        .unwrap();
        assert!(report.scenarios[0].funding_pnl_quote > 0.0);
        assert!(report.scenarios[0].scenario_positive);
    }

    #[test]
    fn negative_funding_credits_short_spot_long_perp() {
        let spot = book(990, 100.0, 101.0);
        let perp = book(995, 98.0, 99.0);
        let scenarios = [scenario("base", -0.001, 99.5, 99.5)];
        let report = analyze_funding_carry(
            &spot,
            &perp,
            &cfg(FundingCarryDirection::ShortSpotLongPerp),
            &scenarios,
        )
        .unwrap();
        assert!(report.scenarios[0].funding_pnl_quote > 0.0);
        assert!(report.scenarios[0].borrow_cost_quote > 0.0);
    }

    #[test]
    fn funding_sign_reversal_is_visible_as_an_adverse_scenario() {
        let spot = book(990, 99.0, 100.0);
        let perp = book(995, 101.0, 102.0);
        let scenarios = [
            scenario("base", 0.001, 100.5, 100.5),
            scenario("funding-reversal", -0.01, 100.5, 100.5),
        ];
        let report = analyze_funding_carry(
            &spot,
            &perp,
            &cfg(FundingCarryDirection::LongSpotShortPerp),
            &scenarios,
        )
        .unwrap();
        assert!(report.scenarios[0].funding_pnl_quote > 0.0);
        assert!(report.scenarios[1].funding_pnl_quote < 0.0);
        assert!(report.best_net_pnl_quote > report.worst_net_pnl_quote);
        assert!(!report.all_scenarios_positive);
    }

    #[test]
    fn adverse_basis_drift_can_erase_carry() {
        let spot = book(990, 99.0, 100.0);
        let perp = book(995, 101.0, 102.0);
        let scenarios = [scenario("basis-widens", 0.0001, 100.0, 105.0)];
        let report = analyze_funding_carry(
            &spot,
            &perp,
            &cfg(FundingCarryDirection::LongSpotShortPerp),
            &scenarios,
        )
        .unwrap();
        assert!(report.scenarios[0].basis_drift_bps > 0.0);
        assert!(report.scenarios[0].net_pnl_quote < 0.0);
        assert!(!report.scenarios[0].scenario_positive);
    }

    #[test]
    fn rehedging_cost_is_explicit() {
        let spot = book(990, 99.0, 100.0);
        let perp = book(995, 101.0, 102.0);
        let mut stressed = scenario("rehedge", 0.001, 100.5, 100.5);
        stressed.rehedge_cost_bps = 50.0;
        let baseline = scenario("base", 0.001, 100.5, 100.5);
        let report = analyze_funding_carry(
            &spot,
            &perp,
            &cfg(FundingCarryDirection::LongSpotShortPerp),
            &[baseline, stressed],
        )
        .unwrap();
        assert!(report.scenarios[1].rehedge_cost_quote > 0.0);
        assert!(report.scenarios[1].net_pnl_quote < report.scenarios[0].net_pnl_quote);
    }

    #[test]
    fn stale_or_skewed_books_block_every_scenario() {
        let spot = book(700, 99.0, 100.0);
        let perp = book(995, 101.0, 102.0);
        let scenarios = [scenario("base", 0.001, 100.5, 100.5)];
        let report = analyze_funding_carry(
            &spot,
            &perp,
            &cfg(FundingCarryDirection::LongSpotShortPerp),
            &scenarios,
        )
        .unwrap();
        assert!(report
            .constraints
            .contains(&FundingCarryConstraint::SpotQuoteStale));
        assert!(report
            .constraints
            .contains(&FundingCarryConstraint::LegTimestampSkew));
        assert_eq!(report.positive_scenarios, 0);
    }

    #[test]
    fn empty_scenario_set_is_rejected() {
        let spot = book(990, 99.0, 100.0);
        let perp = book(995, 101.0, 102.0);
        assert_eq!(
            analyze_funding_carry(
                &spot,
                &perp,
                &cfg(FundingCarryDirection::LongSpotShortPerp),
                &[],
            )
            .unwrap_err(),
            FundingCarryError::EmptyScenarios
        );
    }

    #[test]
    fn repeated_analysis_is_bit_reproducible() {
        let spot = book(990, 99.0, 100.0);
        let perp = book(995, 101.0, 102.0);
        let scenarios = [scenario("base", 0.001, 100.5, 100.5)];
        let config = cfg(FundingCarryDirection::LongSpotShortPerp);
        let a = analyze_funding_carry(&spot, &perp, &config, &scenarios).unwrap();
        let b = analyze_funding_carry(&spot, &perp, &config, &scenarios).unwrap();
        assert_eq!(a, b);
    }
}
