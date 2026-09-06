//! Venue-neutral cross-market arbitrage analysis.
//!
//! Gross top-of-book spread is not enough to call an opportunity profitable.
//! This module walks both books for the requested base size, converts both quote
//! currencies into one common currency, subtracts explicit taker fees and fixed
//! transaction costs, validates timestamps, and keeps balance/inventory
//! verification separate from pure market-edge detection.

use serde::{Deserialize, Serialize};

use crate::orderbook::OrderBook;
use crate::orders::Side;

/// One venue leg used by [`analyze_cross_venue`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrageVenue {
    pub venue: String,
    pub quote_asset: String,
    /// Common-currency units per one unit of this venue's quote currency.
    pub quote_to_common: f32,
    /// Explicit taker fee in basis points.
    pub taker_fee_bps: f32,
    /// Fixed/network/gas/transfer cost allocated to this leg, already expressed
    /// in the common currency.
    pub additional_cost_common: f32,
    /// Available base inventory. Required on the selling leg for a fully
    /// executable verdict.
    pub available_base: Option<f32>,
    /// Available quote balance in the common currency. Required on the buying
    /// leg for a fully executable verdict.
    pub available_quote_common: Option<f32>,
    pub book: OrderBook,
}

/// Cross-venue analysis policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossVenueConfig {
    pub base_asset: String,
    pub common_quote: String,
    pub base_size: f32,
    /// Minimum net return required after all explicit costs, in basis points of
    /// the buy-side acquisition cost.
    pub min_net_profit_bps: f32,
    /// Evaluation timestamp used for staleness checks.
    pub now_ts_ms: i64,
    pub max_quote_age_ms: i64,
    pub max_leg_skew_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArbitrageConstraint {
    BuyDepthInsufficient,
    SellDepthInsufficient,
    BuyQuoteFromFuture,
    SellQuoteFromFuture,
    BuyQuoteStale,
    SellQuoteStale,
    LegTimestampSkew,
    BelowNetProfitThreshold,
    BuyBalanceUnverified,
    SellInventoryUnverified,
    InsufficientBuyQuoteBalance,
    InsufficientSellInventory,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossVenueReport {
    pub base_asset: String,
    pub common_quote: String,
    pub base_size: f32,
    pub buy_venue: String,
    pub sell_venue: String,
    pub buy_quote_asset: String,
    pub sell_quote_asset: String,
    pub buy_ts_ms: i64,
    pub sell_ts_ms: i64,
    pub timestamp_skew_ms: i64,
    pub buy_filled_base: f32,
    pub sell_filled_base: f32,
    pub buy_vwap: f32,
    pub sell_vwap: f32,
    pub buy_slippage_bps: f32,
    pub sell_slippage_bps: f32,
    pub buy_cost_common: f32,
    pub sell_proceeds_common: f32,
    pub gross_spread_bps: f32,
    pub buy_fee_common: f32,
    pub sell_fee_common: f32,
    pub additional_cost_common: f32,
    pub net_profit_common: f32,
    pub net_profit_bps: f32,
    /// True when depth, timestamps and explicit costs still clear the configured
    /// net-profit threshold. This does not imply inventory is available.
    pub market_edge_positive: bool,
    /// True only when `market_edge_positive` is true and the required buy quote
    /// balance and sell base inventory were both supplied and sufficient.
    pub fully_executable: bool,
    pub constraints: Vec<ArbitrageConstraint>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArbitrageError {
    InvalidBaseSize,
    InvalidProfitThreshold,
    InvalidTimePolicy,
    InvalidBuyConversion,
    InvalidSellConversion,
    InvalidBuyFee,
    InvalidSellFee,
    InvalidBuyAdditionalCost,
    InvalidSellAdditionalCost,
}

fn validate_venue(venue: &ArbitrageVenue, buy: bool) -> Result<(), ArbitrageError> {
    if !venue.quote_to_common.is_finite() || venue.quote_to_common <= 0.0
    {
        return Err(if buy {
            ArbitrageError::InvalidBuyConversion
        } else {
            ArbitrageError::InvalidSellConversion
        });
    }
    if !venue.taker_fee_bps.is_finite() || venue.taker_fee_bps < 0.0
    {
        return Err(if buy {
            ArbitrageError::InvalidBuyFee
        } else {
            ArbitrageError::InvalidSellFee
        });
    }
    if !venue.additional_cost_common.is_finite() || venue.additional_cost_common < 0.0
    {
        return Err(if buy {
            ArbitrageError::InvalidBuyAdditionalCost
        } else {
            ArbitrageError::InvalidSellAdditionalCost
        });
    }
    Ok(())
}

fn timestamp_constraints(
    ts_ms: i64,
    now_ts_ms: i64,
    max_age_ms: i64,
    from_future: ArbitrageConstraint,
    stale: ArbitrageConstraint,
    constraints: &mut Vec<ArbitrageConstraint>,
) {
    if ts_ms > now_ts_ms
    {
        constraints.push(from_future);
    }
    else if now_ts_ms - ts_ms > max_age_ms
    {
        constraints.push(stale);
    }
}

/// Analyze buying `base_size` on `buy` and simultaneously selling it on `sell`.
///
/// The function is pure and deterministic. It never submits orders and it never
/// assumes missing balances are sufficient.
pub fn analyze_cross_venue(
    buy: &ArbitrageVenue,
    sell: &ArbitrageVenue,
    cfg: &CrossVenueConfig,
) -> Result<CrossVenueReport, ArbitrageError> {
    if !cfg.base_size.is_finite() || cfg.base_size <= 0.0
    {
        return Err(ArbitrageError::InvalidBaseSize);
    }
    if !cfg.min_net_profit_bps.is_finite()
    {
        return Err(ArbitrageError::InvalidProfitThreshold);
    }
    if cfg.max_quote_age_ms < 0 || cfg.max_leg_skew_ms < 0
    {
        return Err(ArbitrageError::InvalidTimePolicy);
    }
    validate_venue(buy, true)?;
    validate_venue(sell, false)?;

    let buy_fill = buy.book.vwap_to_fill(Side::Buy, cfg.base_size);
    let sell_fill = sell.book.vwap_to_fill(Side::Sell, cfg.base_size);
    let mut constraints = Vec::new();

    if !buy_fill.fully_filled
    {
        constraints.push(ArbitrageConstraint::BuyDepthInsufficient);
    }
    if !sell_fill.fully_filled
    {
        constraints.push(ArbitrageConstraint::SellDepthInsufficient);
    }

    timestamp_constraints(
        buy.book.ts_ms,
        cfg.now_ts_ms,
        cfg.max_quote_age_ms,
        ArbitrageConstraint::BuyQuoteFromFuture,
        ArbitrageConstraint::BuyQuoteStale,
        &mut constraints,
    );
    timestamp_constraints(
        sell.book.ts_ms,
        cfg.now_ts_ms,
        cfg.max_quote_age_ms,
        ArbitrageConstraint::SellQuoteFromFuture,
        ArbitrageConstraint::SellQuoteStale,
        &mut constraints,
    );

    let timestamp_skew_ms = if buy.book.ts_ms >= sell.book.ts_ms
    {
        buy.book.ts_ms - sell.book.ts_ms
    }
    else
    {
        sell.book.ts_ms - buy.book.ts_ms
    };
    if timestamp_skew_ms > cfg.max_leg_skew_ms
    {
        constraints.push(ArbitrageConstraint::LegTimestampSkew);
    }

    let buy_cost_common = buy_fill.vwap * cfg.base_size * buy.quote_to_common;
    let sell_proceeds_common = sell_fill.vwap * cfg.base_size * sell.quote_to_common;
    let gross_spread_bps = if buy_cost_common > f32::MIN_POSITIVE
    {
        (sell_proceeds_common - buy_cost_common) / buy_cost_common * 10_000.0
    }
    else
    {
        0.0
    };
    let buy_fee_common = buy_cost_common * buy.taker_fee_bps / 10_000.0;
    let sell_fee_common = sell_proceeds_common * sell.taker_fee_bps / 10_000.0;
    let additional_cost_common = buy.additional_cost_common + sell.additional_cost_common;
    let net_profit_common = sell_proceeds_common
        - sell_fee_common
        - sell.additional_cost_common
        - buy_cost_common
        - buy_fee_common
        - buy.additional_cost_common;
    let acquisition_cost_common = buy_cost_common + buy_fee_common + buy.additional_cost_common;
    let net_profit_bps = if acquisition_cost_common > f32::MIN_POSITIVE
    {
        net_profit_common / acquisition_cost_common * 10_000.0
    }
    else
    {
        0.0
    };

    if net_profit_bps < cfg.min_net_profit_bps
    {
        constraints.push(ArbitrageConstraint::BelowNetProfitThreshold);
    }

    let market_blocked = constraints.iter().any(|c| {
        matches!(
            c,
            ArbitrageConstraint::BuyDepthInsufficient
                | ArbitrageConstraint::SellDepthInsufficient
                | ArbitrageConstraint::BuyQuoteFromFuture
                | ArbitrageConstraint::SellQuoteFromFuture
                | ArbitrageConstraint::BuyQuoteStale
                | ArbitrageConstraint::SellQuoteStale
                | ArbitrageConstraint::LegTimestampSkew
                | ArbitrageConstraint::BelowNetProfitThreshold
        )
    });
    let market_edge_positive = !market_blocked;

    match buy.available_quote_common
    {
        None => constraints.push(ArbitrageConstraint::BuyBalanceUnverified),
        Some(balance) if !balance.is_finite() || balance < acquisition_cost_common =>
        {
            constraints.push(ArbitrageConstraint::InsufficientBuyQuoteBalance);
        },
        Some(_) => {},
    }
    match sell.available_base
    {
        None => constraints.push(ArbitrageConstraint::SellInventoryUnverified),
        Some(balance) if !balance.is_finite() || balance < cfg.base_size =>
        {
            constraints.push(ArbitrageConstraint::InsufficientSellInventory);
        },
        Some(_) => {},
    }

    let balance_blocked = constraints.iter().any(|c| {
        matches!(
            c,
            ArbitrageConstraint::BuyBalanceUnverified
                | ArbitrageConstraint::SellInventoryUnverified
                | ArbitrageConstraint::InsufficientBuyQuoteBalance
                | ArbitrageConstraint::InsufficientSellInventory
        )
    });

    Ok(CrossVenueReport {
        base_asset: cfg.base_asset.clone(),
        common_quote: cfg.common_quote.clone(),
        base_size: cfg.base_size,
        buy_venue: buy.venue.clone(),
        sell_venue: sell.venue.clone(),
        buy_quote_asset: buy.quote_asset.clone(),
        sell_quote_asset: sell.quote_asset.clone(),
        buy_ts_ms: buy.book.ts_ms,
        sell_ts_ms: sell.book.ts_ms,
        timestamp_skew_ms,
        buy_filled_base: buy_fill.filled,
        sell_filled_base: sell_fill.filled,
        buy_vwap: buy_fill.vwap,
        sell_vwap: sell_fill.vwap,
        buy_slippage_bps: buy_fill.slippage_bps,
        sell_slippage_bps: sell_fill.slippage_bps,
        buy_cost_common,
        sell_proceeds_common,
        gross_spread_bps,
        buy_fee_common,
        sell_fee_common,
        additional_cost_common,
        net_profit_common,
        net_profit_bps,
        market_edge_positive,
        fully_executable: market_edge_positive && !balance_blocked,
        constraints,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orderbook::Level;

    fn venue(
        name: &str,
        quote: &str,
        ts_ms: i64,
        bids: &[(f32, f32)],
        asks: &[(f32, f32)],
    ) -> ArbitrageVenue {
        ArbitrageVenue {
            venue: name.to_string(),
            quote_asset: quote.to_string(),
            quote_to_common: 1.0,
            taker_fee_bps: 10.0,
            additional_cost_common: 0.0,
            available_base: Some(10.0),
            available_quote_common: Some(10_000.0),
            book: OrderBook::new(
                "BTC",
                ts_ms,
                bids.iter().map(|&(p, q)| Level::new(p, q)).collect(),
                asks.iter().map(|&(p, q)| Level::new(p, q)).collect(),
            ),
        }
    }

    fn cfg(size: f32) -> CrossVenueConfig {
        CrossVenueConfig {
            base_asset: "BTC".to_string(),
            common_quote: "USD".to_string(),
            base_size: size,
            min_net_profit_bps: 20.0,
            now_ts_ms: 1_000,
            max_quote_age_ms: 100,
            max_leg_skew_ms: 25,
        }
    }

    #[test]
    fn executable_depth_and_fees_still_leave_positive_edge() {
        let buy = venue("A", "USDT", 990, &[(99.0, 10.0)], &[(100.0, 10.0)]);
        let sell = venue("B", "USDC", 995, &[(103.0, 10.0)], &[(104.0, 10.0)]);
        let report = analyze_cross_venue(&buy, &sell, &cfg(2.0)).unwrap();
        assert!(report.gross_spread_bps > 200.0);
        assert!(report.net_profit_bps > 20.0);
        assert!(report.market_edge_positive);
        assert!(report.fully_executable);
        assert!(report.constraints.is_empty());
    }

    #[test]
    fn top_of_book_edge_can_disappear_when_requested_size_walks_depth() {
        let buy = venue(
            "A",
            "USD",
            990,
            &[(99.0, 10.0)],
            &[(100.0, 0.1), (110.0, 10.0)],
        );
        let sell = venue(
            "B",
            "USD",
            995,
            &[(105.0, 0.1), (101.0, 10.0)],
            &[(106.0, 10.0)],
        );
        let report = analyze_cross_venue(&buy, &sell, &cfg(1.0)).unwrap();
        assert!(report.buy_vwap > 108.0);
        assert!(report.sell_vwap < 102.0);
        assert!(!report.market_edge_positive);
        assert!(report
            .constraints
            .contains(&ArbitrageConstraint::BelowNetProfitThreshold));
    }

    #[test]
    fn insufficient_depth_blocks_market_edge() {
        let buy = venue("A", "USD", 990, &[(99.0, 1.0)], &[(100.0, 0.5)]);
        let sell = venue("B", "USD", 995, &[(103.0, 10.0)], &[(104.0, 10.0)]);
        let report = analyze_cross_venue(&buy, &sell, &cfg(1.0)).unwrap();
        assert!(!report.market_edge_positive);
        assert!(report
            .constraints
            .contains(&ArbitrageConstraint::BuyDepthInsufficient));
    }

    #[test]
    fn quote_conversion_is_applied_before_profitability() {
        let buy = venue("A", "USD", 990, &[(99.0, 10.0)], &[(100.0, 10.0)]);
        let mut sell = venue("B", "EUR", 995, &[(102.0, 10.0)], &[(103.0, 10.0)]);
        sell.quote_to_common = 0.95;
        let report = analyze_cross_venue(&buy, &sell, &cfg(1.0)).unwrap();
        assert!(report.sell_proceeds_common < report.buy_cost_common);
        assert!(!report.market_edge_positive);
    }

    #[test]
    fn stale_or_skewed_quotes_are_never_marked_positive() {
        let buy = venue("A", "USD", 700, &[(99.0, 10.0)], &[(100.0, 10.0)]);
        let sell = venue("B", "USD", 995, &[(103.0, 10.0)], &[(104.0, 10.0)]);
        let report = analyze_cross_venue(&buy, &sell, &cfg(1.0)).unwrap();
        assert!(!report.market_edge_positive);
        assert!(report.constraints.contains(&ArbitrageConstraint::BuyQuoteStale));
        assert!(report
            .constraints
            .contains(&ArbitrageConstraint::LegTimestampSkew));
    }

    #[test]
    fn missing_balances_preserve_market_edge_but_block_executable_verdict() {
        let mut buy = venue("A", "USD", 990, &[(99.0, 10.0)], &[(100.0, 10.0)]);
        let mut sell = venue("B", "USD", 995, &[(103.0, 10.0)], &[(104.0, 10.0)]);
        buy.available_quote_common = None;
        sell.available_base = None;
        let report = analyze_cross_venue(&buy, &sell, &cfg(1.0)).unwrap();
        assert!(report.market_edge_positive);
        assert!(!report.fully_executable);
        assert!(report
            .constraints
            .contains(&ArbitrageConstraint::BuyBalanceUnverified));
        assert!(report
            .constraints
            .contains(&ArbitrageConstraint::SellInventoryUnverified));
    }

    #[test]
    fn repeated_analysis_is_bit_reproducible() {
        let buy = venue("A", "USD", 990, &[(99.0, 10.0)], &[(100.0, 10.0)]);
        let sell = venue("B", "USD", 995, &[(103.0, 10.0)], &[(104.0, 10.0)]);
        let a = analyze_cross_venue(&buy, &sell, &cfg(1.0)).unwrap();
        let b = analyze_cross_venue(&buy, &sell, &cfg(1.0)).unwrap();
        assert_eq!(a, b);
    }
}
