//! Cross-exchange market-making analysis.
//!
//! This module evaluates a maker quote on one venue against the executable
//! hedge available on another venue. It does not submit orders. The output is
//! deliberately scenario-like: both maker-fill directions are evaluated after
//! explicit fees, executable hedge depth and an adverse latency buffer.
//!
//! A positive quoted spread is never sufficient by itself. A side is labelled
//! executable only when the hedge book can absorb the full requested base size,
//! timestamps are usable and the cost-net edge clears the configured threshold.

use serde::{Deserialize, Serialize};

use crate::marketmaking::Quotes;
use crate::orderbook::OrderBook;
use crate::orders::Side;

/// Explicit assumptions for a cross-exchange market-making evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossExchangeMmConfig {
    pub maker_venue: String,
    pub hedge_venue: String,
    /// Base quantity assumed to fill on the maker venue and therefore requiring
    /// an immediate hedge on the second venue.
    pub base_size: f32,
    /// Maker fee rate as a fraction of notional. Negative values represent a
    /// rebate. Values must be greater than `-1`.
    pub maker_fee_rate: f32,
    /// Hedge taker fee rate as a fraction of notional. Values must be greater
    /// than `-1`.
    pub hedge_taker_fee_rate: f32,
    /// One-way time from maker fill observation to hedge execution.
    pub hedge_latency_ms: u64,
    /// Caller-declared adverse price movement per second, in basis points. This
    /// is a stress assumption, not an inferred volatility forecast.
    pub adverse_move_bps_per_second: f32,
    /// Fixed cost charged per completed hedge, in quote currency.
    pub fixed_hedge_cost: f32,
    /// Minimum required net edge in basis points of maker notional.
    pub min_net_edge_bps: f32,
    /// Evaluation clock.
    pub now_ms: i64,
    /// Maximum allowed age for either snapshot.
    pub max_age_ms: i64,
    /// Maximum timestamp skew between maker and hedge snapshots.
    pub max_skew_ms: i64,
}

/// One maker-fill direction and its corresponding cross-venue hedge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossExchangeMmSide {
    /// Maker side that is assumed to fill.
    pub maker_side: Side,
    /// Taker side required on the hedge venue.
    pub hedge_side: Side,
    pub maker_price: f32,
    pub hedge_snapshot_vwap: f32,
    /// Hedge price after applying the explicit adverse latency stress.
    pub hedge_stressed_vwap: f32,
    pub hedge_filled_base: f32,
    pub hedge_fully_filled: bool,
    pub maker_fee: f32,
    pub hedge_fee: f32,
    pub fixed_hedge_cost: f32,
    /// Net quote-currency PnL for this one maker-fill + hedge cycle.
    pub net_pnl: f32,
    /// Net edge in basis points of maker notional.
    pub net_edge_bps: f32,
    /// True only when full hedge depth exists and the configured minimum edge is
    /// exceeded.
    pub executable_positive: bool,
}

/// Deterministic report for both possible maker-fill directions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossExchangeMmReport {
    pub maker_venue: String,
    pub hedge_venue: String,
    pub base_size: f32,
    pub maker_ts_ms: i64,
    pub hedge_ts_ms: i64,
    pub maker_age_ms: i64,
    pub hedge_age_ms: i64,
    pub snapshot_skew_ms: i64,
    /// Adverse hedge stress implied by latency and the caller-declared movement
    /// rate.
    pub latency_buffer_bps: f32,
    /// Maker bid fill: buy on maker, then sell on hedge venue.
    pub maker_bid_fill: CrossExchangeMmSide,
    /// Maker ask fill: sell on maker, then buy on hedge venue.
    pub maker_ask_fill: CrossExchangeMmSide,
    pub positive_sides: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossExchangeMmError {
    InvalidConfig,
    InvalidQuotes,
    StaleSnapshot,
    FutureSnapshot,
    ExcessiveSnapshotSkew,
    EmptyHedgeBook,
}

fn valid_rate(x: f32) -> bool {
    x.is_finite() && x > -1.0
}

fn validate_config(cfg: &CrossExchangeMmConfig) -> bool {
    cfg.base_size.is_finite()
        && cfg.base_size > 0.0
        && valid_rate(cfg.maker_fee_rate)
        && valid_rate(cfg.hedge_taker_fee_rate)
        && cfg.adverse_move_bps_per_second.is_finite()
        && cfg.adverse_move_bps_per_second >= 0.0
        && cfg.fixed_hedge_cost.is_finite()
        && cfg.fixed_hedge_cost >= 0.0
        && cfg.min_net_edge_bps.is_finite()
        && cfg.min_net_edge_bps >= 0.0
        && cfg.max_age_ms >= 0
        && cfg.max_skew_ms >= 0
}

fn quote_valid(q: &Quotes) -> bool {
    q.bid.is_finite() && q.ask.is_finite() && q.bid > 0.0 && q.ask > 0.0 && q.bid <= q.ask
}

fn age_ms(now_ms: i64, ts_ms: i64) -> Result<i64, CrossExchangeMmError> {
    if ts_ms > now_ms
    {
        return Err(CrossExchangeMmError::FutureSnapshot);
    }
    Ok(now_ms - ts_ms)
}

fn edge_bps(net_pnl: f32, maker_price: f32, base_size: f32) -> f32 {
    let notional = maker_price * base_size;
    if notional > 1e-12
    {
        10_000.0 * net_pnl / notional
    }
    else
    {
        0.0
    }
}

/// Evaluate a maker quote against executable hedge depth on another venue.
///
/// `maker_ts_ms` is the timestamp of the state used to generate `maker_quotes`.
/// The hedge timestamp comes from `hedge_book`.
pub fn analyze_cross_exchange_mm(
    maker_quotes: &Quotes,
    maker_ts_ms: i64,
    hedge_book: &OrderBook,
    cfg: &CrossExchangeMmConfig,
) -> Result<CrossExchangeMmReport, CrossExchangeMmError> {
    if !validate_config(cfg)
    {
        return Err(CrossExchangeMmError::InvalidConfig);
    }
    if !quote_valid(maker_quotes)
    {
        return Err(CrossExchangeMmError::InvalidQuotes);
    }
    if hedge_book.best_bid().is_none() || hedge_book.best_ask().is_none()
    {
        return Err(CrossExchangeMmError::EmptyHedgeBook);
    }

    let maker_age_ms = age_ms(cfg.now_ms, maker_ts_ms)?;
    let hedge_age_ms = age_ms(cfg.now_ms, hedge_book.ts_ms)?;
    if maker_age_ms > cfg.max_age_ms || hedge_age_ms > cfg.max_age_ms
    {
        return Err(CrossExchangeMmError::StaleSnapshot);
    }
    let snapshot_skew_ms = (maker_ts_ms - hedge_book.ts_ms).abs();
    if snapshot_skew_ms > cfg.max_skew_ms
    {
        return Err(CrossExchangeMmError::ExcessiveSnapshotSkew);
    }

    let latency_buffer_bps = cfg.adverse_move_bps_per_second * cfg.hedge_latency_ms as f32 / 1000.0;
    let latency_frac = latency_buffer_bps / 10_000.0;

    let hedge_sell = hedge_book.vwap_to_fill(Side::Sell, cfg.base_size);
    let stressed_sell = hedge_sell.vwap * (1.0 - latency_frac);
    let maker_bid_notional = maker_quotes.bid * cfg.base_size;
    let hedge_sell_notional = stressed_sell * hedge_sell.filled;
    let maker_bid_fee = maker_bid_notional * cfg.maker_fee_rate;
    let hedge_sell_fee = hedge_sell_notional * cfg.hedge_taker_fee_rate;
    let bid_net = hedge_sell_notional
        - hedge_sell_fee
        - maker_bid_notional
        - maker_bid_fee
        - cfg.fixed_hedge_cost;
    let bid_edge = edge_bps(bid_net, maker_quotes.bid, cfg.base_size);
    let bid_positive = hedge_sell.fully_filled && bid_edge > cfg.min_net_edge_bps;
    let maker_bid_fill = CrossExchangeMmSide {
        maker_side: Side::Buy,
        hedge_side: Side::Sell,
        maker_price: maker_quotes.bid,
        hedge_snapshot_vwap: hedge_sell.vwap,
        hedge_stressed_vwap: stressed_sell,
        hedge_filled_base: hedge_sell.filled,
        hedge_fully_filled: hedge_sell.fully_filled,
        maker_fee: maker_bid_fee,
        hedge_fee: hedge_sell_fee,
        fixed_hedge_cost: cfg.fixed_hedge_cost,
        net_pnl: bid_net,
        net_edge_bps: bid_edge,
        executable_positive: bid_positive,
    };

    let hedge_buy = hedge_book.vwap_to_fill(Side::Buy, cfg.base_size);
    let stressed_buy = hedge_buy.vwap * (1.0 + latency_frac);
    let maker_ask_notional = maker_quotes.ask * cfg.base_size;
    let hedge_buy_notional = stressed_buy * hedge_buy.filled;
    let maker_ask_fee = maker_ask_notional * cfg.maker_fee_rate;
    let hedge_buy_fee = hedge_buy_notional * cfg.hedge_taker_fee_rate;
    let ask_net = maker_ask_notional
        - maker_ask_fee
        - hedge_buy_notional
        - hedge_buy_fee
        - cfg.fixed_hedge_cost;
    let ask_edge = edge_bps(ask_net, maker_quotes.ask, cfg.base_size);
    let ask_positive = hedge_buy.fully_filled && ask_edge > cfg.min_net_edge_bps;
    let maker_ask_fill = CrossExchangeMmSide {
        maker_side: Side::Sell,
        hedge_side: Side::Buy,
        maker_price: maker_quotes.ask,
        hedge_snapshot_vwap: hedge_buy.vwap,
        hedge_stressed_vwap: stressed_buy,
        hedge_filled_base: hedge_buy.filled,
        hedge_fully_filled: hedge_buy.fully_filled,
        maker_fee: maker_ask_fee,
        hedge_fee: hedge_buy_fee,
        fixed_hedge_cost: cfg.fixed_hedge_cost,
        net_pnl: ask_net,
        net_edge_bps: ask_edge,
        executable_positive: ask_positive,
    };

    Ok(CrossExchangeMmReport {
        maker_venue: cfg.maker_venue.clone(),
        hedge_venue: cfg.hedge_venue.clone(),
        base_size: cfg.base_size,
        maker_ts_ms,
        hedge_ts_ms: hedge_book.ts_ms,
        maker_age_ms,
        hedge_age_ms,
        snapshot_skew_ms,
        latency_buffer_bps,
        positive_sides: usize::from(maker_bid_fill.executable_positive)
            + usize::from(maker_ask_fill.executable_positive),
        maker_bid_fill,
        maker_ask_fill,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marketmaking::Quotes;
    use crate::orderbook::Level;

    fn quotes(bid: f32, ask: f32) -> Quotes {
        Quotes {
            bid,
            ask,
            reservation_price: (bid + ask) / 2.0,
            spread: ask - bid,
            skew: 0.0,
            bid_offset: 0.0,
            ask_offset: 0.0,
        }
    }

    fn hedge_book(ts_ms: i64) -> OrderBook {
        OrderBook::new(
            "BTC-USDT",
            ts_ms,
            vec![Level::new(100.5, 2.0), Level::new(100.4, 4.0)],
            vec![Level::new(100.6, 2.0), Level::new(100.7, 4.0)],
        )
    }

    fn cfg() -> CrossExchangeMmConfig {
        CrossExchangeMmConfig {
            maker_venue: "maker".to_string(),
            hedge_venue: "hedge".to_string(),
            base_size: 1.0,
            maker_fee_rate: -0.0001,
            hedge_taker_fee_rate: 0.0002,
            hedge_latency_ms: 50,
            adverse_move_bps_per_second: 2.0,
            fixed_hedge_cost: 0.0,
            min_net_edge_bps: 1.0,
            now_ms: 10_000,
            max_age_ms: 1_000,
            max_skew_ms: 250,
        }
    }

    #[test]
    fn reports_both_fill_directions() {
        let r = analyze_cross_exchange_mm(&quotes(100.0, 101.0), 9_950, &hedge_book(9_900), &cfg())
            .unwrap();
        assert_eq!(r.maker_bid_fill.hedge_side, Side::Sell);
        assert_eq!(r.maker_ask_fill.hedge_side, Side::Buy);
        assert!(r.maker_bid_fill.hedge_fully_filled);
        assert!(r.maker_ask_fill.hedge_fully_filled);
    }

    #[test]
    fn executable_label_requires_cost_net_edge() {
        let r = analyze_cross_exchange_mm(&quotes(100.0, 101.0), 9_950, &hedge_book(9_900), &cfg())
            .unwrap();
        assert!(r.maker_bid_fill.executable_positive);
        assert!(r.maker_ask_fill.executable_positive);

        let mut expensive = cfg();
        expensive.fixed_hedge_cost = 1.0;
        let r2 =
            analyze_cross_exchange_mm(&quotes(100.0, 101.0), 9_950, &hedge_book(9_900), &expensive)
                .unwrap();
        assert!(!r2.maker_bid_fill.executable_positive);
        assert!(!r2.maker_ask_fill.executable_positive);
    }

    #[test]
    fn latency_stress_is_adverse_on_both_hedges() {
        let r = analyze_cross_exchange_mm(&quotes(100.0, 101.0), 9_950, &hedge_book(9_900), &cfg())
            .unwrap();
        assert!(r.maker_bid_fill.hedge_stressed_vwap < r.maker_bid_fill.hedge_snapshot_vwap);
        assert!(r.maker_ask_fill.hedge_stressed_vwap > r.maker_ask_fill.hedge_snapshot_vwap);
        assert!(r.latency_buffer_bps > 0.0);
    }

    #[test]
    fn insufficient_hedge_depth_cannot_be_executable() {
        let thin = OrderBook::new(
            "BTC-USDT",
            9_900,
            vec![Level::new(100.5, 0.2)],
            vec![Level::new(100.6, 0.2)],
        );
        let r = analyze_cross_exchange_mm(&quotes(100.0, 101.0), 9_950, &thin, &cfg()).unwrap();
        assert!(!r.maker_bid_fill.hedge_fully_filled);
        assert!(!r.maker_ask_fill.hedge_fully_filled);
        assert_eq!(r.positive_sides, 0);
    }

    #[test]
    fn rejects_stale_future_and_skewed_snapshots() {
        let mut stale_cfg = cfg();
        stale_cfg.max_age_ms = 20;
        assert!(matches!(
            analyze_cross_exchange_mm(&quotes(100.0, 101.0), 9_950, &hedge_book(9_900), &stale_cfg),
            Err(CrossExchangeMmError::StaleSnapshot)
        ));

        assert!(matches!(
            analyze_cross_exchange_mm(&quotes(100.0, 101.0), 10_001, &hedge_book(9_900), &cfg()),
            Err(CrossExchangeMmError::FutureSnapshot)
        ));

        let mut skew_cfg = cfg();
        skew_cfg.max_skew_ms = 10;
        assert!(matches!(
            analyze_cross_exchange_mm(&quotes(100.0, 101.0), 9_950, &hedge_book(9_900), &skew_cfg),
            Err(CrossExchangeMmError::ExcessiveSnapshotSkew)
        ));
    }

    #[test]
    fn repeated_analysis_is_deterministic() {
        let q = quotes(100.0, 101.0);
        let b = hedge_book(9_900);
        let c = cfg();
        let a = analyze_cross_exchange_mm(&q, 9_950, &b, &c).unwrap();
        let d = analyze_cross_exchange_mm(&q, 9_950, &b, &c).unwrap();
        assert_eq!(
            a.maker_bid_fill.net_pnl.to_bits(),
            d.maker_bid_fill.net_pnl.to_bits()
        );
        assert_eq!(
            a.maker_ask_fill.net_pnl.to_bits(),
            d.maker_ask_fill.net_pnl.to_bits()
        );
    }
}
