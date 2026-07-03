//! Portfolio & position accounting — the book of record.
//!
//! A [`Position`] carries a **signed** quantity (long `> 0`, short `< 0`), an
//! average entry price, and cumulative realised PnL. Applying a fill nets
//! against the existing position: an opposing fill first *reduces* it (realising
//! PnL on the closed portion), and any overshoot *flips* the position with a
//! fresh average entry at the fill price — the standard one-way (netting) model.
//!
//! An [`Account`] holds cash plus a book of positions. Equity is
//! **mark-to-market asset value**: `cash + Σ qty·mark`, which is correct for
//! both longs (positive inventory) and shorts (negative inventory financed by
//! the proceeds already added to cash). Fees are debited from cash on every fill.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::orders::{Fill, Side};

/// A netted position in one symbol.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Position {
    pub symbol: String,
    /// Signed size: `> 0` long, `< 0` short, `0` flat.
    pub qty: f32,
    /// Average entry price of the open quantity (0 when flat).
    pub avg_entry: f32,
    /// Cumulative realised PnL (gross of fees) from closed quantity.
    pub realized_pnl: f32,
}

impl Position {
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            qty: 0.0,
            avg_entry: 0.0,
            realized_pnl: 0.0,
        }
    }

    pub fn is_flat(&self) -> bool {
        self.qty.abs() < 1e-12
    }

    pub fn is_long(&self) -> bool {
        self.qty > 1e-12
    }

    pub fn is_short(&self) -> bool {
        self.qty < -1e-12
    }

    /// Mark-to-market value of the inventory: `qty · mark` (signed).
    pub fn market_value(&self, mark: f32) -> f32 {
        self.qty * mark
    }

    /// Unrealised PnL at `mark`: `(mark − avg_entry) · qty` (works for shorts
    /// because `qty` is negative).
    pub fn unrealized_pnl(&self, mark: f32) -> f32 {
        if self.is_flat()
        {
            return 0.0;
        }
        (mark - self.avg_entry) * self.qty
    }

    /// Apply a signed size change from a fill and return the realised PnL booked
    /// by this fill (gross of fees). `side` and `qty` describe the fill; `price`
    /// is the execution price.
    pub fn apply(&mut self, side: Side, price: f32, qty: f32) -> f32 {
        let delta = side.sign() * qty; // signed change to the position
        let old = self.qty;
        let mut realized = 0.0f32;

        if old.abs() < 1e-12 || (old > 0.0) == (delta > 0.0)
        {
            // Opening or increasing in the same direction: blend the entry.
            let new_qty = old + delta;
            let old_abs = old.abs();
            let add_abs = delta.abs();
            let denom = old_abs + add_abs;
            self.avg_entry = if denom > 1e-12
            {
                (self.avg_entry * old_abs + price * add_abs) / denom
            }
            else
            {
                price
            };
            self.qty = new_qty;
        }
        else
        {
            // Opposing fill: reduce (and possibly flip).
            let close_abs = old.abs().min(delta.abs());
            // Realised PnL on the closed quantity: for a long being reduced by a
            // sell, (price − entry)·close; for a short reduced by a buy,
            // (entry − price)·close. `old.signum()` captures the sign.
            realized = (price - self.avg_entry) * close_abs * old.signum();
            self.realized_pnl += realized;
            let new_qty = old + delta;
            if new_qty.abs() < 1e-9
            {
                self.qty = 0.0;
                self.avg_entry = 0.0;
            }
            else if (new_qty > 0.0) != (old > 0.0)
            {
                // Flipped through zero: the overshoot opens a fresh position at
                // the fill price.
                self.qty = new_qty;
                self.avg_entry = price;
            }
            else
            {
                // Partial close: same direction, entry unchanged.
                self.qty = new_qty;
            }
        }
        realized
    }
}

/// Liquidation price for an isolated-margin linear position (analytical helper;
/// the paper engine does not force liquidations, but the agent can query the
/// distance-to-liquidation for risk sizing).
///
/// `leverage` is the initial leverage (so initial-margin rate = 1/leverage),
/// `mmr` is the maintenance-margin rate (e.g. 0.005 for 0.5 %).
pub fn liquidation_price(entry: f32, leverage: f32, mmr: f32, side: Side) -> f32 {
    let imr = if leverage > 0.0 { 1.0 / leverage } else { 1.0 };
    match side
    {
        Side::Buy => entry * (1.0 - imr + mmr), // long
        Side::Sell => entry * (1.0 + imr - mmr), // short
    }
}

/// A trading account: cash in quote currency plus a book of positions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub cash: f32,
    pub positions: BTreeMap<String, Position>,
    pub fees_paid: f32,
    pub realized_pnl: f32,
    /// Mark-to-market equity sampled over time (for the performance report).
    pub equity_curve: Vec<f32>,
}

impl Account {
    pub fn new(starting_cash: f32) -> Self {
        Self {
            cash: starting_cash,
            positions: BTreeMap::new(),
            fees_paid: 0.0,
            realized_pnl: 0.0,
            equity_curve: vec![starting_cash],
        }
    }

    /// Position for `symbol`, creating a flat one if absent.
    pub fn position(&mut self, symbol: &str) -> &mut Position {
        self.positions
            .entry(symbol.to_string())
            .or_insert_with(|| Position::new(symbol))
    }

    /// Current signed size in `symbol` (0 if none).
    pub fn qty(&self, symbol: &str) -> f32 {
        self.positions.get(symbol).map(|p| p.qty).unwrap_or(0.0)
    }

    /// Apply a fill for `symbol`/`side`: update cash (including fee), the
    /// position, and the fee/realised tallies.
    pub fn apply_fill(&mut self, symbol: &str, side: Side, fill: &Fill) {
        self.cash += fill.cash_flow(side);
        self.fees_paid += fill.fee;
        let realized = self.position(symbol).apply(side, fill.price, fill.qty);
        self.realized_pnl += realized;
    }

    /// Mark-to-market equity given a price for each held symbol: `cash + Σ qty·mark`.
    /// Symbols missing from `marks` are valued at their average entry (no PnL).
    pub fn equity(&self, marks: &BTreeMap<String, f32>) -> f32 {
        let mut eq = self.cash;
        for (sym, pos) in &self.positions
        {
            let mark = marks.get(sym).copied().unwrap_or(pos.avg_entry);
            eq += pos.market_value(mark);
        }
        eq
    }

    /// Total unrealised PnL across all positions at the given marks.
    pub fn unrealized_pnl(&self, marks: &BTreeMap<String, f32>) -> f32 {
        let mut u = 0.0f32;
        for (sym, pos) in &self.positions
        {
            let mark = marks.get(sym).copied().unwrap_or(pos.avg_entry);
            u += pos.unrealized_pnl(mark);
        }
        u
    }

    /// Sample the current equity onto the curve (called once per bar).
    pub fn mark(&mut self, marks: &BTreeMap<String, f32>) {
        let eq = self.equity(marks);
        self.equity_curve.push(eq);
    }

    /// Gross exposure = Σ |qty·mark|; net exposure = Σ qty·mark. Returned as
    /// `(gross, net)` in quote currency.
    pub fn exposure(&self, marks: &BTreeMap<String, f32>) -> (f32, f32) {
        let mut gross = 0.0f32;
        let mut net = 0.0f32;
        for (sym, pos) in &self.positions
        {
            let mark = marks.get(sym).copied().unwrap_or(pos.avg_entry);
            let mv = pos.market_value(mark);
            gross += mv.abs();
            net += mv;
        }
        (gross, net)
    }
}

/// A rebalancing trade: bring `symbol` to a target signed quantity.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RebalanceTrade {
    pub symbol: String,
    pub side: Side,
    pub qty: f32,
    pub target_qty: f32,
    pub current_qty: f32,
}

/// Compute the trades that move an account to `target_weights` (weight of total
/// equity per symbol; may sum to < 1 to hold cash) at the given `marks`.
///
/// `band` suppresses trades whose weight drift is below the threshold (0 = always
/// trade). Only symbols present in `target_weights` are adjusted.
pub fn rebalance_to_weights(
    account: &Account,
    target_weights: &BTreeMap<String, f32>,
    marks: &BTreeMap<String, f32>,
    band: f32,
) -> Vec<RebalanceTrade> {
    let equity = account.equity(marks);
    let mut trades = Vec::new();
    if equity <= 0.0
    {
        return trades;
    }
    for (sym, &w) in target_weights
    {
        let price = match marks.get(sym)
        {
            Some(p) if *p > 1e-12 => *p,
            _ => continue,
        };
        let current_qty = account.qty(sym);
        let current_weight = current_qty * price / equity;
        if (w - current_weight).abs() < band
        {
            continue;
        }
        let target_qty = w * equity / price;
        let delta = target_qty - current_qty;
        if delta.abs() < 1e-9
        {
            continue;
        }
        trades.push(RebalanceTrade {
            symbol: sym.clone(),
            side: if delta > 0.0 { Side::Buy } else { Side::Sell },
            qty: delta.abs(),
            target_qty,
            current_qty,
        });
    }
    trades
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(price: f32, qty: f32) -> Fill {
        Fill {
            price,
            qty,
            fee: 0.0,
            taker: true,
            ts_ms: 0,
        }
    }

    #[test]
    fn open_and_average_up() {
        let mut p = Position::new("BTC");
        p.apply(Side::Buy, 100.0, 1.0);
        p.apply(Side::Buy, 110.0, 1.0);
        assert!((p.qty - 2.0).abs() < 1e-6);
        assert!((p.avg_entry - 105.0).abs() < 1e-4);
    }

    #[test]
    fn long_realizes_pnl_on_close() {
        let mut p = Position::new("BTC");
        p.apply(Side::Buy, 100.0, 2.0);
        let r = p.apply(Side::Sell, 110.0, 1.0); // close 1 unit at +10
        assert!((r - 10.0).abs() < 1e-4);
        assert!((p.qty - 1.0).abs() < 1e-6);
        assert!((p.avg_entry - 100.0).abs() < 1e-4); // entry unchanged on partial close
    }

    #[test]
    fn short_realizes_pnl() {
        let mut p = Position::new("BTC");
        p.apply(Side::Sell, 100.0, 1.0); // short 1 @100
        assert!(p.is_short());
        let r = p.apply(Side::Buy, 90.0, 1.0); // cover at 90 -> +10
        assert!((r - 10.0).abs() < 1e-4);
        assert!(p.is_flat());
    }

    #[test]
    fn flip_resets_entry_to_fill_price() {
        let mut p = Position::new("BTC");
        p.apply(Side::Buy, 100.0, 1.0); // long 1
        let r = p.apply(Side::Sell, 120.0, 3.0); // close 1 (+20), flip short 2 @120
        assert!((r - 20.0).abs() < 1e-4);
        assert!((p.qty - (-2.0)).abs() < 1e-6);
        assert!((p.avg_entry - 120.0).abs() < 1e-4);
    }

    #[test]
    fn unrealized_pnl_long_and_short() {
        let mut long = Position::new("BTC");
        long.apply(Side::Buy, 100.0, 1.0);
        assert!((long.unrealized_pnl(110.0) - 10.0).abs() < 1e-4);
        let mut short = Position::new("BTC");
        short.apply(Side::Sell, 100.0, 1.0);
        assert!((short.unrealized_pnl(90.0) - 10.0).abs() < 1e-4);
    }

    #[test]
    fn account_equity_conserves_value_minus_fees() {
        let mut acct = Account::new(1000.0);
        let mut f = fill(100.0, 1.0);
        f.fee = 0.5;
        acct.apply_fill("BTC", Side::Buy, &f);
        // cash = 1000 - 100 - 0.5 = 899.5; equity at mark 100 = 899.5 + 100 = 999.5
        let mut marks = BTreeMap::new();
        marks.insert("BTC".to_string(), 100.0);
        assert!((acct.equity(&marks) - 999.5).abs() < 1e-3);
        assert!((acct.fees_paid - 0.5).abs() < 1e-4);
    }

    #[test]
    fn account_short_equity() {
        let mut acct = Account::new(1000.0);
        acct.apply_fill("BTC", Side::Sell, &fill(100.0, 1.0)); // short 1
        let mut marks = BTreeMap::new();
        marks.insert("BTC".to_string(), 90.0);
        // cash = 1000 + 100 = 1100; equity = 1100 + (-1*90) = 1010 (profit 10)
        assert!((acct.equity(&marks) - 1010.0).abs() < 1e-3);
    }

    #[test]
    fn liquidation_price_directions() {
        // 10x long, mmr 0.5% -> liq below entry.
        let liq_long = liquidation_price(100.0, 10.0, 0.005, Side::Buy);
        assert!(liq_long < 100.0 && liq_long > 88.0);
        let liq_short = liquidation_price(100.0, 10.0, 0.005, Side::Sell);
        assert!(liq_short > 100.0 && liq_short < 112.0);
    }

    #[test]
    fn rebalance_generates_trades_to_targets() {
        let mut acct = Account::new(10_000.0);
        // Currently all cash. Target 50% BTC.
        let mut marks = BTreeMap::new();
        marks.insert("BTC".to_string(), 100.0);
        let mut targets = BTreeMap::new();
        targets.insert("BTC".to_string(), 0.5);
        let trades = rebalance_to_weights(&acct, &targets, &marks, 0.0);
        assert_eq!(trades.len(), 1);
        // 50% of 10_000 = 5_000 notional / 100 = 50 units to buy.
        assert!((trades[0].qty - 50.0).abs() < 1e-3);
        assert_eq!(trades[0].side, Side::Buy);

        // Apply it and confirm no further trade within band.
        acct.apply_fill("BTC", Side::Buy, &fill(100.0, 50.0));
        let trades2 = rebalance_to_weights(&acct, &targets, &marks, 0.01);
        assert!(trades2.is_empty());
    }

    #[test]
    fn exposure_gross_and_net() {
        let mut acct = Account::new(10_000.0);
        acct.apply_fill("BTC", Side::Buy, &fill(100.0, 10.0));
        acct.apply_fill("ETH", Side::Sell, &fill(50.0, 20.0));
        let mut marks = BTreeMap::new();
        marks.insert("BTC".to_string(), 100.0);
        marks.insert("ETH".to_string(), 50.0);
        let (gross, net) = acct.exposure(&marks);
        // BTC +1000, ETH -1000 -> gross 2000, net 0.
        assert!((gross - 2000.0).abs() < 1e-2);
        assert!(net.abs() < 1e-2);
    }
}
