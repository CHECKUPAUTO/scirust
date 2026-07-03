//! Order-book microstructure analytics — the read-side an execution algo or a
//! market-making agent needs before it sizes an order.
//!
//! An [`OrderBook`] is a snapshot of resting liquidity: `bids` sorted by price
//! descending, `asks` ascending. From it we derive the mid / micro price, the
//! spread, book imbalance, and — most importantly — the **cost of taking size**:
//! walking the book to compute the VWAP a market order would pay and the
//! resulting slippage. This is what lets the agent answer "how much can I trade
//! without moving the price more than X bps?".

use serde::{Deserialize, Serialize};

use crate::orders::Side;

/// One price level: `price` and the resting `qty` (base units) at it.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Level {
    pub price: f32,
    pub qty: f32,
}

impl Level {
    pub fn new(price: f32, qty: f32) -> Self {
        Self { price, qty }
    }
}

/// A book snapshot. `bids` must be price-descending, `asks` price-ascending.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub ts_ms: i64,
    pub bids: Vec<Level>,
    pub asks: Vec<Level>,
}

impl OrderBook {
    /// Build a book from raw `(price, qty)` pairs, sorting each side canonically.
    pub fn new(symbol: &str, ts_ms: i64, bids: Vec<Level>, asks: Vec<Level>) -> Self {
        let mut b = bids;
        let mut a = asks;
        b.sort_by(|x, y| y.price.partial_cmp(&x.price).unwrap_or(std::cmp::Ordering::Equal));
        a.sort_by(|x, y| x.price.partial_cmp(&y.price).unwrap_or(std::cmp::Ordering::Equal));
        Self {
            symbol: symbol.to_string(),
            ts_ms,
            bids: b,
            asks: a,
        }
    }

    pub fn best_bid(&self) -> Option<Level> {
        self.bids.first().copied()
    }

    pub fn best_ask(&self) -> Option<Level> {
        self.asks.first().copied()
    }

    /// `(best_ask + best_bid) / 2`.
    pub fn mid(&self) -> Option<f32> {
        match (self.best_bid(), self.best_ask())
        {
            (Some(b), Some(a)) => Some((a.price + b.price) / 2.0),
            _ => None,
        }
    }

    /// Size-weighted micro-price `(Pb·Qa + Pa·Qb) / (Qa + Qb)`. Because it
    /// weights each side's price by the *opposite* side's size, the micro-price
    /// leans toward the thinner side — the one more likely to be swept next.
    pub fn micro_price(&self) -> Option<f32> {
        let b = self.best_bid()?;
        let a = self.best_ask()?;
        let denom = a.qty + b.qty;
        if denom <= 1e-12
        {
            return self.mid();
        }
        Some((b.price * a.qty + a.price * b.qty) / denom)
    }

    /// Absolute spread `best_ask − best_bid`.
    pub fn spread(&self) -> Option<f32> {
        match (self.best_bid(), self.best_ask())
        {
            (Some(b), Some(a)) => Some(a.price - b.price),
            _ => None,
        }
    }

    /// Spread in basis points of the mid.
    pub fn spread_bps(&self) -> Option<f32> {
        let s = self.spread()?;
        let m = self.mid()?;
        if m.abs() < 1e-12
        {
            return None;
        }
        Some(10_000.0 * s / m)
    }

    /// Total resting quantity in the top `n` levels of a side (base units).
    pub fn depth(&self, side: Side, n: usize) -> f32 {
        let book = match side
        {
            Side::Buy => &self.asks, // buying consumes asks
            Side::Sell => &self.bids,
        };
        book.iter().take(n).map(|l| l.qty).sum()
    }

    /// Book imbalance over the top `n` levels: `(bidVol − askVol)/(bidVol + askVol)`
    /// in `[-1, 1]`. Positive = more resting bids = buy-side pressure.
    pub fn imbalance(&self, n: usize) -> f32 {
        let bid_vol: f32 = self.bids.iter().take(n).map(|l| l.qty).sum();
        let ask_vol: f32 = self.asks.iter().take(n).map(|l| l.qty).sum();
        let denom = bid_vol + ask_vol;
        if denom <= 1e-12
        {
            return 0.0;
        }
        (bid_vol - ask_vol) / denom
    }

    /// Walk the book to fill `size` base units on the given side. Returns the
    /// [`FillEstimate`] — the VWAP paid, the filled quantity (may be less than
    /// `size` if the book is too thin), and slippage vs the mid.
    pub fn vwap_to_fill(&self, side: Side, size: f32) -> FillEstimate {
        let levels = match side
        {
            Side::Buy => &self.asks,
            Side::Sell => &self.bids,
        };
        let mid = self.mid().unwrap_or(0.0);
        let mut remaining = size.max(0.0);
        let mut cost = 0.0f32;
        let mut filled = 0.0f32;
        let mut worst = None;
        for lvl in levels
        {
            if remaining <= 1e-12
            {
                break;
            }
            let take = remaining.min(lvl.qty);
            cost += take * lvl.price;
            filled += take;
            remaining -= take;
            worst = Some(lvl.price);
        }
        let vwap = if filled > 1e-12 { cost / filled } else { mid };
        let slippage_bps = if mid.abs() > 1e-12
        {
            10_000.0 * (vwap - mid) / mid * side.sign()
        }
        else
        {
            0.0
        };
        FillEstimate {
            requested: size,
            filled,
            vwap,
            worst_price: worst.unwrap_or(mid),
            slippage_bps,
            fully_filled: remaining <= 1e-9,
        }
    }

    /// Base-unit liquidity resting within `bps` of the mid on the side a taker
    /// would consume (asks for a buy, bids for a sell) — "how much can I buy
    /// before moving the price `bps`?".
    pub fn liquidity_within_bps(&self, side: Side, bps: f32) -> f32 {
        let mid = match self.mid()
        {
            Some(m) => m,
            None => return 0.0,
        };
        let band = mid * bps / 10_000.0;
        match side
        {
            Side::Buy =>
            {
                let limit = mid + band;
                self.asks.iter().take_while(|l| l.price <= limit).map(|l| l.qty).sum()
            },
            Side::Sell =>
            {
                let limit = mid - band;
                self.bids.iter().take_while(|l| l.price >= limit).map(|l| l.qty).sum()
            },
        }
    }

    /// A compact analytics bundle for the agent (all the scalars at once).
    pub fn analyze(&self, depth_levels: usize) -> BookAnalysis {
        BookAnalysis {
            symbol: self.symbol.clone(),
            mid: self.mid().unwrap_or(0.0),
            micro_price: self.micro_price().unwrap_or(0.0),
            best_bid: self.best_bid().map(|l| l.price).unwrap_or(0.0),
            best_ask: self.best_ask().map(|l| l.price).unwrap_or(0.0),
            spread: self.spread().unwrap_or(0.0),
            spread_bps: self.spread_bps().unwrap_or(0.0),
            bid_depth: self.depth(Side::Sell, depth_levels),
            ask_depth: self.depth(Side::Buy, depth_levels),
            imbalance: self.imbalance(depth_levels),
        }
    }
}

/// The estimated result of taking `requested` size off the book.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FillEstimate {
    pub requested: f32,
    pub filled: f32,
    pub vwap: f32,
    pub worst_price: f32,
    /// Slippage vs the mid in bps (always ≥ 0 for a taker; sign-normalised).
    pub slippage_bps: f32,
    pub fully_filled: bool,
}

/// Flat, serialisable book summary for a tool response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BookAnalysis {
    pub symbol: String,
    pub mid: f32,
    pub micro_price: f32,
    pub best_bid: f32,
    pub best_ask: f32,
    pub spread: f32,
    pub spread_bps: f32,
    pub bid_depth: f32,
    pub ask_depth: f32,
    pub imbalance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book() -> OrderBook {
        OrderBook::new(
            "BTC/USDT",
            1000,
            vec![Level::new(99.0, 5.0), Level::new(98.0, 10.0), Level::new(97.0, 20.0)],
            vec![Level::new(101.0, 4.0), Level::new(102.0, 8.0), Level::new(103.0, 16.0)],
        )
    }

    #[test]
    fn mid_and_spread() {
        let b = book();
        assert!((b.mid().unwrap() - 100.0).abs() < 1e-4);
        assert!((b.spread().unwrap() - 2.0).abs() < 1e-4);
        assert!((b.spread_bps().unwrap() - 200.0).abs() < 1e-2);
    }

    #[test]
    fn micro_price_leans_to_thin_side() {
        // best bid 99 qty 5, best ask 101 qty 4. Ask is thinner -> micro-price
        // should lean toward the ask (above mid 100).
        let b = book();
        let mp = b.micro_price().unwrap();
        // (99*4 + 101*5)/(9) = (396+505)/9 = 100.11
        assert!((mp - 100.111).abs() < 1e-2, "micro {mp}");
        assert!(mp > b.mid().unwrap());
    }

    #[test]
    fn imbalance_sign() {
        // bids depth 3 levels = 35, asks = 28 -> positive imbalance.
        let b = book();
        let imb = b.imbalance(3);
        assert!(imb > 0.0);
        assert!((imb - (35.0 - 28.0) / (35.0 + 28.0)).abs() < 1e-4);
    }

    #[test]
    fn vwap_to_fill_walks_asks() {
        let b = book();
        // Buy 10: 4@101 + 6@102 = (404 + 612)/10 = 101.6
        let est = b.vwap_to_fill(Side::Buy, 10.0);
        assert!(est.fully_filled);
        assert!((est.vwap - 101.6).abs() < 1e-3, "vwap {}", est.vwap);
        assert!(est.slippage_bps > 0.0);
        assert!((est.worst_price - 102.0).abs() < 1e-4);
    }

    #[test]
    fn vwap_to_fill_reports_partial_when_thin() {
        let b = book();
        // Asks total 4+8+16 = 28. Ask for 100 -> partial fill of 28.
        let est = b.vwap_to_fill(Side::Buy, 100.0);
        assert!(!est.fully_filled);
        assert!((est.filled - 28.0).abs() < 1e-4);
    }

    #[test]
    fn liquidity_within_bps() {
        let b = book();
        // mid 100, 100 bps = 1.0 band -> asks with price <= 101 -> only 101 level (4).
        let liq = b.liquidity_within_bps(Side::Buy, 100.0);
        assert!((liq - 4.0).abs() < 1e-4);
        // 300 bps -> band 3 -> asks <= 103 -> 4+8+16 = 28.
        let liq2 = b.liquidity_within_bps(Side::Buy, 300.0);
        assert!((liq2 - 28.0).abs() < 1e-4);
    }

    #[test]
    fn analyze_bundle() {
        let a = book().analyze(3);
        assert!((a.mid - 100.0).abs() < 1e-4);
        assert!(a.imbalance > 0.0);
        assert!(a.spread_bps > 0.0);
    }

    #[test]
    fn unsorted_input_is_canonicalized() {
        let b = OrderBook::new(
            "X",
            1,
            vec![Level::new(97.0, 1.0), Level::new(99.0, 1.0), Level::new(98.0, 1.0)],
            vec![Level::new(103.0, 1.0), Level::new(101.0, 1.0), Level::new(102.0, 1.0)],
        );
        assert_eq!(b.best_bid().unwrap().price, 99.0);
        assert_eq!(b.best_ask().unwrap().price, 101.0);
    }
}
