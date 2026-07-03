//! Order types, fee/slippage models, and the paper matching engine.
//!
//! This is the execution layer a professional venue exposes: market / limit /
//! stop / stop-limit / trailing-stop / take-profit orders, time-in-force
//! (GTC/IOC/FOK), maker/taker fees, a slippage model, and tick/lot rounding.
//! Fills are simulated deterministically against an OHLC candle using the
//! **standard backtest fill semantics** (a limit buy fills iff the bar traded
//! at or below its price, a stop buy triggers iff the bar traded at or above
//! its stop, …). No network, no real money — the `PaperExchange` in
//! [`crate::backtest`] drives these primitives.
//!
//! Look-ahead discipline: the candle passed to [`simulate_fill`] is the
//! **execution** bar (typically the bar *after* the signal), never the signal
//! bar itself.

use serde::{Deserialize, Serialize};

use crate::market::Candle;

/// Buy or sell.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    /// `+1.0` for a buy, `-1.0` for a sell — the sign of the signed position
    /// delta this side produces.
    pub fn sign(&self) -> f32 {
        match self
        {
            Side::Buy => 1.0,
            Side::Sell => -1.0,
        }
    }

    pub fn opposite(&self) -> Side {
        match self
        {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }

    pub fn label(&self) -> &'static str {
        match self
        {
            Side::Buy => "BUY",
            Side::Sell => "SELL",
        }
    }
}

/// The order's price semantics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum OrderType {
    /// Fill immediately at the prevailing price (taker).
    Market,
    /// Rest until the market trades to `price` or better (maker unless it
    /// crosses on arrival).
    Limit { price: f32 },
    /// Trigger a market order once `stop` is touched.
    StopMarket { stop: f32 },
    /// Trigger a resting limit at `limit` once `stop` is touched.
    StopLimit { stop: f32, limit: f32 },
    /// A take-profit — economically a limit order at `price`.
    TakeProfit { price: f32 },
}

impl OrderType {
    /// The resting/limit price if the type has one (for maker/post-only checks).
    pub fn limit_price(&self) -> Option<f32> {
        match self
        {
            OrderType::Limit { price } | OrderType::TakeProfit { price } => Some(*price),
            OrderType::StopLimit { limit, .. } => Some(*limit),
            _ => None,
        }
    }
}

/// Time-in-force.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TimeInForce {
    /// Good-till-cancelled: rests across bars.
    Gtc,
    /// Immediate-or-cancel: fill what crosses this bar, drop the rest.
    Ioc,
    /// Fill-or-kill: fill entirely on arrival or cancel wholly.
    Fok,
}

/// Lifecycle status of an order.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OrderStatus {
    New,
    Triggered,
    PartiallyFilled,
    Filled,
    Canceled,
    Rejected,
}

/// Maker/taker fee schedule in basis points (1 bp = 0.01 %).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FeeSchedule {
    pub maker_bps: f32,
    pub taker_bps: f32,
}

impl Default for FeeSchedule {
    /// A conservative default in the range of a VIP-0 crypto perp tier
    /// (taker 5 bps, maker 1 bp).
    fn default() -> Self {
        Self {
            maker_bps: 1.0,
            taker_bps: 5.0,
        }
    }
}

impl FeeSchedule {
    /// Fee charged on a fill of `notional` (= price·qty), in quote currency.
    pub fn fee(&self, notional: f32, taker: bool) -> f32 {
        let bps = if taker { self.taker_bps } else { self.maker_bps };
        notional.abs() * bps / 10_000.0
    }
}

/// A constant-plus-linear slippage model. Taker fills pay `base_bps` of
/// half-spread cost plus `impact_bps_per_unit · (size / reference_size)`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SlippageModel {
    /// Fixed cost applied to every taker fill (bps of price).
    pub base_bps: f32,
    /// Linear market-impact coefficient (bps per unit of `size / ref_liquidity`).
    pub impact_bps: f32,
    /// Reference liquidity used to normalise `size` in the impact term.
    pub ref_liquidity: f32,
}

impl Default for SlippageModel {
    fn default() -> Self {
        Self {
            base_bps: 1.0,
            impact_bps: 0.0,
            ref_liquidity: 1.0,
        }
    }
}

impl SlippageModel {
    /// Slippage in bps for a taker order of `size` base units.
    pub fn slippage_bps(&self, size: f32) -> f32 {
        let ref_liq = self.ref_liquidity.max(1e-9);
        self.base_bps + self.impact_bps * (size.abs() / ref_liq)
    }

    /// Apply slippage to a reference price: buys pay up, sells receive less.
    pub fn adjust(&self, reference: f32, side: Side, size: f32) -> f32 {
        let frac = self.slippage_bps(size) / 10_000.0;
        reference * (1.0 + side.sign() * frac)
    }
}

/// A single execution.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Fill {
    pub price: f32,
    pub qty: f32,
    pub fee: f32,
    /// True if this fill removed liquidity (crossed the spread).
    pub taker: bool,
    pub ts_ms: i64,
}

impl Fill {
    /// Signed cash flow to the account from this fill (quote currency),
    /// **including fees**: buying spends cash, selling receives it, and the fee
    /// is always a debit.
    pub fn cash_flow(&self, side: Side) -> f32 {
        -side.sign() * self.price * self.qty - self.fee
    }
}

/// Instrument trading rules — tick size (price granularity), step size
/// (quantity granularity), and minimum notional. Real venues reject orders
/// that violate these, so the paper engine honours them too.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Instrument {
    pub tick_size: f32,
    pub step_size: f32,
    pub min_notional: f32,
}

impl Default for Instrument {
    fn default() -> Self {
        Self {
            tick_size: 0.01,
            step_size: 1e-6,
            min_notional: 0.0,
        }
    }
}

impl Instrument {
    /// Round a price down to the nearest tick.
    pub fn round_price(&self, price: f32) -> f32 {
        if self.tick_size <= 0.0
        {
            return price;
        }
        (price / self.tick_size).round() * self.tick_size
    }

    /// Round a quantity down to the nearest lot step.
    pub fn round_qty(&self, qty: f32) -> f32 {
        if self.step_size <= 0.0
        {
            return qty;
        }
        (qty / self.step_size).floor() * self.step_size
    }

    /// True if `price·qty` meets the minimum notional.
    pub fn meets_min_notional(&self, price: f32, qty: f32) -> bool {
        (price * qty).abs() >= self.min_notional
    }
}

/// A live order in the paper engine, carrying its mutable fill/trail state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: u64,
    pub symbol: String,
    pub side: Side,
    pub order_type: OrderType,
    pub qty: f32,
    pub tif: TimeInForce,
    pub reduce_only: bool,
    pub post_only: bool,
    pub status: OrderStatus,
    pub filled_qty: f32,
    pub avg_fill_price: f32,
    pub ts_ms: i64,
    /// Whether a stop order has already been triggered (armed to fill).
    pub triggered: bool,
}

impl Order {
    /// Build a market order.
    pub fn market(id: u64, symbol: &str, side: Side, qty: f32) -> Self {
        Self::new(id, symbol, side, OrderType::Market, qty)
    }

    /// Build a limit order (GTC by default).
    pub fn limit(id: u64, symbol: &str, side: Side, qty: f32, price: f32) -> Self {
        Self::new(id, symbol, side, OrderType::Limit { price }, qty)
    }

    pub fn new(id: u64, symbol: &str, side: Side, order_type: OrderType, qty: f32) -> Self {
        Self {
            id,
            symbol: symbol.to_string(),
            side,
            order_type,
            qty,
            tif: TimeInForce::Gtc,
            reduce_only: false,
            post_only: false,
            status: OrderStatus::New,
            filled_qty: 0.0,
            avg_fill_price: 0.0,
            ts_ms: 0,
            triggered: false,
        }
    }

    pub fn with_tif(mut self, tif: TimeInForce) -> Self {
        self.tif = tif;
        self
    }

    pub fn reduce_only(mut self) -> Self {
        self.reduce_only = true;
        self
    }

    pub fn post_only(mut self) -> Self {
        self.post_only = true;
        self
    }

    pub fn remaining(&self) -> f32 {
        (self.qty - self.filled_qty).max(0.0)
    }

    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::New | OrderStatus::Triggered | OrderStatus::PartiallyFilled
        )
    }

    /// Record a fill against this order, updating status and average price.
    pub fn apply_fill(&mut self, fill: &Fill) {
        let prev = self.filled_qty;
        let new_qty = prev + fill.qty;
        if new_qty > 1e-12
        {
            self.avg_fill_price = (self.avg_fill_price * prev + fill.price * fill.qty) / new_qty;
        }
        self.filled_qty = new_qty;
        self.status = if self.remaining() <= 1e-9
        {
            OrderStatus::Filled
        }
        else
        {
            OrderStatus::PartiallyFilled
        };
    }
}

/// Attempt to fill an order against a single execution candle.
///
/// Returns `Some(Fill)` for the executable quantity (`order.remaining()`), or
/// `None` if the order does not fill on this bar. Implements the standard
/// backtest fill rules; stop orders that are already `triggered` fall through
/// to their market/limit behaviour.
///
/// `post_only` rejects (returns `None`) if the limit would cross on arrival —
/// the caller should mark the order `Rejected`. This function does not mutate
/// the order; the engine applies the returned fill.
pub fn simulate_fill(
    order: &Order,
    candle: &Candle,
    fees: &FeeSchedule,
    slippage: &SlippageModel,
) -> Option<Fill> {
    let qty = order.remaining();
    if qty <= 1e-12
    {
        return None;
    }
    let side = order.side;
    let (o, h, l) = (candle.open, candle.high, candle.low);

    // Resolve the raw fill price per order type, or None if it doesn't fill.
    let (raw_price, taker) = match order.order_type
    {
        OrderType::Market => (o, true),
        OrderType::Limit { price } | OrderType::TakeProfit { price } =>
        {
            let crosses_on_open = match side
            {
                Side::Buy => o <= price,
                Side::Sell => o >= price,
            };
            // post-only: reject if it would take liquidity immediately.
            if order.post_only && crosses_on_open
            {
                return None;
            }
            let fills = match side
            {
                Side::Buy => l <= price,
                Side::Sell => h >= price,
            };
            if !fills
            {
                return None;
            }
            // Price improvement: a gap through the limit fills at the open.
            let px = match side
            {
                Side::Buy => o.min(price),
                Side::Sell => o.max(price),
            };
            // A limit that is already marketable on arrival (its open is on the
            // crossing side) removes liquidity — it is a taker and pays slippage
            // + the taker fee. A resting limit that only gets hit later is a
            // maker. `post_only` crossing was already rejected above.
            (px, crosses_on_open)
        },
        OrderType::StopMarket { stop } =>
        {
            let triggered = order.triggered
                || match side
                {
                    Side::Buy => h >= stop,
                    Side::Sell => l <= stop,
                };
            if !triggered
            {
                return None;
            }
            // Fill as a market order at the worse of open/stop (gap-aware).
            let px = match side
            {
                Side::Buy => o.max(stop),
                Side::Sell => o.min(stop),
            };
            (px, true)
        },
        OrderType::StopLimit { stop, limit } =>
        {
            let triggered = order.triggered
                || match side
                {
                    Side::Buy => h >= stop,
                    Side::Sell => l <= stop,
                };
            if !triggered
            {
                return None;
            }
            let fills = match side
            {
                Side::Buy => l <= limit,
                Side::Sell => h >= limit,
            };
            if !fills
            {
                return None;
            }
            // The stop triggers intrabar (after the open), so the order becomes
            // aggressive (a taker) as price crosses through the stop. It fills at
            // the resting limit; the limit clamp below guarantees slippage never
            // pushes the price through that limit.
            (limit, true)
        },
    };

    // Taker fills pay slippage; resting (maker) fills do not.
    let mut price = if taker
    {
        slippage.adjust(raw_price, side, qty)
    }
    else
    {
        raw_price
    };
    // A limit/stop-limit price is a hard cap: slippage must never fill through
    // it (a buy never above its limit, a sell never below).
    if let Some(limit) = order.order_type.limit_price()
    {
        price = match side
        {
            Side::Buy => price.min(limit),
            Side::Sell => price.max(limit),
        };
    }
    let notional = price * qty;
    let fee = fees.fee(notional, taker);
    Some(Fill {
        price,
        qty,
        fee,
        taker,
        ts_ms: candle.ts_ms,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candle(o: f32, h: f32, l: f32, c: f32) -> Candle {
        Candle {
            ts_ms: 1000,
            open: o,
            high: h,
            low: l,
            close: c,
            volume: 100.0,
        }
    }

    #[test]
    fn side_helpers() {
        assert_eq!(Side::Buy.sign(), 1.0);
        assert_eq!(Side::Sell.sign(), -1.0);
        assert_eq!(Side::Buy.opposite(), Side::Sell);
    }

    #[test]
    fn fee_schedule_bps() {
        let f = FeeSchedule {
            maker_bps: 1.0,
            taker_bps: 5.0,
        };
        // 10 bps of 10_000 notional = 5.0 for taker, 1.0 for maker
        assert!((f.fee(10_000.0, true) - 5.0).abs() < 1e-4);
        assert!((f.fee(10_000.0, false) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn slippage_pushes_against_taker() {
        let s = SlippageModel {
            base_bps: 10.0,
            impact_bps: 0.0,
            ref_liquidity: 1.0,
        };
        let buy = s.adjust(100.0, Side::Buy, 1.0);
        let sell = s.adjust(100.0, Side::Sell, 1.0);
        assert!(buy > 100.0, "buy should pay up: {buy}");
        assert!(sell < 100.0, "sell should receive less: {sell}");
        assert!((buy - 100.1).abs() < 1e-3); // 10 bps
    }

    #[test]
    fn market_fills_at_open_with_slippage() {
        let o = Order::market(1, "BTC", Side::Buy, 1.0);
        let f = simulate_fill(&o, &candle(100.0, 101.0, 99.0, 100.5), &Default::default(), &Default::default())
            .unwrap();
        assert!(f.taker);
        assert!(f.price >= 100.0); // open + slippage
        assert_eq!(f.qty, 1.0);
    }

    #[test]
    fn limit_buy_fills_only_if_low_touches() {
        let o = Order::limit(1, "BTC", Side::Buy, 1.0, 99.5);
        // Bar dips to 99 -> fills at min(open, limit)=99.5
        let f = simulate_fill(&o, &candle(100.0, 100.5, 99.0, 100.0), &Default::default(), &Default::default());
        assert!(f.is_some());
        assert!((f.unwrap().price - 99.5).abs() < 1e-4);
        // Bar never dips to 99.5 -> no fill.
        let none = simulate_fill(&o, &candle(100.0, 101.0, 99.6, 100.5), &Default::default(), &Default::default());
        assert!(none.is_none());
    }

    #[test]
    fn limit_buy_gap_is_marketable_taker_at_open() {
        // Open already below the limit -> the limit is marketable on arrival, so
        // it fills at the open as a TAKER (removes liquidity). With no slippage
        // the fill is exactly the open; the limit clamp keeps it at/under 99.5.
        let no_slip = SlippageModel { base_bps: 0.0, impact_bps: 0.0, ref_liquidity: 1.0 };
        let o = Order::limit(1, "BTC", Side::Buy, 1.0, 99.5);
        let f = simulate_fill(&o, &candle(99.0, 99.8, 98.0, 99.2), &Default::default(), &no_slip).unwrap();
        assert!((f.price - 99.0).abs() < 1e-4, "gap fill at open, got {}", f.price);
        assert!(f.taker, "a limit that crosses on arrival is a taker");
        assert!(f.price <= 99.5 + 1e-6, "fill must never exceed the limit");
    }

    #[test]
    fn resting_limit_is_maker() {
        // Open above the limit -> the buy limit rests and is only hit intrabar,
        // so it is a maker with no slippage.
        let o = Order::limit(1, "BTC", Side::Buy, 1.0, 99.5);
        let f = simulate_fill(&o, &candle(100.0, 100.5, 99.0, 100.0), &Default::default(), &Default::default())
            .unwrap();
        assert!(!f.taker, "a resting limit is a maker");
        assert!((f.price - 99.5).abs() < 1e-4);
    }

    #[test]
    fn limit_sell_fills_if_high_reaches() {
        let o = Order::limit(1, "BTC", Side::Sell, 1.0, 101.0);
        let f = simulate_fill(&o, &candle(100.0, 101.5, 99.5, 100.5), &Default::default(), &Default::default());
        assert!(f.is_some());
        assert!((f.unwrap().price - 101.0).abs() < 1e-4);
    }

    #[test]
    fn post_only_rejected_when_crossing() {
        // Buy limit at 101 with open 100 -> would cross immediately -> reject.
        let o = Order::limit(1, "BTC", Side::Buy, 1.0, 101.0).post_only();
        let none = simulate_fill(&o, &candle(100.0, 102.0, 99.0, 100.0), &Default::default(), &Default::default());
        assert!(none.is_none());
    }

    #[test]
    fn stop_buy_triggers_on_high() {
        let o = Order::new(1, "BTC", Side::Buy, OrderType::StopMarket { stop: 105.0 }, 1.0);
        // High reaches 106 -> triggers, fills at max(open, stop)=105 + slippage.
        let f = simulate_fill(&o, &candle(102.0, 106.0, 101.0, 105.5), &Default::default(), &Default::default());
        assert!(f.is_some());
        let fill = f.unwrap();
        assert!(fill.taker);
        assert!(fill.price >= 105.0);
        // High never reaches stop -> no trigger.
        let none = simulate_fill(&o, &candle(102.0, 104.0, 101.0, 103.0), &Default::default(), &Default::default());
        assert!(none.is_none());
    }

    #[test]
    fn stop_limit_needs_trigger_then_cross() {
        let o = Order::new(
            1,
            "BTC",
            Side::Sell,
            OrderType::StopLimit { stop: 95.0, limit: 94.0 },
            1.0,
        );
        // Low hits 93 (<=95 trigger) and <=94 limit -> fills at max(open, limit).
        let f = simulate_fill(&o, &candle(96.0, 96.5, 93.0, 94.5), &Default::default(), &Default::default());
        assert!(f.is_some());
        assert!((f.unwrap().price - 94.0).abs() < 1e-4);
    }

    #[test]
    fn instrument_rounding_and_min_notional() {
        let inst = Instrument {
            tick_size: 0.5,
            step_size: 0.1,
            min_notional: 10.0,
        };
        assert!((inst.round_price(100.3) - 100.5).abs() < 1e-4);
        assert!((inst.round_qty(1.27) - 1.2).abs() < 1e-4);
        assert!(inst.meets_min_notional(100.0, 0.2));
        assert!(!inst.meets_min_notional(100.0, 0.05));
    }

    #[test]
    fn apply_fill_updates_average_and_status() {
        let mut o = Order::market(1, "BTC", Side::Buy, 2.0);
        o.apply_fill(&Fill { price: 100.0, qty: 1.0, fee: 0.1, taker: true, ts_ms: 1 });
        assert_eq!(o.status, OrderStatus::PartiallyFilled);
        o.apply_fill(&Fill { price: 102.0, qty: 1.0, fee: 0.1, taker: true, ts_ms: 2 });
        assert_eq!(o.status, OrderStatus::Filled);
        assert!((o.avg_fill_price - 101.0).abs() < 1e-4);
    }

    #[test]
    fn fill_cash_flow_signs() {
        let f = Fill { price: 100.0, qty: 1.0, fee: 0.5, taker: true, ts_ms: 1 };
        // Buying: spend 100 + 0.5 fee = -100.5
        assert!((f.cash_flow(Side::Buy) - (-100.5)).abs() < 1e-4);
        // Selling: receive 100 - 0.5 fee = +99.5
        assert!((f.cash_flow(Side::Sell) - 99.5).abs() < 1e-4);
    }
}
