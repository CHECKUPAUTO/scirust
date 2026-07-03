//! High-frequency microstructure signals — the alpha a fast small-order trader
//! reads off the tape and the top of book.
//!
//! * **Order-Flow Imbalance (OFI)** — Cont–Kukanov–Stoikov: the net change in
//!   best-level depth, the strongest short-horizon predictor of the next price
//!   move.
//! * **Trade-flow imbalance** — signed traded volume `(buy − sell)/(buy + sell)`.
//! * **VPIN** — volume-synchronised probability of informed trading (flow
//!   toxicity), via bulk-volume classification.
//! * **Kyle's λ** — price impact per unit of signed volume, from an OLS fit.
//!
//! All are pure reductions over a slice of L1 quotes or trades. These are the
//! features that tell a market-making / execution agent whether to lean, widen,
//! or pull its quotes.

use serde::{Deserialize, Serialize};

/// One level-1 (top-of-book) quote snapshot.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct L1Quote {
    pub bid_px: f32,
    pub bid_qty: f32,
    pub ask_px: f32,
    pub ask_qty: f32,
}

/// A single trade print. `buyer_is_taker = true` means the aggressor bought
/// (an up-tick in flow), `false` means the aggressor sold.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct TradePrint {
    pub price: f32,
    pub qty: f32,
    pub buyer_is_taker: bool,
}

/// Per-update Order-Flow Imbalance (Cont–Kukanov–Stoikov). For consecutive L1
/// quotes the OFI increment is
///
/// ```text
/// e = [ q_b·1(P_b≥P_b⁻) − q_b⁻·1(P_b≤P_b⁻) ]      (bid contribution)
///   − [ q_a·1(P_a≤P_a⁻) − q_a⁻·1(P_a≥P_a⁻) ]      (ask contribution)
/// ```
///
/// where `⁻` denotes the previous snapshot. Positive ⇒ net buy pressure.
pub fn ofi_increment(prev: &L1Quote, cur: &L1Quote) -> f32 {
    let bid = if cur.bid_px > prev.bid_px
    {
        cur.bid_qty
    }
    else if cur.bid_px < prev.bid_px
    {
        -prev.bid_qty
    }
    else
    {
        cur.bid_qty - prev.bid_qty
    };
    let ask = if cur.ask_px < prev.ask_px
    {
        cur.ask_qty
    }
    else if cur.ask_px > prev.ask_px
    {
        -prev.ask_qty
    }
    else
    {
        cur.ask_qty - prev.ask_qty
    };
    bid - ask
}

/// Cumulative OFI over a series of L1 quotes (sum of per-update increments).
pub fn order_flow_imbalance(quotes: &[L1Quote]) -> f32 {
    if quotes.len() < 2
    {
        return 0.0;
    }
    let mut acc = 0.0f32;
    for w in quotes.windows(2)
    {
        acc += ofi_increment(&w[0], &w[1]);
    }
    acc
}

/// Signed trade-flow imbalance `(buyVol − sellVol)/(buyVol + sellVol)` in
/// `[-1, 1]` over a set of trade prints.
pub fn trade_flow_imbalance(trades: &[TradePrint]) -> f32 {
    let mut buy = 0.0f32;
    let mut sell = 0.0f32;
    for t in trades
    {
        if t.buyer_is_taker
        {
            buy += t.qty;
        }
        else
        {
            sell += t.qty;
        }
    }
    let denom = buy + sell;
    if denom <= 1e-12
    {
        return 0.0;
    }
    (buy - sell) / denom
}

/// VPIN (flow toxicity) via bulk-volume classification.
///
/// Trades are packed into `num_buckets` equal-volume buckets of size
/// `bucket_volume`. Within each bucket the buy fraction is estimated from the
/// standardised price change through the normal CDF (Easley–López de
/// Prado–O'Hara), and `VPIN = mean_j |V_buy_j − V_sell_j| / V_bucket`.
///
/// Returns `None` if there is not enough volume to fill a single bucket.
pub fn vpin(prices: &[f32], volumes: &[f32], bucket_volume: f32, num_buckets: usize) -> Option<f32> {
    let n = prices.len();
    if n < 2 || bucket_volume <= 0.0 || volumes.len() != n
    {
        return None;
    }
    // Price-change volatility for standardisation (sample stdev of diffs).
    let mut diffs = Vec::with_capacity(n - 1);
    for i in 1..n
    {
        diffs.push(prices[i] - prices[i - 1]);
    }
    let mean_d = diffs.iter().sum::<f32>() / diffs.len() as f32;
    let var_d = diffs.iter().map(|d| (d - mean_d).powi(2)).sum::<f32>() / diffs.len().max(1) as f32;
    let sigma = var_d.sqrt().max(1e-9);

    // Accumulate volume into buckets, splitting buy/sell by the BVC weight.
    let mut buckets: Vec<f32> = Vec::new(); // stores |buy - sell| per completed bucket
    let mut cur_buy = 0.0f32;
    let mut cur_sell = 0.0f32;
    let mut cur_vol = 0.0f32;
    for i in 1..n
    {
        let dp = prices[i] - prices[i - 1];
        let z = dp / sigma;
        let buy_frac = normal_cdf(z);
        let mut v = volumes[i];
        // A single trade's volume may straddle a bucket boundary; split it.
        while v > 1e-12
        {
            let room = bucket_volume - cur_vol;
            let take = v.min(room);
            cur_buy += take * buy_frac;
            cur_sell += take * (1.0 - buy_frac);
            cur_vol += take;
            v -= take;
            if cur_vol >= bucket_volume - 1e-9
            {
                buckets.push((cur_buy - cur_sell).abs());
                cur_buy = 0.0;
                cur_sell = 0.0;
                cur_vol = 0.0;
            }
        }
    }
    if buckets.is_empty()
    {
        return None;
    }
    let take = buckets.len().min(num_buckets.max(1));
    let tail = &buckets[buckets.len() - take..];
    let sum: f32 = tail.iter().sum();
    Some((sum / take as f32) / bucket_volume)
}

/// Kyle's λ — price impact per unit of signed volume — from an OLS regression of
/// mid-price changes on signed traded volume. Returns the slope (≥0 for a normal
/// impact). Needs matched `signed_volume[i]` and `mid_change[i]` samples.
pub fn kyle_lambda(signed_volume: &[f32], mid_change: &[f32]) -> f32 {
    let n = signed_volume.len();
    if n < 2 || mid_change.len() != n
    {
        return 0.0;
    }
    let mx = signed_volume.iter().sum::<f32>() / n as f32;
    let my = mid_change.iter().sum::<f32>() / n as f32;
    let mut cov = 0.0f32;
    let mut var = 0.0f32;
    for i in 0..n
    {
        let dx = signed_volume[i] - mx;
        cov += dx * (mid_change[i] - my);
        var += dx * dx;
    }
    if var < 1e-12
    {
        return 0.0;
    }
    cov / var
}

/// Standard normal CDF via the erf approximation (Abramowitz & Stegun 7.1.26).
fn normal_cdf(x: f32) -> f32 {
    0.5 * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

fn erf(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26. Computed in f64 so the polynomial constants
    // keep their full precision, then narrowed to f32.
    let xf = x as f64;
    let sign = if xf < 0.0 { -1.0 } else { 1.0 };
    let xa = xf.abs();
    let t = 1.0 / (1.0 + 0.3275911 * xa);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-xa * xa).exp();
    (sign * y) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ofi_positive_on_bid_lift() {
        let prev = L1Quote { bid_px: 100.0, bid_qty: 5.0, ask_px: 101.0, ask_qty: 5.0 };
        // Bid price ticks up (aggressive buying interest) -> positive OFI.
        let cur = L1Quote { bid_px: 100.5, bid_qty: 6.0, ask_px: 101.0, ask_qty: 5.0 };
        assert!(ofi_increment(&prev, &cur) > 0.0);
    }

    #[test]
    fn ofi_negative_on_ask_drop() {
        let prev = L1Quote { bid_px: 100.0, bid_qty: 5.0, ask_px: 101.0, ask_qty: 5.0 };
        // Ask price ticks down (aggressive selling) -> negative OFI.
        let cur = L1Quote { bid_px: 100.0, bid_qty: 5.0, ask_px: 100.5, ask_qty: 6.0 };
        assert!(ofi_increment(&prev, &cur) < 0.0);
    }

    #[test]
    fn cumulative_ofi_sums() {
        let quotes = vec![
            L1Quote { bid_px: 100.0, bid_qty: 5.0, ask_px: 101.0, ask_qty: 5.0 },
            L1Quote { bid_px: 100.5, bid_qty: 6.0, ask_px: 101.0, ask_qty: 5.0 },
            L1Quote { bid_px: 100.5, bid_qty: 8.0, ask_px: 101.0, ask_qty: 5.0 },
        ];
        assert!(order_flow_imbalance(&quotes) > 0.0);
    }

    #[test]
    fn trade_flow_imbalance_bounds() {
        let trades = vec![
            TradePrint { price: 100.0, qty: 3.0, buyer_is_taker: true },
            TradePrint { price: 100.0, qty: 1.0, buyer_is_taker: false },
        ];
        let imb = trade_flow_imbalance(&trades);
        assert!((imb - 0.5).abs() < 1e-4); // (3-1)/4
        assert!((-1.0..=1.0).contains(&imb));
    }

    #[test]
    fn vpin_in_unit_range() {
        // Alternating up/down moves with volume -> VPIN computable and in [0,1].
        let prices: Vec<f32> = (0..100).map(|i| 100.0 + (i as f32 * 0.5).sin()).collect();
        let volumes = vec![10.0f32; 100];
        let v = vpin(&prices, &volumes, 50.0, 5).unwrap();
        assert!((0.0..=1.0).contains(&v), "VPIN out of range: {v}");
    }

    #[test]
    fn vpin_none_without_enough_volume() {
        let prices = vec![100.0, 101.0];
        let volumes = vec![1.0, 1.0];
        assert!(vpin(&prices, &volumes, 1000.0, 5).is_none());
    }

    #[test]
    fn kyle_lambda_recovers_slope() {
        // mid_change = 0.5 * signed_volume exactly -> lambda = 0.5.
        let sv: Vec<f32> = (-10..10).map(|i| i as f32).collect();
        let mc: Vec<f32> = sv.iter().map(|v| 0.5 * v).collect();
        assert!((kyle_lambda(&sv, &mc) - 0.5).abs() < 1e-4);
    }

    #[test]
    fn normal_cdf_symmetry() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-3);
        assert!((normal_cdf(10.0) - 1.0).abs() < 1e-3);
        assert!(normal_cdf(-10.0).abs() < 1e-3);
    }
}
