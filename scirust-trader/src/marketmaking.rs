//! Optimal market making — the quote-generation layer for a liquidity-providing
//! agent that posts fast, small two-sided micro-orders.
//!
//! Implements the Avellaneda–Stoikov (2008) model: from the mid price `s`, the
//! current signed inventory `q`, the time remaining, and three parameters
//! (risk aversion `γ`, volatility `σ`, order-arrival decay `κ`) it produces a
//! **reservation price** that skews away from inventory and an **optimal spread**
//! around it:
//!
//! ```text
//! reservation r = s − q·γ·σ²·(T−t)
//! optimal spread δ = γ·σ²·(T−t) + (2/γ)·ln(1 + γ/κ)
//! bid = r − δ/2 ,  ask = r + δ/2
//! ```
//!
//! As inventory grows long, `r` (and both quotes) shift down so the ask is more
//! likely to be hit and the position mean-reverts to flat — the core
//! inventory-risk control. A Guéant–Lehalle–Fernandez-Tapia style closed-form
//! spread is also provided for the infinite-horizon / stationary regime.

use serde::{Deserialize, Serialize};

/// Market-making model parameters.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MmParams {
    /// Inventory risk aversion γ (> 0). Larger ⇒ tighter inventory control,
    /// wider skew.
    pub gamma: f32,
    /// Price volatility σ (per unit time, same clock as `time_remaining`).
    pub sigma: f32,
    /// Order-arrival decay κ (liquidity/fill-intensity parameter, > 0).
    pub kappa: f32,
}

impl Default for MmParams {
    fn default() -> Self {
        Self {
            gamma: 0.1,
            sigma: 2.0,
            kappa: 1.5,
        }
    }
}

/// A two-sided quote plus the intermediate quantities the agent can reason about.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Quotes {
    pub bid: f32,
    pub ask: f32,
    /// Inventory-skewed reservation (fair) price.
    pub reservation_price: f32,
    /// Total optimal spread `ask − bid`.
    pub spread: f32,
    /// Skew of the reservation price vs the mid (`reservation − mid`); negative
    /// when long inventory (quotes pushed down to sell).
    pub skew: f32,
    /// Half-spread on the bid side.
    pub bid_offset: f32,
    /// Half-spread on the ask side.
    pub ask_offset: f32,
}

/// Avellaneda–Stoikov reservation price: `s − q·γ·σ²·(T−t)`.
pub fn reservation_price(mid: f32, inventory: f32, time_remaining: f32, p: &MmParams) -> f32 {
    mid - inventory * p.gamma * p.sigma * p.sigma * time_remaining.max(0.0)
}

/// Avellaneda–Stoikov optimal total spread:
/// `γ·σ²·(T−t) + (2/γ)·ln(1 + γ/κ)`.
pub fn optimal_spread(time_remaining: f32, p: &MmParams) -> f32 {
    let g = p.gamma.max(1e-9);
    let k = p.kappa.max(1e-9);
    g * p.sigma * p.sigma * time_remaining.max(0.0) + (2.0 / g) * (1.0 + g / k).ln()
}

/// Full Avellaneda–Stoikov quotes for the given state. `time_remaining` is
/// `T − t` in the same time units as `σ` (use `1.0` for a stationary desk).
pub fn optimal_quotes(mid: f32, inventory: f32, time_remaining: f32, p: &MmParams) -> Quotes {
    let r = reservation_price(mid, inventory, time_remaining, p);
    let spread = optimal_spread(time_remaining, p).max(0.0);
    let half = spread / 2.0;
    let bid = r - half;
    let ask = r + half;
    Quotes {
        bid,
        ask,
        reservation_price: r,
        spread,
        skew: r - mid,
        bid_offset: mid - bid,
        ask_offset: ask - mid,
    }
}

/// Guéant–Lehalle–Fernandez-Tapia stationary (infinite-horizon) approximation of
/// the optimal half-spread, independent of a terminal time:
/// `(1/γ)·ln(1 + γ/κ) + ½·√( (σ²·γ) / (2·κ·A) )·(2q±1)` — here we return the
/// symmetric base half-spread `(1/γ)·ln(1+γ/κ) + ½·√(σ²γ/(2κA))` and the caller
/// applies the inventory tilt via [`optimal_quotes`]. `arrival_a` is the base
/// order-flow intensity `A`.
pub fn glft_half_spread(p: &MmParams, arrival_a: f32) -> f32 {
    let g = p.gamma.max(1e-9);
    let k = p.kappa.max(1e-9);
    let a = arrival_a.max(1e-9);
    (1.0 / g) * (1.0 + g / k).ln() + 0.5 * ((p.sigma * p.sigma * g) / (2.0 * k * a)).max(0.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservation_price_skews_against_inventory() {
        let p = MmParams::default();
        // Long inventory -> reservation below mid (want to sell).
        let r_long = reservation_price(100.0, 5.0, 1.0, &p);
        assert!(r_long < 100.0);
        // Short inventory -> reservation above mid (want to buy).
        let r_short = reservation_price(100.0, -5.0, 1.0, &p);
        assert!(r_short > 100.0);
        // Flat -> reservation == mid.
        assert!((reservation_price(100.0, 0.0, 1.0, &p) - 100.0).abs() < 1e-4);
    }

    #[test]
    fn optimal_spread_positive_and_grows_with_time() {
        let p = MmParams::default();
        let s_short = optimal_spread(0.1, &p);
        let s_long = optimal_spread(1.0, &p);
        assert!(s_short > 0.0);
        assert!(s_long > s_short, "more time-to-horizon widens the spread");
    }

    #[test]
    fn quotes_bracket_reservation_and_skew_with_inventory() {
        let p = MmParams::default();
        let flat = optimal_quotes(100.0, 0.0, 1.0, &p);
        assert!(flat.bid < flat.reservation_price && flat.reservation_price < flat.ask);
        assert!((flat.skew).abs() < 1e-4);
        // Long inventory -> both quotes shift down vs the flat case.
        let long = optimal_quotes(100.0, 5.0, 1.0, &p);
        assert!(long.bid < flat.bid);
        assert!(long.ask < flat.ask);
        assert!(long.skew < 0.0);
        // Spread is the same regardless of inventory (AS property).
        assert!((long.spread - flat.spread).abs() < 1e-3);
    }

    #[test]
    fn glft_half_spread_is_positive() {
        let p = MmParams::default();
        assert!(glft_half_spread(&p, 140.0) > 0.0);
    }

    #[test]
    fn higher_risk_aversion_controls_inventory_harder() {
        let cautious = MmParams { gamma: 0.5, ..Default::default() };
        let bold = MmParams { gamma: 0.05, ..Default::default() };
        let q_cautious = optimal_quotes(100.0, 5.0, 1.0, &cautious);
        let q_bold = optimal_quotes(100.0, 5.0, 1.0, &bold);
        // More risk-averse -> larger inventory skew (reservation further from mid).
        assert!(q_cautious.skew.abs() > q_bold.skew.abs());
    }
}
