//! Trading strategies — the signal layer.
//!
//! A [`Strategy`] maps a window of candles to a [`Signal`]: the **desired target
//! position** (long / short / flat) on the *last* closed bar, a `strength` in
//! `[0, 1]`, and a human-readable `reason`. Returning a target position (rather
//! than raw buy/sell events) lets the backtester and the live agent reconcile to
//! it idempotently. Every strategy evaluates only on closed bars — the
//! backtester executes the signal on the *next* bar to avoid look-ahead.
//!
//! Strategies are constructible by name via [`strategy_from_spec`], so the MCP
//! layer and the opportunity [`crate::scanner`] can spin one up from a string +
//! parameter map the agent supplies in natural language.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::agent::Action;
use crate::indicators;
use crate::market::Candle;

/// A strategy's opinion on the latest bar.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Signal {
    /// Desired target position.
    pub action: Action,
    /// Conviction in `[0, 1]`.
    pub strength: f32,
    /// Why — indicator values that drove the call.
    pub reason: String,
}

impl Signal {
    pub fn new(action: Action, strength: f32, reason: impl Into<String>) -> Self {
        Self {
            action,
            strength: strength.clamp(0.0, 1.0),
            reason: reason.into(),
        }
    }

    pub fn flat(reason: impl Into<String>) -> Self {
        Self::new(Action::Flat, 0.0, reason)
    }
}

/// The strategy interface.
pub trait Strategy {
    /// A stable identifier (e.g. `"sma_cross(10,30)"`).
    fn name(&self) -> String;

    /// Bars of history required before the first meaningful signal.
    fn warmup(&self) -> usize;

    /// Evaluate on `candles`, returning the target position implied by the last
    /// closed bar. `candles` must be chronological.
    fn evaluate(&self, candles: &[Candle]) -> Signal;
}

fn closes(candles: &[Candle]) -> Vec<f32> {
    candles.iter().map(|c| c.close).collect()
}

// ---------------------------------------------------------------------------
// Moving-average crossover (trend following).
// ---------------------------------------------------------------------------

/// SMA (or EMA) crossover: long while the fast average is above the slow one.
#[derive(Debug, Clone)]
pub struct MaCross {
    pub fast: usize,
    pub slow: usize,
    pub exponential: bool,
}

impl MaCross {
    pub fn sma(fast: usize, slow: usize) -> Self {
        Self { fast, slow, exponential: false }
    }
    pub fn ema(fast: usize, slow: usize) -> Self {
        Self { fast, slow, exponential: true }
    }
}

impl Strategy for MaCross {
    fn name(&self) -> String {
        format!(
            "{}_cross({},{})",
            if self.exponential { "ema" } else { "sma" },
            self.fast,
            self.slow
        )
    }
    fn warmup(&self) -> usize {
        self.slow + 2
    }
    fn evaluate(&self, candles: &[Candle]) -> Signal {
        let c = closes(candles);
        let n = c.len();
        if n < self.slow + 1
        {
            return Signal::flat("insufficient history");
        }
        let ma = |p: usize| if self.exponential { indicators::ema(&c, p) } else { indicators::sma(&c, p) };
        let fast = ma(self.fast);
        let slow = ma(self.slow);
        let i = n - 1;
        let (f, s) = (fast[i], slow[i]);
        if f.is_nan() || s.is_nan()
        {
            return Signal::flat("warming up");
        }
        let gap = (f - s) / s.abs().max(1e-9);
        let strength = (gap.abs() * 50.0).clamp(0.0, 1.0);
        if f > s
        {
            Signal::new(Action::Long, strength, format!("fast {f:.2} > slow {s:.2}"))
        }
        else
        {
            Signal::new(Action::Short, strength, format!("fast {f:.2} < slow {s:.2}"))
        }
    }
}

// ---------------------------------------------------------------------------
// RSI mean-reversion.
// ---------------------------------------------------------------------------

/// RSI mean-reversion: long when oversold, short when overbought, else flat.
#[derive(Debug, Clone)]
pub struct RsiReversion {
    pub period: usize,
    pub oversold: f32,
    pub overbought: f32,
}

impl Default for RsiReversion {
    fn default() -> Self {
        Self { period: 14, oversold: 30.0, overbought: 70.0 }
    }
}

impl Strategy for RsiReversion {
    fn name(&self) -> String {
        format!("rsi_reversion({},{:.0},{:.0})", self.period, self.oversold, self.overbought)
    }
    fn warmup(&self) -> usize {
        self.period + 2
    }
    fn evaluate(&self, candles: &[Candle]) -> Signal {
        let c = closes(candles);
        let r = indicators::rsi(&c, self.period);
        let i = c.len().saturating_sub(1);
        let v = r.get(i).copied().unwrap_or(f32::NAN);
        if v.is_nan()
        {
            return Signal::flat("warming up");
        }
        if v < self.oversold
        {
            let strength = ((self.oversold - v) / self.oversold).clamp(0.0, 1.0);
            Signal::new(Action::Long, strength, format!("RSI {v:.1} < oversold {:.0}", self.oversold))
        }
        else if v > self.overbought
        {
            let strength = ((v - self.overbought) / (100.0 - self.overbought)).clamp(0.0, 1.0);
            Signal::new(Action::Short, strength, format!("RSI {v:.1} > overbought {:.0}", self.overbought))
        }
        else
        {
            Signal::flat(format!("RSI {v:.1} neutral"))
        }
    }
}

// ---------------------------------------------------------------------------
// MACD signal-line cross.
// ---------------------------------------------------------------------------

/// MACD: long while the MACD line is above its signal line.
#[derive(Debug, Clone)]
pub struct MacdCross {
    pub fast: usize,
    pub slow: usize,
    pub signal: usize,
}

impl Default for MacdCross {
    fn default() -> Self {
        Self { fast: 12, slow: 26, signal: 9 }
    }
}

impl Strategy for MacdCross {
    fn name(&self) -> String {
        format!("macd({},{},{})", self.fast, self.slow, self.signal)
    }
    fn warmup(&self) -> usize {
        self.slow + self.signal + 2
    }
    fn evaluate(&self, candles: &[Candle]) -> Signal {
        let c = closes(candles);
        let macd = indicators::macd_line(&c, self.fast, self.slow);
        let sig = indicators::macd_signal_line(&c, self.fast, self.slow, self.signal);
        let i = c.len().saturating_sub(1);
        let (m, s) = (macd.get(i).copied().unwrap_or(f32::NAN), sig.get(i).copied().unwrap_or(f32::NAN));
        if m.is_nan() || s.is_nan()
        {
            return Signal::flat("warming up");
        }
        let hist = m - s;
        let strength = (hist.abs() / c[i].abs().max(1e-9) * 200.0).clamp(0.0, 1.0);
        if m > s
        {
            Signal::new(Action::Long, strength, format!("MACD {m:.4} > signal {s:.4}"))
        }
        else
        {
            Signal::new(Action::Short, strength, format!("MACD {m:.4} < signal {s:.4}"))
        }
    }
}

// ---------------------------------------------------------------------------
// Bollinger breakout.
// ---------------------------------------------------------------------------

/// Bollinger breakout: long on a close above the upper band, short below the
/// lower band, flat inside. (Breakout, not mean-reversion.)
#[derive(Debug, Clone)]
pub struct BollingerBreakout {
    pub period: usize,
    pub k: f32,
}

impl Default for BollingerBreakout {
    fn default() -> Self {
        Self { period: 20, k: 2.0 }
    }
}

impl Strategy for BollingerBreakout {
    fn name(&self) -> String {
        format!("bollinger_breakout({},{:.1})", self.period, self.k)
    }
    fn warmup(&self) -> usize {
        self.period + 2
    }
    fn evaluate(&self, candles: &[Candle]) -> Signal {
        let c = closes(candles);
        let upper = indicators::bollinger_band(&c, self.period, self.k, true);
        let lower = indicators::bollinger_band(&c, self.period, self.k, false);
        let i = c.len().saturating_sub(1);
        let (u, l, px) = (
            upper.get(i).copied().unwrap_or(f32::NAN),
            lower.get(i).copied().unwrap_or(f32::NAN),
            c[i],
        );
        if u.is_nan() || l.is_nan()
        {
            return Signal::flat("warming up");
        }
        let width = (u - l).max(1e-9);
        if px > u
        {
            Signal::new(Action::Long, ((px - u) / width).clamp(0.0, 1.0), format!("close {px:.2} > upper {u:.2}"))
        }
        else if px < l
        {
            Signal::new(Action::Short, ((l - px) / width).clamp(0.0, 1.0), format!("close {px:.2} < lower {l:.2}"))
        }
        else
        {
            Signal::flat(format!("close {px:.2} inside band"))
        }
    }
}

// ---------------------------------------------------------------------------
// Donchian breakout (Turtle).
// ---------------------------------------------------------------------------

/// Donchian breakout: long on a new `period`-bar high, short on a new low. The
/// prior bar's channel is used so the breakout is not self-triggering.
#[derive(Debug, Clone)]
pub struct DonchianBreakout {
    pub period: usize,
}

impl Default for DonchianBreakout {
    fn default() -> Self {
        Self { period: 20 }
    }
}

impl Strategy for DonchianBreakout {
    fn name(&self) -> String {
        format!("donchian_breakout({})", self.period)
    }
    fn warmup(&self) -> usize {
        self.period + 2
    }
    fn evaluate(&self, candles: &[Candle]) -> Signal {
        let n = candles.len();
        if n < self.period + 1
        {
            return Signal::flat("insufficient history");
        }
        let highs: Vec<f32> = candles.iter().map(|c| c.high).collect();
        let lows: Vec<f32> = candles.iter().map(|c| c.low).collect();
        let ch = indicators::donchian(&highs, &lows, self.period);
        let i = n - 1;
        // Use the channel formed by bars up to i-1 (exclude the current bar).
        let (u, l) = (ch.upper[i - 1], ch.lower[i - 1]);
        if u.is_nan() || l.is_nan()
        {
            return Signal::flat("warming up");
        }
        let px = candles[i].close;
        if px >= u
        {
            Signal::new(Action::Long, 0.7, format!("close {px:.2} broke {}-bar high {u:.2}", self.period))
        }
        else if px <= l
        {
            Signal::new(Action::Short, 0.7, format!("close {px:.2} broke {}-bar low {l:.2}", self.period))
        }
        else
        {
            Signal::flat(format!("close {px:.2} inside Donchian channel"))
        }
    }
}

// ---------------------------------------------------------------------------
// Supertrend trend-follow.
// ---------------------------------------------------------------------------

/// Supertrend: long while the trend direction is up, short while down.
#[derive(Debug, Clone)]
pub struct SupertrendFollow {
    pub period: usize,
    pub mult: f32,
}

impl Default for SupertrendFollow {
    fn default() -> Self {
        Self { period: 10, mult: 3.0 }
    }
}

impl Strategy for SupertrendFollow {
    fn name(&self) -> String {
        format!("supertrend({},{:.1})", self.period, self.mult)
    }
    fn warmup(&self) -> usize {
        self.period + 3
    }
    fn evaluate(&self, candles: &[Candle]) -> Signal {
        let highs: Vec<f32> = candles.iter().map(|c| c.high).collect();
        let lows: Vec<f32> = candles.iter().map(|c| c.low).collect();
        let c = closes(candles);
        let st = indicators::supertrend(&highs, &lows, &c, self.period, self.mult);
        let i = c.len().saturating_sub(1);
        match st.direction.get(i).copied()
        {
            Some(1) => Signal::new(Action::Long, 0.7, format!("supertrend up, line {:.2}", st.line[i])),
            Some(-1) => Signal::new(Action::Short, 0.7, format!("supertrend down, line {:.2}", st.line[i])),
            _ => Signal::flat("warming up"),
        }
    }
}

// ---------------------------------------------------------------------------
// Time-series momentum.
// ---------------------------------------------------------------------------

/// Momentum: long when the trailing `lookback`-bar return is positive.
#[derive(Debug, Clone)]
pub struct Momentum {
    pub lookback: usize,
    /// Dead-band (fraction) around zero return that stays flat.
    pub threshold: f32,
}

impl Default for Momentum {
    fn default() -> Self {
        Self { lookback: 20, threshold: 0.0 }
    }
}

impl Strategy for Momentum {
    fn name(&self) -> String {
        format!("momentum({})", self.lookback)
    }
    fn warmup(&self) -> usize {
        self.lookback + 1
    }
    fn evaluate(&self, candles: &[Candle]) -> Signal {
        let c = closes(candles);
        let n = c.len();
        if n <= self.lookback
        {
            return Signal::flat("insufficient history");
        }
        let past = c[n - 1 - self.lookback];
        if past.abs() < 1e-9
        {
            return Signal::flat("degenerate base price");
        }
        let ret = c[n - 1] / past - 1.0;
        let strength = (ret.abs() * 10.0).clamp(0.0, 1.0);
        if ret > self.threshold
        {
            Signal::new(Action::Long, strength, format!("{}-bar return {:+.2}%", self.lookback, ret * 100.0))
        }
        else if ret < -self.threshold
        {
            Signal::new(Action::Short, strength, format!("{}-bar return {:+.2}%", self.lookback, ret * 100.0))
        }
        else
        {
            Signal::flat(format!("{}-bar return {:+.2}% flat", self.lookback, ret * 100.0))
        }
    }
}

/// The set of strategy names understood by [`strategy_from_spec`].
pub const STRATEGY_NAMES: &[&str] = &[
    "sma_cross",
    "ema_cross",
    "rsi_reversion",
    "macd",
    "bollinger_breakout",
    "donchian_breakout",
    "supertrend",
    "momentum",
];

fn param(params: &BTreeMap<String, f32>, key: &str, default: f32) -> f32 {
    params.get(key).copied().unwrap_or(default)
}

/// Construct a strategy from a name and a parameter map (the values the agent
/// specifies in natural language, e.g. `{"fast": 10, "slow": 30}`). Unknown
/// parameters are ignored; missing ones fall back to sensible defaults.
pub fn strategy_from_spec(name: &str, params: &BTreeMap<String, f32>) -> Option<Box<dyn Strategy>> {
    match name
    {
        "sma_cross" => Some(Box::new(MaCross::sma(
            param(params, "fast", 10.0) as usize,
            param(params, "slow", 30.0) as usize,
        ))),
        "ema_cross" => Some(Box::new(MaCross::ema(
            param(params, "fast", 12.0) as usize,
            param(params, "slow", 26.0) as usize,
        ))),
        "rsi_reversion" => Some(Box::new(RsiReversion {
            period: param(params, "period", 14.0) as usize,
            oversold: param(params, "oversold", 30.0),
            overbought: param(params, "overbought", 70.0),
        })),
        "macd" => Some(Box::new(MacdCross {
            fast: param(params, "fast", 12.0) as usize,
            slow: param(params, "slow", 26.0) as usize,
            signal: param(params, "signal", 9.0) as usize,
        })),
        "bollinger_breakout" => Some(Box::new(BollingerBreakout {
            period: param(params, "period", 20.0) as usize,
            k: param(params, "k", 2.0),
        })),
        "donchian_breakout" => Some(Box::new(DonchianBreakout {
            period: param(params, "period", 20.0) as usize,
        })),
        "supertrend" => Some(Box::new(SupertrendFollow {
            period: param(params, "period", 10.0) as usize,
            mult: param(params, "mult", 3.0),
        })),
        "momentum" => Some(Box::new(Momentum {
            lookback: param(params, "lookback", 20.0) as usize,
            threshold: param(params, "threshold", 0.0),
        })),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candle(close: f32) -> Candle {
        Candle {
            ts_ms: 0,
            open: close,
            high: close + 1.0,
            low: close - 1.0,
            close,
            volume: 100.0,
        }
    }

    fn series(values: &[f32]) -> Vec<Candle> {
        values.iter().map(|&v| candle(v)).collect()
    }

    #[test]
    fn sma_cross_long_in_uptrend() {
        let candles = series(&(0..60).map(|i| 100.0 + i as f32).collect::<Vec<_>>());
        let s = MaCross::sma(10, 30);
        let sig = s.evaluate(&candles);
        assert_eq!(sig.action, Action::Long);
    }

    #[test]
    fn sma_cross_short_in_downtrend() {
        let candles = series(&(0..60).map(|i| 200.0 - i as f32).collect::<Vec<_>>());
        let s = MaCross::sma(10, 30);
        assert_eq!(s.evaluate(&candles).action, Action::Short);
    }

    #[test]
    fn rsi_reversion_flags_extremes() {
        // Strong uptrend -> RSI high -> short (mean reversion).
        let up = series(&(0..40).map(|i| 100.0 + i as f32).collect::<Vec<_>>());
        assert_eq!(RsiReversion::default().evaluate(&up).action, Action::Short);
        // Strong downtrend -> RSI low -> long.
        let down = series(&(0..40).map(|i| 200.0 - i as f32).collect::<Vec<_>>());
        assert_eq!(RsiReversion::default().evaluate(&down).action, Action::Long);
    }

    #[test]
    fn momentum_direction() {
        let up = series(&(0..40).map(|i| 100.0 + i as f32).collect::<Vec<_>>());
        assert_eq!(Momentum::default().evaluate(&up).action, Action::Long);
    }

    #[test]
    fn supertrend_follows_trend() {
        let up = series(&(0..40).map(|i| 100.0 + i as f32 * 2.0).collect::<Vec<_>>());
        assert_eq!(SupertrendFollow::default().evaluate(&up).action, Action::Long);
    }

    #[test]
    fn factory_builds_all_named_strategies() {
        for name in STRATEGY_NAMES
        {
            let s = strategy_from_spec(name, &BTreeMap::new());
            assert!(s.is_some(), "factory should build {name}");
        }
        assert!(strategy_from_spec("nonexistent", &BTreeMap::new()).is_none());
    }

    #[test]
    fn factory_respects_params() {
        let mut p = BTreeMap::new();
        p.insert("fast".to_string(), 5.0);
        p.insert("slow".to_string(), 20.0);
        let s = strategy_from_spec("sma_cross", &p).unwrap();
        assert_eq!(s.name(), "sma_cross(5,20)");
    }

    #[test]
    fn insufficient_history_is_flat() {
        let candles = series(&[100.0, 101.0, 102.0]);
        assert_eq!(MaCross::sma(10, 30).evaluate(&candles).action, Action::Flat);
    }
}
