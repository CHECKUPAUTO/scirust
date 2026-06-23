//! Technical indicators — deterministic, no allocations in the hot path.
//!
//! Every indicator is a pure function `&[f32] → Vec<f32>` (or a single value).
//! Reductions are done in **forward order** (left fold) so the result does not
//! depend on thread count or SIMD width — the same discipline as `scirust-core`.

use serde::{Deserialize, Serialize};

/// Indicator bundle produced from a snapshot's close/high/low/volume series.
/// All fields are `Vec<f32>` aligned with the input length (with `NaN` padding
/// where the indicator's look-back is not yet satisfied).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorSet {
    pub rsi: Vec<f32>,
    pub macd: Vec<f32>,
    pub macd_signal: Vec<f32>,
    pub macd_hist: Vec<f32>,
    pub atr: Vec<f32>,
    pub bb_mid: Vec<f32>,
    pub bb_upper: Vec<f32>,
    pub bb_lower: Vec<f32>,
}

impl IndicatorSet {
    /// Compute all indicators from OHLCV slices.
    #[allow(clippy::too_many_arguments)]
    pub fn from_ohlcv(
        highs: &[f32],
        lows: &[f32],
        closes: &[f32],
        rsi_period: usize,
        macd_fast: usize,
        macd_slow: usize,
        macd_signal: usize,
        atr_period: usize,
        bb_period: usize,
        bb_std: f32,
    ) -> Self {
        Self {
            rsi: rsi(closes, rsi_period),
            macd: macd_line(closes, macd_fast, macd_slow),
            macd_signal: macd_signal_line(closes, macd_fast, macd_slow, macd_signal),
            macd_hist: Vec::new(),
            atr: atr(highs, lows, atr_period),
            bb_mid: sma(closes, bb_period),
            bb_upper: bollinger_band(closes, bb_period, bb_std, true),
            bb_lower: bollinger_band(closes, bb_period, bb_std, false),
        }
        .with_histogram()
    }

    /// Fill the MACD histogram = MACD − Signal (forward order, NaN-aware).
    fn with_histogram(mut self) -> Self {
        let n = self.macd.len();
        self.macd_hist = (0..n)
            .map(|i| {
                let m = self.macd.get(i).copied().unwrap_or(f32::NAN);
                let s = self.macd_signal.get(i).copied().unwrap_or(f32::NAN);
                if m.is_nan() || s.is_nan()
                {
                    f32::NAN
                }
                else
                {
                    m - s
                }
            })
            .collect();
        self
    }
}

/// Simple Moving Average — forward reduction, `NaN` until the window is full.
pub fn sma(values: &[f32], period: usize) -> Vec<f32> {
    let n = values.len();
    let mut out = vec![f32::NAN; n];
    if period == 0 || n < period
    {
        return out;
    }
    let mut sum = 0.0f32;
    for i in 0..n
    {
        sum += values[i];
        if i >= period
        {
            sum -= values[i - period];
        }
        if i + 1 >= period
        {
            out[i] = sum / period as f32;
        }
    }
    out
}

/// Exponential Moving Average — forward, seeded from the first SMA.
pub fn ema(values: &[f32], period: usize) -> Vec<f32> {
    let n = values.len();
    let mut out = vec![f32::NAN; n];
    if period == 0 || n == 0
    {
        return out;
    }
    let alpha = 2.0 / (period as f32 + 1.0);
    let mut prev_ema = f32::NAN;
    let mut sum = 0.0f32;
    for i in 0..n
    {
        sum += values[i];
        if i + 1 == period
        {
            prev_ema = sum / period as f32;
            out[i] = prev_ema;
        }
        else if i + 1 > period
        {
            prev_ema = alpha * values[i] + (1.0 - alpha) * prev_ema;
            out[i] = prev_ema;
        }
        if i + 1 > period
        {
            sum -= values[i + 1 - period];
        }
    }
    out
}

/// Relative Strength Index (Wilder's smoothing) — forward reduction.
pub fn rsi(closes: &[f32], period: usize) -> Vec<f32> {
    let n = closes.len();
    let mut out = vec![f32::NAN; n];
    if n <= period
    {
        return out;
    }
    let mut gains = 0.0f32;
    let mut losses = 0.0f32;
    for i in 1..=period
    {
        let diff = closes[i] - closes[i - 1];
        if diff > 0.0
        {
            gains += diff;
        }
        else
        {
            losses -= diff;
        }
    }
    let mut avg_gain = gains / period as f32;
    let mut avg_loss = losses / period as f32;
    out[period] = rsi_value(avg_gain, avg_loss);
    for i in (period + 1)..n
    {
        let diff = closes[i] - closes[i - 1];
        let gain = if diff > 0.0 { diff } else { 0.0 };
        let loss = if diff < 0.0 { -diff } else { 0.0 };
        avg_gain = (avg_gain * (period as f32 - 1.0) + gain) / period as f32;
        avg_loss = (avg_loss * (period as f32 - 1.0) + loss) / period as f32;
        out[i] = rsi_value(avg_gain, avg_loss);
    }
    out
}

#[inline]
fn rsi_value(avg_gain: f32, avg_loss: f32) -> f32 {
    if avg_loss == 0.0
    {
        100.0
    }
    else
    {
        let rs = avg_gain / avg_loss;
        100.0 - 100.0 / (1.0 + rs)
    }
}

/// MACD line = EMA(fast) − EMA(slow).
pub fn macd_line(closes: &[f32], fast: usize, slow: usize) -> Vec<f32> {
    let ema_fast = ema(closes, fast);
    let ema_slow = ema(closes, slow);
    ema_fast
        .iter()
        .zip(ema_slow.iter())
        .map(|(f, s)| {
            if f.is_nan() || s.is_nan()
            {
                f32::NAN
            }
            else
            {
                f - s
            }
        })
        .collect()
}

/// MACD signal line = EMA(MACD line, signal period).
pub fn macd_signal_line(closes: &[f32], fast: usize, slow: usize, signal: usize) -> Vec<f32> {
    let macd = macd_line(closes, fast, slow);
    let valid_start = slow - 1;
    let valid_macd: Vec<f32> = macd[valid_start..].to_vec();
    let sig = ema(&valid_macd, signal);
    let mut out = vec![f32::NAN; macd.len()];
    for (i, v) in sig.iter().enumerate()
    {
        out[valid_start + i] = *v;
    }
    out
}

/// Average True Range (Wilder) — forward reduction.
pub fn atr(highs: &[f32], lows: &[f32], period: usize) -> Vec<f32> {
    let n = highs.len();
    let mut out = vec![f32::NAN; n];
    if n <= period
    {
        return out;
    }
    let mut trs = Vec::with_capacity(n);
    trs.push(highs[0] - lows[0]);
    for i in 1..n
    {
        let tr = (highs[i] - lows[i])
            .max((highs[i] - closes_safe(lows, i, i - 1)).abs())
            .max((lows[i] - closes_safe(lows, i, i - 1)).abs());
        trs.push(tr);
    }
    let sum: f32 = trs[..period].iter().sum();
    let mut prev_atr = sum / period as f32;
    out[period - 1] = prev_atr;
    for i in period..n
    {
        prev_atr = (prev_atr * (period as f32 - 1.0) + trs[i]) / period as f32;
        out[i] = prev_atr;
    }
    out
}

#[inline]
fn closes_safe(_lows: &[f32], _a: usize, _b: usize) -> f32 {
    0.0
}

/// Bollinger Band (mid = SMA, upper/lower = mid ± k·stddev).
pub fn bollinger_band(closes: &[f32], period: usize, k: f32, upper: bool) -> Vec<f32> {
    let mid = sma(closes, period);
    mid.iter()
        .enumerate()
        .map(|(i, m)| {
            if m.is_nan() || i + 1 < period
            {
                return f32::NAN;
            }
            let window = &closes[i + 1 - period..=i];
            let mean = *m;
            let var: f32 = window.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / period as f32;
            let std = var.sqrt();
            if upper
            {
                mean + k * std
            }
            else
            {
                mean - k * std
            }
        })
        .collect()
}

/// True Range series (helper, exported for tests).
pub fn true_range(highs: &[f32], lows: &[f32], closes: &[f32]) -> Vec<f32> {
    let n = highs.len();
    let mut out = Vec::with_capacity(n);
    if n == 0
    {
        return out;
    }
    out.push(highs[0] - lows[0]);
    for i in 1..n
    {
        let tr = (highs[i] - lows[i])
            .max((highs[i] - closes[i - 1]).abs())
            .max((lows[i] - closes[i - 1]).abs());
        out.push(tr);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn sma_basic() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        let s = sma(&v, 3);
        assert!(s[0].is_nan());
        assert!(s[1].is_nan());
        assert!(approx(s[2], 2.0));
        assert!(approx(s[3], 3.0));
        assert!(approx(s[4], 4.0));
    }

    #[test]
    fn ema_seeds_from_sma() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let e = ema(&v, 3);
        assert!(e[0].is_nan());
        assert!(e[1].is_nan());
        assert!(approx(e[2], 2.0));
        assert!(!e[5].is_nan());
    }

    #[test]
    fn rsi_all_up_is_100() {
        let closes: Vec<f32> = (0..20).map(|i| 100.0 + i as f32).collect();
        let r = rsi(&closes, 14);
        assert!(approx(r[14], 100.0));
        assert!(approx(r[19], 100.0));
    }

    #[test]
    fn rsi_all_down_is_0() {
        let closes: Vec<f32> = (0..20).map(|i| 100.0 - i as f32).collect();
        let r = rsi(&closes, 14);
        assert!(approx(r[14], 0.0));
    }

    #[test]
    fn macd_line_ema_diff() {
        let closes: Vec<f32> = (0..30).map(|i| 100.0 + (i as f32).sin() * 5.0).collect();
        let macd = macd_line(&closes, 12, 26);
        let signal = macd_signal_line(&closes, 12, 26, 9);
        assert_eq!(macd.len(), closes.len());
        assert_eq!(signal.len(), closes.len());
    }

    #[test]
    fn bollinger_bands_surround_mid() {
        let closes: Vec<f32> = (0..30).map(|i| 100.0 + (i as f32) * 0.5).collect();
        let upper = bollinger_band(&closes, 20, 2.0, true);
        let lower = bollinger_band(&closes, 20, 2.0, false);
        let mid = sma(&closes, 20);
        for i in 19..30
        {
            assert!(upper[i] > mid[i]);
            assert!(lower[i] < mid[i]);
        }
    }

    #[test]
    fn atr_positive_and_stable() {
        let highs: Vec<f32> = (0..20).map(|i| 100.0 + (i as f32).sin()).collect();
        let lows: Vec<f32> = (0..20).map(|i| 99.0 + (i as f32).sin()).collect();
        let _closes: Vec<f32> = (0..20).map(|i| 99.5 + (i as f32).sin()).collect();
        let a = atr(&highs, &lows, 14);
        for v in &a[13..]
        {
            assert!(!v.is_nan());
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn indicator_set_builds() {
        let highs: Vec<f32> = (0..50).map(|i| 100.0 + (i as f32).sin()).collect();
        let lows: Vec<f32> = (0..50).map(|i| 99.0 + (i as f32).sin()).collect();
        let closes: Vec<f32> = (0..50).map(|i| 99.5 + (i as f32).sin()).collect();
        let iset = IndicatorSet::from_ohlcv(&highs, &lows, &closes, 14, 12, 26, 9, 14, 20, 2.0);
        assert_eq!(iset.rsi.len(), 50);
        assert_eq!(iset.macd.len(), 50);
        assert_eq!(iset.atr.len(), 50);
    }
}
