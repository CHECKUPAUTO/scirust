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
            atr: atr(highs, lows, closes, atr_period),
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
///
/// True range needs the *previous close*, so `closes` must be provided (parallel
/// to `highs`/`lows`). Delegates to [`true_range`] for the per-bar TR.
pub fn atr(highs: &[f32], lows: &[f32], closes: &[f32], period: usize) -> Vec<f32> {
    let n = highs.len();
    let mut out = vec![f32::NAN; n];
    if n <= period || lows.len() != n || closes.len() != n
    {
        return out;
    }
    let trs = true_range(highs, lows, closes);
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

// ---------------------------------------------------------------------------
// Extended professional indicator suite.
//
// Everything below is a pure function over OHLCV slices, forward-reduced and
// `NaN`-padded until its look-back is satisfied — the same discipline as the
// core indicators above. These are the tools a pro desk or an exchange charting
// package exposes beyond the RSI/MACD/ATR/Bollinger basics.
// ---------------------------------------------------------------------------

/// Highest value over a trailing window of `period` (NaN until the window fills).
pub fn rolling_max(values: &[f32], period: usize) -> Vec<f32> {
    let n = values.len();
    let mut out = vec![f32::NAN; n];
    if period == 0 || n < period
    {
        return out;
    }
    for i in (period - 1)..n
    {
        let mut m = values[i + 1 - period];
        for &v in &values[i + 2 - period..=i]
        {
            if v > m
            {
                m = v;
            }
        }
        out[i] = m;
    }
    out
}

/// Lowest value over a trailing window of `period` (NaN until the window fills).
pub fn rolling_min(values: &[f32], period: usize) -> Vec<f32> {
    let n = values.len();
    let mut out = vec![f32::NAN; n];
    if period == 0 || n < period
    {
        return out;
    }
    for i in (period - 1)..n
    {
        let mut m = values[i + 1 - period];
        for &v in &values[i + 2 - period..=i]
        {
            if v < m
            {
                m = v;
            }
        }
        out[i] = m;
    }
    out
}

/// Typical price `(H+L+C)/3` — the input to CCI, MFI, VWAP.
pub fn typical_price(highs: &[f32], lows: &[f32], closes: &[f32]) -> Vec<f32> {
    let n = highs.len();
    (0..n)
        .map(|i| (highs[i] + lows[i] + closes[i]) / 3.0)
        .collect()
}

/// Rate of Change (percent): `100·(close[i] − close[i−n]) / close[i−n]`.
pub fn roc(closes: &[f32], period: usize) -> Vec<f32> {
    let n = closes.len();
    let mut out = vec![f32::NAN; n];
    for i in period..n
    {
        let base = closes[i - period];
        if base.abs() > 1e-12
        {
            out[i] = 100.0 * (closes[i] - base) / base;
        }
    }
    out
}

/// Absolute momentum: `close[i] − close[i−n]`.
pub fn momentum(closes: &[f32], period: usize) -> Vec<f32> {
    let n = closes.len();
    let mut out = vec![f32::NAN; n];
    for i in period..n
    {
        out[i] = closes[i] - closes[i - period];
    }
    out
}

/// Rolling Z-score of the close vs a `period`-window mean/stdev.
pub fn zscore(closes: &[f32], period: usize) -> Vec<f32> {
    let n = closes.len();
    let mut out = vec![f32::NAN; n];
    if period < 2 || n < period
    {
        return out;
    }
    for i in (period - 1)..n
    {
        let w = &closes[i + 1 - period..=i];
        let mean: f32 = w.iter().sum::<f32>() / period as f32;
        let var: f32 = w.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (period as f32 - 1.0);
        let sd = var.sqrt();
        if sd > 1e-12
        {
            out[i] = (closes[i] - mean) / sd;
        }
        else
        {
            out[i] = 0.0;
        }
    }
    out
}

/// Fast Stochastic Oscillator. Returns `(%K, %D)` where
/// `%K = 100·(close − LL) / (HH − LL)` over `k_period`, and `%D = SMA(%K, d_period)`.
pub fn stochastic(
    highs: &[f32],
    lows: &[f32],
    closes: &[f32],
    k_period: usize,
    d_period: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = closes.len();
    let hh = rolling_max(highs, k_period);
    let ll = rolling_min(lows, k_period);
    let mut k = vec![f32::NAN; n];
    for i in 0..n
    {
        if !hh[i].is_nan() && !ll[i].is_nan()
        {
            let range = hh[i] - ll[i];
            k[i] = if range > 1e-12
            {
                100.0 * (closes[i] - ll[i]) / range
            }
            else
            {
                50.0
            };
        }
    }
    let d = sma_nan_aware(&k, d_period);
    (k, d)
}

/// SMA that tolerates leading `NaN`s: it starts averaging from the first window
/// of `period` finite values. Used to smooth oscillators (%K → %D).
fn sma_nan_aware(values: &[f32], period: usize) -> Vec<f32> {
    let n = values.len();
    let mut out = vec![f32::NAN; n];
    if period == 0
    {
        return out;
    }
    for i in 0..n
    {
        if i + 1 < period
        {
            continue;
        }
        let w = &values[i + 1 - period..=i];
        if w.iter().any(|x| x.is_nan())
        {
            continue;
        }
        out[i] = w.iter().sum::<f32>() / period as f32;
    }
    out
}

/// Williams %R over `period`: `-100·(HH − close) / (HH − LL)`, ranging `[-100, 0]`.
pub fn williams_r(highs: &[f32], lows: &[f32], closes: &[f32], period: usize) -> Vec<f32> {
    let n = closes.len();
    let hh = rolling_max(highs, period);
    let ll = rolling_min(lows, period);
    let mut out = vec![f32::NAN; n];
    for i in 0..n
    {
        if !hh[i].is_nan() && !ll[i].is_nan()
        {
            let range = hh[i] - ll[i];
            out[i] = if range > 1e-12
            {
                -100.0 * (hh[i] - closes[i]) / range
            }
            else
            {
                -50.0
            };
        }
    }
    out
}

/// Commodity Channel Index over `period`:
/// `(TP − SMA(TP)) / (0.015 · mean_abs_dev(TP))`.
pub fn cci(highs: &[f32], lows: &[f32], closes: &[f32], period: usize) -> Vec<f32> {
    let n = closes.len();
    let tp = typical_price(highs, lows, closes);
    let mut out = vec![f32::NAN; n];
    if period == 0 || n < period
    {
        return out;
    }
    for i in (period - 1)..n
    {
        let w = &tp[i + 1 - period..=i];
        let sma: f32 = w.iter().sum::<f32>() / period as f32;
        let mad: f32 = w.iter().map(|x| (x - sma).abs()).sum::<f32>() / period as f32;
        out[i] = if mad > 1e-12
        {
            (tp[i] - sma) / (0.015 * mad)
        }
        else
        {
            0.0
        };
    }
    out
}

/// On-Balance Volume — cumulative volume signed by the close-to-close direction.
pub fn obv(closes: &[f32], volumes: &[f32]) -> Vec<f32> {
    let n = closes.len();
    let mut out = vec![0.0f32; n];
    if n == 0
    {
        return out;
    }
    for i in 1..n
    {
        out[i] = if closes[i] > closes[i - 1]
        {
            out[i - 1] + volumes[i]
        }
        else if closes[i] < closes[i - 1]
        {
            out[i - 1] - volumes[i]
        }
        else
        {
            out[i - 1]
        };
    }
    out
}

/// Money Flow Index over `period` — a volume-weighted RSI on the typical price.
// `i` indexes the output *and* bounds the trailing money-flow window, so the
// range loop is not needless here.
#[allow(clippy::needless_range_loop)]
pub fn mfi(highs: &[f32], lows: &[f32], closes: &[f32], volumes: &[f32], period: usize) -> Vec<f32> {
    let n = closes.len();
    let tp = typical_price(highs, lows, closes);
    let mut out = vec![f32::NAN; n];
    if n <= period
    {
        return out;
    }
    for i in period..n
    {
        let mut pos = 0.0f32;
        let mut neg = 0.0f32;
        for j in (i + 1 - period)..=i
        {
            if j == 0
            {
                continue;
            }
            let rmf = tp[j] * volumes[j];
            if tp[j] > tp[j - 1]
            {
                pos += rmf;
            }
            else if tp[j] < tp[j - 1]
            {
                neg += rmf;
            }
        }
        out[i] = if neg < 1e-12
        {
            100.0
        }
        else
        {
            let mr = pos / neg;
            100.0 - 100.0 / (1.0 + mr)
        };
    }
    out
}

/// Rolling VWAP over `period`: `Σ(TP·vol) / Σ(vol)` in the window.
pub fn vwap(highs: &[f32], lows: &[f32], closes: &[f32], volumes: &[f32], period: usize) -> Vec<f32> {
    let n = closes.len();
    let tp = typical_price(highs, lows, closes);
    let mut out = vec![f32::NAN; n];
    if period == 0 || n < period
    {
        return out;
    }
    for i in (period - 1)..n
    {
        let mut pv = 0.0f32;
        let mut vol = 0.0f32;
        for j in (i + 1 - period)..=i
        {
            pv += tp[j] * volumes[j];
            vol += volumes[j];
        }
        out[i] = if vol > 1e-12 { pv / vol } else { tp[i] };
    }
    out
}

/// Chaikin Money Flow over `period`.
pub fn chaikin_money_flow(
    highs: &[f32],
    lows: &[f32],
    closes: &[f32],
    volumes: &[f32],
    period: usize,
) -> Vec<f32> {
    let n = closes.len();
    let mut mfv = vec![0.0f32; n];
    for i in 0..n
    {
        let range = highs[i] - lows[i];
        let mult = if range > 1e-12
        {
            ((closes[i] - lows[i]) - (highs[i] - closes[i])) / range
        }
        else
        {
            0.0
        };
        mfv[i] = mult * volumes[i];
    }
    let mut out = vec![f32::NAN; n];
    if period == 0 || n < period
    {
        return out;
    }
    for i in (period - 1)..n
    {
        let mfv_sum: f32 = mfv[i + 1 - period..=i].iter().sum();
        let vol_sum: f32 = volumes[i + 1 - period..=i].iter().sum();
        out[i] = if vol_sum > 1e-12 { mfv_sum / vol_sum } else { 0.0 };
    }
    out
}

/// Directional Movement Index bundle: `+DI`, `−DI`, and Wilder-smoothed `ADX`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DmiSet {
    pub plus_di: Vec<f32>,
    pub minus_di: Vec<f32>,
    pub adx: Vec<f32>,
}

/// Average Directional Index (Wilder). This is the classic implementation:
/// directional movement and true range are Wilder-smoothed (not simple SMAs),
/// then DX is Wilder-smoothed once more into ADX.
pub fn dmi(highs: &[f32], lows: &[f32], closes: &[f32], period: usize) -> DmiSet {
    let n = highs.len();
    let nanv = || vec![f32::NAN; n];
    if n <= period || period == 0
    {
        return DmiSet {
            plus_di: nanv(),
            minus_di: nanv(),
            adx: nanv(),
        };
    }
    // Per-bar +DM, -DM, TR (index 0 undefined -> 0).
    let mut plus_dm = vec![0.0f32; n];
    let mut minus_dm = vec![0.0f32; n];
    let mut tr = vec![0.0f32; n];
    for i in 1..n
    {
        let up = highs[i] - highs[i - 1];
        let down = lows[i - 1] - lows[i];
        plus_dm[i] = if up > down && up > 0.0 { up } else { 0.0 };
        minus_dm[i] = if down > up && down > 0.0 { down } else { 0.0 };
        tr[i] = (highs[i] - lows[i])
            .max((highs[i] - closes[i - 1]).abs())
            .max((lows[i] - closes[i - 1]).abs());
    }
    // Wilder smoothing seeded with the sum over the first `period` bars
    // (indices 1..=period).
    let wilder_seed = |v: &[f32]| -> Vec<f32> {
        let mut out = vec![f32::NAN; n];
        let seed: f32 = v[1..=period].iter().sum();
        out[period] = seed;
        for i in (period + 1)..n
        {
            out[i] = out[i - 1] - out[i - 1] / period as f32 + v[i];
        }
        out
    };
    let s_plus = wilder_seed(&plus_dm);
    let s_minus = wilder_seed(&minus_dm);
    let s_tr = wilder_seed(&tr);

    let mut plus_di = vec![f32::NAN; n];
    let mut minus_di = vec![f32::NAN; n];
    let mut dx = vec![f32::NAN; n];
    for i in period..n
    {
        if s_tr[i].is_nan() || s_tr[i].abs() < 1e-12
        {
            continue;
        }
        let pdi = 100.0 * s_plus[i] / s_tr[i];
        let mdi = 100.0 * s_minus[i] / s_tr[i];
        plus_di[i] = pdi;
        minus_di[i] = mdi;
        let sum = pdi + mdi;
        dx[i] = if sum > 1e-12 { 100.0 * (pdi - mdi).abs() / sum } else { 0.0 };
    }
    // ADX = Wilder-smoothed DX. First ADX at index 2*period-1 = mean of the
    // first `period` DX values (indices period..2*period-1).
    let mut adx = vec![f32::NAN; n];
    let first_adx_idx = 2 * period - 1;
    if first_adx_idx < n
    {
        let seed: f32 = dx[period..first_adx_idx + 1]
            .iter()
            .filter(|v| !v.is_nan())
            .sum::<f32>()
            / period as f32;
        adx[first_adx_idx] = seed;
        for i in (first_adx_idx + 1)..n
        {
            let prev = adx[i - 1];
            let cur = if dx[i].is_nan() { 0.0 } else { dx[i] };
            adx[i] = (prev * (period as f32 - 1.0) + cur) / period as f32;
        }
    }
    DmiSet {
        plus_di,
        minus_di,
        adx,
    }
}

/// A price channel: mid, upper, lower series aligned with the input length.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Channel {
    pub upper: Vec<f32>,
    pub mid: Vec<f32>,
    pub lower: Vec<f32>,
}

/// Donchian Channel: highest-high / lowest-low over `period`, mid = their mean.
pub fn donchian(highs: &[f32], lows: &[f32], period: usize) -> Channel {
    let upper = rolling_max(highs, period);
    let lower = rolling_min(lows, period);
    let mid = upper
        .iter()
        .zip(lower.iter())
        .map(|(u, l)| {
            if u.is_nan() || l.is_nan()
            {
                f32::NAN
            }
            else
            {
                (u + l) / 2.0
            }
        })
        .collect();
    Channel { upper, mid, lower }
}

/// Keltner Channel: `EMA(close, period) ± mult·ATR(period)`.
pub fn keltner(highs: &[f32], lows: &[f32], closes: &[f32], period: usize, mult: f32) -> Channel {
    let mid = ema(closes, period);
    let atr_series = atr(highs, lows, closes, period);
    let n = closes.len();
    let mut upper = vec![f32::NAN; n];
    let mut lower = vec![f32::NAN; n];
    for i in 0..n
    {
        if !mid[i].is_nan() && !atr_series[i].is_nan()
        {
            upper[i] = mid[i] + mult * atr_series[i];
            lower[i] = mid[i] - mult * atr_series[i];
        }
    }
    Channel { upper, mid, lower }
}

/// Supertrend line + direction (`+1` = uptrend/bullish, `−1` = downtrend).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Supertrend {
    pub line: Vec<f32>,
    pub direction: Vec<i8>,
}

/// Supertrend (ATR-banded trend follower). `mult` is typically 2–3.
///
/// Standard carry-over band logic: the final upper band only moves down (and the
/// final lower band only moves up) while price stays on the same side, so the
/// trailing stop never loosens against the trend.
pub fn supertrend(
    highs: &[f32],
    lows: &[f32],
    closes: &[f32],
    period: usize,
    mult: f32,
) -> Supertrend {
    let n = closes.len();
    let atr_series = atr(highs, lows, closes, period);
    let mut final_upper = vec![f32::NAN; n];
    let mut final_lower = vec![f32::NAN; n];
    let mut line = vec![f32::NAN; n];
    let mut direction = vec![0i8; n];
    let mut started = false;
    for i in 0..n
    {
        if atr_series[i].is_nan()
        {
            continue;
        }
        let hl2 = (highs[i] + lows[i]) / 2.0;
        let basic_upper = hl2 + mult * atr_series[i];
        let basic_lower = hl2 - mult * atr_series[i];
        if !started
        {
            final_upper[i] = basic_upper;
            final_lower[i] = basic_lower;
            direction[i] = 1;
            line[i] = final_lower[i];
            started = true;
            continue;
        }
        let prev_fu = final_upper[i - 1];
        let prev_fl = final_lower[i - 1];
        final_upper[i] = if basic_upper < prev_fu || closes[i - 1] > prev_fu
        {
            basic_upper
        }
        else
        {
            prev_fu
        };
        final_lower[i] = if basic_lower > prev_fl || closes[i - 1] < prev_fl
        {
            basic_lower
        }
        else
        {
            prev_fl
        };
        // Direction flip based on close vs the opposing final band.
        let prev_dir = direction[i - 1];
        direction[i] = if prev_dir == 1
        {
            if closes[i] < final_lower[i] { -1 } else { 1 }
        }
        else if closes[i] > final_upper[i]
        {
            1
        }
        else
        {
            -1
        };
        line[i] = if direction[i] == 1 { final_lower[i] } else { final_upper[i] };
    }
    Supertrend { line, direction }
}

#[cfg(test)]
mod tests_ext {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    #[test]
    fn rolling_extrema() {
        let v = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let mx = rolling_max(&v, 3);
        let mn = rolling_min(&v, 3);
        assert!(mx[0].is_nan() && mx[1].is_nan());
        assert!(approx(mx[2], 4.0));
        assert!(approx(mx[5], 9.0));
        assert!(approx(mn[2], 1.0));
        assert!(approx(mn[5], 1.0));
    }

    #[test]
    fn roc_and_momentum() {
        let c = [10.0, 11.0, 12.0, 13.0];
        let r = roc(&c, 1);
        assert!(approx(r[1], 10.0)); // (11-10)/10*100
        let m = momentum(&c, 2);
        assert!(approx(m[2], 2.0)); // 12-10
    }

    #[test]
    fn stochastic_bounds() {
        let highs: Vec<f32> = (0..30).map(|i| 100.0 + (i as f32 * 0.5).sin() * 5.0 + 5.0).collect();
        let lows: Vec<f32> = highs.iter().map(|h| h - 10.0).collect();
        let closes: Vec<f32> = highs.iter().map(|h| h - 5.0).collect();
        let (k, d) = stochastic(&highs, &lows, &closes, 14, 3);
        for i in 16..30
        {
            assert!((0.0..=100.0).contains(&k[i]), "%K out of range: {}", k[i]);
            assert!((0.0..=100.0).contains(&d[i]));
        }
    }

    #[test]
    fn williams_r_in_range() {
        let highs: Vec<f32> = (0..30).map(|i| 100.0 + i as f32).collect();
        let lows: Vec<f32> = highs.iter().map(|h| h - 3.0).collect();
        let closes: Vec<f32> = highs.iter().map(|h| h - 1.0).collect();
        let wr = williams_r(&highs, &lows, &closes, 14);
        for v in &wr[14..]
        {
            assert!((-100.0..=0.0).contains(v), "%R out of range: {v}");
        }
    }

    #[test]
    fn obv_tracks_direction() {
        let c = [10.0, 11.0, 10.5, 12.0];
        let v = [100.0, 200.0, 150.0, 300.0];
        let o = obv(&c, &v);
        assert_eq!(o[0], 0.0);
        assert_eq!(o[1], 200.0); // up
        assert_eq!(o[2], 50.0); // down: 200-150
        assert_eq!(o[3], 350.0); // up: 50+300
    }

    #[test]
    fn mfi_in_range() {
        let highs: Vec<f32> = (0..30).map(|i| 100.0 + (i as f32 * 0.3).sin() * 4.0).collect();
        let lows: Vec<f32> = highs.iter().map(|h| h - 5.0).collect();
        let closes: Vec<f32> = highs.iter().map(|h| h - 2.0).collect();
        let vols: Vec<f32> = (0..30).map(|i| 1000.0 + i as f32 * 10.0).collect();
        let m = mfi(&highs, &lows, &closes, &vols, 14);
        for v in &m[14..]
        {
            assert!((0.0..=100.0).contains(v), "MFI out of range: {v}");
        }
    }

    #[test]
    fn vwap_between_low_and_high() {
        let highs = [10.0, 11.0, 12.0, 13.0];
        let lows = [8.0, 9.0, 10.0, 11.0];
        let closes = [9.0, 10.0, 11.0, 12.0];
        let vols = [100.0, 100.0, 100.0, 100.0];
        let w = vwap(&highs, &lows, &closes, &vols, 2);
        for i in 1..4
        {
            assert!(w[i] >= lows[i] - 2.0 && w[i] <= highs[i] + 2.0);
        }
    }

    #[test]
    fn cci_zero_on_flat_trend_is_finite() {
        let closes: Vec<f32> = (0..30).map(|i| 100.0 + i as f32).collect();
        let highs: Vec<f32> = closes.iter().map(|c| c + 1.0).collect();
        let lows: Vec<f32> = closes.iter().map(|c| c - 1.0).collect();
        let c = cci(&highs, &lows, &closes, 20);
        assert!(c[25].is_finite());
    }

    #[test]
    fn dmi_adx_in_range_and_trend_detected() {
        // Strong uptrend -> +DI should exceed -DI and ADX should rise.
        let highs: Vec<f32> = (0..60).map(|i| 100.0 + i as f32 * 1.0).collect();
        let lows: Vec<f32> = highs.iter().map(|h| h - 2.0).collect();
        let closes: Vec<f32> = highs.iter().map(|h| h - 0.5).collect();
        let set = dmi(&highs, &lows, &closes, 14);
        let last = 59;
        assert!(set.plus_di[last] > set.minus_di[last], "uptrend: +DI should lead");
        assert!(set.adx[last].is_finite());
        assert!((0.0..=100.0).contains(&set.adx[last]), "ADX out of range");
    }

    #[test]
    fn donchian_brackets_price() {
        let highs: Vec<f32> = (0..30).map(|i| 100.0 + (i as f32 * 0.4).sin() * 6.0).collect();
        let lows: Vec<f32> = highs.iter().map(|h| h - 4.0).collect();
        let ch = donchian(&highs, &lows, 20);
        for i in 19..30
        {
            assert!(ch.upper[i] >= ch.mid[i]);
            assert!(ch.mid[i] >= ch.lower[i]);
        }
    }

    #[test]
    fn keltner_bands_surround_mid() {
        let highs: Vec<f32> = (0..40).map(|i| 100.0 + (i as f32 * 0.2).sin() * 3.0 + 2.0).collect();
        let lows: Vec<f32> = highs.iter().map(|h| h - 4.0).collect();
        let closes: Vec<f32> = highs.iter().map(|h| h - 2.0).collect();
        let ch = keltner(&highs, &lows, &closes, 20, 2.0);
        for i in 25..40
        {
            if !ch.mid[i].is_nan()
            {
                assert!(ch.upper[i] > ch.mid[i]);
                assert!(ch.lower[i] < ch.mid[i]);
            }
        }
    }

    #[test]
    fn supertrend_flips_direction() {
        // Up then down: direction must contain both +1 and -1.
        let mut closes: Vec<f32> = (0..30).map(|i| 100.0 + i as f32).collect();
        closes.extend((0..30).map(|i| 130.0 - i as f32 * 1.5));
        let highs: Vec<f32> = closes.iter().map(|c| c + 1.0).collect();
        let lows: Vec<f32> = closes.iter().map(|c| c - 1.0).collect();
        let st = supertrend(&highs, &lows, &closes, 10, 3.0);
        let has_up = st.direction.contains(&1);
        let has_down = st.direction.contains(&-1);
        assert!(has_up && has_down, "supertrend should flip on a reversal");
    }

    #[test]
    fn zscore_centered() {
        let closes: Vec<f32> = (0..30).map(|i| 100.0 + (i as f32 * 0.5).sin()).collect();
        let z = zscore(&closes, 20);
        for v in &z[19..]
        {
            assert!(v.is_finite());
            assert!(v.abs() < 5.0, "z-score unexpectedly large: {v}");
        }
    }
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
        let closes: Vec<f32> = (0..20).map(|i| 99.5 + (i as f32).sin()).collect();
        let a = atr(&highs, &lows, &closes, 14);
        for v in &a[13..]
        {
            assert!(!v.is_nan());
            assert!(*v >= 0.0);
        }
    }

    #[test]
    fn atr_measures_true_range_not_price_level() {
        // Bars around 50000 with ~60-wide ranges. ATR must be on the order of the
        // range (tens), NOT the price level — the pre-fix stub used close=0.0, so
        // TR collapsed to ~high (~50000). Verified separation via a loose bound.
        let highs: Vec<f32> = (0..40)
            .map(|i| 50000.0 + (i as f32 * 0.3).sin() * 5.0 + 30.0)
            .collect();
        let lows: Vec<f32> = highs.iter().map(|h| h - 60.0).collect();
        let closes: Vec<f32> = highs.iter().map(|h| h - 30.0).collect();
        let a = atr(&highs, &lows, &closes, 14);
        let last = *a.last().unwrap();
        assert!(last.is_finite() && last > 0.0);
        assert!(
            last < 500.0,
            "ATR should track the ~60-wide range, got {last}"
        );
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
