//! Market-regime detection — read the *state of the market* before choosing how
//! to trade in it.
//!
//! A strategy that prints money in a calm uptrend can bleed out in a choppy,
//! high-volatility tape; a mean-reversion book that thrives in a range gets run
//! over by a trend. The single most useful thing an agent can know before
//! sizing a position or picking a strategy family is *which regime we are in*.
//! This module classifies that deterministically from an OHLCV history.
//!
//! Three orthogonal reads, then a combined taxonomy:
//!
//! * **Volatility state** — rolling realized volatility, ranked against its own
//!   history into calm / elevated / crisis buckets. Volatility clusters
//!   (Mandelbrot 1963; the GARCH tradition), so the recent level is genuinely
//!   informative about the near future.
//! * **Trend state** — the OLS slope of log-price over a window, normalized by
//!   volatility into a t-statistic-like signal-to-noise ratio, so "trend" means
//!   a move large relative to the noise, not just any drift. → bull / bear /
//!   range.
//! * **Persistence** — the **Hurst exponent** via rescaled-range (R/S) analysis
//!   (Hurst 1951; Mandelbrot & Wallis 1969). `H > 0.5` ⇒ trending/persistent
//!   (momentum has an edge), `H < 0.5` ⇒ mean-reverting/anti-persistent
//!   (fade extremes), `H ≈ 0.5` ⇒ random walk (no persistence edge).
//!
//! The per-bar labels feed an empirical **Markov transition matrix** over the
//! six regimes, from which we derive expected regime durations and the
//! long-run (stationary) occupancy — so the agent can reason about how sticky
//! the current regime is, not just what it is right now.
//!
//! Everything is a pure, forward-order reduction: same candles ⇒ same report.

// Matrix maths (transition matrix, power iteration) and windowed statistics read
// most clearly with explicit `i`/`j` index loops here.
#![allow(clippy::needless_range_loop)]

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::market::Candle;
use crate::metrics::{returns_from_equity, stddev};

/// The six market regimes, in a fixed order that indexes the transition matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Up-trend, low volatility — the easy tape: trend-follow, leverage tolerable.
    BullCalm,
    /// Up-trend, high volatility — real trend but whippy: trim size, widen stops.
    BullVolatile,
    /// No decisive trend — mean-reversion / market-making, fade the extremes.
    Range,
    /// Down-trend, low volatility — orderly decline: risk-off or trend-short.
    BearCalm,
    /// Down-trend, high volatility — disorderly selling: defensive, small size.
    BearVolatile,
    /// Extreme volatility regardless of direction — de-risk, cut leverage.
    Crisis,
}

impl MarketRegime {
    /// The regimes in transition-matrix row/column order.
    pub const ALL: [MarketRegime; 6] = [
        MarketRegime::BullCalm,
        MarketRegime::BullVolatile,
        MarketRegime::Range,
        MarketRegime::BearCalm,
        MarketRegime::BearVolatile,
        MarketRegime::Crisis,
    ];

    /// Index into [`MarketRegime::ALL`].
    pub fn index(self) -> usize {
        MarketRegime::ALL
            .iter()
            .position(|&r| r == self)
            .unwrap_or(2)
    }

    /// Short human label.
    pub fn label(self) -> &'static str {
        match self
        {
            MarketRegime::BullCalm => "bull / calm",
            MarketRegime::BullVolatile => "bull / volatile",
            MarketRegime::Range => "range / choppy",
            MarketRegime::BearCalm => "bear / calm",
            MarketRegime::BearVolatile => "bear / volatile",
            MarketRegime::Crisis => "crisis",
        }
    }

    /// The recommended trading posture for this regime — the actionable
    /// takeaway an agent turns into sizing and strategy-family choices.
    pub fn posture(self) -> &'static str {
        match self
        {
            MarketRegime::BullCalm =>
            {
                "trend-follow long; momentum/breakout strategies; normal-to-higher leverage"
            },
            MarketRegime::BullVolatile =>
            {
                "stay long the trend but cut size and widen stops; volatility is elevated"
            },
            MarketRegime::Range =>
            {
                "mean-reversion / market-making; fade extremes; avoid breakout entries"
            },
            MarketRegime::BearCalm => "risk-off or trend-follow short; reduce gross exposure",
            MarketRegime::BearVolatile =>
            {
                "defensive: small size, short or hedged; disorderly downside"
            },
            MarketRegime::Crisis =>
            {
                "de-risk now: cut leverage, raise cash/hedge; do not add to positions"
            },
        }
    }
}

/// Tuning for [`detect`]. Defaults suit hourly-to-daily crypto bars.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeConfig {
    /// Rolling window (in returns) for realized volatility.
    pub vol_window: usize,
    /// Rolling window (in bars) for the trend regression.
    pub trend_window: usize,
    /// Volatility percentile at/above which a bar is "volatile" (0..1).
    pub elevated_pct: f32,
    /// Volatility percentile at/above which a bar is "crisis" (0..1).
    pub crisis_pct: f32,
    /// `|normalized trend|` below this is treated as no trend (range).
    pub range_t: f32,
    /// Bars per year, for annualizing the reported volatility.
    pub periods_per_year: f32,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            vol_window: 20,
            trend_window: 30,
            elevated_pct: 0.66,
            crisis_pct: 0.90,
            range_t: 1.0,
            periods_per_year: 365.0,
        }
    }
}

/// The regime report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeReport {
    /// Current (most-recent-bar) regime.
    pub current: MarketRegime,
    /// Human label for [`RegimeReport::current`].
    pub current_label: String,
    /// Recommended posture for the current regime.
    pub posture: String,
    /// Latest rolling realized volatility, **annualized**.
    pub realized_vol: f32,
    /// Percentile rank (0..1) of the latest volatility within its own history.
    pub vol_percentile: f32,
    /// Latest per-bar log-price trend slope.
    pub trend_slope: f32,
    /// Latest normalized trend strength (signal-to-noise, t-stat-like).
    pub trend_strength: f32,
    /// Hurst exponent over the full return series.
    pub hurst: f32,
    /// Plain-language reading of the Hurst exponent.
    pub hurst_interpretation: String,
    /// Number of bars that received a regime label.
    pub num_labeled: usize,
    /// Occupancy count per regime, keyed by [`MarketRegime::label`].
    pub regime_counts: BTreeMap<String, usize>,
    /// Regime labels in transition-matrix row/column order.
    pub regime_order: Vec<String>,
    /// Row-stochastic empirical transition matrix `P[i][j] = P(i → j)`.
    pub transition_matrix: Vec<Vec<f32>>,
    /// Expected persistence (in bars) of each observed regime, `1/(1−P_ii)`.
    pub expected_durations: BTreeMap<String, f32>,
    /// Long-run (stationary) occupancy fraction per regime.
    pub stationary: BTreeMap<String, f32>,
}

// ---------------------------------------------------------------------------
// Building blocks.
// ---------------------------------------------------------------------------

/// Annualized realized volatility of a return series: sample stdev × √ppy.
pub fn realized_volatility(returns: &[f32], periods_per_year: f32) -> f32 {
    stddev(returns) * periods_per_year.max(0.0).sqrt()
}

/// OLS slope of `ys` against `x = 0,1,2,…` (per-step slope). 0 for `n < 2`.
fn ols_slope(ys: &[f32]) -> f32 {
    let n = ys.len();
    if n < 2
    {
        return 0.0;
    }
    let nf = n as f32;
    let mean_x = (nf - 1.0) / 2.0;
    let mean_y = ys.iter().sum::<f32>() / nf;
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for (i, &y) in ys.iter().enumerate()
    {
        let dx = i as f32 - mean_x;
        num += dx * (y - mean_y);
        den += dx * dx;
    }
    if den.abs() < 1e-12 { 0.0 } else { num / den }
}

/// OLS slope of `ys` against arbitrary `xs` (both `f64`, for the R/S fit).
fn ols_slope_xy(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2
    {
        return 0.0;
    }
    let nf = n as f64;
    let mean_x = xs.iter().sum::<f64>() / nf;
    let mean_y = ys.iter().sum::<f64>() / nf;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..n
    {
        let dx = xs[i] - mean_x;
        num += dx * (ys[i] - mean_y);
        den += dx * dx;
    }
    if den.abs() < 1e-12 { 0.0 } else { num / den }
}

/// Hurst exponent via rescaled-range (R/S) analysis. `H > 0.5` trending,
/// `H < 0.5` mean-reverting, `H ≈ 0.5` random walk. Falls back to `0.5`
/// (random walk) when the series is too short to estimate.
///
/// Uses `f64` internally: the R/S ratios span orders of magnitude and the
/// log-log fit is sensitive to rounding in the tails.
pub fn hurst_exponent(series: &[f32]) -> f32 {
    let n = series.len();
    if n < 16
    {
        return 0.5;
    }
    let max_lag = n / 2;
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();
    let mut lag = 8usize;
    while lag <= max_lag
    {
        let chunks = n / lag;
        let mut rs_sum = 0.0f64;
        let mut rs_count = 0usize;
        for c in 0..chunks
        {
            let seg = &series[c * lag..(c + 1) * lag];
            let mean = seg.iter().map(|&x| x as f64).sum::<f64>() / lag as f64;
            let mut cum = 0.0f64;
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            let mut var = 0.0f64;
            for &x in seg
            {
                let d = x as f64 - mean;
                cum += d;
                if cum < lo
                {
                    lo = cum;
                }
                if cum > hi
                {
                    hi = cum;
                }
                var += d * d;
            }
            let r = hi - lo;
            let s = (var / lag as f64).sqrt();
            if s > 1e-12 && r > 0.0
            {
                rs_sum += r / s;
                rs_count += 1;
            }
        }
        if rs_count > 0
        {
            xs.push((lag as f64).ln());
            ys.push((rs_sum / rs_count as f64).ln());
        }
        // Geometric-ish spacing of lags — deterministic, no RNG.
        lag = (lag as f64 * 1.6) as usize + 1;
    }
    if xs.len() < 2
    {
        return 0.5;
    }
    (ols_slope_xy(&xs, &ys) as f32).clamp(0.0, 1.0)
}

fn hurst_reading(h: f32) -> &'static str {
    if h > 0.55
    {
        "trending / persistent — momentum and breakout strategies have an edge"
    }
    else if h < 0.45
    {
        "mean-reverting / anti-persistent — favor reversion and market-making"
    }
    else
    {
        "near random-walk — no reliable persistence edge; trade the regime, not the drift"
    }
}

/// Value at percentile `q` (0..1) of a sorted slice, nearest-rank.
fn percentile_sorted(sorted: &[f32], q: f32) -> f32 {
    if sorted.is_empty()
    {
        return 0.0;
    }
    let q = q.clamp(0.0, 1.0);
    let idx = ((q * sorted.len() as f32).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[idx]
}

/// Fraction of `sorted` values `<= v` — the percentile rank of `v`.
fn percentile_rank(sorted: &[f32], v: f32) -> f32 {
    if sorted.is_empty()
    {
        return 0.0;
    }
    let count = sorted.iter().filter(|&&x| x <= v).count();
    count as f32 / sorted.len() as f32
}

fn classify(vol: f32, t: f32, elevated: f32, crisis: f32, range_t: f32) -> MarketRegime {
    if vol >= crisis
    {
        return MarketRegime::Crisis;
    }
    let volatile = vol >= elevated;
    if t > range_t
    {
        if volatile
        {
            MarketRegime::BullVolatile
        }
        else
        {
            MarketRegime::BullCalm
        }
    }
    else if t < -range_t
    {
        if volatile
        {
            MarketRegime::BearVolatile
        }
        else
        {
            MarketRegime::BearCalm
        }
    }
    else
    {
        MarketRegime::Range
    }
}

// ---------------------------------------------------------------------------
// Top-level detection.
// ---------------------------------------------------------------------------

/// Detect the market regime from an OHLCV history. Returns `None` if there are
/// too few candles to label even a single bar.
pub fn detect(candles: &[Candle], cfg: &RegimeConfig) -> Option<RegimeReport> {
    let n = candles.len();
    let w_v = cfg.vol_window.max(2);
    let w_t = cfg.trend_window.max(2);
    // A bar `i` needs `w_v` returns ending at it (⇒ i ≥ w_v) and `w_t`
    // log-closes ending at it (⇒ i ≥ w_t − 1).
    let start = w_v.max(w_t - 1);
    if n <= start
    {
        return None;
    }

    let closes: Vec<f32> = candles.iter().map(|c| c.close).collect();
    let logc: Vec<f32> = closes.iter().map(|&c| c.max(1e-12).ln()).collect();
    let rets = returns_from_equity(&closes); // len n-1; rets[k] spans close k→k+1

    // Per-bar (volatility, normalized-trend) for every labelable bar.
    let mut bar_idx: Vec<usize> = Vec::new();
    let mut vols: Vec<f32> = Vec::new();
    let mut slopes: Vec<f32> = Vec::new();
    let mut tstats: Vec<f32> = Vec::new();
    for i in start..n
    {
        // The `w_v` returns ending at bar i are rets[i-w_v .. i].
        let vol = stddev(&rets[i - w_v..i]);
        let slope = ols_slope(&logc[i + 1 - w_t..=i]);
        // Signal-to-noise over the window: drift·√w / σ.
        let t = if vol > 1e-12
        {
            slope * (w_t as f32).sqrt() / vol
        }
        else
        {
            0.0
        };
        bar_idx.push(i);
        vols.push(vol);
        slopes.push(slope);
        tstats.push(t);
    }

    // Percentile thresholds from the empirical volatility distribution.
    let mut sorted = vols.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let elevated = percentile_sorted(&sorted, cfg.elevated_pct);
    let crisis = percentile_sorted(&sorted, cfg.crisis_pct);

    // Label every bar.
    let labels: Vec<MarketRegime> = (0..vols.len())
        .map(|k| classify(vols[k], tstats[k], elevated, crisis, cfg.range_t))
        .collect();

    // Occupancy counts.
    let mut regime_counts: BTreeMap<String, usize> = BTreeMap::new();
    for &r in &labels
    {
        *regime_counts.entry(r.label().to_string()).or_insert(0) += 1;
    }

    // Empirical transition matrix over the fixed 6-state order.
    let m = MarketRegime::ALL.len();
    let mut counts = vec![vec![0.0f32; m]; m];
    for w in labels.windows(2)
    {
        counts[w[0].index()][w[1].index()] += 1.0;
    }
    let mut transition = vec![vec![0.0f32; m]; m];
    for i in 0..m
    {
        let row_sum: f32 = counts[i].iter().sum();
        if row_sum > 0.0
        {
            for j in 0..m
            {
                transition[i][j] = counts[i][j] / row_sum;
            }
        }
    }

    // Expected duration 1/(1−P_ii) for each observed regime.
    let mut expected_durations: BTreeMap<String, f32> = BTreeMap::new();
    for i in 0..m
    {
        let regime = MarketRegime::ALL[i];
        if regime_counts.contains_key(regime.label())
        {
            let p = transition[i][i];
            let dur = if p < 1.0
            {
                1.0 / (1.0 - p)
            }
            else
            {
                labels.len() as f32
            };
            expected_durations.insert(regime.label().to_string(), dur);
        }
    }

    // Stationary distribution via power iteration on the row-stochastic matrix.
    let stationary_vec = stationary_distribution(&transition, m);
    let mut stationary: BTreeMap<String, f32> = BTreeMap::new();
    for i in 0..m
    {
        if stationary_vec[i] > 0.0
        {
            stationary.insert(MarketRegime::ALL[i].label().to_string(), stationary_vec[i]);
        }
    }

    let last = labels.len() - 1;
    let current = labels[last];
    let vol_pct = percentile_rank(&sorted, vols[last]);
    let hurst = hurst_exponent(&rets);

    Some(RegimeReport {
        current,
        current_label: current.label().to_string(),
        posture: current.posture().to_string(),
        realized_vol: vols[last] * cfg.periods_per_year.max(0.0).sqrt(),
        vol_percentile: vol_pct,
        trend_slope: slopes[last],
        trend_strength: tstats[last],
        hurst,
        hurst_interpretation: hurst_reading(hurst).to_string(),
        num_labeled: labels.len(),
        regime_counts,
        regime_order: MarketRegime::ALL
            .iter()
            .map(|r| r.label().to_string())
            .collect(),
        transition_matrix: transition,
        expected_durations,
        stationary,
    })
}

/// Stationary distribution `π = πP` by power iteration. Rows for unobserved
/// states are all-zero and simply contribute (and receive) nothing; the result
/// is renormalized over the mass that survives. Deterministic: fixed iteration
/// count, uniform start.
fn stationary_distribution(p: &[Vec<f32>], m: usize) -> Vec<f32> {
    // Start uniform over states that have an outgoing row (were observed as a
    // "from" state); if none, fall back to fully uniform.
    let mut active: Vec<bool> = (0..m).map(|i| p[i].iter().sum::<f32>() > 0.0).collect();
    if !active.iter().any(|&a| a)
    {
        active = vec![true; m];
    }
    let n_active = active.iter().filter(|&&a| a).count() as f32;
    let mut v: Vec<f32> = active
        .iter()
        .map(|&a| if a { 1.0 / n_active } else { 0.0 })
        .collect();

    for _ in 0..500
    {
        let mut next = vec![0.0f32; m];
        for i in 0..m
        {
            if v[i] == 0.0
            {
                continue;
            }
            let row_sum: f32 = p[i].iter().sum();
            if row_sum > 0.0
            {
                for j in 0..m
                {
                    next[j] += v[i] * p[i][j];
                }
            }
            else
            {
                // Absorbing/unobserved "from" state: keep its mass in place so
                // the vector stays a probability distribution.
                next[i] += v[i];
            }
        }
        let total: f32 = next.iter().sum();
        if total > 1e-12
        {
            for x in next.iter_mut()
            {
                *x /= total;
            }
        }
        v = next;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candle(ts: i64, close: f32) -> Candle {
        Candle {
            ts_ms: ts,
            open: close,
            high: close * 1.002,
            low: close * 0.998,
            close,
            volume: 100.0,
        }
    }

    fn series(closes: &[f32]) -> Vec<Candle> {
        closes
            .iter()
            .enumerate()
            .map(|(i, &c)| candle(i as i64 * 60_000, c))
            .collect()
    }

    #[test]
    fn realized_vol_scales_with_sqrt_ppy() {
        let r = vec![0.01, -0.01, 0.02, -0.02, 0.015, -0.005];
        let daily = realized_volatility(&r, 1.0);
        let annual = realized_volatility(&r, 365.0);
        assert!((annual / daily - 365.0f32.sqrt()).abs() < 1e-3);
    }

    #[test]
    fn ols_slope_recovers_a_line() {
        let ys: Vec<f32> = (0..20).map(|i| 3.0 + 2.0 * i as f32).collect();
        assert!((ols_slope(&ys) - 2.0).abs() < 1e-3);
    }

    #[test]
    fn hurst_high_for_trend_low_for_meanrevert() {
        // Strong deterministic trend -> persistent -> H well above 0.5.
        let trend: Vec<f32> = (0..256).map(|i| i as f32 * 0.5).collect();
        let h_trend = hurst_exponent(&trend);
        // Alternating series -> anti-persistent -> H below 0.5.
        let flip: Vec<f32> = (0..256)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let h_flip = hurst_exponent(&flip);
        assert!(h_trend > 0.55, "trend hurst {h_trend}");
        assert!(h_flip < 0.45, "flip hurst {h_flip}");
    }

    #[test]
    fn detect_none_when_too_short() {
        let candles = series(&[100.0, 101.0, 102.0]);
        assert!(detect(&candles, &RegimeConfig::default()).is_none());
    }

    #[test]
    fn clean_uptrend_is_bullish() {
        // Smooth low-vol uptrend -> BullCalm, positive trend strength.
        let closes: Vec<f32> = (0..200).map(|i| 100.0 * 1.003f32.powi(i)).collect();
        let cfg = RegimeConfig {
            vol_window: 15,
            trend_window: 20,
            ..Default::default()
        };
        let rep = detect(&series(&closes), &cfg).unwrap();
        assert!(rep.trend_strength > 0.0, "t {}", rep.trend_strength);
        assert!(
            matches!(
                rep.current,
                MarketRegime::BullCalm | MarketRegime::BullVolatile
            ),
            "regime {:?}",
            rep.current
        );
    }

    #[test]
    fn clean_downtrend_is_bearish() {
        let closes: Vec<f32> = (0..200).map(|i| 100.0 * 0.997f32.powi(i)).collect();
        let cfg = RegimeConfig {
            vol_window: 15,
            trend_window: 20,
            ..Default::default()
        };
        let rep = detect(&series(&closes), &cfg).unwrap();
        assert!(rep.trend_strength < 0.0, "t {}", rep.trend_strength);
        assert!(
            matches!(
                rep.current,
                MarketRegime::BearCalm | MarketRegime::BearVolatile
            ),
            "regime {:?}",
            rep.current
        );
    }

    #[test]
    fn choppy_flat_market_is_range() {
        // Oscillate around a level with no drift -> Range.
        let closes: Vec<f32> = (0..200)
            .map(|i| 100.0 + 2.0 * (i as f32 * 0.4).sin())
            .collect();
        let cfg = RegimeConfig {
            vol_window: 15,
            trend_window: 20,
            ..Default::default()
        };
        let rep = detect(&series(&closes), &cfg).unwrap();
        assert_eq!(rep.current, MarketRegime::Range, "t {}", rep.trend_strength);
    }

    #[test]
    fn transition_rows_are_stochastic() {
        let closes: Vec<f32> = (0..300)
            .map(|i| 100.0 + i as f32 * 0.2 + 5.0 * (i as f32 * 0.3).sin())
            .collect();
        let rep = detect(&series(&closes), &RegimeConfig::default()).unwrap();
        for (i, row) in rep.transition_matrix.iter().enumerate()
        {
            let sum: f32 = row.iter().sum();
            // Each row is either all-zero (unobserved from-state) or sums to 1.
            assert!(
                sum.abs() < 1e-4 || (sum - 1.0).abs() < 1e-4,
                "row {i} sum {sum}"
            );
        }
    }

    #[test]
    fn stationary_is_a_distribution() {
        let closes: Vec<f32> = (0..300)
            .map(|i| 100.0 + i as f32 * 0.2 + 5.0 * (i as f32 * 0.3).sin())
            .collect();
        let rep = detect(&series(&closes), &RegimeConfig::default()).unwrap();
        let total: f32 = rep.stationary.values().sum();
        assert!((total - 1.0).abs() < 1e-3, "stationary sum {total}");
        assert!(rep.stationary.values().all(|&x| x >= 0.0));
    }

    #[test]
    fn detect_is_deterministic() {
        let closes: Vec<f32> = (0..250)
            .map(|i| 100.0 + i as f32 * 0.1 + 3.0 * (i as f32 * 0.5).sin())
            .collect();
        let a = detect(&series(&closes), &RegimeConfig::default()).unwrap();
        let b = detect(&series(&closes), &RegimeConfig::default()).unwrap();
        assert_eq!(a.current, b.current);
        assert_eq!(a.realized_vol, b.realized_vol);
        assert_eq!(a.hurst, b.hurst);
    }
}
