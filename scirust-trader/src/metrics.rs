//! Performance & risk metrics — the same statistics a professional trading desk
//! or an exchange's analytics dashboard reports on a strategy.
//!
//! Every function is a pure reduction over a slice of returns or an equity curve,
//! done in **forward order** (left fold) so the result is bit-reproducible and
//! independent of thread count — the same determinism discipline as
//! `scirust-core`. No allocations in the hot loops beyond the obvious.
//!
//! Conventions
//! -----------
//! * A *simple return* series `r_t = equity_t / equity_{t-1} − 1`.
//! * Risk-free rate is expressed **per period** (not annualised) unless noted.
//! * Annualisation multiplies a per-period mean by `periods_per_year` (ppy) and a
//!   per-period stdev by `sqrt(ppy)`. Pick `ppy` from the bar timeframe — see
//!   [`periods_per_year`].

use serde::{Deserialize, Serialize};

/// Number of bars in a trading year for a given interval string (e.g. "1m",
/// "1h", "1d"). Crypto trades 24/7/365, so a day has 1440 one-minute bars and a
/// year has 365 days. Falls back to `365.0` (daily) for anything unrecognised.
pub fn periods_per_year(interval: &str) -> f32 {
    let s = interval.trim().to_lowercase();
    let (num, unit) = s.split_at(s.find(|c: char| c.is_alphabetic()).unwrap_or(s.len()));
    let n: f32 = num.parse().unwrap_or(1.0);
    let per_day = match unit
    {
        "m" | "min" => 1440.0 / n,
        "h" | "hr" => 24.0 / n,
        "d" | "day" => 1.0 / n,
        "w" | "week" => 1.0 / (7.0 * n),
        "s" | "sec" => 86_400.0 / n,
        _ => 1.0,
    };
    (per_day * 365.0).max(1.0)
}

/// Arithmetic mean (forward reduction). `NaN`-free inputs assumed.
pub fn mean(xs: &[f32]) -> f32 {
    if xs.is_empty()
    {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for &x in xs
    {
        sum += x;
    }
    sum / xs.len() as f32
}

/// Sample standard deviation (Bessel-corrected, `n−1`). Returns 0 for `n < 2`.
pub fn stddev(xs: &[f32]) -> f32 {
    let n = xs.len();
    if n < 2
    {
        return 0.0;
    }
    let m = mean(xs);
    let mut acc = 0.0f32;
    for &x in xs
    {
        let d = x - m;
        acc += d * d;
    }
    (acc / (n as f32 - 1.0)).sqrt()
}

/// Population variance (divides by `n`). Used where a full-population second
/// moment is wanted (e.g. the continuous Kelly form).
pub fn variance_pop(xs: &[f32]) -> f32 {
    let n = xs.len();
    if n == 0
    {
        return 0.0;
    }
    let m = mean(xs);
    let mut acc = 0.0f32;
    for &x in xs
    {
        let d = x - m;
        acc += d * d;
    }
    acc / n as f32
}

/// Convert an equity curve to a simple-return series (length `n−1`).
pub fn returns_from_equity(equity: &[f32]) -> Vec<f32> {
    if equity.len() < 2
    {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(equity.len() - 1);
    for i in 1..equity.len()
    {
        let prev = equity[i - 1];
        if prev.abs() > 1e-12
        {
            out.push(equity[i] / prev - 1.0);
        }
        else
        {
            out.push(0.0);
        }
    }
    out
}

/// Total (cumulative) return of an equity curve: `last / first − 1`.
pub fn total_return(equity: &[f32]) -> f32 {
    if equity.len() < 2 || equity[0].abs() < 1e-12
    {
        return 0.0;
    }
    equity[equity.len() - 1] / equity[0] - 1.0
}

/// Compound annual growth rate from an equity curve.
///
/// `CAGR = (end/start)^(ppy / n_periods) − 1`, where `n_periods` is the number
/// of return steps (`equity.len() − 1`).
pub fn cagr(equity: &[f32], periods_per_year: f32) -> f32 {
    let n = equity.len();
    if n < 2 || equity[0] <= 0.0 || equity[n - 1] <= 0.0
    {
        return 0.0;
    }
    let ratio = equity[n - 1] / equity[0];
    let exp = periods_per_year / (n as f32 - 1.0);
    ratio.powf(exp) - 1.0
}

/// Annualised volatility: per-period sample stdev × √ppy.
pub fn volatility(returns: &[f32], periods_per_year: f32) -> f32 {
    stddev(returns) * periods_per_year.max(0.0).sqrt()
}

/// Annualised Sharpe ratio.
///
/// `Sharpe = (mean(r) − rf_per_period) / stdev(r) × √ppy`. Returns 0 when the
/// return series has no dispersion (avoids a divide-by-zero blow-up).
pub fn sharpe(returns: &[f32], rf_per_period: f32, periods_per_year: f32) -> f32 {
    let sd = stddev(returns);
    if sd < 1e-12
    {
        return 0.0;
    }
    let excess = mean(returns) - rf_per_period;
    excess / sd * periods_per_year.max(0.0).sqrt()
}

/// Downside deviation: RMS of returns **below** the target (per period).
/// The denominator is the full sample count `n` (Sortino convention), not the
/// count of downside observations.
pub fn downside_deviation(returns: &[f32], target_per_period: f32) -> f32 {
    if returns.is_empty()
    {
        return 0.0;
    }
    let mut acc = 0.0f32;
    for &r in returns
    {
        let d = (r - target_per_period).min(0.0);
        acc += d * d;
    }
    (acc / returns.len() as f32).sqrt()
}

/// Annualised Sortino ratio: excess return over downside deviation.
pub fn sortino(returns: &[f32], target_per_period: f32, periods_per_year: f32) -> f32 {
    let dd = downside_deviation(returns, target_per_period);
    if dd < 1e-12
    {
        return 0.0;
    }
    let excess = mean(returns) - target_per_period;
    excess / dd * periods_per_year.max(0.0).sqrt()
}

/// Maximum drawdown of an equity curve, as a positive fraction (0.2 = −20 %).
///
/// Returns `(max_drawdown, peak_index, trough_index)`. The peak/trough indices
/// bracket the worst peak-to-valley decline.
pub fn max_drawdown(equity: &[f32]) -> (f32, usize, usize) {
    if equity.len() < 2
    {
        return (0.0, 0, 0);
    }
    let mut peak = equity[0];
    let mut peak_idx = 0usize;
    let mut best_peak_idx = 0usize;
    let mut best_trough_idx = 0usize;
    let mut max_dd = 0.0f32;
    for (i, &e) in equity.iter().enumerate()
    {
        if e > peak
        {
            peak = e;
            peak_idx = i;
        }
        let dd = if peak > 1e-12 { (peak - e) / peak } else { 0.0 };
        if dd > max_dd
        {
            max_dd = dd;
            best_peak_idx = peak_idx;
            best_trough_idx = i;
        }
    }
    (max_dd, best_peak_idx, best_trough_idx)
}

/// Calmar ratio: annualised return (CAGR) divided by max drawdown.
pub fn calmar(equity: &[f32], periods_per_year: f32) -> f32 {
    let (mdd, _, _) = max_drawdown(equity);
    if mdd < 1e-12
    {
        return 0.0;
    }
    cagr(equity, periods_per_year) / mdd
}

/// Ulcer Index: RMS of the percentage drawdown at every point on the curve.
/// A depth-and-duration–aware alternative to max drawdown.
pub fn ulcer_index(equity: &[f32]) -> f32 {
    if equity.is_empty()
    {
        return 0.0;
    }
    let mut peak = equity[0];
    let mut acc = 0.0f32;
    for &e in equity
    {
        if e > peak
        {
            peak = e;
        }
        let dd_pct = if peak > 1e-12 { (e - peak) / peak * 100.0 } else { 0.0 };
        acc += dd_pct * dd_pct;
    }
    (acc / equity.len() as f32).sqrt()
}

/// Historical Value-at-Risk at confidence `1−alpha` (e.g. `alpha = 0.05` for
/// 95 % VaR). Returned as a **positive loss fraction**: the return at the
/// `alpha` quantile, negated. Uses the nearest-rank percentile method.
pub fn value_at_risk(returns: &[f32], alpha: f32) -> f32 {
    if returns.is_empty()
    {
        return 0.0;
    }
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let a = alpha.clamp(0.0, 1.0);
    let idx = ((a * sorted.len() as f32).ceil() as usize).saturating_sub(1);
    let q = sorted[idx.min(sorted.len() - 1)];
    (-q).max(0.0)
}

/// Historical Conditional VaR (Expected Shortfall): the mean loss in the worst
/// `alpha` tail, as a positive loss fraction.
pub fn conditional_value_at_risk(returns: &[f32], alpha: f32) -> f32 {
    if returns.is_empty()
    {
        return 0.0;
    }
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let a = alpha.clamp(0.0, 1.0);
    let count = ((a * sorted.len() as f32).ceil() as usize).max(1).min(sorted.len());
    let mut sum = 0.0f32;
    for &r in &sorted[..count]
    {
        sum += r;
    }
    (-(sum / count as f32)).max(0.0)
}

/// Pearson correlation between two equal-length series. Returns 0 if either has
/// no dispersion or the lengths differ.
pub fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    if n < 2 || b.len() != n
    {
        return 0.0;
    }
    let ma = mean(a);
    let mb = mean(b);
    let mut cov = 0.0f32;
    let mut va = 0.0f32;
    let mut vb = 0.0f32;
    for i in 0..n
    {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    let denom = (va * vb).sqrt();
    if denom < 1e-12
    {
        return 0.0;
    }
    cov / denom
}

/// Beta of an asset's returns against a market's returns:
/// `Cov(asset, market) / Var(market)`.
pub fn beta(asset: &[f32], market: &[f32]) -> f32 {
    let n = asset.len();
    if n < 2 || market.len() != n
    {
        return 0.0;
    }
    let ma = mean(asset);
    let mm = mean(market);
    let mut cov = 0.0f32;
    let mut vm = 0.0f32;
    for i in 0..n
    {
        let da = asset[i] - ma;
        let dm = market[i] - mm;
        cov += da * dm;
        vm += dm * dm;
    }
    if vm < 1e-12
    {
        return 0.0;
    }
    cov / vm
}

/// Kelly fraction from win probability `p` and payoff ratio `b` (avg win /
/// avg loss, both positive): `f* = p − (1 − p) / b`. Clamped to `[0, 1]`.
pub fn kelly_discrete(p: f32, b: f32) -> f32 {
    if b <= 0.0
    {
        return 0.0;
    }
    (p - (1.0 - p) / b).clamp(0.0, 1.0)
}

/// Continuous Kelly fraction from a return series: `mean / variance`
/// (the leverage that maximises expected log-growth under a Gaussian
/// approximation). Clamped to `[0, 1]` so it never recommends leverage here.
pub fn kelly_continuous(returns: &[f32]) -> f32 {
    let var = variance_pop(returns);
    if var < 1e-12
    {
        return 0.0;
    }
    (mean(returns) / var).clamp(0.0, 1.0)
}

/// Statistics over a set of realised trade PnLs (in quote currency).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TradeStats {
    pub num_trades: usize,
    pub num_wins: usize,
    pub num_losses: usize,
    pub win_rate: f32,
    pub gross_profit: f32,
    pub gross_loss: f32,
    /// Gross profit / |gross loss|. `f32::INFINITY` if there are no losses.
    pub profit_factor: f32,
    pub avg_win: f32,
    pub avg_loss: f32,
    /// Per-trade expectancy: `win_rate·avg_win − loss_rate·|avg_loss|`.
    pub expectancy: f32,
    /// Payoff ratio `avg_win / |avg_loss|` (0 if no losses recorded).
    pub payoff_ratio: f32,
}

/// Aggregate realised-PnL statistics from a list of closed-trade PnLs.
pub fn trade_stats(pnls: &[f32]) -> TradeStats {
    let num_trades = pnls.len();
    let mut num_wins = 0usize;
    let mut num_losses = 0usize;
    let mut gross_profit = 0.0f32;
    let mut gross_loss = 0.0f32; // accumulated as a negative number
    for &p in pnls
    {
        if p > 0.0
        {
            num_wins += 1;
            gross_profit += p;
        }
        else if p < 0.0
        {
            num_losses += 1;
            gross_loss += p;
        }
    }
    let win_rate = if num_trades > 0
    {
        num_wins as f32 / num_trades as f32
    }
    else
    {
        0.0
    };
    let avg_win = if num_wins > 0 { gross_profit / num_wins as f32 } else { 0.0 };
    let avg_loss = if num_losses > 0 { gross_loss / num_losses as f32 } else { 0.0 };
    let profit_factor = if gross_loss.abs() > 1e-12
    {
        gross_profit / gross_loss.abs()
    }
    else if gross_profit > 0.0
    {
        f32::INFINITY
    }
    else
    {
        0.0
    };
    // Loss rate is the fraction of *losing* trades, not `1 − win_rate`: a
    // break-even (exactly 0) trade is neither a win nor a loss and must not be
    // counted against expectancy.
    let loss_rate = if num_trades > 0 { num_losses as f32 / num_trades as f32 } else { 0.0 };
    let expectancy = win_rate * avg_win - loss_rate * avg_loss.abs();
    let payoff_ratio = if avg_loss.abs() > 1e-12 { avg_win / avg_loss.abs() } else { 0.0 };
    TradeStats {
        num_trades,
        num_wins,
        num_losses,
        win_rate,
        gross_profit,
        gross_loss,
        profit_factor,
        avg_win,
        avg_loss,
        expectancy,
        payoff_ratio,
    }
}

/// A full performance report — what a backtest or a live account summary
/// hands back to the agent. All ratios are annualised where applicable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub periods_per_year: f32,
    pub total_return: f32,
    pub cagr: f32,
    pub volatility: f32,
    pub sharpe: f32,
    pub sortino: f32,
    pub calmar: f32,
    pub max_drawdown: f32,
    pub ulcer_index: f32,
    pub var_95: f32,
    pub cvar_95: f32,
    pub kelly_continuous: f32,
    pub trades: TradeStats,
}

impl PerformanceReport {
    /// Compute a full report from an equity curve, the closed-trade PnLs, and
    /// the bar timeframe. `rf_per_period` is the per-bar risk-free rate
    /// (usually 0 for crypto).
    pub fn from_curve(
        equity: &[f32],
        trade_pnls: &[f32],
        periods_per_year: f32,
        rf_per_period: f32,
    ) -> Self {
        let returns = returns_from_equity(equity);
        let (mdd, _, _) = max_drawdown(equity);
        PerformanceReport {
            periods_per_year,
            total_return: total_return(equity),
            cagr: cagr(equity, periods_per_year),
            volatility: volatility(&returns, periods_per_year),
            sharpe: sharpe(&returns, rf_per_period, periods_per_year),
            sortino: sortino(&returns, rf_per_period, periods_per_year),
            calmar: calmar(equity, periods_per_year),
            max_drawdown: mdd,
            ulcer_index: ulcer_index(equity),
            var_95: value_at_risk(&returns, 0.05),
            cvar_95: conditional_value_at_risk(&returns, 0.05),
            kelly_continuous: kelly_continuous(&returns),
            trades: trade_stats(trade_pnls),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn ppy_matches_timeframes() {
        assert!(approx(periods_per_year("1d"), 365.0, 1.0));
        assert!(approx(periods_per_year("1h"), 24.0 * 365.0, 1.0));
        assert!(approx(periods_per_year("1m"), 1440.0 * 365.0, 1.0));
        assert!(approx(periods_per_year("15m"), 96.0 * 365.0, 1.0));
        assert!(approx(periods_per_year("4h"), 6.0 * 365.0, 1.0));
    }

    #[test]
    fn returns_and_total() {
        let eq = vec![100.0, 110.0, 99.0];
        let r = returns_from_equity(&eq);
        assert!(approx(r[0], 0.10, 1e-5));
        assert!(approx(r[1], -0.10, 1e-5));
        assert!(approx(total_return(&eq), -0.01, 1e-5));
    }

    #[test]
    fn stddev_matches_hand_computation() {
        // sample stdev of [2,4,4,4,5,5,7,9] is 2.138... (n-1 = 7)
        let xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!(approx(stddev(&xs), 2.13809, 1e-3));
    }

    #[test]
    fn max_drawdown_finds_worst_decline() {
        // peak 120 at idx 2, trough 80 at idx 4 -> dd = 40/120 = 0.3333
        let eq = vec![100.0, 110.0, 120.0, 90.0, 80.0, 130.0];
        let (mdd, peak, trough) = max_drawdown(&eq);
        assert!(approx(mdd, 1.0 / 3.0, 1e-4));
        assert_eq!(peak, 2);
        assert_eq!(trough, 4);
    }

    #[test]
    fn sharpe_zero_when_flat() {
        let r = vec![0.01, 0.01, 0.01];
        assert_eq!(sharpe(&r, 0.0, 365.0), 0.0);
    }

    #[test]
    fn sharpe_positive_for_upward_low_vol() {
        let r = vec![0.01, 0.012, 0.009, 0.011, 0.010];
        assert!(sharpe(&r, 0.0, 365.0) > 0.0);
    }

    #[test]
    fn sortino_ignores_upside_vol() {
        // All positive returns -> downside deviation 0 -> sortino 0 by guard.
        let r = vec![0.01, 0.05, 0.02];
        assert_eq!(sortino(&r, 0.0, 365.0), 0.0);
        // With a loss and a positive mean, downside deviation is non-zero and
        // the ratio is finite and positive.
        let r2 = vec![0.03, -0.01, 0.04];
        assert!(sortino(&r2, 0.0, 365.0) > 0.0);
    }

    #[test]
    fn var_and_cvar_are_positive_losses() {
        let r: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 1000.0).collect();
        let var = value_at_risk(&r, 0.05);
        let cvar = conditional_value_at_risk(&r, 0.05);
        assert!(var > 0.0);
        assert!(cvar >= var, "CVaR ({cvar}) must be >= VaR ({var})");
    }

    #[test]
    fn correlation_perfect_and_anti() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [2.0, 4.0, 6.0, 8.0];
        let c = [8.0, 6.0, 4.0, 2.0];
        assert!(approx(correlation(&a, &b), 1.0, 1e-4));
        assert!(approx(correlation(&a, &c), -1.0, 1e-4));
    }

    #[test]
    fn beta_of_scaled_market() {
        let market = [0.01, -0.02, 0.03, -0.01];
        let asset: Vec<f32> = market.iter().map(|x| x * 2.0).collect();
        assert!(approx(beta(&asset, &market), 2.0, 1e-4));
    }

    #[test]
    fn kelly_discrete_even_money() {
        // p=0.6, b=1 -> f* = 0.6 - 0.4 = 0.2
        assert!(approx(kelly_discrete(0.6, 1.0), 0.2, 1e-5));
        // Negative edge clamps to 0.
        assert_eq!(kelly_discrete(0.4, 1.0), 0.0);
    }

    #[test]
    fn trade_stats_win_rate_and_profit_factor() {
        let pnls = vec![100.0, -50.0, 200.0, -50.0];
        let s = trade_stats(&pnls);
        assert_eq!(s.num_trades, 4);
        assert_eq!(s.num_wins, 2);
        assert_eq!(s.num_losses, 2);
        assert!(approx(s.win_rate, 0.5, 1e-5));
        // gross_profit 300, gross_loss 100 -> pf 3.0
        assert!(approx(s.profit_factor, 3.0, 1e-4));
        assert!(approx(s.avg_win, 150.0, 1e-3));
        assert!(approx(s.avg_loss, -50.0, 1e-3));
        // expectancy = 0.5*150 - 0.5*50 = 50
        assert!(approx(s.expectancy, 50.0, 1e-3));
    }

    #[test]
    fn expectancy_excludes_breakeven_trades() {
        // 1 win (+100), 1 loss (−50), 2 break-even (0.0). loss_rate must be
        // 1/4 = 0.25 (not 1 − win_rate = 0.75).
        let s = trade_stats(&[100.0, -50.0, 0.0, 0.0]);
        assert_eq!(s.num_wins, 1);
        assert_eq!(s.num_losses, 1);
        // expectancy = 0.25*100 − 0.25*50 = 12.5
        assert!(approx(s.expectancy, 12.5, 1e-3), "expectancy {}", s.expectancy);
    }

    #[test]
    fn profit_factor_infinite_with_no_losses() {
        let s = trade_stats(&[10.0, 20.0]);
        assert!(s.profit_factor.is_infinite());
    }

    #[test]
    fn full_report_is_finite() {
        let eq = vec![10_000.0, 10_100.0, 10_050.0, 10_300.0, 10_250.0, 10_500.0];
        let pnls = vec![100.0, -50.0, 250.0, -50.0, 250.0];
        let rep = PerformanceReport::from_curve(&eq, &pnls, periods_per_year("1d"), 0.0);
        assert!(rep.sharpe.is_finite());
        assert!(rep.max_drawdown >= 0.0);
        assert!(rep.total_return > 0.0);
        assert_eq!(rep.trades.num_trades, 5);
    }
}
