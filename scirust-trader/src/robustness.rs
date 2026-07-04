//! Robustness analysis — the honesty layer over the backtester.
//!
//! A scanner that backtests many strategies and keeps the best *will* surface
//! flukes: an edge that only existed in one lucky window, or a curve fit to
//! noise. Two tools push back on that:
//!
//! * **Walk-forward** — split the history into sequential segments and backtest
//!   each independently. A real edge persists across segments; an overfit one
//!   shows up as a single positive window surrounded by losers. The
//!   *consistency* (fraction of profitable windows) is the headline number.
//! * **Monte-Carlo** — bootstrap-resample the trade log to build a distribution
//!   of equity paths, giving percentile bands on the final equity and the max
//!   drawdown, plus the probability of loss and of **ruin**.
//!
//! Both are deterministic: the Monte-Carlo uses a seeded xorshift RNG, so the
//! same inputs and seed always produce the same distribution.

use serde::{Deserialize, Serialize};

use crate::backtest::{BacktestConfig, run_backtest};
use crate::market::Candle;
use crate::strategy::Strategy;

/// A deterministic xorshift64* RNG — reproducible bootstrap sampling.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform index in `[0, n)` (`n` must be > 0).
    fn index(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Nearest-rank percentile of a sorted slice. `q` in `[0, 1]`.
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

// ---------------------------------------------------------------------------
// Walk-forward out-of-sample consistency.
// ---------------------------------------------------------------------------

/// One walk-forward segment's result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowResult {
    pub index: usize,
    pub bars: usize,
    pub total_return: f32,
    pub sharpe: f32,
    pub max_drawdown: f32,
    pub num_trades: usize,
}

/// The aggregate walk-forward report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardReport {
    pub num_windows: usize,
    pub windows: Vec<WindowResult>,
    pub mean_return: f32,
    pub stdev_return: f32,
    /// Number of windows with a positive return.
    pub positive_windows: usize,
    /// Fraction of windows that were profitable — the robustness headline.
    pub consistency: f32,
    pub worst_window_return: f32,
    pub best_window_return: f32,
}

/// Backtest `strategy` on each of `num_windows` sequential segments of
/// `candles` and report the out-of-sample consistency across them.
pub fn walk_forward(
    strategy: &dyn Strategy,
    candles: &[Candle],
    num_windows: usize,
    cfg: &BacktestConfig,
) -> WalkForwardReport {
    let n = candles.len();
    let k = num_windows.max(1).min(n.max(1));
    let seg = n / k;
    let mut windows = Vec::with_capacity(k);
    for w in 0..k
    {
        let start = w * seg;
        // Last window absorbs the remainder.
        let end = if w + 1 == k { n } else { start + seg };
        if end <= start
        {
            continue;
        }
        let slice = &candles[start..end];
        let report = run_backtest(strategy, slice, cfg);
        windows.push(WindowResult {
            index: w,
            bars: slice.len(),
            total_return: report.total_return,
            sharpe: report.performance.sharpe,
            max_drawdown: report.performance.max_drawdown,
            num_trades: report.num_trades,
        });
    }

    let returns: Vec<f32> = windows.iter().map(|w| w.total_return).collect();
    let m = if returns.is_empty()
    {
        0.0
    }
    else
    {
        returns.iter().sum::<f32>() / returns.len() as f32
    };
    let var = if returns.len() < 2
    {
        0.0
    }
    else
    {
        returns.iter().map(|r| (r - m).powi(2)).sum::<f32>() / (returns.len() as f32 - 1.0)
    };
    let positive = windows.iter().filter(|w| w.total_return > 0.0).count();
    let consistency = if windows.is_empty()
    {
        0.0
    }
    else
    {
        positive as f32 / windows.len() as f32
    };
    let worst = returns.iter().copied().fold(f32::INFINITY, f32::min);
    let best = returns.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    WalkForwardReport {
        num_windows: windows.len(),
        windows,
        mean_return: m,
        stdev_return: var.sqrt(),
        positive_windows: positive,
        consistency,
        worst_window_return: if worst.is_finite() { worst } else { 0.0 },
        best_window_return: if best.is_finite() { best } else { 0.0 },
    }
}

// ---------------------------------------------------------------------------
// Monte-Carlo bootstrap of the trade log.
// ---------------------------------------------------------------------------

/// The Monte-Carlo risk report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloReport {
    pub num_paths: usize,
    pub num_trades: usize,
    pub starting_equity: f32,
    pub mean_final: f32,
    pub median_final: f32,
    pub p5_final: f32,
    pub p95_final: f32,
    pub worst_final: f32,
    pub best_final: f32,
    pub median_max_drawdown: f32,
    pub p95_max_drawdown: f32,
    /// P(final equity < starting equity).
    pub prob_loss: f32,
    /// P(equity touches `ruin_threshold` at any point along the path).
    pub prob_ruin: f32,
    pub ruin_threshold: f32,
}

/// Bootstrap-resample `trade_pnls` (with replacement) into `num_paths` equity
/// paths of `trade_pnls.len()` trades each, and summarise the distribution of
/// outcomes. Returns `None` if there are no trades.
///
/// Deterministic in `seed`: the same inputs and seed always yield the same
/// report.
pub fn monte_carlo(
    trade_pnls: &[f32],
    starting_equity: f32,
    num_paths: usize,
    ruin_threshold: f32,
    seed: u64,
) -> Option<MonteCarloReport> {
    let n = trade_pnls.len();
    if n == 0
    {
        return None;
    }
    let paths = num_paths.clamp(1, 200_000);
    let mut rng = Rng::new(seed);

    let mut finals = Vec::with_capacity(paths);
    let mut maxdds = Vec::with_capacity(paths);
    let mut losses = 0usize;
    let mut ruins = 0usize;

    for _ in 0..paths
    {
        let mut equity = starting_equity;
        let mut peak = starting_equity;
        let mut maxdd = 0.0f32;
        let mut ruined = false;
        for _ in 0..n
        {
            let pnl = trade_pnls[rng.index(n)];
            equity += pnl;
            if equity > peak
            {
                peak = equity;
            }
            if peak > 1e-9
            {
                let dd = (peak - equity) / peak;
                if dd > maxdd
                {
                    maxdd = dd;
                }
            }
            if equity <= ruin_threshold
            {
                ruined = true;
            }
        }
        if equity < starting_equity
        {
            losses += 1;
        }
        if ruined
        {
            ruins += 1;
        }
        finals.push(equity);
        maxdds.push(maxdd);
    }

    let mean_final = finals.iter().sum::<f32>() / paths as f32;
    finals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    maxdds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    Some(MonteCarloReport {
        num_paths: paths,
        num_trades: n,
        starting_equity,
        mean_final,
        median_final: percentile_sorted(&finals, 0.5),
        p5_final: percentile_sorted(&finals, 0.05),
        p95_final: percentile_sorted(&finals, 0.95),
        worst_final: finals[0],
        best_final: finals[finals.len() - 1],
        median_max_drawdown: percentile_sorted(&maxdds, 0.5),
        p95_max_drawdown: percentile_sorted(&maxdds, 0.95),
        prob_loss: losses as f32 / paths as f32,
        prob_ruin: ruins as f32 / paths as f32,
        ruin_threshold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::MaCross;

    fn candle(ts: i64, close: f32) -> Candle {
        Candle {
            ts_ms: ts,
            open: close,
            high: close * 1.004,
            low: close * 0.996,
            close,
            volume: 100.0,
        }
    }

    #[test]
    fn rng_is_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..100
        {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn percentiles_are_ordered() {
        let sorted: Vec<f32> = (0..100).map(|i| i as f32).collect();
        assert!(percentile_sorted(&sorted, 0.05) <= percentile_sorted(&sorted, 0.5));
        assert!(percentile_sorted(&sorted, 0.5) <= percentile_sorted(&sorted, 0.95));
    }

    #[test]
    fn walk_forward_reports_consistency() {
        // Clean uptrend -> a long trend-follower should be profitable across
        // most windows -> high consistency.
        let candles: Vec<Candle> = (0..400)
            .map(|i| candle(i as i64 * 60_000, 100.0 + i as f32))
            .collect();
        let cfg = BacktestConfig {
            fees: crate::orders::FeeSchedule {
                maker_bps: 0.0,
                taker_bps: 0.0,
            },
            slippage: crate::orders::SlippageModel {
                base_bps: 0.0,
                impact_bps: 0.0,
                ref_liquidity: 1.0,
            },
            ..Default::default()
        };
        let wf = walk_forward(&MaCross::sma(5, 20), &candles, 4, &cfg);
        assert_eq!(wf.num_windows, 4);
        assert!(wf.consistency > 0.5, "consistency {}", wf.consistency);
        assert!(wf.mean_return.is_finite());
    }

    #[test]
    fn monte_carlo_positive_edge_low_ruin() {
        // Positive-expectancy trade log: many small wins, few small losses.
        let mut pnls = vec![100.0f32; 60];
        pnls.extend(vec![-50.0f32; 40]);
        let rep = monte_carlo(&pnls, 10_000.0, 2000, 0.0, 7).unwrap();
        // Expected sum ~ 60*100 - 40*50 = 4000 -> median final well above start.
        assert!(rep.median_final > 10_000.0, "median {}", rep.median_final);
        assert!(rep.prob_ruin < 0.05, "ruin {}", rep.prob_ruin);
        assert!(rep.p5_final <= rep.median_final && rep.median_final <= rep.p95_final);
    }

    #[test]
    fn monte_carlo_negative_edge_high_loss() {
        // Negative-expectancy log -> high probability of loss.
        let mut pnls = vec![50.0f32; 40];
        pnls.extend(vec![-100.0f32; 60]);
        let rep = monte_carlo(&pnls, 10_000.0, 2000, 0.0, 7).unwrap();
        assert!(rep.prob_loss > 0.8, "prob_loss {}", rep.prob_loss);
    }

    #[test]
    fn monte_carlo_deterministic_in_seed() {
        let pnls = vec![10.0, -5.0, 20.0, -8.0, 3.0];
        let a = monte_carlo(&pnls, 1000.0, 500, 0.0, 123).unwrap();
        let b = monte_carlo(&pnls, 1000.0, 500, 0.0, 123).unwrap();
        assert_eq!(a.median_final, b.median_final);
        assert_eq!(a.prob_ruin, b.prob_ruin);
    }

    #[test]
    fn monte_carlo_none_without_trades() {
        assert!(monte_carlo(&[], 1000.0, 100, 0.0, 1).is_none());
    }

    #[test]
    fn monte_carlo_detects_ruin() {
        // All-loss log with a ruin threshold just below start -> ruin certain.
        let pnls = vec![-200.0f32; 10];
        let rep = monte_carlo(&pnls, 1000.0, 100, 500.0, 1).unwrap();
        assert!(rep.prob_ruin > 0.99, "ruin {}", rep.prob_ruin);
    }
}
