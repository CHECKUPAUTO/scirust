//! Parameter optimization — with an overfitting gate baked in.
//!
//! Sweeping a strategy's parameters and keeping the best backtest is the single
//! fastest way to fool yourself: with enough knobs, *something* always fits the
//! past. This module tunes parameters the honest way, mirroring how a
//! professional desk validates a systematic strategy:
//!
//! 1. **Split** the history into an in-sample **train** portion and an untouched
//!    **holdout** the search never sees.
//! 2. **Search** the parameter grid on *train only*, and rank candidates not by
//!    their single full-period fit but by their **walk-forward out-of-sample
//!    consistency** across independent sub-windows (via [`crate::robustness`]).
//!    A parameter set that only works on one lucky stretch scores poorly even
//!    in-sample.
//! 3. **Confirm** the finalists on the holdout. The train→holdout **Sharpe
//!    degradation** (`overfit_gap`) is the tell: a robust edge barely drops, an
//!    overfit one collapses or flips negative.
//! 4. **Verdict** — a plain-language read the agent can act on: trust it, size
//!    it down, or discard it.
//!
//! Everything is deterministic: the grid is enumerated in a fixed order, the
//! backtester and walk-forward are pure, and ties break by generation order.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::backtest::{BacktestConfig, run_backtest};
use crate::market::Candle;
use crate::robustness::walk_forward;
use crate::strategy::strategy_from_spec;

/// One axis of the search grid: a parameter name and the values to try.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamAxis {
    pub name: String,
    pub values: Vec<f32>,
}

impl ParamAxis {
    pub fn new(name: impl Into<String>, values: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            values,
        }
    }
}

/// How candidates are ranked *in-sample* (on the train split).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Objective {
    /// Mean walk-forward window return, discounted by consistency for a positive
    /// edge — the default. Rewards an edge that is both positive *and* reliable
    /// across sub-periods; ranks losers purely by how much they lose.
    ReturnConsistency,
    /// Fraction of profitable walk-forward windows (tie-broken by mean return).
    Consistency,
    /// Mean walk-forward window return.
    MeanReturn,
    /// The worst single walk-forward window — the most conservative (minimax).
    WorstWindow,
    /// Full train-period Sharpe ratio.
    Sharpe,
}

impl Objective {
    pub fn parse(s: &str) -> Option<Objective> {
        match s.trim().to_lowercase().as_str()
        {
            "return_consistency" | "default" => Some(Objective::ReturnConsistency),
            "consistency" => Some(Objective::Consistency),
            "mean_return" | "return" => Some(Objective::MeanReturn),
            "worst_window" | "worst" | "minimax" => Some(Objective::WorstWindow),
            "sharpe" => Some(Objective::Sharpe),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self
        {
            Objective::ReturnConsistency => "return×consistency (out-of-sample)",
            Objective::Consistency => "walk-forward consistency",
            Objective::MeanReturn => "mean walk-forward return",
            Objective::WorstWindow => "worst walk-forward window",
            Objective::Sharpe => "train Sharpe",
        }
    }
}

/// Tuning for [`optimize`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizeConfig {
    /// Fraction of the history used for the in-sample search; the rest is the
    /// untouched holdout. Clamped to `[0.3, 0.9]`.
    pub train_frac: f32,
    /// Walk-forward windows within the train split.
    pub wf_windows: usize,
    /// The in-sample ranking objective.
    pub objective: Objective,
    /// Leaderboard size (candidates confirmed on the holdout).
    pub top_k: usize,
    /// Hard cap on grid combinations evaluated. A larger grid is strided down to
    /// this many, evenly spread, so the sweep stays bounded.
    pub max_combos: usize,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            train_frac: 0.7,
            wf_windows: 4,
            objective: Objective::ReturnConsistency,
            top_k: 10,
            max_combos: 256,
        }
    }
}

/// One evaluated parameter set, with in-sample and holdout performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub params: BTreeMap<String, f32>,
    /// Full train-period total return.
    pub train_return: f32,
    /// Train-period annualized Sharpe.
    pub train_sharpe: f32,
    /// Fraction of profitable walk-forward windows within train.
    pub train_consistency: f32,
    /// Mean walk-forward window return within train.
    pub train_mean_window: f32,
    /// Worst walk-forward window return within train.
    pub train_worst_window: f32,
    /// The in-sample ranking score (per the chosen objective).
    pub objective_score: f32,
    /// Holdout total return — the honest out-of-sample estimate.
    pub holdout_return: f32,
    /// Holdout annualized Sharpe.
    pub holdout_sharpe: f32,
    /// Holdout max drawdown.
    pub holdout_max_drawdown: f32,
    /// Holdout trade count.
    pub holdout_trades: usize,
    /// Train Sharpe − holdout Sharpe: the overfitting tell. Large positive ⇒ the
    /// in-sample fit flattered the parameters.
    pub overfit_gap: f32,
}

/// The optimization report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizeReport {
    pub strategy: String,
    pub objective: String,
    pub train_bars: usize,
    pub holdout_bars: usize,
    /// Total valid grid combinations before the `max_combos` cap.
    pub grid_size: usize,
    /// Combinations actually evaluated.
    pub num_evaluated: usize,
    /// Whether the grid was strided down to fit `max_combos`.
    pub truncated: bool,
    /// The best candidate by the in-sample objective, confirmed on the holdout.
    pub best: Candidate,
    /// Top-`k` candidates (holdout-confirmed), best first.
    pub leaderboard: Vec<Candidate>,
    /// Plain-language read on whether the best parameters survive out-of-sample.
    pub verdict: String,
}

/// A sensible default parameter grid for each built-in strategy, so the agent
/// can optimize without hand-specifying axes.
pub fn default_axes(strategy_name: &str) -> Vec<ParamAxis> {
    match strategy_name
    {
        "sma_cross" | "ema_cross" => vec![
            ParamAxis::new("fast", vec![5.0, 10.0, 15.0, 20.0]),
            ParamAxis::new("slow", vec![30.0, 50.0, 100.0, 200.0]),
        ],
        "rsi_reversion" => vec![
            ParamAxis::new("period", vec![7.0, 14.0, 21.0]),
            ParamAxis::new("oversold", vec![20.0, 30.0]),
            ParamAxis::new("overbought", vec![70.0, 80.0]),
        ],
        "macd" => vec![
            ParamAxis::new("fast", vec![8.0, 12.0]),
            ParamAxis::new("slow", vec![21.0, 26.0]),
            ParamAxis::new("signal", vec![9.0]),
        ],
        "bollinger_breakout" => vec![
            ParamAxis::new("period", vec![14.0, 20.0, 30.0]),
            ParamAxis::new("k", vec![1.5, 2.0, 2.5]),
        ],
        "donchian_breakout" => vec![ParamAxis::new("period", vec![10.0, 20.0, 55.0])],
        "supertrend" => vec![
            ParamAxis::new("period", vec![7.0, 10.0, 14.0]),
            ParamAxis::new("mult", vec![2.0, 3.0, 4.0]),
        ],
        "momentum" => vec![ParamAxis::new("lookback", vec![10.0, 20.0, 40.0])],
        _ => Vec::new(),
    }
}

/// Cartesian product of the axes as `(name, value)` assignments, in a fixed
/// order. Bounded: stops growing past a hard ceiling so pathological grids can't
/// exhaust memory.
fn cartesian(axes: &[ParamAxis]) -> Vec<Vec<(String, f32)>> {
    const HARD_CEIL: usize = 65_536;
    let mut combos: Vec<Vec<(String, f32)>> = vec![Vec::new()];
    for axis in axes
    {
        if axis.values.is_empty()
        {
            continue;
        }
        let mut next = Vec::new();
        for combo in &combos
        {
            for &v in &axis.values
            {
                let mut c = combo.clone();
                c.push((axis.name.clone(), v));
                next.push(c);
            }
        }
        combos = next;
        if combos.len() > HARD_CEIL
        {
            break;
        }
    }
    combos
}

/// A combo is invalid if it pairs a `fast` window at or above its `slow` window.
fn combo_is_valid(combo: &[(String, f32)]) -> bool {
    let get = |k: &str| combo.iter().find(|(n, _)| n == k).map(|(_, v)| *v);
    match (get("fast"), get("slow"))
    {
        (Some(f), Some(s)) => f < s,
        _ => true,
    }
}

fn objective_score(
    obj: Objective,
    consistency: f32,
    mean_window: f32,
    worst_window: f32,
    train_sharpe: f32,
) -> f32 {
    match obj
    {
        // A positive edge is discounted by how often it fails; a losing edge is
        // ranked by its loss (consistency can't rescue it).
        Objective::ReturnConsistency =>
        {
            if mean_window > 0.0
            {
                mean_window * consistency
            }
            else
            {
                mean_window
            }
        },
        Objective::Consistency => consistency + 1e-6 * mean_window,
        Objective::MeanReturn => mean_window,
        Objective::WorstWindow => worst_window,
        Objective::Sharpe => train_sharpe,
    }
}

fn verdict(best: &Candidate) -> String {
    let ret = best.holdout_return * 100.0;
    if best.holdout_return <= 0.0
    {
        format!(
            "OVERFIT / NO EDGE — the best in-sample parameters lose out-of-sample \
             (holdout return {ret:+.2}%, Sharpe {:.2}). Do not trade this.",
            best.holdout_sharpe
        )
    }
    else if best.overfit_gap > 1.0 || best.holdout_sharpe < 0.5 * best.train_sharpe.max(0.0)
    {
        format!(
            "PARTIAL — positive out-of-sample but materially degraded from in-sample \
             (Sharpe {:.2}→{:.2}, holdout return {ret:+.2}%). Size down and re-validate.",
            best.train_sharpe, best.holdout_sharpe
        )
    }
    else
    {
        format!(
            "ROBUST — holds up out-of-sample (holdout return {ret:+.2}%, Sharpe {:.2}); \
             in-sample→holdout degradation is modest ({:.2} Sharpe).",
            best.holdout_sharpe, best.overfit_gap
        )
    }
}

/// Optimize `strategy_name`'s parameters over `axes`, guarding against
/// overfitting via a train/holdout split and walk-forward in-sample ranking.
///
/// `base_params` fixes any parameters not on an axis (merged into every combo).
/// Returns `None` if there is too little data to form a train and holdout split.
pub fn optimize(
    strategy_name: &str,
    axes: &[ParamAxis],
    base_params: &BTreeMap<String, f32>,
    candles: &[Candle],
    cfg: &BacktestConfig,
    opt: &OptimizeConfig,
) -> Option<OptimizeReport> {
    let n = candles.len();
    let train_frac = opt.train_frac.clamp(0.3, 0.9);
    let split = ((n as f32) * train_frac).round() as usize;
    // Need a usable train and a non-trivial holdout.
    if n < 40 || split < 20 || n - split < 8
    {
        return None;
    }
    let train = &candles[..split];
    let holdout = &candles[split..];

    // Enumerate and validate the grid.
    let all = cartesian(axes);
    let valid: Vec<Vec<(String, f32)>> = all.into_iter().filter(|c| combo_is_valid(c)).collect();
    let grid_size = valid.len();
    if grid_size == 0
    {
        return None;
    }
    let max_combos = opt.max_combos.max(1);
    let step = grid_size.div_ceil(max_combos);
    let sampled: Vec<&Vec<(String, f32)>> = valid.iter().step_by(step).collect();
    let truncated = sampled.len() < grid_size;

    // Phase 1 — evaluate every sampled combo on the train split only.
    let mut candidates: Vec<Candidate> = Vec::with_capacity(sampled.len());
    for combo in &sampled
    {
        let mut params = base_params.clone();
        for (k, v) in combo.iter()
        {
            params.insert(k.clone(), *v);
        }
        // Skip combos the factory can't build.
        let Some(strat) = strategy_from_spec(strategy_name, &params)
        else
        {
            continue;
        };
        let wf = walk_forward(strat.as_ref(), train, opt.wf_windows, cfg);
        let train_bt = run_backtest(strat.as_ref(), train, cfg);
        let score = objective_score(
            opt.objective,
            wf.consistency,
            wf.mean_return,
            wf.worst_window_return,
            train_bt.performance.sharpe,
        );
        candidates.push(Candidate {
            params,
            train_return: train_bt.total_return,
            train_sharpe: train_bt.performance.sharpe,
            train_consistency: wf.consistency,
            train_mean_window: wf.mean_return,
            train_worst_window: wf.worst_window_return,
            objective_score: score,
            // Holdout filled in phase 2 for the finalists only.
            holdout_return: 0.0,
            holdout_sharpe: 0.0,
            holdout_max_drawdown: 0.0,
            holdout_trades: 0,
            overfit_gap: 0.0,
        });
    }
    if candidates.is_empty()
    {
        return None;
    }

    // Rank by the in-sample objective (stable sort ⇒ ties keep generation order).
    candidates.sort_by(|a, b| {
        b.objective_score
            .partial_cmp(&a.objective_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Phase 2 — confirm the finalists on the untouched holdout.
    let top_k = opt.top_k.clamp(1, candidates.len());
    let mut leaderboard: Vec<Candidate> = candidates.into_iter().take(top_k).collect();
    for cand in leaderboard.iter_mut()
    {
        if let Some(strat) = strategy_from_spec(strategy_name, &cand.params)
        {
            let bt = run_backtest(strat.as_ref(), holdout, cfg);
            cand.holdout_return = bt.total_return;
            cand.holdout_sharpe = bt.performance.sharpe;
            cand.holdout_max_drawdown = bt.performance.max_drawdown;
            cand.holdout_trades = bt.num_trades;
            cand.overfit_gap = cand.train_sharpe - bt.performance.sharpe;
        }
    }

    let best = leaderboard[0].clone();
    let verdict_text = verdict(&best);

    Some(OptimizeReport {
        strategy: strategy_name.to_string(),
        objective: opt.objective.label().to_string(),
        train_bars: train.len(),
        holdout_bars: holdout.len(),
        grid_size,
        num_evaluated: leaderboard.len().max(top_k).min(grid_size),
        truncated,
        best,
        leaderboard,
        verdict: verdict_text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> BacktestConfig {
        BacktestConfig {
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
        }
    }

    fn candle(ts: i64, close: f32) -> Candle {
        Candle {
            ts_ms: ts,
            open: close,
            high: close * 1.003,
            low: close * 0.997,
            close,
            volume: 100.0,
        }
    }

    fn trend(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| candle(i as i64 * 60_000, 100.0 + i as f32))
            .collect()
    }

    #[test]
    fn cartesian_is_full_product() {
        let axes = vec![
            ParamAxis::new("a", vec![1.0, 2.0]),
            ParamAxis::new("b", vec![10.0, 20.0, 30.0]),
        ];
        assert_eq!(cartesian(&axes).len(), 6);
    }

    #[test]
    fn invalid_fast_slow_combos_filtered() {
        assert!(!combo_is_valid(&[
            ("fast".into(), 30.0),
            ("slow".into(), 10.0)
        ]));
        assert!(combo_is_valid(&[
            ("fast".into(), 10.0),
            ("slow".into(), 30.0)
        ]));
        assert!(combo_is_valid(&[("period".into(), 14.0)]));
    }

    #[test]
    fn objective_parse_roundtrip() {
        assert_eq!(
            Objective::parse("consistency"),
            Some(Objective::Consistency)
        );
        assert_eq!(Objective::parse("worst"), Some(Objective::WorstWindow));
        assert_eq!(Objective::parse("nope"), None);
    }

    #[test]
    fn none_when_too_little_data() {
        let candles = trend(30);
        let axes = default_axes("sma_cross");
        assert!(
            optimize(
                "sma_cross",
                &axes,
                &BTreeMap::new(),
                &candles,
                &cfg(),
                &OptimizeConfig::default()
            )
            .is_none()
        );
    }

    #[test]
    fn optimizes_and_confirms_on_holdout() {
        // A clean persistent uptrend -> a trend-follower should be profitable
        // both in-sample and out-of-sample -> ROBUST verdict, positive holdout.
        let candles = trend(400);
        let axes = default_axes("sma_cross");
        let rep = optimize(
            "sma_cross",
            &axes,
            &BTreeMap::new(),
            &candles,
            &cfg(),
            &OptimizeConfig::default(),
        )
        .unwrap();
        assert!(rep.train_bars > 0 && rep.holdout_bars > 0);
        assert!(rep.grid_size > 1);
        assert!(!rep.leaderboard.is_empty());
        // Best params come from the factory's parameter names.
        assert!(rep.best.params.contains_key("fast"));
        assert!(rep.best.params.contains_key("slow"));
        // A real trend edge should not collapse out-of-sample.
        assert!(
            rep.best.holdout_return > 0.0,
            "holdout {}",
            rep.best.holdout_return
        );
        assert!(rep.verdict.starts_with("ROBUST") || rep.verdict.starts_with("PARTIAL"));
    }

    #[test]
    fn leaderboard_is_sorted_by_objective() {
        let candles = trend(400);
        let axes = default_axes("sma_cross");
        let rep = optimize(
            "sma_cross",
            &axes,
            &BTreeMap::new(),
            &candles,
            &cfg(),
            &OptimizeConfig::default(),
        )
        .unwrap();
        for w in rep.leaderboard.windows(2)
        {
            assert!(w[0].objective_score >= w[1].objective_score);
        }
    }

    #[test]
    fn max_combos_bounds_evaluation() {
        let candles = trend(400);
        let axes = default_axes("sma_cross"); // 4×4 = 16, minus fast>=slow invalids
        let opt = OptimizeConfig {
            max_combos: 3,
            ..Default::default()
        };
        let rep = optimize("sma_cross", &axes, &BTreeMap::new(), &candles, &cfg(), &opt).unwrap();
        assert!(rep.truncated);
        assert!(rep.leaderboard.len() <= 3);
    }

    #[test]
    fn deterministic_result() {
        let candles = trend(400);
        let axes = default_axes("supertrend");
        let a = optimize(
            "supertrend",
            &axes,
            &BTreeMap::new(),
            &candles,
            &cfg(),
            &OptimizeConfig::default(),
        )
        .unwrap();
        let b = optimize(
            "supertrend",
            &axes,
            &BTreeMap::new(),
            &candles,
            &cfg(),
            &OptimizeConfig::default(),
        )
        .unwrap();
        assert_eq!(a.best.params, b.best.params);
        assert_eq!(a.best.holdout_return, b.best.holdout_return);
        assert_eq!(a.best.objective_score, b.best.objective_score);
    }
}
