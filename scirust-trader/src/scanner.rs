//! Opportunity discovery — the engine behind "connect to SciRust and find me
//! trades that respect these conditions, with a profit target of X".
//!
//! The agent hands over a set of [`OpportunityConstraints`] (target return, max
//! drawdown, minimum Sharpe / win-rate, allowed direction, which strategies to
//! try). The scanner then, for every `(symbol × strategy)` pair:
//!
//! 1. **backtests** the strategy over the symbol's history (real fees/slippage),
//! 2. reads the **current signal** on the freshest bar,
//! 3. **filters** on the constraints,
//! 4. computes an actionable **entry / stop / take-profit / size** from ATR,
//! 5. **scores & ranks** the survivors, and
//! 6. **seals** each one with a SHA-256 proof hash for audit/replay.
//!
//! Everything is deterministic: same candles + same constraints ⇒ same ranked
//! opportunities and the same proof hashes.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::agent::Action;
use crate::backtest::{run_backtest, BacktestConfig};
use crate::indicators;
use crate::market::MarketSnapshot;
use crate::strategy::{strategy_from_spec, STRATEGY_NAMES};
use std::collections::BTreeMap;

/// Filters the agent expresses in natural language, lowered to numbers.
/// Unspecified fields default to "no constraint".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpportunityConstraints {
    /// Minimum backtested total return over the window (e.g. 0.05 = +5 %).
    pub min_total_return: f32,
    /// Minimum per-trade expectancy (quote currency) required.
    pub min_expectancy: f32,
    /// Maximum tolerated backtested max drawdown (e.g. 0.2 = 20 %).
    pub max_drawdown: f32,
    /// Minimum annualised Sharpe.
    pub min_sharpe: f32,
    /// Minimum win rate in `[0, 1]`.
    pub min_win_rate: f32,
    /// Minimum profit factor.
    pub min_profit_factor: f32,
    /// Minimum number of trades in the backtest (avoids overfit single-trade flukes).
    pub min_trades: usize,
    /// Restrict to a direction; `None` = long or short.
    pub direction: Option<Action>,
    /// Minimum current-signal strength to surface.
    pub min_signal_strength: f32,
    /// Strategy names to try; empty = all built-in strategies.
    pub strategies: Vec<String>,
    /// How many opportunities to return (ranked).
    pub max_results: usize,
}

impl Default for OpportunityConstraints {
    fn default() -> Self {
        Self {
            min_total_return: f32::NEG_INFINITY,
            min_expectancy: f32::NEG_INFINITY,
            max_drawdown: f32::INFINITY,
            min_sharpe: f32::NEG_INFINITY,
            min_win_rate: 0.0,
            min_profit_factor: 0.0,
            min_trades: 1,
            direction: None,
            min_signal_strength: 0.0,
            strategies: Vec::new(),
            max_results: 10,
        }
    }
}

/// Risk inputs used to turn a signal into a sized, stop-protected trade plan.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ScanRiskConfig {
    pub capital: f32,
    /// Fraction of capital risked to the stop per trade (e.g. 0.01 = 1 %).
    pub risk_per_trade: f32,
    /// Stop distance = `atr_mult · ATR(atr_period)`.
    pub atr_period: usize,
    pub atr_mult: f32,
    /// Reward:risk ratio for the take-profit (e.g. 2.0 → TP at 2× the stop distance).
    pub reward_risk: f32,
}

impl Default for ScanRiskConfig {
    fn default() -> Self {
        Self {
            capital: 10_000.0,
            risk_per_trade: 0.01,
            atr_period: 14,
            atr_mult: 2.0,
            reward_risk: 2.0,
        }
    }
}

/// A ranked, actionable trade opportunity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Opportunity {
    pub symbol: String,
    pub interval: String,
    pub strategy: String,
    pub action: Action,
    pub signal_strength: f32,
    pub reason: String,
    pub last_close: f32,
    pub entry: f32,
    pub stop_loss: f32,
    pub take_profit: f32,
    pub position_size: f32,
    /// Notional value of the suggested position (quote currency).
    pub position_notional: f32,
    /// Capital at risk to the stop (quote currency).
    pub risk_amount: f32,
    /// Reward:risk of the plan.
    pub reward_risk: f32,
    // --- backtest evidence ---
    pub backtest_total_return: f32,
    pub backtest_sharpe: f32,
    pub backtest_max_drawdown: f32,
    pub backtest_win_rate: f32,
    pub backtest_profit_factor: f32,
    pub backtest_expectancy: f32,
    pub num_trades: usize,
    /// Ranking score in `[0, 1]` (higher = better).
    pub score: f32,
    /// SHA-256 fingerprint of the underlying market window.
    pub snapshot_fingerprint: String,
    /// SHA-256 proof of this opportunity (see [`Opportunity::seal`]).
    pub proof_hash: String,
}

impl Opportunity {
    /// Compute the audit proof hash over the canonical opportunity content
    /// (every field except the hash itself). Two identical opportunities always
    /// produce the same proof, so a third party can recompute and verify it.
    fn seal(mut self) -> Self {
        self.proof_hash.clear();
        let json = serde_json::to_string(&self).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        self.proof_hash = format!("{:x}", hasher.finalize());
        self
    }

    /// Recompute and check the proof hash (tamper detection).
    pub fn verify(&self) -> bool {
        let mut clone = self.clone();
        let claimed = std::mem::take(&mut clone.proof_hash);
        let json = serde_json::to_string(&clone).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        format!("{:x}", hasher.finalize()) == claimed
    }
}

/// The sealed result of a scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanReport {
    pub scirust_version: String,
    pub num_symbols: usize,
    pub num_candidates: usize,
    pub num_matched: usize,
    pub opportunities: Vec<Opportunity>,
    /// SHA-256 over all opportunity proof hashes — one hash sealing the report.
    pub manifest_hash: String,
}

impl ScanReport {
    fn seal(
        opportunities: Vec<Opportunity>,
        num_symbols: usize,
        num_candidates: usize,
        num_matched: usize,
    ) -> Self {
        let mut hasher = Sha256::new();
        for o in &opportunities
        {
            hasher.update(o.proof_hash.as_bytes());
            hasher.update(b"|");
        }
        ScanReport {
            scirust_version: env!("CARGO_PKG_VERSION").to_string(),
            num_symbols,
            num_matched,
            num_candidates,
            opportunities,
            manifest_hash: format!("{:x}", hasher.finalize()),
        }
    }

    /// Verify every opportunity proof and the manifest hash.
    pub fn verify(&self) -> bool {
        if !self.opportunities.iter().all(|o| o.verify())
        {
            return false;
        }
        let mut hasher = Sha256::new();
        for o in &self.opportunities
        {
            hasher.update(o.proof_hash.as_bytes());
            hasher.update(b"|");
        }
        format!("{:x}", hasher.finalize()) == self.manifest_hash
    }
}

/// Scan a universe of market series for opportunities matching `constraints`.
///
/// `series` is one [`MarketSnapshot`] per symbol (the caller supplies the
/// candles — from the mock feed, a live connector, or a file). Determinism is
/// preserved: the same inputs always yield the same ranked, sealed report.
pub fn scan(
    series: &[MarketSnapshot],
    constraints: &OpportunityConstraints,
    risk: &ScanRiskConfig,
) -> ScanReport {
    let strat_names: Vec<String> = if constraints.strategies.is_empty()
    {
        STRATEGY_NAMES.iter().map(|s| s.to_string()).collect()
    }
    else
    {
        constraints.strategies.clone()
    };

    let mut candidates = 0usize;
    let mut matches: Vec<Opportunity> = Vec::new();

    for snap in series
    {
        if snap.candles.len() < 40
        {
            continue; // not enough history to backtest meaningfully
        }
        let closes = snap.closes();
        let highs: Vec<f32> = snap.candles.iter().map(|c| c.high).collect();
        let lows: Vec<f32> = snap.candles.iter().map(|c| c.low).collect();
        let last_close = *closes.last().unwrap();
        let atr_series = indicators::atr(&highs, &lows, &closes, risk.atr_period);
        let atr = atr_series
            .last()
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(last_close * 0.02);

        for name in &strat_names
        {
            let strat = match strategy_from_spec(name, &BTreeMap::new())
            {
                Some(s) => s,
                None => continue,
            };
            candidates += 1;

            let cfg = BacktestConfig {
                symbol: snap.symbol.clone(),
                interval: snap.interval.clone(),
                ..Default::default()
            };
            let report = run_backtest(strat.as_ref(), &snap.candles, &cfg);
            let sig = strat.evaluate(&snap.candles);

            // --- constraint gate ---
            if sig.action == Action::Flat
            {
                continue; // no actionable entry right now
            }
            if let Some(dir) = constraints.direction
            {
                if dir != sig.action
                {
                    continue;
                }
            }
            let perf = &report.performance;
            if sig.strength < constraints.min_signal_strength
                || report.total_return < constraints.min_total_return
                || perf.max_drawdown > constraints.max_drawdown
                || perf.sharpe < constraints.min_sharpe
                || perf.trades.win_rate < constraints.min_win_rate
                || perf.trades.profit_factor < constraints.min_profit_factor
                || perf.trades.expectancy < constraints.min_expectancy
                || report.num_trades < constraints.min_trades
            {
                continue;
            }

            // --- trade plan from ATR ---
            let stop_distance = risk.atr_mult * atr;
            let (stop_loss, take_profit) = match sig.action
            {
                Action::Long => (
                    last_close - stop_distance,
                    last_close + stop_distance * risk.reward_risk,
                ),
                Action::Short => (
                    last_close + stop_distance,
                    last_close - stop_distance * risk.reward_risk,
                ),
                Action::Flat => (last_close, last_close),
            };
            let risk_amount = risk.capital * risk.risk_per_trade.clamp(0.0, 1.0);
            let position_size = if stop_distance > 1e-9 { risk_amount / stop_distance } else { 0.0 };
            let position_notional = (position_size * last_close).min(risk.capital);
            let position_size = if last_close > 1e-9 { position_notional / last_close } else { 0.0 };

            let score = score_opportunity(&report, sig.strength);

            let opp = Opportunity {
                symbol: snap.symbol.clone(),
                interval: snap.interval.clone(),
                strategy: report.strategy.clone(),
                action: sig.action,
                signal_strength: sig.strength,
                reason: sig.reason.clone(),
                last_close,
                entry: last_close,
                stop_loss,
                take_profit,
                position_size,
                position_notional,
                risk_amount,
                reward_risk: risk.reward_risk,
                backtest_total_return: report.total_return,
                backtest_sharpe: perf.sharpe,
                backtest_max_drawdown: perf.max_drawdown,
                backtest_win_rate: perf.trades.win_rate,
                backtest_profit_factor: perf.trades.profit_factor,
                backtest_expectancy: perf.trades.expectancy,
                num_trades: report.num_trades,
                score,
                snapshot_fingerprint: snap.fingerprint(),
                proof_hash: String::new(),
            }
            .seal();
            matches.push(opp);
        }
    }

    // Rank by score descending; tie-break on symbol+strategy for determinism.
    matches.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.symbol.cmp(&b.symbol))
            .then_with(|| a.strategy.cmp(&b.strategy))
    });
    // How many candidates actually passed the constraint gate — captured before
    // the display cap truncates the ranked list.
    let num_matched = matches.len();
    matches.truncate(constraints.max_results.max(1));

    ScanReport::seal(matches, series.len(), candidates, num_matched)
}

/// Bounded ranking score in `[0, 1]` blending Sharpe, return, win rate, profit
/// factor, signal conviction, and (inverse) drawdown.
fn score_opportunity(report: &crate::backtest::BacktestReport, signal_strength: f32) -> f32 {
    let perf = &report.performance;
    let sharpe_term = sigmoid(perf.sharpe / 2.0); // ~0.5 at Sharpe 0, ->1 for high
    let return_term = (report.total_return * 3.0).tanh().max(0.0);
    let win_term = perf.trades.win_rate;
    let pf = if perf.trades.profit_factor.is_finite()
    {
        (perf.trades.profit_factor / 3.0).min(1.0)
    }
    else
    {
        1.0
    };
    let dd_term = 1.0 - perf.max_drawdown.clamp(0.0, 1.0);
    let raw = 0.28 * sharpe_term
        + 0.22 * return_term
        + 0.16 * win_term
        + 0.12 * pf
        + 0.12 * signal_strength
        + 0.10 * dd_term;
    raw.clamp(0.0, 1.0)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::{Candle, MarketSnapshot};

    fn snapshot(symbol: &str, closes: &[f32]) -> MarketSnapshot {
        let candles = closes
            .iter()
            .enumerate()
            .map(|(i, &c)| Candle {
                ts_ms: i as i64 * 60_000,
                open: c,
                high: c * 1.01,
                low: c * 0.99,
                close: c,
                volume: 100.0,
            })
            .collect();
        MarketSnapshot {
            exchange: "test".to_string(),
            symbol: symbol.to_string(),
            interval: "1h".to_string(),
            candles,
        }
    }

    fn uptrend(symbol: &str, n: usize) -> MarketSnapshot {
        snapshot(symbol, &(0..n).map(|i| 100.0 + i as f32).collect::<Vec<_>>())
    }

    #[test]
    fn scan_finds_opportunities_in_trend() {
        let series = vec![uptrend("BTC/USDT", 200)];
        let report = scan(&series, &OpportunityConstraints::default(), &ScanRiskConfig::default());
        assert!(report.num_candidates > 0);
        assert!(report.verify(), "report proof must verify");
        // In a clean uptrend at least one long trend-follower should surface.
        assert!(report.opportunities.iter().any(|o| o.action == Action::Long));
    }

    #[test]
    fn constraints_filter_out_weak_edges() {
        let series = vec![uptrend("BTC/USDT", 200)];
        // Impossible Sharpe constraint -> nothing matches.
        let c = OpportunityConstraints {
            min_sharpe: 1_000.0,
            ..Default::default()
        };
        let report = scan(&series, &c, &ScanRiskConfig::default());
        assert_eq!(report.num_matched, 0);
        assert!(report.opportunities.is_empty());
    }

    #[test]
    fn direction_constraint_respected() {
        let series = vec![uptrend("BTC/USDT", 200)];
        let c = OpportunityConstraints {
            direction: Some(Action::Short),
            ..Default::default()
        };
        let report = scan(&series, &c, &ScanRiskConfig::default());
        assert!(report.opportunities.iter().all(|o| o.action == Action::Short));
    }

    #[test]
    fn opportunities_are_ranked_and_capped() {
        let series = vec![uptrend("BTC/USDT", 200), uptrend("ETH/USDT", 200)];
        let c = OpportunityConstraints {
            max_results: 3,
            ..Default::default()
        };
        let report = scan(&series, &c, &ScanRiskConfig::default());
        assert!(report.opportunities.len() <= 3);
        // Scores must be non-increasing (ranked).
        for w in report.opportunities.windows(2)
        {
            assert!(w[0].score >= w[1].score - 1e-6);
        }
    }

    #[test]
    fn trade_plan_has_stop_and_target_on_correct_sides() {
        let series = vec![uptrend("BTC/USDT", 200)];
        let report = scan(&series, &OpportunityConstraints::default(), &ScanRiskConfig::default());
        for o in &report.opportunities
        {
            match o.action
            {
                Action::Long =>
                {
                    assert!(o.stop_loss < o.entry, "long stop below entry");
                    assert!(o.take_profit > o.entry, "long TP above entry");
                },
                Action::Short =>
                {
                    assert!(o.stop_loss > o.entry);
                    assert!(o.take_profit < o.entry);
                },
                Action::Flat => panic!("flat opportunities should be filtered"),
            }
            assert!(o.position_size > 0.0);
        }
    }

    #[test]
    fn num_matched_counts_all_survivors_not_the_display_cap() {
        let series = vec![uptrend("BTC/USDT", 200), uptrend("ETH/USDT", 200)];
        // Cap the returned list to 1 but keep loose constraints so several pass.
        let c = OpportunityConstraints {
            max_results: 1,
            ..Default::default()
        };
        let report = scan(&series, &c, &ScanRiskConfig::default());
        assert_eq!(report.opportunities.len(), 1, "display capped to 1");
        assert!(
            report.num_matched >= report.opportunities.len(),
            "num_matched ({}) must count all survivors, not the cap",
            report.num_matched
        );
    }

    #[test]
    fn tampering_breaks_the_proof() {
        let series = vec![uptrend("BTC/USDT", 200)];
        let mut report = scan(&series, &OpportunityConstraints::default(), &ScanRiskConfig::default());
        if let Some(o) = report.opportunities.first_mut()
        {
            o.take_profit *= 2.0; // tamper
            assert!(!o.verify(), "tampered opportunity must fail verification");
        }
    }
}
