//! Risk engine — position sizing, stop-loss, and drawdown control.
//!
//! All risk calculations are **deterministic** and derived from the certified
//! bounds produced by `certify.rs`. The engine never lets a position exceed
//! the max-drawdown constraint, and the stop-loss is set from the certified
//! interval width.

use serde::{Deserialize, Serialize};

use crate::agent::{Action, CertifiedPrediction};

/// Risk parameters for the trading session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Total capital (e.g. 10_000 USDT).
    pub capital: f32,
    /// Maximum fraction of capital per position (0.0–1.0).
    pub max_position_fraction: f32,
    /// Maximum drawdown allowed before circuit breaker triggers (0.0–1.0).
    pub max_drawdown: f32,
    /// Stop-loss multiplier: stop = entry ± k × uncertainty.
    pub stop_loss_k: f32,
    /// Minimum confidence (1 − uncertainty/|midpoint|) to take a position.
    pub min_confidence: f32,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            capital: 10_000.0,
            max_position_fraction: 0.10,
            max_drawdown: 0.10,
            stop_loss_k: 3.0,
            min_confidence: 0.3,
        }
    }
}

/// A sized position with stop-loss and take-profit levels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub action: Action,
    pub entry_price: f32,
    pub size: f32,
    pub stop_loss: f32,
    pub take_profit: f32,
    pub max_loss: f32,
    pub max_loss_pct: f32,
    pub confidence: f32,
    pub allowed: bool,
    pub reason: String,
}

/// Compute the position from a certified prediction and risk config.
pub fn size_position(pred: &CertifiedPrediction, cfg: &RiskConfig) -> Position {
    let entry = pred.last_close;
    let midpoint = pred.bounds.midpoint;
    let uncertainty = pred.bounds.uncertainty;
    let action = pred.action;
    let confidence = if uncertainty > 0.0 {
        (1.0 - uncertainty / midpoint.abs().max(1e-6)).clamp(0.0, 1.0)
    } else {
        1.0
    };

    // Stop-loss: entry ± k × uncertainty (opposite side of the trade).
    let (stop_loss, take_profit) = match action {
        Action::Long => (
            entry - cfg.stop_loss_k * uncertainty,
            entry + cfg.stop_loss_k * uncertainty * 2.0,
        ),
        Action::Short => (
            entry + cfg.stop_loss_k * uncertainty,
            entry - cfg.stop_loss_k * uncertainty * 2.0,
        ),
        Action::Flat => (entry, entry),
    };

    // Position size: fraction of capital scaled by confidence.
    let base_size = cfg.capital * cfg.max_position_fraction;
    let size = match action {
        Action::Flat => 0.0,
        _ => base_size * confidence,
    };

    // Max loss if stop is hit: fraction of position value at risk.
    let max_loss_frac = match action {
        Action::Long => (entry - stop_loss).abs() / entry,
        Action::Short => (stop_loss - entry).abs() / entry,
        Action::Flat => 0.0,
    };
    let max_loss = size * max_loss_frac;
    let max_loss_pct = if cfg.capital > 0.0 { max_loss / cfg.capital } else { 0.0 };

    // Gate checks.
    let allowed;
    let reason;
    if action == Action::Flat {
        allowed = false;
        reason = "flat: no position".to_string();
    } else if confidence < cfg.min_confidence {
        allowed = false;
        reason = format!("confidence {:.3} < min {:.3}", confidence, cfg.min_confidence);
    } else if max_loss_pct > cfg.max_drawdown {
        allowed = false;
        reason = format!(
            "max_loss_pct {:.4} > max_drawdown {:.4}",
            max_loss_pct, cfg.max_drawdown
        );
    } else {
        allowed = true;
        reason = format!("allowed: size={:.2} confidence={:.3}", size, confidence);
    }

    Position {
        action,
        entry_price: entry,
        size,
        stop_loss,
        take_profit,
        max_loss,
        max_loss_pct,
        confidence,
        allowed,
        reason,
    }
}

/// Track cumulative drawdown across a sequence of positions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownTracker {
    pub peak: f32,
    pub current: f32,
    pub max_drawdown_seen: f32,
    pub circuit_breaker_triggered: bool,
}

impl DrawdownTracker {
    pub fn new(capital: f32) -> Self {
        Self {
            peak: capital,
            current: capital,
            max_drawdown_seen: 0.0,
            circuit_breaker_triggered: false,
        }
    }

    /// Record a realized PnL and update drawdown.
    pub fn record_pnl(&mut self, pnl: f32, cfg: &RiskConfig) {
        self.current += pnl;
        if self.current > self.peak {
            self.peak = self.current;
        }
        let dd = if self.peak > 0.0 {
            (self.peak - self.current) / self.peak
        } else {
            0.0
        };
        if dd > self.max_drawdown_seen {
            self.max_drawdown_seen = dd;
        }
        if dd >= cfg.max_drawdown {
            self.circuit_breaker_triggered = true;
        }
    }

    /// Check if trading should be halted.
    pub fn should_halt(&self) -> bool {
        self.circuit_breaker_triggered
    }
}

/// Simulate a backtest with risk management applied to each decision.
/// Returns the final equity and drawdown stats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub initial_capital: f32,
    pub final_equity: f32,
    pub total_return: f32,
    pub max_drawdown_seen: f32,
    pub num_trades: usize,
    pub num_allowed: usize,
    pub num_blocked: usize,
    pub circuit_breaker_triggered: bool,
}

/// Run a simplified backtest: for each decision, compute position, simulate
/// the next-bar return, and update the drawdown tracker.
pub fn run_backtest(
    predictions: &[CertifiedPrediction],
    cfg: &RiskConfig,
) -> BacktestResult {
    let mut tracker = DrawdownTracker::new(cfg.capital);
    let mut num_trades = 0;
    let mut num_allowed = 0;
    let mut num_blocked = 0;

    for pred in predictions {
        if tracker.should_halt() {
            break;
        }
        num_trades += 1;
        let pos = size_position(pred, cfg);
        if !pos.allowed {
            num_blocked += 1;
            continue;
        }
        num_allowed += 1;

        // Simplified PnL: use the certified midpoint as the expected return.
        let pnl = pos.size * pred.bounds.midpoint;
        tracker.record_pnl(pnl, cfg);
    }

    BacktestResult {
        initial_capital: cfg.capital,
        final_equity: tracker.current,
        total_return: (tracker.current - cfg.capital) / cfg.capital,
        max_drawdown_seen: tracker.max_drawdown_seen,
        num_trades,
        num_allowed,
        num_blocked,
        circuit_breaker_triggered: tracker.circuit_breaker_triggered,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certify::{CertifiedBounds, Interval};
    use std::collections::BTreeMap;

    fn make_pred(action: Action, midpoint: f32, uncertainty: f32, close: f32) -> CertifiedPrediction {
        CertifiedPrediction {
            symbol: "BTC/USDT".to_string(),
            action,
            raw_prediction: midpoint,
            bounds: CertifiedBounds {
                eps: 0.01,
                output: Interval::new(midpoint - uncertainty, midpoint + uncertainty),
                midpoint,
                uncertainty,
                weights_fingerprint: "abc".to_string(),
            },
            feature_attribution: BTreeMap::new(),
            snapshot_fingerprint: "def".to_string(),
            weights_fingerprint: "abc".to_string(),
            last_close: close,
        }
    }

    #[test]
    fn long_position_has_stop_below_entry() {
        let pred = make_pred(Action::Long, 0.02, 0.005, 50_000.0);
        let cfg = RiskConfig::default();
        let pos = size_position(&pred, &cfg);
        assert!(pos.stop_loss < pos.entry_price);
        assert!(pos.take_profit > pos.entry_price);
        assert!(pos.allowed);
    }

    #[test]
    fn short_position_has_stop_above_entry() {
        let pred = make_pred(Action::Short, -0.02, 0.005, 50_000.0);
        let cfg = RiskConfig::default();
        let pos = size_position(&pred, &cfg);
        assert!(pos.stop_loss > pos.entry_price);
        assert!(pos.take_profit < pos.entry_price);
        assert!(pos.allowed);
    }

    #[test]
    fn flat_is_not_allowed() {
        let pred = make_pred(Action::Flat, 0.0, 0.01, 50_000.0);
        let cfg = RiskConfig::default();
        let pos = size_position(&pred, &cfg);
        assert!(!pos.allowed);
        assert_eq!(pos.size, 0.0);
    }

    #[test]
    fn low_confidence_blocks_position() {
        let pred = make_pred(Action::Long, 0.001, 0.01, 50_000.0);
        let mut cfg = RiskConfig::default();
        cfg.min_confidence = 0.9;
        let pos = size_position(&pred, &cfg);
        assert!(!pos.allowed);
        assert!(pos.reason.contains("confidence"));
    }

    #[test]
    fn excessive_loss_blocks_position() {
        // Midpoint=0.5 (50% return), uncertainty=0.1 (10%), stop_k=100 → wide stop.
        let pred = make_pred(Action::Long, 0.5, 0.1, 100.0);
        let mut cfg = RiskConfig::default();
        cfg.max_drawdown = 0.01;
        cfg.stop_loss_k = 100.0;
        cfg.min_confidence = 0.0;
        let pos = size_position(&pred, &cfg);
        // stop = 100 - 100*0.1 = 100 - 10 = 90 → max_loss_frac = 10/100 = 0.1
        // size = 10000 * 0.1 * 0.8 = 800 → max_loss = 800*0.1 = 80
        // max_loss_pct = 80/10000 = 0.008 < 0.01... still under.
        // Use max_position_fraction = 1.0 instead.
        let cfg2 = RiskConfig {
            capital: 1000.0,
            max_position_fraction: 1.0,
            max_drawdown: 0.01,
            stop_loss_k: 100.0,
            min_confidence: 0.0,
        };
        let pos = size_position(&pred, &cfg2);
        // size = 1000 * 1.0 * 0.8 = 800
        // max_loss = 800 * 0.1 = 80
        // max_loss_pct = 80/1000 = 0.08 > 0.01 → blocked
        assert!(!pos.allowed, "should be blocked, reason: {}", pos.reason);
        assert!(pos.reason.contains("max_loss_pct"), "reason: {}", pos.reason);
    }

    #[test]
    fn drawdown_tracker_triggers_circuit_breaker() {
        let mut tracker = DrawdownTracker::new(10_000.0);
        let cfg = RiskConfig {
            max_drawdown: 0.05,
            ..Default::default()
        };
        tracker.record_pnl(-600.0, &cfg);
        assert!(tracker.should_halt());
    }

    #[test]
    fn backtest_runs_and_reports() {
        let preds = vec![
            make_pred(Action::Long, 0.02, 0.005, 50_000.0),
            make_pred(Action::Short, -0.01, 0.003, 51_000.0),
            make_pred(Action::Flat, 0.0, 0.01, 50_500.0),
        ];
        let cfg = RiskConfig::default();
        let result = run_backtest(&preds, &cfg);
        assert_eq!(result.num_trades, 3);
        assert_eq!(result.num_allowed, 2);
        assert_eq!(result.num_blocked, 1);
    }

    #[test]
    fn backtest_circuit_breaker_stops_early() {
        let preds: Vec<CertifiedPrediction> = (0..20)
            .map(|_| make_pred(Action::Long, -0.05, 0.001, 50_000.0))
            .collect();
        let mut cfg = RiskConfig::default();
        cfg.max_drawdown = 0.02;
        let result = run_backtest(&preds, &cfg);
        assert!(result.circuit_breaker_triggered);
        assert!(result.num_trades < 20);
    }
}