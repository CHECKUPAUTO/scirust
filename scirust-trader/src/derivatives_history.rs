//! Rolling crypto-derivatives features over timestamp-aligned snapshots.
//!
//! The core [`crate::derivatives`] module describes one perpetual-futures
//! snapshot. This module adds deterministic historical context without turning
//! any single metric into a trading recommendation: funding and basis
//! standardisation, open-interest momentum/acceleration, liquidation intensity,
//! and a simple price/open-interest state taxonomy.

use serde::{Deserialize, Serialize};

use crate::derivatives::{
    DerivativesSnapshot, basis_bps, liquidation_imbalance, open_interest_change,
};

/// Joint direction of mark price and open interest between two observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PriceOiRegime {
    PriceUpOiUp,
    PriceUpOiDown,
    PriceDownOiUp,
    PriceDownOiDown,
    /// At least one of price or open interest did not change.
    FlatOrUnchanged,
    /// Inputs are non-finite, non-positive where a price is required, or absent.
    Unavailable,
}

/// Historical feature configuration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct DerivativesHistoryConfig {
    /// Rolling observations used for z-scores and percentile ranks.
    pub window: usize,
    /// Number of observations between the open-interest momentum endpoints.
    pub oi_lookback: usize,
}

impl Default for DerivativesHistoryConfig {
    fn default() -> Self {
        Self {
            window: 20,
            oi_lookback: 5,
        }
    }
}

/// Deterministic rolling context for the newest derivatives snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DerivativesHistoryReport {
    pub symbol: String,
    pub ts_ms: i64,
    pub observations: usize,
    pub window: usize,
    pub funding_zscore: Option<f32>,
    /// Empirical rank in `[0, 1]` within the trailing window.
    pub funding_percentile: Option<f32>,
    pub basis_zscore: Option<f32>,
    /// Empirical rank in `[0, 1]` within the trailing window.
    pub basis_percentile: Option<f32>,
    /// One-observation open-interest percentage change.
    pub open_interest_change_pct: Option<f32>,
    /// Percentage change from `oi_lookback` observations ago to now.
    pub open_interest_momentum_pct: Option<f32>,
    /// Latest one-step OI % change minus the preceding one-step OI % change.
    pub open_interest_acceleration_pct: Option<f32>,
    /// Non-negative long + short liquidation amount for the newest snapshot.
    pub liquidation_total: f32,
    pub liquidation_imbalance: f32,
    pub liquidation_zscore: Option<f32>,
    pub liquidation_percentile: Option<f32>,
    pub price_oi_regime: PriceOiRegime,
}

/// Sample z-score of the newest value within a trailing window.
///
/// Returns `None` when the window has fewer than two observations, does not fit
/// in the input, or contains non-finite values. A constant window returns
/// `Some(0.0)` because the newest observation is exactly at the window mean.
pub fn latest_zscore(values: &[f32], window: usize) -> Option<f32> {
    if window < 2 || values.len() < window
    {
        return None;
    }
    let tail = &values[values.len() - window..];
    if tail.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    let n = window as f32;
    let mean = tail.iter().sum::<f32>() / n;
    let variance = tail
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f32>()
        / (n - 1.0);
    let sd = variance.sqrt();
    if sd <= f32::EPSILON
    {
        Some(0.0)
    }
    else
    {
        Some((tail[window - 1] - mean) / sd)
    }
}

/// Empirical percentile rank of the newest value in a trailing window.
///
/// The rank is `count(x <= latest) / window`, hence is always in `(0, 1]` for a
/// valid non-empty window. Ties are intentionally inclusive and deterministic.
pub fn latest_percentile(values: &[f32], window: usize) -> Option<f32> {
    if window == 0 || values.len() < window
    {
        return None;
    }
    let tail = &values[values.len() - window..];
    if tail.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    let latest = tail[window - 1];
    let count = tail.iter().filter(|&&v| v <= latest).count();
    Some(count as f32 / window as f32)
}

/// Non-negative liquidation amount for one observation.
pub fn liquidation_total(snapshot: &DerivativesSnapshot) -> f32 {
    let long = if snapshot.long_liquidations.is_finite()
    {
        snapshot.long_liquidations.max(0.0)
    }
    else
    {
        0.0
    };
    let short = if snapshot.short_liquidations.is_finite()
    {
        snapshot.short_liquidations.max(0.0)
    }
    else
    {
        0.0
    };
    long + short
}

/// Classify the newest one-step mark-price/open-interest state.
pub fn price_oi_regime(
    previous: &DerivativesSnapshot,
    current: &DerivativesSnapshot,
) -> PriceOiRegime {
    if !previous.mark_price.is_finite()
        || !current.mark_price.is_finite()
        || !previous.open_interest.is_finite()
        || !current.open_interest.is_finite()
        || previous.mark_price <= 0.0
        || current.mark_price <= 0.0
        || previous.open_interest < 0.0
        || current.open_interest < 0.0
    {
        return PriceOiRegime::Unavailable;
    }

    let price_delta = current.mark_price - previous.mark_price;
    let oi_delta = current.open_interest - previous.open_interest;
    if price_delta > 0.0 && oi_delta > 0.0
    {
        PriceOiRegime::PriceUpOiUp
    }
    else if price_delta > 0.0 && oi_delta < 0.0
    {
        PriceOiRegime::PriceUpOiDown
    }
    else if price_delta < 0.0 && oi_delta > 0.0
    {
        PriceOiRegime::PriceDownOiUp
    }
    else if price_delta < 0.0 && oi_delta < 0.0
    {
        PriceOiRegime::PriceDownOiDown
    }
    else
    {
        PriceOiRegime::FlatOrUnchanged
    }
}

fn one_step_oi_change(history: &[DerivativesSnapshot]) -> Option<f32> {
    if history.len() < 2
    {
        return None;
    }
    let previous = &history[history.len() - 2];
    let current = &history[history.len() - 1];
    open_interest_change(previous.open_interest, current.open_interest).map(|c| c.percent)
}

fn oi_momentum(history: &[DerivativesSnapshot], lookback: usize) -> Option<f32> {
    if lookback == 0 || history.len() <= lookback
    {
        return None;
    }
    let current = &history[history.len() - 1];
    let previous = &history[history.len() - 1 - lookback];
    open_interest_change(previous.open_interest, current.open_interest).map(|c| c.percent)
}

fn oi_acceleration(history: &[DerivativesSnapshot]) -> Option<f32> {
    if history.len() < 3
    {
        return None;
    }
    let a = &history[history.len() - 3];
    let b = &history[history.len() - 2];
    let c = &history[history.len() - 1];
    let prior = open_interest_change(a.open_interest, b.open_interest)?.percent;
    let latest = open_interest_change(b.open_interest, c.open_interest)?.percent;
    Some(latest - prior)
}

/// Analyze the newest snapshot in `history` using only trailing observations.
///
/// The input order is assumed chronological. The report remains useful with
/// short history: fields whose look-back is not yet satisfied are `None` rather
/// than silently switching to a smaller window.
pub fn analyze_history(
    history: &[DerivativesSnapshot],
    cfg: &DerivativesHistoryConfig,
) -> Option<DerivativesHistoryReport> {
    let current = history.last()?;
    let funding: Vec<f32> = history.iter().map(|s| s.funding_rate).collect();
    let basis: Vec<f32> = history
        .iter()
        .map(|s| basis_bps(s.mark_price, s.spot_price).unwrap_or(f32::NAN))
        .collect();
    let liquidations: Vec<f32> = history.iter().map(liquidation_total).collect();
    let previous = history.get(history.len().wrapping_sub(2));
    let price_oi_regime = previous
        .map(|p| price_oi_regime(p, current))
        .unwrap_or(PriceOiRegime::Unavailable);

    Some(DerivativesHistoryReport {
        symbol: current.symbol.clone(),
        ts_ms: current.ts_ms,
        observations: history.len(),
        window: cfg.window,
        funding_zscore: latest_zscore(&funding, cfg.window),
        funding_percentile: latest_percentile(&funding, cfg.window),
        basis_zscore: latest_zscore(&basis, cfg.window),
        basis_percentile: latest_percentile(&basis, cfg.window),
        open_interest_change_pct: one_step_oi_change(history),
        open_interest_momentum_pct: oi_momentum(history, cfg.oi_lookback),
        open_interest_acceleration_pct: oi_acceleration(history),
        liquidation_total: liquidation_total(current),
        liquidation_imbalance: liquidation_imbalance(
            current.long_liquidations,
            current.short_liquidations,
        ),
        liquidation_zscore: latest_zscore(&liquidations, cfg.window),
        liquidation_percentile: latest_percentile(&liquidations, cfg.window),
        price_oi_regime,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(
        ts_ms: i64,
        mark_price: f32,
        funding_rate: f32,
        open_interest: f32,
        long_liquidations: f32,
        short_liquidations: f32,
    ) -> DerivativesSnapshot {
        DerivativesSnapshot {
            symbol: "BTCUSDT".to_string(),
            ts_ms,
            spot_price: 100.0,
            mark_price,
            index_price: 100.0,
            funding_rate,
            funding_interval_hours: 8.0,
            open_interest,
            long_liquidations,
            short_liquidations,
        }
    }

    #[test]
    fn latest_zscore_uses_sample_standard_deviation() {
        let z = latest_zscore(&[1.0, 2.0, 3.0], 3).unwrap();
        assert!((z - 1.0).abs() < 1e-6);
        assert!(latest_zscore(&[1.0], 2).is_none());
        assert!((latest_zscore(&[5.0, 5.0, 5.0], 3).unwrap()).abs() < 1e-6);
    }

    #[test]
    fn percentile_is_empirical_and_inclusive_on_ties() {
        assert!((latest_percentile(&[1.0, 2.0, 3.0], 3).unwrap() - 1.0).abs() < 1e-6);
        assert!((latest_percentile(&[2.0, 2.0, 1.0], 3).unwrap() - (1.0 / 3.0)).abs() < 1e-6);
        assert!(latest_percentile(&[1.0, f32::NAN], 2).is_none());
    }

    #[test]
    fn price_oi_taxonomy_covers_all_direction_pairs() {
        let base = snapshot(0, 100.0, 0.0, 100.0, 0.0, 0.0);
        assert_eq!(
            price_oi_regime(&base, &snapshot(1, 101.0, 0.0, 110.0, 0.0, 0.0)),
            PriceOiRegime::PriceUpOiUp
        );
        assert_eq!(
            price_oi_regime(&base, &snapshot(1, 101.0, 0.0, 90.0, 0.0, 0.0)),
            PriceOiRegime::PriceUpOiDown
        );
        assert_eq!(
            price_oi_regime(&base, &snapshot(1, 99.0, 0.0, 110.0, 0.0, 0.0)),
            PriceOiRegime::PriceDownOiUp
        );
        assert_eq!(
            price_oi_regime(&base, &snapshot(1, 99.0, 0.0, 90.0, 0.0, 0.0)),
            PriceOiRegime::PriceDownOiDown
        );
    }

    #[test]
    fn history_report_combines_standardized_derivatives_context() {
        let history = vec![
            snapshot(1, 100.0, 0.0, 100.0, 2.0, 8.0),
            snapshot(2, 101.0, 0.001, 110.0, 5.0, 15.0),
            snapshot(3, 102.0, 0.002, 120.0, 10.0, 20.0),
        ];
        let cfg = DerivativesHistoryConfig {
            window: 3,
            oi_lookback: 2,
        };
        let report = analyze_history(&history, &cfg).unwrap();

        assert!((report.funding_zscore.unwrap() - 1.0).abs() < 1e-6);
        assert!((report.funding_percentile.unwrap() - 1.0).abs() < 1e-6);
        assert!((report.basis_zscore.unwrap() - 1.0).abs() < 1e-5);
        assert!((report.basis_percentile.unwrap() - 1.0).abs() < 1e-6);
        assert!((report.open_interest_change_pct.unwrap() - (100.0 / 11.0)).abs() < 1e-5);
        assert!((report.open_interest_momentum_pct.unwrap() - 20.0).abs() < 1e-6);
        let expected_acceleration = 100.0 / 11.0 - 10.0;
        assert!(
            (report.open_interest_acceleration_pct.unwrap() - expected_acceleration).abs() < 1e-5
        );
        assert!((report.liquidation_total - 30.0).abs() < 1e-6);
        assert!((report.liquidation_zscore.unwrap() - 1.0).abs() < 1e-6);
        assert!((report.liquidation_percentile.unwrap() - 1.0).abs() < 1e-6);
        assert!((report.liquidation_imbalance - (1.0 / 3.0)).abs() < 1e-6);
        assert_eq!(report.price_oi_regime, PriceOiRegime::PriceUpOiUp);
    }

    #[test]
    fn short_history_keeps_unsatisfied_features_explicitly_missing() {
        let history = vec![snapshot(1, 100.0, 0.001, 100.0, 0.0, 0.0)];
        let cfg = DerivativesHistoryConfig {
            window: 3,
            oi_lookback: 2,
        };
        let report = analyze_history(&history, &cfg).unwrap();
        assert!(report.funding_zscore.is_none());
        assert!(report.basis_percentile.is_none());
        assert!(report.open_interest_change_pct.is_none());
        assert_eq!(report.price_oi_regime, PriceOiRegime::Unavailable);
    }
}
