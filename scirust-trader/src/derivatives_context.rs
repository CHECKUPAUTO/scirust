//! Cross-metric context for crypto derivatives histories.
//!
//! This module deliberately reports *relationships* between observed market
//! variables without converting them into trading signals. It complements
//! [`crate::derivatives_history`] with one-step change vectors, descriptive
//! divergence flags, and explicit liquidation-cluster detection.

use serde::{Deserialize, Serialize};

use crate::derivatives::{DerivativesSnapshot, basis_bps, open_interest_change};
use crate::derivatives_history::{latest_percentile, latest_zscore, liquidation_total};

/// One-step changes between two aligned derivatives observations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DerivativesChange {
    pub mark_return_pct: Option<f32>,
    pub funding_change_bps: Option<f32>,
    pub basis_change_bps: Option<f32>,
    pub open_interest_change_pct: Option<f32>,
}

/// Descriptive sign disagreements between price and other derivatives metrics.
///
/// A flag means only that the latest changes have opposite signs. It is not an
/// entry/exit recommendation and carries no assumed predictive direction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DivergenceFlags {
    pub price_vs_funding_opposed: Option<bool>,
    pub price_vs_basis_opposed: Option<bool>,
    pub price_vs_open_interest_opposed: Option<bool>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct LiquidationClusterConfig {
    pub window: usize,
    pub z_threshold: f32,
    pub percentile_threshold: f32,
}

impl Default for LiquidationClusterConfig {
    fn default() -> Self {
        Self {
            window: 20,
            z_threshold: 2.0,
            percentile_threshold: 0.95,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LiquidationClusterReport {
    pub current_total: f32,
    pub zscore: f32,
    pub percentile: f32,
    pub is_cluster: bool,
}

fn pct_change(previous: f32, current: f32) -> Option<f32> {
    if !previous.is_finite()
        || !current.is_finite()
        || previous.abs() <= f32::EPSILON
    {
        return None;
    }
    Some((current - previous) / previous * 100.0)
}

fn opposed(a: Option<f32>, b: Option<f32>) -> Option<bool> {
    let a = a?;
    let b = b?;
    if !a.is_finite() || !b.is_finite()
    {
        return None;
    }
    if a.abs() <= f32::EPSILON || b.abs() <= f32::EPSILON
    {
        return Some(false);
    }
    Some(a.signum() != b.signum())
}

/// Compute one-step mark, funding, basis and open-interest changes.
pub fn changes(
    previous: &DerivativesSnapshot,
    current: &DerivativesSnapshot,
) -> DerivativesChange {
    let previous_basis = basis_bps(previous.mark_price, previous.spot_price);
    let current_basis = basis_bps(current.mark_price, current.spot_price);
    let basis_change_bps = match (previous_basis, current_basis)
    {
        (Some(a), Some(b)) => Some(b - a),
        _ => None,
    };
    let funding_change_bps = if previous.funding_rate.is_finite() && current.funding_rate.is_finite()
    {
        Some((current.funding_rate - previous.funding_rate) * 10_000.0)
    }
    else
    {
        None
    };

    DerivativesChange {
        mark_return_pct: pct_change(previous.mark_price, current.mark_price),
        funding_change_bps,
        basis_change_bps,
        open_interest_change_pct: open_interest_change(
            previous.open_interest,
            current.open_interest,
        )
        .map(|c| c.percent),
    }
}

/// Compare the signs of price change with funding, basis, and OI changes.
pub fn divergence_flags(change: &DerivativesChange) -> DivergenceFlags {
    DivergenceFlags {
        price_vs_funding_opposed: opposed(change.mark_return_pct, change.funding_change_bps),
        price_vs_basis_opposed: opposed(change.mark_return_pct, change.basis_change_bps),
        price_vs_open_interest_opposed: opposed(
            change.mark_return_pct,
            change.open_interest_change_pct,
        ),
    }
}

/// Detect an unusually large liquidation observation relative to its own
/// trailing history.
///
/// Both the z-score and empirical-percentile gates must pass. This makes the
/// threshold explicit and avoids labeling a moderate observation a cluster from
/// only one normalization convention.
pub fn detect_liquidation_cluster(
    history: &[DerivativesSnapshot],
    cfg: &LiquidationClusterConfig,
) -> Option<LiquidationClusterReport> {
    if cfg.window < 2
        || !cfg.z_threshold.is_finite()
        || cfg.z_threshold < 0.0
        || !cfg.percentile_threshold.is_finite()
        || !(0.0..=1.0).contains(&cfg.percentile_threshold)
    {
        return None;
    }
    let totals: Vec<f32> = history.iter().map(liquidation_total).collect();
    let zscore = latest_zscore(&totals, cfg.window)?;
    let percentile = latest_percentile(&totals, cfg.window)?;
    let current_total = *totals.last()?;
    Some(LiquidationClusterReport {
        current_total,
        zscore,
        percentile,
        is_cluster: zscore >= cfg.z_threshold && percentile >= cfg.percentile_threshold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(
        ts_ms: i64,
        spot: f32,
        mark: f32,
        funding: f32,
        oi: f32,
        long_liq: f32,
        short_liq: f32,
    ) -> DerivativesSnapshot {
        DerivativesSnapshot {
            symbol: "BTCUSDT".to_string(),
            ts_ms,
            spot_price: spot,
            mark_price: mark,
            index_price: spot,
            funding_rate: funding,
            funding_interval_hours: 8.0,
            open_interest: oi,
            long_liquidations: long_liq,
            short_liquidations: short_liq,
        }
    }

    #[test]
    fn change_vector_uses_explicit_units() {
        let a = snapshot(1, 100.0, 101.0, 0.001, 100.0, 0.0, 0.0);
        let b = snapshot(2, 100.0, 102.0, 0.0005, 90.0, 0.0, 0.0);
        let c = changes(&a, &b);
        assert!((c.mark_return_pct.unwrap() - (100.0 / 101.0)).abs() < 1e-5);
        assert!((c.funding_change_bps.unwrap() + 5.0).abs() < 1e-6);
        assert!((c.basis_change_bps.unwrap() - 100.0).abs() < 1e-4);
        assert!((c.open_interest_change_pct.unwrap() + 10.0).abs() < 1e-6);
    }

    #[test]
    fn divergence_flags_are_descriptive_only() {
        let c = DerivativesChange {
            mark_return_pct: Some(1.0),
            funding_change_bps: Some(-2.0),
            basis_change_bps: Some(3.0),
            open_interest_change_pct: Some(-4.0),
        };
        let flags = divergence_flags(&c);
        assert_eq!(flags.price_vs_funding_opposed, Some(true));
        assert_eq!(flags.price_vs_basis_opposed, Some(false));
        assert_eq!(flags.price_vs_open_interest_opposed, Some(true));
    }

    #[test]
    fn liquidation_cluster_requires_both_gates() {
        let history = vec![
            snapshot(1, 100.0, 100.0, 0.0, 100.0, 5.0, 5.0),
            snapshot(2, 100.0, 100.0, 0.0, 100.0, 5.0, 5.0),
            snapshot(3, 100.0, 100.0, 0.0, 100.0, 5.0, 5.0),
            snapshot(4, 100.0, 100.0, 0.0, 100.0, 50.0, 50.0),
        ];
        let cfg = LiquidationClusterConfig {
            window: 4,
            z_threshold: 1.0,
            percentile_threshold: 1.0,
        };
        let report = detect_liquidation_cluster(&history, &cfg).unwrap();
        assert_eq!(report.current_total, 100.0);
        assert_eq!(report.percentile, 1.0);
        assert!(report.zscore > 1.0);
        assert!(report.is_cluster);
    }

    #[test]
    fn repeated_analysis_is_bit_reproducible() {
        let history = vec![
            snapshot(1, 100.0, 100.0, 0.0, 100.0, 1.0, 2.0),
            snapshot(2, 100.0, 101.0, 0.001, 105.0, 3.0, 4.0),
            snapshot(3, 100.0, 102.0, 0.002, 110.0, 5.0, 6.0),
        ];
        let cfg = LiquidationClusterConfig {
            window: 3,
            z_threshold: 1.0,
            percentile_threshold: 0.9,
        };
        let first = detect_liquidation_cluster(&history, &cfg);
        let second = detect_liquidation_cluster(&history, &cfg);
        assert_eq!(first, second);

        let c1 = changes(&history[1], &history[2]);
        let c2 = changes(&history[1], &history[2]);
        assert_eq!(c1, c2);
        assert_eq!(divergence_flags(&c1), divergence_flags(&c2));
    }
}
