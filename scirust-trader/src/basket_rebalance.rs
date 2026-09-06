//! Cost-aware basket/index rebalancing.
//!
//! The existing portfolio layer computes the quantity changes required to reach
//! target weights. This module adds the execution-planning constraints needed
//! for a basket or index rebalance: explicit per-asset costs, a drift band,
//! minimum trade notionals and a deterministic cap on one-way turnover.
//!
//! This is a planner only. It does not submit orders and it does not claim that
//! the cost assumptions will be realized by a live venue.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::orders::Side;
use crate::portfolio::{Account, rebalance_to_weights};

/// Per-asset trading-cost assumptions used by the planner.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RebalanceCostAssumption {
    /// Fee rate as a fraction of notional. Negative values represent rebates.
    pub fee_rate: f32,
    /// Expected adverse execution slippage in basis points.
    pub slippage_bps: f32,
    /// Fixed quote-currency cost charged when a trade is retained in the plan.
    pub fixed_cost: f32,
}

impl Default for RebalanceCostAssumption {
    fn default() -> Self {
        Self {
            fee_rate: 0.0,
            slippage_bps: 0.0,
            fixed_cost: 0.0,
        }
    }
}

/// Portfolio-level planning constraints.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BasketRebalanceConfig {
    /// Ignore assets whose absolute weight drift is smaller than this fraction.
    pub drift_band: f32,
    /// Maximum one-way turnover as a fraction of current marked equity. A value
    /// of `0.25` permits planned notional equal to at most 25% of equity.
    pub max_turnover_fraction: f32,
    /// Drop post-cap trades below this quote-currency notional.
    pub min_trade_notional: f32,
}

/// One retained trade in the rebalance plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasketRebalanceTrade {
    pub symbol: String,
    pub side: Side,
    pub price: f32,
    pub current_qty: f32,
    /// Quantity implied by the unconstrained target weight.
    pub unconstrained_target_qty: f32,
    /// Quantity actually planned after the portfolio turnover cap is applied.
    pub planned_qty: f32,
    pub planned_notional: f32,
    pub target_weight: f32,
    pub fee_rate: f32,
    pub slippage_bps: f32,
    pub estimated_fee: f32,
    pub estimated_slippage_cost: f32,
    pub fixed_cost: f32,
    pub estimated_total_cost: f32,
}

/// Deterministic basket rebalance plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasketRebalancePlan {
    pub equity: f32,
    pub target_weight_sum: f32,
    /// One-way turnover before applying the cap, as a fraction of equity.
    pub requested_turnover_fraction: f32,
    /// Proportional scale applied to every candidate trade before the minimum
    /// notional filter. `1` means the cap did not bind.
    pub turnover_scale: f32,
    /// One-way turnover of retained planned trades, as a fraction of equity.
    pub planned_turnover_fraction: f32,
    pub turnover_cap_binding: bool,
    pub estimated_total_cost: f32,
    pub skipped_small_trades: usize,
    pub trades: Vec<BasketRebalanceTrade>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BasketRebalanceError {
    InvalidConfig,
    InvalidTargetWeights,
    InvalidMark(String),
    MissingCostAssumption(String),
    InvalidCostAssumption(String),
    NonPositiveEquity,
}

fn valid_cost(c: RebalanceCostAssumption) -> bool {
    c.fee_rate.is_finite()
        && c.fee_rate > -1.0
        && c.slippage_bps.is_finite()
        && c.slippage_bps >= 0.0
        && c.fixed_cost.is_finite()
        && c.fixed_cost >= 0.0
}

fn validate_config(cfg: BasketRebalanceConfig) -> bool {
    cfg.drift_band.is_finite()
        && cfg.drift_band >= 0.0
        && cfg.max_turnover_fraction.is_finite()
        && cfg.max_turnover_fraction >= 0.0
        && cfg.min_trade_notional.is_finite()
        && cfg.min_trade_notional >= 0.0
}

/// Build a cost-aware, turnover-capped rebalance plan.
///
/// Assets that should be exited must appear explicitly in `target_weights` with
/// weight `0`. Every target symbol must have a positive mark and an explicit
/// cost assumption so that the planner never silently assumes free execution.
pub fn plan_basket_rebalance(
    account: &Account,
    target_weights: &BTreeMap<String, f32>,
    marks: &BTreeMap<String, f32>,
    costs: &BTreeMap<String, RebalanceCostAssumption>,
    cfg: BasketRebalanceConfig,
) -> Result<BasketRebalancePlan, BasketRebalanceError> {
    if !validate_config(cfg) {
        return Err(BasketRebalanceError::InvalidConfig);
    }

    let mut target_weight_sum = 0.0f32;
    for (symbol, weight) in target_weights {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(BasketRebalanceError::InvalidTargetWeights);
        }
        target_weight_sum += *weight;
        let mark = marks
            .get(symbol)
            .copied()
            .ok_or_else(|| BasketRebalanceError::InvalidMark(symbol.clone()))?;
        if !mark.is_finite() || mark <= 0.0 {
            return Err(BasketRebalanceError::InvalidMark(symbol.clone()));
        }
        let cost = costs
            .get(symbol)
            .copied()
            .ok_or_else(|| BasketRebalanceError::MissingCostAssumption(symbol.clone()))?;
        if !valid_cost(cost) {
            return Err(BasketRebalanceError::InvalidCostAssumption(symbol.clone()));
        }
    }
    if !target_weight_sum.is_finite() || target_weight_sum > 1.0 + 1e-6 {
        return Err(BasketRebalanceError::InvalidTargetWeights);
    }

    let equity = account.equity(marks);
    if !equity.is_finite() || equity <= 0.0 {
        return Err(BasketRebalanceError::NonPositiveEquity);
    }

    let candidates = rebalance_to_weights(account, target_weights, marks, cfg.drift_band);
    let requested_notional: f32 = candidates
        .iter()
        .map(|trade| trade.qty * marks.get(&trade.symbol).copied().unwrap_or(0.0))
        .sum();
    let requested_turnover_fraction = requested_notional / equity;
    let allowed_notional = cfg.max_turnover_fraction * equity;
    let turnover_scale = if requested_notional <= 1e-12 {
        1.0
    } else {
        (allowed_notional / requested_notional).clamp(0.0, 1.0)
    };
    let turnover_cap_binding = turnover_scale < 1.0 - 1e-7;

    let mut trades = Vec::new();
    let mut planned_notional_total = 0.0f32;
    let mut estimated_total_cost = 0.0f32;
    let mut skipped_small_trades = 0usize;

    for candidate in candidates {
        let price = marks[&candidate.symbol];
        let planned_qty = candidate.qty * turnover_scale;
        let planned_notional = planned_qty * price;
        if planned_notional + 1e-9 < cfg.min_trade_notional || planned_qty <= 1e-12 {
            skipped_small_trades += 1;
            continue;
        }

        let cost = costs[&candidate.symbol];
        let estimated_fee = planned_notional * cost.fee_rate;
        let estimated_slippage_cost = planned_notional * cost.slippage_bps / 10_000.0;
        let estimated_trade_cost = estimated_fee + estimated_slippage_cost + cost.fixed_cost;

        planned_notional_total += planned_notional;
        estimated_total_cost += estimated_trade_cost;
        trades.push(BasketRebalanceTrade {
            symbol: candidate.symbol.clone(),
            side: candidate.side,
            price,
            current_qty: candidate.current_qty,
            unconstrained_target_qty: candidate.target_qty,
            planned_qty,
            planned_notional,
            target_weight: target_weights[&candidate.symbol],
            fee_rate: cost.fee_rate,
            slippage_bps: cost.slippage_bps,
            estimated_fee,
            estimated_slippage_cost,
            fixed_cost: cost.fixed_cost,
            estimated_total_cost: estimated_trade_cost,
        });
    }

    Ok(BasketRebalancePlan {
        equity,
        target_weight_sum,
        requested_turnover_fraction,
        turnover_scale,
        planned_turnover_fraction: planned_notional_total / equity,
        turnover_cap_binding,
        estimated_total_cost,
        skipped_small_trades,
        trades,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orders::Fill;

    fn fill(price: f32, qty: f32) -> Fill {
        Fill {
            price,
            qty,
            fee: 0.0,
            taker: true,
            ts_ms: 0,
        }
    }

    fn base_inputs() -> (
        Account,
        BTreeMap<String, f32>,
        BTreeMap<String, f32>,
        BTreeMap<String, RebalanceCostAssumption>,
    ) {
        let account = Account::new(10_000.0);
        let mut targets = BTreeMap::new();
        targets.insert("BTC".to_string(), 0.5);
        targets.insert("ETH".to_string(), 0.3);

        let mut marks = BTreeMap::new();
        marks.insert("BTC".to_string(), 100.0);
        marks.insert("ETH".to_string(), 50.0);

        let mut costs = BTreeMap::new();
        costs.insert(
            "BTC".to_string(),
            RebalanceCostAssumption {
                fee_rate: 0.001,
                slippage_bps: 5.0,
                fixed_cost: 0.25,
            },
        );
        costs.insert(
            "ETH".to_string(),
            RebalanceCostAssumption {
                fee_rate: 0.001,
                slippage_bps: 8.0,
                fixed_cost: 0.10,
            },
        );
        (account, targets, marks, costs)
    }

    fn cfg(max_turnover_fraction: f32) -> BasketRebalanceConfig {
        BasketRebalanceConfig {
            drift_band: 0.0,
            max_turnover_fraction,
            min_trade_notional: 0.0,
        }
    }

    #[test]
    fn uncapped_plan_reaches_requested_notional() {
        let (account, targets, marks, costs) = base_inputs();
        let plan = plan_basket_rebalance(&account, &targets, &marks, &costs, cfg(1.0)).unwrap();
        assert_eq!(plan.trades.len(), 2);
        assert!(!plan.turnover_cap_binding);
        assert!((plan.requested_turnover_fraction - 0.8).abs() < 1e-5);
        assert!((plan.planned_turnover_fraction - 0.8).abs() < 1e-5);
        assert!(plan.estimated_total_cost > 0.0);
    }

    #[test]
    fn turnover_cap_scales_all_trades_proportionally() {
        let (account, targets, marks, costs) = base_inputs();
        let plan = plan_basket_rebalance(&account, &targets, &marks, &costs, cfg(0.2)).unwrap();
        assert!(plan.turnover_cap_binding);
        assert!((plan.turnover_scale - 0.25).abs() < 1e-5);
        assert!((plan.planned_turnover_fraction - 0.2).abs() < 1e-5);
        assert!((plan.trades[0].planned_notional + plan.trades[1].planned_notional - 2_000.0).abs() < 1e-3);
    }

    #[test]
    fn minimum_notional_drops_small_post_cap_trades() {
        let (account, targets, marks, costs) = base_inputs();
        let mut c = cfg(0.01);
        c.min_trade_notional = 40.0;
        let plan = plan_basket_rebalance(&account, &targets, &marks, &costs, c).unwrap();
        assert_eq!(plan.trades.len(), 1);
        assert_eq!(plan.skipped_small_trades, 1);
        assert_eq!(plan.trades[0].symbol, "BTC");
    }

    #[test]
    fn drift_band_reuses_existing_portfolio_semantics() {
        let (mut account, targets, marks, costs) = base_inputs();
        account.apply_fill("BTC", Side::Buy, &fill(100.0, 49.5));
        account.apply_fill("ETH", Side::Buy, &fill(50.0, 60.0));
        let c = BasketRebalanceConfig {
            drift_band: 0.01,
            max_turnover_fraction: 1.0,
            min_trade_notional: 0.0,
        };
        let plan = plan_basket_rebalance(&account, &targets, &marks, &costs, c).unwrap();
        assert!(plan.trades.is_empty());
    }

    #[test]
    fn every_target_requires_explicit_cost_assumption() {
        let (account, targets, marks, mut costs) = base_inputs();
        costs.remove("ETH");
        assert!(matches!(
            plan_basket_rebalance(&account, &targets, &marks, &costs, cfg(1.0)),
            Err(BasketRebalanceError::MissingCostAssumption(symbol)) if symbol == "ETH"
        ));
    }

    #[test]
    fn invalid_target_sum_is_rejected() {
        let (account, mut targets, marks, costs) = base_inputs();
        targets.insert("BTC".to_string(), 0.8);
        targets.insert("ETH".to_string(), 0.7);
        assert!(matches!(
            plan_basket_rebalance(&account, &targets, &marks, &costs, cfg(1.0)),
            Err(BasketRebalanceError::InvalidTargetWeights)
        ));
    }

    #[test]
    fn repeated_plan_is_deterministic() {
        let (account, targets, marks, costs) = base_inputs();
        let a = plan_basket_rebalance(&account, &targets, &marks, &costs, cfg(0.2)).unwrap();
        let b = plan_basket_rebalance(&account, &targets, &marks, &costs, cfg(0.2)).unwrap();
        assert_eq!(a.trades.len(), b.trades.len());
        assert_eq!(a.turnover_scale.to_bits(), b.turnover_scale.to_bits());
        assert_eq!(
            a.estimated_total_cost.to_bits(),
            b.estimated_total_cost.to_bits()
        );
    }
}
