//! Deterministic bounded-grid plan construction.
//!
//! A grid is an allocation and order-lifecycle policy, not a directional
//! closed-bar [`crate::strategy::Strategy`]. This module constructs validated
//! entry/exit order templates using the existing venue-neutral instrument and
//! order primitives. The paper/live execution layer remains responsible for
//! fills, fees, slippage, cancellation, and reconciliation.

use serde::{Deserialize, Serialize};

use crate::orders::{Instrument, Order, OrderType, Side};

/// Lifecycle of one grid level. State transitions are explicit so an execution
/// runtime can persist and replay them deterministically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GridLevelState {
    Idle,
    EntryWorking,
    EntryFilled,
    ExitWorking,
    Complete,
}

/// Input to [`plan_grid`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfig {
    pub symbol: String,
    pub side: Side,
    /// Inclusive lower price bound.
    pub start_price: f32,
    /// Inclusive upper price bound.
    pub end_price: f32,
    /// Number of entry levels, including both bounds.
    pub levels: usize,
    /// Total quote-currency budget distributed equally across levels before
    /// instrument rounding.
    pub total_quote: f32,
    /// Minimum quote amount requested per level.
    pub min_order_quote: f32,
    /// Minimum relative distance between adjacent rounded entry prices.
    /// `0.001` means 0.1%.
    pub min_spread_fraction: f32,
    /// Relative take-profit distance from each filled level.
    pub take_profit_fraction: f32,
    /// Execution throttle carried with the plan. The planner never opens more
    /// orders itself; a runtime must enforce this value.
    pub max_open_orders: usize,
}

/// One grid level with entry and reduce-only exit templates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridLevel {
    pub index: usize,
    pub price: f32,
    pub requested_quote: f32,
    pub rounded_quote: f32,
    pub base_qty: f32,
    pub take_profit_price: f32,
    pub entry_order: Order,
    pub exit_order: Order,
    pub state: GridLevelState,
}

/// Fully validated grid plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridPlan {
    pub symbol: String,
    pub side: Side,
    pub start_price: f32,
    pub end_price: f32,
    pub requested_quote_total: f32,
    pub rounded_quote_total: f32,
    pub max_open_orders: usize,
    pub levels: Vec<GridLevel>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GridPlanError {
    InvalidBounds,
    TooFewLevels,
    InvalidBudget,
    InvalidMinOrderQuote,
    InvalidMinSpread,
    InvalidTakeProfit,
    InvalidMaxOpenOrders,
    RoundedPriceDuplicate {
        index: usize,
    },
    SpreadTooTight {
        index: usize,
    },
    RoundedQuantityZero {
        index: usize,
    },
    BelowMinNotional {
        index: usize,
        rounded_notional: f32,
        min_notional: f32,
    },
}

fn exit_price(entry: f32, side: Side, fraction: f32, instrument: &Instrument) -> f32 {
    let raw = match side {
        Side::Buy => entry * (1.0 + fraction),
        Side::Sell => entry * (1.0 - fraction),
    };
    instrument.round_price(raw)
}

/// Construct an arithmetic grid between the inclusive bounds.
///
/// Every level is retained or the whole plan is rejected. This prevents a venue
/// rule or rounding change from silently altering the intended allocation.
pub fn plan_grid(cfg: &GridConfig, instrument: &Instrument) -> Result<GridPlan, GridPlanError> {
    if !cfg.start_price.is_finite()
        || !cfg.end_price.is_finite()
        || cfg.start_price <= 0.0
        || cfg.end_price <= cfg.start_price
    {
        return Err(GridPlanError::InvalidBounds);
    }
    if cfg.levels < 2 {
        return Err(GridPlanError::TooFewLevels);
    }
    if !cfg.total_quote.is_finite() || cfg.total_quote <= 0.0 {
        return Err(GridPlanError::InvalidBudget);
    }
    if !cfg.min_order_quote.is_finite() || cfg.min_order_quote <= 0.0 {
        return Err(GridPlanError::InvalidMinOrderQuote);
    }
    if !cfg.min_spread_fraction.is_finite() || cfg.min_spread_fraction < 0.0 {
        return Err(GridPlanError::InvalidMinSpread);
    }
    if !cfg.take_profit_fraction.is_finite()
        || cfg.take_profit_fraction <= 0.0
        || cfg.take_profit_fraction >= 1.0
    {
        return Err(GridPlanError::InvalidTakeProfit);
    }
    if cfg.max_open_orders == 0 || cfg.max_open_orders > cfg.levels {
        return Err(GridPlanError::InvalidMaxOpenOrders);
    }

    let requested_quote = cfg.total_quote / cfg.levels as f32;
    if requested_quote < cfg.min_order_quote {
        return Err(GridPlanError::InvalidBudget);
    }

    let step = (cfg.end_price - cfg.start_price) / (cfg.levels - 1) as f32;
    let mut levels = Vec::with_capacity(cfg.levels);
    let mut previous_price: Option<f32> = None;
    let mut rounded_quote_total = 0.0f32;

    for index in 0..cfg.levels {
        let raw_price = cfg.start_price + step * index as f32;
        let price = instrument.round_price(raw_price);
        if !price.is_finite() || price <= 0.0 {
            return Err(GridPlanError::InvalidBounds);
        }

        if let Some(previous) = previous_price {
            if (price - previous).abs() <= f32::EPSILON {
                return Err(GridPlanError::RoundedPriceDuplicate { index });
            }
            let relative = (price - previous).abs() / previous.abs().max(f32::MIN_POSITIVE);
            if relative < cfg.min_spread_fraction {
                return Err(GridPlanError::SpreadTooTight { index });
            }
        }

        let base_qty = instrument.round_qty(requested_quote / price);
        if !base_qty.is_finite() || base_qty <= 0.0 {
            return Err(GridPlanError::RoundedQuantityZero { index });
        }
        let rounded_quote = price * base_qty;
        if !instrument.meets_min_notional(price, base_qty) {
            return Err(GridPlanError::BelowMinNotional {
                index,
                rounded_notional: rounded_quote,
                min_notional: instrument.min_notional,
            });
        }

        let take_profit_price = exit_price(price, cfg.side, cfg.take_profit_fraction, instrument);
        if !take_profit_price.is_finite() || take_profit_price <= 0.0 {
            return Err(GridPlanError::InvalidTakeProfit);
        }

        let entry_id = (index as u64) * 2 + 1;
        let exit_id = entry_id + 1;
        let entry_order = Order::limit(entry_id, &cfg.symbol, cfg.side, base_qty, price).post_only();
        let exit_order = Order::new(
            exit_id,
            &cfg.symbol,
            cfg.side.opposite(),
            OrderType::TakeProfit {
                price: take_profit_price,
            },
            base_qty,
        )
        .reduce_only();

        rounded_quote_total += rounded_quote;
        levels.push(GridLevel {
            index,
            price,
            requested_quote,
            rounded_quote,
            base_qty,
            take_profit_price,
            entry_order,
            exit_order,
            state: GridLevelState::Idle,
        });
        previous_price = Some(price);
    }

    Ok(GridPlan {
        symbol: cfg.symbol.clone(),
        side: cfg.side,
        start_price: levels.first().map(|l| l.price).unwrap_or(cfg.start_price),
        end_price: levels.last().map(|l| l.price).unwrap_or(cfg.end_price),
        requested_quote_total: cfg.total_quote,
        rounded_quote_total,
        max_open_orders: cfg.max_open_orders,
        levels,
    })
}

/// Apply one legal lifecycle transition. Invalid jumps return `false` and leave
/// the state unchanged.
pub fn transition_level(level: &mut GridLevel, next: GridLevelState) -> bool {
    let legal = matches!(
        (level.state, next),
        (GridLevelState::Idle, GridLevelState::EntryWorking)
            | (GridLevelState::EntryWorking, GridLevelState::Idle)
            | (GridLevelState::EntryWorking, GridLevelState::EntryFilled)
            | (GridLevelState::EntryFilled, GridLevelState::ExitWorking)
            | (GridLevelState::ExitWorking, GridLevelState::EntryFilled)
            | (GridLevelState::ExitWorking, GridLevelState::Complete)
            | (GridLevelState::Complete, GridLevelState::Idle)
    );
    if legal {
        level.state = next;
    }
    legal
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instrument() -> Instrument {
        Instrument {
            tick_size: 1.0,
            step_size: 0.01,
            min_notional: 5.0,
        }
    }

    fn config(side: Side) -> GridConfig {
        GridConfig {
            symbol: "BTCUSDT".to_string(),
            side,
            start_price: 90.0,
            end_price: 110.0,
            levels: 3,
            total_quote: 300.0,
            min_order_quote: 5.0,
            min_spread_fraction: 0.05,
            take_profit_fraction: 0.02,
            max_open_orders: 2,
        }
    }

    #[test]
    fn grid_preserves_bounds_budget_and_order_semantics() {
        let plan = plan_grid(&config(Side::Buy), &instrument()).unwrap();
        assert_eq!(plan.levels.len(), 3);
        assert_eq!(plan.max_open_orders, 2);
        assert!((plan.levels[0].price - 90.0).abs() < 1e-6);
        assert!((plan.levels[1].price - 100.0).abs() < 1e-6);
        assert!((plan.levels[2].price - 110.0).abs() < 1e-6);
        assert!(plan.rounded_quote_total <= plan.requested_quote_total);
        for level in &plan.levels {
            assert!(level.entry_order.post_only);
            assert!(level.exit_order.reduce_only);
            assert_eq!(level.state, GridLevelState::Idle);
            assert!(level.take_profit_price > level.price);
        }
    }

    #[test]
    fn sell_grid_take_profit_is_below_entry() {
        let plan = plan_grid(&config(Side::Sell), &instrument()).unwrap();
        assert!(plan.levels.iter().all(|level| level.take_profit_price < level.price));
    }

    #[test]
    fn too_tight_grid_after_rounding_is_rejected() {
        let mut cfg = config(Side::Buy);
        cfg.start_price = 100.0;
        cfg.end_price = 101.0;
        cfg.min_spread_fraction = 0.01;
        assert!(matches!(
            plan_grid(&cfg, &instrument()),
            Err(GridPlanError::RoundedPriceDuplicate { .. }) | Err(GridPlanError::SpreadTooTight { .. })
        ));
    }

    #[test]
    fn invalid_budget_and_throttle_are_rejected() {
        let mut cfg = config(Side::Buy);
        cfg.total_quote = 10.0;
        assert_eq!(plan_grid(&cfg, &instrument()).unwrap_err(), GridPlanError::InvalidBudget);

        let mut cfg = config(Side::Buy);
        cfg.max_open_orders = 4;
        assert_eq!(
            plan_grid(&cfg, &instrument()).unwrap_err(),
            GridPlanError::InvalidMaxOpenOrders
        );
    }

    #[test]
    fn lifecycle_only_accepts_declared_transitions() {
        let mut level = plan_grid(&config(Side::Buy), &instrument()).unwrap().levels.remove(0);
        assert!(transition_level(&mut level, GridLevelState::EntryWorking));
        assert!(transition_level(&mut level, GridLevelState::EntryFilled));
        assert!(!transition_level(&mut level, GridLevelState::Complete));
        assert!(transition_level(&mut level, GridLevelState::ExitWorking));
        assert!(transition_level(&mut level, GridLevelState::Complete));
        assert!(transition_level(&mut level, GridLevelState::Idle));
    }
}
