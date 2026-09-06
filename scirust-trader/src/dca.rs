//! Deterministic dollar-cost-averaging plan construction.
//!
//! DCA is an allocation/execution policy rather than a closed-bar directional
//! [`crate::strategy::Strategy`]. This module therefore builds explicit order
//! levels on top of [`crate::orders`] without introducing a second matching
//! engine. The existing paper execution layer remains authoritative for fills,
//! fees, slippage, and order lifecycle semantics.
//!
//! A plan is descriptive: it does not claim an edge or profitability. Every
//! price, quote allocation, rounded base quantity, and risk reference is exposed
//! so a caller can simulate and audit the plan before use.

use serde::{Deserialize, Serialize};

use crate::orders::{Instrument, Order, OrderType, Side};

/// Liquidity intent for DCA child orders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DcaMode {
    /// Rest each child as a post-only GTC limit order at its configured level.
    Maker,
    /// Submit a market order when the configured trigger level is activated.
    Taker,
}

/// Input to [`plan_dca`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DcaConfig {
    pub symbol: String,
    pub side: Side,
    /// Quote-currency notional allocated to each level.
    pub quote_amounts: Vec<f32>,
    /// Trigger/reference price for each level. For `Maker`, this is also the
    /// limit price. For `Taker`, it is only the price used to size the planned
    /// base quantity; the eventual market fill remains determined by the paper
    /// or venue execution engine.
    pub prices: Vec<f32>,
    pub mode: DcaMode,
    /// Fractional distance from the weighted entry, e.g. `0.02` = 2%.
    pub take_profit_pct: Option<f32>,
    /// Fractional distance from the weighted entry, e.g. `0.01` = 1%.
    pub stop_loss_pct: Option<f32>,
}

/// One validated DCA level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DcaLevel {
    pub index: usize,
    pub trigger_price: f32,
    pub requested_quote: f32,
    /// Quote notional after tick/lot rounding.
    pub rounded_quote: f32,
    pub base_qty: f32,
    /// Order template. Timestamps and real fill prices are intentionally left to
    /// the execution layer.
    pub order: Order,
}

/// Deterministic DCA plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DcaPlan {
    pub symbol: String,
    pub side: Side,
    pub mode: DcaMode,
    pub levels: Vec<DcaLevel>,
    pub requested_quote_total: f32,
    pub rounded_quote_total: f32,
    pub total_base_qty: f32,
    /// Quantity-weighted entry reference after instrument rounding.
    pub weighted_entry_price: f32,
    /// Reference only; no exit order is submitted by this planner.
    pub take_profit_price: Option<f32>,
    /// Reference only; no exit order is submitted by this planner.
    pub stop_loss_price: Option<f32>,
}

/// Validation failure while constructing a DCA plan.
#[derive(Debug, Clone, PartialEq)]
pub enum DcaPlanError {
    Empty,
    LengthMismatch {
        quote_amounts: usize,
        prices: usize,
    },
    InvalidPrice {
        index: usize,
    },
    InvalidQuoteAmount {
        index: usize,
    },
    InvalidTakeProfit,
    InvalidStopLoss,
    RoundedQuantityZero {
        index: usize,
    },
    BelowMinNotional {
        index: usize,
        rounded_notional: f32,
        min_notional: f32,
    },
}

fn valid_fraction(value: Option<f32>) -> bool {
    value.map(|v| v.is_finite() && v >= 0.0).unwrap_or(true)
}

fn exit_reference(
    weighted_entry: f32,
    side: Side,
    fraction: Option<f32>,
    profit: bool,
) -> Option<f32> {
    let fraction = fraction?;
    let direction = match (side, profit)
    {
        (Side::Buy, true) | (Side::Sell, false) => 1.0,
        (Side::Buy, false) | (Side::Sell, true) => -1.0,
    };
    Some(weighted_entry * (1.0 + direction * fraction))
}

/// Build a validated DCA plan using the venue-neutral [`Instrument`] rules.
///
/// The planner rejects levels that become zero-sized or fall below the minimum
/// notional after price/quantity rounding. It never silently drops a level,
/// because doing so would change the user's allocation schedule.
pub fn plan_dca(cfg: &DcaConfig, instrument: &Instrument) -> Result<DcaPlan, DcaPlanError> {
    if cfg.prices.is_empty() || cfg.quote_amounts.is_empty()
    {
        return Err(DcaPlanError::Empty);
    }
    if cfg.prices.len() != cfg.quote_amounts.len()
    {
        return Err(DcaPlanError::LengthMismatch {
            quote_amounts: cfg.quote_amounts.len(),
            prices: cfg.prices.len(),
        });
    }
    if !valid_fraction(cfg.take_profit_pct)
    {
        return Err(DcaPlanError::InvalidTakeProfit);
    }
    if !valid_fraction(cfg.stop_loss_pct)
    {
        return Err(DcaPlanError::InvalidStopLoss);
    }

    let mut levels = Vec::with_capacity(cfg.prices.len());
    let mut requested_quote_total = 0.0f32;
    let mut rounded_quote_total = 0.0f32;
    let mut total_base_qty = 0.0f32;
    let mut weighted_notional = 0.0f32;

    for (index, (&raw_price, &requested_quote)) in
        cfg.prices.iter().zip(cfg.quote_amounts.iter()).enumerate()
    {
        if !raw_price.is_finite() || raw_price <= 0.0
        {
            return Err(DcaPlanError::InvalidPrice { index });
        }
        if !requested_quote.is_finite() || requested_quote <= 0.0
        {
            return Err(DcaPlanError::InvalidQuoteAmount { index });
        }

        let trigger_price = instrument.round_price(raw_price);
        if !trigger_price.is_finite() || trigger_price <= 0.0
        {
            return Err(DcaPlanError::InvalidPrice { index });
        }
        let base_qty = instrument.round_qty(requested_quote / trigger_price);
        if !base_qty.is_finite() || base_qty <= 0.0
        {
            return Err(DcaPlanError::RoundedQuantityZero { index });
        }
        let rounded_quote = trigger_price * base_qty;
        if !instrument.meets_min_notional(trigger_price, base_qty)
        {
            return Err(DcaPlanError::BelowMinNotional {
                index,
                rounded_notional: rounded_quote,
                min_notional: instrument.min_notional,
            });
        }

        let id = index as u64 + 1;
        let order = match cfg.mode
        {
            DcaMode::Maker =>
            {
                Order::limit(id, &cfg.symbol, cfg.side, base_qty, trigger_price).post_only()
            },
            DcaMode::Taker => Order::market(id, &cfg.symbol, cfg.side, base_qty),
        };

        requested_quote_total += requested_quote;
        rounded_quote_total += rounded_quote;
        total_base_qty += base_qty;
        weighted_notional += trigger_price * base_qty;
        levels.push(DcaLevel {
            index,
            trigger_price,
            requested_quote,
            rounded_quote,
            base_qty,
            order,
        });
    }

    let weighted_entry_price = if total_base_qty > 0.0
    {
        weighted_notional / total_base_qty
    }
    else
    {
        0.0
    };

    Ok(DcaPlan {
        symbol: cfg.symbol.clone(),
        side: cfg.side,
        mode: cfg.mode,
        levels,
        requested_quote_total,
        rounded_quote_total,
        total_base_qty,
        weighted_entry_price,
        take_profit_price: exit_reference(
            weighted_entry_price,
            cfg.side,
            cfg.take_profit_pct,
            true,
        ),
        stop_loss_price: exit_reference(weighted_entry_price, cfg.side, cfg.stop_loss_pct, false),
    })
}

/// True when a DCA level's order template is a resting maker order.
pub fn is_maker_level(level: &DcaLevel) -> bool {
    matches!(level.order.order_type, OrderType::Limit { .. }) && level.order.post_only
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instrument() -> Instrument {
        Instrument {
            tick_size: 0.5,
            step_size: 0.01,
            min_notional: 5.0,
        }
    }

    fn config(side: Side, mode: DcaMode) -> DcaConfig {
        DcaConfig {
            symbol: "BTCUSDT".to_string(),
            side,
            quote_amounts: vec![100.0, 200.0],
            prices: vec![100.2, 90.2],
            mode,
            take_profit_pct: Some(0.10),
            stop_loss_pct: Some(0.05),
        }
    }

    #[test]
    fn maker_plan_rounds_and_preserves_explicit_budget_accounting() {
        let plan = plan_dca(&config(Side::Buy, DcaMode::Maker), &instrument()).unwrap();
        assert_eq!(plan.levels.len(), 2);
        assert!(plan.levels.iter().all(is_maker_level));
        assert!((plan.levels[0].trigger_price - 100.0).abs() < 1e-6);
        assert!((plan.levels[0].base_qty - 1.0).abs() < 1e-6);
        assert!((plan.levels[1].trigger_price - 90.0).abs() < 1e-6);
        assert!((plan.levels[1].base_qty - 2.22).abs() < 1e-6);
        assert!((plan.requested_quote_total - 300.0).abs() < 1e-6);
        assert!(plan.rounded_quote_total <= plan.requested_quote_total);
    }

    #[test]
    fn buy_exit_references_have_expected_direction() {
        let plan = plan_dca(&config(Side::Buy, DcaMode::Taker), &instrument()).unwrap();
        let tp = plan.take_profit_price.unwrap();
        let sl = plan.stop_loss_price.unwrap();
        assert!(tp > plan.weighted_entry_price);
        assert!(sl < plan.weighted_entry_price);
        assert!(matches!(plan.levels[0].order.order_type, OrderType::Market));
    }

    #[test]
    fn sell_exit_references_reverse_direction() {
        let plan = plan_dca(&config(Side::Sell, DcaMode::Maker), &instrument()).unwrap();
        let tp = plan.take_profit_price.unwrap();
        let sl = plan.stop_loss_price.unwrap();
        assert!(tp < plan.weighted_entry_price);
        assert!(sl > plan.weighted_entry_price);
    }

    #[test]
    fn invalid_or_unequal_inputs_are_rejected() {
        let mut cfg = config(Side::Buy, DcaMode::Maker);
        cfg.prices.pop();
        assert!(matches!(
            plan_dca(&cfg, &instrument()),
            Err(DcaPlanError::LengthMismatch { .. })
        ));

        let mut cfg = config(Side::Buy, DcaMode::Maker);
        cfg.prices[0] = 0.0;
        assert_eq!(
            plan_dca(&cfg, &instrument()).unwrap_err(),
            DcaPlanError::InvalidPrice { index: 0 }
        );
    }

    #[test]
    fn min_notional_is_checked_after_rounding() {
        let cfg = DcaConfig {
            symbol: "BTCUSDT".to_string(),
            side: Side::Buy,
            quote_amounts: vec![4.99],
            prices: vec![100.0],
            mode: DcaMode::Maker,
            take_profit_pct: None,
            stop_loss_pct: None,
        };
        assert!(matches!(
            plan_dca(&cfg, &instrument()),
            Err(DcaPlanError::BelowMinNotional { index: 0, .. })
        ));
    }

    #[test]
    fn invalid_risk_distances_are_not_silently_clamped() {
        let mut cfg = config(Side::Buy, DcaMode::Maker);
        cfg.take_profit_pct = Some(-0.01);
        assert_eq!(
            plan_dca(&cfg, &instrument()).unwrap_err(),
            DcaPlanError::InvalidTakeProfit
        );
    }
}
