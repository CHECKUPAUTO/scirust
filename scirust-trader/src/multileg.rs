//! Multi-leg execution state and recovery semantics.
//!
//! Cross-venue arbitrage, basis and funding-carry strategies can be exposed when
//! one leg fills before another. This module makes that state explicit. It does
//! not choose a universal recovery policy: callers can request either completion
//! of outstanding target quantities or neutralization of quantities already
//! filled. Both plans are deterministic and remain separate from venue I/O.

use serde::{Deserialize, Serialize};

use crate::orders::Side;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LegIntent {
    pub leg_id: String,
    pub venue: String,
    pub symbol: String,
    pub side: Side,
    pub target_qty: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LegProgress {
    pub intent: LegIntent,
    pub filled_qty: f32,
    pub avg_fill_price: f32,
    pub fees_paid: f32,
}

impl LegProgress {
    pub fn remaining_qty(&self) -> f32 {
        (self.intent.target_qty - self.filled_qty).max(0.0)
    }

    pub fn fill_fraction(&self) -> f32 {
        if self.intent.target_qty <= 1e-12
        {
            0.0
        }
        else
        {
            (self.filled_qty / self.intent.target_qty).clamp(0.0, 1.0)
        }
    }

    pub fn complete(&self) -> bool {
        self.remaining_qty() <= 1e-9
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MultiLegStatus {
    /// No leg has filled yet.
    Unfilled,
    /// Some quantity has filled but all legs are at the same completion ratio
    /// within the configured tolerance.
    BalancedPartial,
    /// Leg completion ratios differ materially, creating explicit leg risk.
    Imbalanced,
    /// Every leg has reached its declared target quantity.
    Complete,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MultiLegState {
    pub strategy_id: String,
    pub legs: Vec<LegProgress>,
    /// Maximum permitted difference between any two leg fill fractions before
    /// the state is labelled imbalanced.
    pub fill_fraction_tolerance: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecoveryMode {
    /// Continue in each original direction until every target is complete.
    CompleteOutstanding,
    /// Trade opposite each already-filled quantity so every leg returns flat.
    NeutralizeFilled,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecoveryAction {
    pub leg_id: String,
    pub venue: String,
    pub symbol: String,
    pub side: Side,
    pub qty: f32,
    pub mode: RecoveryMode,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RecoveryPlan {
    pub strategy_id: String,
    pub observed_status: MultiLegStatus,
    pub max_fill_fraction_gap: f32,
    pub mode: RecoveryMode,
    pub actions: Vec<RecoveryAction>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MultiLegError {
    EmptyStrategy,
    InvalidTolerance,
    InvalidLeg { leg_id: String },
    DuplicateLegId { leg_id: String },
    UnknownLeg { leg_id: String },
    InvalidFill { leg_id: String },
}

impl MultiLegState {
    pub fn new(
        strategy_id: &str,
        intents: Vec<LegIntent>,
        fill_fraction_tolerance: f32,
    ) -> Result<Self, MultiLegError> {
        if strategy_id.is_empty() || intents.is_empty()
        {
            return Err(MultiLegError::EmptyStrategy);
        }
        if !fill_fraction_tolerance.is_finite() || !(0.0..=1.0).contains(&fill_fraction_tolerance)
        {
            return Err(MultiLegError::InvalidTolerance);
        }

        let mut seen = std::collections::BTreeSet::new();
        let mut legs = Vec::with_capacity(intents.len());
        for intent in intents
        {
            if intent.leg_id.is_empty()
                || intent.venue.is_empty()
                || intent.symbol.is_empty()
                || !intent.target_qty.is_finite()
                || intent.target_qty <= 0.0
            {
                return Err(MultiLegError::InvalidLeg {
                    leg_id: intent.leg_id,
                });
            }
            if !seen.insert(intent.leg_id.clone())
            {
                return Err(MultiLegError::DuplicateLegId {
                    leg_id: intent.leg_id,
                });
            }
            legs.push(LegProgress {
                intent,
                filled_qty: 0.0,
                avg_fill_price: 0.0,
                fees_paid: 0.0,
            });
        }
        Ok(Self {
            strategy_id: strategy_id.to_string(),
            legs,
            fill_fraction_tolerance,
        })
    }

    pub fn record_fill(
        &mut self,
        leg_id: &str,
        price: f32,
        qty: f32,
        fee: f32,
    ) -> Result<(), MultiLegError> {
        let leg = self
            .legs
            .iter_mut()
            .find(|leg| leg.intent.leg_id == leg_id)
            .ok_or_else(|| MultiLegError::UnknownLeg {
                leg_id: leg_id.to_string(),
            })?;
        if !price.is_finite()
            || price <= 0.0
            || !qty.is_finite()
            || qty <= 0.0
            || !fee.is_finite()
            || fee < 0.0
            || leg.filled_qty + qty > leg.intent.target_qty + 1e-6
        {
            return Err(MultiLegError::InvalidFill {
                leg_id: leg_id.to_string(),
            });
        }
        let previous = leg.filled_qty;
        let next = previous + qty;
        leg.avg_fill_price = (leg.avg_fill_price * previous + price * qty) / next.max(1e-12);
        leg.filled_qty = next;
        leg.fees_paid += fee;
        Ok(())
    }

    pub fn max_fill_fraction_gap(&self) -> f32 {
        if self.legs.is_empty()
        {
            return 0.0;
        }
        let mut min_fraction = 1.0f32;
        let mut max_fraction = 0.0f32;
        for leg in &self.legs
        {
            let fraction = leg.fill_fraction();
            min_fraction = min_fraction.min(fraction);
            max_fraction = max_fraction.max(fraction);
        }
        max_fraction - min_fraction
    }

    pub fn status(&self) -> MultiLegStatus {
        if self.legs.iter().all(|leg| leg.complete())
        {
            return MultiLegStatus::Complete;
        }
        if self.legs.iter().all(|leg| leg.filled_qty <= 1e-12)
        {
            return MultiLegStatus::Unfilled;
        }
        if self.max_fill_fraction_gap() > self.fill_fraction_tolerance
        {
            MultiLegStatus::Imbalanced
        }
        else
        {
            MultiLegStatus::BalancedPartial
        }
    }

    /// Produce a deterministic recovery plan without mutating the observed
    /// multi-leg state.
    pub fn recovery_plan(&self, mode: RecoveryMode) -> RecoveryPlan {
        let mut actions = Vec::new();
        for leg in &self.legs
        {
            let (side, qty) = match mode
            {
                RecoveryMode::CompleteOutstanding => (leg.intent.side, leg.remaining_qty()),
                RecoveryMode::NeutralizeFilled => (leg.intent.side.opposite(), leg.filled_qty),
            };
            if qty > 1e-9
            {
                actions.push(RecoveryAction {
                    leg_id: leg.intent.leg_id.clone(),
                    venue: leg.intent.venue.clone(),
                    symbol: leg.intent.symbol.clone(),
                    side,
                    qty,
                    mode,
                });
            }
        }
        RecoveryPlan {
            strategy_id: self.strategy_id.clone(),
            observed_status: self.status(),
            max_fill_fraction_gap: self.max_fill_fraction_gap(),
            mode,
            actions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intents() -> Vec<LegIntent> {
        vec![
            LegIntent {
                leg_id: "buy-spot".to_string(),
                venue: "A".to_string(),
                symbol: "BTC-USDT".to_string(),
                side: Side::Buy,
                target_qty: 1.0,
            },
            LegIntent {
                leg_id: "sell-perp".to_string(),
                venue: "B".to_string(),
                symbol: "BTC-PERP".to_string(),
                side: Side::Sell,
                target_qty: 1.0,
            },
        ]
    }

    #[test]
    fn one_filled_leg_is_explicitly_imbalanced() {
        let mut state = MultiLegState::new("basis-1", intents(), 0.05).unwrap();
        state.record_fill("buy-spot", 100.0, 1.0, 0.1).unwrap();
        assert_eq!(state.status(), MultiLegStatus::Imbalanced);
        assert!((state.max_fill_fraction_gap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn completion_recovery_only_targets_missing_quantity() {
        let mut state = MultiLegState::new("basis-1", intents(), 0.05).unwrap();
        state.record_fill("buy-spot", 100.0, 1.0, 0.1).unwrap();
        state.record_fill("sell-perp", 101.0, 0.25, 0.1).unwrap();
        let plan = state.recovery_plan(RecoveryMode::CompleteOutstanding);
        assert_eq!(plan.observed_status, MultiLegStatus::Imbalanced);
        assert_eq!(plan.actions.len(), 1);
        assert_eq!(plan.actions[0].leg_id, "sell-perp");
        assert_eq!(plan.actions[0].side, Side::Sell);
        assert!((plan.actions[0].qty - 0.75).abs() < 1e-6);
    }

    #[test]
    fn neutralization_reverses_every_observed_fill() {
        let mut state = MultiLegState::new("basis-1", intents(), 0.05).unwrap();
        state.record_fill("buy-spot", 100.0, 1.0, 0.1).unwrap();
        state.record_fill("sell-perp", 101.0, 0.25, 0.1).unwrap();
        let plan = state.recovery_plan(RecoveryMode::NeutralizeFilled);
        assert_eq!(plan.actions.len(), 2);
        assert_eq!(plan.actions[0].side, Side::Sell);
        assert!((plan.actions[0].qty - 1.0).abs() < 1e-6);
        assert_eq!(plan.actions[1].side, Side::Buy);
        assert!((plan.actions[1].qty - 0.25).abs() < 1e-6);
    }

    #[test]
    fn equal_partial_progress_is_not_labelled_imbalanced() {
        let mut state = MultiLegState::new("basis-1", intents(), 0.05).unwrap();
        state.record_fill("buy-spot", 100.0, 0.5, 0.0).unwrap();
        state.record_fill("sell-perp", 101.0, 0.5, 0.0).unwrap();
        assert_eq!(state.status(), MultiLegStatus::BalancedPartial);
        assert!(state.max_fill_fraction_gap().abs() < 1e-6);
    }

    #[test]
    fn complete_state_has_empty_completion_recovery() {
        let mut state = MultiLegState::new("basis-1", intents(), 0.05).unwrap();
        state.record_fill("buy-spot", 100.0, 1.0, 0.0).unwrap();
        state.record_fill("sell-perp", 101.0, 1.0, 0.0).unwrap();
        assert_eq!(state.status(), MultiLegStatus::Complete);
        assert!(
            state
                .recovery_plan(RecoveryMode::CompleteOutstanding)
                .actions
                .is_empty()
        );
    }

    #[test]
    fn overfill_is_rejected() {
        let mut state = MultiLegState::new("basis-1", intents(), 0.05).unwrap();
        assert!(matches!(
            state.record_fill("buy-spot", 100.0, 1.1, 0.0),
            Err(MultiLegError::InvalidFill { .. })
        ));
    }
}
