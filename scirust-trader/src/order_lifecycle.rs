//! Deterministic venue order lifecycle.
//!
//! The paper `Order` type records fill state. This runtime layer adds the
//! asynchronous states and normalized venue events needed for submit/accept,
//! partial fills, cancel races, amendments and connection state. Events are
//! sequence-checked so deterministic replay can reject gaps in local ordering.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::orders::OrderType;
use crate::venue::{VenueExecutionEvent, VenueOrderCommand, VenueOrderRequest};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LifecycleStatus {
    PendingSubmit,
    Open,
    PartiallyFilled,
    PendingCancel,
    Filled,
    Canceled,
    Rejected,
}

impl LifecycleStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Filled | Self::Canceled | Self::Rejected)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LifecycleOrder {
    pub request: VenueOrderRequest,
    pub venue_order_id: Option<String>,
    pub status: LifecycleStatus,
    pub filled_qty: f32,
    pub avg_fill_price: f32,
    pub cumulative_fee: f32,
    pub revision: u64,
    pub last_ts_ms: i64,
    pub reject_reason: Option<String>,
}

impl LifecycleOrder {
    pub fn remaining(&self) -> f32 {
        (self.request.qty - self.filled_qty).max(0.0)
    }

    pub fn is_active(&self) -> bool {
        !self.status.is_terminal()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LifecycleError {
    DuplicateClientOrderId(u64),
    UnknownClientOrderId(u64),
    NonMonotonicSequence { previous: u64, incoming: u64 },
    InvalidTransition { client_order_id: u64 },
    InvalidFill { client_order_id: u64 },
    InvalidAmend { client_order_id: u64 },
}

/// Single-venue lifecycle state. Use one instance per execution adapter so the
/// venue event sequence remains unambiguous.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleBook {
    pub venue: String,
    pub connected: bool,
    pub last_sequence: Option<u64>,
    pub orders: BTreeMap<u64, LifecycleOrder>,
}

impl LifecycleBook {
    pub fn new(venue: &str) -> Self {
        Self {
            venue: venue.to_string(),
            connected: true,
            last_sequence: None,
            orders: BTreeMap::new(),
        }
    }

    /// Register a locally-created order before its submit command is sent.
    pub fn register_submission(
        &mut self,
        request: VenueOrderRequest,
        ts_ms: i64,
    ) -> Result<VenueOrderCommand, LifecycleError> {
        if self.orders.contains_key(&request.client_order_id)
        {
            return Err(LifecycleError::DuplicateClientOrderId(
                request.client_order_id,
            ));
        }
        let id = request.client_order_id;
        self.orders.insert(
            id,
            LifecycleOrder {
                request: request.clone(),
                venue_order_id: None,
                status: LifecycleStatus::PendingSubmit,
                filled_qty: 0.0,
                avg_fill_price: 0.0,
                cumulative_fee: 0.0,
                revision: 0,
                last_ts_ms: ts_ms,
                reject_reason: None,
            },
        );
        Ok(VenueOrderCommand::Submit(request))
    }

    /// Mark a local cancel intent. Fills remain legal while cancellation is in
    /// flight because exchanges can race a fill against a cancel request.
    pub fn request_cancel(
        &mut self,
        client_order_id: u64,
    ) -> Result<VenueOrderCommand, LifecycleError> {
        let order = self
            .orders
            .get_mut(&client_order_id)
            .ok_or(LifecycleError::UnknownClientOrderId(client_order_id))?;
        if !matches!(
            order.status,
            LifecycleStatus::Open | LifecycleStatus::PartiallyFilled
        )
        {
            return Err(LifecycleError::InvalidTransition { client_order_id });
        }
        order.status = LifecycleStatus::PendingCancel;
        Ok(VenueOrderCommand::Cancel { client_order_id })
    }

    /// Construct an amend command after validating it against current local fill
    /// state. The local order changes only after an `Amended` venue event.
    pub fn request_amend(
        &self,
        client_order_id: u64,
        new_qty: f32,
        new_limit_price: Option<f32>,
    ) -> Result<VenueOrderCommand, LifecycleError> {
        let order = self
            .orders
            .get(&client_order_id)
            .ok_or(LifecycleError::UnknownClientOrderId(client_order_id))?;
        if !matches!(
            order.status,
            LifecycleStatus::Open | LifecycleStatus::PartiallyFilled
        ) || !new_qty.is_finite()
            || new_qty <= 0.0
            || new_qty + 1e-9 < order.filled_qty
            || new_limit_price.is_some_and(|p| !p.is_finite() || p <= 0.0)
        {
            return Err(LifecycleError::InvalidAmend { client_order_id });
        }
        Ok(VenueOrderCommand::Amend {
            client_order_id,
            new_qty,
            new_limit_price,
        })
    }

    fn check_sequence(&self, incoming: u64) -> Result<(), LifecycleError> {
        if let Some(previous) = self.last_sequence
        {
            if incoming <= previous
            {
                return Err(LifecycleError::NonMonotonicSequence { previous, incoming });
            }
        }
        Ok(())
    }

    /// Apply one normalized venue event. State mutates only after the event has
    /// passed sequence and transition validation.
    pub fn apply_event(&mut self, event: VenueExecutionEvent) -> Result<(), LifecycleError> {
        let sequence = event.sequence();
        self.check_sequence(sequence)?;

        match event
        {
            VenueExecutionEvent::Disconnected { .. } =>
            {
                self.connected = false;
            },
            VenueExecutionEvent::Reconnected { .. } =>
            {
                self.connected = true;
            },
            VenueExecutionEvent::Accepted {
                ts_ms,
                client_order_id,
                venue_order_id,
                ..
            } =>
            {
                let order = self.order_mut(client_order_id)?;
                if order.status != LifecycleStatus::PendingSubmit
                {
                    return Err(LifecycleError::InvalidTransition { client_order_id });
                }
                order.venue_order_id = Some(venue_order_id);
                order.status = LifecycleStatus::Open;
                order.last_ts_ms = ts_ms;
            },
            VenueExecutionEvent::Rejected {
                ts_ms,
                client_order_id,
                reason,
                ..
            } =>
            {
                let order = self.order_mut(client_order_id)?;
                if order.status != LifecycleStatus::PendingSubmit
                {
                    return Err(LifecycleError::InvalidTransition { client_order_id });
                }
                order.status = LifecycleStatus::Rejected;
                order.reject_reason = Some(reason);
                order.last_ts_ms = ts_ms;
            },
            VenueExecutionEvent::Fill {
                ts_ms,
                client_order_id,
                price,
                qty,
                fee,
                ..
            } =>
            {
                let order = self.order_mut(client_order_id)?;
                if !matches!(
                    order.status,
                    LifecycleStatus::Open
                        | LifecycleStatus::PartiallyFilled
                        | LifecycleStatus::PendingCancel
                ) || !price.is_finite()
                    || price <= 0.0
                    || !qty.is_finite()
                    || qty <= 0.0
                    || !fee.is_finite()
                    || qty > order.remaining() + 1e-6
                {
                    return Err(LifecycleError::InvalidFill { client_order_id });
                }
                let previous = order.filled_qty;
                let next = previous + qty;
                order.avg_fill_price =
                    (order.avg_fill_price * previous + price * qty) / next.max(1e-12);
                order.filled_qty = next;
                order.cumulative_fee += fee;
                order.last_ts_ms = ts_ms;
                order.status = if order.remaining() <= 1e-9
                {
                    LifecycleStatus::Filled
                }
                else if order.status == LifecycleStatus::PendingCancel
                {
                    LifecycleStatus::PendingCancel
                }
                else
                {
                    LifecycleStatus::PartiallyFilled
                };
            },
            VenueExecutionEvent::Canceled {
                ts_ms,
                client_order_id,
                ..
            } =>
            {
                let order = self.order_mut(client_order_id)?;
                if !matches!(
                    order.status,
                    LifecycleStatus::Open
                        | LifecycleStatus::PartiallyFilled
                        | LifecycleStatus::PendingCancel
                )
                {
                    return Err(LifecycleError::InvalidTransition { client_order_id });
                }
                order.status = LifecycleStatus::Canceled;
                order.last_ts_ms = ts_ms;
            },
            VenueExecutionEvent::Amended {
                ts_ms,
                client_order_id,
                new_qty,
                new_limit_price,
                ..
            } =>
            {
                let order = self.order_mut(client_order_id)?;
                if !matches!(
                    order.status,
                    LifecycleStatus::Open | LifecycleStatus::PartiallyFilled
                ) || !new_qty.is_finite()
                    || new_qty <= 0.0
                    || new_qty + 1e-9 < order.filled_qty
                    || new_limit_price.is_some_and(|p| !p.is_finite() || p <= 0.0)
                {
                    return Err(LifecycleError::InvalidAmend { client_order_id });
                }
                if let Some(price) = new_limit_price
                {
                    order.request.order_type =
                        amend_limit_price(order.request.order_type, price)
                            .ok_or(LifecycleError::InvalidAmend { client_order_id })?;
                }
                order.request.qty = new_qty;
                order.revision += 1;
                order.last_ts_ms = ts_ms;
                if order.remaining() <= 1e-9
                {
                    order.status = LifecycleStatus::Filled;
                }
            },
        }

        self.last_sequence = Some(sequence);
        Ok(())
    }

    fn order_mut(&mut self, client_order_id: u64) -> Result<&mut LifecycleOrder, LifecycleError> {
        self.orders
            .get_mut(&client_order_id)
            .ok_or(LifecycleError::UnknownClientOrderId(client_order_id))
    }
}

fn amend_limit_price(order_type: OrderType, price: f32) -> Option<OrderType> {
    match order_type
    {
        OrderType::Limit { .. } => Some(OrderType::Limit { price }),
        OrderType::TakeProfit { .. } => Some(OrderType::TakeProfit { price }),
        OrderType::StopLimit { stop, .. } => Some(OrderType::StopLimit { stop, limit: price }),
        OrderType::Market | OrderType::StopMarket { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orders::{Side, TimeInForce};

    fn request() -> VenueOrderRequest {
        VenueOrderRequest {
            client_order_id: 1,
            symbol: "BTC-USDT".to_string(),
            side: Side::Buy,
            order_type: OrderType::Limit { price: 100.0 },
            qty: 2.0,
            tif: TimeInForce::Gtc,
            reduce_only: false,
            post_only: true,
        }
    }

    fn accepted(sequence: u64) -> VenueExecutionEvent {
        VenueExecutionEvent::Accepted {
            sequence,
            ts_ms: 100,
            client_order_id: 1,
            venue_order_id: "v-1".to_string(),
        }
    }

    #[test]
    fn submit_accept_partial_fill_fill_is_deterministic() {
        let mut book = LifecycleBook::new("x");
        book.register_submission(request(), 90).unwrap();
        book.apply_event(accepted(1)).unwrap();
        book.apply_event(VenueExecutionEvent::Fill {
            sequence: 2,
            ts_ms: 110,
            client_order_id: 1,
            price: 100.0,
            qty: 0.5,
            fee: 0.01,
            taker: false,
        })
        .unwrap();
        assert_eq!(book.orders[&1].status, LifecycleStatus::PartiallyFilled);
        assert!((book.orders[&1].remaining() - 1.5).abs() < 1e-6);

        book.apply_event(VenueExecutionEvent::Fill {
            sequence: 3,
            ts_ms: 120,
            client_order_id: 1,
            price: 101.0,
            qty: 1.5,
            fee: 0.03,
            taker: false,
        })
        .unwrap();
        assert_eq!(book.orders[&1].status, LifecycleStatus::Filled);
        assert!((book.orders[&1].avg_fill_price - 100.75).abs() < 1e-6);
    }

    #[test]
    fn cancel_race_allows_fill_before_cancel_ack() {
        let mut book = LifecycleBook::new("x");
        book.register_submission(request(), 90).unwrap();
        book.apply_event(accepted(1)).unwrap();
        book.request_cancel(1).unwrap();
        assert_eq!(book.orders[&1].status, LifecycleStatus::PendingCancel);

        book.apply_event(VenueExecutionEvent::Fill {
            sequence: 2,
            ts_ms: 110,
            client_order_id: 1,
            price: 100.0,
            qty: 0.5,
            fee: 0.0,
            taker: false,
        })
        .unwrap();
        assert_eq!(book.orders[&1].status, LifecycleStatus::PendingCancel);

        book.apply_event(VenueExecutionEvent::Canceled {
            sequence: 3,
            ts_ms: 115,
            client_order_id: 1,
        })
        .unwrap();
        assert_eq!(book.orders[&1].status, LifecycleStatus::Canceled);
        assert!((book.orders[&1].filled_qty - 0.5).abs() < 1e-6);
    }

    #[test]
    fn amendment_cannot_reduce_below_filled_quantity() {
        let mut book = LifecycleBook::new("x");
        book.register_submission(request(), 90).unwrap();
        book.apply_event(accepted(1)).unwrap();
        book.apply_event(VenueExecutionEvent::Fill {
            sequence: 2,
            ts_ms: 110,
            client_order_id: 1,
            price: 100.0,
            qty: 1.0,
            fee: 0.0,
            taker: false,
        })
        .unwrap();
        assert!(matches!(
            book.request_amend(1, 0.5, Some(99.0)),
            Err(LifecycleError::InvalidAmend { .. })
        ));
    }

    #[test]
    fn amendment_updates_revision_after_ack() {
        let mut book = LifecycleBook::new("x");
        book.register_submission(request(), 90).unwrap();
        book.apply_event(accepted(1)).unwrap();
        book.request_amend(1, 3.0, Some(99.5)).unwrap();
        book.apply_event(VenueExecutionEvent::Amended {
            sequence: 2,
            ts_ms: 120,
            client_order_id: 1,
            new_qty: 3.0,
            new_limit_price: Some(99.5),
        })
        .unwrap();
        assert_eq!(book.orders[&1].revision, 1);
        assert_eq!(book.orders[&1].request.qty, 3.0);
        assert_eq!(book.orders[&1].request.order_type.limit_price(), Some(99.5));
    }

    #[test]
    fn duplicate_or_out_of_order_sequences_are_rejected() {
        let mut book = LifecycleBook::new("x");
        book.register_submission(request(), 90).unwrap();
        book.apply_event(accepted(10)).unwrap();
        assert!(matches!(
            book.apply_event(VenueExecutionEvent::Disconnected {
                sequence: 10,
                ts_ms: 130
            }),
            Err(LifecycleError::NonMonotonicSequence { .. })
        ));
    }

    #[test]
    fn disconnect_reconnect_is_replayable_state() {
        let mut book = LifecycleBook::new("x");
        book.apply_event(VenueExecutionEvent::Disconnected {
            sequence: 1,
            ts_ms: 100,
        })
        .unwrap();
        assert!(!book.connected);
        book.apply_event(VenueExecutionEvent::Reconnected {
            sequence: 2,
            ts_ms: 200,
        })
        .unwrap();
        assert!(book.connected);
    }
}
