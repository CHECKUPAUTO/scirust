//! Deterministic replay of normalized execution-runtime actions.
//!
//! Replay includes both local intents and venue events. This matters because an
//! exchange acknowledgement alone is not enough to reconstruct the local order:
//! the original normalized request must also be part of the record. Replaying
//! the same ordered records from an empty [`LifecycleBook`] yields the same
//! final lifecycle state or the same first validation error.

use serde::{Deserialize, Serialize};

use crate::order_lifecycle::{LifecycleBook, LifecycleError};
use crate::venue::{VenueExecutionEvent, VenueOrderRequest};

/// One record in the deterministic runtime journal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RuntimeRecord {
    /// Local submit intent recorded before the command is sent to an adapter.
    SubmitIntent {
        ts_ms: i64,
        request: VenueOrderRequest,
    },
    /// Local cancel intent. The venue acknowledgement is a separate record.
    CancelIntent {
        ts_ms: i64,
        client_order_id: u64,
    },
    /// Local amend intent. State changes only when an `Amended` venue event is
    /// replayed, mirroring the live lifecycle semantics.
    AmendIntent {
        ts_ms: i64,
        client_order_id: u64,
        new_qty: f32,
        new_limit_price: Option<f32>,
    },
    VenueEvent(VenueExecutionEvent),
}

impl RuntimeRecord {
    pub fn ts_ms(&self) -> i64 {
        match self {
            Self::SubmitIntent { ts_ms, .. }
            | Self::CancelIntent { ts_ms, .. }
            | Self::AmendIntent { ts_ms, .. } => *ts_ms,
            Self::VenueEvent(event) => event.ts_ms(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReplayError {
    /// Runtime journal timestamps moved backwards. Equal timestamps are allowed
    /// because multiple causally ordered actions can share one millisecond.
    NonMonotonicTimestamp {
        previous_ts_ms: i64,
        incoming_ts_ms: i64,
        record_index: usize,
    },
    Lifecycle {
        record_index: usize,
        error: LifecycleError,
    },
}

/// Result of a successful replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayReport {
    pub records_applied: usize,
    pub first_ts_ms: Option<i64>,
    pub last_ts_ms: Option<i64>,
    pub final_state: LifecycleBook,
}

/// Replay an ordered journal from an empty single-venue lifecycle book.
///
/// Runtime-record timestamp ordering is checked in addition to the strictly
/// increasing venue sequence enforced by [`LifecycleBook`]. This catches a
/// journal whose local intents were accidentally reordered around venue events.
pub fn replay_runtime(
    venue: &str,
    records: &[RuntimeRecord],
) -> Result<ReplayReport, ReplayError> {
    let mut book = LifecycleBook::new(venue);
    let mut previous_ts_ms = None;

    for (index, record) in records.iter().enumerate() {
        let ts_ms = record.ts_ms();
        if let Some(previous) = previous_ts_ms {
            if ts_ms < previous {
                return Err(ReplayError::NonMonotonicTimestamp {
                    previous_ts_ms: previous,
                    incoming_ts_ms: ts_ms,
                    record_index: index,
                });
            }
        }

        let result = match record {
            RuntimeRecord::SubmitIntent { ts_ms, request } => book
                .register_submission(request.clone(), *ts_ms)
                .map(|_| ()),
            RuntimeRecord::CancelIntent {
                client_order_id, ..
            } => book.request_cancel(*client_order_id).map(|_| ()),
            RuntimeRecord::AmendIntent {
                client_order_id,
                new_qty,
                new_limit_price,
                ..
            } => book
                .request_amend(*client_order_id, *new_qty, *new_limit_price)
                .map(|_| ()),
            RuntimeRecord::VenueEvent(event) => book.apply_event(event.clone()),
        };

        if let Err(error) = result {
            return Err(ReplayError::Lifecycle {
                record_index: index,
                error,
            });
        }
        previous_ts_ms = Some(ts_ms);
    }

    Ok(ReplayReport {
        records_applied: records.len(),
        first_ts_ms: records.first().map(RuntimeRecord::ts_ms),
        last_ts_ms: records.last().map(RuntimeRecord::ts_ms),
        final_state: book,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::order_lifecycle::LifecycleStatus;
    use crate::orders::{OrderType, Side, TimeInForce};

    fn request() -> VenueOrderRequest {
        VenueOrderRequest {
            client_order_id: 42,
            symbol: "BTC-USDT".to_string(),
            side: Side::Buy,
            order_type: OrderType::Limit { price: 100.0 },
            qty: 2.0,
            tif: TimeInForce::Gtc,
            reduce_only: false,
            post_only: true,
        }
    }

    fn journal() -> Vec<RuntimeRecord> {
        vec![
            RuntimeRecord::SubmitIntent {
                ts_ms: 100,
                request: request(),
            },
            RuntimeRecord::VenueEvent(VenueExecutionEvent::Accepted {
                sequence: 1,
                ts_ms: 101,
                client_order_id: 42,
                venue_order_id: "v-42".to_string(),
            }),
            RuntimeRecord::VenueEvent(VenueExecutionEvent::Fill {
                sequence: 2,
                ts_ms: 102,
                client_order_id: 42,
                price: 100.0,
                qty: 0.5,
                fee: 0.01,
                taker: false,
            }),
            RuntimeRecord::CancelIntent {
                ts_ms: 103,
                client_order_id: 42,
            },
            RuntimeRecord::VenueEvent(VenueExecutionEvent::Canceled {
                sequence: 3,
                ts_ms: 104,
                client_order_id: 42,
            }),
        ]
    }

    #[test]
    fn same_journal_replays_to_bit_identical_state() {
        let a = replay_runtime("x", &journal()).unwrap();
        let b = replay_runtime("x", &journal()).unwrap();
        let oa = &a.final_state.orders[&42];
        let ob = &b.final_state.orders[&42];
        assert_eq!(oa.status, LifecycleStatus::Canceled);
        assert_eq!(oa.status, ob.status);
        assert_eq!(oa.filled_qty.to_bits(), ob.filled_qty.to_bits());
        assert_eq!(oa.avg_fill_price.to_bits(), ob.avg_fill_price.to_bits());
        assert_eq!(oa.cumulative_fee.to_bits(), ob.cumulative_fee.to_bits());
        assert_eq!(a.final_state.last_sequence, Some(3));
    }

    #[test]
    fn replay_detects_local_timestamp_reordering() {
        let mut records = journal();
        records[3] = RuntimeRecord::CancelIntent {
            ts_ms: 90,
            client_order_id: 42,
        };
        assert!(matches!(
            replay_runtime("x", &records),
            Err(ReplayError::NonMonotonicTimestamp { record_index: 3, .. })
        ));
    }

    #[test]
    fn replay_surfaces_first_invalid_lifecycle_transition() {
        let records = vec![RuntimeRecord::VenueEvent(VenueExecutionEvent::Accepted {
            sequence: 1,
            ts_ms: 100,
            client_order_id: 999,
            venue_order_id: "unknown".to_string(),
        })];
        assert!(matches!(
            replay_runtime("x", &records),
            Err(ReplayError::Lifecycle { record_index: 0, .. })
        ));
    }
}
