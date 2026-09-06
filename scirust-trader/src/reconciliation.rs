//! Reconciliation of local lifecycle state against a normalized venue snapshot.
//!
//! Reconciliation is observational: it reports divergence but does not silently
//! rewrite local state. Recovery policy belongs to a separate runtime layer so
//! an agent can inspect exactly what disagreed before deciding or applying a
//! deterministic recovery plan.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::order_lifecycle::{LifecycleBook, LifecycleStatus};

/// Normalized venue-side order snapshot imported from an adapter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VenueOrderSnapshot {
    pub client_order_id: u64,
    pub venue_order_id: Option<String>,
    pub qty: f32,
    pub filled_qty: f32,
    pub avg_fill_price: f32,
    pub status: LifecycleStatus,
}

impl VenueOrderSnapshot {
    pub fn validate(&self) -> bool {
        self.qty.is_finite()
            && self.qty > 0.0
            && self.filled_qty.is_finite()
            && self.filled_qty >= 0.0
            && self.filled_qty <= self.qty + 1e-6
            && self.avg_fill_price.is_finite()
            && self.avg_fill_price >= 0.0
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ReconciliationConfig {
    /// Absolute base-quantity tolerance for normalized venue rounding noise.
    pub qty_tolerance: f32,
    /// Relative average-fill-price tolerance in basis points.
    pub price_tolerance_bps: f32,
}

impl Default for ReconciliationConfig {
    fn default() -> Self {
        Self {
            qty_tolerance: 1e-6,
            price_tolerance_bps: 0.01,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReconciliationDiscrepancy {
    LocalActiveMissingOnVenue {
        client_order_id: u64,
        local_status: LifecycleStatus,
    },
    UnexpectedVenueOrder {
        client_order_id: u64,
        venue_status: LifecycleStatus,
    },
    VenueOrderIdMismatch {
        client_order_id: u64,
        local: Option<String>,
        venue: Option<String>,
    },
    StatusMismatch {
        client_order_id: u64,
        local: LifecycleStatus,
        venue: LifecycleStatus,
    },
    QuantityMismatch {
        client_order_id: u64,
        local_qty: f32,
        venue_qty: f32,
    },
    FilledQuantityMismatch {
        client_order_id: u64,
        local_filled_qty: f32,
        venue_filled_qty: f32,
    },
    AverageFillPriceMismatch {
        client_order_id: u64,
        local_avg_fill_price: f32,
        venue_avg_fill_price: f32,
    },
    InvalidVenueSnapshot {
        client_order_id: u64,
    },
    DuplicateVenueSnapshot {
        client_order_id: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReconciliationReport {
    pub local_orders: usize,
    pub venue_orders: usize,
    pub local_active_orders: usize,
    pub venue_active_orders: usize,
    pub consistent: bool,
    pub discrepancies: Vec<ReconciliationDiscrepancy>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReconciliationError {
    InvalidConfig,
}

fn price_matches(local: f32, venue: f32, tolerance_bps: f32) -> bool {
    if local == 0.0 && venue == 0.0 {
        return true;
    }
    let reference = local.abs().max(venue.abs());
    if reference <= 1e-12 {
        return (local - venue).abs() <= 1e-12;
    }
    (local - venue).abs() / reference * 10_000.0 <= tolerance_bps
}

/// Compare local order state to one venue snapshot without mutating either.
pub fn reconcile_orders(
    local: &LifecycleBook,
    venue_snapshot: &[VenueOrderSnapshot],
    cfg: ReconciliationConfig,
) -> Result<ReconciliationReport, ReconciliationError> {
    if !cfg.qty_tolerance.is_finite()
        || cfg.qty_tolerance < 0.0
        || !cfg.price_tolerance_bps.is_finite()
        || cfg.price_tolerance_bps < 0.0
    {
        return Err(ReconciliationError::InvalidConfig);
    }

    let mut discrepancies = Vec::new();
    let mut venue_by_id = BTreeMap::new();
    for remote in venue_snapshot {
        if !remote.validate() {
            discrepancies.push(ReconciliationDiscrepancy::InvalidVenueSnapshot {
                client_order_id: remote.client_order_id,
            });
            continue;
        }
        if venue_by_id.insert(remote.client_order_id, remote).is_some() {
            discrepancies.push(ReconciliationDiscrepancy::DuplicateVenueSnapshot {
                client_order_id: remote.client_order_id,
            });
        }
    }

    for (client_order_id, local_order) in &local.orders {
        match venue_by_id.get(client_order_id) {
            None => {
                if local_order.is_active() {
                    discrepancies.push(
                        ReconciliationDiscrepancy::LocalActiveMissingOnVenue {
                            client_order_id: *client_order_id,
                            local_status: local_order.status,
                        },
                    );
                }
            }
            Some(remote) => {
                if local_order.venue_order_id != remote.venue_order_id {
                    discrepancies.push(ReconciliationDiscrepancy::VenueOrderIdMismatch {
                        client_order_id: *client_order_id,
                        local: local_order.venue_order_id.clone(),
                        venue: remote.venue_order_id.clone(),
                    });
                }
                if local_order.status != remote.status {
                    discrepancies.push(ReconciliationDiscrepancy::StatusMismatch {
                        client_order_id: *client_order_id,
                        local: local_order.status,
                        venue: remote.status,
                    });
                }
                if (local_order.request.qty - remote.qty).abs() > cfg.qty_tolerance {
                    discrepancies.push(ReconciliationDiscrepancy::QuantityMismatch {
                        client_order_id: *client_order_id,
                        local_qty: local_order.request.qty,
                        venue_qty: remote.qty,
                    });
                }
                if (local_order.filled_qty - remote.filled_qty).abs() > cfg.qty_tolerance {
                    discrepancies.push(ReconciliationDiscrepancy::FilledQuantityMismatch {
                        client_order_id: *client_order_id,
                        local_filled_qty: local_order.filled_qty,
                        venue_filled_qty: remote.filled_qty,
                    });
                }
                if !price_matches(
                    local_order.avg_fill_price,
                    remote.avg_fill_price,
                    cfg.price_tolerance_bps,
                ) {
                    discrepancies.push(ReconciliationDiscrepancy::AverageFillPriceMismatch {
                        client_order_id: *client_order_id,
                        local_avg_fill_price: local_order.avg_fill_price,
                        venue_avg_fill_price: remote.avg_fill_price,
                    });
                }
            }
        }
    }

    for (client_order_id, remote) in &venue_by_id {
        if !local.orders.contains_key(client_order_id) {
            discrepancies.push(ReconciliationDiscrepancy::UnexpectedVenueOrder {
                client_order_id: *client_order_id,
                venue_status: remote.status,
            });
        }
    }

    let local_active_orders = local.orders.values().filter(|o| o.is_active()).count();
    let venue_active_orders = venue_by_id
        .values()
        .filter(|o| !o.status.is_terminal())
        .count();

    Ok(ReconciliationReport {
        local_orders: local.orders.len(),
        venue_orders: venue_by_id.len(),
        local_active_orders,
        venue_active_orders,
        consistent: discrepancies.is_empty(),
        discrepancies,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orders::{OrderType, Side, TimeInForce};
    use crate::venue::{VenueExecutionEvent, VenueOrderRequest};

    fn local_open() -> LifecycleBook {
        let mut book = LifecycleBook::new("x");
        let request = VenueOrderRequest {
            client_order_id: 1,
            symbol: "BTC-USDT".to_string(),
            side: Side::Buy,
            order_type: OrderType::Limit { price: 100.0 },
            qty: 2.0,
            tif: TimeInForce::Gtc,
            reduce_only: false,
            post_only: true,
        };
        book.register_submission(request, 90).unwrap();
        book.apply_event(VenueExecutionEvent::Accepted {
            sequence: 1,
            ts_ms: 100,
            client_order_id: 1,
            venue_order_id: "v-1".to_string(),
        })
        .unwrap();
        book
    }

    fn matching_remote() -> VenueOrderSnapshot {
        VenueOrderSnapshot {
            client_order_id: 1,
            venue_order_id: Some("v-1".to_string()),
            qty: 2.0,
            filled_qty: 0.0,
            avg_fill_price: 0.0,
            status: LifecycleStatus::Open,
        }
    }

    #[test]
    fn identical_normalized_state_is_consistent() {
        let report = reconcile_orders(
            &local_open(),
            &[matching_remote()],
            ReconciliationConfig::default(),
        )
        .unwrap();
        assert!(report.consistent);
        assert!(report.discrepancies.is_empty());
    }

    #[test]
    fn missing_active_local_order_is_reported() {
        let report = reconcile_orders(
            &local_open(),
            &[],
            ReconciliationConfig::default(),
        )
        .unwrap();
        assert!(!report.consistent);
        assert!(matches!(
            report.discrepancies.as_slice(),
            [ReconciliationDiscrepancy::LocalActiveMissingOnVenue {
                client_order_id: 1,
                ..
            }]
        ));
    }

    #[test]
    fn unexpected_venue_order_is_reported() {
        let local = LifecycleBook::new("x");
        let report = reconcile_orders(
            &local,
            &[matching_remote()],
            ReconciliationConfig::default(),
        )
        .unwrap();
        assert!(report.discrepancies.iter().any(|d| matches!(
            d,
            ReconciliationDiscrepancy::UnexpectedVenueOrder { client_order_id: 1, .. }
        )));
    }

    #[test]
    fn quantity_status_and_price_drift_are_all_visible() {
        let mut local = local_open();
        local.apply_event(VenueExecutionEvent::Fill {
            sequence: 2,
            ts_ms: 110,
            client_order_id: 1,
            price: 100.0,
            qty: 0.5,
            fee: 0.01,
            taker: false,
        })
        .unwrap();

        let remote = VenueOrderSnapshot {
            client_order_id: 1,
            venue_order_id: Some("v-1".to_string()),
            qty: 3.0,
            filled_qty: 0.7,
            avg_fill_price: 101.0,
            status: LifecycleStatus::Open,
        };
        let report = reconcile_orders(&local, &[remote], ReconciliationConfig::default()).unwrap();
        assert!(!report.consistent);
        assert!(report.discrepancies.len() >= 4);
    }

    #[test]
    fn quantity_and_price_tolerances_are_explicit() {
        let mut remote = matching_remote();
        remote.qty += 1e-7;
        let cfg = ReconciliationConfig {
            qty_tolerance: 1e-6,
            price_tolerance_bps: 0.1,
        };
        assert!(reconcile_orders(&local_open(), &[remote], cfg).unwrap().consistent);
    }

    #[test]
    fn duplicate_snapshot_ids_are_never_silently_accepted() {
        let remote = matching_remote();
        let report = reconcile_orders(
            &local_open(),
            &[remote.clone(), remote],
            ReconciliationConfig::default(),
        )
        .unwrap();
        assert!(report.discrepancies.iter().any(|d| matches!(
            d,
            ReconciliationDiscrepancy::DuplicateVenueSnapshot { client_order_id: 1 }
        )));
    }
}
