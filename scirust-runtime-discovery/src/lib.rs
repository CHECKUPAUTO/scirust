//! Leakage-aware runtime feature discovery for SciRust policies.
//!
//! This crate deliberately separates hypothesis generation from scientific
//! acceptance. A generated feature is only a typed proposal. It must still be
//! instrumented, evaluated on development data, ablated, and confirmed on an
//! untouched split before it can affect a runtime policy.

mod catalog;
mod proposal;
mod schema;

pub use catalog::generate_catalog;
pub use proposal::{
    ProposalBatch, ProposalRejection, ProposalReview, RankedHypothesis, review_proposals,
    summarize_rejections,
};
pub use schema::{
    ComputeClass, DiscoveryRequest, EvidenceBoundary, FeatureCatalog, FeatureFamily,
    FeatureHypothesis, RejectedHypothesis, RuntimeCost, TemporalAvailability,
};
