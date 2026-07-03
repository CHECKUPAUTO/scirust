//! # scirust-agtech — precision-agriculture compliance & reproducibility
//!
//! Deterministic, pure-Rust building blocks for precision-agriculture
//! data pipelines and machinery safety compliance (ISO 25119, ISO 18497,
//! ISOBUS/ISO 11783 context — see `docs/DOMAIN_ROADMAP.md` D7):
//!
//! - [`outlier_filter`] / [`idw`] — a reproducible yield-map cleaning
//!   pipeline (global + local outlier filters, inverse-distance-weighted
//!   interpolation) addressing the documented divergence between GIS
//!   tools' default settings (Walczykova et al. 2018).
//! - [`agpl`] — the ISO 25119-2 risk-parameter data model (Severity,
//!   Exposure, Controllability). Deliberately does **not** compute an
//!   AgPL from those parameters — see [`agpl`]'s module doc for why.

pub mod agpl;
pub mod idw;
pub mod outlier_filter;

pub use agpl::{Controllability, Exposure, RiskParameters, Severity};
pub use idw::idw_interpolate;
pub use outlier_filter::{global_filter, local_filter};

use serde::{Deserialize, Serialize};

/// Une mesure de rendement géoréférencée.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct YieldPoint {
    pub x: f64,
    pub y: f64,
    pub yield_value: f64,
}
