//! # scirust-spc — Statistical Process Control
//!
//! Deterministic SPC for manufacturing FDC (semiconductor, pharma, agro):
//!
//! - [`ControlChart`] + [`western_electric`] — Shewhart chart with the Western
//!   Electric run rules (catches shifts a single 3σ test misses).
//! - [`EwmaChart`] — EWMA chart, sensitive to small sustained shifts.
//! - [`CusumChart`] — tabular two-sided CUSUM chart with Siegmund ARLs, the
//!   fastest detector of small sustained shifts.
//! - [`HotellingT2`] — multivariate T² monitoring of correlated quality variables.
//! - [`constants`] — Shewhart variable-chart constants (A2/A3/D3/D4/B3/B4, c4,
//!   d2/d3) for subgroup X-bar / R / S charts, validated against the canonical
//!   published tables.
//!
//! Pairs naturally with `scirust-pdm`'s conformal guard for a guaranteed
//! false-alarm rate on the fault-detection layer.

pub mod constants;
pub mod cusum;
pub mod ewma;
pub mod hotelling;
pub mod shewhart;

pub use cusum::{CusumChart, CusumSide, arl_one_sided};
pub use ewma::EwmaChart;
pub use hotelling::HotellingT2;
pub use shewhart::{ControlChart, WesternElectric, western_electric};
