//! # scirust-spc — Statistical Process Control
//!
//! Deterministic SPC for manufacturing FDC (semiconductor, pharma, agro):
//!
//! - [`ControlChart`] + [`western_electric`] — Shewhart chart with the Western
//!   Electric run rules (catches shifts a single 3σ test misses).
//! - [`EwmaChart`] — EWMA chart, sensitive to small sustained shifts.
//! - [`HotellingT2`] — multivariate T² monitoring of correlated quality variables.
//!
//! Pairs naturally with `scirust-pdm`'s conformal guard for a guaranteed
//! false-alarm rate on the fault-detection layer.

pub mod ewma;
pub mod hotelling;
pub mod shewhart;

pub use ewma::EwmaChart;
pub use hotelling::HotellingT2;
pub use shewhart::{ControlChart, WesternElectric, western_electric};
