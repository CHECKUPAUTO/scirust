//! # scirust-bms — battery management
//!
//! Deterministic, guarantee-carrying battery analytics for EV and grid storage:
//!
//! - [`BatteryEkf`] — State-of-Charge estimation via an Extended Kalman Filter
//!   over a 1-RC equivalent-circuit model (built on `scirust-estimation`).
//! - [`ThermalGuard`] — thermal-runaway early warning from the *accelerating*
//!   rate of temperature rise, before the critical temperature is reached.
//! - [`ConformalSoh`] — distribution-free State-of-Health bounds with coverage
//!   `≥ 1 − α` (built on the predictive-maintenance conformal machinery).
//!
//! Every estimator is bit-reproducible and validated against an oracle.

pub mod capacity;
pub mod dual;
pub mod soc;
pub mod soh;
pub mod thermal;

pub use capacity::RlsCapacity;
pub use dual::DualEstimator;
pub use soc::{BatteryEkf, CellParams};
pub use soh::ConformalSoh;
pub use thermal::{ThermalGuard, ThermalState};
