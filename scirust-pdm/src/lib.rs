//! SciRust Predictive Maintenance (PDM)
//!
//! Industrial-grade degradation tracking, Remaining Useful Life (RUL) estimation,
//! and specialized fault detectors for automotive production lines.
//!
//! ## Modules
//! - **health** — Health Index computation from feature streams
//! - **rul** — Remaining Useful Life estimation (linear, exponential, MLP-based)
//! - **change_detection** — CUSUM, Page-Hinkley test for regime shifts
//! - **detectors** — Specialized fault detectors (imbalance, misalignment, bearing, cavitation)

pub mod change_detection;
pub mod conformal_guard;
pub mod conformal_rul;
pub mod detectors;
pub mod health;
pub mod rul;

pub use change_detection::{CUSUM, ChangePoint, PageHinkley};
pub use conformal_guard::{ConformalGuard, GuardVerdict};
pub use conformal_rul::ConformalRul;
pub use detectors::{
    BearingFaultDetector, CavitationDetector, FaultReport, FaultSeverity, FaultType,
    ImbalanceDetector, MisalignmentDetector,
};
pub use health::{HealthIndex, HealthState};
pub use rul::{ExponentialRulEstimator, LinearRulEstimator, RulEstimator, RulPrediction};
