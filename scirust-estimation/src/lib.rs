//! # scirust-estimation — deterministic state estimation
//!
//! Pure-Rust, bit-reproducible state estimators for industrial sensing:
//!
//! - [`KalmanFilter`] — the linear Kalman filter (fixed-order `f64`).
//! - [`Ekf`] — the Extended Kalman filter (nonlinear `f`/`h` via closures + Jacobians).
//! - [`IntervalFilter`] — set-membership estimation with a **containment
//!   guarantee**: a box that provably brackets the true state given bounded
//!   noise — the certified counterpart to the Kalman filter's probabilistic
//!   estimate.
//!
//! Shared infrastructure for the battery (BMS), sensor-fusion and structural
//! verticals. Every operation accumulates in a fixed order, so a run is
//! bit-identical across machines — the determinism guarantee the rest of
//! SciRust upholds, extended to estimation.

pub mod ekf;
pub mod imm;
pub mod interval;
pub mod kalman;
pub mod linalg;
pub mod particle;
pub mod rls;
pub mod smoother;
pub mod ud;
pub mod ukf;

pub use ekf::Ekf;
pub use imm::{Imm, ImmModel};
pub use interval::IntervalFilter;
pub use kalman::KalmanFilter;
pub use linalg::Mat;
pub use particle::ParticleFilter;
pub use rls::{RlsFilter, VectorRls};
pub use smoother::RtsSmoother;
pub use ud::UdFilter;
pub use ukf::Ukf;
