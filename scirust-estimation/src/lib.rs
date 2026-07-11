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

pub mod directional;
pub mod ekf;
pub mod imm;
pub mod interval;
pub mod kalman;
pub mod linalg;
pub mod mimo_fir;
pub mod particle;
pub mod qr_rls;
pub mod rls;
pub mod rls_const;
pub mod smoother;
pub mod squared_givens;
pub mod ud;
pub mod ukf;

pub use directional::DirectionalRls;
pub use ekf::Ekf;
pub use imm::{Imm, ImmModel};
pub use interval::IntervalFilter;
pub use kalman::KalmanFilter;
pub use linalg::Mat;
pub use mimo_fir::{LambdaFit, MimoFirRls, tune_lambda};
pub use particle::ParticleFilter;
pub use qr_rls::{QrRls, QrRlsConst, QrRlsMimo};
pub use rls::{RlsFilter, VectorRls};
pub use rls_const::RlsFilterConst;
pub use smoother::RtsSmoother;
pub use squared_givens::{GivensQrdRls, SquaredGivensRls};
pub use ud::UdFilter;
pub use ukf::Ukf;
