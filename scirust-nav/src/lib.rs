//! # scirust-nav — deterministic navigation and positioning
//!
//! Pure-Rust, bit-reproducible building blocks for industrial navigation:
//!
//! - [`Ins2d`] — a planar inertial dead-reckoning mechanization (integrate
//!   acceleration to velocity and position in a local tangent frame).
//! - [`GnssInsFusion`] — a loosely-coupled GNSS/INS Kalman filter: the IMU
//!   drives a high-rate prediction; intermittent GNSS position fixes correct
//!   it. During a GNSS outage the estimate dead-reckons and its uncertainty
//!   grows; when fixes resume it is pulled back.
//! - [`tdoa`] — time-difference-of-arrival multilateration: locate an emitter
//!   from arrival-time *differences* across sensors of known position. The same
//!   geometry locates a partial-discharge or acoustic-emission source from
//!   sensor arrival times.
//!
//! Every operation accumulates in a fixed order, so a run is bit-identical
//! across machines — the determinism guarantee the rest of SciRust upholds.

pub mod fusion;
pub mod ins;
pub mod tdoa;

pub use fusion::GnssInsFusion;
pub use ins::Ins2d;
pub use tdoa::{tdoa_locate_2d, TdoaSolution};
