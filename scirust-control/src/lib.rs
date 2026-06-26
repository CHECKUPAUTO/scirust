//! # scirust-control — deterministic control with certified input constraints
//!
//! Closes the loop on the SciRust sensing/estimation stack:
//!
//! - [`Pid`] — PID with anti-windup and relay (Åström–Hägglund) auto-tuning.
//! - [`dlqr`] — discrete infinite-horizon LQR (Riccati).
//! - [`solve_box_qp`] — box-constrained convex QP (projected gradient).
//! - [`LinearMpc`] — condensed linear MPC whose box-QP projection makes the
//!   applied input **feasible by construction** (certified input-constraint
//!   satisfaction).
//! - [`detect_oscillation`] — control-loop oscillation/stiction monitoring.
//!
//! Pure Rust, fixed-order `f64` ⇒ bit-reproducible control moves.
//!
//! Commercial use is gated by [`ControlModule`]: unlock the module against a
//! signed entitlement ([`scirust_license`]) before building controllers. The raw
//! constructors remain available for noncommercial use under the dual license.

pub mod license;
pub mod lqr;
pub mod monitor;
pub mod mpc;
pub mod pid;
pub mod qp;

pub use license::ControlModule;
pub use lqr::dlqr;
pub use monitor::{OscillationReport, detect_oscillation};
pub use mpc::LinearMpc;
pub use pid::{Pid, RelayTuning, relay_autotune};
pub use qp::solve_box_qp;
