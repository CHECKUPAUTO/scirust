//! # scirust-metrology — measurement assurance
//!
//! - [`gum`] — GUM uncertainty propagation: combined standard uncertainty by
//!   sensitivity coefficients and by Monte-Carlo (Supplement 1).
//! - [`allan`] — Allan variance / deviation for sensor and clock stability.
//!
//! Deterministic — the measurement-trust layer under every other vertical.

pub mod allan;
pub mod gum;

pub use allan::{allan_curve, allan_deviation};
pub use gum::{combined_uncertainty, monte_carlo};
