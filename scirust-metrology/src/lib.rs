//! # scirust-metrology — measurement assurance
//!
//! - [`gum`] — GUM uncertainty propagation: combined standard uncertainty by
//!   sensitivity coefficients and by Monte-Carlo (Supplement 1).
//! - [`expanded`] — GUM expanded uncertainty: Welch–Satterthwaite effective
//!   degrees of freedom, the coverage factor `k`, and the coverage interval.
//! - [`allan`] — Allan variance / deviation for sensor and clock stability.
//!
//! Deterministic — the measurement-trust layer under every other vertical.

pub mod allan;
pub mod expanded;
pub mod gum;

pub use allan::{allan_curve, allan_deviation};
pub use expanded::{
    ExpandedUncertainty, coverage_factor, effective_dof, expanded_uncertainty, t_quantile,
};
pub use gum::{combined_uncertainty, monte_carlo};
