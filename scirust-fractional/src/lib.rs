//! # `scirust-fractional` — deterministic fractional calculus
//!
//! This crate supplies explicit, auditable discretizations of fractional
//! derivatives without duplicating SciRust's existing special functions.
//! Gamma evaluations are delegated to `scirust-special`.
//!
//! Implemented operators:
//!
//! - Grünwald–Letnikov coefficients;
//! - a left-sided Riemann–Liouville derivative on a uniform grid;
//! - a left-sided Caputo derivative using the L1 scheme, on a uniform grid
//!   ([`caputo_l1_uniform`]) or an explicitly non-uniform grid
//!   ([`caputo_l1_nonuniform`]).
//!
//! The first release deliberately supports only `0 < alpha < 1`. Higher
//! orders, fast history compression and multidimensional Riesz operators
//! require distinct numerical contracts and will be added separately.
//!
//! ## Example
//!
//! ```
//! use scirust_fractional::{FractionalOrder, caputo_l1_uniform};
//!
//! let alpha = FractionalOrder::new(0.5).expect("valid order");
//! let step = 0.25;
//! let samples: Vec<f64> = (0..=16).map(|i| i as f64 * step).collect();
//!
//! let derivative = caputo_l1_uniform(&samples, step, alpha)
//!     .expect("valid uniform-grid derivative");
//!
//! assert!(derivative.is_finite());
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod caputo;
mod caputo_nonuniform;
mod error;
mod grunwald_letnikov;
mod order;
mod validation;

pub use caputo::caputo_l1_uniform;
pub use caputo_nonuniform::caputo_l1_nonuniform;
pub use error::FractionalError;
pub use grunwald_letnikov::{grunwald_letnikov_weights, riemann_liouville_gl_uniform};
pub use order::FractionalOrder;
