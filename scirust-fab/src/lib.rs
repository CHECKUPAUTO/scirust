//! # scirust-fab — semiconductor-fab process control
//!
//! Deterministic, pure-Rust building blocks for wafer-fab process control
//! (SEMI E10/E58/E116 context — see `docs/DOMAIN_ROADMAP.md` D6):
//!
//! - [`r2r::EwmaR2rController`] — EWMA run-to-run recipe control, the
//!   canonical feedback loop that adjusts the next lot's recipe from the
//!   previous lot's measured output.
//! - [`pca::Pca`] — PCA-based multivariate fault detection (T²/SPE),
//!   built on [`scirust_spc`]'s and [`scirust_solvers`]'s existing
//!   primitives rather than duplicating them.
//!
//! For univariate/simple multivariate SPC (Shewhart, EWMA monitoring
//! charts, Hotelling T² on raw variables), use [`scirust_spc`] directly —
//! this crate adds the *control* (R2R) and *PCA-based* FDC layers on top,
//! it does not re-implement what [`scirust_spc`] already provides.

pub mod pca;
pub mod r2r;

pub use pca::Pca;
pub use r2r::EwmaR2rController;
