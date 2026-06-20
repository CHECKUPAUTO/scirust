//! # scirust-water — deterministic water-network diagnostics
//!
//! Pure-Rust tools for leak detection and surge analysis in pressurised water
//! distribution:
//!
//! - [`leak`] — acoustic **leak correlation**: a leak radiates broadband noise
//!   that reaches two sensors bracketing a pipe segment with a delay set by
//!   where it is. Cross-correlating the two signals recovers that delay, and the
//!   pipe geometry turns it into a position — the method field crews use with a
//!   leak correlator.
//! - [`transient`] — **water-hammer** physics: the Joukowsky pressure surge from
//!   a sudden velocity change, and the Korteweg wave speed from fluid and pipe
//!   elasticity. Surge is what bursts mains; these are the governing equations.
//!
//! Every operation accumulates in a fixed order, so a run is bit-identical
//! across machines — the determinism guarantee the rest of SciRust upholds.

pub mod leak;
pub mod transient;

pub use leak::{locate_leak, LeakLocation};
pub use transient::{joukowsky_surge, korteweg_wave_speed};
