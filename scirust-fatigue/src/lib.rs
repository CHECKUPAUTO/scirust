//! # scirust-fatigue — structural fatigue analysis
//!
//! Deterministic, pure-Rust fatigue-life primitives (DO-178C/DO-333
//! determinism context — see `docs/DOMAIN_ROADMAP.md` D4):
//!
//! - [`rainflow`] — ASTM E1049-85 rainflow cycle counting, ported and
//!   verified against an established reference implementation.
//! - [`miner`] — Palmgren-Miner cumulative damage from counted cycles
//!   and a user-supplied S-N curve.
//!
//! **Honest scope**: this crate covers the cycle-counting and
//! damage-accumulation math. It does not implement deterministic
//! fixed-point flight-control-law arithmetic or certified bounds for
//! learned components — those remain a partnership-scale undertaking
//! (see D4's entry in `docs/DOMAIN_ROADMAP.md`).

pub mod miner;
pub mod rainflow;

pub use miner::{PowerLawSnCurve, miner_damage};
pub use rainflow::{Cycle, Reversal, aggregate_by_range, count_cycles, rainflow_count, reversals};
