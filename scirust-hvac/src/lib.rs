//! # scirust-hvac — building / HVAC analytics
//!
//! - [`fdd`] — Air-Handling-Unit Fault Detection & Diagnostics (ASHRAE
//!   Guideline 36-style physical-residual rules: mixing, cooling coil, economizer).
//! - [`nilm`] — Non-Intrusive Load Monitoring: disaggregate a whole-building
//!   power trace into appliance on/off events by step detection + signature matching.
//!
//! Deterministic, pure Rust.

pub mod fdd;
pub mod nilm;

pub use fdd::{AhuFault, AhuReading, diagnose_ahu};
pub use nilm::{LoadEvent, disaggregate};
