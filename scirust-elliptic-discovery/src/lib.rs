#![forbid(unsafe_code)]
//! Exact, deterministic experimentation on locally generated toy elliptic curves.
//!
//! This crate deliberately accepts only small prime fields and locally specified
//! curve parameters. It has no key, address, SEC 1, network, or blockchain API.
//! All arithmetic is exact and delegates modular primitives to scirust-modalg.

pub mod curve;
pub mod enumerate;
pub mod field;
pub mod orders;

pub use curve::{CurveError, ToyCurve, ToyPoint};
pub use field::{FieldError, Fp, PrimeError, ToyPrime};
