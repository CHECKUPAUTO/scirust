#![forbid(unsafe_code)]
//! Exact, deterministic experimentation on locally generated toy elliptic curves.
//!
//! This crate deliberately accepts only small prime fields and locally specified
//! curve parameters. It has no key, address, SEC 1, network, or blockchain API.
//! All arithmetic is exact and delegates modular primitives to scirust-modalg.

pub mod canonical;
pub mod curve;
pub mod enumerate;
pub mod experiment;
pub mod field;
pub mod falsify;
pub mod orders;
pub mod report;
pub mod scope;

pub use curve::{CurveError, ToyCurve, ToyPoint};
pub use experiment::{Corpus, CorpusCurve, ExperimentManifest};
pub use field::{FieldError, Fp, PrimeError, ToyPrime};
pub use falsify::{Counterexample, first_point_counterexample};
pub use report::ExperimentReport;
pub use scope::{CorpusKind, LocalResearchCase, ScopeError};
