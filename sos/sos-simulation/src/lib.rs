//! # `sos-simulation` — the SOS Simulation Engine (backend-independent core)
//!
//! A **simulation is an experiment whose executor is a solver** (RFC-0002 §08.5).
//! The Simulation Engine's job is to present a **backend-independent interface**,
//! so the Discovery loop is identical whether evidence comes from a PDE solve, a
//! signal-processing pipeline, or a wet-lab instrument.
//!
//! This crate is that interface, plus honest determinism and record/replay:
//!
//! * [`Simulate`] — the syscall the Discovery loop names instead of a concrete
//!   backend. `sos-scirust` provides the default solver implementations; any
//!   other backend implements the same trait and **declares its own**
//!   [`level`](Simulate::level). This crate implements **no solver** (Invariant
//!   VIII) — no stub.
//! * [`Observation`] — a result stamped with the [`DeterminismLevel`](sos_core::DeterminismLevel)
//!   the backend **realized**, so no result is ever presented as more reproducible
//!   than its backend allows: an `L3` observation reproduces to the bit, an `L1`
//!   in distribution given the seed.
//! * [`Vcr`] — the record/replay memo of the effect boundary: run a simulation
//!   once, replay it thereafter. This is what lets an expensive or one-shot
//!   simulation live inside a reproducible workflow.
//!
//! ## What is deliberately *not* here yet
//!
//! No solver (ODE, PDE, FFT, …) lives here — those are `sos-scirust` backends
//! implementing [`Simulate`] (Invariant VIII). The world-touching effect
//! boundary's **capability authorization** (signed, least-privilege) is enforced
//! by the Workflow Engine's executor seam (`sos-workflow` + `sos-registry`); this
//! crate is the backend-independent interface and the deterministic record/replay
//! half.
//!
//! ## Example
//!
//! ```
//! use sos_core::{DeterminismLevel, SemVer, canonical::{Canonical, CanonicalEncoder}};
//! use sos_simulation::{Simulate, SimDescriptor, Observation, Vcr, SimError};
//!
//! // A tiny bit-exact (L3) integer "solver": sum 0..n, offset by the seed.
//! struct Summation;
//! struct Range { n: u64 }
//! impl Canonical for Range { fn encode(&self, e: &mut CanonicalEncoder) { e.u64(self.n); } }
//!
//! impl Simulate for Summation {
//!     type Config = Range;
//!     type Output = u64;
//!     fn descriptor(&self) -> SimDescriptor { SimDescriptor::new("summation", SemVer::new(1, 0, 0)) }
//!     fn level(&self) -> DeterminismLevel { DeterminismLevel::L3 }
//!     fn run(&self, cfg: &Range, seed: u64) -> Result<Observation<u64>, SimError> {
//!         Ok(Observation::new((0..cfg.n).sum::<u64>().wrapping_add(seed), self.level(), seed))
//!     }
//! }
//!
//! let sim = Summation;
//! // The result carries its honest determinism level.
//! let obs = sim.run(&Range { n: 5 }, 0).unwrap();
//! assert_eq!(obs.output, 10);
//! assert_eq!(obs.level(), DeterminismLevel::L3);
//!
//! // The VCR records the first run and replays it thereafter (identical, free).
//! let mut vcr = Vcr::new();
//! let first = vcr.observe(&sim, &Range { n: 5 }, 0).unwrap();
//! assert!(!first.replayed);
//! let again = vcr.observe(&sim, &Range { n: 5 }, 0).unwrap();
//! assert!(again.replayed);
//! assert_eq!(first.observation, again.observation);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod error;
pub mod observation;
pub mod simulate;
pub mod vcr;

pub use error::{Result, SimError};
pub use observation::Observation;
pub use simulate::{SimDescriptor, Simulate};
pub use vcr::{Recorded, Vcr};
