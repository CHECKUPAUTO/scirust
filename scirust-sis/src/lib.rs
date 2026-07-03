//! `scirust-sis` — Safety Instrumented Systems (IEC 61511), the
//! process-safety analogue of `scirust-func-safety` (ISO 26262/automotive).
//!
//! Builds the SIS "systems and logic" layer on top of the pure quantitative
//! reliability math already in `scirust-reliability` (`PFDavg`/`PFH`/SIL for
//! the 1oo1/1oo2/2oo2/2oo3/1oo3 MooN family):
//!
//! - [`voting`] — `M`-out-of-`N` voting architectures: evaluate per-channel
//!   votes into a trip decision, and dispatch to the matching `PFDavg`
//!   formula.
//! - [`sif_loop`] — a full Safety Instrumented Function loop (sensors →
//!   logic solver → final elements), whose total `PFDavg` is the sum of its
//!   subsystems' (standard ISA-TR84.00.02 SIL-verification practice).
//! - [`fault_injection`] — simulate a real demand against a set of
//!   dangerous-undetected channel failures, and classify the outcome
//!   (safe trip / dangerous failure / spurious trip).
//! - [`cause_effect`] — cause-and-effect matrices: detected conditions
//!   mapped to safety actions, evaluated deterministically.
//! - [`proof_test`] — inverts `PFDavg` to size the longest proof-test
//!   interval meeting a target (via `scirust-solvers::roots::bisection`,
//!   since the quadratic/cubic MooN forms have no closed-form inverse).
//! - [`audit`] — SHA-256 hash-chained log of trip decisions and
//!   cause-and-effect matrix changes, motivated directly by Triton/Trisis
//!   (2017): SIS logic that can be reprogrammed without a tamper-evident
//!   trail is a demonstrated attack target, not a hypothetical one.
//! - [`reactor_trip`] — channel-bypass reconfiguration for nuclear
//!   reactor-trip voting (IEC 61513 §6.2.3.5): reduces `N` without
//!   changing `M` when a channel goes into maintenance bypass. Does
//!   **not** implement ISA-67.04 setpoint methodology — see that
//!   module's doc for the honest boundary.
//!
//! See `README.md` for the IEC 61511/61508 citations and the Triton/Trisis
//! background in full.

pub mod audit;
pub mod cause_effect;
pub mod error;
pub mod fault_injection;
pub mod proof_test;
pub mod reactor_trip;
pub mod sif_loop;
pub mod voting;

pub use cause_effect::CauseEffectMatrix;
pub use error::{SisError, SisResult};
pub use fault_injection::{TripOutcome, TripSimulationResult, simulate_demand};
pub use proof_test::max_proof_test_interval;
pub use reactor_trip::{architecture_with_bypass, pfd_avg_during_bypass};
pub use sif_loop::{SifLoop, Subsystem};
pub use voting::Architecture;
