//! # `sos-planner` — the SOS Planning Engine (deterministic core)
//!
//! The mandate names one major objective: **maximize scientific information.**
//! This engine turns expected-information-gain estimates into the decision *"run
//! this experiment next"* — or the honest *"information is exhausted, stop"* (SDE
//! §05). It is deterministic and **integer fixed-point** (EIG in millibits,
//! `1 bit == 1000`), so a plan is reproducible and citable, with no opaque score
//! (Invariant VI).
//!
//! * [`Estimate`] — an EIG estimate that **carries its own uncertainty** (point,
//!   standard error, determinism level). EIG is expensive and biased to estimate,
//!   so the planner never treats it as exact.
//! * [`Cost`] — the experiment cost model (compute, time, samples, risk).
//! * [`UtilityPolicy`] — the explicit, versioned "value of information" policy;
//!   the default is `U = EIG / cost`.
//! * [`Planner`] / [`GreedyPlanner`] — [`recommend`](Planner::recommend) ranks
//!   candidates by utility and returns a [`Plan`]: the ranked designs (each with
//!   its EIG, cost, and utility) plus a [`StopVerdict`] — run `ξ*`, or
//!   [`InformationExhausted`](StopVerdict::InformationExhausted).
//! * [`StoppingRule`] — the composable stop rules of the discovery loop
//!   (`posterior_mass > p`, `eig < ε`, `budget_exhausted`, `any`/`all`).
//!
//! ## What is deliberately *not* here yet
//!
//! **Computing** EIG — the nested-expectation numerics (closed-form GP predictive
//! variance, nested Monte-Carlo, variational bounds) — is the single hardest piece
//! of numerical work and lives in `sos-scirust` (`scirust-gp` / `scirust-stats`)
//! per Invariant VIII. This crate **consumes** EIG [`Estimate`]s and turns them
//! into decisions; it computes no EIG itself. The heavier value-of-information
//! policies (`knowledge_gradient`, `min_max_regret`) and non-myopic look-ahead are
//! likewise deferred — no stub.
//!
//! ## Example — the worked micro-example (SDE §05.7)
//!
//! ```
//! use sos_core::{DeterminismLevel, HashAlgo, ObjectId};
//! use sos_planner::{Candidate, Cost, Estimate, GreedyPlanner, Planner, StopVerdict, UtilityPolicy};
//!
//! let design = |tag: &[u8]| ObjectId::compute(HashAlgo::default(), b"design", tag);
//! let (a, b) = (design(b"A"), design(b"B"));
//!
//! // Design A: 0.05 bits at cost 1.  Design B: 0.9 bits at cost 2.
//! let candidates = [
//!     Candidate::new(a, Estimate::new(50,  10, DeterminismLevel::L2), Cost::new(1, 0, 0, 0)),
//!     Candidate::new(b, Estimate::new(900, 50, DeterminismLevel::L2), Cost::new(2, 0, 0, 0)),
//! ];
//!
//! // eig_per_cost: U_A = 50*1000/1 = 50_000, U_B = 900*1000/2 = 450_000 ⇒ B wins.
//! let plan = GreedyPlanner::new()
//!     .recommend(&candidates, UtilityPolicy::EigPerCost, 10)
//!     .unwrap();
//! assert_eq!(plan.verdict, StopVerdict::Recommend(b));
//! assert_eq!(plan.best().unwrap().experiment, b);
//!
//! // If instead every design is 0.02 ± 0.03 bits, nothing clears the floor:
//! // information is exhausted — a first-class output, not a silent loop.
//! let noisy = [Candidate::new(a, Estimate::new(20, 30, DeterminismLevel::L1), Cost::new(1, 0, 0, 0))];
//! let stop = GreedyPlanner::new().recommend(&noisy, UtilityPolicy::EigPerCost, 100).unwrap();
//! assert_eq!(stop.verdict, StopVerdict::InformationExhausted);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod error;
pub mod estimate;
pub mod plan;
pub mod planner;
pub mod policy;
pub mod stopping;

pub use error::{PlannerError, Result};
pub use estimate::{Cost, Estimate, MILLIBITS_PER_BIT};
pub use plan::{Candidate, Plan, RankedDesign, StopVerdict, seal_plan};
pub use planner::{GreedyPlanner, Planner};
pub use policy::{EXCLUDED, UTILITY_SCALE, UtilityPolicy};
pub use stopping::{StopSignals, StoppingRule};
