//! # `sos-scirust` — the SOS Computational Backend Adapter
//!
//! Invariant VIII (backend independence is structural, RFC-0002 §01 §VIII)
//! names this crate as the **only** SOS crate permitted to depend on
//! `scirust-*` — a rule `sos/scripts/lint-deps.py` now checks mechanically,
//! not just by convention. Every other engine stays backend-agnostic; this
//! crate is where real numerics enter, wrapped behind the engine's own,
//! unmodified types.
//!
//! ## What's here
//!
//! [`eig`] — a closed-form expected-information-gain estimator for
//! `sos-planner`, wrapping [`scirust_gp::GaussianProcess`]'s exact posterior
//! variance via the Gaussian-channel mutual-information formula. This is
//! gap #1 (tier 1) of the `sos-scirust` scoping plan: `sos-planner` already
//! ships the ranking/stopping-rule machinery and deliberately consumes
//! [`sos_planner::Estimate`]s rather than computing them; this crate is the
//! real computation, not a change to that contract.
//!
//! ## What is deliberately not here yet
//!
//! Per gap #1's remaining tiers: the Bayesian-optimization search loop
//! (`scirust-automl::bayesian_optimize`) and the seeded nested-Monte-Carlo
//! fallback for non-Gaussian likelihoods (`scirust-stats::SplitMix64`) are
//! separate, documented follow-on increments. So are gaps #2–8 (the
//! `sos-workflow` `StageExecutor`, `sos-simulation` backends, and the rest) —
//! each is its own increment, not stubbed here. Registry-mediated resolution
//! (binding a `sos-registry` `PluginDescriptor` to this estimator) is also
//! deferred: `sos-scirust` is documented as the in-process "Static Rust...
//! the default" transport (RFC-0002 §10 §1), so direct construction is the
//! expected shape until a caller actually needs to swap implementations.
//!
//! ## Example
//!
//! ```
//! use scirust_gp::{GaussianProcess, Rbf};
//! use sos_core::{HashAlgo, ObjectId};
//! use sos_planner::{Cost, GreedyPlanner, Planner, StopVerdict, UtilityPolicy};
//! use sos_scirust::GpEigEstimator;
//!
//! // A GP fit to a few observations — real numerics, not a placeholder.
//! let x = vec![vec![0.0], vec![1.0], vec![2.0]];
//! let y = vec![0.0, 1.0, 0.0];
//! let kernel = Rbf { lengthscale: 1.0, variance: 1.0 };
//! let gp = GaussianProcess::fit(&x, &y, kernel, 1e-6).unwrap();
//! let est = GpEigEstimator::new(gp, 0.05).unwrap();
//!
//! // Build real Candidates — sos-planner's ranking machinery is untouched.
//! let design = |tag: &[u8]| ObjectId::compute(HashAlgo::default(), b"design", tag);
//! let unexplored = est.candidate(design(b"far"), &[50.0], Cost::new(1, 0, 0, 0));
//! let explored = est.candidate(design(b"near"), &[1.0], Cost::new(1, 0, 0, 0));
//!
//! let plan = GreedyPlanner::new()
//!     .recommend(&[explored, unexplored], UtilityPolicy::EigPerCost, 1)
//!     .unwrap();
//! // The unexplored design has far higher posterior variance, hence EIG.
//! assert_eq!(plan.verdict, StopVerdict::Recommend(unexplored.experiment));
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod eig;
pub mod error;

pub use eig::GpEigEstimator;
pub use error::{Result, ScirustError};
