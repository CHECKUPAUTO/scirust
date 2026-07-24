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
//! **Gap #1** — `sos-planner` already ships the ranking/stopping-rule
//! machinery and deliberately consumes [`sos_planner::Estimate`]s rather than
//! computing them; this crate is the real computation, not a change to that
//! contract. All three EIG tiers SDE §08 §3 names are landed:
//!
//! * [`eig`] — **tier 1**, closed-form: [`GpEigEstimator`] wraps
//!   [`scirust_gp::GaussianProcess`]'s exact posterior variance via the
//!   Gaussian-channel mutual-information formula. Exact, `L3`, zero standard
//!   error.
//! * [`bo`] — **tier 2**, search: [`bo::BoResult`] reuses
//!   `scirust-automl`'s seeded `bayesian_optimize`/`expected_improvement` loop
//!   to maximize `sos-planner`'s own [`sos_planner::UtilityPolicy::utility`]
//!   over a *continuous* design box, rather than ranking a pre-enumerated
//!   discrete set. Seeded, `L1` (SDE §08 §6's `automl` classification),
//!   even though the EIG value at the returned point is itself exact.
//! * [`nmc`] — **tier 3**, nested Monte Carlo:
//!   [`nmc::NestedMcEigEstimator`] estimates EIG for *discrete hypothesis
//!   discrimination with non-Gaussian likelihoods* — a finite set of
//!   `scirust-stats` [`scirust_stats::DiscreteDistribution`]s (`Poisson`,
//!   `Binomial`, ...), one per hypothesis. The inner Bayes update is exact
//!   (the hypothesis set is small and finite); only the outer expectation
//!   over the observation is Monte Carlo, seeded via `scirust-stats`'
//!   `SplitMix64`. `L1`, and — unlike tiers 1/2 — a genuinely non-zero
//!   standard error, computed for real rather than asserted.
//!
//! **Gap #3** (`sos-simulation` backends) — `sos-simulation` ships the
//! backend-independent [`sos_simulation::Simulate`] syscall, honest
//! determinism stamping, and record/replay, but implements no solver, the
//! same Invariant VIII boundary gap #1 respects for `sos-planner`:
//!
//! * [`ode`] — two backends, both integrating `dy/dt = f(t, y)`, at the two
//!   determinism levels SDE §08 §2 names for this family:
//!     * [`ode::Rk4OdeSimulator`] — `scirust_solvers`'s fixed-step RK4. `L3`,
//!       seedless-deterministic (RFC-0002 §08 §1's classification of
//!       `scirust-solvers` itself) — a fixed sequence of scalar `f64`
//!       operations with no adaptive branching.
//!     * [`ode::Dopri5OdeSimulator`] — `scirust_solvers`'s adaptive
//!       Dormand-Prince 5(4). `L2`, not `L3`: every step's accept/reject
//!       decision branches on a computed error norm against a declared
//!       tolerance, the textbook "iterative solver to a tolerance" case.
//!       Because `Observation` has no dedicated certificate field, the
//!       certificate lives in the output: [`ode::CertifiedTrajectory`] carries
//!       the trajectory *and* the `rtol`/`atol`/accepted/rejected-step
//!       bookkeeping that bounds its accuracy.
//! * [`quadrature`] — a third `L2`-plus-certificate entry:
//!   [`quadrature::QuadratureSimulator`] estimates `∫ₐᵇ f(x) dx` via
//!   `scirust_solvers`'s adaptive Simpson quadrature, using the *strict*
//!   variant that errors on depth exhaustion rather than silently returning a
//!   non-compliant estimate — a certificate that could be wrong isn't one.
//!   [`quadrature::CertifiedIntegral`] needs no accepted/rejected bookkeeping
//!   the way [`ode::CertifiedTrajectory`] does: a *successful* strict call is
//!   by construction guaranteed to have met the declared tolerance.
//!
//! [`ode`] and [`quadrature`] share their `f64`-quantization and
//! `scirust-solvers`-error-mapping helpers via [`solver`] rather than
//! duplicating them.
//!
//! ## What is deliberately not here yet
//!
//! Gap #2 (the `sos-workflow` `StageExecutor`) needs a dispatch/registry
//! mechanism first — a materially different, larger increment — and gaps
//! #4–8 are untouched; each is its own increment, not stubbed here.
//! Registry-mediated resolution (binding a `sos-registry` `PluginDescriptor`
//! to any capability above) is also deferred: `sos-scirust` is documented as
//! the in-process "Static Rust... the default" transport (RFC-0002 §10 §1),
//! so direct construction is the expected shape until a caller actually needs
//! to swap implementations. Within gap #3 itself, `scirust-solvers`' nonlinear
//! (Newton/Broyden) is a separate follow-on backend — root-finding does not
//! obviously fit `Simulate`'s "observe an experiment" framing the way
//! integration does, so it needs its own look rather than a mechanical
//! repeat of this pattern — and `scirust-signal`/`scirust-sim`'s executor
//! kinds (SDE §08 §2) are untouched.
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

pub mod bo;
pub mod eig;
pub mod error;
pub mod nmc;
pub mod ode;
pub mod quadrature;
mod solver;

pub use bo::BoResult;
pub use eig::GpEigEstimator;
pub use error::{Result, ScirustError};
pub use nmc::NestedMcEigEstimator;
pub use ode::{Dopri5OdeSimulator, Rk4OdeSimulator};
pub use quadrature::QuadratureSimulator;
