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
//! * [`root`] — [`root::BroydenRootSimulator`] solves `F(x) = 0` with
//!   `scirust_solvers`' quasi-Newton Broyden iteration. Also `L2`, and the
//!   certificate it carries is unusually load-bearing: a root solve is
//!   *basin-dependent*, so [`root::CertifiedRoot`] records
//!   [`started_from`](root::CertifiedRoot::started_from) alongside the residual
//!   — a root reported without its starting point cannot be reproduced or even
//!   correctly interpreted.
//! * [`spectrum`] — [`spectrum::PeriodogramSimulator`] computes a windowed
//!   periodogram with `scirust-signal`'s FFT: the first backend here whose
//!   observation answers *"what is in this signal?"* rather than *"what does
//!   this system do?"*. `L3` — and its module docs are explicit that `L3`
//!   asserts bit-reproducibility and **not** accuracy, since a single
//!   periodogram is a biased, high-variance PSD estimator whose variance does
//!   not shrink with record length. Like [`root::CertifiedRoot`], its output
//!   carries the choice that shaped it: [`spectrum::Spectrum::window`].
//!
//! * [`model`] — [`model::CatalogSimulator`] runs `scirust-sim`'s
//!   oracle-tested domain models (SIR/SEIR, Lotka-Volterra, RC circuits, Van
//!   der Pol, damped oscillators, ...) on its fixed-step RK4. `L3`, and the
//!   reason it exists is not the integrator — [`ode::Rk4OdeSimulator`] already
//!   integrates — but **identity**. That backend's model is an anonymous
//!   closure, so an [`ode::OdeConfig`]'s content address fixes the span, the
//!   step and the initial state while leaving the physics to whichever closure
//!   the handler was built with; the same digest can denote two different
//!   experiments. A [`model::ModelSpec`] is a name plus a parameter vector, so
//!   a [`model::ModelRun`]'s address *determines what will be computed* — the
//!   property a study manifest needs, since it identifies a stage's
//!   configuration by digest and cannot inline a right-hand side.
//!   [`model::ModeledTrajectoryBody`] makes such a run a first-class stored
//!   object that can answer which model produced it.
//!
//! [`ode`], [`quadrature`], [`root`], [`spectrum`] and [`model`] share their
//! `f64`-quantization helpers — and the solver backends their
//! `scirust-solvers`-error mapping — via [`solver`] rather than duplicating
//! them.
//!
//! **Gaps #2, #4–8** — the `sos-workflow` `StageExecutor` now dispatches real
//! work through [`stage::OdeStageHandler`] and [`stage::CatalogStageHandler`],
//! and [`verdict`] supplies the agreement predicates `sos-repro` re-execution
//! checks against. The two handlers are the same mechanism with different
//! guarantees, which is the clearest statement of what [`model`] adds: the
//! catalogue one can be asked
//! [`model_at`](stage::CatalogStageHandler::model_at) — *which model will this
//! stage integrate?* — from a [`Plan`](sos_workflow::Plan) alone, before
//! anything runs, and the closure-based one has no such answer to give.
//!
//! ## What is deliberately not here yet
//!
//! Registry-mediated resolution (binding a `sos-registry` `PluginDescriptor`
//! to any capability above) is deferred: `sos-scirust` is documented as
//! the in-process "Static Rust... the default" transport (RFC-0002 §10 §1),
//! so direct construction is the expected shape until a caller actually needs
//! to swap implementations. Within the signal family, Welch averaging and
//! spectrograms are follow-on work rather than something [`spectrum`] pretends
//! to.
//!
//! Only two of the five [`Simulate`](sos_simulation::Simulate) backends have
//! stage handlers ([`ode::Rk4OdeSimulator`] and
//! [`model::CatalogSimulator`]); [`ode::Dopri5OdeSimulator`], [`quadrature`],
//! [`root`] and [`spectrum`] do not, and no other engine's stages have
//! handlers at all. `sos-cli` still has no handler registry, so nothing in
//! that binary can bind a study's plugins to code — `sos run` is not real
//! yet, and that is the remaining gap rather than anything in this crate.
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

pub mod bayes;
pub mod bo;
pub mod eig;
pub mod error;
pub mod model;
pub mod nmc;
pub mod ode;
pub mod quadrature;
pub mod root;
mod solver;
pub mod spectrum;
pub mod stage;
pub mod verdict;

pub use bayes::{BayesFactor, log10_bayes_factor};
pub use bo::BoResult;
pub use eig::GpEigEstimator;
pub use error::{Result, ScirustError};
pub use model::{
    CatalogSimulator, ModelKind, ModelRun, ModelSpec, ModeledTrajectory, ModeledTrajectoryBody,
    UnknownModel,
};
pub use nmc::NestedMcEigEstimator;
pub use ode::{Dopri5OdeSimulator, Rk4OdeSimulator};
pub use quadrature::QuadratureSimulator;
pub use root::{BroydenRootSimulator, CertifiedRoot, RootConfig};
pub use spectrum::{PeriodogramSimulator, Spectrum, SpectrumConfig, WindowKind};
pub use stage::{OdeStageHandler, TrajectoryBody, config_address};
pub use verdict::{Agreement, VerdictError, certify_root};
