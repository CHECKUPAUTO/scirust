//! # `scirust-sim` — multi-domain simulation environments
//!
//! SciRust has oracle-tested numerical integrators (`scirust-solvers`,
//! `scirust-stiff`) and domain verticals full of physics *formulas*, but until
//! this crate there was no common way to say "here is a system, step it
//! through time, and let an agent interact with it". This crate provides that
//! layer, in three parts:
//!
//! 1. **A deterministic time-stepping engine** ([`engine`]) — the [`System`]
//!    trait describes any continuous-time dynamical system `y' = f(t, y)`;
//!    [`simulate`] integrates it with classical fixed-step RK4 and returns a
//!    [`Trajectory`]. The [`SecondOrderSystem`] trait describes mechanical
//!    systems `q'' = a(t, q, q')` and [`simulate_second_order`] integrates
//!    them with the *symplectic* (semi-implicit) Euler method, whose energy
//!    error stays bounded over long horizons where the explicit method drifts.
//! 2. **A gym-style interaction layer** ([`env`]) — the [`Environment`] trait
//!    (`reset` / `step(action) -> observation, reward, done`) for
//!    agent-in-the-loop simulation, with ready-made control environments
//!    ([`envs::CartPole`], [`envs::GridWorld`]) and an episode runner.
//! 3. **Ready-made domain models**, each validated in its test module against
//!    an analytic solution or a conservation law:
//!    - [`mechanics`] — spring–mass–damper, pendulum, projectile with drag;
//!    - [`orbital`] — planar two-body Kepler problem;
//!    - [`epidemiology`] — SIR and SEIR compartmental models;
//!    - [`ecology`] — Lotka–Volterra predator–prey, logistic growth;
//!    - [`chemistry`] — consecutive reactions (Bateman), reversible reaction;
//!    - [`thermal`] — Newton cooling, 1-D heat conduction (method of lines);
//!    - [`electrical`] — RC charging, series RLC;
//!    - [`stochastic`] — geometric Brownian motion, Ornstein–Uhlenbeck,
//!      and an M/M/1 queue by discrete-event simulation.
//!
//! Everything is self-contained: the integrators, the [`SplitMix64`] random
//! generator and every model are implemented here. There are no dependencies,
//! no `unsafe`, no global state and no ambient randomness — stochastic models
//! take an explicit `seed`, so a simulation is a pure function of its inputs
//! and identical runs yield bit-identical results.
//!
//! ## Interoperability
//!
//! [`System::derivatives`] uses the same in-place shape as the closures taken
//! by `scirust_solvers::ode::dopri5` and by `scirust-stiff`, so any model in
//! this crate can be handed to those integrators with a one-line adapter
//! (`|t, y, dydt| model.derivatives(t, y, dydt)`) when adaptive or stiff
//! stepping is needed; stiff plants (e.g. Robertson kinetics) belong there.
//! The [`Environment`] trait mirrors the `(state, reward, done)` step shape of
//! `scirust_learning::rl::Env`, so RL agents can drive these environments
//! through a thin adapter.
//!
//! ## Error handling
//!
//! Fallible operations return [`SimError`]. Malformed inputs (non-finite or
//! non-positive steps, dimension mismatches) and numerical blow-up are
//! reported rather than panicking.
//!
//! ## Example
//!
//! Simulate an SIR epidemic and check the invariant the integrator must keep:
//!
//! ```
//! use scirust_sim::epidemiology::Sir;
//! use scirust_sim::simulate;
//!
//! // R0 = beta/gamma = 3: an epidemic in a fully susceptible population.
//! let sir = Sir::new(0.6, 0.2).expect("valid rates");
//! let traj = simulate(&sir, &[0.999, 0.001, 0.0], 0.0, 60.0, 0.05).expect("integrates");
//!
//! // The infected fraction first grew...
//! let peak = traj.column(1).unwrap().iter().cloned().fold(0.0, f64::max);
//! assert!(peak > 0.2);
//! // ...and S + I + R stayed exactly 1 (RK4 preserves linear invariants).
//! let last = traj.last_state().unwrap();
//! assert!((last[0] + last[1] + last[2] - 1.0).abs() < 1e-12);
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod chemistry;
pub mod ecology;
pub mod electrical;
pub mod engine;
pub mod env;
pub mod envs;
pub mod epidemiology;
pub mod mechanics;
pub mod orbital;
pub mod rng;
pub mod stochastic;
pub mod thermal;

pub use engine::{
    SecondOrderSystem, SimError, System, Trajectory, simulate, simulate_second_order,
};
pub use env::{Environment, Step, run_episode};
pub use rng::SplitMix64;
