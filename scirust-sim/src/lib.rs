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
//!    [`simulate`] integrates it with classical fixed-step RK4, while
//!    [`simulate_adaptive`] uses the error-controlled **Dormand–Prince 5(4)**
//!    method (the `ode45` scheme) to choose the step size automatically —
//!    small through fast transients, large through smooth stretches — and
//!    both return a [`Trajectory`]. The [`SecondOrderSystem`] trait describes
//!    mechanical systems `q'' = a(t, q, q')` and [`simulate_second_order`]
//!    integrates them with the *symplectic* (semi-implicit) Euler method,
//!    whose energy error stays bounded over long horizons where the explicit
//!    method drifts.
//! 2. **A gym-style interaction layer** ([`env`]) — the [`Environment`] trait
//!    (`reset` / `step(action) -> observation, reward, done`) for
//!    agent-in-the-loop simulation, with ready-made control environments
//!    ([`envs::CartPole`], [`envs::GridWorld`]) and an episode runner.
//! 3. **Ready-made domain models**, each validated in its test module against
//!    an analytic solution or a conservation law:
//!    - [`mechanics`] — spring–mass–damper, pendulum, projectile with drag,
//!      and the chaotic double pendulum (energy conservation and sensitive
//!      dependence on initial conditions as oracles);
//!    - [`orbital`] — planar two-body Kepler problem;
//!    - [`epidemiology`] — SIR and SEIR compartmental models;
//!    - [`ecology`] — Lotka–Volterra predator–prey, logistic growth;
//!    - [`chemistry`] — consecutive reactions (Bateman), reversible reaction,
//!      and the stiff Robertson benchmark (integrated via the `stiff` feature);
//!    - [`thermal`] — Newton cooling, 1-D heat conduction (method of lines);
//!    - [`electrical`] — RC charging, series RLC;
//!    - [`stochastic`] — geometric Brownian motion, Ornstein–Uhlenbeck,
//!      and an M/M/1 queue by discrete-event simulation;
//!    - [`pharmacokinetics`] — oral one-compartment and IV two-compartment
//!      drug models (closed-form and AUC oracles);
//!    - [`rigid_body`] — torque-free rotation (Euler's equations), with
//!      energy/angular-momentum conservation and the intermediate-axis
//!      instability;
//!    - [`battery`] — a Thévenin (1-RC) battery cell with self-heating
//!      (the `scirust-bms` plant), coulomb-counting and RC/thermal oracles;
//!    - [`hvac`] — a 2R2C single-zone building thermal model
//!      (the `scirust-hvac` plant), with an exact linear steady state;
//!    - [`grid`] — the synchronous-machine swing equation
//!      (the `scirust-grid` plant), with equilibrium, small-signal frequency
//!      and an energy invariant.
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
//! by `scirust_solvers::ode::dopri5` and by `scirust-stiff`. For genuinely
//! **stiff** plants (e.g. the [`chemistry::Robertson`] kinetics) the explicit
//! engine would need an impractically small step; enabling the optional
//! **`stiff`** feature adds [`stiff_bridge`], which integrates any [`System`]
//! with `scirust-stiff`'s L-stable Backward Euler and adaptive Rosenbrock-W
//! methods.
//! The [`Environment`] trait mirrors the `(state, reward, done)` step shape of
//! `scirust_learning::rl::Env`. Enabling the optional **`rl`** feature adds
//! [`rl_bridge::RlEnv`], an adapter that presents any
//! [`Environment`] + [`FiniteActionSpace`] as a `scirust_learning::rl::Env`, so
//! the existing tabular/PPO/deep agents train on these environments unchanged
//! (a tabular Q-learner finding the shortest path on [`envs::GridWorld`] is a
//! test of the feature). The feature is off by default, keeping the core crate
//! dependency-free.
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

pub mod battery;
pub mod chemistry;
pub mod ecology;
pub mod electrical;
pub mod engine;
pub mod env;
pub mod envs;
pub mod epidemiology;
pub mod grid;
pub mod hvac;
pub mod mechanics;
pub mod orbital;
pub mod pharmacokinetics;
pub mod rigid_body;
#[cfg(feature = "rl")]
pub mod rl_bridge;
pub mod rng;
#[cfg(feature = "stiff")]
pub mod stiff_bridge;
pub mod stochastic;
pub mod thermal;

pub use engine::{
    SecondOrderSystem, SimError, System, Trajectory, simulate, simulate_adaptive,
    simulate_second_order,
};
pub use env::{Environment, FiniteActionSpace, Step, run_episode};
pub use rng::SplitMix64;
