# SciRust — Multi-Domain Simulation Environments (2026-07-10)

Answer to the request: *"je veux que scirust offre des capacités, un
environnement de simulation dans divers domaines"*. This round ships
**`scirust-sim`**, a new workspace crate that turns SciRust's existing
numerical machinery into ready-to-use, oracle-tested **simulation
environments** across eight scientific and engineering domains, plus the
agent-in-the-loop abstraction the roadmap had listed as a gap ("RL gym
abstraction").

## Why a new crate (the gap)

A survey of the 100+ crate workspace before this round found:

- **Integrators exist** (`scirust-solvers::ode::{dopri5, rk4_fixed}`,
  `scirust-stiff`) but only as free functions over closures — no `System` /
  plant abstraction anywhere.
- **Two RL-environment shapes already existed and were duplicated**
  (`scirust-learning::rl::Env`, `scirust-rl-algo::AlgoEnv`) with no concrete
  physical environments behind them.
- The **domain verticals** (`water`, `grid`, `bms`, `hvac`, `biomed`,
  `maritime`, …) contain physics *formulas* and estimators, but **no
  time-stepping simulator** — nothing to generate the very trajectories those
  estimators are meant to consume.
- `scirust-events-*` is streaming event *detection*, not discrete-event
  simulation: the workspace had **no simulation clock / event queue** at all.

`scirust-sim` fills exactly that layer, without duplicating any of the above.

## What shipped

### Engine (`engine.rs`)
- `trait System` — `y' = f(t, y)`, in-place derivative with the same shape as
  the `dopri5` closures (one-line adapter to the adaptive/stiff integrators).
- `simulate` — classical fixed-step RK4 → `Trajectory { t, y }`; validated
  inputs, `SimError` on malformed requests, `NonFinite` on blow-up, bounded
  step budget, lands exactly on `t_end`. An order-4 convergence test measures
  the error ratio at ~16 when halving `h`.
- `trait SecondOrderSystem` + `simulate_second_order` — **symplectic
  (semi-implicit) Euler** for mechanical systems; `FirstOrderForm` adapts any
  second-order system back to RK4. The orbital test integrates ten orbits at
  200 steps/orbit: the symplectic radius stays < 1.05 while explicit Euler
  exceeds 1.3 — the textbook demonstration, now a regression test.

### Agent-in-the-loop layer (`env.rs`, `envs.rs`)
- `trait Environment` — gym-style `reset` / `step(action) → Step
  { observation, reward, done }`, typed observations/actions, mirroring
  `scirust_learning::rl::Env` for a future thin bridge; `run_episode` driver.
- **`CartPole`** — the Barto–Sutton–Anderson task with the reference
  implementation's constants and update order; seeded resets make episodes
  bit-replayable. Test: the classic lean-direction policy survives ≥ 3× the
  steps of a blind constant push.
- **`GridWorld`** — deterministic shortest-path MDP with walls; the greedy
  policy provably takes exactly the Manhattan distance.

### Deterministic randomness (`rng.rs`)
- `SplitMix64` (Vigna's algorithm) — **validated against the published
  reference vectors** (`0xe220a8397b1dcdaf…` for seed 0, cross-checked with an
  independent implementation), uniform/Gaussian (Box–Muller)/exponential
  variates. No ambient randomness anywhere: every stochastic model takes an
  explicit `seed`, per the workspace reproducibility convention.

### The eight domains (each `System`-implementing, each oracle-tested)

| Module | Models | Oracle |
|---|---|---|
| `mechanics` | spring–mass–damper, nonlinear pendulum, projectile with linear drag | underdamped closed form to 1e-6; energy conserved to 1e-9 (spring) / 1e-7 (pendulum at amplitude 2 rad); exact drag solution to 1e-8 |
| `orbital` | planar two-body Kepler | circular orbit closes after `2π√(r³/μ)` to 1e-6; energy & angular momentum conserved to 1e-9; symplectic-vs-explicit demonstration |
| `epidemiology` | SIR, SEIR | `S+I+R` conserved to 1e-12 (RK4 preserves linear invariants); growth ⇔ R₀ > 1; exact final-size relation `ln s∞ = -R₀(1-s∞)` to 1e-3; SEIR peak provably later than SIR |
| `ecology` | Lotka–Volterra, logistic | first integral `δx - γ ln x + βy - α ln y` conserved to 1e-6; return-after-period near equilibrium; logistic closed form |
| `chemistry` | consecutive A→B→C, reversible A⇌B | Bateman solution to 1e-8; mass conserved to 1e-12; relaxation to K = k_f/k_r |
| `thermal` | Newton cooling, 1-D heat rod (method of lines) | closed form to 1e-9; **discrete** sine-mode eigenvalue decay to 1e-6; steady-state linear profile; maximum principle |
| `electrical` | RC, series RLC | closed forms to 1e-9/1e-6; passivity (`dE/dt = -R·i² ≤ 0`) at every step; lossless LC conserves energy |
| `stochastic` | GBM, Ornstein–Uhlenbeck, M/M/1 queue (discrete-event) | exact transition laws (σ=0 degenerates to the exact exponential to 1e-12); moments vs theory; M/M/1 matches L = ρ/(1−ρ), utilization ρ, W = 1/(μ−λ) and Little's law on a 200 000-time-unit run |

## Conventions (the workspace's strictest, no exceptions)

Pure Rust, **zero dependencies**, stable-compatible, `#![forbid(unsafe_code)]`,
`#![deny(missing_docs)]`, no panics on malformed input (`SimError` implements
`Display + Error`), deterministic with explicit seeds, runnable crate-level
doctest, inline oracle tests — the same bar as the seven crates of the
2026-07-10 domain-roadmap round.

## Verification

- `cargo test -p scirust-sim` — **66 tests + 1 doctest, green**.
- `cargo clippy -p scirust-sim --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-sim -- --check` — clean.
- `cargo miri test -p scirust-sim` — **green** (41 executed under the
  interpreter, heavy accuracy/statistics runs `cfg_attr(miri, ignore)`d per
  the `scirust-stiff` precedent); the crate is added to the CI **Miri gate**.
  Note: the two bit-identity tests are also Miri-ignored because Miri
  *deliberately* perturbs transcendental float intrinsics; native jobs enforce
  them.

## Natural follow-ups (not in this round)

1. A feature-gated adapter implementing `scirust_learning::rl::Env` for any
   `scirust-sim::Environment`, so the existing PPO/tabular/deep agents drive
   CartPole/GridWorld unchanged (and `scirust-rl-algo`'s duplicated `AlgoEnv`
   can converge on the same shape).
2. Vertical plants implementing `System` (battery RC-thermal model for `bms`,
   water-hammer line for `water`, zone thermal model for `hvac`) — the
   verticals supply the formulas; the simulator now exists to host them.
3. Adapters to `dopri5`/`rosenbrock23` for adaptive/stiff stepping of
   `System` implementors (the trait shape was chosen to make this a
   one-liner), e.g. Robertson kinetics in `chemistry`.
4. MCP tools (`sim_run`, `sim_episode`) exposing the environments to agents
   through `scirust-mcp`, following the pattern of the SIS/tolerance tools.
