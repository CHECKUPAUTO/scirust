# SciRust — Simulation Environments, Round XII (2026-07-11)

Follow-up to rounds I–XI. With the three architectural follow-ups closed, this
round is purely additive: a new oracle-tested domain model. Round IX added the
library's first *chaotic* system (the double pendulum); this round adds its
counterpart — the first *limit-cycle* system, the **Van der Pol oscillator**.
Together they cover the two hallmark behaviours of nonlinear dynamics.

## `electrical::VanDerPol`

The self-sustaining nonlinear oscillator `x'' - μ·(1 - x²)·x' + x = 0`, state
`y = [x, v]`, implementing `System` like every other model. It lives in
`electrical` — its historical home, Balthasar van der Pol's triode-circuit
work. A public `energy()` method returns `E = ½·(x² + v²)`, whose rate
`dE/dt = μ·(1 - x²)·v²` is the physics that makes the system oscillate.

The defining feature: the nonlinear damping injects energy inside the strip
`|x| < 1` and removes it outside, so **every** trajectory (except the unstable
fixed point at the origin) spirals onto one and the *same* stable periodic
orbit — unlike a linear oscillator, whose amplitude is fixed by its initial
condition. At large `μ` it stiffens into a relaxation oscillator (integrable via
the `stiff` feature).

## The oracles

A limit cycle has no closed-form trajectory, so the tests check the properties
that define the regime:

1. **Global limit-cycle attractor.** A trajectory starting just off the
   origin (inside) and one starting far outside both settle onto the same
   orbit, and its amplitude is the classic **≈ 2** — the headline test.
2. **`μ = 0` recovers the harmonic oscillator.** With no nonlinearity the
   equation is `x'' + x = 0`: the numeric solution matches `x(t) = cos t` and
   the energy `½` is conserved (a closed-form oracle).
3. **Self-oscillation mechanism.** `dE/dt = μ·(1 - x²)·v²` is verified positive
   inside `|x| < 1`, negative outside, and zero on the boundary — a fast,
   analytic (non-Miri-ignored) derivative test.
4. Constructor/validation (rejects non-finite or negative `μ`; `energy`
   returns `None` on a malformed state).

## Verification

- `cargo test -p scirust-sim` — **102 tests + 2 doctests green** (+4).
- `cargo clippy -p scirust-sim --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-sim -- --check` — clean.
- The two heavy settling runs are `#[cfg_attr(miri, ignore)]`, matching the
  crate convention; the analytic `dE/dt` and validation tests run under Miri.

## Status

The multi-domain simulation environment now spans **15 domains**, with both
canonical nonlinear-dynamics behaviours represented (chaos: double pendulum;
limit cycle: Van der Pol). Remaining ideas stay additive (e.g. a CSTR reactor,
or exposing Van der Pol as a `sim_*` MCP tool to showcase the stiff solver at
large `μ`).
