# SciRust — Simulation Environments, Round IX (2026-07-11)

Follow-up to rounds I–VIII. Every prior domain in `scirust-sim` is *regular*:
its trajectories are smooth, and small changes in the start produce small
changes in the outcome. This round adds the library's first **chaotic** system
— the canonical example of deterministic chaos — to `scirust-sim::mechanics`.

## What shipped

### `mechanics::DoublePendulum`
Two bobs (`m1`, `m2`) on two rigid rods (`l1`, `l2`): the first hangs from a
fixed pivot, the second from the first bob. Angles `θ1`, `θ2` are measured from
the downward vertical; the state is `[θ1, ω1, θ2, ω2]`. It implements `System`
in first-order form like every other mechanics model, using the standard
Lagrangian accelerations

```
θ1'' = [ −g(2m1+m2)sinθ1 − m2·g·sin(θ1−2θ2)
         − 2·sinΔ·m2·(ω2²·l2 + ω1²·l1·cosΔ) ] / [ l1·(2m1 + m2 − m2·cos2Δ) ]
θ2'' = [ 2·sinΔ·(ω1²·l1·(m1+m2) + g(m1+m2)cosθ1 + ω2²·l2·m2·cosΔ) ]
                                       / [ l2·(2m1 + m2 − m2·cos2Δ) ]
```

with `Δ = θ1 − θ2`. A public `energy` method returns the total mechanical
energy (kinetic — including the `cos(θ1−θ2)` cross term from bob-2's compound
velocity — plus gravitational potential).

## The two oracles

A chaotic system has no closed-form trajectory to check against, so the tests
verify the two properties that *define* the regime:

1. **Energy conservation.** Total energy is a first integral of the flow, so it
   must stay constant along any trajectory regardless of how wild the motion
   is. Integrated with the adaptive Dormand–Prince 5(4) solver at a tight
   tolerance from a high-energy (chaotic) start, the energy holds to **1e-6
   relative** over the whole run — the integrator stays on the constant-energy
   surface even as the trajectory itself becomes unpredictable.

2. **Sensitive dependence on initial conditions.** Two starts differing by
   **1e-8** in `θ1` only, integrated with the *identical* fixed-step RK4 (so any
   divergence is physical, not a numerical artifact), separate to **O(1)** in
   phase space by the end — an amplification of more than **1e6×**. This is the
   positive-Lyapunov-exponent signature no regular system can produce; the same
   test on, say, the single pendulum would show the separation staying at 1e-8.

Plus the usual constructor/validation test (rejects non-positive or non-finite
masses, lengths, gravity; `energy` returns `None` on a malformed state instead
of panicking).

## Verification

- `cargo test -p scirust-sim` — **98 tests + 2 doctests green** (+3 for the
  double pendulum).
- `cargo clippy -p scirust-sim --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-sim -- --check` — clean.
- The two heavy chaotic runs are `#[cfg_attr(miri, ignore)]`, matching the
  crate's convention for long transcendental-heavy accuracy tests.

## What remains

1. `System` impls wired directly into the vertical crates (the remaining
   architectural follow-up — needs a dependency-direction decision).
2. A `sim_stiff` MCP tool exposing the round-VI Rosenbrock bridge (needs the
   `stiff` feature enabled on `scirust-mcp`).
3. Further domain models (e.g. Van der Pol / limit-cycle oscillator, a CSTR
   reactor) in the same oracle-tested style.
