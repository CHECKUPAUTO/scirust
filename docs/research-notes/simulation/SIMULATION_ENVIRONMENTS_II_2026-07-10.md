# SciRust — Simulation Environments, Round II (2026-07-10)

Follow-up to `SIMULATION_ENVIRONMENTS_2026-07-10.md` / PR #288, which shipped
the `scirust-sim` crate. This round extends it along the two axes the original
report flagged as the natural next steps: a better **integrator** and more
**domains**. Everything stays inside the crate's strictest-in-the-workspace
conventions — pure Rust, zero dependencies, `#![forbid(unsafe_code)]`,
`#![deny(missing_docs)]`, deterministic, oracle-tested.

## 1. Error-controlled adaptive integration (`engine::simulate_adaptive`)

The round-I engine offered only fixed-step methods; the single biggest
usability gap was that a user had to guess a step size, and a problem with a
fast transient followed by a slow tail forced a tiny step over the whole span.

`simulate_adaptive` closes that gap with the **Dormand–Prince 5(4)** embedded
Runge–Kutta pair — the method behind MATLAB's `ode45`:

- Seven-stage tableau; the fifth-order solution advances the state, the
  embedded fourth-order solution gives the local-error estimate.
- **Automatic initial step** via the Hairer–Nørsett–Wanner heuristic (balance
  the scaled sizes of `y`, `f`, and a finite-difference second derivative).
- **Elementary I-controller** (safety 0.9, clamp [0.2, 5.0], exponent 1/5),
  with Hairer's "no growth right after a rejection" rule to avoid a
  reject/accept limit cycle.
- **FSAL** (first-same-as-last): the seventh stage of an accepted step is the
  first stage of the next, so a smooth run costs ~6 derivative evals/step.
- A `SimError::StepUnderflow` guard: a finite-time singularity or an
  unreachable tolerance is reported, never silently mis-integrated.

**Oracles (all in the test module):**
- reproduces `e^{-t}` over `[0,10]` to 1e-8 per sample in <300 accepted steps
  (fixed RK4 at that accuracy needs `h≈6e-3`, ~1700 steps);
- endpoint error *decreases* monotonically as the tolerance tightens;
- on `y'=-50y` the mean accepted step in the settled tail is >5× the mean step
  through the initial transient (the controller genuinely adapts);
- the vector-valued oscillator (via `FirstOrderForm`) matches `cos`/`sin` to
  1e-8;
- lands exactly on `t_end`; rejects malformed tolerances/dimensions;
- a `y'=y²` finite-time blow-up returns `StepUnderflow`/`NonFinite`, not a
  bogus success.

## 2. Pharmacokinetics (`pharmacokinetics`)

A genuinely new domain (clinical pharmacology), chosen because its models have
clean closed forms — strong oracles.

- **`OralOneCompartment`** — first-order absorption from a gut depot into a
  one-compartment body. Central amount is the **Bateman function**; peak at the
  analytic `t_max = ln(k_a/k_e)/(k_a−k_e)`.
- **`TwoCompartmentIv`** — IV bolus into a central compartment exchanging with
  a peripheral one. Central amount is the **biexponential** `A·e^{−αt}+B·e^{−βt}`
  with the hybrid rate constants α,β (roots of `s²+(k₁₀+k₁₂+k₂₁)s+k₁₀k₂₁`).

**Oracles:** both match their closed forms to 1e-7; the peripheral compartment
rises then falls while total body amount decreases monotonically (one-way
elimination); and — tying the two features together — the central-compartment
**AUC**, obtained by trapezoid over an `simulate_adaptive` trajectory, recovers
the exact `dose/k₁₀` to 0.1%.

## 3. Rigid-body dynamics (`rigid_body`)

Torque-free rotation about the principal axes, i.e. **Euler's equations**
`Iᵢω̇ᵢ = (Iⱼ−Iₖ)ωⱼωₖ`. Aerospace/robotics attitude dynamics — a distinct
domain from the round-I mechanics module.

**Oracles:**
- rotational kinetic energy and `|L|²` are exact invariants, conserved to 1e-8
  over a long asymmetric-top run;
- the **symmetric top** (`I₁=I₂`) precesses at the closed-form rate
  `Ω=(I₃−I₁)ω₃/I₁` — `ω₃` constant, transverse magnitude constant, state
  returning after one precession period;
- the **intermediate-axis theorem** (tennis-racket / Dzhanibekov effect): a
  body spun about its middle axis tumbles (off-axis components grow to O(1)),
  while spins about the min and max axes stay stable — reproduced qualitatively.

## Verification

- `cargo test -p scirust-sim` — **83 tests + 2 doctests, green**.
- `cargo clippy -p scirust-sim --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-sim -- --check` — clean.
- `cargo miri test -p scirust-sim` — green (45 executed, the heavy
  accuracy/precession/tumbling runs `cfg_attr(miri, ignore)`d per the
  `scirust-stiff` precedent). The crate is already in the CI Miri gate.

## What remains (still open, unchanged from round I)

1. A feature-gated `scirust_learning::rl::Env` adapter for `Environment`.
2. Vertical plants implementing `System` (battery RC-thermal, water-hammer
   line, HVAC zone) — now with an adaptive integrator ready to drive them.
3. Stiff plants (Robertson kinetics) routed to `scirust-stiff` via the shared
   closure shape.
4. MCP tools (`sim_run`, `sim_episode`) exposing the environments to agents.
