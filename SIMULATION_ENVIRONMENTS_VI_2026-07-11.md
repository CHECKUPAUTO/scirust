# SciRust — Simulation Environments, Round VI (2026-07-11)

Follow-up to rounds I–V. This round implements the next open item: **stiff
plants routed to `scirust-stiff`**. The `scirust-sim` engine is explicit (RK4,
Dormand–Prince 5(4), symplectic Euler), so it cannot efficiently integrate
*stiff* systems — those with a fast transient feeding a much slower mode. The
crate's own interoperability note said such plants "belong" in `scirust-stiff`;
this round makes that concrete with a bridge and the canonical stiff benchmark.

## What shipped

### `chemistry::Robertson` (default, zero-dep)
The Robertson autocatalytic system — the textbook stiff-ODE benchmark (rate
constants `k₁=0.04`, `k₂=3·10⁷`, `k₃=10⁴`, nine orders of magnitude apart). It
implements `System` like every other model; total mass `a+b+c` is a linear
invariant (the three derivative terms sum to zero — checked in a default test).

### `stiff_bridge` (behind the optional `stiff` feature)
Two functions integrate any `System` with `scirust-stiff`'s implicit methods:
- `simulate_rosenbrock` — adaptive, linearly-implicit Rosenbrock-W(2,3)
  (`ode23s`-type), the recommended stiff integrator;
- `simulate_backward_euler` — fixed-step, L-stable Backward Euler.

The adapter bridges the shape mismatch (`System::derivatives` writes in place;
`scirust-stiff`'s closures return a `Vec`) and maps `Solution`/`StiffError`
back to `Trajectory`/`SimError`, so a solver failure returns an error rather
than panicking.

### Oracles
- `y' = -50y` integrated by both methods matches `e^{-50t}` (Rosenbrock to
  1e-6; Backward Euler stays bounded and accurate at a step far above the
  explicit stability limit);
- **Robertson at t = 0.4** matches the reference solution (a ≈ 0.9851,
  b ≈ 3.4·10⁻⁵, c ≈ 0.0149) with mass conserved to 1e-6 throughout;
- Backward Euler and Rosenbrock **agree** on Robertson at t = 0.4 (two
  independent stiff methods);
- Robertson integrated to t = 1000 crosses several decades from the ~10⁻⁴
  initial transient, with the majority of the mass converted to C;
- the payoff: explicit `simulate` (RK4) with a coarse step on Robertson returns
  `NonFinite` (it blows up), while `simulate_rosenbrock` succeeds on the same
  span — the reason the bridge exists.

## Zero cost by default

The `stiff` feature is the only thing that pulls `scirust-stiff`. The default
build stays zero-dependency and Miri-clean; the feature is covered by dedicated
CI steps (`cargo test` / `cargo clippy -p scirust-sim --features stiff`),
mirroring the `rl` and wgpu feature jobs.

## Verification

- `cargo test -p scirust-sim` (default) — **95 tests + 2 doctests, green**.
- `cargo test -p scirust-sim --features stiff` — **102 tests + 3 doctests,
  green** (the 7 bridge tests + the bridge doctest).
- `cargo clippy -p scirust-sim --all-targets -- -D warnings` and
  `... --features stiff ...` — clean.
- `cargo fmt -p scirust-sim -- --check` — clean.

## A note on `scirust-stiff`'s adaptive floor

`rosenbrock23` sets its minimum step to `span·1e-10`, so an *enormous* horizon
(e.g. Robertson to t = 4·10⁵) raises the floor above what the stiff initial
layer needs and it reports `StepUnderflow`. The bridge surfaces this honestly
(as `SimError::StepUnderflow`); the tests integrate to horizons the method
handles (t ≤ 1000). Lifting that floor is a possible future `scirust-stiff`
improvement, not a bridge concern.

## What remains

1. Unifying `scirust_rl_algo::AlgoEnv` onto the shared `Environment` trait.
2. Optionally, `System` impls wired directly into the vertical crates.
3. More `sim_*` MCP tools and a possible `sim_stiff` tool exposing this bridge.
