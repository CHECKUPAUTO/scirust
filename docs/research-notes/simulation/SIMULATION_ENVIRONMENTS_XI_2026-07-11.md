# SciRust — Simulation Environments, Round XI (2026-07-11)

Follow-up to rounds I–X. This round takes the last open architectural item —
"`System` impls wired directly into the vertical crates" — in its
**non-disruptive** form: a vertical crate exposes *its own* dynamics through the
shared `scirust_sim::System` trait, the reverse of `scirust-sim` re-declaring a
vertical's physics.

## The design question, resolved

`System` lives in `scirust-sim`. For a vertical to implement it, the vertical
must depend on `scirust-sim`. The *literal* reading of the follow-up — have
`scirust-sim` source its models *from* the verticals — would instead force
`scirust-sim` to depend on the verticals, costing it the zero-dependency
property that is its main selling point.

So this round does the acyclic, additive version: **the vertical depends on
`scirust-sim` (optionally, behind a feature) and implements `System`.**
`scirust-sim` depends on no vertical, so there is no cycle, and its default
build is untouched.

The target was chosen by surveying the verticals: grid and HVAC have no
continuous-time dynamics at all (only static estimators), and BMS's 1-RC model
is already re-implemented verbatim in `scirust-sim`. `scirust-biomed`'s
glucose-insulin plant is the clean fit — a genuinely *new* dynamics that
`scirust-sim` does not already have.

## What shipped

### `scirust-biomed::control::sim::GlucoseSystem` (feature `sim`)
`scirust-biomed`'s CBF safety filter (`control::barrier`) models the plant it
controls — `dG/dt = -a·(G - G_b) - k·u` — but only ever evaluates its
instantaneous derivative inside the barrier constraint. `GlucoseSystem` wraps
those same parameters (`control::GlucoseModel`) plus a constant insulin
infusion `u` and implements `scirust_sim::System` (`dim = 1`, state `[G]`), so
`scirust-sim`'s engine integrates the plant forward in time directly.

It also exposes:
- `steady_state()` → `G* = G_b - (k/a)·u`;
- `exact(g0, t)` → the closed-form `G(t) = G* + (G0 - G*)·e^{-a·t}` (the linear
  ODE integrates exactly), used as the integrator's oracle.

### Feature + CI wiring
`scirust-biomed` gains an optional `sim` feature (`sim = ["dep:scirust-sim"]`);
the default build stays as it was (deps: only `scirust-core` + `serde`).
Dedicated CI steps (`cargo test` / `cargo clippy --features sim`) mirror the
existing `rl` / `stiff` / wgpu feature jobs, since the default workspace run
does not build the feature-gated module.

## Verification

- `cargo test -p scirust-biomed` (default) — **41 tests green** (unchanged).
- `cargo test -p scirust-biomed --features sim` — **44 tests + 1 doctest green**
  (+3: the numeric trajectory matches the closed-form solution to 1e-6 under a
  constant infusion; with `u = 0` it relaxes monotonically to `G_b`; the
  derivative vanishes at `G*`).
- `cargo clippy -p scirust-biomed --all-targets -- -D warnings` and
  `… --features sim …` — clean.
- `cargo fmt -p scirust-biomed -- --check` — clean.

## What this closes

All three follow-ups from the round-VII report are now delivered:
1. ✅ `AlgoEnv` unified onto the shared `Env` trait (round VII).
2. ✅ (this round) a vertical exposes its own `System` — the non-disruptive
   direction, keeping `scirust-sim` zero-dependency.
3. ✅ More `sim_*` MCP tools + the `sim_stiff` tool (rounds VIII, X).

Remaining ideas are purely additive (more oracle-tested domains — Van der Pol,
CSTR — or the same `sim`-feature pattern applied to a second vertical), not
architectural.
