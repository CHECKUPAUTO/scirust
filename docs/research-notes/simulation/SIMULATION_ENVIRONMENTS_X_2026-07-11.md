# SciRust — Simulation Environments, Round X (2026-07-11)

Follow-up to rounds I–IX. This round delivers the `sim_stiff` MCP tool that
rounds VI and VIII flagged as deferred: it exposes `scirust-sim`'s implicit
**stiff** integrator (the round-VI Rosenbrock bridge to `scirust-stiff`) as an
agent-callable Model Context Protocol tool, bringing the `sim_*` family to six.

## What shipped

### `sim_stiff_robertson`
Integrates the **Robertson autocatalytic reaction** — the canonical stiff-ODE
benchmark, whose rate constants span nine orders of magnitude
(`k₁=0.04`, `k₂=3·10⁷`, `k₃=10⁴`) — with `scirust-sim`'s adaptive,
linearly-implicit **Rosenbrock-W(2,3)** solver via the `stiff_bridge`. The three
rate constants, the initial `[a, b, c]` and the horizon are all parameterizable.
It returns:
- the final species concentrations `[a, b, c]`;
- the total mass `a+b+c` and a `mass_conserved` flag (mass is a linear invariant
  the integrator must preserve);
- the fraction of mass converted to C;
- the step count.

The point of the tool is the same as the bridge's: an explicit method (RK4)
would need an impractically small step — or blow up — on Robertson's fast
initial transient, whereas the implicit Rosenbrock method's stability is
decoupled from it.

### Feature wiring
`scirust-mcp` now enables `scirust-sim`'s optional **`stiff`** feature, which
pulls `scirust-stiff` into the server build (only there — the default
`scirust-sim` build stays zero-dependency). This was the dependency-surface
decision round VIII deferred; the MCP server is the right place to accept it,
since it already aggregates every vertical.

## Verification

- `cargo test -p scirust-mcp` — **149 tests green** (+2: a Hairer & Wanner
  reference oracle at t = 0.4 — a ≈ 0.9851, b ≈ 3.4·10⁻⁵, c ≈ 0.0149, mass
  conserved to 1e-6 — and an input-validation test).
- The `default_registry` coverage test now also asserts `sim_stiff_robertson`.
- `cargo clippy -p scirust-mcp --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-mcp -- --check` — clean.

Docs: README `scirust-mcp` bullet and the `sim.rs` module header extended;
`CHANGELOG.md` entry added.

## What remains

1. `System` impls wired directly into the vertical crates — the last
   architectural follow-up. It carries a genuine dependency-direction tension
   (the `System` trait lives in `scirust-sim`, so a vertical implementing it
   must depend on `scirust-sim`, and sourcing the models *from* the verticals
   would cost `scirust-sim` its zero-dependency property). Worth a design
   decision before implementing rather than guessing.
2. Further oracle-tested domain models (Van der Pol / limit-cycle oscillator,
   a CSTR reactor) in the same style.
