# SciRust — Simulation Environments, Round IV (2026-07-11)

Follow-up to rounds I–III. Those built the engine, the adaptive integrator,
new scientific domains and the RL bridge. This round implements the next open
item: **giving the industrial verticals runnable plants**. The very first
capability map of the workspace noted that `scirust-bms`, `scirust-hvac`,
`scirust-grid` (and the other verticals) ship *physics formulas and
estimators* but **no time-stepping simulator** — nothing to generate the
trajectories those estimators are meant to consume. Three new `scirust-sim`
modules close that for the three flagship verticals, each implementing the
`System` trait, oracle-tested, dependency-free.

## The three plants

### `battery` — Thévenin 1-RC cell with self-heating (`scirust-bms`)
State `[soc, v_rc, temp]`: coulomb-counting SoC, a polarization overpotential
(one RC branch), and a lumped thermal node heated by the ohmic + polarization
losses. Oracles:
- SoC is a linear invariant → RK4 tracks the exact `soc0 − I·t/(cap·3600)` to
  round-off (1e-12);
- the overpotential matches the closed form `I·R₁·(1 − e^{−t/τ})`;
- the temperature relaxes to `T_amb + P_gen·R_th` (checked to 1e-3);
- terminal voltage sags under load; charging (`I < 0`) raises SoC.

### `hvac` — 2R2C single-zone building (`scirust-hvac`)
State `[t_air, t_wall]`: an air node and a wall thermal mass, driven by a
constant outside temperature and HVAC power. Oracles:
- the **exact linear steady state** `t_air = t_out + Q·(R_aw+R_wo)`,
  `t_wall = t_out + Q·R_wo`;
- with `Q = 0` the zone relaxes biexponentially to the outside temperature and
  never overshoots the initial-to-outside band.

### `grid` — synchronous-machine swing equation (`scirust-grid`)
`δ'' = (ω_s/2H)(P_m − P_max sin δ) − (D/2H)δ'`, a `SecondOrderSystem`. Oracles:
- the equilibrium `δ* = asin(P_m/P_max)` is a fixed point, and the small-signal
  frequency `√((ω_s/2H)·P_max·cos δ*)` matches (~1.17 Hz for the sample machine);
- with `D = 0` the transient energy `½δ'² − (ω_s/2H)(P_m δ + P_max cos δ)` is
  conserved and a small oscillation returns after one period (symplectic
  integrator);
- with damping the rotor settles back to `δ*`;
- `P_m > P_max` reports *no equilibrium* (loss of synchronism).

## Why in `scirust-sim` (not the vertical crates)

Keeping the plants here preserves `scirust-sim`'s zero-dependency,
Miri-clean, self-contained design and avoids making every vertical crate
depend on `scirust-sim`. The modules are named and documented after the
verticals they model, and they consume the same `System` trait, so a vertical
can adopt one directly. (Wiring `System` impls *into* the vertical crates
remains an option if those crates later want the simulator inline.)

## Verification

- `cargo test -p scirust-sim` — **94 tests + 2 doctests, green** (+11 from the
  three plants; the default build stays zero-dependency).
- `cargo test -p scirust-sim --features rl` — **97 tests + 3 doctests, green**.
- `cargo clippy -p scirust-sim --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-sim -- --check` — clean.
- `cargo miri test -p scirust-sim` — green (50 executed, the heavy integration
  runs `cfg_attr(miri, ignore)`d per the `scirust-stiff` precedent).

## What remains

1. Stiff plants (e.g. a detailed electrochemical or Robertson-style kinetics)
   routed to `scirust-stiff` via the shared closure shape.
2. MCP tools (`sim_run`, `sim_episode`) exposing the environments and plants to
   agents through `scirust-mcp`.
3. Unifying `scirust_rl_algo::AlgoEnv` onto the shared environment trait.
4. Optionally, `System` impls wired directly into the vertical crates for
   inline simulation there.
