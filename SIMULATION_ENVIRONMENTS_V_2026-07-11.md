# SciRust — Simulation Environments, Round V (2026-07-11)

Follow-up to rounds I–IV. Those built `scirust-sim` (engine, adaptive
integrator, thirteen oracle-tested domains, the RL bridge, and the industrial
vertical plants). This round makes the simulations **agent-callable**: they
are now exposed as Model Context Protocol tools in `scirust-mcp`, so any agent
(the in-house `scirust-sciagent` SLM, Claude, ChatGPT, a script) can run a
scenario with a single typed tool call and get back the key metrics — no
integration code to write.

## What shipped

`scirust-mcp` gains `scirust-sim` as a dependency and a new `tools::sim`
module registering three tools (same `McpTool { name, description,
input_schema, handler }` shape, SHA-256 hash-chained audit log per call, as
every other tool):

- **`sim_epidemic`** — Kermack–McKendrick SIR. Inputs `beta`, `gamma`
  (+ optional `initial_infected`, `days`, `dt`). Returns `r0`,
  `peak_infected_fraction`, `peak_day`, `final_attack_rate`, and an `epidemic`
  boolean (`R0 > 1`).
- **`sim_battery_discharge`** — the Thévenin 1-RC cell plant (`scirust-bms`) at
  constant current. Returns final SoC / terminal voltage / temperature, the
  steady-state temperature and the polarization time constant, and a
  `depleted` flag.
- **`sim_grid_stability`** — the swing-equation machine (`scirust-grid`).
  Returns whether a synchronous operating point exists, the equilibrium angle
  `asin(P_m/P_max)`, the small-signal electromechanical frequency (Hz), and —
  when a `disturbance_angle_rad` + `duration_s` are supplied — a transient
  `settled` verdict.

Each handler validates its JSON inputs (missing / non-numeric / non-finite /
out-of-range) and maps `scirust-sim`'s `SimError` to a message string, so a
malformed call returns a helpful error rather than panicking.

## Why this matters

The very first capability map noted that `scirust-mcp` was SciRust's
"connect any agent to the platform" layer. Until now it exposed solvers,
dev tools, discovery and the vertical *primitives* — but not the ability to
*run a simulation*. An agent supervising a battery or a grid node can now ask
"what happens if I draw 4 A for an hour?" or "does this machine stay in step
after a 0.4 rad kick?" and get a deterministic, audited answer.

## Verification

- `cargo test -p scirust-mcp` — **143 tests, green** (the 6 new tool tests +
  the registry's name-collision / presence checks, which now also assert the
  three `sim_*` tools are registered).
- `cargo clippy -p scirust-mcp --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-mcp -- --check` — clean.

The oracle values match the underlying `scirust-sim` tests: `sim_epidemic`
with `beta=0.6, gamma=0.2` reports `R0 = 3`, a peak above 20% and a ~94%
attack rate; `sim_battery_discharge` reproduces the exact coulomb-counting SoC
(`1 − 400/7200 ≈ 0.9444` after 100 s at 4 A from a 2 A·h cell); and
`sim_grid_stability` recovers the `π/6` equilibrium, the ~1.17 Hz mode, and a
`settled` transient under damping.

## What remains

1. Stiff plants routed to `scirust-stiff` for genuinely stiff kinetics.
2. Unifying `scirust_rl_algo::AlgoEnv` onto the shared `Environment` trait.
3. Optionally, `System` impls wired directly into the vertical crates.
4. More `sim_*` tools (e.g. `sim_hvac_zone`, `sim_pk_dose`) as agents ask for
   them — the pattern is now one small module per tool.
