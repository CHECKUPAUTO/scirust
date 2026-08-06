# SciRust — Simulation Environments, Round VIII (2026-07-11)

Follow-up to rounds I–VII. This round advances the third open item from the
round-VII report: **more `sim_*` MCP tools**. Two `scirust-sim` domains that
already existed as tested plants but had no agent-callable surface are now
exposed as Model Context Protocol tools, bringing the `sim_*` family from three
tools to five.

## What shipped

Both tools live in `scirust-mcp/src/tools/sim.rs`, register through
`sim_tools()`, and carry the same SHA-256 hash-chained audit log per call as the
rest of the server. Neither needs a Cargo feature or a new dependency — they use
the default, zero-dependency `scirust-sim` modules.

### `sim_hvac_zone`
Drives the **2R2C single-zone building thermal** model (`scirust-sim::hvac`, the
`scirust-hvac` plant): an air node coupled through the wall thermal mass to a
fixed outside temperature, with a constant HVAC heat input `q_hvac`. It returns:
- the **exact linear steady state** — air `t_out + Q·(R_aw+R_wo)`, wall
  `t_out + Q·R_wo` (the model's analytic oracle);
- the zone heat-loss **conductance** `1/(R_aw+R_wo)` in W/K;
- the air and wall temperatures reached after `duration_s`, plus a
  `reached_steady_state` flag.

The default horizon is `12·c_wall·(r_aw+r_wo)` — about a dozen of the dominant
(slow) thermal time constants — so the zone has genuinely settled by the end
unless the caller overrides it.

### `sim_pharmacokinetics_oral`
Drives the **oral one-compartment** PK model (`scirust-sim::pharmacokinetics`):
a gut depot holding the bioavailable fraction `F` of the dose empties at rate
`k_a` into a central compartment that eliminates at rate `k_e`, giving the
Bateman plasma-concentration curve. It returns:
- the peak concentration **C_max** and the time **t_max** it occurs (read from
  the trajectory, so it is robust even at the `k_a = k_e` singularity where the
  closed-form `t_max` does not exist);
- when `k_a ≠ k_e`, the **analytic `t_max`** `ln(k_a/k_e)/(k_a−k_e)` as a
  cross-check;
- the terminal elimination **half-life** `ln(2)/k_e`;
- the **exact total exposure** `AUC(0..∞) = F·dose/(V·k_e)` (closed form, so it
  needs no numerical tail);
- the plasma concentration at the end of the horizon.

The default horizon is ten elimination half-lives.

## Verification

- `cargo test -p scirust-mcp` — **147 tests green** (+4: two per new tool — an
  oracle test and an input-validation test — e.g. the HVAC tool reproducing the
  `t_air = 130 °C`, `t_wall = 105 °C`, `conductance = 4 W/K` steady state, and
  the PK tool reproducing `t_max ≈ 1.65`, `AUC ≈ 10.667`,
  `half-life ≈ 2.773`).
- The `default_registry` collision/coverage test now also asserts both new tool
  names are registered.
- `cargo clippy -p scirust-mcp --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-mcp -- --check` — clean.

## What remains

1. Optionally, `System` impls wired directly into the vertical crates (so a
   vertical exposes its own steppable dynamics instead of `scirust-sim`
   re-declaring them).
2. A `sim_stiff` MCP tool exposing the round-VI Rosenbrock bridge — deferred
   because it requires enabling `scirust-sim`'s optional `stiff` feature on
   `scirust-mcp` (and thus pulling `scirust-stiff` into the server build), a
   distinct dependency-surface decision best made on its own.
