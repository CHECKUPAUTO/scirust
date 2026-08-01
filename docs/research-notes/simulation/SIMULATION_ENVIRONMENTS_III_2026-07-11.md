# SciRust — Simulation Environments, Round III (2026-07-11)

Follow-up to rounds I and II (`SIMULATION_ENVIRONMENTS_2026-07-10.md`,
`SIMULATION_ENVIRONMENTS_II_2026-07-10.md`). This round implements the
**#1 open item** both prior reports flagged: the reinforcement-learning
bridge that connects `scirust-sim`'s environments to the agents already in
`scirust-learning::rl`.

## The gap it closes

`scirust-sim` shipped a gym-style `Environment` trait deliberately shaped like
`scirust_learning::rl::Env` (both step to `(state, reward, done)`), but nothing
actually connected the two — so the CartPole and GridWorld environments could
not be driven by the existing tabular / PPO / deep agents. The RL trait needs
one thing `Environment` doesn't provide: the ability to enumerate the legal
actions (`available_actions`).

## What shipped

### `FiniteActionSpace` (core, no new deps)
A trait `FiniteActionSpace: Environment { fn actions(&self) -> Vec<Self::Action>; }`,
implemented for `CartPole` (`Left`/`Right`) and `GridWorld` (the four moves).
`Move` now derives `Hash` so it can key a Q-table. This lives in the default,
dependency-free part of the crate.

### `rl_bridge::RlEnv` (behind the optional `rl` feature)
`RlEnv<E>` wraps any `E: Environment + FiniteActionSpace` and implements
`scirust_learning::rl::Env` — observation → RL `State`, action → RL `Action`,
reward and `done` pass through, and `available_actions` returns the wrapped
environment's action set. The existing agents drive it with no changes.

The `rl` feature is **off by default**: it is the only thing that pulls
`scirust-learning` (and `rand`, for the test). The default build stays
zero-dependency and Miri-clean; the feature is covered by dedicated CI steps
(`cargo test` / `cargo clippy -p scirust-sim --features rl`), mirroring the
existing portable-simd and wgpu feature jobs.

### Oracles
- **End-to-end learning proof**: a `TabularAgent<(usize,usize), Move>` trained
  on `RlEnv<GridWorld>` for 5 000 episodes converges so that its greedy policy
  reaches the goal in **exactly the Manhattan distance** — the shortest path.
  This is the concrete demonstration that a `scirust-learning` agent trains on
  a `scirust-sim` environment.
- **Adapter plumbing**: a hand-coded optimal policy driven purely through the
  RL `Env` trait also takes the Manhattan distance, and `available_actions`
  returns the right sets for both GridWorld (4) and CartPole (2) — checked
  independently of the learning dynamics.

## Verification

- `cargo test -p scirust-sim` (default) — **83 tests + 2 doctests, green**;
  unchanged from round II, so the Miri gate and zero-dep guarantee are intact.
- `cargo test -p scirust-sim --features rl` — **86 tests + 3 doctests, green**
  (the three new bridge tests + the bridge doctest).
- `cargo clippy -p scirust-sim --all-targets -- -D warnings` and
  `... --features rl ...` — both clean.
- `cargo fmt -p scirust-sim -- --check` — clean.

## What remains

The remaining round-I/II follow-ups are unchanged and still open:
1. Vertical plants implementing `System` (battery RC-thermal, water-hammer
   line, HVAC zone) — now with both an adaptive integrator *and* the RL bridge
   ready to drive them.
2. Stiff plants (Robertson kinetics) routed to `scirust-stiff`.
3. MCP tools (`sim_run`, `sim_episode`) exposing the environments to agents.
4. Optionally, unifying `scirust_rl_algo::AlgoEnv` onto the same trait to
   remove that crate's duplicated environment shape.
