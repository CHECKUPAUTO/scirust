# SciRust — Simulation Environments, Round VII (2026-07-11)

Follow-up to rounds I–VI. This round closes the first open item listed at the
end of round VI: **unifying `scirust_rl_algo::AlgoEnv` onto the shared
`scirust_learning::rl::Env` trait**, removing a duplicated environment
abstraction from the workspace.

## The duplication

`scirust-rl-algo` (RL-based algorithm discovery) defined its own environment
trait:

```rust
pub trait AlgoEnv {
    fn reset(&mut self) -> AlgoSearchState;
    fn step(&mut self, action: &AlgoAction) -> (AlgoSearchState, f64, bool);
    fn observe(&self) -> AlgoSearchState;
    fn reward(&self, state: &AlgoSearchState) -> f64;
    fn available_actions(&self, state: &AlgoSearchState) -> Vec<AlgoAction>;
    fn is_terminal(&self, state: &AlgoSearchState) -> bool;
}
```

The first three method shapes (`reset`, `step`, `available_actions`) are
*exactly* the `scirust_learning::rl::Env` trait — a crate `scirust-rl-algo`
already depends on directly. The `(next_state, reward, done)` step tuple is
identical. Two names for one contract.

## What changed

`AlgoEnv` is now a **sub-trait** of the shared `Env`:

```rust
pub trait AlgoEnv: Env<State = AlgoSearchState, Action = AlgoAction> {
    fn observe(&self) -> AlgoSearchState;
    fn reward(&self, state: &AlgoSearchState) -> f64;
    fn is_terminal(&self, state: &AlgoSearchState) -> bool;
}
```

- `reset` / `step` / `available_actions` come from `Env` (with the associated
  types pinned to `AlgoSearchState` / `AlgoAction`).
- `AlgoEnv` keeps only what is genuinely specific to algorithm search: the
  `observe` accessor, the weighted `reward` decomposition
  (correctness · efficiency · simplicity), and the `is_terminal` predicate.
- `AlgoSearchEnv` now carries an `impl Env for AlgoSearchEnv` block (the three
  shared methods, plus `type State`/`type Action`) followed by an
  `impl AlgoEnv` block (the three specific methods). The bodies are unchanged.

Because `AlgoSearchEnv` implements `Env`, the shared tabular / policy-gradient
agents in `scirust-learning` now apply to the algorithm-search environment
directly — there is no longer a private environment abstraction that the rest
of the workspace has to bridge to.

## Why this shape (and not a wrapper)

Round VI listed three ways to remove the duplication; this is the one that
*deletes* the duplicate rather than papering over it with an adapter. An
adapter (à la `rl_bridge::RlEnv`) is the right tool when the two traits live in
crates that must stay decoupled; here `scirust-rl-algo` **already** depends on
`scirust-learning`, so making `AlgoEnv` a sub-trait removes the second
definition outright at zero indirection cost. No public method signature
changed — callers of `env.reset()` / `env.step()` / `env.available_actions()`
keep working, they just need `Env` in scope (already the case within the
crate).

## Verification

- `cargo test -p scirust-rl-algo` — **47 tests green** (including the three
  `AlgoEnv` tests exercising `reset` / `step` / `is_terminal` through the split
  traits).
- `cargo clippy -p scirust-rl-algo --all-targets -- -D warnings` — clean.
- `cargo fmt -p scirust-rl-algo -- --check` — clean.
- `cargo build -p scirust-industrial` (the only other crate that names
  `scirust-rl-algo`) — green; it references the crate in a capability string
  only, so nothing downstream broke.

## What remains

1. Optionally, `System` impls wired directly into the vertical crates (so a
   vertical exposes its own steppable dynamics instead of `scirust-sim`
   re-declaring them).
2. More `sim_*` MCP tools (HVAC set-point, PK dose schedule) and a possible
   `sim_stiff` tool exposing the round-VI Rosenbrock bridge.
