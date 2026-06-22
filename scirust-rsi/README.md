# scirust-rsi — Recursive Self-Improvement (bounded & sandboxed)

Pure-Rust, deterministic implementations of the algorithms that let a learning
system **improve itself** — in the precise machine-learning sense, not the
science-fiction one.

> Every loop here is a *terminating, non-regressing, reproducible* procedure
> that improves a measured objective. None of them execute generated code,
> touch the host, or modify their own binary.

## Algorithms

| Module | Algorithm | The self-improvement signal |
|---|---|---|
| `refine` | **Self-Refine** | critique-and-revise loop on a single solution |
| `star` | **STaR** (Self-Taught Reasoner) | retrain on the system's own *correct* attempts |
| `expert_iteration` | **Expert Iteration** (ExIt / AlphaZero-style) | distil a search-augmented "expert" back into the policy |
| `evo` | **(1+λ)-ES + Rechenberg's 1/5 rule** | the optimiser self-tunes its own mutation strength σ |
| `pbt` | **Population-Based Training** | members copy winners and perturb their own hyper-parameters |

All five run on the same elitist primitive (`ascend`), so they share the same
guarantees.

## Safety model

Recursive self-improvement is only as safe as its bounds. This crate makes them
explicit and enforced by a `Guard` on every loop:

- **Bounded** — `max_iters` caps the run; it always terminates.
- **Monotone / non-regressing** — adoption is *elitist*: a candidate replaces the
  incumbent only if it is measurably better (`min_delta`), so best-so-far never
  decreases. `Report::is_monotone()` verifies this after the fact.
- **Convergence-aware** — `patience` stops the loop once improvement stalls;
  `target` stops it once the goal is met.
- **Sandboxed** — loops operate on data structures and a scalar `Fitness`. They
  never run generated code or self-modify.
- **Reproducible** — every loop is seeded; same seed ⇒ same run.

## Quick start

```rust
use scirust_rsi::{Guard, evo::OnePlusLambda};

// Minimise the sphere function (maximise its negation) in 5 dims.
let opt = OnePlusLambda::new(0xC0FFEE).lambda(8).sigma0(0.5);
let guard = Guard::new().max_iters(500).target(-1e-6);
let (x, fit, report) =
    opt.optimize(vec![3.0; 5], |x| -x.iter().map(|v| v * v).sum::<f64>(), &guard);

assert!(report.is_monotone());
```

Run the examples:

```sh
cargo run -p scirust-rsi --example rsi_demo        # all five loops, offline
cargo run -p scirust-rsi --example llm_refine      # LLM self-refine (mock model)
cargo run -p scirust-rsi --example nn_evolution    # (1+λ)-ES trains a real scirust-core MLP
cargo run -p scirust-rsi --example optimizer_bench # ES vs PBT neuro-evolution on the same MLP (reproducible)
cargo run -p scirust-rsi --example expert_iteration_nn  # Expert Iteration trains a real MLP (expert = local search)
cargo run -p scirust-rsi --example claude_refine --features anthropic   # live Claude (needs ANTHROPIC_API_KEY)
cargo test  -p scirust-rsi
```

## Extending

Implement the trait for your task and hand it to the matching driver:
`RefineTask` → `SelfRefiner`, `BootstrapTask` → `Star`,
`ExpertIterationTask` → `ExpertIteration`, `PbtTask` → `Pbt`. For continuous
objectives, pass a `Fn(&[f64]) -> Fitness` closure straight to `OnePlusLambda`.

Part of the [SciRust](../README.md) pure-Rust scientific-computing stack.
