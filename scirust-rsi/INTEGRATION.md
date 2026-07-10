# Using `scirust-rsi` from an agent (e.g. the `Memorithm/RSI` repo)

`scirust-rsi` is the *engine* for a recursive-self-improvement agent: it owns the
bounded, elitist, reproducible loop (propose → evaluate → keep-if-better →
repeat). The **agent** supplies the two things the engine cannot: a *generator*
of candidates (often an LLM or a program-synthesiser) and an *evaluator*.

```
            ┌──────────────────────── scirust-rsi (the engine) ───────────────────────┐
 agent  →   │  Guard (bounds) ──► ascend / Star / ExpertIteration / Pbt (elitist loop) │
            └──────────────▲────────────────────────────────────┬─────────────────────┘
                           │ propose(candidate)                  │ keep if strictly better
                           │                                     ▼
                  your generator                          your evaluator
            (LLM / scirust-algogen / scirust-synthesis)   (tests, benchmark, oracle)
```

## 1. Depend on it

`scirust-rsi` is a member of the `scirust` workspace and is not published to
crates.io, so consume it as a **git dependency** that selects the single
package by name:

```toml
# RSI/Cargo.toml
[dependencies]
scirust-rsi = { git = "https://github.com/Memorithm/scirust", branch = "master" }
# Optional extra building blocks for *generating* candidate algorithms:
scirust-algogen   = { git = "https://github.com/Memorithm/scirust", branch = "master" }
scirust-synthesis = { git = "https://github.com/Memorithm/scirust", branch = "master" }
```

Or, if you want the whole umbrella, `scirust = { git = ... }` and reach the
engine via `scirust::rsi`.

## 2. Pick the loop that matches your improvement signal

| Your situation | Implement | Driver |
|---|---|---|
| One artefact you critique-and-revise | `RefineTask` | `SelfRefiner` |
| A model that learns from its own correct attempts | `BootstrapTask` | `Star` |
| A fast policy you can augment with search | `ExpertIterationTask` | `ExpertIteration` |
| You tune hyper-parameters during training | `PbtTask` | `Pbt` |
| A continuous objective `Fn(&[f64]) -> Fitness` | — | `OnePlusLambda` |
| **An LLM proposes candidates** | `Generator` + `Critic` | `LlmRefine` |
| Wire a loop from plain closures (no new type) | — | `adapters::FnRefine` |

For the LLM path, implement `Generator::propose` (one call to your model) and
`Critic::score` (your evaluator), then run `LlmRefine` — a bounded, elitist
best-of-`n` self-refine loop. See `src/llm.rs` and `examples/llm_refine.rs`.
Every `Guard` also accepts a wall-clock `time_budget` (`Guard::time_budget`).

A ready-made Claude-backed generator ships behind the optional `anthropic`
feature — no need to write the HTTP yourself:

```toml
scirust-rsi = { git = "https://github.com/Memorithm/scirust", branch = "master", features = ["anthropic"] }
```

```rust
use scirust_rsi::llm::anthropic::ClaudeGenerator;
let mut generator = ClaudeGenerator::from_env()?      // reads ANTHROPIC_API_KEY
    .model("claude-opus-4-8")                          // default model
    .max_tokens(2048);
// hand `&mut generator` to `LlmRefine::run` alongside your `Critic`.
```

## 3. The agent loop in ~20 lines

```rust
use rand::rngs::StdRng;
use scirust_rsi::refine::{RefineTask, SelfRefiner};
use scirust_rsi::{Fitness, Guard};

/// An algorithm the agent is trying to improve (source string, AST, params...).
#[derive(Clone)]
struct Algo(String);

struct ImproveAlgo;
impl RefineTask for ImproveAlgo {
    type Solution = Algo;

    fn initial(&self, _rng: &mut StdRng) -> Algo {
        Algo("/* seed implementation */".into())
    }

    // EVALUATOR: compile + run the test-suite / benchmark, return a score.
    fn score(&self, a: &Algo) -> Fitness {
        // e.g. fraction of tests passed, minus a complexity penalty.
        evaluate_with_tests(&a.0)
    }

    // GENERATOR: ask the LLM / synthesiser for a critiqued revision.
    fn refine(&self, a: &Algo, rng: &mut StdRng) -> Algo {
        Algo(call_generator(&a.0, rng)) // your LLM/codegen call goes here
    }
}

let (best, report) = SelfRefiner::new(42)
    .run(&ImproveAlgo, &Guard::new().max_iters(50).patience(8).target(1.0));
assert!(report.is_monotone()); // the agent can never ship a regression
# fn evaluate_with_tests(_: &str) -> f64 { 1.0 }
# fn call_generator(_: &str, _: &mut StdRng) -> String { String::new() }
```

## 4. Safety contract the agent inherits for free

- **Termination** — `Guard::max_iters` caps every run.
- **No regression** — adoption is elitist (`min_delta`); a worse candidate is
  rejected. `Report::is_monotone()` proves the best-so-far never decreased.
- **Convergence** — `patience` / `target` stop the loop cleanly.
- **Sandbox** — the engine only ever calls *your* `score`/`refine`. It executes
  no code and never self-modifies; if your evaluator runs generated code, run it
  in *your* sandbox.
- **Reproducible** — seed in, identical run out.

This is the same "propose, evaluate, keep if provably better, repeat" shape as a
Darwin-Gödel-Machine / STOP loop, but with the bounds made explicit and tested.
