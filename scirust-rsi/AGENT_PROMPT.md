# Agent prompt — drive an RSI loop with `scirust-rsi`

Ready-to-paste prompt for a Claude Code session **scoped to `Memorithm/RSI`**
(this scirust session does not have access to that repo). It makes `RSI` consume
the `scirust-rsi` engine to generate and then recursively improve algorithms,
with the safety contract enforced and verified.

```text
Context
-------
The Memorithm/scirust repository now contains a `scirust-rsi` crate: a pure-Rust,
deterministic and BOUNDED engine for recursive self-improvement (elitist loop
"propose → evaluate → keep if STRICTLY better → repeat").
Integration docs: scirust-rsi/INTEGRATION.md on the master branch.
It exposes the RefineTask / BootstrapTask / ExpertIterationTask / PbtTask traits,
the OnePlusLambda driver, the `ascend` primitive and the `Guard` safeguard
(max_iters, patience, target, min_delta) with `Report::is_monotone()`.

Objective
---------
Make the Memorithm/RSI repository a CONSUMER of scirust-rsi: an agent that
GENERATES candidate algorithms then IMPROVES them in a loop, without ever
regressing. Develop on a new branch, do NOT push to main directly.

Steps
-----
1. Inspect the current state of Memorithm/RSI (structure, Cargo.toml, what exists).
   Also read scirust-rsi/INTEGRATION.md and scirust-rsi/src/lib.rs in scirust.
2. Add the git dependency:
     scirust-rsi = { git = "https://github.com/Memorithm/scirust", branch = "master" }
   (and scirust-algogen / scirust-synthesis if you want a real code generator).
3. Implement the trait suited to RSI's task (RefineTask by default) where:
     - `score`  = EVALUATOR: compiles/tests the candidate, returns a Fitness
                  (e.g. fraction of passing tests − complexity penalty);
     - `refine` = GENERATOR: produces a critiqued revision of the candidate.
   If no LLM generator is wired, start with a deterministic generator (symbolic
   mutations via scirust-algogen) so everything is testable and reproducible.
4. Drive the loop with an explicit Guard
   (e.g. Guard::new().max_iters(50).patience(8).target(...)) and keep the Report.
5. VERIFICATION (mandatory, do not skip):
     - `cargo build` and `cargo test` pass;
     - a test proves that `report.is_monotone()` is true (non-regression);
     - a test proves termination (iterations ≤ max_iters);
     - run a small example `cargo run --example ...` that shows a Fitness
       improving then stabilizing, and logs the Report.
     - `cargo clippy` clean.
6. Document in RSI's README: how to launch the agent, the safety contract
   (bounds, non-regression, evaluator sandbox, reproducible seed),
   and the fact that all generated code is executed in YOUR sandbox, not by the engine.
7. Commit with clear messages, push the branch, open a PR towards main.
   DO NOT MERGE without my agreement. Give me the PR link and the summary of the
   verifications (actual test outputs).

Safeguards
----------
- Do not generate/execute any code outside a sandbox that YOU control in RSI.
- The loop must remain bounded and elitist: no regression may be adopted. If you
  cannot guarantee it, stop and explain why.
- Report faithfully: if a test fails, show the output; do not claim it is green
  if it is not.
```
