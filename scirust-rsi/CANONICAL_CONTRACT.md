# Canonical `scirust-rsi` contract

`Memorithm/scirust/scirust-rsi` is the **canonical implementation** of SciRust's bounded self-improvement primitives.

Downstream Memorithm repositories, including `Memorithm/RSI`, must consume this implementation from an **exact reviewed SciRust commit** rather than maintaining a behaviorally independent copy. Compatibility adapters may exist downstream, but semantic fixes to the engine belong here first.

## Stable downstream contract

The contract intentionally stays small:

- `Fitness = f64`, with larger values always better;
- `Guard` supplies hard loop bounds (`max_iters`) plus optional `patience`, `target`, `min_delta` and time budget;
- `refine::RefineTask` supplies `initial`, `score` and `refine`;
- `refine::SelfRefiner::new(seed).run(task, guard)` returns the kept solution and an auditable `Report`;
- `Report::history` is the **best-so-far incumbent after each executed iteration**;
- `Report::is_monotone()` means that this kept incumbent history never decreases;
- a worse candidate that is evaluated and rejected does **not** make the report non-monotone;
- for a deterministic task/evaluator, the same seed and configuration reproduce the same run.

The integration tests in `tests/canonical_contract.rs` freeze these semantics explicitly for downstream consumers.

## Ownership boundary with RSI

This crate provides bounded search/refinement algorithms over data structures and scalar objectives. Its sandbox claim is correspondingly narrow: the core algorithms do not execute generated host code or mutate repositories.

`Memorithm/RSI` owns the separate empirical engineering loop that can materialize source-code candidates and invoke build/test/benchmark evaluators. That real-code execution must retain its own isolation, allowlists, COGNO hard gates, resource bounds and audit trail. Consuming `scirust-rsi` does not weaken or replace those controls.

## Compatibility discipline

A downstream migration must:

1. choose an already reviewed and green SciRust commit;
2. pin that exact immutable revision;
3. run the downstream feature/integration tests against it;
4. record the exact SciRust revision in the cross-repository compatibility set;
5. update the pin only through another reviewed downstream change.

Moving branch names such as `master` are development inputs, not qualified compatibility identifiers.

## Change policy

Changes that alter any of the stable semantics above require:

- focused tests in this crate;
- an explicit compatibility note for downstream consumers;
- green SciRust CI on the exact final PR head before merge;
- downstream pin updates only after the upstream merge SHA is known.

This policy keeps one source of truth while allowing RSI, SciAgent and other systems to build richer orchestration around it.