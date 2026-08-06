# Phase 4 specification

## Objective

Add deterministic online rank adaptation above the Phase 3 strict-budget
planner. The controller consumes one planner proposal per budget observation
and decides whether to retain or replace the active key/value rank pair.

## Transition policy

The controller applies transitions in this order:

1. the first valid proposal initializes the active plan;
2. a plan that no longer fits the current strict budget is replaced immediately;
3. a quality-compliant proposal immediately replaces a non-compliant active plan;
4. every other rank change requires `confirmation_steps` consecutive identical
   rank-pair proposals;
5. observing the active rank pair clears any pending proposal.

The policy never retains a representation whose persistent bytes exceed the
current strict budget.

## Determinism

Pending state contains only the proposed key/value rank pair and its consecutive
observation count. Equal inputs produce equal transitions, reasons, counters and
FNV-1a step fingerprints. No wall clock, random scheduler or platform service is
consulted.

## Trace

Each step records:

- current budget and quality scale;
- raw Phase 3 proposal;
- active plan after hysteresis;
- transition reason;
- pending confirmation count;
- whether a proposal was suppressed;
- active reconstruction and attention quality;
- stable step fingerprint.

## Invariants

1. Key and value ranks remain independent.
2. The active plan always fits the current strict persistent budget.
3. Budget-forced downgrades are immediate.
4. Quality recovery is immediate.
5. Discretionary changes require consecutive confirmation.
6. Returning to the active rank pair clears pending state.
7. Repeated timeline execution is bit-deterministic on the same target and build.
8. No `unsafe` code or external dependency is introduced.

## Acceptance gates

- `cargo +nightly-2026-07-02 fmt --all -- --check`;
- `cargo +1.89.0 clippy --all-targets -- -D warnings`;
- all debug, release and documentation tests pass;
- the harness emits four timelines, 52 steps and 31 CSV columns;
- two release harness runs are byte-identical;
- every active step remains within budget;
- the suite contains forced, recovery, confirmed and suppressed transitions;
- exact timelines finish at their intrinsic ranks;
- no production SciRust crate is modified.

## Deliberate non-goals

Phase 4 does not implement:

- basis retraining from a live token stream;
- per-token ranks;
- token eviction or cache tiering;
- scalar or product quantization;
- residual/outlier channels;
- GPU or SIMD kernels;
- production integration.
