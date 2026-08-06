# Phase 3 specification

## Objective

Turn fixed-rank projection into a deterministic budget planner that selects
separate key and value ranks under a strict persistent-storage limit.

## Persistent budget

The budget includes:

- per-token key coefficients;
- per-token value coefficients;
- shared key basis;
- shared value basis.

It excludes transient attention scratch because that storage is shared by
executions and does not grow with the persistent cache representation.

For token count `T`, dense dimension `D`, key rank `Rk` and value rank `Rv`:

```text
latent_bytes = 4 * (T + D) * (Rk + Rv)
dense_bytes  = 8 * T * D
```

## Deterministic selection

Every nested rank pair is evaluated in ascending key-rank/value-rank order.
Candidates over budget are discarded. Selection is lexicographic:

1. candidates satisfying all quality targets are preferred;
2. among satisfying candidates, minimum persistent bytes wins;
3. remaining ties use attention error, reconstruction error and ranks;
4. when no candidate satisfies the guard, the minimum worst target ratio wins.

The report never represents a failed quality guard as success.

## Pareto frontier

A candidate is dominated when another budget-feasible candidate is no worse in
persistent bytes, key reconstruction RMS, value reconstruction RMS and
attention maximum absolute error, and is strictly better in at least one of
those objectives.

## Invariants

1. Key and value ranks remain independent.
2. The selected representation never exceeds the strict byte budget.
3. Rank-one/rank-one storage is the explicit minimum accepted budget.
4. Quality success requires both reconstruction targets and the attention guard.
5. Candidate ordering and tie breaking are deterministic.
6. The Pareto frontier contains only non-dominated candidates.
7. Repeated suite execution is bit-deterministic on the same target and build.
8. No `unsafe` code or external dependency is introduced.

## Acceptance gates

- `cargo +nightly-2026-07-02 fmt --all -- --check`;
- `cargo +1.89.0 clippy --all-targets -- -D warnings`;
- all debug, release and documentation tests pass;
- the harness emits 12 scenarios and 31 CSV columns;
- two release harness runs are byte-identical;
- four exact low-rank scenarios recover their intrinsic ranks;
- every selected candidate is within budget;
- quality-guard failures remain explicit;
- no production SciRust crate is modified.

## Deliberate non-goals

Phase 3 does not implement:

- adaptive per-token rank;
- token eviction or cache tiering;
- product or scalar quantization;
- residual/outlier channels;
- GPU or SIMD kernels;
- production integration.
