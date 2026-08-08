# Elastic Latent KV — Phase 22: bounded deterministic top-k ranking

## Status

Phase 22 removes the quadratic ranking cost from the common bounded top-k path of the Phase 19 deterministic WGPU sampler while preserving the exact CPU sampling contract used by Phases 20 and 21.

## Problem

Phase 21 eliminates the generated-token host data dependency, but its sampler still inherits the Phase 19 correctness baseline: one WGPU invocation performs a full selection-sort ranking of the vocabulary for every non-greedy sample.

For vocabulary size `V`, that baseline performs `V` selection passes and O(V²) probability comparisons even when the caller asks to keep only `K << V` tokens.

The CPU contract does not require the rejected `V-K` tokens to be fully sorted. It only requires:

1. the first `K` entries to be the same probability-descending/token-id-ascending prefix as a full sort;
2. every rejected token to be zeroed;
3. top-p to inspect the ranked non-zero prefix in the same order;
4. the categorical draw to scan probabilities in original token-id order;
5. PCG state advancement to remain unchanged.

## Scope

When `0 < top_k < V`, the existing sampler now:

- initialises the same probability and order scratch arrays as Phase 19;
- executes only the first `K` deterministic selection passes;
- leaves the tail unsorted but still as an exact permutation of all rejected token ids;
- zeros that whole tail exactly once;
- runs nucleus selection only across the ranked non-zero prefix;
- performs the same final token-id-order categorical draw with the same emulated PCG state.

The resulting ranking work is O(V·K) rather than O(V²) for bounded top-k.

## Exact fallback

Phase 22 deliberately keeps the original full-ranking behavior when:

- `top_k == 0` (top-k disabled);
- `top_k >= V` (top-k is non-restrictive).

Greedy decoding (`temperature <= 0` or `top_k == 1`) still bypasses ranking and consumes no RNG draw.

This phase does not claim that the one-invocation kernel is throughput-optimal. It removes unnecessary deterministic work before introducing parallel ranking in a later phase.

## Determinism

The ordering contract is unchanged:

- probability descending;
- exact probability ties broken by lower token id;
- top-p keeps the smallest ranked prefix reaching the requested mass;
- the random draw consumes exactly one PCG word for each stochastic sample;
- rejected/invalid host inputs do not advance state.

Partial selection is sufficient because every selection-sort swap preserves a permutation in the unsorted tail. After the first `K` passes, positions `[0, K)` are the exact full-sort prefix and `[K, V)` contains every rejected token exactly once.

## Public observability

`WgpuDeterministicSampler` exposes:

- `ranking_passes_per_sample()` — `K` for the bounded path, `V` for full ranking, zero for greedy shortcuts;
- `uses_bounded_top_k_fast_path()` — whether the O(V·K) path is active.

These describe algorithmic work; they are not timing or throughput claims.

## Validation

The permanent Phase 22 gate requires:

- nightly-2026-07-02 rustfmt;
- strict Clippy with `-D warnings`;
- Rust 1.89.0 MSRV check;
- Mesa lavapipe execution of the deterministic sampler integration suite;
- Mesa lavapipe execution of the Phase 21 device-feedback suite as a non-regression proof.

The sampler suite must prove:

1. exact CPU/WGPU seeded temperature parity on the full-ranking fallback;
2. exact CPU/WGPU combined top-k + top-p parity on the bounded path;
3. exact lower-token-id ordering for equal-probability ties at the top-k boundary;
4. `K` ranking passes are reported for bounded top-k;
5. `V` ranking passes remain for disabled/non-restrictive top-k;
6. greedy still performs zero ranking passes and consumes no RNG draw;
7. invalid shapes/non-finite host logits do not advance sampler state.

## Non-goals

Phase 22 does not:

- parallelize selection across WGPU invocations or workgroups;
- change the CPU `SamplingConfig` API;
- change PCG or floating-point sampling semantics;
- optimize pure top-p with `top_k == 0`;
- add CUDA sampling/device-feedback parity;
- quantize model or KV state;
- make a wall-clock speedup claim without a dedicated benchmark.
