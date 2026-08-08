# Elastic Latent KV — Phase 24: exact parallel bounded top-k prototype

## Status

Phase 24 adds an **additive** WGPU sampler prototype that parallelizes only the bounded top-k candidate comparisons. The Phase 22 `WgpuDeterministicSampler` remains the production/oracle WGPU path until this prototype proves both exact parity and a material benefit under the Phase 23 benchmark contract.

## Design

`WgpuParallelTopKSampler` launches one workgroup of 64 lanes. The constructor specializes the WGSL source for the exact configured `top_k`, currently restricted to `2 <= K < V` and `K <= 256`.

For each selected rank:

1. each lane scans a strided subset of the remaining order positions;
2. each lane chooses its local best candidate using the total order `(probability descending, token id ascending)`;
3. a fixed 64-lane workgroup reduction chooses the global best candidate;
4. lane 0 performs the same selection-sort swap as the Phase 22 oracle;
5. a workgroup barrier makes the swap visible before the next rank.

This preserves the exact first-K prefix of the sequential selection sort.

## Determinism boundary

Only comparison-only work is parallelized. Floating-point reductions that can depend on accumulation order remain scalar on lane 0:

- top-p normalization sum;
- top-p cumulative prefix;
- final categorical probability sum;
- final categorical token-id scan;
- PCG state transition and random draw.

The maximum-logit reduction is parallel because it uses only `max`, not floating-point addition. Exact ties in candidate selection continue to prefer the lower token id.

All workgroup barriers are reached uniformly. The resident `enabled` flag suppresses useful work without changing barrier control flow.

## Scope and limitations

The prototype requires:

- a WGPU adapter supporting at least 64 invocations per workgroup;
- finite positive temperature;
- finite `top_p` in `0..=1`;
- `2 <= top_k < vocab_size`;
- `top_k <= 256`.

Greedy, unbounded top-k, and larger K continue to use the Phase 22 sampler. Phase 24 does not replace or silently redirect the production sampler.

## Validation

The permanent gate requires:

- nightly-2026-07-02 rustfmt;
- strict Clippy with warnings denied;
- exact Rust 1.89.0 compatibility;
- Mesa lavapipe execution of the Phase 24 parallel sampler tests;
- Mesa lavapipe execution of the Phase 22 sequential sampler suite as the oracle regression.

The Phase 24 tests prove:

1. exact seeded token-stream equality against the CPU sampler;
2. exact equality against `WgpuDeterministicSampler` on the same logits/config/seed;
3. exact lower-token-id tie behavior under combined top-k + top-p;
4. exact reset/replay without changing resident allocation;
5. the workgroup-lane and ranking-pass observability contract.

## Performance policy

Phase 24 makes **no speedup claim**. The new sampler is deliberately additive so Phase 23 can compare the same public `sample()` boundary for sequential and parallel implementations on a real adapter. If the parallel workgroup is not materially beneficial, it will not be promoted into the Phase 21 device-feedback path.

## Non-goals

Phase 24 does not:

- parallelize probability summation or PCG state;
- change the CPU sampling API or semantics;
- integrate the parallel sampler into Phase 21 generation yet;
- add CUDA sampler parity;
- claim lavapipe timing as GPU performance;
- support arbitrary K beyond 256;
- use multi-workgroup global synchronization.
