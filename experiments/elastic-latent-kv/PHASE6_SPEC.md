# Phase 6 specification — deterministic latent and residual quantization

## Objective

Phase 6 quantizes the persistent latent coefficients and sparse residual values
introduced by the earlier Elastic Latent KV phases. It compares an FP32
reference with deterministic row-wise symmetric INT8 and packed INT4 formats,
while preserving a strict persistent byte budget.

## Representation

Each key/value side contains:

1. an FP32 identity-prefix basis used as the deterministic reference basis;
2. one coefficient row per cached token;
3. zero or more sparse residual `(u16 index, value)` pairs per token;
4. one symmetric scale per quantized row;
5. either FP32 payload bytes, signed INT8 payload bytes, or two signed INT4
   values packed into each byte.

Zero-valued rows use a deterministic scale of `1.0` and all-zero payloads.
INT4 uses the signed range `[-7, 7]`; the unused `-8` code is never emitted.

## Planning

The planner enumerates independent formats for:

- key coefficients;
- value coefficients;
- key residual values;
- value residual values.

For components with zero columns, only FP32/empty storage is considered. Every
candidate accounts for:

- FP32 basis bytes;
- coefficient payloads;
- per-row scales;
- packed INT4 bytes;
- residual `u16` indices;
- residual value payloads.

Candidates exceeding the strict byte budget are rejected. Among candidates
meeting all quality guards, the planner selects the smallest storage footprint,
then the smallest worst normalized target ratio, then a stable format tuple.
If no quality-feasible candidate exists, the lowest normalized-error candidate
is retained for diagnostics.

## Quality gates

Each candidate is measured against the FP32 latent-plus-residual oracle using:

- key reconstruction relative RMS;
- value reconstruction relative RMS;
- maximum absolute attention-output error.

The standard suite contains 12 deterministic scenarios over dimensions 16, 32
and 64. Three exact scenarios require FP32 equality. Nine structured scenarios
exercise INT8 and INT4 coefficient/residual combinations under strict budgets.

## Non-goals

Phase 6 does not implement:

- learned quantization scales;
- stochastic rounding;
- per-channel basis quantization;
- production GPU/SIMD kernels;
- live model integration;
- cache eviction or HOT/WARM/COLD tiering.
