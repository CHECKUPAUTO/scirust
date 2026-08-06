# Elastic Latent KV — Phase 5

## Scope

Phase 5 introduces a deterministic fixed-slot sparse residual channel above the fixed-rank latent key/value representation established by Phases 0–4.

The phase remains an isolated CPU research experiment. It does not add quantization, GPU kernels, per-layer integration, model checkpoints, production serving code or `unsafe` blocks.

## Representation

For every token:

- latent key coefficients store the projection onto a shared key basis;
- latent value coefficients store the projection onto a shared value basis;
- up to `s_k` key residual coordinates are stored as `(u16 index, f32 value)` pairs;
- up to `s_v` value residual coordinates are stored in the same format.

Residual coordinates are selected by descending absolute residual magnitude. Equal magnitudes are resolved by the lowest dense coordinate index. Empty reserved slots use `u16::MAX` and zero.

## Reconstruction-free attention

For a query `q`, token key coefficients `a_i`, key basis `B_k` and sparse key residual `r_i^k`:

```text
score_i = (B_k^T q) · a_i + q · r_i^k
```

For value coefficients `b_i`, value basis `B_v`, normalized weights `w_i` and sparse value residual `r_i^v`:

```text
output = B_v (Σ_i w_i b_i) + Σ_i w_i r_i^v
```

No dense key or value vector is reconstructed on the attention path.

## Persistent storage accounting

The planner accounts exactly for:

- latent coefficients: `tokens × (r_k + r_v) × 4` bytes;
- shared bases: `dimension × (r_k + r_v) × 4` bytes;
- residual indices: `tokens × (s_k + s_v) × 2` bytes;
- residual values: `tokens × (s_k + s_v) × 4` bytes.

Scratch buffers and generated reports are not persistent cache payload.

## Planner

The planner enumerates every tuple:

```text
(r_k, r_v, s_k, s_v)
```

inside configured maxima. Candidates exceeding the strict byte budget are rejected before numerical evaluation.

Candidate ordering is deterministic:

1. quality-compliant candidates precede non-compliant candidates;
2. compliant candidates minimize persistent bytes, then total rank/slot complexity;
3. non-compliant candidates minimize the worst normalized quality-target ratio;
4. stable rank/slot tuple ordering resolves remaining ties.

A zero-residual baseline is selected independently from candidates where `s_k = s_v = 0` under the same budget.

## Deterministic corpus

The standard suite contains 12 scenarios:

- dimensions 16, 32 and 64;
- one pure low-rank case per dimension;
- three structured sparse-residual cases per dimension;
- four queries and 20 cached tokens per case;
- exact reconstruction and attention targets;
- deterministic seeds and stable FNV-1a output fingerprints.

Structured residual coordinates are placed outside the maximum latent basis prefix, so the residual channel is measured independently from adding more shared basis vectors.

## Gates

- `cargo +nightly-2026-07-02 fmt --all -- --check`
- `cargo +1.89.0 clippy --all-targets -- -D warnings`
- `cargo +1.89.0 test --all-targets -- --test-threads=1`
- `cargo +1.89.0 test --all-targets --release -- --test-threads=1`
- `cargo +1.89.0 test --doc`
- two byte-identical release harness executions;
- 12 report rows and 41 stable CSV columns;
- every selected candidate remains under budget;
- every standard selected candidate satisfies all exact quality guards;
- at least nine scenarios strictly improve the zero-residual baseline.
