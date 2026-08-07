# Elastic Latent KV — Phase 11 runtime completion

## Why this follow-up exists

The original Phase 11 merge implemented a deterministic fixed-capacity lifecycle controller. It tracked logical positions, basis versions and HOT/WARM/COLD temperatures and emitted `LifecycleAction` values describing target compression. The initial Phase 13 runtime consumed those actions only as transition counters: its Phase 8 backend still used one homogeneous representation, and decode rejected the next token once `capacity_tokens` was reached.

This follow-up completes the runtime contract requested for Phase 11: **material deterministic recompression on tier transitions plus sliding-window eviction with fixed allocation**.

## Row migration primitives

`ResidualQuantizedLatentKvCache` now exposes two allocation-stable operations:

- `reconstruct_token_into(token, key, value)` reconstructs one resident sparse-residual latent row into caller-owned dense buffers;
- `remove_oldest()` shifts the active encoded prefix left and reuses the already allocated tail slot.

The operations cover coefficient payloads, quantization scales, residual indices, residual payloads and residual scales. No vector grows during reconstruction or removal.

Reconstruction is intentionally used only at lifecycle migration boundaries. Normal Phase 8 single-tier attention remains reconstruction-free.

## Physical HOT/WARM/COLD backend

`TieredResidualLatentBackend` owns up to three independently encoded Phase 8 caches:

- **HOT**: `LifecycleConfig::hot_tokens`, HOT formats and base rank divided by the HOT rank divisor;
- **WARM**: `LifecycleConfig::warm_tokens`, WARM formats and rank divisor;
- **COLD**: the remaining capacity, COLD formats and rank divisor.

Residual slots are capped by both the selected Phase 9 channel plan and the target tier's `maximum_residual_slots`.

When a tier is full:

1. the oldest row of the source tier is reconstructed into fixed migration scratch;
2. room is recursively established in the next colder tier;
3. the source row is removed without releasing/reallocating storage;
4. the dense scratch row is encoded with the target tier's rank, coefficient format, residual format and residual-slot cap;
5. if COLD is already full, its oldest row is evicted before reuse.

This makes the Phase 11 compression target materially observable in stored rank/format, not merely in metadata.

## Global attention semantics

A tiered cache must not independently softmax HOT, WARM and COLD and combine three context vectors. That would change the attention distribution.

The lifecycle backend therefore evaluates one global softmax over all resident rows in chronological order:

`COLD oldest→newest, WARM oldest→newest, HOT oldest→newest`.

Keys/values are reconstructed one resident row at a time into preallocated scratch. Scores are stored in one preallocated capacity-sized vector; the denominator is global; value accumulation uses the same global weights.

This correctness-first Phase 11 path can later receive a reconstruction-free multi-tier kernel without changing its externally visible semantics.

## Budget accounting

The Phase 13 runtime now checks the actual tiered layout rather than assuming the homogeneous Phase 9 estimate. For each head it:

1. selects a deterministic Phase 9 plan;
2. materializes the configured lifecycle tiers;
3. compares the sum of tier-specific Phase 8 persistent estimates with that head's strict budget;
4. deterministically reduces the planner budget and retries when duplicated tier bases / target formats would exceed the global head budget;
5. fails with the existing `BudgetInfeasible` policy error if no feasible plan remains.

`allocated_ceiling_bytes` is checked against all tier cache allocations plus fixed global-attention and migration scratch owned by the lifecycle backend.

Quality telemetry is recomputed for the physically active lifecycle tiers using the same Phase 9 rank/residual profile and format-retention constants. It no longer reports the homogeneous plan quality as if colder tiers had identical precision.

## Sliding-window runtime

`ElasticLatentDecodeRuntime::decode_step` no longer rejects a token merely because the resident window reached `capacity_tokens`.

On every step:

- the material tiered backend migrates/recompresses/evicts as needed before appending;
- attention sees at most `capacity_tokens` most recent rows;
- `LatentKvLifecycle::admit` performs the corresponding logical ring eviction;
- lifecycle actions still provide deterministic transition telemetry;
- `last_lifecycle_evictions` and `total_lifecycle_evictions` expose logical eviction counts.

The legacy `CapacityExhausted` error variant is retained for source compatibility, but normal lifecycle-backed decode no longer emits it.

## Fixed-allocation guarantee

The Phase 11 implementation preallocates:

- every non-empty tier cache at its final window capacity;
- one global score vector of length `capacity_tokens`;
- one dense key scratch and one dense value scratch for attention;
- one dense key scratch and one dense value scratch for migration.

Tests append well beyond capacity and assert that `packed_bytes()` / runtime `allocated_bytes` remain unchanged while evictions increase.

The existing `AttentionBackend::attention` contract still returns an owned `Vec<f32>` and the Phase 13 projection path also returns owned vectors. This completion therefore claims **zero persistent allocation growth**, not zero transient allocations for the entire legacy numeric decode API.

## Validation targets

The dedicated read-only CI gate validates:

```bash
cargo +nightly-2026-07-02 fmt --all -- --check
cargo +nightly-2026-07-02 clippy -p scirust-core --all-targets --locked -- -D warnings
cargo +1.89.0 check -p scirust-core --all-targets --locked
cargo +1.89.0 test -p scirust-core reconstruction_and_oldest_removal_preserve_fixed_allocation --locked
cargo +1.89.0 test -p scirust-core tiered_latent_kv_backend::tests --locked
cargo +1.89.0 test -p scirust-core elastic_latent_runtime::tests --locked
cargo +1.89.0 run --quiet --locked -p scirust-core --example latent_kv_phase13
```

The full repository CI remains the merge gate after the targeted validation is green.
