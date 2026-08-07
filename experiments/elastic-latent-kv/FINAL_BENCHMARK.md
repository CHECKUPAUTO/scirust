# Elastic Latent KV — Final Benchmark Contract

This benchmark closes the Phase 7–13 implementation sequence with one reproducible comparison harness. It is intentionally precise about what is and is not measured.

## Compared implementations

The harness compares four public production-facing cache paths from `scirust-core`:

1. `PlainKvCache`: dense FP32 reference;
2. `LatentQuantizedBackend`: Phase 7, fixed half-rank INT4 latent coefficients;
3. `ResidualLatentQuantizedBackend`: Phase 8, fixed half-rank INT4 coefficients plus two INT4 sparse residual slots;
4. `AdaptiveResidualLatentBackend`: the Phase 9 strict-budget policy materialized through the validated Phase 8 backend.

Phases 10–13 remain covered by their dedicated deterministic validation harnesses. The final benchmark focuses on the cache representation and attention-layer cost that can be compared fairly across all four implementations without inventing a model-level tokenizer, sampler, vocabulary projection, or device-resident KV path that SciRust does not yet expose.

## Canonical contexts

The default context lengths are:

- 1,024 tokens;
- 4,096 tokens;
- 16,384 tokens;
- 32,768 tokens.

The benchmark reserves one additional cache position for the timed decode token.

## Measurement method

For each context and implementation:

1. construct a fresh fixed-capacity backend set for all attention heads;
2. deterministically prefill cache K/V vectors directly, outside the decode timer;
3. time one real `kv_backend::decode_step` for the next token, including q/k/v/o projections and backend attention;
4. repeat from a fresh cache and report the median duration;
5. compare the numerical output with the dense reference from the same context;
6. verify that every repeated numerical output has the same FNV fingerprint.

Direct K/V prefill keeps setup O(context) instead of executing every historical attention query and turning a 32K cache benchmark into an O(context²) trajectory benchmark. The reported `cache_prefill_*` fields therefore measure cache ingestion only. They are not model prefill throughput.

## Reported fields

The CSV includes:

- actual cache allocation reported through `AttentionBackend::packed_bytes()`;
- adaptive planned persistent bytes;
- compression ratio and memory saved versus dense;
- cache-prefill duration and cache-ingest tokens/s;
- `attention_ttft_proxy_ns`: cold one-token attention-layer decode latency at context zero;
- long-context decode latency for one token;
- attention-layer tokens/s, equal to one timed decode divided by its median latency;
- maximum absolute output error against dense;
- deterministic output status and fingerprint;
- adaptive selected-plan quality and fingerprint.

`attention_ttft_proxy_ns` is deliberately named a proxy. It is not end-to-end model TTFT. Likewise, `attention_tokens_per_second` is not full-model generation throughput. End-to-end TTFT, logits error and model tokens/s require a complete model-level inference harness and model weights; this benchmark does not fabricate those values.

## Determinism

Timing values are expected to vary with host load. Numerical output is required to remain deterministic. CI therefore validates fingerprints and structural invariants rather than requiring byte-identical timing CSV files.

## Running the canonical benchmark

```bash
cargo +1.89.0 run --quiet --release --locked -p scirust-core --example elastic_latent_kv_benchmark > elastic-latent-kv-final.csv
```

The defaults run the canonical 1K/4K/16K/32K contexts with three repetitions per implementation.

Environment controls:

```bash
SCIRUST_ELASTIC_KV_BENCH_CONTEXTS=1024,4096,16384,32768 \
SCIRUST_ELASTIC_KV_BENCH_REPEATS=5 \
SCIRUST_ELASTIC_KV_BENCH_BUDGET_BPS=3000 \
cargo +1.89.0 run --quiet --release --locked -p scirust-core \
  --example elastic_latent_kv_benchmark > elastic-latent-kv-final.csv
```

`SCIRUST_ELASTIC_KV_BENCH_BUDGET_BPS` is the adaptive per-head target budget as basis points of the corresponding dense payload. The planner still enforces the minimum feasible representation when basis overhead dominates very short contexts.

## CI smoke validation

CI uses short contexts and two repetitions. Its purpose is to prove:

- pinned formatting and Clippy cleanliness;
- Rust 1.89 compatibility;
- successful construction of all four implementations;
- positive timing samples;
- exact row structure;
- deterministic numerical fingerprints;
- dense memory accounting equality;
- compressed allocations below dense on the smoke contexts;
- adaptive planned bytes never exceeding adaptive allocated bytes;
- non-zero adaptive quality and plan fingerprints.

CI smoke timing is not published as a performance claim.
