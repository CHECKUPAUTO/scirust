# SciRust Runtime Feature Discovery

`scirust-runtime-discovery` generates deterministic, typed hypotheses for new
runtime-policy features. It does **not** declare a feature scientifically valid.
Acceptance remains the responsibility of the existing causal, sequential, GP,
NSGA-II, symbolic-regression, ablation, and untouched-confirmation pipeline.

## Design rules

- No task outcome may be available to a runtime feature.
- No future generation state may be read.
- Every hypothesis declares its source signals, temporal availability, runtime
  cost, expected failure mode, and ablation group.
- Missing instrumentation is reported as a rejected hypothesis; it is never
  silently invented.
- Catalog generation is deterministic and byte-stable after JSON formatting.
- Historical holdout failures may motivate hypotheses, but those prompts may not
  be reused to select or tune the replacement policy.

## Generate the Elastic Cache catalog

```bash
cargo +nightly-2026-07-02 run \
  --manifest-path scirust-runtime-discovery/Cargo.toml \
  --bin runtime-feature-discovery \
  -- \
  --request scirust-runtime-discovery/examples/elastic_cache_request.json \
  --output /tmp/elastic-cache-feature-catalog.json
```

## Intended integration with SciAgent

SciAgent is an hypothesis proposer, not a scientific oracle. A future
`runtime-feature-discovery` tool will provide SciAgent with:

1. anonymized development false positives and matched safe neighbours;
2. the current runtime signal registry;
3. the typed schema in this crate;
4. a strict instruction to return only JSON hypotheses.

The Rust validator then rejects leakage, non-determinism, unavailable signals,
and invalid cost declarations before any instrumentation or experiment is run.

## Scientific lifecycle

1. Generate a proposal catalog from development evidence only.
2. Select instrumentable hypotheses under an explicit compute budget.
3. Add deterministic instrumentation to the model runtime.
4. Recollect atomic counterfactual branches on new development prompts.
5. Run causal discovery, CRF, GP, NSGA-II, and symbolic ablations.
6. Freeze the feature set and policy.
7. Evaluate once on a new untouched confirmation set.

A negative result remains a valid result; the runtime must stay fail-closed.
