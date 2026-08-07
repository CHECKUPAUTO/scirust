# Elastic Latent KV — Phase 18: WGPU-resident greedy MiniLLM

## Status

Phase 18 extends the Phase 17 resident encoder boundary across the complete greedy incremental `MiniLLM` token step.

## Scope

A Phase 18 runtime snapshots immutable inference parameters from `MiniLLM` through a read-only core API, then keeps the following data resident on WGPU:

- token embedding table;
- all Phase 17 encoder weights and per-layer latent KV rings;
- MiniLLM's post-encoder LayerNorm parameters;
- LM-head weights and bias;
- current decode position;
- intermediate hidden rows.

For each token the runtime executes three ordered WGPU dispatches without host-visible intermediate state:

1. embedding lookup plus sinusoidal positional encoding;
2. the complete Phase 17 resident Transformer encoder;
3. MiniLLM final LayerNorm, LM-head projection and greedy argmax.

## Host/device boundary

Per greedy incremental token the host transfers exactly:

- one `u32` token id to WGPU (4 bytes);
- one `u32` greedy next-token id back to the host (4 bytes).

Hidden vectors and logits never round-trip through host memory.

The generic WGPU adapter may still allocate command metadata, bind groups or encoders per dispatch. Phase 18 therefore does not claim zero host allocations per token.

## Greedy semantics

The existing CPU `argmax_row` keeps the later token id when two finite logits compare equal. Phase 18 scans vocabulary ids in ascending order and replaces the current winner on `>=`, preserving the same highest-id tie break.

## Core snapshot contract

`MiniLLM::inference_snapshot()` returns immutable references to the model components needed by an accelerator backend. Internal fields remain private and the training/tape path is unchanged.

## Determinism baseline

All Phase 18 kernels use one invocation and fixed loop ordering. This is a correctness/reproducibility baseline rather than a throughput claim.

## Validation

The dedicated Phase 18 gate requires:

- repository rustfmt on `nightly-2026-07-02`;
- strict Clippy with `-D warnings` and the `wgpu` feature;
- exact MSRV Rust `1.89.0` check;
- Mesa lavapipe execution of the Phase 18 integration tests.

The integration suite covers:

- exact greedy generated-token sequence parity with `MiniLLM::generate_ids_cached` using full-rank identity bases;
- a 4-byte upload / 4-byte download telemetry boundary;
- invalid token and out-of-order position rejection;
- reset without persistent allocation growth.

## Non-goals

Phase 18 does not yet provide:

- temperature, top-k or top-p sampling on device;
- seeded stochastic sampling parity;
- CUDA-resident MiniLLM inference;
- throughput-oriented parallel kernels;
- INT8/INT4 resident model or KV tiers;
- HOT/WARM/COLD device migration;
- tokenizer execution on device.
