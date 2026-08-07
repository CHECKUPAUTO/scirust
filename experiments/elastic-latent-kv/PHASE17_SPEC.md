# Elastic Latent KV — Phase 17: WGPU-resident Transformer encoder

## Status

Phase 17 extends the Phase 16 device-resident boundary from one pre-LN Transformer block to one complete incremental `TransformerEncoder`.

## Scope

For a single decode token, the WGPU runtime keeps the following work on device in one deterministic dispatch:

- every encoder block's LayerNorm 1;
- Q/K/V projections;
- per-layer, per-head Elastic Latent KV append;
- fixed-order latent score, stable softmax and value aggregation;
- dense head reconstruction and O projection;
- both residual paths;
- LayerNorm 2;
- FFN1 + ReLU + FFN2;
- the encoder's final LayerNorm.

The persistent state contains separate latent K/V rings for every encoder layer while reusing one scratch region because layers execute sequentially.

## Host/device boundary

Per incremental token the host transfers exactly:

1. one `d_model` FP32 input row to WGPU;
2. one `d_model` FP32 final encoder row back to the host.

There is no host readback or re-upload between encoder blocks.

The existing compute adapter may still allocate launch metadata, bind groups or command encoders per dispatch. Phase 17 therefore does **not** claim zero host allocations per launch.

## Determinism baseline

The Phase 17 kernel intentionally uses one compute invocation and fixed loop ordering. This is a correctness and reproducibility baseline, not a throughput claim.

## Validation

The dedicated Phase 17 gate requires:

- repository rustfmt on `nightly-2026-07-02`;
- strict Clippy with `-D warnings` and the `wgpu` feature;
- exact MSRV Rust `1.89.0` check;
- Mesa lavapipe execution of the Phase 17 integration tests.

The integration suite covers:

- full-rank parity against legacy two-layer `TransformerEncoder::infer_step`, including final LayerNorm;
- lower-rank sliding-window wrap against an independent CPU latent encoder oracle;
- out-of-order position rejection and reset without persistent allocation growth.

## Non-goals

Phase 17 does not yet provide:

- embeddings or positional encoding on device;
- a device-resident LM/classification output head;
- CUDA-resident encoder parity;
- throughput-oriented parallel kernels;
- quantized INT8/INT4 resident tiers;
- HOT/WARM/COLD migration on device.
