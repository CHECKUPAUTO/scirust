# FLAT M28 — SciRust resident baseline

This slice measures SciRust's existing resident WGPU attention composition against FLAT's public fused grouped-forward pipeline on the same `WgpuContext` and the same resident Q/K/V buffers.

The SciRust baseline is the existing public composition:

1. resident `Q·Kᵀ` GEMM;
2. resident scale + optional causal mask;
3. resident row softmax;
4. resident probability·V GEMM.

Each primitive uses the production SciRust WGPU implementation and therefore submits its own command buffer. The FLAT path records one fused grouped-forward dispatch. The paired benchmark covers MHA (`batch=1`, one query head and one KV head), causal and non-causal modes. GQA/MQA are not attributed to the SciRust naive baseline because that public baseline does not provide an equivalent native grouped-head contract.

Both paths are correctness-gated against `forward_reference_grouped` before timing. H2D upload and D2H readback are excluded from timed regions. The benchmark reports a fresh-output FLAT scope and a reused-output FLAT scope separately because buffer-allocation policy is part of the observable public-path contract.

Run:

```bash
cargo run --release --locked -p scirust-gpu \
  --features flat-attention --example flat_m28_naive_vs_fused
```

Environment overrides:

- `SCIRUST_M28_SEQ_LEN`
- `SCIRUST_M28_HEAD_DIM`
- `SCIRUST_M28_WARMUPS`
- `SCIRUST_M28_REPEATS`

The CSV includes median/p95 timings, paired ratios and parity evidence. `performance_claim=none` is emitted deliberately: the numbers are benchmark evidence for the measured adapter and scope, not a universal speedup claim.

The sovereignty boundary is unchanged: Rust-native host code and WGPU/WGSL only; no project-authored C/C++ or C ABI bridge and no mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN or vendor SDK.
