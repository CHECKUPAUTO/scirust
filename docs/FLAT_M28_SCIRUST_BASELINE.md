# FLAT M28 — SciRust resident baseline

This slice measures SciRust's existing resident WGPU attention composition against FLAT's public fused grouped-forward pipeline on the same `WgpuContext`. Each path receives its own pre-resident Q/K/V buffers populated from exactly the same deterministic bytes; buffer creation/upload is outside timing. This keeps the comparison on one WGPU device without exposing SciRust's private `GpuMatrix` storage handle solely for benchmarking.

The SciRust baseline is the existing public composition:

1. resident `Q·Kᵀ` GEMM;
2. resident scale + optional causal mask;
3. resident row softmax;
4. resident probability·V GEMM.

Each primitive uses the production SciRust WGPU implementation and therefore submits its own command buffer. The FLAT path records one fused grouped-forward dispatch. The paired benchmark covers MHA (`batch=1`, one query head and one KV head), causal and non-causal modes. GQA/MQA are not attributed to the SciRust naive baseline because that public baseline does not provide an equivalent native grouped-head contract.

Both paths are correctness-gated against `forward_reference_grouped` before timing. H2D upload and D2H readback are excluded from timed regions. The benchmark reports a fresh-output FLAT scope and a reused-output FLAT scope separately because buffer-allocation policy is part of the observable public-path contract.

The measurement loop rotates the three timed paths (SciRust multi-dispatch, FLAT with fresh output, FLAT with reused output) through first/middle/last execution position over complete three-iteration cycles. This avoids structurally assigning one path to the same thermal/cache/order position on every repeat.

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

## Physical Thor release-evidence gate

`.github/workflows/flat-m28-thor-release-evidence.yml` runs the paired benchmark on the self-hosted physical Jetson Thor under the shared SciRust GPU lock. The idle observation and benchmark execute under the same lock boundary. An unreadable compute-occupancy query fails closed; a known detached production `cuda_pretrain` workload is never killed and is allowed to finish naturally while the workflow reserves the next idle boundary. Unknown GPU compute activity fails the qualification.

The workflow checks out and verifies the exact source revision being reported, requires the physical NVIDIA Thor to be visible through Vulkan, forces the Vulkan WGPU backend, and rejects benchmark output unless the selected WGPU adapter reports `NVIDIA Tegra NVIDIA Thor`. NVIDIA inventory alone is not treated as proof that the measured WGPU work executed on Thor.

The release-evidence sweep covers `seq_len` 128 and 512 with `head_dim` 64 and 128; each benchmark invocation measures both causal and non-causal attention with 3 warmups and 9 timed repeats. The same correctness oracle must pass before any timing row is emitted.

This workflow is evidence collection, not a performance assertion. A release-level improvement statement may be promoted only after the exact-head output has been inspected and accepted for the explicitly measured workload(s). Any run collected while Thor occupancy is unverified or while another compute workload is active is rejected as release evidence. Software-adapter timing does not substitute for this physical-device evidence.

The sovereignty boundary is unchanged: Rust-native host code and WGPU/WGSL only; no project-authored C/C++ or C ABI bridge and no mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN or vendor SDK.
