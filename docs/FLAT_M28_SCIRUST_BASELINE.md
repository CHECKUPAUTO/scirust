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

The CSV includes the selected WGPU adapter/backend, median/p95 timings, paired ratios and parity evidence. `performance_claim=none` is emitted deliberately: the numbers are benchmark evidence for the measured adapter and scope, not a universal speedup claim.

## Physical Thor release-evidence gate

`.github/workflows/flat-m28-thor-release-evidence.yml` runs the paired benchmark on the self-hosted physical Jetson Thor under the shared SciRust GPU lock. The idle observation and benchmark execute under the same lock boundary. An unreadable compute-occupancy query fails closed; a known detached production `cuda_pretrain` workload is never killed and is allowed to finish naturally while the workflow reserves the next idle boundary. Unknown GPU compute activity fails the qualification.

The workflow checks out and verifies the exact source revision being reported, requires the physical NVIDIA Thor to be visible through Vulkan, forces the Vulkan WGPU backend, and rejects benchmark output unless the selected WGPU adapter reports `NVIDIA Tegra NVIDIA Thor` and the context reports the actual backend as `Vulkan`. NVIDIA inventory alone is not treated as proof that the measured WGPU work executed on Thor.

The release-evidence sweep covers `seq_len` 128 and 512 with `head_dim` 64 and 128; each benchmark invocation measures both causal and non-causal attention with 3 warmups and 9 timed repeats. The same correctness oracle must pass before any timing row is emitted.

This workflow is evidence collection, not a performance assertion. A release-level improvement statement may be promoted only after the exact-head output has been inspected and accepted for the explicitly measured workload(s). Any run collected while Thor occupancy is unverified or while another compute workload is active is rejected as release evidence. Software-adapter timing does not substitute for this physical-device evidence.

## Accepted physical Thor evidence (2026-08-19)

GitHub Actions run `32307039698` on exact SciRust head `08f7cfe0b3b8e8d384d8a1082927dff0c208d41c` (PR #1283, FLAT pinned to `43b4c0ba`, benchmark using the qualified vec4 MHA pipeline) executed successfully on the persistent physical Thor runner, adapter `NVIDIA Tegra NVIDIA Thor`, backend Vulkan. The full protocol was enforced: exact-head checkout, compile before GPU reservation, GPU lock with independent contention proof, Thor/Vulkan identity verification, idle/cooldown window, contamination watchdog, alternating three-path rotation, oracle correctness gate before timing, and empty post-run occupancy.

Recorded rows (MHA batch=1 heads=1, 3 warmups, 9 repeats, medians in microseconds):

| seq | dim | causal | naive multi-dispatch µs | FLAT fresh µs | FLAT reused µs | naive / FLAT fresh | naive parity max abs | FLAT parity max abs |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|
| 128 | 64 | no | 614.177 | 1656.725 | 1668.029 | 0.370718 | 0.00000101 | 0.00000036 |
| 128 | 64 | yes | 613.565 | 1658.649 | 1659.056 | 0.369919 | 0.00000048 | 0.00000036 |
| 128 | 128 | no | 767.075 | 1795.122 | 1787.974 | 0.427311 | 0.00000077 | 0.00000042 |
| 128 | 128 | yes | 637.464 | 1409.010 | 1394.252 | 0.452420 | 0.00000072 | 0.00000036 |
| 512 | 64 | no | 968.741 | 2738.882 | 2757.189 | 0.353699 | 0.00000179 | 0.00000060 |
| 512 | 64 | yes | 952.955 | 1873.936 | 1878.974 | 0.508531 | 0.00000137 | 0.00000060 |
| 512 | 128 | no | 1487.919 | 3169.659 | 3182.216 | 0.469426 | 0.00000167 | 0.00000072 |
| 512 | 128 | yes | 1430.353 | 1993.530 | 2037.510 | 0.717498 | 0.00000125 | 0.00000060 |

**Negative result.** The FLAT fused grouped-forward pipeline — measured on its qualified vec4 MHA path — is slower than the SciRust naive multi-dispatch baseline in all eight measured MHA rows (naive/FLAT ratios 0.35–0.72). Correctness parity is excellent (FLAT parity errors are lower than naive in every row), but the roadmap 1.0 performance gate — measured improvement over SciRust's previous multi-dispatch attention for supported target workloads — is **not satisfied** by the FLAT grouped-forward pipeline for these MHA prefill geometries on physical Thor.

This evidence identifies the bottleneck: the FLAT Q4 tiled kernel's workgroup overhead at MHA heads=1 (64 invocations per 4×8 tile) cannot beat the naive per-row GEMM composition for small-head-count prefill. FLAT's measured advantages remain in GQA geometries (K/V amortization) and decode: the M48 decode candidate (1.17x–1.22x over M15) and M53 asymmetric prefill vec4 (1.09x–1.18x over portable) are separate, device/workload-scoped evidence and do not close this MHA gate. `performance_claim=none` remains in force; this negative result is retained as required by the Phase O retention rule.

The sovereignty boundary is unchanged: Rust-native host code and WGPU/WGSL only; no project-authored C/C++ or C ABI bridge and no mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN or vendor SDK.
