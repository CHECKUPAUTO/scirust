# FLAT M50 — product GQA prefill proof

M50 closes the missing product-level attention comparison between SciAgent's previous resident multi-dispatch GQA prefill composition and the FLAT M32 feature-on route currently pinned by SciRust.

The paired paths are the exact production entrypoints:

- legacy: `GpuChain::gqa_attention`, including head-local Q/K RoPE, per-query-head slicing, single-head attention, placement and accumulation;
- FLAT: `WgpuFlatM11Bridge::forward`, including fused Q/K RoPE and native grouped GQA attention.

Both consume the same resident raw projected Q/K/V values on one SciRust-owned WGPU context. Upload and readback are outside timing. Each timed call includes the output allocation and synchronization observable in the corresponding product route. Correctness is checked by downloading both outputs before timing and requiring elementwise agreement within the established FLAT tolerance.

The physical qualification covers SciAgent's native `q_heads=8`, `kv_heads=2` geometry, sequence lengths 128 and 512, head dimensions 64 and 128, and causal/non-causal attention. It uses 3 warmups and 12 measured repeats with alternating execution order after 300 seconds of continuous Thor idleness under the `/dev/nvidia0` exclusion contract.

The benchmark emits `performance_claim=none`. A bounded product claim may be recorded only after clean exact-head Thor/Vulkan output is inspected. A candidate result must not be generalized to other devices, shapes or end-to-end model latency.

SciRust remains pinned to FLAT revision `31a33f5e7193dda5ab777c079154ec5ee49ddf4b`. This slice changes no product route, dependency, fallback or sovereignty boundary. Host integration remains Rust-native and GPU execution remains WGPU/WGSL, with no mandatory C/C++, C ABI bridge, CUDA C++/`nvcc`, vendor SDK, WMMA/WGMMA, CUTLASS or cuDNN.
