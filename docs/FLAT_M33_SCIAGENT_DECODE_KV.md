# FLAT M33 — SciAgent resident decode/KV qualification

M33 qualifies SciAgent's already-wired FLAT decode path as an end-to-end resident KV-cache integration rather than introducing a second decode implementation.

## Runtime contract

With `scirust-sciagent/flat-attention` enabled:

- prompt prefill uses the M32 resident grouped-forward route;
- each layer stores RoPE-rotated K and raw V in fixed-capacity SciRust-owned `WgpuDenseKvCache` allocations;
- each subsequent token uses `query_len = 1` and the existing FLAT M15 pre-rotated-K path;
- Q, active K, active V and attention context remain WGPU-resident across the attention boundary;
- the existing KV prefix is not rebuilt on the host and no host K/V round-trip is introduced into the token loop;
- the final vocabulary logits retain the existing SciRust host-visible sampling boundary.

The feature-off path remains the explicit SciRust reference/fallback. FLAT itself does not silently select that path.

## Cache lifecycle and EOS

`generate_cached` allocates fresh per-layer caches for every generation request, so replay after completion starts from an empty logical cache without retaining hidden state from the previous request.

M33 adds `generate_cached_until_eos`, which uses the same M32 prefill + M15 decode path but terminates immediately after a generated token matches the caller-provided EOS set. An empty EOS set is required to be behaviorally identical to ordinary `generate_cached`.

`flat_m33_lifecycle` verifies deterministic replay, fresh-cache reset semantics, immediate EOS termination and the empty-EOS equivalence contract on a WGPU adapter.

## Real-device benchmark gate

`flat_m33_thor_bench` is the M33 model-level measurement harness. It runs through the public resident generation API and records:

- exact SciRust source revision;
- WGPU adapter name;
- model geometry and GQA head geometry;
- prompt length and generated-token count;
- warm-up and measured repeat counts;
- median prompt-prefill latency;
- median and p95 incremental decode latency per token;
- decode tokens/s.

The benchmark uses paired fresh-cache runs. `generate_cached(prompt, 1)` measures prompt prefill plus first-token selection; `generate_cached(prompt, N)` measures the same prefill followed by `N - 1` incremental decode forwards. The per-pair difference is divided by the actual decode-forward count. Execution order alternates to reduce systematic ordering bias.

The repository's self-hosted Jetson Thor workflow runs the lifecycle parity test and this benchmark on the real ARM64 Thor WGPU/Vulkan adapter. Results from software Vulkan remain correctness/qualification evidence only and must not be promoted as physical-GPU performance.

The harness emits `performance_claim=none`; measured values become a performance claim only when the exact real-device log/artifact and commit are cited.

## Sovereignty

M33 remains Rust-native host code plus WGPU/WGSL. It introduces no project-authored C/C++, C ABI bridge, mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN or vendor SDK. The Thor workflow may inspect machine state, but FLAT's build and execution path does not depend on CUDA SDK components.
