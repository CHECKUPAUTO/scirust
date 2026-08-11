# FLAT M32 — SciAgent resident prefill integration

M32 routes SciAgent's opt-in resident prompt prefill attention through the already-qualified FLAT M11 native GQA/MQA grouped-forward pipeline.

## Execution boundary

With `scirust-sciagent/flat-attention` enabled, each prefill layer keeps projected Q/K/V in SciRust-owned WGPU buffers and passes those buffers directly to `WgpuFlatM11Bridge::forward`. FLAT performs fused Q/K RoPE plus causal grouped attention and returns the context as a resident `GpuMatrix`; SciRust then continues with the existing output projection, residual, MLP, cache seeding and final LM head.

With the feature disabled, the existing `GpuChain::gqa_attention` path is unchanged and remains the explicit oracle/fallback. There is no silent fallback inside the FLAT feature path.

The prompt K cache is still seeded from SciRust's resident `rope_heads` result because M15 decode consumes pre-rotated K. This is a device-to-device resident operation and does not introduce a host Q/K/V round-trip.

## Correctness gate

`flat_runtime_decode` now exercises the complete M32 prefill + M15 decode route and compares generated tokens against the existing whole-sequence resident greedy path on fixed prompts. The WGPU CI job runs this parity test on lavapipe.

## Performance evidence

M32 makes no new speedup claim. Existing M28 paired resident-attention benchmarks remain low-level evidence for legacy multi-dispatch versus FLAT. A model-level prefill latency claim must be backed by a paired benchmark on the same real adapter, exact commit, model geometry, prompt length, warm-up count and measurement protocol before any promotion statement is made.

## Sovereignty

Rust-native host code plus WGPU/WGSL only. No project-authored C/C++, C ABI bridge, mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN, or vendor SDK is introduced.
