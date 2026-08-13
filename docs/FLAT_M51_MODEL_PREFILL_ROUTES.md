# FLAT M51 — model-level prefill route ablation

M50 proves that the current SciAgent causal GQA attention route is faster than
the previous resident multi-dispatch composition on the measured physical Thor
matrix. M51 tests whether that kernel-level gain survives the complete resident
model prefill and whether SciRust can remove duplicated K rotary work.

The three routes run on the same `ResidentModel`, weights and WGPU context:

- `Legacy`: SciRust head-local Q/K RoPE plus multi-dispatch GQA;
- `FlatRawK`: the current M32 product route, with Q/K RoPE fused by FLAT;
- `FlatPreRotatedKReuse`: SciRust rotates K once, passes the resulting resident
  buffer to FLAT's pre-rotated-K entry point, and appends that same buffer to the
  KV cache.

The candidate is not the default. `FlatRawK` remains the product default while
M51 is evidence-only. The benchmark covers prompts 128 and 512 with the M33
model geometry (`d_model=512`, 8 layers, GQA 8/2, `d_ff=1408`). It runs complete
prefill through projections, attention, KV-cache construction, residuals, MLPs,
final normalization and the tied vocabulary head. The last-position logits of
both FLAT routes must agree with the legacy route before timing.

Qualification requires the persistent physical Thor runners, Vulkan, the
shared `/dev/nvidia0` exclusion object, 300 seconds of continuous idleness and
continuous SciAgent contamination monitoring. Uploading the model is outside
timing; every route includes its own request-local KV-cache allocation and the
public model-level work it performs.

The benchmark emits `performance_claim=none`. No route may be promoted unless
clean exact-head Thor evidence shows a repeatable improvement in the measured
scope. The implementation stays Rust-native with WGPU/WGSL and adds no C/C++,
C ABI, vendor SDK, CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS or cuDNN requirement.
