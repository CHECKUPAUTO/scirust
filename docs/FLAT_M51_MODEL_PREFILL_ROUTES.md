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

The candidate was never the default. `FlatRawK` remained the product route while
M51 was evidence-only. The benchmark covered prompts 128 and 512 with the M33
model geometry (`d_model=512`, 8 layers, GQA 8/2, `d_ff=1408`). It runs complete
prefill through projections, attention, KV-cache construction, residuals, MLPs,
final normalization and the tied vocabulary head. The last-position logits of
both FLAT routes must agree with the legacy route before timing.

Qualification requires the persistent physical Thor runners, Vulkan, the
shared `/dev/nvidia0` exclusion object, 300 seconds of continuous idleness and
continuous SciAgent contamination monitoring. Uploading the model is outside
timing; every route includes its own request-local KV-cache allocation and the
public model-level work it performs.

## Physical Thor result and retention decision

Exact SciRust head `93a54de3b699d1b040d5d1e8d94afb3755531b1f`
passed all parity and physical-device gates. Full-logit relative errors were
`8.37e-6` for prompt 128 and `1.13e-5` for prompt 512.

The current product route beat the legacy model-level prefill in both measured
cases:

- prompt 128: 50.407171 ms versus 73.733820 ms (`1.462731x`);
- prompt 512: 254.610215 ms versus 258.952283 ms (`1.017054x`).

Pre-rotated-K reuse did not improve either case. It took 52.004822 ms at prompt
128 (`current/candidate=0.969279`) and 254.627142 ms at prompt 512
(`current/candidate=0.999934`). The candidate is therefore rejected and its
route selector, benchmark executable and active workflow are removed after the
evidence merge. The merged commit retains complete reproducibility in history.

The result is bounded to this physical device, model and workload. The broad
marker remains `performance_claim=none`. The retained implementation stays
Rust-native with WGPU/WGSL and adds no C/C++, C ABI, vendor SDK, CUDA
C++/`nvcc`, WMMA/WGMMA, CUTLASS or cuDNN requirement.
