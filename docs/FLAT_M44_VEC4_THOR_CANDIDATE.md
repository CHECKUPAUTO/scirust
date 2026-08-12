# FLAT M44 vec4 physical-Thor candidate evidence

This evidence gate evaluates the opt-in FLAT `Q4Vec4Mha` grouped-forward candidate without changing SciRust's product FLAT dependency pin.

The candidate is compiled in a temporary Cargo project against exact FLAT commit `bf69a1e1328769a1d1e26b3f2f0f813aa5c0b62c`. The benchmark uses one SciRust-owned WGPU context and compares the existing SciRust resident multi-dispatch attention path, FLAT portable grouped forward with prepared bindings, and FLAT vec4 MHA with prepared bindings. Upload and readback are outside the timed region.

Physical evidence is accepted only when the adapter is `NVIDIA Tegra NVIDIA Thor` through Vulkan, the GPU has been continuously idle for 300 seconds, no `cuda_pretrain` process appears during the measured window, post-run compute occupancy is empty, and both FLAT variants match the scalar grouped oracle for O and LSE. The measured matrix is sequence length 128/512, head dimension 64/128, causal and non-causal, with three warmups and twelve measured iterations whose execution order rotates among all three paths.

This gate reports `performance_claim=none`. A default-route or release-speed claim requires reviewing the exact physical results first. Previous Thor timing runs made before SciAgent exclusion was enforceable remain diagnostic history only and are not promotion evidence.
