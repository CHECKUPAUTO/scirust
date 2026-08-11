# FLAT M31 — SciRust WGPU adapter

SciRust consumes FLAT through an opt-in WGPU adapter while retaining ownership of the device, queue, resident matrices and synchronization boundary.

## Stable contract

`WgpuFlatStableAdapter` validates native GQA/MQA geometry and attention configuration through `flat_attention::api::v1` before entering the existing qualified resident training bridge. The concrete SciRust `GpuMatrix` handles remain resident and are passed directly to FLAT's caller-owned WGPU pipelines; the versioned API does not require a host copy.

The current grouped training contract is self-attention (`query_len == kv_len == seq_len`). FLAT's stable shape supports asymmetric query/KV lengths, but this adapter does not claim an asymmetric training capability until the SciRust training bridge implements it.

## Ownership and dispatch

For a supported forward→backward training request:

- Q/K/V/dO remain SciRust-owned device buffers;
- FLAT forward and backward are recorded into one caller-owned command encoder;
- one queue submission is performed by the convenience `forward_backward` boundary;
- O/LSE/dQ/dK/dV remain resident;
- intermediate packing uses device-to-device copies only;
- there is no host readback of Q/K/V/O in the bridge.

The forward path itself is FLAT's fused grouped-attention dispatch and does not materialize the old SciRust NxN score/probability matrices. The backward remains FLAT's correctness-first recomputation kernel.

## Fallback policy

`FlatStableFallbackPolicy::ReturnError` is deliberate. The opt-in stable adapter never silently switches to the legacy SciRust multi-dispatch attention implementation. A higher-level runtime may choose an explicit fallback after receiving the error, but that policy is outside this adapter.

## Performance evidence

M28's paired SciRust benchmark separately compares the legacy resident multi-dispatch MHA composition with FLAT under the same WGPU device and correctness oracle. Software Vulkan results are qualification evidence, not a universal speedup claim. Hardware-specific promotion remains benchmark-gated.

## Sovereignty

The integration remains Rust-native host code plus WGPU/WGSL. It introduces no project-authored C/C++, C ABI bridge, mandatory CUDA C++/`nvcc`, WMMA/WGMMA, CUTLASS, cuDNN, or vendor SDK.
