# FLAT M53 — asymmetric vec4 product-attention candidate

M53 evaluates FLAT-ATTENTION PR #86's opt-in asymmetric vec4 kernel through
SciRust's real resident M11 bridge. The candidate is not the product default.

The benchmark compares three routes on one WGPU context and the same resident
Q/K/V buffers: SciRust's previous multi-dispatch GQA composition, the current
portable FLAT product route, and the M53 vec4 FLAT route. Upload and readback
are outside timing. Every row checks portable output against legacy and vec4
output against portable before emitting measurements.

The fixed matrix is GQA 8/2, sequence lengths 128 and 512, D64 and D128,
causal and non-causal attention, 3 warmups, and 12 rotated-order repeats. The
workflow accepts only a persistent physical Thor runner, locks `/dev/nvidia0`,
requires a continuous five-minute idle window, rejects foreign compute during
timing, and verifies empty post-run occupancy.

Immutable candidate inputs:

- SciRust base: `a8c4739bc3a657cf9ee18e0defa7e3c88c09d456`;
- FLAT candidate: `43b4c0ba08e109ac7025a01a01837da6927d05d0`;
- FLAT PR: #86.

No result is accepted until the exact SciRust PR head and workflow run are
recorded here. The candidate may be promoted only if the clean physical data
improves the current portable route without losing parity, followed by a
full-model SciAgent prefill qualification. Otherwise it will be removed.

The route remains Rust-native WGPU/WGSL with no mandatory C/C++, C ABI, CUDA
C++, vendor SDK, or vendor-specific shader extension. `performance_claim=none`
remains in force.
