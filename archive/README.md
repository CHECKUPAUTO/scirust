# archive/ — code removed from the compilation tree

Repository policy: **100 % of the code under `*/src/` is compiled, wired and
tested**. This directory keeps, out of the build, historical sources that do
not satisfy this contract, with their exact state. Nothing here is compiled by
the workspace; everything is recoverable (and the git history is authoritative).

| Origin | Content | State found | To bring it back to life |
|---|---|---|---|
| `scirust-gpu/src/*.rs` (8 files) | WGSL kernels (wgpu), cuBLAS matmul (cudarc), GPU tensor, GPU quant | never declared as `mod`; `wgpu`/`cudarc` dependencies absent from the workspace; the `scirust-core` API has drifted since | add the optional deps, declare the modules behind `cfg(feature)`, realign on the current API, validate against the CPU oracle |
| `scirust-simd/neon.rs` | NEON kernels | abandoned duplicate — the active NEON kernels live in `scirust-simd/src/dispatch.rs` (tested on aarch64) | n/a (prefer dispatch.rs) |
| `scirust-simd/sve.rs` | SVE kernels | uses `sv*` intrinsics unavailable in `std::arch`; does not compile | wait for Rust's SVE stabilization, or switch to inline asm like `scirust_simd::sve::sve_vector_length_elements` |
| `scirust-core/quant/` ("Pillar 5") | int8/int4/bf16 SIMD | unwired draft containing **incorrect** kernels (0x7F mask on signed values, erroneous lane recombination, bf16 sign-bit corruption); duplicates the **validated** int8 path (`scirust-core/src/quantization.rs` + `scirust-runtime` audit binaries) | restart from the validated path; any resumption requires bit-exact equivalence tests against the scalar |

Decision documented in `scirust_complete_audit_report.md` (reliability update):
"everything is wired" compliance + ban on unvalidated duplication.
