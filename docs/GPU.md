# GPU — status and roadmap

> **Honest status (today): GPU compute is _not_ wired into the build.**
> This page documents what actually exists, why hardware GPU is not yet a
> claimed capability, and what re-wiring it requires. It used to describe a
> one-line GPU API (`GpuContext`, `ConvGpuPipelines`, `set_global_gpu_context`,
> `Conv2d::on_gpu`); that API is **archived, not compiled** — keeping the doc
> as-is would have been an overclaim, which the project does not do.

## What exists today

- **`scirust-gpu` ships one real, tested path**: a deterministic CPU reference
  backend (`CpuBackend`) with a fixed accumulation order, exposed through the
  `RawComputeBackend` trait and the `GpuAccelerator` dispatcher. It is the
  bit-tolerant oracle any future GPU backend must be validated against.
- The `WgpuBackend` / `CudaBackend` placeholders return
  `BackendError::Unavailable` — they **never fabricate results**.
- `scirust-core` routes all compute through CPU/SIMD kernels
  (AVX2/SSE2/NEON, runtime-dispatched), which are the tested production path.
- `--features wgpu` currently compiles nothing extra (the feature is a
  reserved switch, not an implementation).

```rust
use scirust_gpu::{GpuAccelerator, BackendError};

let acc = GpuAccelerator::cpu();                 // the wired, tested path
let c = acc.matmul(&a, &b, m, k, n)?;            // real GEMM, deterministic

// Device paths are honest about not being wired yet:
let gpu = GpuAccelerator::Wgpu(scirust_gpu::WgpuBackend);
assert!(matches!(gpu.matmul(&a, &b, m, k, n), Err(BackendError::Unavailable("wgpu"))));
```

## Why GPU is not claimed yet

The project rule is **no claim without a test**. A truthful "GPU accelerated"
claim needs all of:

1. **A CI runner that can execute it.** wgpu compute needs a Vulkan/Metal/DX12
   adapter (or a software Vulkan ICD such as Mesa *lavapipe*). The standard
   hosted runners — and the dev container — have none, so a wgpu path cannot
   be tested here.
2. **A determinism story.** GPU floating-point is not bit-identical to the CPU
   path across drivers; the project's bit-exact guarantee requires a
   documented, bit-*tolerant* CPU oracle for any GPU result. `CpuBackend` is
   that oracle.
3. **A supply-chain decision.** `wgpu` pulls a large transitive tree
   (`wgpu-hal`, `naga`, …) that must clear `cargo deny` and is weighed against
   the "pure Rust, minimal, auditable" posture.

Until those hold, shipping a wgpu backend would only reproduce the dishonest
stub that was just removed.

## Re-wiring plan (roadmap P2.2)

See [`docs/INDUSTRIAL_ROADMAP.md`](INDUSTRIAL_ROADMAP.md) §P2.2. The intended
path is **wgpu only** (portable; testable in CI via a software Vulkan adapter),
with the archived WGSL kernels in [`archive/scirust-gpu/`](../archive/scirust-gpu/)
re-aligned to the current `scirust-core` API and validated against `CpuBackend`.
CUDA/cuBLAS stays out of scope until a GPU runner exists.

The archived sources include working drafts of: a `saxpy`/`relu` WGSL compute
path (`wgpu_backend.rs`), a GPU tensor/context, im2col-based Conv2d pipelines,
and a cuBLAS matmul — preserved for reference, not built.

## Historical result (not reproducible from this build)

A cuBLAS-backed BF16 matmul once reached ~63 TFLOPS on an NVIDIA Jetson Thor
(aarch64), validated against a CPU oracle. This is a **historical measurement**
from the archived code, not a current capability — see
`scirust_complete_audit_report.md` §5.
