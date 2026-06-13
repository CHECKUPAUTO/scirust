# GPU — status and roadmap

> **Status: a real, portable GPU compute path is wired behind the `wgpu`
> feature and tested against the CPU oracle.** It is *not* on by default.
> This page documents exactly what exists, how it is tested, and what remains.
> (An earlier version of this page described a one-line `Conv2d::on_gpu` API
> that did not exist — that archived API is not compiled; the doc was corrected
> rather than left as an overclaim.)

## What exists today

- **CPU reference backend** (`CpuBackend`, always built): a deterministic,
  fixed-accumulation-order GEMM exposed through the `RawComputeBackend` trait
  and the `GpuAccelerator` dispatcher. It is the bit-tolerant oracle the GPU
  path is validated against.
- **Portable GPU GEMM via wgpu** (`WgpuBackend`, feature `wgpu`): a real general
  WGSL compute shader (`C = alpha·op(A)·op(B) + beta·C`, with optional
  transposes, row-major `f32`) executed on a Vulkan/Metal/DX12/GL adapter.
  Validated against `CpuBackend` within a documented floating-point tolerance
  (GPU accumulation order is not bit-identical to the scalar path).
- **Autograd-tape integration** (`WgpuEngine`, feature `wgpu`): `WgpuEngine`
  implements the tape's `GpuEngine` hook. Attach it with
  `Tape::with_gpu_engine(WgpuEngine::new().unwrap())`, and `Var::matmul_gpu`
  runs **both its forward and backward GEMMs on the GPU** (the backward's
  `dA = g·Bᵀ` and `dB = Aᵀ·g` use the transpose path). The device + pipeline
  are created once and reused across the pass. Validated end to end against the
  CPU tape (forward + both gradients) on lavapipe.
- **CUDA** (`CudaBackend`): out of scope until a GPU CI runner exists; always
  returns `BackendError::Unavailable`. The archived cuBLAS draft is kept in
  `archive/scirust-gpu/`.
- Default training/inference still routes through the CPU/SIMD kernels
  (AVX2/SSE2/NEON); the GPU path is opt-in (feature + an attached engine),
  keeping the bit-exact default guarantee intact.
- **Conv2d on the GPU**: when a `WgpuEngine` is attached to the tape, the
  Conv2d im2col GEMMs — forward `W·col`, backward `dW = dout·colᵀ` and
  `dInput = Wᵀ·dout` — run on the engine (validated end to end against the CPU
  Conv2d on lavapipe). im2col/col2im themselves stay on the CPU for now.
- **VRAM-resident matmul chains** (`GpuChain`): keep an intermediate activation
  in GPU memory across a sequence of GEMMs — upload inputs once, chain
  `matmul`s on `GpuMatrix` handles, download only the final result. The
  intermediate never round-trips to the CPU. Oracle-validated on lavapipe.

```rust
use scirust_gpu::GpuChain;

let chain = GpuChain::new().expect("a Vulkan adapter");
let a = chain.upload(&a_data, 2, 3);
let b = chain.upload(&b_data, 3, 2);
let c = chain.upload(&c_data, 2, 4);
let t = chain.matmul(&a, &b)?;   // T = A·B stays in VRAM …
let out = chain.matmul(&t, &c)?; // … and feeds the next GEMM directly
let result = chain.download(&out)?; // only the final matrix is downloaded
```

```rust
use scirust_gpu::{GpuAccelerator, BackendError, WgpuBackend, RawComputeBackend};

// Always-available, deterministic CPU reference path:
let acc = GpuAccelerator::cpu();
let c = acc.matmul(&a, &b, m, k, n)?;            // real GEMM, bit-deterministic

// Standalone portable GPU GEMM (requires `--features wgpu` and an adapter):
match WgpuBackend.gemm_f32(&a, &b, m, k, n) {
    Ok(gpu_c) => { /* validated against CpuBackend within tolerance */ }
    Err(BackendError::Unavailable("wgpu")) => { /* no feature / no adapter — honest */ }
    Err(e) => return Err(e),
}
```

GPU-accelerated autograd (feature `wgpu`):

```rust
use scirust_core::autodiff::reverse::Tape;
use scirust_gpu::WgpuEngine;

let tape = match WgpuEngine::new() {
    Some(engine) => Tape::new().with_gpu_engine(engine),  // forward + backward on GPU
    None => Tape::new(),                                   // no adapter → CPU tape
};
let a = tape.input(/* … */);
let b = tape.input(/* … */);
let c = a.matmul_gpu(b);   // GEMM on the GPU engine when attached, else CPU
let loss = c.sum();
tape.backward(loss.idx()); // dA = g·Bᵀ, dB = Aᵀ·g also on the GPU
```

## How it's tested (no claim without a test)

The wgpu GEMM is exercised in CI on a **software Vulkan adapter** — Mesa
*lavapipe* (`llvmpipe`) — so the assertion path runs without physical GPU
hardware:

```bash
sudo apt-get install -y mesa-vulkan-drivers vulkan-tools   # provides lavapipe
cargo test -p scirust-gpu --features wgpu
```

The tests compare the wgpu result to `CpuBackend` with a relative-Frobenius
tolerance (`< 1e-4`). If no adapter can be acquired, the GPU tests skip rather
than fail; CI installs lavapipe so they actually execute. The CI job is
`GPU (wgpu / lavapipe)` in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

## Determinism note

GPU floating-point is not bit-identical to the scalar CPU path (different
accumulation order, possible FMA), so the bit-exact guarantee does **not**
extend to the GPU backend. `CpuBackend` is the documented bit-tolerant oracle;
GPU output is asserted equal to it within tolerance, not bit-for-bit.

## Supply chain

Enabling `wgpu` pulls a larger transitive tree (`wgpu-hal`, `naga`, `ash`, …).
That tree clears `cargo deny` (advisories, licences, bans, sources) as part of
CI. The dependency is **optional** — default builds and the standard gates do
not compile it.

## Roadmap (P2.2 and beyond)

See [`docs/INDUSTRIAL_ROADMAP.md`](INDUSTRIAL_ROADMAP.md) §P2.2. Done: portable
wgpu GEMM (oracle-validated via lavapipe), its plumbing into the autograd tape
(`WgpuEngine` + `Var::matmul_gpu` forward/backward), Conv2d's im2col GEMMs, and
a VRAM-resident matmul-chain API (`GpuChain`). Next:

- **Transparent tape residency**: make `matmul_gpu`/Conv2d on the tape keep
  intermediates resident automatically. This requires the tape's value storage
  (`DeviceTensor`, today a CPU `Tensor`) to become lazily-materialised GPU
  storage and the rest of the forward op-set (bias, activations) to be
  device-resident — a larger change, payoff only on real GPU hardware.
- Move im2col/col2im onto the GPU — the archived pipelines in
  [`archive/scirust-gpu/`](../archive/scirust-gpu/) are the reference.
- More ops (elementwise, reductions) behind the same trait.
- CUDA/cuBLAS only once a hardware GPU runner exists.

## Historical result (not reproducible from this build)

A cuBLAS-backed BF16 matmul once reached ~63 TFLOPS on an NVIDIA Jetson Thor
(aarch64), validated against a CPU oracle. This is a **historical measurement**
from the archived code, not a current capability — see
`scirust_complete_audit_report.md` §5.
