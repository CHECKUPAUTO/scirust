//! # scirust-cuda — Route B (CUDA + Tensor cores) backend for the Jetson Thor
//!
//! The mixed-precision counterpart to `scirust-gpu`'s portable wgpu path. Where
//! Route A (fp32 WGSL) is kernel-efficiency bound at **<5 % of the Thor's peak**
//! (measured: ~34 tok/s training / ~3.5 tok/s 350M decode — see
//! `scirust-sciagent/JETSON_THOR.md`), this backend runs the GEMMs in **bf16 on
//! Blackwell Tensor cores** with fp32 accumulation. The **B0 feasibility gate**
//! measured **12.9×** the fp32 GFLOP/s for exactly these matmuls on the Thor
//! (39.1 vs 3.0 TFLOP/s), which is why this crate exists — see
//! `scirust-sciagent/ROUTE_B.md` for the full design and phased plan.
//!
//! ## Precision
//!
//! bf16 inputs, **fp32 accumulate** (cuBLASLt `CUBLAS_COMPUTE_32F`), fp32 master
//! weights held by the optimizer. bf16 over fp16 because its exponent range
//! matches fp32 — **no loss scaling** needed. Results are therefore **not**
//! bit-identical to the CPU/fp32 reference; they agree within a bf16-appropriate
//! relative tolerance (~`5e-2`), and every op is checked against the CPU oracle
//! brick by brick, exactly as Route A was.
//!
//! ## Build gating
//!
//! Everything is behind the **`cuda`** feature. With it off, this crate pulls no
//! CUDA dependency and compiles anywhere (CI included); the public surface is just
//! [`available`]. With it on, it needs the CUDA toolkit — i.e. the Thor — to build
//! and run. This mirrors how Route A's GPU tests skip without an adapter, one step
//! further: Route B can't even be *built* without CUDA, so its tests are Thor-only.

#[cfg(feature = "cuda")]
mod chain;
#[cfg(feature = "cuda")]
pub use chain::{CudaChain, CudaF32, CudaMatrix};

/// Whether this build includes the CUDA backend (the `cuda` feature). `false`
/// builds carry no CUDA dependency and expose no device API.
pub const fn available() -> bool {
    cfg!(feature = "cuda")
}
