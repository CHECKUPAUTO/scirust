//! Unified compute-backend abstraction targeted by the `#[gpu]` macro.
//!
//! ## Honesty policy (repo-wide)
//! Code under `*/src/` is wired and tested and never claims a capability it
//! does not have.
//!
//! - **CPU reference backend** ([`CpuBackend`]) — always built, deterministic,
//!   oracle-grade GEMM. This is the bit-tolerant oracle a GPU result is
//!   validated against.
//! - **Portable GPU** ([`WgpuBackend`]) — real WGSL compute path behind the
//!   `wgpu` feature (Vulkan/Metal/DX12/GL).
//! - **CUDA** ([`CudaBackend`]) — placeholder until a GPU CI runner exists.
//! - **Deterministic compute** — Kahan summation, INT8 quantized GEMM (bit-exact
//!   via integer arithmetic), and fixed-order accumulation.
//! - **Kernel library** — tiled 16×16 SGEMM, fused GEMM+bias+activation,
//!   extended activations (gelu, silu, sigmoid, tanh, elu, softplus, etc.),
//!   deterministic reductions with Kahan compensation, INT8 GEMM.
//! - **Operations** — CPU reference ops (activations, LayerNorm, RMSNorm,
//!   reductions) for oracle validation.
//! - **Fusion engine** — compile GEMM → bias → activation sequences into a
//!   single GPU dispatch.
//! - **VRAM-resident tensor** — `GpuTensor` for device-resident autograd.
//! - **GPU im2col/col2im** — keep Conv2d chains entirely in VRAM.
//!
//! ## Determinism guarantee
//!
//! SciRust's GPU determinism is built on three strategies:
//! 1. **Integer arithmetic** (INT8 → INT32 accumulation) — mathematically exact.
//! 2. **Kahan compensated summation** for FP32 when integer paths aren't suitable.
//! 3. **Fixed dispatch ordering** — reproducible accumulation sequences.
//!
//! The CPU oracle (`CpuBackend`) remains the bit-exact reference forever.

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec, vec::Vec};

#[cfg(feature = "wgpu")]
mod chain;
#[cfg(feature = "wgpu")]
mod conv_gpu;
#[cfg(feature = "wgpu")]
pub mod deterministic;
#[cfg(feature = "wgpu")]
mod deterministic_gpu;
#[cfg(feature = "wgpu")]
mod engine;
#[cfg(feature = "wgpu")]
mod fusion;
#[cfg(feature = "wgpu")]
pub mod kernels;
#[cfg(feature = "wgpu")]
pub mod ops;
#[cfg(feature = "wgpu")]
mod tensor;
#[cfg(feature = "wgpu")]
mod wgpu_backend;

#[cfg(feature = "wgpu")]
pub use chain::{BlockCache, BlockWeights, GpuChain, ModelWeights};
#[cfg(feature = "wgpu")]
pub use conv_gpu::{COL2IM_WGSL, IM2COL_WGSL, cpu_col2im, cpu_im2col};
#[cfg(feature = "wgpu")]
pub use deterministic_gpu::{DeterministicGpu, DeterministicValidator};
#[cfg(feature = "wgpu")]
pub use engine::WgpuEngine;
#[cfg(feature = "wgpu")]
pub use fusion::{FusedLayer, FusionNode, plan_fusion};
#[cfg(feature = "wgpu")]
pub use tensor::GpuTensor;
#[cfg(feature = "wgpu")]
pub use wgpu_backend::{GpuMatrix, WgpuContext, wgpu_scale_causal_mask, wgpu_softmax};

/// Error returned when a compute backend cannot service a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    /// The requested hardware backend is not wired in this build (see P2.2).
    Unavailable(&'static str),
    /// Operand dimensions are inconsistent for the requested operation.
    ShapeMismatch(String),
}

impl core::fmt::Display for BackendError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self
        {
            BackendError::Unavailable(name) => write!(
                f,
                "compute backend `{name}` is not wired in this build (roadmap P2.2)"
            ),
            BackendError::ShapeMismatch(msg) => write!(f, "shape mismatch: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BackendError {}

/// Result specialised for backend operations.
pub type BackendResult<T> = Result<T, BackendError>;

/// Hardware abstraction targeted by the `#[gpu]` macro.
///
/// `gemm_f32` computes the row-major product `C(m×n) = A(m×k) · B(k×n)`.
/// Implementations must return an honest [`BackendError`] rather than
/// fabricated data when they cannot perform the operation.
pub trait RawComputeBackend {
    /// Stable identifier for the backend (e.g. `"cpu"`, `"wgpu"`).
    fn device_name(&self) -> &'static str;
    /// Row-major GEMM `C = A · B`.
    fn gemm_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> BackendResult<Vec<f32>>;
}

/// Validate that `a` and `b` hold exactly the elements an `m×k · k×n` GEMM needs.
fn check_gemm_dims(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> BackendResult<()> {
    if a.len() != m * k
    {
        return Err(BackendError::ShapeMismatch(format!(
            "A has {} elements, expected m*k = {}*{} = {}",
            a.len(),
            m,
            k,
            m * k
        )));
    }
    if b.len() != k * n
    {
        return Err(BackendError::ShapeMismatch(format!(
            "B has {} elements, expected k*n = {}*{} = {}",
            b.len(),
            k,
            n,
            k * n
        )));
    }
    Ok(())
}

/// CPU reference backend — always available, deterministic, oracle-grade.
pub struct CpuBackend;

impl RawComputeBackend for CpuBackend {
    fn device_name(&self) -> &'static str {
        "cpu"
    }

    fn gemm_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> BackendResult<Vec<f32>> {
        check_gemm_dims(a, b, m, k, n)?;
        let mut out = vec![0.0f32; m * n];
        for i in 0..m
        {
            for j in 0..n
            {
                let mut acc = 0.0f32;
                for p in 0..k
                {
                    acc += a[i * k + p] * b[p * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        Ok(out)
    }
}

/// Portable GPU backend (wgpu, Vulkan/Metal/DX12/GL).
pub struct WgpuBackend;

impl RawComputeBackend for WgpuBackend {
    fn device_name(&self) -> &'static str {
        "wgpu"
    }

    fn gemm_f32(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> BackendResult<Vec<f32>> {
        #[cfg(feature = "wgpu")]
        {
            check_gemm_dims(a, b, m, k, n)?;
            wgpu_backend::wgpu_gemm(a, b, m, k, n)
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = (a, b, m, k, n);
            Err(BackendError::Unavailable("wgpu"))
        }
    }
}

/// CUDA/cuBLAS backend placeholder.
pub struct CudaBackend;

impl RawComputeBackend for CudaBackend {
    fn device_name(&self) -> &'static str {
        "cuda"
    }

    fn gemm_f32(
        &self,
        _a: &[f32],
        _b: &[f32],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> BackendResult<Vec<f32>> {
        Err(BackendError::Unavailable("cuda"))
    }
}

/// Transparent hardware dispatcher.
pub enum GpuAccelerator {
    Cpu(CpuBackend),
    Wgpu(WgpuBackend),
    Cuda(CudaBackend),
}

impl GpuAccelerator {
    pub fn cpu() -> Self {
        GpuAccelerator::Cpu(CpuBackend)
    }

    pub fn device_name(&self) -> &'static str {
        match self
        {
            GpuAccelerator::Cpu(b) => b.device_name(),
            GpuAccelerator::Wgpu(b) => b.device_name(),
            GpuAccelerator::Cuda(b) => b.device_name(),
        }
    }

    pub fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> BackendResult<Vec<f32>> {
        match self
        {
            GpuAccelerator::Cpu(backend) => backend.gemm_f32(a, b, m, k, n),
            GpuAccelerator::Wgpu(backend) => backend.gemm_f32(a, b, m, k, n),
            GpuAccelerator::Cuda(backend) => backend.gemm_f32(a, b, m, k, n),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_gemm_matches_hand_computed_oracle() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = CpuBackend.gemm_f32(&a, &b, 2, 3, 2).unwrap();
        assert_eq!(c, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn cpu_gemm_identity_is_passthrough() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let id = [1.0, 0.0, 0.0, 1.0];
        assert_eq!(CpuBackend.gemm_f32(&a, &id, 2, 2, 2).unwrap(), a.to_vec());
    }

    #[test]
    fn cpu_gemm_is_bit_deterministic() {
        let a: Vec<f32> = (0..12).map(|i| (i as f32).sin()).collect();
        let b: Vec<f32> = (0..12).map(|i| (i as f32).cos()).collect();
        let first = CpuBackend.gemm_f32(&a, &b, 3, 4, 3).unwrap();
        let second = CpuBackend.gemm_f32(&a, &b, 3, 4, 3).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn shape_mismatch_is_reported() {
        let err = CpuBackend
            .gemm_f32(&[1.0, 2.0], &[1.0], 2, 2, 1)
            .unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    #[test]
    fn device_backends_are_honest_not_fake() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        assert_eq!(
            CudaBackend.gemm_f32(&a, &b, 2, 2, 2),
            Err(BackendError::Unavailable("cuda"))
        );
        #[cfg(not(feature = "wgpu"))]
        assert_eq!(
            WgpuBackend.gemm_f32(&a, &b, 2, 2, 2),
            Err(BackendError::Unavailable("wgpu"))
        );
    }

    #[test]
    fn accelerator_dispatches_and_reports_device() {
        let cpu = GpuAccelerator::cpu();
        assert_eq!(cpu.device_name(), "cpu");
        let a = [1.0, 2.0, 3.0, 4.0];
        let id = [1.0, 0.0, 0.0, 1.0];
        assert_eq!(cpu.matmul(&a, &id, 2, 2, 2).unwrap(), a.to_vec());
    }
}
