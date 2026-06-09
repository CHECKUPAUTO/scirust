//! Abstraction unifiée pour l'exécution GPU (#[gpu]).

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Trait d'abstraction matériel ciblé par la macro #[gpu].
pub trait RawComputeBackend {
    fn device_name(&self) -> &'static str;
    fn gemm_f32(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32>;
}

pub struct WgpuBackend;
impl RawComputeBackend for WgpuBackend {
    fn device_name(&self) -> &'static str { "wgpu" }
    fn gemm_f32(&self, _a: &[f32], _b: &[f32], m: usize, _k: usize, n: usize) -> Vec<f32> {
        vec![0.0; m * n]
    }
}

pub struct CudaBackend;
impl RawComputeBackend for CudaBackend {
    fn device_name(&self) -> &'static str { "cuda" }
    fn gemm_f32(&self, _a: &[f32], _b: &[f32], m: usize, _k: usize, n: usize) -> Vec<f32> {
        vec![0.0; m * n]
    }
}

/// Dispatcher matériel transparent.
pub enum GpuAccelerator {
    Wgpu(WgpuBackend),
    Cuda(CudaBackend),
    CpuFallback,
}

impl GpuAccelerator {
    pub fn matmul(&self, a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        match self {
            GpuAccelerator::Wgpu(backend) => backend.gemm_f32(a, b, m, k, n),
            GpuAccelerator::Cuda(backend) => backend.gemm_f32(a, b, m, k, n),
            GpuAccelerator::CpuFallback => {
                let mut out = vec![0.0; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0.0;
                        for p in 0..k { acc += a[i*k+p] * b[p*n+j]; }
                        out[i*n+j] = acc;
                    }
                }
                out
            }
        }
    }
}
