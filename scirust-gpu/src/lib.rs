pub mod dispatch {
    /// A trait for data that can be processed on GPU or CPU.
    pub trait GpuData: Send + Sync {
        type Elem;
        fn as_slice(&self) -> &[Self::Elem];
        fn as_mut_slice(&mut self) -> &mut [Self::Elem];
    }

    impl GpuData for Vec<f32> {
        type Elem = f32;
        fn as_slice(&self) -> &[f32] { self }
        fn as_mut_slice(&mut self) -> &mut [f32] { self }
    }

    impl GpuData for [f32] {
        type Elem = f32;
        fn as_slice(&self) -> &[f32] { self }
        fn as_mut_slice(&mut self) -> &mut [f32] { self }
    }

    #[cfg(feature = "cpu-fallback")]
    pub fn gpu_or_cpu<F>(data: &mut [f32], kernel: F)
    where
        F: Fn(&mut [f32]) + Sync,
    {
        // For very small workloads, don't bother with rayon overhead
        if data.len() < 1024 {
            kernel(data);
        } else {
            use rayon::prelude::*;
            data.par_chunks_mut(1024).for_each(|chunk| {
                kernel(chunk);
            });
        }
    }

    #[cfg(not(feature = "cpu-fallback"))]
    pub fn gpu_or_cpu<F>(data: &mut [f32], kernel: F)
    where
        F: Fn(&mut [f32]),
    {
        #[cfg(feature = "cuda")]
        {
            // Initial foundation for CUDA dispatch.
            // In a real implementation, we would use `cust` to:
            // 1. Copy data to device
            // 2. Launch kernel
            // 3. Copy back to host
            // For now, this is a placeholder showing the architectural intent.
            eprintln!("[scirust-gpu] CUDA feature active: Preparing to launch kernel...");
            kernel(data);
        }

        #[cfg(not(feature = "cuda"))]
        {
            kernel(data);
        }
    }
}

pub mod error;
pub mod quantize;
pub mod quant_train;
#[cfg(feature = "legacy-cust")]
pub mod cuda_backend;
#[cfg(feature = "legacy-cust")]
pub mod cuda_turboquant;
pub mod wgpu_backend;
pub mod gpu_tensor;
pub mod gpu_gemm;

#[cfg(feature = "cuda")]
pub mod cublas;
