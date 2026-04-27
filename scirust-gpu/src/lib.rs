pub mod dispatch {
    #[cfg(feature = "cpu-fallback")]
    pub fn gpu_or_cpu<F>(data: &mut [f32], kernel: F)
    where
        F: Fn(&mut [f32]) + Sync,
    {
        use rayon::prelude::*;
        data.par_chunks_mut(1024).for_each(|chunk| {
            kernel(chunk);
        });
    }

    #[cfg(not(feature = "cpu-fallback"))]
    pub fn gpu_or_cpu<F>(data: &mut [f32], kernel: F)
    where
        F: Fn(&mut [f32]),
    {
        kernel(data);
    }
}

pub mod wgpu_backend;
pub mod cuda_backend;

pub mod gpu_tensor;

pub mod gpu_gemm;
