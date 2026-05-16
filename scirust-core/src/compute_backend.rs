//! Abstraction GPU — ComputeBackend trait avec fallback CPU et CUDA

/// Trait unifié pour l'exécution de kernels sur différents backends
pub trait ComputeBackend {
    fn is_available(&self) -> bool;
    fn execute_kernel(&self, kernel: &[f32], data: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>>;
}

/// Backend CPU — toujours disponible
pub struct CpuFallback;

impl ComputeBackend for CpuFallback {
    fn is_available(&self) -> bool { true }

    fn execute_kernel(&self, kernel: &[f32], data: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Convolution simplifiée
        let mut out = vec![0.0f32; data.len()];
        let half_k = kernel.len() / 2;
        for i in 0..data.len() {
            let mut sum = 0.0f32;
            for (j, &k) in kernel.iter().enumerate() {
                let idx = i as isize + j as isize - half_k as isize;
                if idx >= 0 && (idx as usize) < data.len() {
                    sum += data[idx as usize] * k;
                }
            }
            out[i] = sum;
        }
        Ok(out)
    }
}

/// Backend CUDA — si GPU NVIDIA disponible
pub struct CudaBackend;

impl ComputeBackend for CudaBackend {
    fn is_available(&self) -> bool {
        std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
    }

    fn execute_kernel(&self, _kernel: &[f32], data: &[f32]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Stub: TODO implémentation CUDA réelle avec cudarc
        Ok(data.to_vec())
    }
}

/// Sélectionne le meilleur backend disponible
pub fn get_backend() -> Box<dyn ComputeBackend> {
    #[cfg(feature = "gpu")]
    {
        let cuda = CudaBackend;
        if cuda.is_available() {
            return Box::new(cuda);
        }
    }
    Box::new(CpuFallback)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_fallback() {
        let backend = CpuFallback;
        assert!(backend.is_available());
        let kernel = vec![1.0f32, 0.0, -1.0]; // edge detection
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = backend.execute_kernel(&kernel, &data).unwrap();
        assert_eq!(result.len(), data.len());
    }

    #[test]
    fn test_get_backend() {
        let backend = get_backend();
        assert!(backend.is_available());
    }
}
