//! Abstraction GPU — ComputeBackend trait avec fallback CPU et CUDA
//!
//! ## Sécurité numérique
//! - `check_finite` sur les entrées kernel/data avant exécution
//! - Détection overflow dans la convolution CPU
//! - `get_backend()` retourne `Result` — pas de stub silencieux
//! - CUDA stub remplacé par `Err(BackendError::UnsupportedBackend)` honnête

use thiserror::Error;

/// Erreur du backend de calcul.
#[derive(Debug, Error)]
pub enum BackendError {
    #[error("NaN or Inf detected in input data (index {idx}, value {value:.3e})")]
    NanDetected { idx: usize, value: f32 },

    #[error("overflow detected during convolution at index {idx}: value {value:.3e}")]
    Overflow { idx: usize, value: f32 },

    #[error("CUDA backend not available on this hardware/configuration")]
    UnsupportedBackend,

    #[error("internal compute error: {0}")]
    Internal(String),
}

/// Result spécialisé pour les opérations backend.
pub type BackendResult<T> = Result<T, BackendError>;

/// Vérifie qu'un slice ne contient ni NaN ni Inf.
fn check_finite_slice(data: &[f32], _label: &str) -> BackendResult<()> {
    for (i, &v) in data.iter().enumerate()
    {
        if !v.is_finite()
        {
            return Err(BackendError::NanDetected { idx: i, value: v });
        }
    }
    Ok(())
}

/// Trait unifié pour l'exécution de kernels sur différents backends.
pub trait ComputeBackend {
    fn is_available(&self) -> bool;
    fn execute_kernel(&self, kernel: &[f32], data: &[f32]) -> BackendResult<Vec<f32>>;
}

/// Backend CPU — toujours disponible.
pub struct CpuFallback;

impl ComputeBackend for CpuFallback {
    fn is_available(&self) -> bool {
        true
    }

    fn execute_kernel(&self, kernel: &[f32], data: &[f32]) -> BackendResult<Vec<f32>> {
        // Vérifier l'intégrité des entrées
        check_finite_slice(kernel, "kernel")?;
        check_finite_slice(data, "data")?;

        if kernel.is_empty()
        {
            return Err(BackendError::Internal("empty kernel".into()));
        }

        // Convolution simplifiée avec détection d'overflow
        let mut out = vec![0.0f32; data.len()];
        let half_k = kernel.len() / 2;

        #[allow(clippy::needless_range_loop)]
        for i in 0..data.len()
        {
            let mut sum = 0.0f64; // accumuler en f64 pour réduire overflow
            for (j, &k) in kernel.iter().enumerate()
            {
                let idx = i as isize + j as isize - half_k as isize;
                if idx >= 0 && (idx as usize) < data.len()
                {
                    let product = data[idx as usize] as f64 * k as f64;
                    sum += product;
                }
            }

            // Vérifier overflow avant cast
            if !sum.is_finite() || sum.abs() > f32::MAX as f64
            {
                return Err(BackendError::Overflow {
                    idx: i,
                    value: sum as f32,
                });
            }
            out[i] = sum as f32;
        }

        Ok(out)
    }
}

/// Backend CUDA — si GPU NVIDIA disponible.
pub struct CudaBackend;

impl ComputeBackend for CudaBackend {
    fn is_available(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    fn execute_kernel(&self, _kernel: &[f32], _data: &[f32]) -> BackendResult<Vec<f32>> {
        // Vrai signal : pas de stub trompeur
        Err(BackendError::UnsupportedBackend)
    }
}

/// Sélectionne le meilleur backend disponible.
/// Retourne `Err` si aucun backend ne peut être initialisé.
pub fn get_backend() -> BackendResult<Box<dyn ComputeBackend>> {
    #[cfg(feature = "gpu")]
    {
        let cuda = CudaBackend;
        if cuda.is_available()
        {
            return Ok(Box::new(cuda));
        }
    }

    // Fallback CPU
    let cpu = CpuFallback;
    if cpu.is_available()
    {
        return Ok(Box::new(cpu));
    }

    Err(BackendError::UnsupportedBackend)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_fallback() {
        let backend = CpuFallback;
        assert!(backend.is_available());
        let kernel = vec![1.0f32, 0.0, -1.0];
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = backend.execute_kernel(&kernel, &data).unwrap();
        assert_eq!(result.len(), data.len());
    }

    #[test]
    fn test_get_backend() {
        let backend = get_backend().unwrap();
        assert!(backend.is_available());
    }

    #[test]
    fn test_nan_detected() {
        let backend = CpuFallback;
        let kernel = vec![1.0f32, f32::NAN];
        let data = vec![1.0, 2.0, 3.0];
        let result = backend.execute_kernel(&kernel, &data);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BackendError::NanDetected { .. }
        ));
    }

    #[test]
    fn test_cuda_stub_returns_err() {
        let cuda = CudaBackend;
        let result = cuda.execute_kernel(&[1.0], &[2.0]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            BackendError::UnsupportedBackend
        ));
    }
}
