//! Backend de convolution CPU vérifié.
//!
//! ## Sécurité numérique
//! - `check_finite` sur les entrées kernel/data avant exécution
//! - Détection overflow dans la convolution CPU
//!
//! L'accélération matricielle GPU est fournie par `scirust-gpu`; ce module ne
//! déclare aucun backend matériel qu'il ne peut réellement exécuter.

use thiserror::Error;

/// Erreur du backend de calcul.
#[derive(Debug, Error)]
pub enum BackendError {
    #[error("NaN or Inf detected in input data (index {idx}, value {value:.3e})")]
    NanDetected { idx: usize, value: f32 },

    #[error("overflow detected during convolution at index {idx}: value {value:.3e}")]
    Overflow { idx: usize, value: f32 },

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

/// Retourne le backend de convolution disponible dans `scirust-core`.
///
/// Les backends GPU ont une API matricielle distincte dans `scirust-gpu` et ne
/// sont donc pas présentés ici comme des implémentations interchangeables.
pub fn get_backend() -> BackendResult<Box<dyn ComputeBackend>> {
    Ok(Box::new(CpuFallback))
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
}
