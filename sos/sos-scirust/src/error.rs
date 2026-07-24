//! Computational-backend-adapter error type.

use thiserror::Error;

/// Errors produced by the `sos-scirust` adapters.
#[derive(Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum ScirustError {
    /// The underlying `scirust-gp` fit failed.
    #[error("Gaussian process fit failed: {0}")]
    Gp(#[from] scirust_gp::GpError),
    /// A declared observation noise variance was not finite and strictly
    /// positive. Every real sensor/measurement has some noise; a value of
    /// zero would claim infinite information from a single observation, and
    /// NaN/infinity are never a meaningful noise variance.
    #[error("observation noise variance must be finite and positive, got {0}")]
    NonPositiveNoise(f64),
}

/// Convenience alias for `sos-scirust` results.
pub type Result<T> = core::result::Result<T, ScirustError>;
