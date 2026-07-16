//! Error type shared across the crate.

use core::fmt;

/// A convenience `Result` alias for fallible ITD operations.
pub type Result<T> = core::result::Result<T, ItdError>;

/// Errors raised by grid construction, field operators and the simulation
/// driver. These mirror the validation the reference implementation performs
/// (finite values, minimum grid sizes, matching shapes, strictly increasing
/// coordinates, and boundary-mode restrictions).
#[derive(Debug, Clone, PartialEq)]
pub enum ItdError {
    /// A spacing or coordinate value was not finite and strictly positive /
    /// strictly increasing as required.
    InvalidGeometry(String),
    /// A field did not match the shape implied by its geometry, or two fields
    /// that must agree in shape did not.
    ShapeMismatch(String),
    /// A grid was too small for the requested operator (gradients and
    /// vorticity need at least three points per direction; a spatial mean
    /// needs at least two).
    TooFewPoints(String),
    /// A field or parameter contained a non-finite value.
    NonFinite(String),
    /// The structural weights were not five finite, non-negative values with a
    /// strictly positive sum.
    InvalidWeights(String),
    /// A requested boundary mode is not supported in the given configuration
    /// (e.g. a periodic mean on a non-uniform rectilinear grid).
    UnsupportedBoundary(String),
}

impl fmt::Display for ItdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            ItdError::InvalidGeometry(m) => write!(f, "invalid geometry: {m}"),
            ItdError::ShapeMismatch(m) => write!(f, "shape mismatch: {m}"),
            ItdError::TooFewPoints(m) => write!(f, "too few grid points: {m}"),
            ItdError::NonFinite(m) => write!(f, "non-finite value: {m}"),
            ItdError::InvalidWeights(m) => write!(f, "invalid structural weights: {m}"),
            ItdError::UnsupportedBoundary(m) => write!(f, "unsupported boundary mode: {m}"),
        }
    }
}

impl std::error::Error for ItdError {}
