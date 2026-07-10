//! Error type returned by every fallible function of the crate.

use std::fmt;

/// Error produced when an input fails validation or an iterative solver
/// cannot reach its tolerance.
///
/// Every public function validates its arguments before computing and
/// returns one of these variants instead of panicking or silently
/// propagating NaN.
#[derive(Debug, Clone, PartialEq)]
pub enum FluidsError {
    /// A non-finite value (NaN or ±∞) was supplied for the named argument.
    NonFinite {
        /// Name of the offending argument.
        name: &'static str,
    },
    /// The named argument must be strictly positive but was not.
    NonPositive {
        /// Name of the offending argument.
        name: &'static str,
        /// Value that was supplied.
        value: f64,
    },
    /// The named argument must be non-negative but was negative.
    Negative {
        /// Name of the offending argument.
        name: &'static str,
        /// Value that was supplied.
        value: f64,
    },
    /// The named argument fell outside the documented validity range.
    OutOfRange {
        /// Name of the offending argument.
        name: &'static str,
        /// Value that was supplied.
        value: f64,
        /// Inclusive lower bound of the validity range.
        min: f64,
        /// Inclusive upper bound of the validity range.
        max: f64,
    },
    /// An iterative solver failed to converge (should not happen for
    /// inputs inside the documented ranges; reported rather than looping).
    NoConvergence {
        /// Human-readable name of the quantity being solved for.
        what: &'static str,
    },
}

impl fmt::Display for FluidsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            FluidsError::NonFinite { name } =>
            {
                write!(f, "argument `{name}` is non-finite (NaN or infinity)")
            },
            FluidsError::NonPositive { name, value } =>
            {
                write!(f, "argument `{name}` must be > 0, got {value}")
            },
            FluidsError::Negative { name, value } =>
            {
                write!(f, "argument `{name}` must be >= 0, got {value}")
            },
            FluidsError::OutOfRange {
                name,
                value,
                min,
                max,
            } =>
            {
                write!(
                    f,
                    "argument `{name}` = {value} outside validity range [{min}, {max}]"
                )
            },
            FluidsError::NoConvergence { what } =>
            {
                write!(f, "iterative solve of {what} did not converge")
            },
        }
    }
}

impl std::error::Error for FluidsError {}

/// Require `v` to be finite.
pub(crate) fn finite(name: &'static str, v: f64) -> Result<f64, FluidsError> {
    if v.is_finite()
    {
        Ok(v)
    }
    else
    {
        Err(FluidsError::NonFinite { name })
    }
}

/// Require `v` to be finite and strictly positive.
pub(crate) fn positive(name: &'static str, v: f64) -> Result<f64, FluidsError> {
    finite(name, v)?;
    if v > 0.0
    {
        Ok(v)
    }
    else
    {
        Err(FluidsError::NonPositive { name, value: v })
    }
}

/// Require `v` to be finite and non-negative.
pub(crate) fn non_negative(name: &'static str, v: f64) -> Result<f64, FluidsError> {
    finite(name, v)?;
    if v >= 0.0
    {
        Ok(v)
    }
    else
    {
        Err(FluidsError::Negative { name, value: v })
    }
}

/// Require `v` to be finite and inside `[min, max]`.
pub(crate) fn in_range(name: &'static str, v: f64, min: f64, max: f64) -> Result<f64, FluidsError> {
    finite(name, v)?;
    if (min..=max).contains(&v)
    {
        Ok(v)
    }
    else
    {
        Err(FluidsError::OutOfRange {
            name,
            value: v,
            min,
            max,
        })
    }
}
