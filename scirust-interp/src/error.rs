//! Error type returned by fallible interpolant constructors.

use std::fmt;

/// Error produced when node data fails validation.
///
/// Every multi-point constructor validates its `xs`/`ys` before building the
/// interpolant and returns one of these variants instead of panicking. The
/// input is assumed to be well-typed (`f64` slices); the checks concern
/// *shape* and *ordering*, not type errors.
#[derive(Debug, Clone, PartialEq)]
pub enum InterpError {
    /// `xs` and `ys` had different lengths.
    LengthMismatch {
        /// Number of abscissae supplied.
        xs: usize,
        /// Number of ordinates supplied.
        ys: usize,
    },
    /// Fewer points than the method requires (e.g. Akima needs at least 5).
    TooFewPoints {
        /// Number of points supplied.
        got: usize,
        /// Minimum number of points the method needs.
        need: usize,
    },
    /// The abscissae were not strictly increasing at the given index
    /// (a duplicate or an out-of-order value: `xs[index] <= xs[index - 1]`).
    NotStrictlyIncreasing {
        /// Index at which strict monotonicity was violated.
        index: usize,
    },
    /// A non-finite value (NaN or ±∞) was found at the given position.
    NonFinite {
        /// Index of the offending value (in whichever array it occurred).
        index: usize,
    },
}

impl fmt::Display for InterpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            InterpError::LengthMismatch { xs, ys } =>
            {
                write!(f, "length mismatch: xs has {xs} points but ys has {ys}")
            },
            InterpError::TooFewPoints { got, need } =>
            {
                write!(f, "too few points: got {got}, need at least {need}")
            },
            InterpError::NotStrictlyIncreasing { index } =>
            {
                write!(
                    f,
                    "xs must be strictly increasing: violated at index {index}"
                )
            },
            InterpError::NonFinite { index } =>
            {
                write!(f, "non-finite value (NaN or infinity) at index {index}")
            },
        }
    }
}

impl std::error::Error for InterpError {}
