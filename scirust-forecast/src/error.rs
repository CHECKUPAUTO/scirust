//! Error type returned by the fallible forecasting routines.

use std::fmt;

/// Error produced when an input series or parameter fails validation.
///
/// Every fallible constructor and metric validates its arguments up front and
/// returns one of these variants instead of panicking. The checks concern the
/// *shape* of the data (length, seasonal period) and the *range* of the
/// smoothing parameters, all of which must hold for the underlying recurrence
/// to be well defined.
#[derive(Debug, Clone, PartialEq)]
pub enum ForecastError {
    /// The input series was empty (no observations to fit).
    EmptySeries,
    /// A smoothing parameter fell outside the valid closed interval `[0, 1]`
    /// (this also catches `NaN`).
    InvalidSmoothing {
        /// Name of the offending parameter (`"alpha"`, `"beta"`, `"gamma"`).
        name: &'static str,
        /// The value that was supplied.
        value: f64,
    },
    /// The series was too short for the requested operation.
    SeriesTooShort {
        /// Number of observations supplied.
        got: usize,
        /// Minimum number of observations the operation requires.
        need: usize,
    },
    /// The seasonal period was zero (a period must be at least one).
    InvalidPeriod {
        /// The period that was supplied.
        period: usize,
    },
    /// The moving-average window was zero (a window must be at least one).
    InvalidWindow {
        /// The window length that was supplied.
        window: usize,
    },
    /// The autoregressive order was zero (an order must be at least one).
    InvalidOrder {
        /// The order that was supplied.
        order: usize,
    },
    /// Two slices that were required to be the same length were not.
    LengthMismatch {
        /// Length of the first (typically `actual`) slice.
        left: usize,
        /// Length of the second (typically `pred`) slice.
        right: usize,
    },
    /// A zero was found among the actual values while computing MAPE, which
    /// would require dividing by zero.
    ZeroActual {
        /// Index of the zero actual value.
        index: usize,
    },
}

impl fmt::Display for ForecastError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            ForecastError::EmptySeries => write!(f, "the input series was empty"),
            ForecastError::InvalidSmoothing { name, value } =>
            {
                write!(
                    f,
                    "smoothing parameter `{name}` = {value} is outside [0, 1]"
                )
            },
            ForecastError::SeriesTooShort { got, need } =>
            {
                write!(
                    f,
                    "series too short: got {got} observations, need at least {need}"
                )
            },
            ForecastError::InvalidPeriod { period } =>
            {
                write!(f, "invalid seasonal period {period}: must be at least 1")
            },
            ForecastError::InvalidWindow { window } =>
            {
                write!(
                    f,
                    "invalid moving-average window {window}: must be at least 1"
                )
            },
            ForecastError::InvalidOrder { order } =>
            {
                write!(
                    f,
                    "invalid autoregressive order {order}: must be at least 1"
                )
            },
            ForecastError::LengthMismatch { left, right } =>
            {
                write!(f, "length mismatch: {left} vs {right}")
            },
            ForecastError::ZeroActual { index } =>
            {
                write!(
                    f,
                    "actual value at index {index} is zero; MAPE is undefined"
                )
            },
        }
    }
}

impl std::error::Error for ForecastError {}
