//! Errors reported by fractional-calculus operators.

use std::error::Error;
use std::fmt;

/// Errors produced while validating or evaluating a fractional operator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FractionalError {
    /// The fractional order is non-finite or outside the supported interval
    /// `0 < alpha < 1`.
    InvalidOrder(f64),

    /// The uniform sample spacing is non-finite or non-positive.
    InvalidStep(f64),

    /// The supplied sample sequence is empty.
    EmptySamples,

    /// The operator requires at least two samples.
    TooFewSamples,

    /// A sample at the indicated index is not finite.
    NonFiniteSample(usize),

    /// The sample sequence and sample-time sequence have different lengths.
    MismatchedLengths {
        /// Number of supplied samples.
        samples: usize,
        /// Number of supplied sample times.
        sample_times: usize,
    },

    /// A sample time at the indicated index is not finite.
    NonFiniteSampleTime(usize),

    /// Sample times are not strictly increasing at the indicated index.
    NonMonotonicSampleTimes {
        /// Index at which `sample_times[index] <= sample_times[index - 1]`.
        index: usize,
    },
}

impl fmt::Display for FractionalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidOrder(alpha) =>
            {
                write!(
                    formatter,
                    "fractional order must be finite and satisfy 0 < alpha < 1; got {alpha}"
                )
            },
            Self::InvalidStep(step) =>
            {
                write!(
                    formatter,
                    "uniform sample step must be finite and positive; got {step}"
                )
            },
            Self::EmptySamples => write!(formatter, "sample sequence must not be empty"),
            Self::TooFewSamples =>
            {
                write!(formatter, "this operator requires at least two samples")
            },
            Self::NonFiniteSample(index) =>
            {
                write!(formatter, "sample at index {index} is not finite")
            },
            Self::MismatchedLengths {
                samples,
                sample_times,
            } => write!(
                formatter,
                "sample sequence has length {samples} but sample-time sequence has length \
                 {sample_times}"
            ),
            Self::NonFiniteSampleTime(index) =>
            {
                write!(formatter, "sample time at index {index} is not finite")
            },
            Self::NonMonotonicSampleTimes { index } => write!(
                formatter,
                "sample times must be strictly increasing; violated at index {index}"
            ),
        }
    }
}

impl Error for FractionalError {}
