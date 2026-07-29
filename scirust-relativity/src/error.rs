//! Errors reported by differential-geometry operations.

use std::error::Error;
use std::fmt;

/// Errors produced while evaluating metrics and connections.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RelativityError {
    /// A coordinate component is not finite.
    NonFiniteCoordinate(usize),

    /// A metric component is not finite.
    NonFiniteMetricComponent {
        /// Row index.
        row: usize,
        /// Column index.
        column: usize,
    },

    /// The metric matrix is singular or numerically non-invertible.
    SingularMetric,

    /// The finite-difference step is non-finite or non-positive.
    InvalidDifferenceStep(f64),

    /// A curvature-tensor component evaluated to a non-finite value.
    NonFiniteCurvatureComponent {
        /// Short name of the offending quantity (for example `"riemann"`,
        /// `"ricci"`, `"ricci_scalar"`, `"einstein"`, `"kretschmann"`).
        quantity: &'static str,
    },

    /// The number of integration substeps requested for parallel transport is
    /// zero (at least one substep is required).
    InvalidTransportResolution,

    /// A parallel-transported vector component evaluated to a non-finite value.
    NonFiniteTransportedVector,

    /// The affine length requested for a geodesic-deviation integration is
    /// non-finite or non-positive.
    InvalidAffineLength(f64),

    /// A geodesic-deviation (Jacobi) field component evaluated to a non-finite
    /// value.
    NonFiniteDeviationVector,

    /// The geodesic underlying an exponential-map evaluation could not be
    /// integrated to a finite endpoint (for example it left the regular chart).
    ExponentialMapIntegrationFailed,

    /// The logarithm-map Newton iteration did not reach the requested tolerance
    /// within the allowed number of iterations.
    LogarithmMapDidNotConverge,

    /// The timelike-norm floor requested for an orthonormal-frame (tetrad)
    /// construction is non-finite or non-positive.
    InvalidTetradFloor(f64),

    /// The vector chosen as the timelike leg of a tetrad is not timelike:
    /// `g(u, u) > -floor` under the `(-,+,+,+)` convention (this includes a
    /// non-finite norm).
    NonTimelikeFrameVector {
        /// The offending norm `g(u, u)`.
        metric_norm: f64,
    },

    /// A tetrad leg evaluated to a non-finite component.
    NonFiniteTetradLeg,

    /// Fewer than `dimension` independent orthonormal legs could be built, so
    /// the frame is degenerate.
    DegenerateFrame {
        /// Number of independent legs that were built.
        legs_found: usize,
        /// The dimension `D` (the required number of legs).
        dimension: usize,
    },

    /// Synge's world function or one of its gradient components evaluated to a
    /// non-finite value.
    NonFiniteWorldFunction,
}

impl fmt::Display for RelativityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::NonFiniteCoordinate(index) =>
            {
                write!(formatter, "coordinate at index {index} is not finite")
            },
            Self::NonFiniteMetricComponent { row, column } =>
            {
                write!(
                    formatter,
                    "metric component ({row}, {column}) is not finite"
                )
            },
            Self::SingularMetric =>
            {
                write!(
                    formatter,
                    "metric is singular or numerically non-invertible"
                )
            },
            Self::InvalidDifferenceStep(step) =>
            {
                write!(
                    formatter,
                    "finite-difference step must be finite and positive; got {step}"
                )
            },
            Self::NonFiniteCurvatureComponent { quantity } =>
            {
                write!(formatter, "curvature quantity '{quantity}' is not finite")
            },
            Self::InvalidTransportResolution =>
            {
                write!(
                    formatter,
                    "parallel-transport resolution must be at least one substep"
                )
            },
            Self::NonFiniteTransportedVector =>
            {
                write!(formatter, "parallel-transported vector is not finite")
            },
            Self::InvalidAffineLength(length) =>
            {
                write!(
                    formatter,
                    "affine length must be finite and positive; got {length}"
                )
            },
            Self::NonFiniteDeviationVector =>
            {
                write!(formatter, "geodesic-deviation vector is not finite")
            },
            Self::ExponentialMapIntegrationFailed =>
            {
                write!(
                    formatter,
                    "exponential-map geodesic could not be integrated to a finite endpoint"
                )
            },
            Self::LogarithmMapDidNotConverge =>
            {
                write!(
                    formatter,
                    "logarithm-map Newton iteration did not converge to tolerance"
                )
            },
            Self::InvalidTetradFloor(floor) =>
            {
                write!(
                    formatter,
                    "tetrad timelike-norm floor must be finite and positive; got {floor}"
                )
            },
            Self::NonTimelikeFrameVector { metric_norm } =>
            {
                write!(
                    formatter,
                    "tetrad timelike leg is not timelike; g(u, u) = {metric_norm}"
                )
            },
            Self::NonFiniteTetradLeg =>
            {
                write!(formatter, "tetrad leg has a non-finite component")
            },
            Self::DegenerateFrame {
                legs_found,
                dimension,
            } =>
            {
                write!(
                    formatter,
                    "degenerate frame: only {legs_found} of {dimension} independent legs were built"
                )
            },
            Self::NonFiniteWorldFunction =>
            {
                write!(formatter, "world function or its gradient is not finite")
            },
        }
    }
}

impl Error for RelativityError {}
