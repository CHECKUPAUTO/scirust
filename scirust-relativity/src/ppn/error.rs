//! Typed errors for PPN parameter extraction.

use std::error::Error;
use std::fmt;

/// Errors reported by the PPN (Eddington–Robertson `gamma`, `beta`) extraction
/// framework. Every failure on user-controlled numerical input is one of these;
/// the extractor never panics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PpnError {
    /// The mass scale `G M` is non-finite or not strictly positive.
    InvalidMassScale(f64),

    /// The radial domain is malformed: bounds non-finite, non-positive, or
    /// `radius_min >= radius_max`.
    InvalidRadialDomain {
        /// Requested inner radius.
        radius_min: f64,
        /// Requested outer radius.
        radius_max: f64,
    },

    /// A radius passed to the metric is non-finite or not strictly positive.
    InvalidMetricRadius(f64),

    /// The strongest-field sample has compactness `U = G M / radius` outside the
    /// weak-field window the extraction is valid in.
    CompactnessOutOfRange {
        /// The offending compactness value.
        compactness: f64,
        /// The weak-field upper bound.
        maximum: f64,
    },

    /// Fewer samples than the requested extrapolation degree needs.
    InsufficientSamples {
        /// Number of samples available.
        available: usize,
        /// Number required (`degree + 1`).
        required: usize,
    },

    /// A metric component evaluated to a non-finite value at `radius`.
    NonFiniteMetricValue {
        /// The radius at which the evaluation failed.
        radius: f64,
    },

    /// The spatial metric is not conformally flat at `radius`, so the radial
    /// coordinate is not isotropic and the PPN formulas do not apply. This is
    /// how areal-coordinate metrics are rejected rather than silently misused.
    NonIsotropicCoordinates {
        /// The radius at which the isotropy check failed.
        radius: f64,
        /// The relative mismatch between the spatial conformal ratios.
        relative_mismatch: f64,
    },

    /// The metric is not (sufficiently) asymptotically flat at `radius`:
    /// `g_00 + 1` or `A - 1` is not a weak-field perturbation there.
    NonAsymptoticallyFlat {
        /// The radius at which the check failed.
        radius: f64,
    },

    /// The requested extrapolation degree is outside the supported range.
    UnsupportedExtrapolationOrder {
        /// Requested degree.
        order: usize,
        /// Maximum supported degree.
        maximum: usize,
    },

    /// The least-squares system has an exactly singular pivot.
    SingularFit,

    /// The least-squares system is too ill-conditioned to trust.
    IllConditionedFit {
        /// The conditioning indicator (scaled minimum pivot magnitude).
        conditioning: f64,
    },

    /// A computed estimate evaluated to a non-finite value.
    NonFiniteEstimate,
}

impl fmt::Display for PpnError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidMassScale(mass) => write!(
                formatter,
                "mass scale G M must be finite and strictly positive; got {mass}"
            ),
            Self::InvalidRadialDomain {
                radius_min,
                radius_max,
            } => write!(
                formatter,
                "radial domain must satisfy 0 < radius_min < radius_max; got [{radius_min}, {radius_max}]"
            ),
            Self::InvalidMetricRadius(radius) => write!(
                formatter,
                "metric radius must be finite and strictly positive; got {radius}"
            ),
            Self::CompactnessOutOfRange {
                compactness,
                maximum,
            } => write!(
                formatter,
                "strongest-field compactness U = {compactness} exceeds the weak-field bound {maximum}"
            ),
            Self::InsufficientSamples {
                available,
                required,
            } => write!(
                formatter,
                "insufficient samples: {available} available, {required} required for the degree"
            ),
            Self::NonFiniteMetricValue { radius } =>
            {
                write!(formatter, "metric value is not finite at radius {radius}")
            },
            Self::NonIsotropicCoordinates {
                radius,
                relative_mismatch,
            } => write!(
                formatter,
                "spatial metric is not conformally flat at radius {radius} (relative mismatch \
                 {relative_mismatch}); the radial coordinate is not isotropic"
            ),
            Self::NonAsymptoticallyFlat { radius } => write!(
                formatter,
                "metric is not a weak-field perturbation of Minkowski at radius {radius}"
            ),
            Self::UnsupportedExtrapolationOrder { order, maximum } => write!(
                formatter,
                "extrapolation degree {order} is unsupported (1..={maximum})"
            ),
            Self::SingularFit => write!(formatter, "least-squares system is singular"),
            Self::IllConditionedFit { conditioning } => write!(
                formatter,
                "least-squares system is ill-conditioned (indicator {conditioning})"
            ),
            Self::NonFiniteEstimate =>
            {
                write!(formatter, "a computed PPN estimate is not finite")
            },
        }
    }
}

impl Error for PpnError {}
