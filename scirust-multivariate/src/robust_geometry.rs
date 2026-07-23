//! Robust scaling and scale-aware geometry.
//!
//! Raw Euclidean distance silently depends on the units and magnitudes of each
//! coordinate: multiplying one column by `1e6` (say, expressing a pressure in
//! pascal instead of megapascal) completely rewrites neighbourhoods, clusters,
//! and outlier scores. This module provides *fitted* geometry models that remove
//! that accidental dependence:
//!
//! - [`RobustScaler`] — a fitted per-coordinate location/scale model
//!   (standard-deviation, MAD, or IQR based) with an explicit
//!   [`ZeroScalePolicy`] for degenerate dimensions;
//! - [`FittedDistanceMetric`] — raw Euclidean, a relative (norm-ratio) distance,
//!   a robust diagonal distance backed by a fitted scaler, and a regularized
//!   Mahalanobis distance;
//! - [`FeatureDescriptor`] — a units boundary carrying each feature's physical
//!   [`scirust_units::Dimension`], so dimensionally incoherent comparisons can be
//!   rejected before any distance is computed.
//!
//! # Invariance groups (documented per metric, never overstated)
//!
//! | Metric | Invariant to | Not invariant to |
//! |---|---|---|
//! | `RawEuclidean` | rigid motions | any rescaling |
//! | `RelativeNorm` | common positive rescaling (in the `ε`-inactive regime) | per-coordinate rescaling, translation |
//! | `RobustDiagonal` (refit) | positive per-coordinate rescaling + translation | rotations, general affine maps |
//! | `RegularizedMahalanobis` (refit, `ridge = 0`) | affine maps in exact arithmetic only; the ridge and floating point break exact equivariance | — |
//!
//! No metric here is claimed to be *robust affine invariant*: the Mahalanobis
//! variant uses the **classical** mean and covariance (plus an explicit ridge),
//! so a minority of gross outliers can distort it. That is why it is named
//! `RegularizedMahalanobis` and not `RobustAffineInvariant`.
//!
//! # Robust statistics reuse
//!
//! Per-coordinate medians, MADs and IQRs come from `scirust-stats`
//! ([`scirust_stats::robust`]); this module does not re-implement them.
//!
//! # Determinism
//!
//! Fitting and evaluation are pure loops in fixed order with no hidden RNG and
//! no thread-dependent reductions: the same input matrix and configuration
//! produce bit-identical models and distances.

use std::fmt;

use serde::{Deserialize, Serialize};

use scirust_stats::robust::{MadConsistency, interquartile_range, median_absolute_deviation};
use scirust_stats::{ChiSquared, Distribution, Normal, RobustStatsError, describe};

use crate::{Matrix, invert_lower_triangular};

/// Errors returned by robust scaling and fitted geometry.
#[derive(Debug, Clone, PartialEq)]
pub enum RobustGeometryError {
    /// The input matrix has zero rows or zero columns.
    EmptyMatrix,
    /// A row's length differs from the matrix's declared column count.
    RaggedMatrix {
        /// Index of the offending row.
        row: usize,
        /// The declared column count.
        expected: usize,
        /// The offending row's actual length.
        found: usize,
    },
    /// A matrix entry is `NaN` or `±∞`.
    NonFiniteValue {
        /// Row of the offending entry.
        row: usize,
        /// Column of the offending entry.
        col: usize,
        /// The non-finite value.
        value: f64,
    },
    /// The estimator needs more observations than were supplied (for example the
    /// unbiased standard deviation needs at least two rows).
    InsufficientSamples {
        /// Minimum number of rows required.
        required: usize,
        /// Number of rows supplied.
        found: usize,
    },
    /// `minimum_scale` is negative or non-finite.
    InvalidMinimumScale {
        /// The rejected threshold.
        minimum_scale: f64,
    },
    /// A dimension's fitted scale is at or below `minimum_scale` and the policy
    /// is [`ZeroScalePolicy::Error`].
    DegenerateDimension {
        /// The degenerate dimension (column index).
        dimension: usize,
        /// Its fitted scale.
        scale: f64,
    },
    /// Every dimension was dropped by [`ZeroScalePolicy::DropDimension`], so no
    /// geometry remains.
    NoActiveDimensions,
    /// An input's column count does not match the fitted model's.
    DimensionCountMismatch {
        /// The fitted column count.
        expected: usize,
        /// The supplied column count.
        found: usize,
    },
    /// The two points passed to a distance have different lengths.
    LengthMismatch {
        /// Length of the first point.
        left: usize,
        /// Length of the second point.
        right: usize,
    },
    /// A coordinate passed to a distance is `NaN` or `±∞`.
    NonFiniteCoordinate {
        /// Index of the offending coordinate.
        index: usize,
        /// The non-finite value.
        value: f64,
    },
    /// `epsilon` for the relative-norm distance is not strictly positive and
    /// finite.
    InvalidEpsilon {
        /// The rejected value.
        epsilon: f64,
    },
    /// `ridge` for the regularized Mahalanobis distance is negative or
    /// non-finite.
    InvalidRidge {
        /// The rejected value.
        ridge: f64,
    },
    /// The (ridge-regularized) scatter matrix is not positive definite: the
    /// strict Cholesky factorization met a non-positive or non-finite pivot.
    /// A larger ridge usually resolves this.
    SingularScatter {
        /// Index of the failing pivot.
        pivot_index: usize,
    },
    /// A per-dimension robust scale estimate failed.
    ScaleEstimation {
        /// The dimension being fitted.
        dimension: usize,
        /// The underlying robust-statistics error.
        source: RobustStatsError,
    },
    /// A robust scale estimate inside the OGK scatter construction failed (for a
    /// column or a pairwise sum/difference — for example an overflow to a
    /// non-finite value).
    ScatterScaleEstimation {
        /// The underlying robust-statistics error.
        source: RobustStatsError,
    },
    /// A fitted location or scale overflowed to a non-finite value on finite
    /// input (for example a mean beyond `f64::MAX` or a MAD whose
    /// normal-consistency product overflows). Reported explicitly: a non-finite
    /// scale must never silently bypass the zero-scale policy.
    NonFiniteScale {
        /// The dimension whose estimate overflowed.
        dimension: usize,
        /// The non-finite fitted scale (the location may be the overflowing
        /// quantity instead; the scale is reported as fitted).
        scale: f64,
    },
    /// Two feature descriptors carry incompatible physical dimensions where the
    /// metric requires a single common dimension.
    IncompatibleFeatureDimensions {
        /// Index of the reference descriptor.
        first_index: usize,
        /// Index of the incompatible descriptor.
        second_index: usize,
    },
    /// The number of feature descriptors does not match the fitted model's
    /// column count.
    DescriptorCountMismatch {
        /// The fitted column count.
        expected: usize,
        /// The number of descriptors supplied.
        found: usize,
    },
    /// No feature descriptors were supplied.
    EmptyDescriptors,
}

impl fmt::Display for RobustGeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyMatrix => f.write_str("matrix has zero rows or zero columns"),
            Self::RaggedMatrix {
                row,
                expected,
                found,
            } => write!(
                f,
                "row {row} has {found} entries but the matrix declares {expected} columns"
            ),
            Self::NonFiniteValue { row, col, value } =>
            {
                write!(f, "matrix entry ({row}, {col}) is not finite ({value})")
            },
            Self::InsufficientSamples { required, found } => write!(
                f,
                "estimator requires at least {required} observations, got {found}"
            ),
            Self::InvalidMinimumScale { minimum_scale } => write!(
                f,
                "minimum scale {minimum_scale} must be finite and non-negative"
            ),
            Self::DegenerateDimension { dimension, scale } => write!(
                f,
                "dimension {dimension} has degenerate scale {scale} under the Error policy"
            ),
            Self::NoActiveDimensions =>
            {
                f.write_str("every dimension was dropped; no geometry remains")
            },
            Self::DimensionCountMismatch { expected, found } => write!(
                f,
                "input has {found} columns but the fitted model expects {expected}"
            ),
            Self::LengthMismatch { left, right } =>
            {
                write!(f, "points have different lengths ({left} vs {right})")
            },
            Self::NonFiniteCoordinate { index, value } =>
            {
                write!(f, "coordinate {index} is not finite ({value})")
            },
            Self::InvalidEpsilon { epsilon } =>
            {
                write!(f, "epsilon {epsilon} must be finite and strictly positive")
            },
            Self::InvalidRidge { ridge } =>
            {
                write!(f, "ridge {ridge} must be finite and non-negative")
            },
            Self::SingularScatter { pivot_index } => write!(
                f,
                "scatter matrix is not positive definite (Cholesky pivot {pivot_index}); \
consider a larger ridge"
            ),
            Self::ScaleEstimation { dimension, source } =>
            {
                write!(
                    f,
                    "fitting robust scale for dimension {dimension}: {source}"
                )
            },
            Self::ScatterScaleEstimation { source } =>
            {
                write!(f, "fitting a robust scale inside OGK scatter: {source}")
            },
            Self::NonFiniteScale { dimension, scale } => write!(
                f,
                "fitted location or scale for dimension {dimension} is not finite \
(scale = {scale}); the estimate overflowed"
            ),
            Self::IncompatibleFeatureDimensions {
                first_index,
                second_index,
            } => write!(
                f,
                "feature {second_index} has a physical dimension incompatible with feature \
{first_index}, but the metric requires one common dimension"
            ),
            Self::DescriptorCountMismatch { expected, found } => write!(
                f,
                "{found} feature descriptors supplied but the fitted model has {expected} columns"
            ),
            Self::EmptyDescriptors => f.write_str("no feature descriptors were supplied"),
        }
    }
}

impl std::error::Error for RobustGeometryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self
        {
            Self::ScaleEstimation { source, .. } => Some(source),
            Self::ScatterScaleEstimation { source } => Some(source),
            _ => None,
        }
    }
}

/// Which per-coordinate scale estimator [`RobustScaler::fit`] uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RobustScaleMethod {
    /// Unbiased (`n − 1`) sample standard deviation with the mean as location.
    /// Classical: **not** robust to outliers; provided as the non-robust
    /// baseline. Requires at least two rows.
    StandardDeviation,
    /// Normal-consistent median absolute deviation (`MAD · 1.4826`) with the
    /// median as location, so that on normal data it estimates the standard
    /// deviation. Robust to a minority of outliers per coordinate.
    MedianAbsoluteDeviation,
    /// Interquartile range (`Q3 − Q1`, type-7 quantiles) with the median as
    /// location. Robust to a minority of outliers per coordinate.
    InterquartileRange,
}

/// What [`RobustScaler::fit`] does with a dimension whose fitted scale is at or
/// below [`RobustScalerConfig::minimum_scale`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZeroScalePolicy {
    /// Fail with [`RobustGeometryError::DegenerateDimension`].
    Error,
    /// Replace the degenerate scale by exactly `1.0` (the dimension still
    /// participates, unscaled).
    UnitScale,
    /// Mark the dimension inactive. Inactive dimensions transform to `0.0`,
    /// inverse-transform to the fitted location, and contribute nothing to
    /// distances.
    DropDimension,
}

/// Configuration for [`RobustScaler::fit`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RobustScalerConfig {
    /// Subtract the fitted per-coordinate location during [`RobustScaler::transform`].
    /// The location is fitted and stored regardless, so dropped dimensions can
    /// always be restored by [`RobustScaler::inverse_transform`].
    pub center: bool,
    /// The per-coordinate scale estimator.
    pub scale_method: RobustScaleMethod,
    /// Policy for dimensions whose scale is at or below `minimum_scale`.
    pub zero_scale_policy: ZeroScalePolicy,
    /// Threshold at or below which a fitted scale counts as degenerate. Must be
    /// finite and non-negative; `0.0` flags only exactly-zero scales.
    pub minimum_scale: f64,
}

impl Default for RobustScalerConfig {
    /// Centering on, normal-consistent MAD, `Error` on zero scale, threshold `0`.
    fn default() -> Self {
        Self {
            center: true,
            scale_method: RobustScaleMethod::MedianAbsoluteDeviation,
            zero_scale_policy: ZeroScalePolicy::Error,
            minimum_scale: 0.0,
        }
    }
}

/// A fitted per-coordinate location/scale model.
///
/// Fitting is deterministic; the fitted state is exposed so that models can be
/// inspected, serialized, and shipped alongside their provenance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobustScaler {
    /// Fitted per-coordinate location (mean for [`RobustScaleMethod::StandardDeviation`],
    /// median otherwise). Always fitted, even when `config.center` is `false`.
    pub location: Vec<f64>,
    /// Fitted per-coordinate scale after the zero-scale policy was applied.
    /// Inactive dimensions keep their raw (degenerate) fitted value here for
    /// inspection; they are never used to divide.
    pub scale: Vec<f64>,
    /// `true` for dimensions that participate in transforms and distances.
    pub active_dimensions: Vec<bool>,
    /// The configuration this model was fitted with.
    pub config: RobustScalerConfig,
}

impl RobustScaler {
    /// Fit a scaler on `data` (rows = observations, columns = features).
    ///
    /// # Errors
    ///
    /// Typed [`RobustGeometryError`] on an empty or ragged matrix, non-finite
    /// entries, an invalid `minimum_scale`, insufficient rows for the chosen
    /// estimator, a degenerate dimension under [`ZeroScalePolicy::Error`], or
    /// when every dimension is dropped.
    pub fn fit(data: &Matrix, config: RobustScalerConfig) -> Result<Self, RobustGeometryError> {
        validate_matrix(data)?;
        if !(config.minimum_scale.is_finite() && config.minimum_scale >= 0.0)
        {
            return Err(RobustGeometryError::InvalidMinimumScale {
                minimum_scale: config.minimum_scale,
            });
        }
        if config.scale_method == RobustScaleMethod::StandardDeviation && data.rows < 2
        {
            return Err(RobustGeometryError::InsufficientSamples {
                required: 2,
                found: data.rows,
            });
        }

        let mut location = Vec::with_capacity(data.cols);
        let mut scale = Vec::with_capacity(data.cols);
        let mut active_dimensions = Vec::with_capacity(data.cols);

        for j in 0..data.cols
        {
            let column = data.col(j);
            let (loc, raw_scale) =
                match config.scale_method
                {
                    RobustScaleMethod::StandardDeviation =>
                    {
                        (describe::mean(&column), describe::variance(&column).sqrt())
                    },
                    RobustScaleMethod::MedianAbsoluteDeviation =>
                    {
                        let mad = median_absolute_deviation(&column, MadConsistency::Normal)
                            .map_err(|source| RobustGeometryError::ScaleEstimation {
                                dimension: j,
                                source,
                            })?;
                        (describe::median(&column), mad)
                    },
                    RobustScaleMethod::InterquartileRange =>
                    {
                        let iqr = scirust_stats::robust::interquartile_range(&column).map_err(
                            |source| RobustGeometryError::ScaleEstimation {
                                dimension: j,
                                source,
                            },
                        )?;
                        (describe::median(&column), iqr)
                    },
                };

            // A location or scale estimate can overflow to a non-finite value
            // even on finite input (a huge mean, a MAD whose normal-consistency
            // product exceeds f64::MAX). A non-finite scale must never be
            // marked active — `inf <= minimum_scale` is false, which would
            // silently bypass every zero-scale policy — so it is a typed
            // failure, not a silent fallback.
            if !loc.is_finite() || !raw_scale.is_finite()
            {
                return Err(RobustGeometryError::NonFiniteScale {
                    dimension: j,
                    scale: raw_scale,
                });
            }

            if raw_scale <= config.minimum_scale
            {
                match config.zero_scale_policy
                {
                    ZeroScalePolicy::Error =>
                    {
                        return Err(RobustGeometryError::DegenerateDimension {
                            dimension: j,
                            scale: raw_scale,
                        });
                    },
                    ZeroScalePolicy::UnitScale =>
                    {
                        location.push(loc);
                        scale.push(1.0);
                        active_dimensions.push(true);
                    },
                    ZeroScalePolicy::DropDimension =>
                    {
                        location.push(loc);
                        scale.push(raw_scale);
                        active_dimensions.push(false);
                    },
                }
            }
            else
            {
                location.push(loc);
                scale.push(raw_scale);
                active_dimensions.push(true);
            }
        }

        if !active_dimensions.iter().any(|&a| a)
        {
            return Err(RobustGeometryError::NoActiveDimensions);
        }

        Ok(Self {
            location,
            scale,
            active_dimensions,
            config,
        })
    }

    /// Number of fitted dimensions (columns).
    pub fn dimension_count(&self) -> usize {
        self.scale.len()
    }

    /// Transform `data` into the fitted scale-free coordinates.
    ///
    /// Active dimensions map to `(x − location) / scale` (or `x / scale` when
    /// `config.center` is `false`); inactive (dropped) dimensions map to `0.0`.
    ///
    /// # Errors
    ///
    /// Typed [`RobustGeometryError`] on an empty/ragged/non-finite matrix or a
    /// column-count mismatch with the fitted model.
    pub fn transform(&self, data: &Matrix) -> Result<Matrix, RobustGeometryError> {
        validate_matrix(data)?;
        if data.cols != self.dimension_count()
        {
            return Err(RobustGeometryError::DimensionCountMismatch {
                expected: self.dimension_count(),
                found: data.cols,
            });
        }
        let mut out = data.clone();
        for row in &mut out.data
        {
            for (j, value) in row.iter_mut().enumerate()
            {
                *value = self.transform_coordinate(*value, j);
            }
        }
        Ok(out)
    }

    /// Map the transformed coordinates back to the original units.
    ///
    /// Active dimensions map to `x · scale + location` (or `x · scale` when
    /// `config.center` is `false`); inactive dimensions are restored to the
    /// fitted location (their constant value in the training data).
    ///
    /// # Errors
    ///
    /// Typed [`RobustGeometryError`] on an empty/ragged/non-finite matrix or a
    /// column-count mismatch with the fitted model.
    pub fn inverse_transform(&self, data: &Matrix) -> Result<Matrix, RobustGeometryError> {
        validate_matrix(data)?;
        if data.cols != self.dimension_count()
        {
            return Err(RobustGeometryError::DimensionCountMismatch {
                expected: self.dimension_count(),
                found: data.cols,
            });
        }
        let mut out = data.clone();
        for row in &mut out.data
        {
            for (j, value) in row.iter_mut().enumerate()
            {
                *value = if self.active_dimensions[j]
                {
                    if self.config.center
                    {
                        *value * self.scale[j] + self.location[j]
                    }
                    else
                    {
                        *value * self.scale[j]
                    }
                }
                else
                {
                    self.location[j]
                };
            }
        }
        Ok(out)
    }

    /// One coordinate of the forward transform.
    fn transform_coordinate(&self, x: f64, j: usize) -> f64 {
        if self.active_dimensions[j]
        {
            if self.config.center
            {
                (x - self.location[j]) / self.scale[j]
            }
            else
            {
                x / self.scale[j]
            }
        }
        else
        {
            0.0
        }
    }
}

/// A fitted distance model. Each variant documents its exact formula and its
/// invariance group; none claims more than it delivers.
///
/// This enum intentionally has no serde derives: the
/// [`RegularizedMahalanobis`](FittedDistanceMetric::RegularizedMahalanobis)
/// variant embeds a [`Matrix`], which does not serialize.
#[derive(Debug, Clone, PartialEq)]
pub enum FittedDistanceMetric {
    /// Plain Euclidean distance `‖x − y‖₂`. Invariant to rigid motions only;
    /// any rescaling of coordinates changes it.
    RawEuclidean,
    /// The norm-ratio distance
    /// `d(x, y) = ‖x − y‖₂ / max(‖x‖₂, ‖y‖₂, ε)`.
    ///
    /// Invariant to a *common* positive rescaling `x → λx, y → λy` exactly when
    /// `ε` is inactive on both sides (`max(‖x‖, ‖y‖) ≥ ε` before and after);
    /// near the origin the `ε` floor regularizes the ratio (in particular
    /// `d(0, 0) = 0`). **Not** invariant to independent per-coordinate
    /// rescaling, and not translation invariant.
    RelativeNorm {
        /// Strictly positive floor on the denominator.
        epsilon: f64,
    },
    /// `sqrt(Σ_active ((xⱼ − yⱼ) / scaleⱼ)²)` over the scaler's active
    /// dimensions. When the scaler is refit on rescaled data, this distance is
    /// invariant (within floating-point tolerance) to any positive independent
    /// per-coordinate rescaling and to translations. Not invariant to rotations.
    RobustDiagonal {
        /// The fitted per-coordinate scaling model.
        scaler: RobustScaler,
    },
    /// `sqrt((x − y)ᵀ S⁻¹ (x − y))` with
    /// `S = classical covariance + ridge · I`.
    ///
    /// **Honest naming:** the location and scatter are the *classical* mean and
    /// covariance — a minority of gross outliers can distort them, so this is
    /// *not* a robust metric, and the ridge plus floating point break exact
    /// affine equivariance. It is provided as the correlation-aware baseline.
    RegularizedMahalanobis {
        /// The fitted location (classical column means).
        location: Vec<f64>,
        /// The inverse of the ridge-regularized scatter.
        inverse_scatter: Matrix,
        /// The ridge that was added to the scatter's diagonal.
        ridge: f64,
    },
}

impl FittedDistanceMetric {
    /// Fit a [`FittedDistanceMetric::RobustDiagonal`] metric on `data`.
    ///
    /// # Errors
    ///
    /// Propagates [`RobustScaler::fit`] errors.
    pub fn fit_robust_diagonal(
        data: &Matrix,
        config: RobustScalerConfig,
    ) -> Result<Self, RobustGeometryError> {
        Ok(Self::RobustDiagonal {
            scaler: RobustScaler::fit(data, config)?,
        })
    }

    /// Fit a [`FittedDistanceMetric::RegularizedMahalanobis`] metric on `data`:
    /// classical column means, population covariance (`1/n`), `ridge` added to
    /// the diagonal, then a strict Cholesky inversion.
    ///
    /// # Errors
    ///
    /// Typed [`RobustGeometryError`] on invalid input, an invalid ridge, or a
    /// scatter that is not positive definite after regularization
    /// ([`RobustGeometryError::SingularScatter`] — never a silent fallback).
    pub fn fit_regularized_mahalanobis(
        data: &Matrix,
        ridge: f64,
    ) -> Result<Self, RobustGeometryError> {
        validate_matrix(data)?;
        if !(ridge.is_finite() && ridge >= 0.0)
        {
            return Err(RobustGeometryError::InvalidRidge { ridge });
        }
        let (centered, location) = data.center();
        let mut scatter = centered.cov_matrix();
        for i in 0..scatter.rows
        {
            scatter.data[i][i] += ridge;
        }
        let l = strict_cholesky(&scatter)?;
        let l_inv = invert_lower_triangular(&l);
        let inverse_scatter = l_inv.transpose().mul(&l_inv);
        Ok(Self::RegularizedMahalanobis {
            location,
            inverse_scatter,
            ridge,
        })
    }

    /// The number of coordinates this metric expects, when it is fitted
    /// (`None` for the unfitted `RawEuclidean` / `RelativeNorm` variants, which
    /// accept any length).
    pub fn fitted_dimension_count(&self) -> Option<usize> {
        match self
        {
            Self::RawEuclidean | Self::RelativeNorm { .. } => None,
            Self::RobustDiagonal { scaler } => Some(scaler.dimension_count()),
            Self::RegularizedMahalanobis { location, .. } => Some(location.len()),
        }
    }

    /// The distance between two points under this metric.
    ///
    /// # Errors
    ///
    /// Typed [`RobustGeometryError`] on a length mismatch (between the points or
    /// against the fitted model), a non-finite coordinate, or an invalid
    /// `epsilon`. No `NaN` is ever returned.
    pub fn distance(&self, x: &[f64], y: &[f64]) -> Result<f64, RobustGeometryError> {
        validate_point_pair(x, y)?;
        if let Some(expected) = self.fitted_dimension_count()
        {
            if x.len() != expected
            {
                return Err(RobustGeometryError::DimensionCountMismatch {
                    expected,
                    found: x.len(),
                });
            }
        }
        match self
        {
            Self::RawEuclidean => Ok(euclidean(x, y)),
            Self::RelativeNorm { epsilon } =>
            {
                if !(epsilon.is_finite() && *epsilon > 0.0)
                {
                    return Err(RobustGeometryError::InvalidEpsilon { epsilon: *epsilon });
                }
                let denominator = norm(x).max(norm(y)).max(*epsilon);
                Ok(euclidean(x, y) / denominator)
            },
            Self::RobustDiagonal { scaler } =>
            {
                let mut sum = 0.0;
                for j in 0..x.len()
                {
                    if scaler.active_dimensions[j]
                    {
                        let d = (x[j] - y[j]) / scaler.scale[j];
                        sum += d * d;
                    }
                }
                Ok(sum.sqrt())
            },
            Self::RegularizedMahalanobis {
                inverse_scatter, ..
            } =>
            {
                let n = x.len();
                let mut diff = vec![0.0; n];
                for i in 0..n
                {
                    diff[i] = x[i] - y[i];
                }
                let temp = inverse_scatter.mul_vec(&diff);
                let mut d2 = 0.0;
                for i in 0..n
                {
                    d2 += diff[i] * temp[i];
                }
                // S⁻¹ = L⁻ᵀL⁻¹ is positive semi-definite, so d² ≥ 0 in exact
                // arithmetic; clamp the tiny negative rounding residue.
                Ok(d2.max(0.0).sqrt())
            },
        }
    }

    /// Validate that `descriptors` are dimensionally coherent with this metric.
    ///
    /// `RawEuclidean` and `RelativeNorm` sum squared differences of *raw*
    /// coordinates, which is only meaningful when every feature carries the same
    /// physical dimension. `RobustDiagonal` and `RegularizedMahalanobis` divide
    /// by fitted per-coordinate scales first, rendering coordinates
    /// dimensionless, so mixed dimensions are accepted — but the descriptor
    /// count must match the fitted model.
    ///
    /// # Errors
    ///
    /// [`RobustGeometryError::EmptyDescriptors`],
    /// [`RobustGeometryError::IncompatibleFeatureDimensions`], or
    /// [`RobustGeometryError::DescriptorCountMismatch`].
    pub fn validate_feature_descriptors(
        &self,
        descriptors: &[FeatureDescriptor],
    ) -> Result<(), RobustGeometryError> {
        if descriptors.is_empty()
        {
            return Err(RobustGeometryError::EmptyDescriptors);
        }
        if let Some(expected) = self.fitted_dimension_count()
        {
            if descriptors.len() != expected
            {
                return Err(RobustGeometryError::DescriptorCountMismatch {
                    expected,
                    found: descriptors.len(),
                });
            }
        }
        match self
        {
            Self::RawEuclidean | Self::RelativeNorm { .. } =>
            {
                let reference = descriptors[0].dimension;
                for (index, descriptor) in descriptors.iter().enumerate().skip(1)
                {
                    if descriptor.dimension != reference
                    {
                        return Err(RobustGeometryError::IncompatibleFeatureDimensions {
                            first_index: 0,
                            second_index: index,
                        });
                    }
                }
                Ok(())
            },
            Self::RobustDiagonal { .. } | Self::RegularizedMahalanobis { .. } => Ok(()),
        }
    }
}

/// Metadata tying a feature column to its physical dimension.
///
/// This is the units boundary: industrial inputs are converted to coherent SI
/// magnitudes (`f64`) *before* entering a matrix, and this descriptor records
/// what each column means so dimensionally incoherent raw comparisons can be
/// rejected via [`FittedDistanceMetric::validate_feature_descriptors`].
///
/// No serde derives: [`scirust_units::Dimension`] does not serialize.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureDescriptor {
    /// Human-readable feature name.
    pub name: String,
    /// The feature's physical dimension in the SI base-dimension basis.
    pub dimension: scirust_units::Dimension,
}

/// Validate shape and finiteness of a matrix: at least one row and one column,
/// every row exactly `cols` long, every entry finite.
fn validate_matrix(data: &Matrix) -> Result<(), RobustGeometryError> {
    if data.rows == 0 || data.cols == 0 || data.data.len() != data.rows
    {
        return Err(RobustGeometryError::EmptyMatrix);
    }
    for (i, row) in data.data.iter().enumerate()
    {
        if row.len() != data.cols
        {
            return Err(RobustGeometryError::RaggedMatrix {
                row: i,
                expected: data.cols,
                found: row.len(),
            });
        }
        for (j, &value) in row.iter().enumerate()
        {
            if !value.is_finite()
            {
                return Err(RobustGeometryError::NonFiniteValue {
                    row: i,
                    col: j,
                    value,
                });
            }
        }
    }
    Ok(())
}

/// Validate a pair of distance arguments: equal lengths, all coordinates finite.
fn validate_point_pair(x: &[f64], y: &[f64]) -> Result<(), RobustGeometryError> {
    if x.len() != y.len()
    {
        return Err(RobustGeometryError::LengthMismatch {
            left: x.len(),
            right: y.len(),
        });
    }
    for (index, &value) in x.iter().enumerate()
    {
        if !value.is_finite()
        {
            return Err(RobustGeometryError::NonFiniteCoordinate { index, value });
        }
    }
    for (index, &value) in y.iter().enumerate()
    {
        if !value.is_finite()
        {
            return Err(RobustGeometryError::NonFiniteCoordinate { index, value });
        }
    }
    Ok(())
}

/// `‖x − y‖₂` with a fixed left-to-right accumulation order.
fn euclidean(x: &[f64], y: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len()
    {
        let d = x[i] - y[i];
        sum += d * d;
    }
    sum.sqrt()
}

/// `‖x‖₂` with a fixed left-to-right accumulation order.
fn norm(x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for &v in x
    {
        sum += v * v;
    }
    sum.sqrt()
}

/// Strict Cholesky factorization: returns the lower factor `L` with `M = LLᵀ`,
/// or the index of the first non-positive (or non-finite) pivot.
///
/// This is deliberately distinct from the crate's private regularizing
/// `cholesky` (used by the historical Mahalanobis helpers), which silently
/// substitutes a ridge pivot: here a non-positive-definite input is a **typed
/// error**, because the caller controls regularization explicitly through the
/// `ridge` parameter of
/// [`FittedDistanceMetric::fit_regularized_mahalanobis`].
fn strict_cholesky(m: &Matrix) -> Result<Matrix, RobustGeometryError> {
    let n = m.rows;
    let mut l = Matrix::zeros(n, n);
    for i in 0..n
    {
        for j in 0..=i
        {
            let mut s = 0.0;
            for k in 0..j
            {
                s += l.data[i][k] * l.data[j][k];
            }
            if i == j
            {
                let val = m.data[i][i] - s;
                if !(val.is_finite() && val > 0.0)
                {
                    return Err(RobustGeometryError::SingularScatter { pivot_index: i });
                }
                l.data[i][j] = val.sqrt();
            }
            else
            {
                l.data[i][j] = (m.data[i][j] - s) / l.data[j][j];
            }
        }
    }
    Ok(l)
}

// ─────────────────── Robust affine scatter (OGK, phase 4E.1) ───────────────────
//
// The metrics above are diagonal (`RobustScaler`) or classical
// (`RegularizedMahalanobis`); neither resists *multivariate* outliers that are
// unremarkable coordinate-by-coordinate (a point off the correlation axis). This
// adds a deterministic Orthogonalized Gnanadesikan–Kettenring (OGK) location and
// scatter estimator (Maronna & Zamar, 2002), reusing this crate's `jacobi_eigen`
// and `scirust-stats` robust scales — no new dependency edge, no RNG.
//
// Equivariance actually achieved (never overstated). OGK is **exactly**
// equivariant under translation and per-coordinate positive scaling, and only
// **approximately** affine-equivariant: it is *not* exactly affine-equivariant in
// finite samples — only the classical covariance (in exact arithmetic) and an
// exact MCD are. Rotation behaviour is *measured* in the benchmark, not claimed.

/// The robust univariate scale driving OGK's pairwise step and projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RobustUnivariateScale {
    /// Median absolute deviation, scaled to normal consistency.
    MedianAbsoluteDeviation,
    /// Interquartile range, scaled to normal consistency.
    InterquartileRange,
}

/// The multivariate location/scatter estimator.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RobustScatterMethod {
    /// Classical column means and population covariance — a baseline inside this
    /// API (not robust to multivariate outliers; kept for honest comparison).
    Classical,
    /// Deterministic OGK robust location and scatter.
    Ogk {
        /// The univariate scale used throughout.
        scale: RobustUnivariateScale,
        /// Apply the hard-reweighting refinement (a χ²-cutoff C-step loop).
        reweight: bool,
    },
}

/// What equivariance a fitted model actually achieves — stated, not overstated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AchievedEquivariance {
    /// Classical covariance: affine-equivariant in exact arithmetic only (the
    /// ridge and floating point break exact equivariance).
    AffineExactArithmetic,
    /// OGK: exactly equivariant under translation and per-coordinate positive
    /// scaling; only approximately affine-equivariant (rotations pass through the
    /// eigenbasis but are not reproduced exactly in finite samples).
    TranslationScalingExactApproximateAffine,
}

/// Configuration for [`RobustScatterModel::fit`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RobustScatterConfig {
    /// The estimator.
    pub method: RobustScatterMethod,
    /// Ridge added to the scatter diagonal before inversion (`>= 0`). `0.0`
    /// leaves the estimate unregularized; a non-positive-definite scatter then
    /// surfaces as [`RobustGeometryError::SingularScatter`] — never a silent
    /// classical fallback.
    pub ridge: f64,
    /// Maximum hard-reweighting iterations (used only when `reweight` is true).
    pub maximum_iterations: usize,
    /// Relative Frobenius change in the scatter below which reweighting stops.
    pub relative_tolerance: f64,
    /// A per-coordinate robust scale at or below this floor marks the dimension
    /// inactive (and is floored to a positive value to avoid division blow-up).
    /// Must be finite and `>= 0`.
    pub minimum_scale: f64,
}

impl Default for RobustScatterConfig {
    fn default() -> Self {
        Self {
            method: RobustScatterMethod::Ogk {
                scale: RobustUnivariateScale::MedianAbsoluteDeviation,
                reweight: true,
            },
            ridge: 0.0,
            maximum_iterations: 10,
            relative_tolerance: 1.0e-9,
            minimum_scale: 1.0e-12,
        }
    }
}

/// A machine-readable account of a robust-scatter fit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobustScatterReport {
    /// Which estimator produced the model.
    pub method: RobustScatterMethod,
    /// The equivariance actually achieved.
    pub achieved_equivariance: AchievedEquivariance,
    /// Rows contributing to the final estimate (equals the row count when no
    /// reweighting was applied).
    pub effective_sample_count: usize,
    /// Rows rejected by the reweighting cutoff (`0` when no reweighting).
    pub reweighted_outlier_count: usize,
    /// Reweighting iterations performed (`0` when no reweighting).
    pub iterations: usize,
    /// Whether reweighting reached its stability/tolerance criterion.
    pub converged: bool,
    /// Non-fatal notes (degenerate dimensions, skipped reweighting, …).
    pub warnings: Vec<String>,
}

/// A fitted robust multivariate location/scatter model.
///
/// Not `Serialize`/`Deserialize`: it holds [`Matrix`] values, which (by this
/// crate's convention, like [`FittedDistanceMetric`]) do not implement serde.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustScatterModel {
    /// Robust location (one entry per column).
    pub location: Vec<f64>,
    /// Robust scatter matrix (`p × p`, ridge already added).
    pub scatter: Matrix,
    /// Inverse of `scatter`.
    pub inverse_scatter: Matrix,
    /// Per-dimension activity flag (`false` where the robust scale was at or
    /// below `minimum_scale`).
    pub active_dimensions: Vec<bool>,
    /// The fit report.
    pub report: RobustScatterReport,
}

impl RobustScatterModel {
    /// Fit a robust location/scatter model on `data` (rows = observations).
    ///
    /// # Errors
    ///
    /// Typed [`RobustGeometryError`] on an empty/ragged/non-finite matrix, fewer
    /// than two rows, an invalid ridge or `minimum_scale`, a robust-scale failure
    /// ([`RobustGeometryError::ScatterScaleEstimation`]), a non-finite scale
    /// ([`RobustGeometryError::NonFiniteScale`]), or a scatter that is not
    /// positive definite after regularization
    /// ([`RobustGeometryError::SingularScatter`]).
    pub fn fit(data: &Matrix, config: RobustScatterConfig) -> Result<Self, RobustGeometryError> {
        validate_matrix(data)?;
        if !(config.ridge.is_finite() && config.ridge >= 0.0)
        {
            return Err(RobustGeometryError::InvalidRidge {
                ridge: config.ridge,
            });
        }
        if !(config.minimum_scale.is_finite() && config.minimum_scale >= 0.0)
        {
            return Err(RobustGeometryError::InvalidMinimumScale {
                minimum_scale: config.minimum_scale,
            });
        }
        if data.rows < 2
        {
            return Err(RobustGeometryError::InsufficientSamples {
                required: 2,
                found: data.rows,
            });
        }

        match config.method
        {
            RobustScatterMethod::Classical => Self::fit_classical(data, config),
            RobustScatterMethod::Ogk { scale, reweight } =>
            {
                Self::fit_ogk(data, config, scale, reweight)
            },
        }
    }

    fn fit_classical(
        data: &Matrix,
        config: RobustScatterConfig,
    ) -> Result<Self, RobustGeometryError> {
        let (centered, location) = data.center();
        let mut scatter = centered.cov_matrix();
        for i in 0..scatter.rows
        {
            scatter.data[i][i] += config.ridge;
        }
        let inverse_scatter = invert_positive_definite(&scatter)?;
        Ok(Self {
            location,
            scatter,
            inverse_scatter,
            active_dimensions: vec![true; data.cols],
            report: RobustScatterReport {
                method: config.method,
                achieved_equivariance: AchievedEquivariance::AffineExactArithmetic,
                effective_sample_count: data.rows,
                reweighted_outlier_count: 0,
                iterations: 0,
                converged: true,
                warnings: Vec::new(),
            },
        })
    }

    fn fit_ogk(
        data: &Matrix,
        config: RobustScatterConfig,
        scale_method: RobustUnivariateScale,
        reweight: bool,
    ) -> Result<Self, RobustGeometryError> {
        let n = data.rows;
        let p = data.cols;
        let mut warnings: Vec<String> = Vec::new();

        // Step 1 — robust per-column scales; flag and floor degenerate columns.
        let mut floored = vec![0.0_f64; p];
        let mut active = vec![true; p];
        for j in 0..p
        {
            let column: Vec<f64> = (0..n).map(|i| data.data[i][j]).collect();
            let scale = univariate_scale(&column, scale_method)?;
            if !scale.is_finite()
            {
                return Err(RobustGeometryError::NonFiniteScale {
                    dimension: j,
                    scale,
                });
            }
            active[j] = scale > config.minimum_scale;
            if !active[j]
            {
                warnings.push(format!(
                    "dimension {j} has robust scale {scale:.3e} <= minimum_scale {:.3e}; marked inactive",
                    config.minimum_scale
                ));
            }
            floored[j] = scale.max(config.minimum_scale).max(f64::MIN_POSITIVE);
        }

        // Scaled data Y = X D^{-1}.
        let mut y = Matrix::zeros(n, p);
        for i in 0..n
        {
            for (j, &scale) in floored.iter().enumerate()
            {
                y.data[i][j] = data.data[i][j] / scale;
            }
        }

        // Step 2 — the GK pairwise matrix U (unit diagonal, symmetric).
        let mut u = Matrix::zeros(p, p);
        for j in 0..p
        {
            u.data[j][j] = 1.0;
        }
        for j in 0..p
        {
            for k in (j + 1)..p
            {
                let mut sum = vec![0.0_f64; n];
                let mut diff = vec![0.0_f64; n];
                for i in 0..n
                {
                    sum[i] = y.data[i][j] + y.data[i][k];
                    diff[i] = y.data[i][j] - y.data[i][k];
                }
                let scale_sum = univariate_scale(&sum, scale_method)?;
                let scale_diff = univariate_scale(&diff, scale_method)?;
                let entry = 0.25 * (scale_sum * scale_sum - scale_diff * scale_diff);
                u.data[j][k] = entry;
                u.data[k][j] = entry;
            }
        }

        // Step 3 — eigendecompose U with a deterministic order and sign.
        let (raw_values, raw_vectors) = crate::jacobi_eigen(&u);
        let vectors = canonicalize_eigenvectors(&raw_values, raw_vectors);

        // Step 4 — project the scaled data onto the eigenbasis: V = Y E.
        let mut projected = Matrix::zeros(n, p);
        for i in 0..n
        {
            for (l, vector) in vectors.iter().enumerate()
            {
                let mut acc = 0.0;
                for (k, &component) in vector.iter().enumerate()
                {
                    acc += y.data[i][k] * component;
                }
                projected.data[i][l] = acc;
            }
        }

        // Step 5 — robust scale and location of every projected coordinate.
        let mut gamma = vec![0.0_f64; p];
        let mut nu = vec![0.0_f64; p];
        for l in 0..p
        {
            let column: Vec<f64> = (0..n).map(|i| projected.data[i][l]).collect();
            gamma[l] = univariate_scale(&column, scale_method)?;
            nu[l] = describe::median(&column);
        }

        // Step 6 — assemble in Y-space, then Step 7 — map back to X-space.
        // Σ_X[a][b] = d_a d_b Σᵧ[a][b],  Σᵧ[a][b] = Σ_l γ_l² e_l[a] e_l[b];
        // μ_X[a]    = d_a μᵧ[a],         μᵧ[a]    = Σ_l ν_l e_l[a].
        let mut scatter = Matrix::zeros(p, p);
        for a in 0..p
        {
            for b in 0..p
            {
                let mut acc = 0.0;
                for l in 0..p
                {
                    acc += gamma[l] * gamma[l] * vectors[l][a] * vectors[l][b];
                }
                scatter.data[a][b] = floored[a] * floored[b] * acc;
            }
        }
        // Symmetrize away floating-point asymmetry before factorization.
        for a in 0..p
        {
            for b in (a + 1)..p
            {
                let mean = 0.5 * (scatter.data[a][b] + scatter.data[b][a]);
                scatter.data[a][b] = mean;
                scatter.data[b][a] = mean;
            }
        }
        let mut location = vec![0.0_f64; p];
        for a in 0..p
        {
            let mut acc = 0.0;
            for l in 0..p
            {
                acc += nu[l] * vectors[l][a];
            }
            location[a] = floored[a] * acc;
        }

        for a in 0..p
        {
            scatter.data[a][a] += config.ridge;
        }
        let mut inverse_scatter = invert_positive_definite(&scatter)?;

        let mut effective_sample_count = n;
        let mut reweighted_outlier_count = 0;
        let mut iterations = 0;
        let mut converged = true;

        // Step 8 — optional hard-reweighting refinement (χ²-cutoff C-steps).
        if reweight
        {
            converged = false;
            let chi_median = ChiSquared::new(p as f64).quantile(0.5);
            let chi_cutoff = ChiSquared::new(p as f64).quantile(0.975);
            let mut previous: Option<Vec<usize>> = None;

            for _ in 0..config.maximum_iterations.max(1)
            {
                iterations += 1;
                let distances = squared_distances(data, &location, &inverse_scatter);
                let median_distance = describe::median(&distances);
                let correction = if chi_median > 0.0
                {
                    (median_distance / chi_median).max(f64::MIN_POSITIVE)
                }
                else
                {
                    1.0
                };
                let threshold = chi_cutoff * correction;
                let retained: Vec<usize> = (0..n).filter(|&i| distances[i] <= threshold).collect();

                if retained.len() < p + 1
                {
                    warnings.push(format!(
                        "reweighting stopped: only {} of {n} rows passed the χ² cutoff (< p+1 = {})",
                        retained.len(),
                        p + 1
                    ));
                    converged = true;
                    break;
                }

                let subset = select_rows(data, &retained);
                let (centered, new_location) = subset.center();
                let mut new_scatter = centered.cov_matrix();
                for a in 0..p
                {
                    new_scatter.data[a][a] += config.ridge;
                }
                let new_inverse = invert_positive_definite(&new_scatter)?;

                let stable = previous.as_ref() == Some(&retained);
                let change = frobenius_relative_change(&scatter, &new_scatter);

                location = new_location;
                scatter = new_scatter;
                inverse_scatter = new_inverse;
                effective_sample_count = retained.len();
                reweighted_outlier_count = n - retained.len();

                if stable || change < config.relative_tolerance
                {
                    converged = true;
                    break;
                }
                previous = Some(retained);
            }
        }

        Ok(Self {
            location,
            scatter,
            inverse_scatter,
            active_dimensions: active,
            report: RobustScatterReport {
                method: config.method,
                achieved_equivariance:
                    AchievedEquivariance::TranslationScalingExactApproximateAffine,
                effective_sample_count,
                reweighted_outlier_count,
                iterations,
                converged,
                warnings,
            },
        })
    }

    /// Squared robust Mahalanobis distance of `point` from the fitted location.
    ///
    /// # Errors
    ///
    /// [`RobustGeometryError::DimensionCountMismatch`] on a length mismatch;
    /// [`RobustGeometryError::NonFiniteCoordinate`] on a non-finite coordinate.
    pub fn mahalanobis_squared(&self, point: &[f64]) -> Result<f64, RobustGeometryError> {
        if point.len() != self.location.len()
        {
            return Err(RobustGeometryError::DimensionCountMismatch {
                expected: self.location.len(),
                found: point.len(),
            });
        }
        for (index, &value) in point.iter().enumerate()
        {
            if !value.is_finite()
            {
                return Err(RobustGeometryError::NonFiniteCoordinate { index, value });
            }
        }
        let mut difference = vec![0.0_f64; point.len()];
        for i in 0..point.len()
        {
            difference[i] = point[i] - self.location[i];
        }
        let transformed = self.inverse_scatter.mul_vec(&difference);
        let mut squared = 0.0;
        for i in 0..point.len()
        {
            squared += difference[i] * transformed[i];
        }
        Ok(squared)
    }

    /// Robust Mahalanobis distance (the non-negative square root of
    /// [`Self::mahalanobis_squared`]).
    ///
    /// # Errors
    ///
    /// Same as [`Self::mahalanobis_squared`].
    pub fn mahalanobis(&self, point: &[f64]) -> Result<f64, RobustGeometryError> {
        Ok(self.mahalanobis_squared(point)?.max(0.0).sqrt())
    }
}

/// A robust univariate scale, scaled to normal consistency.
fn univariate_scale(
    values: &[f64],
    method: RobustUnivariateScale,
) -> Result<f64, RobustGeometryError> {
    match method
    {
        RobustUnivariateScale::MedianAbsoluteDeviation =>
        {
            median_absolute_deviation(values, MadConsistency::Normal)
                .map_err(|source| RobustGeometryError::ScatterScaleEstimation { source })
        },
        RobustUnivariateScale::InterquartileRange =>
        {
            let raw = interquartile_range(values)
                .map_err(|source| RobustGeometryError::ScatterScaleEstimation { source })?;
            // Normal consistency: IQR / (2·Φ⁻¹(0.75)) matches σ under normality.
            let factor = 2.0 * Normal::standard().quantile(0.75);
            Ok(raw / factor)
        },
    }
}

/// Sort eigenpairs by eigenvalue descending (stable on ties by original index)
/// and fix each eigenvector's sign so its first entry above a small tolerance is
/// positive — a fully deterministic basis regardless of the solver's output.
fn canonicalize_eigenvectors(values: &[f64], vectors: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
    order
        .into_iter()
        .map(|i| {
            let mut vector = vectors[i].clone();
            let pivot = vector
                .iter()
                .copied()
                .find(|value| value.abs() > 1.0e-12)
                .unwrap_or(1.0);
            if pivot < 0.0
            {
                for value in &mut vector
                {
                    *value = -*value;
                }
            }
            vector
        })
        .collect()
}

/// Invert a positive-definite matrix through a strict Cholesky factorization.
fn invert_positive_definite(matrix: &Matrix) -> Result<Matrix, RobustGeometryError> {
    let lower = strict_cholesky(matrix)?;
    let lower_inverse = invert_lower_triangular(&lower);
    Ok(lower_inverse.transpose().mul(&lower_inverse))
}

/// Per-row squared Mahalanobis distances of `data` from `location` under
/// `inverse_scatter`.
fn squared_distances(data: &Matrix, location: &[f64], inverse_scatter: &Matrix) -> Vec<f64> {
    let p = data.cols;
    (0..data.rows)
        .map(|i| {
            let mut difference = vec![0.0_f64; p];
            for j in 0..p
            {
                difference[j] = data.data[i][j] - location[j];
            }
            let transformed = inverse_scatter.mul_vec(&difference);
            let mut squared = 0.0;
            for j in 0..p
            {
                squared += difference[j] * transformed[j];
            }
            squared
        })
        .collect()
}

/// Build the sub-matrix of `data` selecting `rows` in order.
fn select_rows(data: &Matrix, rows: &[usize]) -> Matrix {
    let mut out = Matrix::zeros(rows.len(), data.cols);
    for (target, &source) in rows.iter().enumerate()
    {
        out.data[target] = data.data[source].clone();
    }
    out
}

/// Relative Frobenius change `‖a − b‖_F / ‖b‖_F` (falls back to `‖a − b‖_F` when
/// `b` is all zeros).
fn frobenius_relative_change(a: &Matrix, b: &Matrix) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for i in 0..a.rows
    {
        for j in 0..a.cols
        {
            let delta = a.data[i][j] - b.data[i][j];
            numerator += delta * delta;
            denominator += b.data[i][j] * b.data[i][j];
        }
    }
    if denominator > 0.0
    {
        (numerator / denominator).sqrt()
    }
    else
    {
        numerator.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "expected {expected}, got {actual} (tol {tol})"
        );
    }

    fn sample_matrix() -> Matrix {
        Matrix::from_slice(&[
            &[1.0, 100.0, -3.0],
            &[2.0, 200.0, -1.0],
            &[3.0, 300.0, 1.0],
            &[4.0, 400.0, 3.0],
            &[5.0, 500.0, 5.0],
        ])
    }

    fn mad_config() -> RobustScalerConfig {
        RobustScalerConfig::default()
    }

    #[test]
    fn scaler_round_trip_restores_data() {
        let data = sample_matrix();
        for method in [
            RobustScaleMethod::StandardDeviation,
            RobustScaleMethod::MedianAbsoluteDeviation,
            RobustScaleMethod::InterquartileRange,
        ]
        {
            for center in [true, false]
            {
                let config = RobustScalerConfig {
                    center,
                    scale_method: method,
                    ..RobustScalerConfig::default()
                };
                let scaler = RobustScaler::fit(&data, config).unwrap();
                let transformed = scaler.transform(&data).unwrap();
                let restored = scaler.inverse_transform(&transformed).unwrap();
                for i in 0..data.rows
                {
                    for j in 0..data.cols
                    {
                        assert_close(restored.data[i][j], data.data[i][j], 1e-9);
                    }
                }
            }
        }
    }

    #[test]
    fn scaler_transform_centers_and_scales() {
        // Column 0 of the sample: median 3, normal-consistent MAD 1.4826.
        let data = sample_matrix();
        let scaler = RobustScaler::fit(&data, mad_config()).unwrap();
        assert_close(scaler.location[0], 3.0, 1e-12);
        assert_close(scaler.scale[0], 1.482_602_218_505_602, 1e-9);
        let transformed = scaler.transform(&data).unwrap();
        // Median row maps to 0 in every column.
        for j in 0..data.cols
        {
            assert_close(transformed.data[2][j], 0.0, 1e-12);
        }
    }

    #[test]
    fn scaler_fit_is_deterministic() {
        let data = sample_matrix();
        let a = RobustScaler::fit(&data, mad_config()).unwrap();
        let b = RobustScaler::fit(&data, mad_config()).unwrap();
        assert_eq!(a, b);
        for (x, y) in a.scale.iter().zip(b.scale.iter())
        {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn scaler_is_row_order_invariant_for_order_statistics() {
        // Median/MAD/IQR are order statistics: permuting rows leaves the sorted
        // column unchanged, so the fitted model is bit-identical.
        let data = sample_matrix();
        let permuted = Matrix::from_slice(&[
            &[5.0, 500.0, 5.0],
            &[3.0, 300.0, 1.0],
            &[1.0, 100.0, -3.0],
            &[4.0, 400.0, 3.0],
            &[2.0, 200.0, -1.0],
        ]);
        for method in [
            RobustScaleMethod::MedianAbsoluteDeviation,
            RobustScaleMethod::InterquartileRange,
        ]
        {
            let config = RobustScalerConfig {
                scale_method: method,
                ..RobustScalerConfig::default()
            };
            let a = RobustScaler::fit(&data, config).unwrap();
            let b = RobustScaler::fit(&permuted, config).unwrap();
            for (x, y) in a.scale.iter().zip(b.scale.iter())
            {
                assert_eq!(x.to_bits(), y.to_bits());
            }
            for (x, y) in a.location.iter().zip(b.location.iter())
            {
                assert_eq!(x.to_bits(), y.to_bits());
            }
        }
    }

    #[test]
    fn zero_scale_error_policy_rejects_constant_column() {
        let data = Matrix::from_slice(&[&[1.0, 7.0], &[2.0, 7.0], &[3.0, 7.0]]);
        let result = RobustScaler::fit(&data, mad_config());
        assert!(matches!(
            result,
            Err(RobustGeometryError::DegenerateDimension { dimension: 1, .. })
        ));
    }

    #[test]
    fn zero_scale_unit_policy_keeps_constant_column() {
        let data = Matrix::from_slice(&[&[1.0, 7.0], &[2.0, 7.0], &[3.0, 7.0]]);
        let config = RobustScalerConfig {
            zero_scale_policy: ZeroScalePolicy::UnitScale,
            ..RobustScalerConfig::default()
        };
        let scaler = RobustScaler::fit(&data, config).unwrap();
        assert_close(scaler.scale[1], 1.0, 0.0);
        assert!(scaler.active_dimensions[1]);
        let transformed = scaler.transform(&data).unwrap();
        // (7 − 7) / 1 = 0.
        assert_close(transformed.data[0][1], 0.0, 0.0);
    }

    #[test]
    fn zero_scale_drop_policy_drops_and_restores_constant_column() {
        let data = Matrix::from_slice(&[&[1.0, 7.0], &[2.0, 7.0], &[3.0, 7.0]]);
        let config = RobustScalerConfig {
            zero_scale_policy: ZeroScalePolicy::DropDimension,
            ..RobustScalerConfig::default()
        };
        let scaler = RobustScaler::fit(&data, config).unwrap();
        assert!(!scaler.active_dimensions[1]);
        let transformed = scaler.transform(&data).unwrap();
        assert_close(transformed.data[0][1], 0.0, 0.0);
        let restored = scaler.inverse_transform(&transformed).unwrap();
        assert_close(restored.data[0][1], 7.0, 1e-12);
        assert_close(restored.data[0][0], 1.0, 1e-12);
    }

    #[test]
    fn all_constant_matrix_drops_everything() {
        let data = Matrix::from_slice(&[&[7.0, 7.0], &[7.0, 7.0]]);
        let config = RobustScalerConfig {
            zero_scale_policy: ZeroScalePolicy::DropDimension,
            ..RobustScalerConfig::default()
        };
        assert_eq!(
            RobustScaler::fit(&data, config),
            Err(RobustGeometryError::NoActiveDimensions)
        );
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let empty = Matrix::zeros(0, 0);
        assert_eq!(
            RobustScaler::fit(&empty, mad_config()),
            Err(RobustGeometryError::EmptyMatrix)
        );

        let mut ragged = sample_matrix();
        ragged.data[1].pop();
        assert!(matches!(
            RobustScaler::fit(&ragged, mad_config()),
            Err(RobustGeometryError::RaggedMatrix {
                row: 1,
                expected: 3,
                found: 2
            })
        ));

        let mut non_finite = sample_matrix();
        non_finite.data[2][1] = f64::NAN;
        assert!(matches!(
            RobustScaler::fit(&non_finite, mad_config()),
            Err(RobustGeometryError::NonFiniteValue { row: 2, col: 1, .. })
        ));

        let bad_minimum = RobustScalerConfig {
            minimum_scale: -1.0,
            ..RobustScalerConfig::default()
        };
        assert!(matches!(
            RobustScaler::fit(&sample_matrix(), bad_minimum),
            Err(RobustGeometryError::InvalidMinimumScale { .. })
        ));

        let one_row = Matrix::from_slice(&[&[1.0, 2.0]]);
        let std_config = RobustScalerConfig {
            scale_method: RobustScaleMethod::StandardDeviation,
            ..RobustScalerConfig::default()
        };
        assert_eq!(
            RobustScaler::fit(&one_row, std_config),
            Err(RobustGeometryError::InsufficientSamples {
                required: 2,
                found: 1
            })
        );

        let narrow = Matrix::from_slice(&[&[1.0], &[2.0]]);
        let scaler = RobustScaler::fit(&sample_matrix(), mad_config()).unwrap();
        assert!(matches!(
            scaler.transform(&narrow),
            Err(RobustGeometryError::DimensionCountMismatch {
                expected: 3,
                found: 1
            })
        ));
    }

    #[test]
    fn robust_diagonal_is_invariant_to_coordinate_rescaling_after_refit() {
        let data = sample_matrix();
        let metric = FittedDistanceMetric::fit_robust_diagonal(&data, mad_config()).unwrap();

        let factors = [1.0e-3, 1.0e6, 42.0];
        let mut rescaled = data.clone();
        for row in &mut rescaled.data
        {
            for (j, &f) in factors.iter().enumerate()
            {
                row[j] *= f;
            }
        }
        let metric_rescaled =
            FittedDistanceMetric::fit_robust_diagonal(&rescaled, mad_config()).unwrap();

        for i in 0..data.rows
        {
            for k in (i + 1)..data.rows
            {
                let d = metric.distance(&data.data[i], &data.data[k]).unwrap();
                let dr = metric_rescaled
                    .distance(&rescaled.data[i], &rescaled.data[k])
                    .unwrap();
                assert_close(dr, d, 1e-9 * (1.0 + d));
            }
        }
    }

    #[test]
    fn robust_diagonal_is_translation_invariant_after_refit() {
        let data = sample_matrix();
        let metric = FittedDistanceMetric::fit_robust_diagonal(&data, mad_config()).unwrap();

        let shift = [1000.0, -500.0, 3.0];
        let mut shifted = data.clone();
        for row in &mut shifted.data
        {
            for (j, &s) in shift.iter().enumerate()
            {
                row[j] += s;
            }
        }
        let metric_shifted =
            FittedDistanceMetric::fit_robust_diagonal(&shifted, mad_config()).unwrap();

        for i in 0..data.rows
        {
            for k in (i + 1)..data.rows
            {
                let d = metric.distance(&data.data[i], &data.data[k]).unwrap();
                let ds = metric_shifted
                    .distance(&shifted.data[i], &shifted.data[k])
                    .unwrap();
                assert_close(ds, d, 1e-9 * (1.0 + d));
            }
        }
    }

    #[test]
    fn relative_norm_is_invariant_to_common_scaling() {
        let metric = FittedDistanceMetric::RelativeNorm { epsilon: 1.0e-12 };
        let x = [3.0, 4.0];
        let y = [6.0, 8.0];
        let d = metric.distance(&x, &y).unwrap();
        let lambda = 1.0e6;
        let xs = [x[0] * lambda, x[1] * lambda];
        let ys = [y[0] * lambda, y[1] * lambda];
        let ds = metric.distance(&xs, &ys).unwrap();
        assert_close(ds, d, 1e-12 * (1.0 + d));
        // Hand-computed: ‖x−y‖ = 5, max norm = 10, d = 0.5.
        assert_close(d, 0.5, 1e-12);
    }

    #[test]
    fn relative_norm_handles_zero_vectors_without_nan() {
        let metric = FittedDistanceMetric::RelativeNorm { epsilon: 1.0e-9 };
        let zero = [0.0, 0.0];
        let d = metric.distance(&zero, &zero).unwrap();
        assert_close(d, 0.0, 0.0);
        assert!(!d.is_nan());
    }

    #[test]
    fn relative_norm_rejects_invalid_epsilon() {
        for epsilon in [0.0, -1.0, f64::NAN, f64::INFINITY]
        {
            let metric = FittedDistanceMetric::RelativeNorm { epsilon };
            assert!(matches!(
                metric.distance(&[1.0], &[2.0]),
                Err(RobustGeometryError::InvalidEpsilon { .. })
            ));
        }
    }

    #[test]
    fn raw_euclidean_matches_hand_computed() {
        let metric = FittedDistanceMetric::RawEuclidean;
        let d = metric.distance(&[0.0, 0.0], &[3.0, 4.0]).unwrap();
        assert_close(d, 5.0, 1e-12);
    }

    #[test]
    fn mahalanobis_on_identity_covariance_matches_euclidean() {
        // Two symmetric unit-variance uncorrelated columns: population
        // covariance is (2/5)·diag on this fixture; use an explicit isotropic
        // fixture instead where covariance is exactly I.
        let data = Matrix::from_slice(&[
            &[1.0, 0.0],
            &[-1.0, 0.0],
            &[0.0, 1.0],
            &[0.0, -1.0],
            &[1.0, 0.0],
            &[-1.0, 0.0],
            &[0.0, 1.0],
            &[0.0, -1.0],
        ]);
        // Population covariance = diag(0.5, 0.5); with ridge 0 the metric is
        // sqrt(2)·Euclidean.
        let metric = FittedDistanceMetric::fit_regularized_mahalanobis(&data, 0.0).unwrap();
        let d = metric.distance(&[0.0, 0.0], &[1.0, 0.0]).unwrap();
        assert_close(d, (2.0_f64).sqrt(), 1e-9);
    }

    #[test]
    fn mahalanobis_singular_scatter_is_a_typed_error_not_a_fallback() {
        // Second column is an exact copy of the first: covariance is singular.
        let data = Matrix::from_slice(&[&[1.0, 1.0], &[2.0, 2.0], &[3.0, 3.0], &[4.0, 4.0]]);
        let result = FittedDistanceMetric::fit_regularized_mahalanobis(&data, 0.0);
        assert!(matches!(
            result,
            Err(RobustGeometryError::SingularScatter { .. })
        ));
        // An explicit ridge resolves it.
        let metric = FittedDistanceMetric::fit_regularized_mahalanobis(&data, 1.0e-6).unwrap();
        let d = metric.distance(&[1.0, 1.0], &[2.0, 2.0]).unwrap();
        assert!(d.is_finite() && d > 0.0);
    }

    #[test]
    fn mahalanobis_rejects_invalid_ridge() {
        let data = sample_matrix();
        for ridge in [-1.0, f64::NAN, f64::INFINITY]
        {
            assert!(matches!(
                FittedDistanceMetric::fit_regularized_mahalanobis(&data, ridge),
                Err(RobustGeometryError::InvalidRidge { .. })
            ));
        }
    }

    #[test]
    fn distance_validates_inputs() {
        let metric = FittedDistanceMetric::RawEuclidean;
        assert!(matches!(
            metric.distance(&[1.0, 2.0], &[1.0]),
            Err(RobustGeometryError::LengthMismatch { left: 2, right: 1 })
        ));
        assert!(matches!(
            metric.distance(&[f64::NAN], &[1.0]),
            Err(RobustGeometryError::NonFiniteCoordinate { index: 0, .. })
        ));
        assert!(matches!(
            metric.distance(&[1.0], &[f64::INFINITY]),
            Err(RobustGeometryError::NonFiniteCoordinate { index: 0, .. })
        ));

        let fitted =
            FittedDistanceMetric::fit_robust_diagonal(&sample_matrix(), mad_config()).unwrap();
        assert!(matches!(
            fitted.distance(&[1.0], &[2.0]),
            Err(RobustGeometryError::DimensionCountMismatch {
                expected: 3,
                found: 1
            })
        ));
    }

    #[test]
    fn feature_descriptor_validation_enforces_uniform_dimension_for_raw_metrics() {
        use scirust_units::Dimension;
        let uniform = vec![
            FeatureDescriptor {
                name: "x".to_string(),
                dimension: Dimension::LENGTH,
            },
            FeatureDescriptor {
                name: "y".to_string(),
                dimension: Dimension::LENGTH,
            },
        ];
        let mixed = vec![
            FeatureDescriptor {
                name: "position".to_string(),
                dimension: Dimension::LENGTH,
            },
            FeatureDescriptor {
                name: "pressure".to_string(),
                dimension: Dimension::PRESSURE,
            },
        ];

        let raw = FittedDistanceMetric::RawEuclidean;
        assert!(raw.validate_feature_descriptors(&uniform).is_ok());
        assert!(matches!(
            raw.validate_feature_descriptors(&mixed),
            Err(RobustGeometryError::IncompatibleFeatureDimensions {
                first_index: 0,
                second_index: 1
            })
        ));
        assert_eq!(
            raw.validate_feature_descriptors(&[]),
            Err(RobustGeometryError::EmptyDescriptors)
        );

        // A fitted diagonal metric renders coordinates dimensionless: mixed
        // dimensions are fine, but the count must match.
        let two_column = Matrix::from_slice(&[&[1.0, 10.0], &[2.0, 30.0], &[3.0, 20.0]]);
        let fitted = FittedDistanceMetric::fit_robust_diagonal(&two_column, mad_config()).unwrap();
        assert!(fitted.validate_feature_descriptors(&mixed).is_ok());
        let three = vec![mixed[0].clone(), mixed[1].clone(), mixed[0].clone()];
        assert!(matches!(
            fitted.validate_feature_descriptors(&three),
            Err(RobustGeometryError::DescriptorCountMismatch {
                expected: 2,
                found: 3
            })
        ));
    }

    #[test]
    fn scaler_serialization_round_trips() {
        let scaler = RobustScaler::fit(&sample_matrix(), mad_config()).unwrap();
        let json = serde_json::to_string(&scaler).unwrap();
        let back: RobustScaler = serde_json::from_str(&json).unwrap();
        assert_eq!(scaler, back);
    }

    #[test]
    fn overflowing_scale_estimate_is_a_typed_error_not_a_silent_active_dimension() {
        // MAD path: raw MAD 1.7e308 overflows when multiplied by the
        // normal-consistency factor. Before the finiteness gate this produced
        // scale = +inf marked ACTIVE (`inf <= minimum_scale` is false),
        // silently bypassing every zero-scale policy.
        let data = Matrix::from_slice(&[&[0.0], &[1.7e308], &[-1.7e308]]);
        for policy in [
            ZeroScalePolicy::Error,
            ZeroScalePolicy::UnitScale,
            ZeroScalePolicy::DropDimension,
        ]
        {
            let config = RobustScalerConfig {
                zero_scale_policy: policy,
                ..RobustScalerConfig::default()
            };
            assert!(matches!(
                RobustScaler::fit(&data, config),
                Err(RobustGeometryError::NonFiniteScale { dimension: 0, .. })
            ));
        }

        // Standard-deviation path: a CONSTANT column at 1.7e308 has true scale
        // zero, but the mean/variance accumulators overflow; the honest result
        // is the typed overflow error, never an active infinite scale.
        let constant_huge = Matrix::from_slice(&[&[1.7e308], &[1.7e308], &[1.7e308]]);
        let config = RobustScalerConfig {
            scale_method: RobustScaleMethod::StandardDeviation,
            ..RobustScalerConfig::default()
        };
        assert!(matches!(
            RobustScaler::fit(&constant_huge, config),
            Err(RobustGeometryError::NonFiniteScale { dimension: 0, .. })
        ));
    }

    #[test]
    fn no_silent_nan_from_transforms() {
        let data = sample_matrix();
        let scaler = RobustScaler::fit(&data, mad_config()).unwrap();
        let transformed = scaler.transform(&data).unwrap();
        for row in &transformed.data
        {
            for &v in row
            {
                assert!(v.is_finite());
            }
        }
    }
}

#[cfg(test)]
mod ogk_scatter_tests {
    use super::*;

    fn mat(data: Vec<Vec<f64>>) -> Matrix {
        let rows = data.len();
        let cols = data[0].len();
        Matrix { rows, cols, data }
    }

    fn ogk(reweight: bool, ridge: f64) -> RobustScatterConfig {
        RobustScatterConfig {
            method: RobustScatterMethod::Ogk {
                scale: RobustUnivariateScale::MedianAbsoluteDeviation,
                reweight,
            },
            ridge,
            ..RobustScatterConfig::default()
        }
    }

    /// A deterministic, healthy-rank correlated 2-D cloud (slope 0.8 + bounded
    /// wiggle), no RNG.
    fn correlated_cloud(n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| {
                let x = (i as f64) - (n as f64) / 2.0;
                let wiggle = ((i * 7) % 13) as f64 - 6.0;
                vec![x, 0.8 * x + wiggle]
            })
            .collect()
    }

    /// Two near-independent coordinates with equal spread — isotropic, so `U` has
    /// a repeated eigenvalue.
    fn isotropic_cloud(n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| {
                let a = ((i * 5) % 17) as f64 - 8.0;
                let b = ((i * 11) % 17) as f64 - 8.0;
                vec![a, b]
            })
            .collect()
    }

    fn eigenvalues_2x2(m: &Matrix) -> (f64, f64) {
        let a = m.data[0][0];
        let b = m.data[0][1];
        let d = m.data[1][1];
        let half = (a + d) / 2.0;
        let root = (((a - d) / 2.0).powi(2) + b * b).sqrt();
        (half + root, half - root)
    }

    fn rel_close(actual: f64, expected: f64, relative: f64) -> bool {
        (actual - expected).abs() <= relative * expected.abs().max(1.0)
    }

    #[test]
    fn ogk_fits_a_correlated_cloud_positive_definite() {
        let data = mat(correlated_cloud(40));
        let model = RobustScatterModel::fit(&data, ogk(true, 0.0)).unwrap();
        assert_eq!(model.location.len(), 2);
        assert!(model.scatter.data[0][0] > 0.0 && model.scatter.data[1][1] > 0.0);
        let det = model.scatter.data[0][0] * model.scatter.data[1][1]
            - model.scatter.data[0][1] * model.scatter.data[1][0];
        assert!(det > 0.0, "scatter must be positive definite");
        assert!(
            model.scatter.data[0][1] > 0.0,
            "positive correlation preserved"
        );
        assert!(model.active_dimensions.iter().all(|&a| a));
    }

    #[test]
    fn ogk_is_translation_equivariant() {
        let base = correlated_cloud(40);
        let shift = [10.0, -7.0];
        let shifted: Vec<Vec<f64>> = base
            .iter()
            .map(|r| vec![r[0] + shift[0], r[1] + shift[1]])
            .collect();
        let a = RobustScatterModel::fit(&mat(base), ogk(false, 0.0)).unwrap();
        let b = RobustScatterModel::fit(&mat(shifted), ogk(false, 0.0)).unwrap();
        for (j, &offset) in shift.iter().enumerate()
        {
            assert!((b.location[j] - a.location[j] - offset).abs() < 1e-9);
        }
        for i in 0..2
        {
            for j in 0..2
            {
                assert!((b.scatter.data[i][j] - a.scatter.data[i][j]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn ogk_scales_scatter_quadratically_and_location_linearly() {
        let base = correlated_cloud(40);
        let c = 3.0;
        let scaled: Vec<Vec<f64>> = base.iter().map(|r| vec![c * r[0], c * r[1]]).collect();
        let a = RobustScatterModel::fit(&mat(base), ogk(false, 0.0)).unwrap();
        let b = RobustScatterModel::fit(&mat(scaled), ogk(false, 0.0)).unwrap();
        for j in 0..2
        {
            assert!(rel_close(b.location[j], c * a.location[j], 1e-9));
        }
        for i in 0..2
        {
            for j in 0..2
            {
                assert!(rel_close(
                    b.scatter.data[i][j],
                    c * c * a.scatter.data[i][j],
                    1e-9
                ));
            }
        }
    }

    #[test]
    fn ogk_is_coordinate_scaling_equivariant() {
        let base = correlated_cloud(40);
        let (sx, sy) = (2.0, 5.0);
        let scaled: Vec<Vec<f64>> = base.iter().map(|r| vec![sx * r[0], sy * r[1]]).collect();
        let a = RobustScatterModel::fit(&mat(base), ogk(false, 0.0)).unwrap();
        let b = RobustScatterModel::fit(&mat(scaled), ogk(false, 0.0)).unwrap();
        assert!(rel_close(b.location[0], sx * a.location[0], 1e-9));
        assert!(rel_close(b.location[1], sy * a.location[1], 1e-9));
        assert!(rel_close(
            b.scatter.data[0][0],
            sx * sx * a.scatter.data[0][0],
            1e-9
        ));
        assert!(rel_close(
            b.scatter.data[1][1],
            sy * sy * a.scatter.data[1][1],
            1e-9
        ));
        assert!(rel_close(
            b.scatter.data[0][1],
            sx * sy * a.scatter.data[0][1],
            1e-9
        ));
    }

    #[test]
    fn ogk_rotation_equivariance_is_approximate_raw_and_tighter_reweighted() {
        // OGK is NOT exactly affine/rotation equivariant (only translation and
        // per-coordinate scaling are exact). We MEASURE it and state both facts:
        // raw OGK preserves the scatter spectrum only loosely under rotation,
        // while the reweighting step — a classical covariance on the retained
        // inliers, which IS affine-equivariant — tightens it markedly.
        let base = correlated_cloud(48);
        let theta = 0.5_f64;
        let (c, s) = (theta.cos(), theta.sin());
        let rotated: Vec<Vec<f64>> = base
            .iter()
            .map(|r| vec![c * r[0] - s * r[1], s * r[0] + c * r[1]])
            .collect();

        let raw_a = RobustScatterModel::fit(&mat(base.clone()), ogk(false, 0.0)).unwrap();
        let raw_b = RobustScatterModel::fit(&mat(rotated.clone()), ogk(false, 0.0)).unwrap();
        let re_a = RobustScatterModel::fit(&mat(base), ogk(true, 0.0)).unwrap();
        let re_b = RobustScatterModel::fit(&mat(rotated), ogk(true, 0.0)).unwrap();

        let spectrum_gap = |x: &Matrix, y: &Matrix| {
            let (x1, x2) = eigenvalues_2x2(x);
            let (y1, y2) = eigenvalues_2x2(y);
            let g1 = (x1 - y1).abs() / x1.abs().max(1.0);
            let g2 = (x2 - y2).abs() / x2.abs().max(1.0);
            g1.max(g2)
        };
        let raw_gap = spectrum_gap(&raw_a.scatter, &raw_b.scatter);
        let reweighted_gap = spectrum_gap(&re_a.scatter, &re_b.scatter);

        // Both are bounded (approximate equivariance, not wild), and reweighting
        // is at least as good — here strictly tighter.
        assert!(
            raw_gap < 0.30,
            "raw rotation gap {raw_gap} out of expected band"
        );
        assert!(
            reweighted_gap < 0.12,
            "reweighted rotation gap {reweighted_gap}"
        );
        assert!(
            reweighted_gap <= raw_gap + 1e-12,
            "reweighting should not worsen rotation equivariance ({reweighted_gap} vs {raw_gap})"
        );
    }

    #[test]
    fn ogk_flags_zero_scale_dimensions_without_failing() {
        // Append a constant third column (zero robust scale).
        let data: Vec<Vec<f64>> = correlated_cloud(40)
            .into_iter()
            .map(|mut r| {
                r.push(4.0);
                r
            })
            .collect();
        let model = RobustScatterModel::fit(&mat(data), ogk(false, 1e-6)).unwrap();
        assert!(model.active_dimensions[0] && model.active_dimensions[1]);
        assert!(!model.active_dimensions[2], "constant column is inactive");
        assert!(!model.report.warnings.is_empty());
    }

    #[test]
    fn ogk_singular_scatter_is_a_typed_error_and_ridge_recovers() {
        // Perfectly collinear data: y = 2x, no off-line spread → singular.
        let collinear: Vec<Vec<f64>> = (0..30)
            .map(|i| {
                let x = i as f64 - 15.0;
                vec![x, 2.0 * x]
            })
            .collect();
        let err = RobustScatterModel::fit(&mat(collinear.clone()), ogk(false, 0.0)).unwrap_err();
        assert!(matches!(err, RobustGeometryError::SingularScatter { .. }));
        // A positive ridge regularizes it to positive definite.
        let ok = RobustScatterModel::fit(&mat(collinear), ogk(false, 0.5));
        assert!(ok.is_ok());
    }

    #[test]
    fn ogk_is_deterministic_and_permutation_stable() {
        let data = mat(correlated_cloud(40));
        let first = RobustScatterModel::fit(&data, ogk(true, 0.0)).unwrap();
        let second = RobustScatterModel::fit(&data, ogk(true, 0.0)).unwrap();
        assert_eq!(first, second, "same input, bit-identical model");

        // Reversing the rows must not change the estimate (up to float order).
        let reversed: Vec<Vec<f64>> = correlated_cloud(40).into_iter().rev().collect();
        let permuted = RobustScatterModel::fit(&mat(reversed), ogk(true, 0.0)).unwrap();
        for j in 0..2
        {
            assert!((permuted.location[j] - first.location[j]).abs() < 1e-8);
            for k in 0..2
            {
                assert!((permuted.scatter.data[j][k] - first.scatter.data[j][k]).abs() < 1e-8);
            }
        }
    }

    fn contaminated() -> (Matrix, Vec<usize>) {
        let mut rows = correlated_cloud(40);
        let outliers: Vec<usize> = (40..48).collect();
        for k in 0..8
        {
            let t = k as f64;
            rows.push(vec![10.0 + t, -25.0 - 2.0 * t]);
        }
        (mat(rows), outliers)
    }

    #[test]
    fn ogk_resists_gross_multivariate_contamination_where_classical_fails() {
        let (data, _) = contaminated();
        let robust = RobustScatterModel::fit(&data, ogk(true, 0.0)).unwrap();
        let classical = RobustScatterModel::fit(
            &data,
            RobustScatterConfig {
                method: RobustScatterMethod::Classical,
                ridge: 0.0,
                ..RobustScatterConfig::default()
            },
        )
        .unwrap();
        // The clean cloud is positively correlated; robust keeps that sign and
        // rejects a positive number of outliers.
        assert!(
            robust.scatter.data[0][1] > 0.0,
            "robust keeps positive correlation"
        );
        assert!(robust.report.reweighted_outlier_count > 0);
        // Classical correlation is dragged down (toward or below zero) by the
        // anti-correlated contamination — strictly weaker than robust's.
        assert!(
            classical.scatter.data[0][1] < robust.scatter.data[0][1],
            "classical off-diagonal {} should be below robust {}",
            classical.scatter.data[0][1],
            robust.scatter.data[0][1]
        );
    }

    #[test]
    fn ogk_mahalanobis_ranks_the_true_outliers_highest() {
        let (data, outliers) = contaminated();
        let model = RobustScatterModel::fit(&data, ogk(true, 0.0)).unwrap();
        let mut ranked: Vec<(usize, f64)> = (0..data.rows)
            .map(|i| {
                let row: Vec<f64> = data.data[i].clone();
                (i, model.mahalanobis_squared(&row).unwrap())
            })
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top: std::collections::BTreeSet<usize> = ranked
            .iter()
            .take(outliers.len())
            .map(|(i, _)| *i)
            .collect();
        let expected: std::collections::BTreeSet<usize> = outliers.into_iter().collect();
        assert_eq!(
            top, expected,
            "the top robust distances must be the injected outliers"
        );
    }

    #[test]
    fn ogk_handles_repeated_eigenvalues_isotropic_data() {
        let data = mat(isotropic_cloud(51));
        let first = RobustScatterModel::fit(&data, ogk(false, 0.0)).unwrap();
        let second = RobustScatterModel::fit(&data, ogk(false, 0.0)).unwrap();
        assert_eq!(first, second);
        assert!(first.scatter.data[0][0] > 0.0 && first.scatter.data[1][1] > 0.0);
        // Near-isotropic: off-diagonal small relative to the diagonal.
        let scale = first.scatter.data[0][0].max(first.scatter.data[1][1]);
        assert!(first.scatter.data[0][1].abs() < 0.4 * scale);
    }

    #[test]
    fn classical_method_matches_direct_mean_and_covariance() {
        let data = mat(correlated_cloud(40));
        let model = RobustScatterModel::fit(
            &data,
            RobustScatterConfig {
                method: RobustScatterMethod::Classical,
                ridge: 0.0,
                ..RobustScatterConfig::default()
            },
        )
        .unwrap();
        let (centered, means) = data.center();
        let cov = centered.cov_matrix();
        for (j, &mean) in means.iter().enumerate()
        {
            assert!((model.location[j] - mean).abs() < 1e-9);
            for (k, &covariance) in cov.data[j].iter().enumerate()
            {
                assert!((model.scatter.data[j][k] - covariance).abs() < 1e-9);
            }
        }
        assert_eq!(
            model.report.achieved_equivariance,
            AchievedEquivariance::AffineExactArithmetic
        );
    }

    #[test]
    fn ogk_rejects_invalid_configuration_and_input() {
        let data = mat(correlated_cloud(40));
        assert!(matches!(
            RobustScatterModel::fit(&data, ogk(false, -1.0)).unwrap_err(),
            RobustGeometryError::InvalidRidge { .. }
        ));
        assert!(matches!(
            RobustScatterModel::fit(
                &data,
                RobustScatterConfig {
                    minimum_scale: -0.1,
                    ..ogk(false, 0.0)
                }
            )
            .unwrap_err(),
            RobustGeometryError::InvalidMinimumScale { .. }
        ));
        let single = mat(vec![vec![1.0, 2.0]]);
        assert!(matches!(
            RobustScatterModel::fit(&single, ogk(false, 0.0)).unwrap_err(),
            RobustGeometryError::InsufficientSamples { .. }
        ));
    }
}
