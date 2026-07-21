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

use scirust_stats::robust::{MadConsistency, median_absolute_deviation};
use scirust_stats::{RobustStatsError, describe};

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
