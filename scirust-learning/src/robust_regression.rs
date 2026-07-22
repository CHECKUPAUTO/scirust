//! Deterministic robust linear/affine regression (SRCC program, phase 724).
//!
//! General multi-feature regression infrastructure with typed errors and
//! convergence metadata:
//!
//! - **Ordinary least squares** (multi-output) solved by Householder QR from
//!   `scirust-solvers` — never by normal equations;
//! - **Iteratively reweighted least squares** with the [`RobustLoss`] family
//!   (Huber, Tukey bisquare, smoothed absolute loss) and a deterministic
//!   MAD-based residual scale;
//! - **Trimmed least squares** with deterministic residual ranking, stable
//!   tie-breaking, fixed-point iteration and explicit cycle detection;
//! - **Median-of-means regression** with a seeded deterministic partition and
//!   coordinate-wise parameter-median aggregation (a documented heuristic).
//!
//! The historical [`crate::linear_regression`] and [`crate::polynomial_fit`]
//! baselines are untouched; everything here is new and opt-in.
//!
//! # Reuse, not reinvention
//!
//! Dense linear algebra (matrix storage, Householder QR, least-squares solve)
//! comes from `scirust-solvers`; robust residual scales come from
//! `scirust-stats` (normal-consistent MAD); optional feature standardization
//! reuses `scirust-multivariate`'s fitted [`RobustScaler`] — no second matrix
//! package, no second robust scaler.
//!
//! # Determinism contract
//!
//! No hidden RNG: the only pseudo-randomness is the explicit seed of
//! [`RobustRegressionMethod::MedianOfMeans`], driving a `SplitMix64`
//! Fisher–Yates permutation. Residual ranking uses `f64::total_cmp` with
//! original-index tie-breaks. Identical inputs and configuration produce
//! bit-identical reports. Row order is *not* a free invariance: QR rotations
//! depend on row order at the last-bit level (results agree to solver
//! tolerance, verified by test, not bit-for-bit), and the median-of-means
//! partition composes the seeded permutation with the caller's row order.
//!
//! # Breakdown honesty
//!
//! No method here resists arbitrary majority corruption. Huber-type IRLS has a
//! redescending/bounded influence but a breakdown point that shrinks with
//! leverage; an `h`-trimmed fit tolerates at most a `1 − h` contamination
//! fraction; median-of-means requires a majority of uncontaminated blocks.
//! The benchmark exercises the failure side of each of these claims.

use core::fmt;

use scirust_multivariate::{
    Matrix as MvMatrix, RobustGeometryError, RobustScaleMethod, RobustScaler, RobustScalerConfig,
    ZeroScalePolicy,
};
use scirust_solvers::linalg::{Matrix, qr_decompose, solve_qr_least_squares};
use scirust_stats::robust::{MadConsistency, median_absolute_deviation};
use scirust_stats::{SplitMix64, describe};

/// A regression dataset: `n` rows of `p` features and `k` targets, with
/// optional non-negative per-sample weights.
#[derive(Clone, Debug, PartialEq)]
pub struct RegressionDataset {
    /// Feature matrix (`n × p`).
    pub features: Matrix,
    /// Target matrix (`n × k`); use one column for single-output regression.
    pub targets: Matrix,
    /// Optional per-sample weights (finite, non-negative, positive total).
    pub sample_weights: Option<Vec<f64>>,
}

/// A fitted linear/affine model.
///
/// When feature standardization was requested, `feature_location` and
/// `feature_scale` hold the fitted per-feature location/scale and
/// `coefficients` live in the standardized space; [`LinearRegressionModel::predict`]
/// applies the same standardization to its input, so predictions are always in
/// the original target units.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearRegressionModel {
    /// Coefficient matrix (`p × k`).
    pub coefficients: Matrix,
    /// Per-output intercept (`k` entries; all zero when `fit_intercept` was
    /// false).
    pub intercept: Vec<f64>,
    /// Fitted per-feature location when standardization was requested.
    pub feature_location: Option<Vec<f64>>,
    /// Fitted per-feature scale when standardization was requested.
    pub feature_scale: Option<Vec<f64>>,
}

impl LinearRegressionModel {
    /// Predicts targets (`m × k`) for a feature matrix (`m × p`).
    pub fn predict(&self, features: &Matrix) -> Result<Matrix, RobustRegressionError> {
        let p = self.coefficients.rows();
        let k = self.coefficients.cols();

        if features.cols() != p
        {
            return Err(RobustRegressionError::FeatureCountMismatch {
                expected: p,
                found: features.cols(),
            });
        }

        validate_finite_matrix(features, false)?;

        let mut predictions = Matrix::zeros(features.rows(), k);

        for row in 0..features.rows()
        {
            for output in 0..k
            {
                let mut value = self.intercept[output];

                for feature in 0..p
                {
                    let mut x = features[(row, feature)];

                    if let (Some(location), Some(scale)) =
                        (&self.feature_location, &self.feature_scale)
                    {
                        x = (x - location[feature]) / scale[feature];
                    }

                    value += self.coefficients[(feature, output)] * x;
                }

                predictions[(row, output)] = value;
            }
        }

        Ok(predictions)
    }
}

/// Robust loss families for [`RobustRegressionMethod::IterativelyReweightedLeastSquares`].
///
/// The IRLS weight of a residual `r` at robust scale `σ` is:
///
/// - `Squared`: `w = 1` (plain weighted least squares);
/// - `AbsoluteApprox { epsilon }`: `w = 1 / √(r² + ε²)` — a **smoothed
///   approximation** of the absolute loss, honestly named: it is *not* exact
///   least absolute deviations;
/// - `Huber { delta }`: `w = 1` if `|r| ≤ δσ`, else `δσ / |r|`;
/// - `TukeyBisquare { cutoff }`: `w = (1 − (r/(cσ))²)²` if `|r| < cσ`, else
///   `0` — **non-convex and redescending**: the result can depend on the
///   (deterministic OLS) initialization and reach a local minimum only.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RobustLoss {
    /// Plain squared loss (weights always one).
    Squared,
    /// Smoothed absolute loss `√(r² + ε²)`; requires `epsilon > 0`.
    AbsoluteApprox {
        /// Smoothing width; behaves like exact absolute loss as `ε → 0` but is
        /// never exact.
        epsilon: f64,
    },
    /// Huber loss with threshold `delta` in robust-scale units; requires
    /// `delta > 0`.
    Huber {
        /// Transition point between quadratic and linear regimes, in units of
        /// the MAD residual scale.
        delta: f64,
    },
    /// Tukey bisquare with cutoff `cutoff` in robust-scale units; requires
    /// `cutoff > 0`.
    TukeyBisquare {
        /// Rejection point (weight is exactly zero beyond it), in units of the
        /// MAD residual scale.
        cutoff: f64,
    },
}

/// Which estimator fits the model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RobustRegressionMethod {
    /// Ordinary (optionally weighted, optionally ridge-regularized) least
    /// squares via Householder QR. Supports multiple outputs.
    OrdinaryLeastSquares,
    /// Iteratively reweighted least squares with the configured
    /// [`RobustLoss`]. Single output only.
    IterativelyReweightedLeastSquares,
    /// Iterated trimmed least squares: keep the `⌊h·n⌋` smallest squared
    /// residuals, refit, repeat to a fixed point (with explicit cycle
    /// detection). Single output only; requires `0.5 < h ≤ 1`.
    TrimmedLeastSquares {
        /// Fraction `h` of samples retained at every iteration.
        retained_fraction: f64,
    },
    /// Median-of-means: fit ordinary least squares per seeded block and
    /// aggregate parameter vectors coordinate-wise by median. Single output
    /// only.
    ///
    /// The coordinate-wise parameter median is a **documented heuristic**: it
    /// is not affine equivariant and carries no optimality guarantee; its
    /// robustness holds only when a majority of blocks are uncontaminated.
    MedianOfMeans {
        /// Number of blocks (each must keep at least as many rows as fitted
        /// columns).
        block_count: usize,
        /// Seed of the deterministic `SplitMix64` Fisher–Yates permutation
        /// applied to row indices before contiguous blocking.
        seed: u64,
    },
}

/// Configuration for [`fit_robust_regression`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RobustRegressionConfig {
    /// The estimator.
    pub method: RobustRegressionMethod,
    /// The IRLS loss (ignored by the other methods).
    pub loss: RobustLoss,
    /// Fit a per-output intercept.
    pub fit_intercept: bool,
    /// Ridge penalty on the feature coefficients (never on the intercept),
    /// implemented by row augmentation of the QR design. `0.0` disables it.
    pub ridge: f64,
    /// Maximum IRLS / trimming iterations.
    pub maximum_iterations: usize,
    /// Absolute coefficient-change tolerance for convergence.
    pub absolute_tolerance: f64,
    /// Relative coefficient-change tolerance for convergence.
    pub relative_tolerance: f64,
    /// Optional fitted feature standardization (reusing
    /// [`RobustScaler`]); a degenerate feature scale is a typed
    /// error, never a silent unit fallback.
    pub scale_method: Option<RobustScaleMethod>,
}

impl Default for RobustRegressionConfig {
    fn default() -> Self {
        Self {
            method: RobustRegressionMethod::OrdinaryLeastSquares,
            loss: RobustLoss::Squared,
            fit_intercept: true,
            ridge: 0.0,
            maximum_iterations: 50,
            absolute_tolerance: 1.0e-10,
            relative_tolerance: 1.0e-8,
            scale_method: None,
        }
    }
}

/// Diagnostics accompanying a fitted model.
#[derive(Clone, Debug, PartialEq)]
pub struct RobustRegressionReport {
    /// The fitted model.
    pub model: LinearRegressionModel,
    /// Iterations performed (`1` for the single-pass methods).
    pub iterations: usize,
    /// Whether the iterative method converged (always `true` for single-pass
    /// methods). Non-convergence is reported here, with a warning — never
    /// silently hidden.
    pub converged: bool,
    /// Final objective: the weighted residual sum of squares
    /// `Σᵢ wᵢ·‖rᵢ‖²` using the final per-sample weights (for trimmed fits the
    /// rejected samples carry weight zero; for median-of-means it is the
    /// unweighted full-data value under the aggregated model).
    pub objective: f64,
    /// Normal-consistent MAD of the final (single-output: scalar; multi-output:
    /// row-norm) residuals. `0.0` when more than half the residuals vanish.
    pub residual_scale: f64,
    /// Number of samples with meaningful final weight (`> 1e-12`).
    pub effective_sample_count: usize,
    /// Samples excluded by the trimmed method (empty for the others), in
    /// ascending index order.
    pub rejected_sample_indices: Vec<usize>,
    /// Final per-sample weights in caller row order.
    pub final_weights: Vec<f64>,
    /// Human-readable caveats (non-convergence, degenerate residual scale,
    /// heuristic aggregation, …). Never used to hide a typed failure.
    pub warnings: Vec<String>,
}

/// Errors returned by [`fit_robust_regression`].
///
/// `f64` payloads make this `PartialEq` only; match on variants when a payload
/// may be `NaN`.
#[derive(Clone, Debug, PartialEq)]
pub enum RobustRegressionError {
    /// The dataset has zero rows, zero feature columns, or zero target
    /// columns.
    EmptyDataset,
    /// Feature and target row counts differ.
    RowCountMismatch {
        /// Feature rows.
        features: usize,
        /// Target rows.
        targets: usize,
    },
    /// A prediction input has the wrong number of feature columns.
    FeatureCountMismatch {
        /// Fitted feature count.
        expected: usize,
        /// Supplied feature count.
        found: usize,
    },
    /// A feature entry is `NaN` or `±∞`.
    NonFiniteFeature {
        /// Row of the offending entry.
        row: usize,
        /// Column of the offending entry.
        col: usize,
    },
    /// A target entry is `NaN` or `±∞`.
    NonFiniteTarget {
        /// Row of the offending entry.
        row: usize,
        /// Column of the offending entry.
        col: usize,
    },
    /// A sample weight is `NaN` or `±∞`.
    NonFiniteWeight {
        /// Index of the offending weight.
        index: usize,
    },
    /// A sample weight is negative.
    NegativeWeight {
        /// Index of the offending weight.
        index: usize,
    },
    /// The sample weights sum to a non-positive total.
    ZeroTotalWeight,
    /// The weight vector's length differs from the row count.
    WeightCountMismatch {
        /// Row count.
        rows: usize,
        /// Weight count.
        weights: usize,
    },
    /// Fewer effective rows than fitted columns; the least-squares problem is
    /// underdetermined.
    InsufficientSamples {
        /// Minimum rows required (fitted columns).
        required: usize,
        /// Rows available.
        found: usize,
    },
    /// The selected method supports a single output only.
    UnsupportedMultiOutput {
        /// Number of target columns supplied.
        outputs: usize,
    },
    /// `ridge` is negative or non-finite.
    InvalidRidge,
    /// A tolerance is non-positive or non-finite, or `maximum_iterations` is
    /// zero.
    InvalidTolerance,
    /// A loss parameter (`epsilon`, `delta`, `cutoff`) is non-positive or
    /// non-finite.
    InvalidLossParameter,
    /// The trimmed retained fraction is outside `(0.5, 1]`.
    InvalidRetainedFraction,
    /// The median-of-means block count is zero or exceeds the row count.
    InvalidBlockCount,
    /// A median-of-means block has fewer rows than fitted columns.
    InsufficientBlockSamples {
        /// Index of the offending block.
        block: usize,
        /// Rows in that block.
        rows: usize,
        /// Minimum rows required.
        required: usize,
    },
    /// A feature's fitted robust scale is degenerate (or overflowed) under the
    /// requested standardization.
    DegenerateFeatureScale {
        /// The offending feature column.
        dimension: usize,
    },
    /// The trimming or reweighting left no sample with positive weight.
    AllSamplesRejected,
    /// The QR least-squares solve failed (rank deficiency without ridge, or a
    /// numerical failure); the payload carries the solver's message.
    LeastSquaresSolveFailed {
        /// Display of the underlying solver error.
        detail: String,
    },
}

impl fmt::Display for RobustRegressionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyDataset => formatter.write_str("dataset has no rows, features, or targets"),
            Self::RowCountMismatch { features, targets } => write!(
                formatter,
                "feature rows {features} do not match target rows {targets}"
            ),
            Self::FeatureCountMismatch { expected, found } => write!(
                formatter,
                "expected {expected} feature columns, found {found}"
            ),
            Self::NonFiniteFeature { row, col } =>
            {
                write!(formatter, "feature ({row}, {col}) is not finite")
            },
            Self::NonFiniteTarget { row, col } =>
            {
                write!(formatter, "target ({row}, {col}) is not finite")
            },
            Self::NonFiniteWeight { index } =>
            {
                write!(formatter, "sample weight {index} is not finite")
            },
            Self::NegativeWeight { index } =>
            {
                write!(formatter, "sample weight {index} is negative")
            },
            Self::ZeroTotalWeight =>
            {
                formatter.write_str("sample weights sum to a non-positive total")
            },
            Self::WeightCountMismatch { rows, weights } => write!(
                formatter,
                "weight count {weights} does not match row count {rows}"
            ),
            Self::InsufficientSamples { required, found } => write!(
                formatter,
                "at least {required} rows are required, found {found}"
            ),
            Self::UnsupportedMultiOutput { outputs } => write!(
                formatter,
                "this method supports a single output, found {outputs} target columns"
            ),
            Self::InvalidRidge => formatter.write_str("ridge must be finite and non-negative"),
            Self::InvalidTolerance => formatter.write_str(
                "tolerances must be finite and positive and maximum iterations non-zero",
            ),
            Self::InvalidLossParameter =>
            {
                formatter.write_str("loss parameters must be finite and positive")
            },
            Self::InvalidRetainedFraction =>
            {
                formatter.write_str("retained fraction must lie in (0.5, 1]")
            },
            Self::InvalidBlockCount =>
            {
                formatter.write_str("block count must be at least 1 and at most the row count")
            },
            Self::InsufficientBlockSamples {
                block,
                rows,
                required,
            } => write!(
                formatter,
                "median-of-means block {block} has {rows} rows but needs {required}"
            ),
            Self::DegenerateFeatureScale { dimension } => write!(
                formatter,
                "feature {dimension} has a degenerate robust scale under the requested standardization"
            ),
            Self::AllSamplesRejected => formatter.write_str("no sample retained a positive weight"),
            Self::LeastSquaresSolveFailed { detail } =>
            {
                write!(formatter, "least-squares solve failed: {detail}")
            },
        }
    }
}

impl std::error::Error for RobustRegressionError {}

/// Weight below which a sample is not counted in `effective_sample_count`.
const EFFECTIVE_WEIGHT_FLOOR: f64 = 1.0e-12;

/// Fits a robust regression model.
///
/// See the module documentation for the estimator catalogue, determinism
/// contract, and breakdown limits.
pub fn fit_robust_regression(
    dataset: &RegressionDataset,
    config: RobustRegressionConfig,
) -> Result<RobustRegressionReport, RobustRegressionError> {
    validate_dataset(dataset)?;
    validate_config(&config)?;

    let n = dataset.features.rows();
    let k = dataset.targets.cols();

    // Optional fitted standardization (a degenerate scale is a typed error).
    let standardization = fit_standardization(&dataset.features, config.scale_method)?;

    let standardized = match &standardization
    {
        Some((location, scale)) => standardize(&dataset.features, location, scale),
        None => dataset.features.clone(),
    };

    let base_weights: Vec<f64> = match &dataset.sample_weights
    {
        Some(weights) => weights.clone(),
        None => vec![1.0; n],
    };

    match config.method
    {
        RobustRegressionMethod::OrdinaryLeastSquares => fit_ordinary(
            dataset,
            &standardized,
            &standardization,
            &base_weights,
            config,
        ),
        RobustRegressionMethod::IterativelyReweightedLeastSquares =>
        {
            require_single_output(k)?;
            fit_irls(
                dataset,
                &standardized,
                &standardization,
                &base_weights,
                config,
            )
        },
        RobustRegressionMethod::TrimmedLeastSquares { retained_fraction } =>
        {
            require_single_output(k)?;
            fit_trimmed(
                dataset,
                &standardized,
                &standardization,
                &base_weights,
                config,
                retained_fraction,
            )
        },
        RobustRegressionMethod::MedianOfMeans { block_count, seed } =>
        {
            require_single_output(k)?;
            fit_median_of_means(
                dataset,
                &standardized,
                &standardization,
                &base_weights,
                config,
                block_count,
                seed,
            )
        },
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_dataset(dataset: &RegressionDataset) -> Result<(), RobustRegressionError> {
    let n = dataset.features.rows();
    let p = dataset.features.cols();
    let k = dataset.targets.cols();

    if n == 0 || p == 0 || k == 0
    {
        return Err(RobustRegressionError::EmptyDataset);
    }

    if dataset.targets.rows() != n
    {
        return Err(RobustRegressionError::RowCountMismatch {
            features: n,
            targets: dataset.targets.rows(),
        });
    }

    validate_finite_matrix(&dataset.features, false)?;
    validate_finite_matrix(&dataset.targets, true)?;

    if let Some(weights) = &dataset.sample_weights
    {
        if weights.len() != n
        {
            return Err(RobustRegressionError::WeightCountMismatch {
                rows: n,
                weights: weights.len(),
            });
        }

        let mut total = 0.0;

        for (index, &weight) in weights.iter().enumerate()
        {
            if !weight.is_finite()
            {
                return Err(RobustRegressionError::NonFiniteWeight { index });
            }

            if weight < 0.0
            {
                return Err(RobustRegressionError::NegativeWeight { index });
            }

            total += weight;
        }

        if total <= 0.0
        {
            return Err(RobustRegressionError::ZeroTotalWeight);
        }
    }

    Ok(())
}

fn validate_finite_matrix(matrix: &Matrix, is_target: bool) -> Result<(), RobustRegressionError> {
    for row in 0..matrix.rows()
    {
        for col in 0..matrix.cols()
        {
            if !matrix[(row, col)].is_finite()
            {
                return Err(
                    if is_target
                    {
                        RobustRegressionError::NonFiniteTarget { row, col }
                    }
                    else
                    {
                        RobustRegressionError::NonFiniteFeature { row, col }
                    },
                );
            }
        }
    }

    Ok(())
}

fn validate_config(config: &RobustRegressionConfig) -> Result<(), RobustRegressionError> {
    if !config.ridge.is_finite() || config.ridge < 0.0
    {
        return Err(RobustRegressionError::InvalidRidge);
    }

    if !config.absolute_tolerance.is_finite()
        || config.absolute_tolerance <= 0.0
        || !config.relative_tolerance.is_finite()
        || config.relative_tolerance <= 0.0
        || config.maximum_iterations == 0
    {
        return Err(RobustRegressionError::InvalidTolerance);
    }

    match config.loss
    {
        RobustLoss::Squared =>
        {},
        RobustLoss::AbsoluteApprox { epsilon } =>
        {
            if !epsilon.is_finite() || epsilon <= 0.0
            {
                return Err(RobustRegressionError::InvalidLossParameter);
            }
        },
        RobustLoss::Huber { delta } =>
        {
            if !delta.is_finite() || delta <= 0.0
            {
                return Err(RobustRegressionError::InvalidLossParameter);
            }
        },
        RobustLoss::TukeyBisquare { cutoff } =>
        {
            if !cutoff.is_finite() || cutoff <= 0.0
            {
                return Err(RobustRegressionError::InvalidLossParameter);
            }
        },
    }

    match config.method
    {
        RobustRegressionMethod::TrimmedLeastSquares { retained_fraction } =>
        {
            if !retained_fraction.is_finite() || retained_fraction <= 0.5 || retained_fraction > 1.0
            {
                return Err(RobustRegressionError::InvalidRetainedFraction);
            }
        },
        RobustRegressionMethod::MedianOfMeans { block_count: 0, .. } =>
        {
            return Err(RobustRegressionError::InvalidBlockCount);
        },
        _ =>
        {},
    }

    Ok(())
}

fn require_single_output(outputs: usize) -> Result<(), RobustRegressionError> {
    if outputs != 1
    {
        return Err(RobustRegressionError::UnsupportedMultiOutput { outputs });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Standardization
// ---------------------------------------------------------------------------

type Standardization = Option<(Vec<f64>, Vec<f64>)>;

fn fit_standardization(
    features: &Matrix,
    scale_method: Option<RobustScaleMethod>,
) -> Result<Standardization, RobustRegressionError> {
    let Some(method) = scale_method
    else
    {
        return Ok(None);
    };

    let data = MvMatrix {
        rows: features.rows(),
        cols: features.cols(),
        data: (0..features.rows())
            .map(|row| features.row(row).to_vec())
            .collect(),
    };

    let scaler = RobustScaler::fit(
        &data,
        RobustScalerConfig {
            center: true,
            scale_method: method,
            // A constant (or degenerate) feature cannot be standardized; with
            // an intercept it is exactly collinear anyway. This is a typed
            // error, never a silent unit-scale fallback.
            zero_scale_policy: ZeroScalePolicy::Error,
            minimum_scale: 0.0,
        },
    )
    .map_err(map_scaler_error)?;

    Ok(Some((scaler.location, scaler.scale)))
}

fn map_scaler_error(error: RobustGeometryError) -> RobustRegressionError {
    match error
    {
        RobustGeometryError::DegenerateDimension { dimension, .. }
        | RobustGeometryError::NonFiniteScale { dimension, .. } =>
        {
            RobustRegressionError::DegenerateFeatureScale { dimension }
        },
        RobustGeometryError::InsufficientSamples { required, found } =>
        {
            RobustRegressionError::InsufficientSamples { required, found }
        },
        other => RobustRegressionError::LeastSquaresSolveFailed {
            detail: other.to_string(),
        },
    }
}

fn standardize(features: &Matrix, location: &[f64], scale: &[f64]) -> Matrix {
    let mut standardized = features.clone();

    for row in 0..standardized.rows()
    {
        for col in 0..standardized.cols()
        {
            standardized[(row, col)] = (standardized[(row, col)] - location[col]) / scale[col];
        }
    }

    standardized
}

// ---------------------------------------------------------------------------
// Weighted QR least squares core
// ---------------------------------------------------------------------------

/// Solves the (optionally weighted, optionally ridge-augmented) least-squares
/// problem on a subset of rows, returning per-output stacked coefficients
/// (`p_fit × k`, intercept last when fitted).
///
/// The design keeps the caller's ascending row order; ridge rows (one per
/// feature coefficient, never for the intercept) are appended after the data
/// rows with unit weight.
fn weighted_least_squares(
    standardized: &Matrix,
    targets: &Matrix,
    rows: &[usize],
    weights: &[f64],
    fit_intercept: bool,
    ridge: f64,
) -> Result<Matrix, RobustRegressionError> {
    let p = standardized.cols();
    let k = targets.cols();
    let fitted_columns = p + usize::from(fit_intercept);
    let ridge_rows = if ridge > 0.0 { p } else { 0 };
    let design_rows = rows.len() + ridge_rows;

    if design_rows < fitted_columns
    {
        return Err(RobustRegressionError::InsufficientSamples {
            required: fitted_columns,
            found: rows.len(),
        });
    }

    let sqrt_ridge = ridge.sqrt();

    let mut design = Matrix::zeros(design_rows, fitted_columns);
    let mut rhs = vec![vec![0.0; design_rows]; k];

    for (design_row, &row) in rows.iter().enumerate()
    {
        let sqrt_weight = weights[row].sqrt();

        for col in 0..p
        {
            design[(design_row, col)] = sqrt_weight * standardized[(row, col)];
        }

        if fit_intercept
        {
            design[(design_row, p)] = sqrt_weight;
        }

        for (output, rhs_column) in rhs.iter_mut().enumerate()
        {
            rhs_column[design_row] = sqrt_weight * targets[(row, output)];
        }
    }

    for ridge_index in 0..ridge_rows
    {
        design[(rows.len() + ridge_index, ridge_index)] = sqrt_ridge;
    }

    let factorization =
        qr_decompose(design).map_err(|error| RobustRegressionError::LeastSquaresSolveFailed {
            detail: error.to_string(),
        })?;

    let mut coefficients = Matrix::zeros(fitted_columns, k);

    for (output, rhs_column) in rhs.iter().enumerate()
    {
        let solution = solve_qr_least_squares(&factorization, rhs_column).map_err(|error| {
            RobustRegressionError::LeastSquaresSolveFailed {
                detail: error.to_string(),
            }
        })?;

        for (row, &value) in solution.iter().enumerate()
        {
            coefficients[(row, output)] = value;
        }
    }

    Ok(coefficients)
}

/// Splits stacked coefficients into a model (feature block + optional
/// intercept row).
fn build_model(
    stacked: &Matrix,
    feature_count: usize,
    fit_intercept: bool,
    standardization: &Standardization,
) -> LinearRegressionModel {
    let k = stacked.cols();
    let mut coefficients = Matrix::zeros(feature_count, k);

    for feature in 0..feature_count
    {
        for output in 0..k
        {
            coefficients[(feature, output)] = stacked[(feature, output)];
        }
    }

    let intercept = if fit_intercept
    {
        (0..k)
            .map(|output| stacked[(feature_count, output)])
            .collect()
    }
    else
    {
        vec![0.0; k]
    };

    let (feature_location, feature_scale) = match standardization
    {
        Some((location, scale)) => (Some(location.clone()), Some(scale.clone())),
        None => (None, None),
    };

    LinearRegressionModel {
        coefficients,
        intercept,
        feature_location,
        feature_scale,
    }
}

/// Per-row residual norms (single output: |r|; multiple outputs: the row's
/// Euclidean residual norm), plus the raw single-output residuals when `k == 1`.
fn residual_norms(
    standardized: &Matrix,
    targets: &Matrix,
    stacked: &Matrix,
    fit_intercept: bool,
) -> Vec<f64> {
    let p = standardized.cols();
    let k = targets.cols();

    (0..standardized.rows())
        .map(|row| {
            let mut squared = 0.0;

            for output in 0..k
            {
                let mut prediction = if fit_intercept
                {
                    stacked[(p, output)]
                }
                else
                {
                    0.0
                };

                for feature in 0..p
                {
                    prediction += stacked[(feature, output)] * standardized[(row, feature)];
                }

                let residual = targets[(row, output)] - prediction;
                squared += residual * residual;
            }

            squared.sqrt()
        })
        .collect()
}

fn weighted_objective(residuals: &[f64], weights: &[f64]) -> f64 {
    residuals
        .iter()
        .zip(weights)
        .fold(0.0, |sum, (residual, weight)| {
            sum + weight * residual * residual
        })
}

fn robust_residual_scale(residuals: &[f64]) -> f64 {
    // Inputs are validated finite, and predictions from a successful QR solve
    // are finite, so the only MAD failure mode (non-finite input) cannot
    // occur; an empty slice is excluded by validation.
    median_absolute_deviation(residuals, MadConsistency::Normal).unwrap_or(0.0)
}

fn coefficient_change(previous: &Matrix, current: &Matrix) -> f64 {
    previous
        .data()
        .iter()
        .zip(current.data())
        .fold(0.0, |max, (a, b)| max.max((a - b).abs()))
}

fn coefficient_magnitude(stacked: &Matrix) -> f64 {
    stacked.data().iter().fold(0.0, |max, v| max.max(v.abs()))
}

// ---------------------------------------------------------------------------
// Estimators
// ---------------------------------------------------------------------------

fn fit_ordinary(
    dataset: &RegressionDataset,
    standardized: &Matrix,
    standardization: &Standardization,
    base_weights: &[f64],
    config: RobustRegressionConfig,
) -> Result<RobustRegressionReport, RobustRegressionError> {
    let n = standardized.rows();
    let rows: Vec<usize> = (0..n).collect();

    let stacked = weighted_least_squares(
        standardized,
        &dataset.targets,
        &rows,
        base_weights,
        config.fit_intercept,
        config.ridge,
    )?;

    let residuals = residual_norms(
        standardized,
        &dataset.targets,
        &stacked,
        config.fit_intercept,
    );

    Ok(RobustRegressionReport {
        model: build_model(
            &stacked,
            standardized.cols(),
            config.fit_intercept,
            standardization,
        ),
        iterations: 1,
        converged: true,
        objective: weighted_objective(&residuals, base_weights),
        residual_scale: robust_residual_scale(&residuals),
        effective_sample_count: base_weights
            .iter()
            .filter(|&&weight| weight > EFFECTIVE_WEIGHT_FLOOR)
            .count(),
        rejected_sample_indices: Vec::new(),
        final_weights: base_weights.to_vec(),
        warnings: Vec::new(),
    })
}

fn loss_weight(loss: RobustLoss, residual: f64, scale: f64) -> f64 {
    match loss
    {
        RobustLoss::Squared => 1.0,
        RobustLoss::AbsoluteApprox { epsilon } =>
        {
            1.0 / (residual * residual + epsilon * epsilon).sqrt()
        },
        RobustLoss::Huber { delta } =>
        {
            let threshold = delta * scale;

            if residual.abs() <= threshold || residual == 0.0
            {
                1.0
            }
            else
            {
                threshold / residual.abs()
            }
        },
        RobustLoss::TukeyBisquare { cutoff } =>
        {
            let threshold = cutoff * scale;

            if residual.abs() >= threshold
            {
                0.0
            }
            else
            {
                let ratio = residual / threshold;
                let inner = 1.0 - ratio * ratio;
                inner * inner
            }
        },
    }
}

fn fit_irls(
    dataset: &RegressionDataset,
    standardized: &Matrix,
    standardization: &Standardization,
    base_weights: &[f64],
    config: RobustRegressionConfig,
) -> Result<RobustRegressionReport, RobustRegressionError> {
    let n = standardized.rows();
    let rows: Vec<usize> = (0..n).collect();
    let mut warnings = Vec::new();

    // Deterministic initialization: the ordinary least-squares solution.
    let mut stacked = weighted_least_squares(
        standardized,
        &dataset.targets,
        &rows,
        base_weights,
        config.fit_intercept,
        config.ridge,
    )?;

    let mut weights = base_weights.to_vec();
    let mut converged = false;
    let mut iterations = 0;

    while iterations < config.maximum_iterations
    {
        iterations += 1;

        let residuals = residual_norms(
            standardized,
            &dataset.targets,
            &stacked,
            config.fit_intercept,
        );

        let scale = robust_residual_scale(&residuals);

        if scale == 0.0
            && !matches!(
                config.loss,
                RobustLoss::Squared | RobustLoss::AbsoluteApprox { .. }
            )
        {
            // A zero robust scale means a majority of residuals vanish: the
            // scale-dependent losses degenerate to their σ→0 limit (weight one
            // on interpolated samples, zero elsewhere). Reported, not hidden.
            warnings.push(
                "residual MAD is zero; scale-dependent weights use the sigma->0 limit".to_string(),
            );

            for (index, residual) in residuals.iter().enumerate()
            {
                weights[index] = if *residual == 0.0
                {
                    base_weights[index]
                }
                else
                {
                    0.0
                };
            }
        }
        else
        {
            for (index, residual) in residuals.iter().enumerate()
            {
                weights[index] = base_weights[index] * loss_weight(config.loss, *residual, scale);
            }
        }

        if weights.iter().all(|&weight| weight <= 0.0)
        {
            return Err(RobustRegressionError::AllSamplesRejected);
        }

        let updated = weighted_least_squares(
            standardized,
            &dataset.targets,
            &rows,
            &weights,
            config.fit_intercept,
            config.ridge,
        )?;

        let change = coefficient_change(&stacked, &updated);
        let magnitude = coefficient_magnitude(&updated);

        stacked = updated;

        if change <= config.absolute_tolerance + config.relative_tolerance * magnitude
        {
            converged = true;
            break;
        }
    }

    if !converged
    {
        warnings.push(format!(
            "IRLS did not converge within {} iterations",
            config.maximum_iterations
        ));
    }

    if matches!(config.loss, RobustLoss::TukeyBisquare { .. })
    {
        warnings.push(
            "Tukey bisquare is non-convex; the result depends on the deterministic OLS \
initialization and may be a local minimum"
                .to_string(),
        );
    }

    let residuals = residual_norms(
        standardized,
        &dataset.targets,
        &stacked,
        config.fit_intercept,
    );

    Ok(RobustRegressionReport {
        model: build_model(
            &stacked,
            standardized.cols(),
            config.fit_intercept,
            standardization,
        ),
        iterations,
        converged,
        objective: weighted_objective(&residuals, &weights),
        residual_scale: robust_residual_scale(&residuals),
        effective_sample_count: weights
            .iter()
            .filter(|&&weight| weight > EFFECTIVE_WEIGHT_FLOOR)
            .count(),
        rejected_sample_indices: Vec::new(),
        final_weights: weights,
        warnings,
    })
}

fn fit_trimmed(
    dataset: &RegressionDataset,
    standardized: &Matrix,
    standardization: &Standardization,
    base_weights: &[f64],
    config: RobustRegressionConfig,
    retained_fraction: f64,
) -> Result<RobustRegressionReport, RobustRegressionError> {
    let n = standardized.rows();
    let retained_count = ((n as f64) * retained_fraction).floor() as usize;
    let mut warnings = Vec::new();

    if retained_count == 0
    {
        return Err(RobustRegressionError::AllSamplesRejected);
    }

    // Deterministic initialization: fit on every sample, then iterate the
    // retained set to a fixed point.
    let all_rows: Vec<usize> = (0..n).collect();

    let mut stacked = weighted_least_squares(
        standardized,
        &dataset.targets,
        &all_rows,
        base_weights,
        config.fit_intercept,
        config.ridge,
    )?;

    let mut retained: Vec<usize> = all_rows.clone();
    let mut seen_sets: Vec<Vec<usize>> = Vec::new();
    let mut previous_objective = f64::INFINITY;
    let mut converged = false;
    let mut iterations = 0;

    while iterations < config.maximum_iterations
    {
        iterations += 1;

        let residuals = residual_norms(
            standardized,
            &dataset.targets,
            &stacked,
            config.fit_intercept,
        );

        // Deterministic ranking: ascending residual, original index breaking
        // ties.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| residuals[a].total_cmp(&residuals[b]).then(a.cmp(&b)));

        let mut next_retained: Vec<usize> = order[..retained_count].to_vec();
        next_retained.sort_unstable();

        // Trimmed objective of the freshly ranked set under the current fit.
        // The concentration step never increases it, so objective stagnation
        // is the honest convergence criterion: a near-exact fit leaves many
        // floating-point-tied residuals whose ranked subset can churn among
        // equivalent sets without changing the objective.
        let objective: f64 = next_retained
            .iter()
            .map(|&row| base_weights[row] * residuals[row] * residuals[row])
            .sum();

        if next_retained == retained
            || (previous_objective - objective).abs() <= config.absolute_tolerance
        {
            converged = true;
            break;
        }

        if seen_sets.contains(&next_retained)
        {
            warnings.push(
                "trimmed retained set entered a cycle; keeping the current fixed point".to_string(),
            );
            break;
        }

        seen_sets.push(retained.clone());
        retained = next_retained;
        previous_objective = objective;

        stacked = weighted_least_squares(
            standardized,
            &dataset.targets,
            &retained,
            base_weights,
            config.fit_intercept,
            config.ridge,
        )?;
    }

    if !converged && warnings.is_empty()
    {
        warnings.push(format!(
            "trimming did not reach a fixed point within {} iterations",
            config.maximum_iterations
        ));
    }

    let residuals = residual_norms(
        standardized,
        &dataset.targets,
        &stacked,
        config.fit_intercept,
    );

    let mut final_weights = vec![0.0; n];

    for &row in &retained
    {
        final_weights[row] = base_weights[row];
    }

    let rejected_sample_indices: Vec<usize> =
        (0..n).filter(|row| !retained.contains(row)).collect();

    Ok(RobustRegressionReport {
        model: build_model(
            &stacked,
            standardized.cols(),
            config.fit_intercept,
            standardization,
        ),
        iterations,
        converged,
        objective: weighted_objective(&residuals, &final_weights),
        residual_scale: robust_residual_scale(&residuals),
        effective_sample_count: final_weights
            .iter()
            .filter(|&&weight| weight > EFFECTIVE_WEIGHT_FLOOR)
            .count(),
        rejected_sample_indices,
        final_weights,
        warnings,
    })
}

fn fit_median_of_means(
    dataset: &RegressionDataset,
    standardized: &Matrix,
    standardization: &Standardization,
    base_weights: &[f64],
    config: RobustRegressionConfig,
    block_count: usize,
    seed: u64,
) -> Result<RobustRegressionReport, RobustRegressionError> {
    let n = standardized.rows();
    let p = standardized.cols();
    let fitted_columns = p + usize::from(config.fit_intercept);

    if block_count > n
    {
        return Err(RobustRegressionError::InvalidBlockCount);
    }

    // Deterministic seeded Fisher-Yates permutation of the row indices,
    // followed by as-even-as-possible contiguous blocking (the first
    // `n mod block_count` blocks take one extra row). The result composes the
    // permutation with the caller's row order: deterministic for a fixed seed,
    // but not invariant to reordering the input rows.
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = SplitMix64::new(seed);
    let mut index = n;

    while index > 1
    {
        index -= 1;
        let swap = (rng.next_u64() % (index as u64 + 1)) as usize;
        order.swap(index, swap);
    }

    let base = n / block_count;
    let remainder = n % block_count;

    let mut block_coefficients: Vec<Vec<f64>> = Vec::with_capacity(block_count);
    let mut start = 0;

    for block in 0..block_count
    {
        let size = base + usize::from(block < remainder);
        let end = start + size;

        let mut rows: Vec<usize> = order[start..end].to_vec();
        rows.sort_unstable();

        if rows.len() < fitted_columns
        {
            return Err(RobustRegressionError::InsufficientBlockSamples {
                block,
                rows: rows.len(),
                required: fitted_columns,
            });
        }

        let stacked = weighted_least_squares(
            standardized,
            &dataset.targets,
            &rows,
            base_weights,
            config.fit_intercept,
            config.ridge,
        )?;

        block_coefficients.push(stacked.data().to_vec());

        start = end;
    }

    // Coordinate-wise median across blocks: deterministic, but a documented
    // heuristic (not affine equivariant, no optimality certificate).
    let mut aggregated = Matrix::zeros(fitted_columns, 1);

    for coefficient in 0..fitted_columns
    {
        let values: Vec<f64> = block_coefficients
            .iter()
            .map(|block| block[coefficient])
            .collect();

        aggregated[(coefficient, 0)] = describe::median(&values);
    }

    let residuals = residual_norms(
        standardized,
        &dataset.targets,
        &aggregated,
        config.fit_intercept,
    );

    Ok(RobustRegressionReport {
        model: build_model(&aggregated, p, config.fit_intercept, standardization),
        iterations: 1,
        converged: true,
        objective: weighted_objective(&residuals, base_weights),
        residual_scale: robust_residual_scale(&residuals),
        effective_sample_count: base_weights
            .iter()
            .filter(|&&weight| weight > EFFECTIVE_WEIGHT_FLOOR)
            .count(),
        rejected_sample_indices: Vec::new(),
        final_weights: base_weights.to_vec(),
        warnings: vec![
            "median-of-means aggregates block coefficients by coordinate-wise median: \
a deterministic heuristic that is not affine equivariant and requires a majority of \
uncontaminated blocks"
                .to_string(),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual} (tolerance {tolerance})"
        );
    }

    /// Deterministic fixture: `y = 5 + 2·x₀ − 3·x₁` over a fixed grid, with
    /// `outliers` gross `+100` target corruptions on the first rows.
    fn affine_fixture(outliers: usize) -> RegressionDataset {
        let mut rows = Vec::new();
        let mut targets = Vec::new();

        for i in 0..20
        {
            let x0 = (i % 5) as f64;
            let x1 = (i / 5) as f64 - 1.5;
            let mut y = 5.0 + 2.0 * x0 - 3.0 * x1;

            if i < outliers
            {
                y += 100.0;
            }

            rows.push(x0);
            rows.push(x1);
            targets.push(y);
        }

        RegressionDataset {
            features: Matrix::from_row_major(20, 2, rows),
            targets: Matrix::from_row_major(20, 1, targets),
            sample_weights: None,
        }
    }

    fn config(method: RobustRegressionMethod) -> RobustRegressionConfig {
        RobustRegressionConfig {
            method,
            ..RobustRegressionConfig::default()
        }
    }

    #[test]
    fn ols_recovers_exact_affine_model() {
        let dataset = affine_fixture(0);

        let report = fit_robust_regression(
            &dataset,
            config(RobustRegressionMethod::OrdinaryLeastSquares),
        )
        .unwrap();

        assert_close(report.model.coefficients[(0, 0)], 2.0, 1.0e-9);
        assert_close(report.model.coefficients[(1, 0)], -3.0, 1.0e-9);
        assert_close(report.model.intercept[0], 5.0, 1.0e-9);
        assert!(report.converged);
        assert_eq!(report.iterations, 1);
        assert!(report.objective < 1.0e-16);
        assert_eq!(report.effective_sample_count, 20);
    }

    #[test]
    fn ols_without_intercept_recovers_linear_model() {
        // y = 4·x over a one-feature grid through the origin.
        let features = Matrix::from_row_major(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let targets = Matrix::from_row_major(5, 1, vec![4.0, 8.0, 12.0, 16.0, 20.0]);

        let dataset = RegressionDataset {
            features,
            targets,
            sample_weights: None,
        };

        let report = fit_robust_regression(
            &dataset,
            RobustRegressionConfig {
                fit_intercept: false,
                ..config(RobustRegressionMethod::OrdinaryLeastSquares)
            },
        )
        .unwrap();

        assert_close(report.model.coefficients[(0, 0)], 4.0, 1.0e-10);
        assert_eq!(report.model.intercept, vec![0.0]);
    }

    #[test]
    fn ols_matches_historical_simple_linear_regression() {
        // Cross-check against the crate's existing 1-D baseline.
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.1, 2.9, 5.2, 6.8, 9.1, 10.9];

        let (slope, intercept) = crate::linear_regression(&x, &y);

        let dataset = RegressionDataset {
            features: Matrix::from_row_major(6, 1, x.to_vec()),
            targets: Matrix::from_row_major(6, 1, y.to_vec()),
            sample_weights: None,
        };

        let report = fit_robust_regression(
            &dataset,
            config(RobustRegressionMethod::OrdinaryLeastSquares),
        )
        .unwrap();

        assert_close(report.model.coefficients[(0, 0)], slope, 1.0e-9);
        assert_close(report.model.intercept[0], intercept, 1.0e-9);
    }

    #[test]
    fn ols_recovers_multi_output_models() {
        // Two outputs sharing the design: y₀ = 1 + x₀, y₁ = −2 + 3·x₁.
        let dataset = affine_fixture(0);

        let mut targets = Vec::new();

        for i in 0..20
        {
            let x0 = dataset.features[(i, 0)];
            let x1 = dataset.features[(i, 1)];
            targets.push(1.0 + x0);
            targets.push(-2.0 + 3.0 * x1);
        }

        let dataset = RegressionDataset {
            features: dataset.features,
            targets: Matrix::from_row_major(20, 2, targets),
            sample_weights: None,
        };

        let report = fit_robust_regression(
            &dataset,
            config(RobustRegressionMethod::OrdinaryLeastSquares),
        )
        .unwrap();

        assert_close(report.model.coefficients[(0, 0)], 1.0, 1.0e-9);
        assert_close(report.model.coefficients[(1, 0)], 0.0, 1.0e-9);
        assert_close(report.model.intercept[0], 1.0, 1.0e-9);
        assert_close(report.model.coefficients[(0, 1)], 0.0, 1.0e-9);
        assert_close(report.model.coefficients[(1, 1)], 3.0, 1.0e-9);
        assert_close(report.model.intercept[1], -2.0, 1.0e-9);
    }

    #[test]
    fn huber_resists_minority_outliers_that_break_ols() {
        let dataset = affine_fixture(3);

        let ols = fit_robust_regression(
            &dataset,
            config(RobustRegressionMethod::OrdinaryLeastSquares),
        )
        .unwrap();

        let huber = fit_robust_regression(
            &dataset,
            RobustRegressionConfig {
                loss: RobustLoss::Huber { delta: 1.345 },
                ..config(RobustRegressionMethod::IterativelyReweightedLeastSquares)
            },
        )
        .unwrap();

        let ols_error = (ols.model.coefficients[(0, 0)] - 2.0).abs()
            + (ols.model.coefficients[(1, 0)] + 3.0).abs()
            + (ols.model.intercept[0] - 5.0).abs();

        let huber_error = (huber.model.coefficients[(0, 0)] - 2.0).abs()
            + (huber.model.coefficients[(1, 0)] + 3.0).abs()
            + (huber.model.intercept[0] - 5.0).abs();

        assert!(ols_error > 1.0, "OLS unexpectedly resisted: {ols_error}");
        assert!(huber.converged);
        assert!(
            huber_error < 1.0e-3,
            "Huber failed to recover: {huber_error}"
        );
        assert!(huber_error * 100.0 < ols_error);
    }

    #[test]
    fn trimmed_recovers_exactly_and_reports_rejected_outliers() {
        let dataset = affine_fixture(3);

        let report = fit_robust_regression(
            &dataset,
            config(RobustRegressionMethod::TrimmedLeastSquares {
                retained_fraction: 0.8,
            }),
        )
        .unwrap();

        assert!(report.converged);
        assert_close(report.model.coefficients[(0, 0)], 2.0, 1.0e-8);
        assert_close(report.model.coefficients[(1, 0)], -3.0, 1.0e-8);
        assert_close(report.model.intercept[0], 5.0, 1.0e-8);

        for outlier in 0..3
        {
            assert!(
                report.rejected_sample_indices.contains(&outlier),
                "outlier {outlier} was not rejected"
            );
        }

        assert_eq!(report.effective_sample_count, 16);
        assert_eq!(
            report.final_weights.iter().filter(|&&w| w == 0.0).count(),
            4
        );
    }

    /// Wider fixture for the block estimator: sixty rows so every block keeps
    /// a well-conditioned design (small blocks over a coarse grid produce
    /// near-collinear block fits, which the parameter median cannot repair —
    /// that is the documented heuristic limit, not a target of this test).
    fn wide_affine_fixture(outliers: usize) -> RegressionDataset {
        let mut rows = Vec::new();
        let mut targets = Vec::new();

        for i in 0..60
        {
            let x0 = (i % 6) as f64 + 0.25 * ((i % 7) as f64);
            let x1 = ((i / 6) as f64) * 0.5 - 2.0;
            let mut y = 5.0 + 2.0 * x0 - 3.0 * x1;

            if i < outliers
            {
                y += 100.0;
            }

            rows.push(x0);
            rows.push(x1);
            targets.push(y);
        }

        RegressionDataset {
            features: Matrix::from_row_major(60, 2, rows),
            targets: Matrix::from_row_major(60, 1, targets),
            sample_weights: None,
        }
    }

    #[test]
    fn median_of_means_is_seeded_and_reproducible() {
        // Each outlier can poison an entire block, so the guarantee needs
        // `outliers <= floor((blocks - 1) / 2)`: two outliers over five blocks
        // leave at least three clean blocks for ANY seed. On noiseless data
        // the clean blocks fit exactly, so the coordinate-wise median lands on
        // an exact clean value.
        let dataset = wide_affine_fixture(2);

        let method = RobustRegressionMethod::MedianOfMeans {
            block_count: 5,
            seed: 0xFEED_5EED,
        };

        let first = fit_robust_regression(&dataset, config(method)).unwrap();
        let second = fit_robust_regression(&dataset, config(method)).unwrap();

        assert_eq!(first, second);

        assert_close(first.model.coefficients[(0, 0)], 2.0, 1.0e-6);
        assert_close(first.model.coefficients[(1, 0)], -3.0, 1.0e-6);
        assert_close(first.model.intercept[0], 5.0, 1.0e-6);
        assert!(!first.warnings.is_empty());
    }

    #[test]
    fn median_of_means_breaks_when_outliers_reach_a_block_majority() {
        // Honest negative: eight outliers over five blocks can contaminate a
        // majority of blocks, and the parameter median then tracks the
        // contaminated fits. The estimator never claims otherwise.
        let dataset = wide_affine_fixture(8);

        let report = fit_robust_regression(
            &dataset,
            config(RobustRegressionMethod::MedianOfMeans {
                block_count: 5,
                seed: 0xFEED_5EED,
            }),
        )
        .unwrap();

        let error = (report.model.coefficients[(0, 0)] - 2.0).abs()
            + (report.model.coefficients[(1, 0)] + 3.0).abs();

        assert!(
            error > 0.5,
            "majority-block contamination unexpectedly recovered (error {error})"
        );
    }

    #[test]
    fn duplicate_feature_without_ridge_is_a_typed_solve_failure() {
        // Two identical columns: rank deficient without regularization.
        let mut rows = Vec::new();

        for i in 0..10
        {
            let x = i as f64;
            rows.push(x);
            rows.push(x);
        }

        let targets: Vec<f64> = (0..10).map(|i| 1.0 + (i as f64)).collect();

        let dataset = RegressionDataset {
            features: Matrix::from_row_major(10, 2, rows),
            targets: Matrix::from_row_major(10, 1, targets),
            sample_weights: None,
        };

        let plain = fit_robust_regression(
            &dataset,
            config(RobustRegressionMethod::OrdinaryLeastSquares),
        );

        assert!(matches!(
            plain,
            Err(RobustRegressionError::LeastSquaresSolveFailed { .. })
        ));

        // An explicit ridge resolves the degeneracy: the caller's choice, not
        // a silent fallback.
        let ridged = fit_robust_regression(
            &dataset,
            RobustRegressionConfig {
                ridge: 1.0e-6,
                ..config(RobustRegressionMethod::OrdinaryLeastSquares)
            },
        )
        .unwrap();

        let prediction = ridged
            .model
            .predict(&Matrix::from_row_major(1, 2, vec![4.0, 4.0]))
            .unwrap();

        assert_close(prediction[(0, 0)], 5.0, 1.0e-2);
    }

    #[test]
    fn zero_weight_samples_are_ignored() {
        let mut dataset = affine_fixture(1);

        let mut weights = vec![1.0; 20];
        weights[0] = 0.0;
        dataset.sample_weights = Some(weights);

        let report = fit_robust_regression(
            &dataset,
            config(RobustRegressionMethod::OrdinaryLeastSquares),
        )
        .unwrap();

        assert_close(report.model.coefficients[(0, 0)], 2.0, 1.0e-9);
        assert_close(report.model.intercept[0], 5.0, 1.0e-9);
        assert_eq!(report.effective_sample_count, 19);
    }

    #[test]
    fn non_convergence_is_reported_not_hidden() {
        let dataset = affine_fixture(5);

        let report = fit_robust_regression(
            &dataset,
            RobustRegressionConfig {
                loss: RobustLoss::Huber { delta: 1.345 },
                maximum_iterations: 1,
                absolute_tolerance: 1.0e-15,
                relative_tolerance: 1.0e-15,
                ..config(RobustRegressionMethod::IterativelyReweightedLeastSquares)
            },
        )
        .unwrap();

        assert!(!report.converged);
        assert_eq!(report.iterations, 1);
        assert!(
            report
                .warnings
                .iter()
                .any(|warning| warning.contains("did not converge"))
        );
    }

    #[test]
    fn feature_rescaling_with_standardization_preserves_predictions() {
        let dataset = affine_fixture(2);

        let probe = Matrix::from_row_major(1, 2, vec![2.0, 0.5]);

        let baseline = fit_robust_regression(
            &dataset,
            RobustRegressionConfig {
                loss: RobustLoss::Huber { delta: 1.345 },
                scale_method: Some(RobustScaleMethod::MedianAbsoluteDeviation),
                ..config(RobustRegressionMethod::IterativelyReweightedLeastSquares)
            },
        )
        .unwrap();

        // Rescale feature 0 by 1e6 and feature 1 by 1e-3; the standardized fit
        // must produce the same predictions at correspondingly rescaled
        // probes.
        let factors = [1.0e6, 1.0e-3];
        let mut rows = Vec::new();

        for i in 0..20
        {
            rows.push(dataset.features[(i, 0)] * factors[0]);
            rows.push(dataset.features[(i, 1)] * factors[1]);
        }

        let rescaled_dataset = RegressionDataset {
            features: Matrix::from_row_major(20, 2, rows),
            targets: dataset.targets.clone(),
            sample_weights: None,
        };

        let rescaled = fit_robust_regression(
            &rescaled_dataset,
            RobustRegressionConfig {
                loss: RobustLoss::Huber { delta: 1.345 },
                scale_method: Some(RobustScaleMethod::MedianAbsoluteDeviation),
                ..config(RobustRegressionMethod::IterativelyReweightedLeastSquares)
            },
        )
        .unwrap();

        let rescaled_probe = Matrix::from_row_major(1, 2, vec![2.0 * factors[0], 0.5 * factors[1]]);

        let baseline_prediction = baseline.model.predict(&probe).unwrap();
        let rescaled_prediction = rescaled.model.predict(&rescaled_probe).unwrap();

        assert_close(
            baseline_prediction[(0, 0)],
            rescaled_prediction[(0, 0)],
            1.0e-6,
        );
    }

    #[test]
    fn row_permutation_agrees_to_solver_tolerance() {
        let dataset = affine_fixture(3);

        let baseline = fit_robust_regression(
            &dataset,
            config(RobustRegressionMethod::TrimmedLeastSquares {
                retained_fraction: 0.8,
            }),
        )
        .unwrap();

        // Reverse the rows: QR rotations differ at the last bit, so agreement
        // is to tolerance, not bit-for-bit (documented).
        let n = dataset.features.rows();
        let mut rows = Vec::new();
        let mut targets = Vec::new();

        for i in (0..n).rev()
        {
            rows.push(dataset.features[(i, 0)]);
            rows.push(dataset.features[(i, 1)]);
            targets.push(dataset.targets[(i, 0)]);
        }

        let reversed = RegressionDataset {
            features: Matrix::from_row_major(n, 2, rows),
            targets: Matrix::from_row_major(n, 1, targets),
            sample_weights: None,
        };

        let permuted = fit_robust_regression(
            &reversed,
            config(RobustRegressionMethod::TrimmedLeastSquares {
                retained_fraction: 0.8,
            }),
        )
        .unwrap();

        assert_close(
            baseline.model.coefficients[(0, 0)],
            permuted.model.coefficients[(0, 0)],
            1.0e-8,
        );
        assert_close(
            baseline.model.intercept[0],
            permuted.model.intercept[0],
            1.0e-8,
        );
    }

    #[test]
    fn fitting_is_bit_deterministic() {
        let dataset = affine_fixture(3);

        for method in [
            RobustRegressionMethod::OrdinaryLeastSquares,
            RobustRegressionMethod::IterativelyReweightedLeastSquares,
            RobustRegressionMethod::TrimmedLeastSquares {
                retained_fraction: 0.8,
            },
            RobustRegressionMethod::MedianOfMeans {
                block_count: 4,
                seed: 7,
            },
        ]
        {
            let configuration = RobustRegressionConfig {
                loss: RobustLoss::Huber { delta: 1.345 },
                ..config(method)
            };

            let first = fit_robust_regression(&dataset, configuration).unwrap();
            let second = fit_robust_regression(&dataset, configuration).unwrap();

            assert_eq!(first, second);
        }
    }

    #[test]
    fn invalid_inputs_are_typed_errors() {
        let empty = RegressionDataset {
            features: Matrix::zeros(0, 0),
            targets: Matrix::zeros(0, 0),
            sample_weights: None,
        };

        assert_eq!(
            fit_robust_regression(&empty, RobustRegressionConfig::default()),
            Err(RobustRegressionError::EmptyDataset)
        );

        let mut dataset = affine_fixture(0);
        dataset.targets[(3, 0)] = f64::NAN;

        assert_eq!(
            fit_robust_regression(&dataset, RobustRegressionConfig::default()),
            Err(RobustRegressionError::NonFiniteTarget { row: 3, col: 0 })
        );

        let mut dataset = affine_fixture(0);
        dataset.features[(2, 1)] = f64::INFINITY;

        assert_eq!(
            fit_robust_regression(&dataset, RobustRegressionConfig::default()),
            Err(RobustRegressionError::NonFiniteFeature { row: 2, col: 1 })
        );

        let mut dataset = affine_fixture(0);
        dataset.sample_weights = Some(vec![-1.0; 20]);

        assert_eq!(
            fit_robust_regression(&dataset, RobustRegressionConfig::default()),
            Err(RobustRegressionError::NegativeWeight { index: 0 })
        );

        let mut dataset = affine_fixture(0);
        dataset.sample_weights = Some(vec![0.0; 20]);

        assert_eq!(
            fit_robust_regression(&dataset, RobustRegressionConfig::default()),
            Err(RobustRegressionError::ZeroTotalWeight)
        );
    }

    #[test]
    fn invalid_configurations_are_typed_errors() {
        let dataset = affine_fixture(0);

        assert_eq!(
            fit_robust_regression(
                &dataset,
                RobustRegressionConfig {
                    ridge: -1.0,
                    ..RobustRegressionConfig::default()
                },
            ),
            Err(RobustRegressionError::InvalidRidge)
        );

        assert_eq!(
            fit_robust_regression(
                &dataset,
                RobustRegressionConfig {
                    maximum_iterations: 0,
                    ..RobustRegressionConfig::default()
                },
            ),
            Err(RobustRegressionError::InvalidTolerance)
        );

        assert_eq!(
            fit_robust_regression(
                &dataset,
                RobustRegressionConfig {
                    loss: RobustLoss::Huber { delta: -1.0 },
                    ..RobustRegressionConfig::default()
                },
            ),
            Err(RobustRegressionError::InvalidLossParameter)
        );

        assert_eq!(
            fit_robust_regression(
                &dataset,
                config(RobustRegressionMethod::TrimmedLeastSquares {
                    retained_fraction: 0.5,
                }),
            ),
            Err(RobustRegressionError::InvalidRetainedFraction)
        );

        assert_eq!(
            fit_robust_regression(
                &dataset,
                config(RobustRegressionMethod::MedianOfMeans {
                    block_count: 0,
                    seed: 0,
                }),
            ),
            Err(RobustRegressionError::InvalidBlockCount)
        );

        assert_eq!(
            fit_robust_regression(
                &dataset,
                config(RobustRegressionMethod::MedianOfMeans {
                    block_count: 21,
                    seed: 0,
                }),
            ),
            Err(RobustRegressionError::InvalidBlockCount)
        );

        // Ten blocks of two rows cannot fit three columns each.
        assert!(matches!(
            fit_robust_regression(
                &dataset,
                config(RobustRegressionMethod::MedianOfMeans {
                    block_count: 10,
                    seed: 0,
                }),
            ),
            Err(RobustRegressionError::InsufficientBlockSamples { .. })
        ));
    }

    #[test]
    fn robust_methods_reject_multiple_outputs() {
        let single = affine_fixture(0);

        let dataset = RegressionDataset {
            targets: {
                let mut targets = Matrix::zeros(20, 2);

                for i in 0..20
                {
                    targets[(i, 0)] = single.targets[(i, 0)];
                    targets[(i, 1)] = 1.0;
                }

                targets
            },
            features: single.features,
            sample_weights: None,
        };

        assert_eq!(
            fit_robust_regression(
                &dataset,
                config(RobustRegressionMethod::IterativelyReweightedLeastSquares),
            ),
            Err(RobustRegressionError::UnsupportedMultiOutput { outputs: 2 })
        );
    }

    #[test]
    fn constant_feature_with_standardization_is_a_typed_error() {
        let mut dataset = affine_fixture(0);

        for i in 0..20
        {
            dataset.features[(i, 1)] = 7.0;
        }

        assert_eq!(
            fit_robust_regression(
                &dataset,
                RobustRegressionConfig {
                    scale_method: Some(RobustScaleMethod::MedianAbsoluteDeviation),
                    ..RobustRegressionConfig::default()
                },
            ),
            Err(RobustRegressionError::DegenerateFeatureScale { dimension: 1 })
        );
    }

    #[test]
    fn underdetermined_designs_are_rejected() {
        let dataset = RegressionDataset {
            features: Matrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            targets: Matrix::from_row_major(2, 1, vec![1.0, 2.0]),
            sample_weights: None,
        };

        assert_eq!(
            fit_robust_regression(&dataset, RobustRegressionConfig::default()),
            Err(RobustRegressionError::InsufficientSamples {
                required: 4,
                found: 2
            })
        );
    }
}
