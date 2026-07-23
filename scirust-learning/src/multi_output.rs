//! Robust multi-output regression (phase 4E.4).
//!
//! Industrial sensor systems often predict several correlated channels at once.
//! The crate's IRLS is single-output; this fits all `k` outputs jointly with a
//! configurable **residual geometry** that makes the difference between three
//! distinct notions of robustness explicit:
//!
//! - [`MultiOutputResidualGeometry::IndependentOutputs`] — **independent marginal
//!   robustness**: each output is down-weighted by its own residual, so a row can
//!   be trusted for one channel and rejected for another. Blind to output
//!   correlation.
//! - [`MultiOutputResidualGeometry::Euclidean`] — **joint isotropic robustness**:
//!   one weight per row from the Euclidean norm of its standardized residual
//!   vector. A row is in or out for all channels together, but the geometry
//!   ignores cross-output covariance.
//! - [`MultiOutputResidualGeometry::RobustMahalanobis`] — **joint multivariate
//!   robustness** with **output-covariance modelling**: one weight per row from
//!   the OGK-robust Mahalanobis norm of its residual vector, so correlated
//!   contamination that is unremarkable per channel is still caught.
//!
//! None of these claims arbitrary-output-contamination robustness: a row whose
//! *every* output is corrupted in a coordinated way can still mislead. Missing
//! outputs are handled by an explicit policy (rows with any non-finite target are
//! dropped and reported). Determinism: fixed-order IRLS with the deterministic
//! OGK scatter; identical inputs give a bit-identical fit.

use core::fmt;

use scirust_multivariate::{
    Matrix as GeometryMatrix, RobustScatterConfig, RobustScatterMethod, RobustScatterModel,
    RobustUnivariateScale,
};
use scirust_solvers::linalg::{Matrix, qr_decompose, solve_qr_least_squares};
use scirust_stats::robust::{MadConsistency, median_absolute_deviation};
use scirust_stats::{ChiSquared, Distribution};

use crate::robust_regression::RobustLoss;

/// A floor on any robust scale before it divides.
const SCALE_FLOOR: f64 = 1.0e-12;

/// How per-row IRLS weights are formed from the multi-output residual vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiOutputResidualGeometry {
    /// Independent per-output down-weighting (marginal robustness).
    IndependentOutputs,
    /// One weight per row from the Euclidean norm of the standardized residual.
    Euclidean,
    /// One weight per row from the OGK-robust Mahalanobis norm of the residual
    /// (joint multivariate robustness with output covariance).
    RobustMahalanobis,
}

/// Configuration for [`fit_multi_output_robust`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiOutputRobustConfig {
    /// The residual geometry driving the weights.
    pub residual_geometry: MultiOutputResidualGeometry,
    /// The IRLS loss (its weight function is applied to standardized residuals).
    pub loss: RobustLoss,
    /// Fit a per-output intercept.
    pub fit_intercept: bool,
    /// Ridge penalty on the feature coefficients (never the intercept).
    pub ridge: f64,
    /// Maximum IRLS iterations.
    pub maximum_iterations: usize,
    /// Relative coefficient-change tolerance for convergence.
    pub tolerance: f64,
    /// Standardize each output by its robust scale before fitting (recommended so
    /// heterogeneous output scales are comparable under the joint geometries).
    pub standardize_outputs: bool,
}

impl Default for MultiOutputRobustConfig {
    fn default() -> Self {
        Self {
            residual_geometry: MultiOutputResidualGeometry::RobustMahalanobis,
            loss: RobustLoss::TukeyBisquare { cutoff: 4.685 },
            fit_intercept: true,
            ridge: 0.0,
            maximum_iterations: 50,
            tolerance: 1.0e-9,
            standardize_outputs: true,
        }
    }
}

/// A reproducible multi-output robust regression report.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiOutputRobustReport {
    /// Coefficients (`p × k`), in the original (un-standardized) output units.
    pub coefficients: Matrix,
    /// Per-output intercepts (`k`; all zero when `fit_intercept` was false).
    pub intercepts: Vec<f64>,
    /// Per-output final robust residual scale (original output units).
    pub output_scales: Vec<f64>,
    /// IRLS iterations performed.
    pub iterations: usize,
    /// Whether IRLS reached its tolerance.
    pub converged: bool,
    /// Rows dropped for a non-finite feature or target (ascending).
    pub dropped_rows: Vec<usize>,
    /// Non-fatal notes.
    pub warnings: Vec<String>,
}

impl MultiOutputRobustReport {
    /// Predict the `k` outputs for one feature row.
    pub fn predict(&self, features: &[f64]) -> Vec<f64> {
        let k = self.intercepts.len();
        let p = self.coefficients.rows();
        (0..k)
            .map(|output| {
                let mut value = self.intercepts[output];
                for (feature, &x) in features.iter().enumerate().take(p)
                {
                    value += self.coefficients[(feature, output)] * x;
                }
                value
            })
            .collect()
    }
}

/// Typed multi-output regression errors.
#[derive(Debug, Clone, PartialEq)]
pub enum MultiOutputError {
    /// The design has zero rows, features, or outputs.
    EmptyDesign,
    /// The feature and target row counts differ.
    RowCountMismatch {
        /// Feature rows.
        features: usize,
        /// Target rows.
        targets: usize,
    },
    /// A feature entry is non-finite (features cannot be dropped like targets).
    NonFiniteFeature {
        /// Row.
        row: usize,
        /// Column.
        col: usize,
        /// Value.
        value: f64,
    },
    /// A configuration value was out of range.
    InvalidConfig {
        /// What was wrong.
        detail: String,
    },
    /// Too few finite rows remained after dropping missing outputs.
    TooFewObservations {
        /// Minimum rows required.
        required: usize,
        /// Rows remaining.
        found: usize,
    },
    /// The (weighted, ridge-augmented) least-squares solve was rank deficient.
    RankDeficient {
        /// The output whose solve failed.
        output: usize,
    },
}

impl fmt::Display for MultiOutputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyDesign =>
            {
                formatter.write_str("the design has zero rows, features, or outputs")
            },
            Self::RowCountMismatch { features, targets } => write!(
                formatter,
                "feature rows {features} do not match target rows {targets}"
            ),
            Self::NonFiniteFeature { row, col, value } =>
            {
                write!(
                    formatter,
                    "non-finite feature {value} at row {row}, column {col}"
                )
            },
            Self::InvalidConfig { detail } => write!(formatter, "invalid configuration: {detail}"),
            Self::TooFewObservations { required, found } => write!(
                formatter,
                "multi-output regression needs at least {required} finite rows, found {found}"
            ),
            Self::RankDeficient { output } =>
            {
                write!(
                    formatter,
                    "the weighted design for output {output} is rank deficient"
                )
            },
        }
    }
}

impl std::error::Error for MultiOutputError {}

/// Fit a robust multi-output regression.
///
/// # Errors
///
/// [`MultiOutputError`] on empty/mismatched input, a non-finite feature, an
/// invalid configuration, too few finite rows, or a rank-deficient weighted solve.
pub fn fit_multi_output_robust(
    features: &Matrix,
    targets: &Matrix,
    config: MultiOutputRobustConfig,
) -> Result<MultiOutputRobustReport, MultiOutputError> {
    let n = features.rows();
    let p = features.cols();
    let k = targets.cols();
    if n == 0 || p == 0 || k == 0
    {
        return Err(MultiOutputError::EmptyDesign);
    }
    if targets.rows() != n
    {
        return Err(MultiOutputError::RowCountMismatch {
            features: n,
            targets: targets.rows(),
        });
    }
    validate_config(&config)?;

    // Features must be finite (they cannot be imputed like a missing output).
    for i in 0..n
    {
        for j in 0..p
        {
            let value = features[(i, j)];
            if !value.is_finite()
            {
                return Err(MultiOutputError::NonFiniteFeature {
                    row: i,
                    col: j,
                    value,
                });
            }
        }
    }

    // Missing-output policy: drop any row with a non-finite target.
    let mut dropped_rows = Vec::new();
    let mut kept: Vec<usize> = Vec::new();
    for i in 0..n
    {
        if (0..k).any(|j| !targets[(i, j)].is_finite())
        {
            dropped_rows.push(i);
        }
        else
        {
            kept.push(i);
        }
    }
    let m = kept.len();
    let fitted_columns = p + usize::from(config.fit_intercept);
    if m < fitted_columns + 1
    {
        return Err(MultiOutputError::TooFewObservations {
            required: fitted_columns + 1,
            found: m,
        });
    }

    let mut warnings = Vec::new();

    // Standardize each output by its robust scale (fit on kept rows).
    let output_scale: Vec<f64> = (0..k)
        .map(|j| {
            if config.standardize_outputs
            {
                let column: Vec<f64> = kept.iter().map(|&i| targets[(i, j)]).collect();
                robust_scale(&column)
            }
            else
            {
                1.0
            }
        })
        .collect();

    // Kept feature / standardized-target views.
    let feature_rows: Vec<Vec<f64>> = kept
        .iter()
        .map(|&i| (0..p).map(|j| features[(i, j)]).collect())
        .collect();
    let target_rows: Vec<Vec<f64>> = kept
        .iter()
        .map(|&i| (0..k).map(|j| targets[(i, j)] / output_scale[j]).collect())
        .collect();

    // Initial OLS per output (unit weights).
    let mut beta = vec![vec![0.0_f64; fitted_columns]; k]; // beta[output][column]
    for (output, beta_column) in beta.iter_mut().enumerate()
    {
        let response: Vec<f64> = target_rows.iter().map(|row| row[output]).collect();
        *beta_column = weighted_ols(
            &feature_rows,
            &response,
            None,
            config.fit_intercept,
            config.ridge,
        )
        .ok_or(MultiOutputError::RankDeficient { output })?;
    }

    let mut iterations = 0;
    let mut converged = false;

    for _ in 0..config.maximum_iterations
    {
        iterations += 1;

        // Residual matrix (standardized output units).
        let residuals: Vec<Vec<f64>> = (0..m)
            .map(|i| {
                (0..k)
                    .map(|output| {
                        target_rows[i][output]
                            - predict_row(&feature_rows[i], &beta[output], config.fit_intercept)
                    })
                    .collect()
            })
            .collect();

        let weights = row_weights(
            &residuals,
            k,
            config.residual_geometry,
            config.loss,
            &mut warnings,
        );

        let mut next = beta.clone();
        let mut max_change = 0.0_f64;
        for (output, next_column) in next.iter_mut().enumerate()
        {
            let response: Vec<f64> = target_rows.iter().map(|row| row[output]).collect();
            let column_weights: Vec<f64> = match &weights
            {
                RowWeights::PerOutput(matrix) => (0..m).map(|i| matrix[i][output]).collect(),
                RowWeights::PerRow(vector) => vector.clone(),
            };
            let solved = weighted_ols(
                &feature_rows,
                &response,
                Some(&column_weights),
                config.fit_intercept,
                config.ridge,
            )
            .ok_or(MultiOutputError::RankDeficient { output })?;
            for (old, new) in beta[output].iter().zip(&solved)
            {
                max_change = max_change.max((old - new).abs());
            }
            *next_column = solved;
        }
        let magnitude = beta
            .iter()
            .flat_map(|column| column.iter())
            .fold(1.0_f64, |acc, value| acc.max(value.abs()));
        beta = next;
        if max_change <= config.tolerance * magnitude
        {
            converged = true;
            break;
        }
    }

    // Un-standardize coefficients and intercepts back to original output units.
    let mut coefficients = Matrix::zeros(p, k);
    let mut intercepts = vec![0.0_f64; k];
    for output in 0..k
    {
        for feature in 0..p
        {
            coefficients[(feature, output)] = beta[output][feature] * output_scale[output];
        }
        if config.fit_intercept
        {
            intercepts[output] = beta[output][p] * output_scale[output];
        }
    }

    // Final robust residual scale per output (original units).
    let output_scales: Vec<f64> = (0..k)
        .map(|output| {
            let residual: Vec<f64> = (0..m)
                .map(|i| {
                    (target_rows[i][output]
                        - predict_row(&feature_rows[i], &beta[output], config.fit_intercept))
                        * output_scale[output]
                })
                .collect();
            robust_scale(&residual)
        })
        .collect();

    Ok(MultiOutputRobustReport {
        coefficients,
        intercepts,
        output_scales,
        iterations,
        converged,
        dropped_rows,
        warnings,
    })
}

fn validate_config(config: &MultiOutputRobustConfig) -> Result<(), MultiOutputError> {
    if !(config.ridge.is_finite() && config.ridge >= 0.0)
    {
        return Err(MultiOutputError::InvalidConfig {
            detail: "ridge must be finite and non-negative".to_string(),
        });
    }
    if !(config.tolerance.is_finite() && config.tolerance > 0.0)
    {
        return Err(MultiOutputError::InvalidConfig {
            detail: "tolerance must be finite and positive".to_string(),
        });
    }
    Ok(())
}

/// Per-row IRLS weights, either one per (row, output) or one per row.
enum RowWeights {
    PerOutput(Vec<Vec<f64>>),
    PerRow(Vec<f64>),
}

fn row_weights(
    residuals: &[Vec<f64>],
    k: usize,
    geometry: MultiOutputResidualGeometry,
    loss: RobustLoss,
    warnings: &mut Vec<String>,
) -> RowWeights {
    match geometry
    {
        MultiOutputResidualGeometry::IndependentOutputs =>
        {
            let scales: Vec<f64> = (0..k)
                .map(|output| {
                    robust_scale(&residuals.iter().map(|row| row[output]).collect::<Vec<_>>())
                })
                .collect();
            let matrix: Vec<Vec<f64>> = residuals
                .iter()
                .map(|row| {
                    (0..k)
                        .map(|output| loss_weight(row[output] / scales[output], loss))
                        .collect()
                })
                .collect();
            RowWeights::PerOutput(matrix)
        },
        MultiOutputResidualGeometry::Euclidean =>
        {
            let scales: Vec<f64> = (0..k)
                .map(|output| {
                    robust_scale(&residuals.iter().map(|row| row[output]).collect::<Vec<_>>())
                })
                .collect();
            let norms: Vec<f64> = residuals
                .iter()
                .map(|row| {
                    (0..k)
                        .map(|output| (row[output] / scales[output]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                })
                .collect();
            let reference = joint_reference(&norms, k);
            let weights: Vec<f64> = norms
                .iter()
                .map(|&norm| loss_weight(norm / reference, loss))
                .collect();
            RowWeights::PerRow(weights)
        },
        MultiOutputResidualGeometry::RobustMahalanobis => match mahalanobis_norms(residuals, k)
        {
            Some(distances) =>
            {
                let reference = (ChiSquared::new(k as f64).quantile(0.5))
                    .sqrt()
                    .max(SCALE_FLOOR);
                let weights: Vec<f64> = distances
                    .iter()
                    .map(|&d| loss_weight(d / reference, loss))
                    .collect();
                RowWeights::PerRow(weights)
            },
            None =>
            {
                warnings.push(
                        "robust residual scatter unavailable (singular); falling back to Euclidean geometry"
                            .to_string(),
                    );
                row_weights(
                    residuals,
                    k,
                    MultiOutputResidualGeometry::Euclidean,
                    loss,
                    &mut Vec::new(),
                )
            },
        },
    }
}

/// OGK-robust Mahalanobis norms of the residual vectors (`None` if the robust
/// scatter is singular).
fn mahalanobis_norms(residuals: &[Vec<f64>], k: usize) -> Option<Vec<f64>> {
    let m = residuals.len();
    let geometry = GeometryMatrix {
        rows: m,
        cols: k,
        data: residuals.to_vec(),
    };
    let config = RobustScatterConfig {
        method: RobustScatterMethod::Ogk {
            scale: RobustUnivariateScale::MedianAbsoluteDeviation,
            reweight: true,
        },
        ridge: 1.0e-9,
        ..RobustScatterConfig::default()
    };
    let model = RobustScatterModel::fit(&geometry, config).ok()?;
    let mut distances = Vec::with_capacity(m);
    for row in &geometry.data
    {
        distances.push(model.mahalanobis(row).ok()?);
    }
    Some(distances)
}

/// A robust scale for the norms, normalizing the typical `χ_k` magnitude to ~1.
fn joint_reference(norms: &[f64], k: usize) -> f64 {
    let median = describe_median(norms);
    let chi = (ChiSquared::new(k as f64).quantile(0.5)).sqrt();
    (median / chi.max(SCALE_FLOOR)).max(SCALE_FLOOR)
}

/// Normal-consistent robust scale (MAD), floored.
fn robust_scale(values: &[f64]) -> f64 {
    median_absolute_deviation(values, MadConsistency::Normal)
        .unwrap_or(0.0)
        .max(SCALE_FLOOR)
}

fn describe_median(values: &[f64]) -> f64 {
    scirust_stats::describe::median(values)
}

/// The IRLS weight `ψ(u)/u` of the loss at a standardized residual `u`.
fn loss_weight(u: f64, loss: RobustLoss) -> f64 {
    match loss
    {
        RobustLoss::Squared => 1.0,
        RobustLoss::AbsoluteApprox { epsilon } => 1.0 / (u * u + epsilon * epsilon).sqrt(),
        RobustLoss::Huber { delta } =>
        {
            let magnitude = u.abs();
            if magnitude <= delta
            {
                1.0
            }
            else
            {
                delta / magnitude
            }
        },
        RobustLoss::TukeyBisquare { cutoff } =>
        {
            let t = u / cutoff;
            if t.abs() <= 1.0
            {
                let quad = 1.0 - t * t;
                quad * quad
            }
            else
            {
                0.0
            }
        },
    }
}

fn predict_row(features: &[f64], beta: &[f64], fit_intercept: bool) -> f64 {
    let p = features.len();
    let mut value = 0.0;
    for (j, &feature) in features.iter().enumerate()
    {
        value += feature * beta[j];
    }
    if fit_intercept
    {
        value += beta[p];
    }
    value
}

/// Weighted (and optionally ridge-augmented) OLS through the QR of the
/// `√w`-scaled design `[features | intercept]`. `None` if rank deficient.
fn weighted_ols(
    feature_rows: &[Vec<f64>],
    response: &[f64],
    weights: Option<&[f64]>,
    fit_intercept: bool,
    ridge: f64,
) -> Option<Vec<f64>> {
    let m = feature_rows.len();
    let p = feature_rows[0].len();
    let fitted_columns = p + usize::from(fit_intercept);
    let ridge_rows = if ridge > 0.0 { p } else { 0 };
    let rows = m + ridge_rows;
    if rows < fitted_columns
    {
        return None;
    }

    let mut design = Matrix::zeros(rows, fitted_columns);
    let mut rhs = vec![0.0_f64; rows];
    for i in 0..m
    {
        let root = weights.map_or(1.0, |w| w[i].max(0.0).sqrt());
        for j in 0..p
        {
            design[(i, j)] = root * feature_rows[i][j];
        }
        if fit_intercept
        {
            design[(i, p)] = root;
        }
        rhs[i] = root * response[i];
    }
    let root_ridge = ridge.sqrt();
    for r in 0..ridge_rows
    {
        design[(m + r, r)] = root_ridge;
    }

    let factorization = qr_decompose(design).ok()?;
    if factorization.rcond().unwrap_or(0.0) < 1.0e-12
    {
        return None;
    }
    solve_qr_least_squares(&factorization, &rhs).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// x = i over 0..n; true outputs y1 = 2x, y2 = -x. `corrupt` may mutate the
    /// output pair for a row (to inject contamination or missing values).
    fn dataset(n: usize, corrupt: impl Fn(usize, &mut [f64; 2])) -> (Matrix, Matrix) {
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n * 2);
        for i in 0..n
        {
            let x = i as f64;
            let mut y = [2.0 * x, -x];
            corrupt(i, &mut y);
            xs.push(x);
            ys.push(y[0]);
            ys.push(y[1]);
        }
        (
            Matrix::from_row_major(n, 1, xs),
            Matrix::from_row_major(n, 2, ys),
        )
    }

    fn fit(
        features: &Matrix,
        targets: &Matrix,
        geometry: MultiOutputResidualGeometry,
    ) -> MultiOutputRobustReport {
        fit_multi_output_robust(
            features,
            targets,
            MultiOutputRobustConfig {
                residual_geometry: geometry,
                ..MultiOutputRobustConfig::default()
            },
        )
        .unwrap()
    }

    fn close(report: &MultiOutputRobustReport, tolerance: f64) -> bool {
        (report.coefficients[(0, 0)] - 2.0).abs() < tolerance
            && (report.coefficients[(0, 1)] + 1.0).abs() < tolerance
    }

    #[test]
    fn recovers_clean_multi_output_coefficients() {
        let (features, targets) = dataset(60, |_, _| {});
        for geometry in [
            MultiOutputResidualGeometry::IndependentOutputs,
            MultiOutputResidualGeometry::Euclidean,
            MultiOutputResidualGeometry::RobustMahalanobis,
        ]
        {
            let report = fit(&features, &targets, geometry);
            assert!(
                close(&report, 0.01),
                "{geometry:?}: {:?}",
                report.coefficients
            );
            assert_eq!(report.intercepts.len(), 2);
        }
    }

    #[test]
    fn resists_a_single_corrupted_output() {
        // Corrupt only output 1 on ten rows; output 2 stays clean.
        let (features, targets) = dataset(60, |i, y| {
            if i % 6 == 0
            {
                y[0] += 400.0;
            }
        });
        let report = fit(
            &features,
            &targets,
            MultiOutputResidualGeometry::IndependentOutputs,
        );
        assert!(
            close(&report, 0.1),
            "independent should recover both: {:?}",
            report.coefficients
        );
    }

    #[test]
    fn resists_minority_rows_corrupted_across_all_outputs() {
        let (features, targets) = dataset(60, |i, y| {
            if i % 6 == 0
            {
                y[0] += 400.0;
                y[1] -= 400.0;
            }
        });
        let report = fit(
            &features,
            &targets,
            MultiOutputResidualGeometry::RobustMahalanobis,
        );
        assert!(
            close(&report, 0.1),
            "mahalanobis should recover: {:?}",
            report.coefficients
        );
    }

    #[test]
    fn robust_mahalanobis_resists_correlated_contamination() {
        // A cluster of rows shifted along a correlated direction (both outputs up).
        let (features, targets) = dataset(60, |i, y| {
            if (20..30).contains(&i)
            {
                y[0] += 30.0;
                y[1] += 30.0;
            }
        });
        let report = fit(
            &features,
            &targets,
            MultiOutputResidualGeometry::RobustMahalanobis,
        );
        assert!(
            close(&report, 0.2),
            "mahalanobis under correlated contamination: {:?}",
            report.coefficients
        );
    }

    #[test]
    fn handles_output_scale_differences() {
        // Output 2 expressed in a 1000x unit; standardization must recover -1000.
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for i in 0..60
        {
            let x = i as f64;
            xs.push(x);
            ys.push(2.0 * x);
            ys.push(-1000.0 * x);
        }
        let features = Matrix::from_row_major(60, 1, xs);
        let targets = Matrix::from_row_major(60, 2, ys);
        let report = fit(
            &features,
            &targets,
            MultiOutputResidualGeometry::RobustMahalanobis,
        );
        assert!((report.coefficients[(0, 0)] - 2.0).abs() < 0.01);
        assert!(
            (report.coefficients[(0, 1)] + 1000.0).abs() < 5.0,
            "{}",
            report.coefficients[(0, 1)]
        );
    }

    #[test]
    fn drops_and_reports_missing_output_rows() {
        let (features, targets) = dataset(60, |i, y| {
            if i == 10 || i == 25
            {
                y[1] = f64::NAN;
            }
        });
        let report = fit(
            &features,
            &targets,
            MultiOutputResidualGeometry::IndependentOutputs,
        );
        assert_eq!(report.dropped_rows, vec![10, 25]);
        assert!(close(&report, 0.05));
    }

    #[test]
    fn is_deterministic() {
        let (features, targets) = dataset(60, |i, y| {
            if i % 6 == 0
            {
                y[0] += 400.0;
            }
        });
        let a = fit(
            &features,
            &targets,
            MultiOutputResidualGeometry::RobustMahalanobis,
        );
        let b = fit(
            &features,
            &targets,
            MultiOutputResidualGeometry::RobustMahalanobis,
        );
        assert_eq!(a, b);
    }

    #[test]
    fn invalid_inputs_are_typed_errors() {
        let (features, targets) = dataset(40, |_, _| {});
        assert!(matches!(
            fit_multi_output_robust(
                &features,
                &targets,
                MultiOutputRobustConfig {
                    ridge: -1.0,
                    ..MultiOutputRobustConfig::default()
                }
            )
            .unwrap_err(),
            MultiOutputError::InvalidConfig { .. }
        ));
        let mismatched = Matrix::from_row_major(3, 2, vec![1.0; 6]);
        assert!(matches!(
            fit_multi_output_robust(&features, &mismatched, MultiOutputRobustConfig::default())
                .unwrap_err(),
            MultiOutputError::RowCountMismatch { .. }
        ));
    }

    #[test]
    fn predict_returns_all_outputs() {
        let (features, targets) = dataset(60, |_, _| {});
        let report = fit(&features, &targets, MultiOutputResidualGeometry::Euclidean);
        let prediction = report.predict(&[10.0]);
        assert_eq!(prediction.len(), 2);
        assert!((prediction[0] - 20.0).abs() < 0.2);
        assert!((prediction[1] + 10.0).abs() < 0.2);
    }
}
