//! Linear quantile regression via iteratively-reweighted least squares.
//!
//! Ordinary least squares estimates the conditional *mean*; under heavy tails the
//! mean is a poor summary and its `±σ` band lies. **Quantile regression** targets a
//! conditional quantile `τ ∈ (0, 1)` directly by minimizing the pinball (check)
//! loss `ρ_τ(r) = r · (τ − 𝟙[r < 0])`, so `τ = 0.5` is median (least-absolute)
//! regression and a pair `(τ_lo, τ_hi)` gives a **native** prediction interval
//! `[q_{τ_lo}(x), q_{τ_hi}(x)]` whose width adapts to the local noise — no
//! distributional assumption.
//!
//! The pinball loss is a *weighted* absolute loss with weight `τ` on positive
//! residuals and `1 − τ` on negative ones, so it is fit here by the classical
//! Schlossmacher IRLS: start from OLS, then repeatedly solve a weighted least
//! squares with `wᵢ = (τ if rᵢ ≥ 0 else 1 − τ) / max(|rᵢ|, ε)` — reusing the
//! crate's tested weighted-OLS path ([`fit_robust_regression`]) — to a fixed point.
//! This is a **smoothed approximation**, honestly named: the `ε` floor that keeps
//! the weights finite biases the fit slightly away from the exact linear-programming
//! quantile solution, and the reported [`QuantileRegressionReport::converged`] flag
//! and pinball loss say what actually happened. Deterministic: OLS is RNG-free and
//! the reweighting is a fixed recurrence — no seed.

use core::fmt;

use crate::robust_regression::{
    LinearRegressionModel, RegressionDataset, RobustRegressionConfig, RobustRegressionError,
    RobustRegressionMethod, fit_robust_regression,
};

/// Configuration for [`fit_quantile_regression`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantileRegressionConfig {
    /// Target quantile in the open interval `(0, 1)`.
    pub tau: f64,
    /// Maximum IRLS iterations.
    pub maximum_iterations: usize,
    /// Absolute coefficient-change tolerance for convergence.
    pub tolerance: f64,
    /// Residual floor keeping the IRLS weights finite (`> 0`).
    pub epsilon: f64,
    /// Fit a per-output intercept.
    pub fit_intercept: bool,
    /// Ridge penalty passed to the inner weighted least squares (`0` disables it).
    pub ridge: f64,
}

impl QuantileRegressionConfig {
    /// A configuration for the quantile `tau` with sensible defaults.
    pub fn new(tau: f64) -> Self {
        Self {
            tau,
            maximum_iterations: 100,
            tolerance: 1.0e-8,
            epsilon: 1.0e-6,
            fit_intercept: true,
            ridge: 0.0,
        }
    }
}

impl Default for QuantileRegressionConfig {
    fn default() -> Self {
        Self::new(0.5)
    }
}

/// A fitted quantile regressor and its diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct QuantileRegressionReport {
    /// The fitted linear model (predicts the conditional `τ`-quantile).
    pub model: LinearRegressionModel,
    /// The quantile that was fitted.
    pub tau: f64,
    /// IRLS iterations performed.
    pub iterations: usize,
    /// Whether the coefficient sequence converged within tolerance.
    pub converged: bool,
    /// Mean pinball loss on the training data under the final model.
    pub pinball_loss: f64,
}

/// Errors returned by [`fit_quantile_regression`].
#[derive(Clone, Debug, PartialEq)]
pub enum QuantileRegressionError {
    /// `tau` was not in the open interval `(0, 1)`.
    InvalidTau {
        /// The rejected quantile.
        tau: f64,
    },
    /// The residual floor `epsilon` was not strictly positive.
    NonPositiveEpsilon {
        /// The rejected floor.
        epsilon: f64,
    },
    /// The underlying weighted least squares failed.
    Regression(RobustRegressionError),
}

impl fmt::Display for QuantileRegressionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::InvalidTau { tau } =>
            {
                write!(
                    formatter,
                    "quantile tau {tau} must lie in the open interval (0, 1)"
                )
            },
            Self::NonPositiveEpsilon { epsilon } =>
            {
                write!(
                    formatter,
                    "residual floor epsilon {epsilon} must be positive"
                )
            },
            Self::Regression(error) =>
            {
                write!(formatter, "weighted least squares failed: {error}")
            },
        }
    }
}

impl std::error::Error for QuantileRegressionError {}

impl From<RobustRegressionError> for QuantileRegressionError {
    fn from(error: RobustRegressionError) -> Self {
        Self::Regression(error)
    }
}

/// The pinball (check) loss `ρ_τ(r) = r · (τ − 𝟙[r < 0])`.
fn pinball(residual: f64, tau: f64) -> f64 {
    if residual >= 0.0
    {
        tau * residual
    }
    else
    {
        (tau - 1.0) * residual
    }
}

/// Single-output residuals `yᵢ − ŷᵢ` under `model`.
fn residuals(
    model: &LinearRegressionModel,
    dataset: &RegressionDataset,
) -> Result<Vec<f64>, QuantileRegressionError> {
    let predictions = model.predict(&dataset.features)?;
    Ok((0..dataset.targets.rows())
        .map(|row| dataset.targets[(row, 0)] - predictions[(row, 0)])
        .collect())
}

/// Maximum absolute change between two coefficient matrices (`p × 1`).
fn coefficient_shift(previous: &LinearRegressionModel, current: &LinearRegressionModel) -> f64 {
    let rows = current.coefficients.rows();
    let mut shift: f64 = 0.0;

    for row in 0..rows
    {
        let delta = (current.coefficients[(row, 0)] - previous.coefficients[(row, 0)]).abs();
        shift = shift.max(delta);
    }

    for (a, b) in current.intercept.iter().zip(&previous.intercept)
    {
        shift = shift.max((a - b).abs());
    }

    shift
}

/// Fits a linear quantile regression by Schlossmacher IRLS.
///
/// # Errors
///
/// [`QuantileRegressionError::InvalidTau`] / [`QuantileRegressionError::NonPositiveEpsilon`]
/// for a malformed configuration, and [`QuantileRegressionError::Regression`] when the
/// inner weighted least squares rejects the data (shape, finiteness, degeneracy).
pub fn fit_quantile_regression(
    dataset: &RegressionDataset,
    config: QuantileRegressionConfig,
) -> Result<QuantileRegressionReport, QuantileRegressionError> {
    if !(config.tau > 0.0 && config.tau < 1.0)
    {
        return Err(QuantileRegressionError::InvalidTau { tau: config.tau });
    }

    if !config.epsilon.is_finite() || config.epsilon <= 0.0
    {
        return Err(QuantileRegressionError::NonPositiveEpsilon {
            epsilon: config.epsilon,
        });
    }

    let base = RobustRegressionConfig {
        method: RobustRegressionMethod::OrdinaryLeastSquares,
        fit_intercept: config.fit_intercept,
        ridge: config.ridge,
        ..RobustRegressionConfig::default()
    };

    // Start from ordinary least squares (weights all one).
    let mut model = fit_robust_regression(dataset, base)?.model;
    let mut converged = false;
    let mut iterations = 0;

    for _ in 0..config.maximum_iterations
    {
        iterations += 1;

        let current_residuals = residuals(&model, dataset)?;
        let weights: Vec<f64> = current_residuals
            .iter()
            .map(|&residual| {
                let asymmetric = if residual >= 0.0
                {
                    config.tau
                }
                else
                {
                    1.0 - config.tau
                };
                asymmetric / residual.abs().max(config.epsilon)
            })
            .collect();

        let weighted = RegressionDataset {
            features: dataset.features.clone(),
            targets: dataset.targets.clone(),
            sample_weights: Some(weights),
        };

        let next = fit_robust_regression(&weighted, base)?.model;

        if coefficient_shift(&model, &next) < config.tolerance
        {
            model = next;
            converged = true;
            break;
        }

        model = next;
    }

    let final_residuals = residuals(&model, dataset)?;
    let pinball_loss = final_residuals
        .iter()
        .map(|&residual| pinball(residual, config.tau))
        .sum::<f64>()
        / final_residuals.len().max(1) as f64;

    Ok(QuantileRegressionReport {
        model,
        tau: config.tau,
        iterations,
        converged,
        pinball_loss,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirust_solvers::linalg::Matrix;

    fn dataset(features: &[&[f64]], targets: &[f64]) -> RegressionDataset {
        let rows = features.len();
        let cols = features[0].len();
        let mut flat = Vec::with_capacity(rows * cols);
        for row in features
        {
            flat.extend_from_slice(row);
        }
        RegressionDataset {
            features: Matrix::from_row_major(rows, cols, flat),
            targets: Matrix::from_row_major(rows, 1, targets.to_vec()),
            sample_weights: None,
        }
    }

    #[test]
    fn median_regression_recovers_a_clean_line() {
        // y = 2x + 1 exactly; the median fit must recover it.
        let features: Vec<Vec<f64>> = (0..20).map(|i| vec![i as f64]).collect();
        let rows: Vec<&[f64]> = features.iter().map(Vec::as_slice).collect();
        let targets: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 1.0).collect();
        let data = dataset(&rows, &targets);

        let report = fit_quantile_regression(&data, QuantileRegressionConfig::new(0.5)).unwrap();
        assert!((report.model.coefficients[(0, 0)] - 2.0).abs() < 1e-3);
        assert!((report.model.intercept[0] - 1.0).abs() < 1e-3);
        assert!(report.pinball_loss < 1e-6);
    }

    #[test]
    fn quantiles_are_ordered() {
        // Noisy-ish data: the 0.1 quantile line should sit below the 0.9 line.
        let features: Vec<Vec<f64>> = (0..60).map(|i| vec![i as f64]).collect();
        let rows: Vec<&[f64]> = features.iter().map(Vec::as_slice).collect();
        let targets: Vec<f64> = (0..60)
            .map(|i| i as f64 + if i % 2 == 0 { 5.0 } else { -5.0 })
            .collect();
        let data = dataset(&rows, &targets);

        let low = fit_quantile_regression(&data, QuantileRegressionConfig::new(0.1)).unwrap();
        let high = fit_quantile_regression(&data, QuantileRegressionConfig::new(0.9)).unwrap();

        // Evaluate both fitted lines at the mean predictor; low ≤ high.
        let x = Matrix::from_row_major(1, 1, vec![30.0]);
        let low_prediction = low.model.predict(&x).unwrap()[(0, 0)];
        let high_prediction = high.model.predict(&x).unwrap()[(0, 0)];
        assert!(low_prediction <= high_prediction);
    }

    #[test]
    fn the_median_line_balances_residual_signs() {
        // For an odd number of points on a noisy line, the median fit should leave
        // roughly half the residuals on each side.
        let features: Vec<Vec<f64>> = (0..41).map(|i| vec![i as f64]).collect();
        let rows: Vec<&[f64]> = features.iter().map(Vec::as_slice).collect();
        let targets: Vec<f64> = (0..41)
            .map(|i| 0.5 * i as f64 + ((i * 13) % 7) as f64 - 3.0)
            .collect();
        let data = dataset(&rows, &targets);

        let report = fit_quantile_regression(&data, QuantileRegressionConfig::new(0.5)).unwrap();
        let predictions = report.model.predict(&data.features).unwrap();
        let above = (0..data.targets.rows())
            .filter(|&r| data.targets[(r, 0)] > predictions[(r, 0)])
            .count();
        // Between a third and two-thirds — the median is not wildly off-balance.
        assert!((13..=28).contains(&above), "unbalanced: {above}/41 above");
    }

    #[test]
    fn rejects_a_malformed_configuration() {
        let features: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
        let rows: Vec<&[f64]> = features.iter().map(Vec::as_slice).collect();
        let data = dataset(&rows, &[0.0, 1.0, 2.0, 3.0, 4.0]);

        assert_eq!(
            fit_quantile_regression(&data, QuantileRegressionConfig::new(0.0)).unwrap_err(),
            QuantileRegressionError::InvalidTau { tau: 0.0 }
        );
        let bad_epsilon = QuantileRegressionConfig {
            epsilon: 0.0,
            ..QuantileRegressionConfig::new(0.5)
        };
        assert_eq!(
            fit_quantile_regression(&data, bad_epsilon).unwrap_err(),
            QuantileRegressionError::NonPositiveEpsilon { epsilon: 0.0 }
        );
    }

    #[test]
    fn fit_is_deterministic() {
        let features: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64, (i * i) as f64]).collect();
        let rows: Vec<&[f64]> = features.iter().map(Vec::as_slice).collect();
        let targets: Vec<f64> = (0..30).map(|i| 3.0 * i as f64 - 1.0).collect();
        let data = dataset(&rows, &targets);

        let a = fit_quantile_regression(&data, QuantileRegressionConfig::new(0.7)).unwrap();
        let b = fit_quantile_regression(&data, QuantileRegressionConfig::new(0.7)).unwrap();
        assert_eq!(a, b);
    }
}
