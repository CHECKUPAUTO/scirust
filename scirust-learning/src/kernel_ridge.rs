//! Deterministic RBF kernel ridge regression — a multivariate nonlinear regressor.
//!
//! Where [`crate::robust_regression`] fits an affine map and
//! [`crate::isotonic`] a monotone one-dimensional recalibration, this fits a
//! genuinely **multivariate nonlinear** function through the kernel trick: with
//! the radial basis function kernel `k(x, z) = exp(−γ‖x − z‖²)`, kernel ridge
//! regression is the closed-form minimiser of `‖Kα − y‖² + λ αᵀKα`, whose dual
//! solution solves the symmetric positive-definite system
//!
//! ```text
//!   (K + λI) α = y
//! ```
//!
//! and predicts `f(x) = Σᵢ αᵢ k(x, xᵢ)`. It captures feature interactions and
//! curvature a linear or rank-preserving model cannot — the model class beyond
//! a one-dimensional monotone recalibration.
//!
//! # Reuse, not reinvention
//!
//! The SPD solve is `scirust-solvers`' Cholesky factorization
//! (`cholesky_decompose` + `solve_cholesky`); `λ > 0` makes `K + λI` strictly
//! positive definite, so the factorization is well posed. No second linear
//! algebra package.
//!
//! # Determinism contract
//!
//! No RNG. The kernel matrix, its Cholesky factor and the dual solve are all
//! deterministic functions of the inputs, so identical training data and
//! hyper-parameters give a bit-identical model and identical predictions on
//! every run. (Row order is not claimed as a free invariance: the kernel
//! matrix and its factorization depend on it at the last-bit level.)
//!
//! # Scope honesty
//!
//! Kernel ridge is a smoother, not a robust estimator: every training residual
//! enters the squared loss, so a grossly corrupted target still bends the fit.
//! It buys nonlinearity, not contamination resistance. The bandwidth `γ` and
//! ridge `λ` must be chosen (e.g. on a validation split); this module fits a
//! given pair and does not select them.

use core::fmt;

use scirust_solvers::linalg::{Matrix, cholesky_decompose, solve_cholesky};

/// Hyper-parameters for [`KernelRidgeRegression::fit`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KernelRidgeConfig {
    /// RBF bandwidth `γ > 0` in `k(x, z) = exp(−γ‖x − z‖²)`.
    pub gamma: f64,
    /// Ridge `λ > 0` added to the kernel diagonal.
    pub ridge: f64,
}

/// A fitted RBF kernel ridge model: its support points, their dual weights, and
/// the bandwidth.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelRidgeRegression {
    support: Vec<Vec<f64>>,
    dual_weights: Vec<f64>,
    gamma: f64,
}

/// Errors returned by [`KernelRidgeRegression::fit`].
#[derive(Clone, Debug, PartialEq)]
pub enum KernelRidgeError {
    /// No training rows were supplied.
    EmptyDataset,
    /// The feature-row count and target count differ.
    LengthMismatch {
        /// Number of feature rows.
        features: usize,
        /// Number of targets.
        targets: usize,
    },
    /// A feature row has a different width from the first.
    InconsistentRowWidth {
        /// The offending row index.
        row: usize,
        /// The expected width (row 0's width).
        expected: usize,
        /// The width found.
        found: usize,
    },
    /// A feature value is not finite.
    NonFiniteFeature {
        /// Row of the offending value.
        row: usize,
        /// Column of the offending value.
        column: usize,
    },
    /// A target value is not finite.
    NonFiniteTarget {
        /// Index of the offending target.
        index: usize,
    },
    /// The bandwidth `γ` was not strictly positive.
    NonPositiveGamma {
        /// The offending bandwidth.
        gamma: f64,
    },
    /// The ridge `λ` was not strictly positive.
    NonPositiveRidge {
        /// The offending ridge.
        ridge: f64,
    },
    /// The linear solver failed (e.g. the regularized kernel was numerically
    /// not positive definite).
    SolverFailure {
        /// The solver's own message.
        detail: String,
    },
}

impl fmt::Display for KernelRidgeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            Self::EmptyDataset =>
            {
                write!(
                    formatter,
                    "kernel ridge regression needs at least one sample"
                )
            },
            Self::LengthMismatch { features, targets } => write!(
                formatter,
                "feature/target length mismatch: {features} rows, {targets} targets"
            ),
            Self::InconsistentRowWidth {
                row,
                expected,
                found,
            } => write!(
                formatter,
                "feature row {row} has width {found}, expected {expected}"
            ),
            Self::NonFiniteFeature { row, column } => write!(
                formatter,
                "feature at row {row}, column {column} is not finite"
            ),
            Self::NonFiniteTarget { index } =>
            {
                write!(formatter, "target at index {index} is not finite")
            },
            Self::NonPositiveGamma { gamma } =>
            {
                write!(
                    formatter,
                    "bandwidth gamma {gamma} is not strictly positive"
                )
            },
            Self::NonPositiveRidge { ridge } =>
            {
                write!(formatter, "ridge {ridge} is not strictly positive")
            },
            Self::SolverFailure { detail } =>
            {
                write!(formatter, "kernel solve failed: {detail}")
            },
        }
    }
}

impl std::error::Error for KernelRidgeError {}

/// Squared Euclidean distance between two equal-length rows.
fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

impl KernelRidgeRegression {
    /// Fits RBF kernel ridge regression to `features → targets`.
    ///
    /// # Errors
    ///
    /// Returns a [`KernelRidgeError`] when the dataset is empty, the rows are
    /// ragged, the lengths disagree, any value is non-finite, `γ`/`λ` are not
    /// strictly positive, or the regularized kernel fails to factorize.
    ///
    /// # Example
    ///
    /// ```
    /// use scirust_learning::{KernelRidgeConfig, KernelRidgeRegression};
    /// // A nonlinear (radial) target a linear model cannot fit.
    /// let x = [vec![-2.0], vec![-1.0], vec![0.0], vec![1.0], vec![2.0]];
    /// let y = [4.0, 1.0, 0.0, 1.0, 4.0];
    /// let model =
    ///     KernelRidgeRegression::fit(&x, &y, KernelRidgeConfig { gamma: 0.5, ridge: 1e-6 }).unwrap();
    /// // Near-interpolation at a training point with a tiny ridge.
    /// assert!((model.predict(&[2.0]) - 4.0).abs() < 0.1);
    /// ```
    pub fn fit(
        features: &[Vec<f64>],
        targets: &[f64],
        config: KernelRidgeConfig,
    ) -> Result<Self, KernelRidgeError> {
        if features.is_empty()
        {
            return Err(KernelRidgeError::EmptyDataset);
        }

        if features.len() != targets.len()
        {
            return Err(KernelRidgeError::LengthMismatch {
                features: features.len(),
                targets: targets.len(),
            });
        }

        if config.gamma <= 0.0 || !config.gamma.is_finite()
        {
            return Err(KernelRidgeError::NonPositiveGamma {
                gamma: config.gamma,
            });
        }

        if config.ridge <= 0.0 || !config.ridge.is_finite()
        {
            return Err(KernelRidgeError::NonPositiveRidge {
                ridge: config.ridge,
            });
        }

        let width = features[0].len();

        for (row, features_row) in features.iter().enumerate()
        {
            if features_row.len() != width
            {
                return Err(KernelRidgeError::InconsistentRowWidth {
                    row,
                    expected: width,
                    found: features_row.len(),
                });
            }

            for (column, &value) in features_row.iter().enumerate()
            {
                if !value.is_finite()
                {
                    return Err(KernelRidgeError::NonFiniteFeature { row, column });
                }
            }
        }

        for (index, &target) in targets.iter().enumerate()
        {
            if !target.is_finite()
            {
                return Err(KernelRidgeError::NonFiniteTarget { index });
            }
        }

        let n = features.len();
        let mut data = vec![0.0; n * n];

        // A = K + λI; the RBF kernel has unit diagonal, so A[i][i] = 1 + λ, and
        // K is symmetric so each off-diagonal pair is computed once.
        for i in 0..n
        {
            data[i * n + i] = 1.0 + config.ridge;

            for j in (i + 1)..n
            {
                let entry = (-config.gamma * squared_distance(&features[i], &features[j])).exp();
                data[i * n + j] = entry;
                data[j * n + i] = entry;
            }
        }

        let matrix = Matrix::from_row_major(n, n, data);

        let factor =
            cholesky_decompose(matrix).map_err(|error| KernelRidgeError::SolverFailure {
                detail: error.to_string(),
            })?;

        let dual_weights =
            solve_cholesky(&factor, targets).map_err(|error| KernelRidgeError::SolverFailure {
                detail: error.to_string(),
            })?;

        Ok(Self {
            support: features.to_vec(),
            dual_weights,
            gamma: config.gamma,
        })
    }

    /// Predicts the fitted response at `query` (assumed to have the training
    /// feature width).
    pub fn predict(&self, query: &[f64]) -> f64 {
        self.support
            .iter()
            .zip(&self.dual_weights)
            .map(|(support, &weight)| {
                weight * (-self.gamma * squared_distance(query, support)).exp()
            })
            .sum()
    }

    /// Predicts the fitted response for every row in `queries`.
    pub fn predict_slice(&self, queries: &[Vec<f64>]) -> Vec<f64> {
        queries.iter().map(|query| self.predict(query)).collect()
    }

    /// The number of support (training) points.
    pub fn support_len(&self) -> usize {
        self.support.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    fn rmse(predictions: &[f64], truth: &[f64]) -> f64 {
        let sum: f64 = predictions
            .iter()
            .zip(truth)
            .map(|(p, t)| (p - t).powi(2))
            .sum();
        (sum / predictions.len() as f64).sqrt()
    }

    #[test]
    fn near_interpolates_training_points_with_a_tiny_ridge() {
        let x = [vec![-2.0], vec![-1.0], vec![0.0], vec![1.0], vec![2.0]];
        let y = [4.0, 1.0, 0.0, 1.0, 4.0]; // radial, not linear
        let model = KernelRidgeRegression::fit(
            &x,
            &y,
            KernelRidgeConfig {
                gamma: 0.5,
                ridge: 1e-8,
            },
        )
        .unwrap();

        for (row, &target) in x.iter().zip(&y)
        {
            assert!(approx_eq(model.predict(row), target, 0.02), "at {row:?}");
        }
    }

    #[test]
    fn fits_a_nonlinear_interaction_a_linear_model_cannot() {
        // y = x0 * x1 (pure interaction, zero linear/marginal signal).
        let mut x = Vec::new();
        let mut y = Vec::new();
        for a in [-2.0, -1.0, 0.0, 1.0, 2.0]
        {
            for b in [-2.0, -1.0, 0.0, 1.0, 2.0]
            {
                x.push(vec![a, b]);
                y.push(a * b);
            }
        }

        let model = KernelRidgeRegression::fit(
            &x,
            &y,
            KernelRidgeConfig {
                gamma: 0.25,
                ridge: 1e-4,
            },
        )
        .unwrap();
        let predictions = model.predict_slice(&x);
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        let baseline = rmse(&vec![mean; y.len()], &y);

        // The kernel fit must explain the interaction far better than the mean.
        assert!(
            rmse(&predictions, &y) < 0.1 * baseline,
            "kernel ridge failed to capture the interaction"
        );
    }

    #[test]
    fn predictions_are_deterministic() {
        let x = [
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
            vec![0.0, 0.0],
        ];
        let y = [1.0, 1.0, 2.0, 0.0];
        let config = KernelRidgeConfig {
            gamma: 0.3,
            ridge: 0.01,
        };

        let a = KernelRidgeRegression::fit(&x, &y, config).unwrap();
        let b = KernelRidgeRegression::fit(&x, &y, config).unwrap();
        assert_eq!(a, b);
        assert_eq!(
            a.predict(&[0.5, 0.5]).to_bits(),
            b.predict(&[0.5, 0.5]).to_bits()
        );
    }

    #[test]
    fn larger_ridge_shrinks_toward_a_smoother_fit() {
        let x = [vec![-2.0], vec![-1.0], vec![0.0], vec![1.0], vec![2.0]];
        let y = [4.0, 1.0, 0.0, 1.0, 4.0];

        let sharp = KernelRidgeRegression::fit(
            &x,
            &y,
            KernelRidgeConfig {
                gamma: 0.5,
                ridge: 1e-6,
            },
        )
        .unwrap();
        let smooth = KernelRidgeRegression::fit(
            &x,
            &y,
            KernelRidgeConfig {
                gamma: 0.5,
                ridge: 5.0,
            },
        )
        .unwrap();

        // Heavy ridge pulls the peak prediction down toward the mean.
        assert!(smooth.predict(&[2.0]) < sharp.predict(&[2.0]));
    }

    #[test]
    fn rejects_malformed_input() {
        assert_eq!(
            KernelRidgeRegression::fit(
                &[],
                &[],
                KernelRidgeConfig {
                    gamma: 1.0,
                    ridge: 1.0
                }
            ),
            Err(KernelRidgeError::EmptyDataset)
        );
        assert_eq!(
            KernelRidgeRegression::fit(
                &[vec![1.0]],
                &[1.0, 2.0],
                KernelRidgeConfig {
                    gamma: 1.0,
                    ridge: 1.0
                }
            ),
            Err(KernelRidgeError::LengthMismatch {
                features: 1,
                targets: 2,
            })
        );
        assert!(matches!(
            KernelRidgeRegression::fit(
                &[vec![1.0], vec![1.0, 2.0]],
                &[1.0, 2.0],
                KernelRidgeConfig {
                    gamma: 1.0,
                    ridge: 1.0
                }
            ),
            Err(KernelRidgeError::InconsistentRowWidth { row: 1, .. })
        ));
        assert!(matches!(
            KernelRidgeRegression::fit(
                &[vec![1.0], vec![2.0]],
                &[1.0, 2.0],
                KernelRidgeConfig {
                    gamma: 0.0,
                    ridge: 1.0
                }
            ),
            Err(KernelRidgeError::NonPositiveGamma { .. })
        ));
        assert!(matches!(
            KernelRidgeRegression::fit(
                &[vec![1.0], vec![2.0]],
                &[1.0, 2.0],
                KernelRidgeConfig {
                    gamma: 1.0,
                    ridge: -1.0
                }
            ),
            Err(KernelRidgeError::NonPositiveRidge { .. })
        ));
        assert!(matches!(
            KernelRidgeRegression::fit(
                &[vec![1.0], vec![f64::NAN]],
                &[1.0, 2.0],
                KernelRidgeConfig {
                    gamma: 1.0,
                    ridge: 1.0
                }
            ),
            Err(KernelRidgeError::NonFiniteFeature { row: 1, column: 0 })
        ));
    }
}
