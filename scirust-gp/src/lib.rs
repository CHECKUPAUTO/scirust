//! `scirust-gp` — Gaussian process (GP) regression in pure, dependency-free Rust.
//!
//! This crate implements exact Gaussian process regression with a small family
//! of stationary kernels (squared-exponential / RBF and Matérn-3/2, -5/2). All
//! linear algebra — dense Cholesky factorization and triangular solves — is
//! implemented internally, so the crate has zero dependencies, uses no `unsafe`,
//! and is fully deterministic (no randomness).
//!
//! Given training inputs `x`, targets `y`, a [`Kernel`], and a noise variance,
//! [`GaussianProcess::fit`] forms the covariance matrix `K = k(x_i, x_j) +
//! noise·I`, factors it as `K = L Lᵀ` (lower-triangular Cholesky), and solves
//! `K α = y`. Prediction at a test point `x*` returns the posterior mean
//! `k*ᵀ α` and variance `k(x*, x*) − vᵀv` where `L v = k*`.
//!
//! # Example
//!
//! ```
//! use scirust_gp::{GaussianProcess, Rbf};
//!
//! // A few noiseless observations of a smooth function.
//! let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
//! let y = vec![0.0, 1.0, 0.0, -1.0];
//! let kernel = Rbf { lengthscale: 1.0, variance: 1.0 };
//!
//! let gp = GaussianProcess::fit(&x, &y, kernel, 1e-10).unwrap();
//!
//! // The posterior interpolates the training data at the observed inputs.
//! let (mean, var) = gp.predict(&[1.0]);
//! assert!((mean - 1.0).abs() < 1e-4);
//! assert!(var >= 0.0 && var < 1e-3);
//!
//! // The log marginal likelihood is a finite score of the model fit.
//! assert!(gp.log_marginal_likelihood().is_finite());
//! ```
#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::fmt;

/// Errors returned by fallible Gaussian process operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GpError {
    /// The training data was empty (no inputs or no targets).
    EmptyData,
    /// The number of input rows did not match the number of targets, or the
    /// input rows did not all share a common (non-zero) feature dimension.
    ShapeMismatch {
        /// Number of input rows supplied.
        inputs: usize,
        /// Number of targets supplied.
        targets: usize,
    },
    /// The covariance matrix `K = k(x_i, x_j) + noise·I` was not positive
    /// definite, so its Cholesky factorization failed. This typically means the
    /// noise variance is too small (e.g. zero) for the given data — for
    /// instance duplicate inputs with no observation noise.
    NotPositiveDefinite,
}

impl fmt::Display for GpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self
        {
            GpError::EmptyData => write!(f, "training data is empty"),
            GpError::ShapeMismatch { inputs, targets } => write!(
                f,
                "shape mismatch: {inputs} input rows but {targets} targets (or inconsistent input dimensions)"
            ),
            GpError::NotPositiveDefinite => write!(
                f,
                "covariance matrix is not positive definite (Cholesky failed); increase the noise variance"
            ),
        }
    }
}

impl std::error::Error for GpError {}

/// A positive-definite covariance function over pairs of feature vectors.
///
/// Implementations compute `k(a, b)` for two feature vectors of equal
/// dimension. The kernels provided by this crate are stationary: they depend
/// only on the Euclidean distance `r = ‖a − b‖`.
pub trait Kernel {
    /// Evaluate the kernel on two feature vectors `a` and `b` of equal length.
    fn eval(&self, a: &[f64], b: &[f64]) -> f64;
}

/// Squared distance `‖a − b‖²` between two equal-length feature vectors.
///
/// If the slices differ in length only the overlapping prefix contributes;
/// callers are expected to pass vectors of matching dimension.
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let d = ai - bi;
            d * d
        })
        .sum()
}

/// Squared-exponential (RBF / Gaussian) kernel.
///
/// `k(a, b) = variance · exp(−‖a − b‖² / (2 · lengthscale²))`.
///
/// This is the infinitely differentiable, very smooth kernel; `lengthscale`
/// controls how quickly correlations decay with distance and `variance` is the
/// prior (marginal) variance `k(a, a)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rbf {
    /// Characteristic length scale `ℓ > 0`; larger values give smoother, more
    /// slowly varying functions.
    pub lengthscale: f64,
    /// Signal variance `σ² > 0`, equal to the prior variance `k(a, a)`.
    pub variance: f64,
}

impl Kernel for Rbf {
    fn eval(&self, a: &[f64], b: &[f64]) -> f64 {
        let r2 = sq_dist(a, b);
        self.variance * (-r2 / (2.0 * self.lengthscale * self.lengthscale)).exp()
    }
}

/// Matérn-3/2 kernel.
///
/// With `r = ‖a − b‖` and `d = √3 · r / lengthscale`,
/// `k(a, b) = variance · (1 + d) · exp(−d)`.
///
/// Sample paths are once-differentiable — rougher than the [`Rbf`] kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matern32 {
    /// Characteristic length scale `ℓ > 0`.
    pub lengthscale: f64,
    /// Signal variance `σ² > 0`, equal to the prior variance `k(a, a)`.
    pub variance: f64,
}

impl Kernel for Matern32 {
    fn eval(&self, a: &[f64], b: &[f64]) -> f64 {
        let r = sq_dist(a, b).sqrt();
        let d = 3.0_f64.sqrt() * r / self.lengthscale;
        self.variance * (1.0 + d) * (-d).exp()
    }
}

/// Matérn-5/2 kernel.
///
/// With `r = ‖a − b‖` and `d = √5 · r / lengthscale`,
/// `k(a, b) = variance · (1 + d + d²/3) · exp(−d)`.
///
/// Sample paths are twice-differentiable — a common default that sits between
/// the roughness of [`Matern32`] and the smoothness of [`Rbf`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matern52 {
    /// Characteristic length scale `ℓ > 0`.
    pub lengthscale: f64,
    /// Signal variance `σ² > 0`, equal to the prior variance `k(a, a)`.
    pub variance: f64,
}

impl Kernel for Matern52 {
    fn eval(&self, a: &[f64], b: &[f64]) -> f64 {
        let r = sq_dist(a, b).sqrt();
        let d = 5.0_f64.sqrt() * r / self.lengthscale;
        self.variance * (1.0 + d + d * d / 3.0) * (-d).exp()
    }
}

/// Dense lower-triangular Cholesky factorization of a symmetric matrix.
///
/// Returns `L` such that `A = L Lᵀ`, or `None` if `A` is not positive definite
/// (a non-positive or non-finite pivot is encountered). Only the lower triangle
/// of `A` is read.
fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n
    {
        for j in 0..=i
        {
            let dot: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            let sum = a[i][j] - dot;
            if i == j
            {
                if sum <= 0.0 || sum.is_nan()
                {
                    // Non-positive pivot (or NaN) ⇒ not positive definite.
                    return None;
                }
                l[i][i] = sum.sqrt();
            }
            else
            {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Some(l)
}

/// Solve `L z = b` for a lower-triangular `L` by forward substitution.
fn forward_substitution(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut z = vec![0.0; n];
    for i in 0..n
    {
        let mut sum = b[i];
        for k in 0..i
        {
            sum -= l[i][k] * z[k];
        }
        z[i] = sum / l[i][i];
    }
    z
}

/// Solve `Lᵀ x = z` for a lower-triangular `L` by back substitution.
///
/// `Lᵀ` is upper-triangular with `(Lᵀ)[i][k] = L[k][i]`.
fn back_substitution(l: &[Vec<f64>], z: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev()
    {
        let mut sum = z[i];
        for k in (i + 1)..n
        {
            sum -= l[k][i] * x[k];
        }
        x[i] = sum / l[i][i];
    }
    x
}

/// A fitted Gaussian process regressor.
///
/// Construct one with [`GaussianProcess::fit`], then query the posterior with
/// [`GaussianProcess::predict`] / [`GaussianProcess::predict_many`] and score
/// the fit with [`GaussianProcess::log_marginal_likelihood`].
#[derive(Debug, Clone)]
pub struct GaussianProcess<K: Kernel> {
    /// Training inputs (each a feature vector of common dimension).
    x: Vec<Vec<f64>>,
    /// Lower-triangular Cholesky factor of `K = k(x_i, x_j) + noise·I`.
    l: Vec<Vec<f64>>,
    /// Solution of `K α = y`.
    alpha: Vec<f64>,
    /// The kernel used to build `K` and to form test covariances.
    kernel: K,
    /// Precomputed log marginal likelihood of the training data.
    lml: f64,
}

impl<K: Kernel> GaussianProcess<K> {
    /// Fit a Gaussian process to training inputs `x` and targets `y`.
    ///
    /// Builds the covariance matrix `K` with entries `k(x_i, x_j)` plus
    /// `noise_variance` on the diagonal, computes its Cholesky factor `L`, and
    /// solves `K α = y`.
    ///
    /// # Errors
    ///
    /// * [`GpError::EmptyData`] if `x` or `y` is empty.
    /// * [`GpError::ShapeMismatch`] if `x.len() != y.len()` or the rows of `x`
    ///   do not all share the same (non-zero) feature dimension.
    /// * [`GpError::NotPositiveDefinite`] if `K` is not positive definite (its
    ///   Cholesky factorization fails), e.g. with zero noise and duplicate
    ///   inputs.
    pub fn fit(x: &[Vec<f64>], y: &[f64], kernel: K, noise_variance: f64) -> Result<Self, GpError> {
        if x.is_empty() || y.is_empty()
        {
            return Err(GpError::EmptyData);
        }
        if x.len() != y.len()
        {
            return Err(GpError::ShapeMismatch {
                inputs: x.len(),
                targets: y.len(),
            });
        }
        let dim = x[0].len();
        if dim == 0 || x.iter().any(|row| row.len() != dim)
        {
            return Err(GpError::ShapeMismatch {
                inputs: x.len(),
                targets: y.len(),
            });
        }

        let n = x.len();
        let mut kmat = vec![vec![0.0; n]; n];
        for (i, row) in kmat.iter_mut().enumerate()
        {
            for (j, cell) in row.iter_mut().enumerate()
            {
                let mut v = kernel.eval(&x[i], &x[j]);
                if i == j
                {
                    v += noise_variance;
                }
                *cell = v;
            }
        }

        let l = cholesky(&kmat).ok_or(GpError::NotPositiveDefinite)?;

        // Solve K α = y via K = L Lᵀ: forward solve L z = y, back solve Lᵀ α = z.
        let z = forward_substitution(&l, y);
        let alpha = back_substitution(&l, &z);

        // Log marginal likelihood: −½ yᵀα − Σ ln L_ii − (n/2) ln(2π).
        let y_alpha: f64 = y.iter().zip(alpha.iter()).map(|(&yi, &ai)| yi * ai).sum();
        let log_diag: f64 = (0..n).map(|i| l[i][i].ln()).sum();
        let lml = -0.5 * y_alpha - log_diag - (n as f64) / 2.0 * (2.0 * std::f64::consts::PI).ln();

        Ok(GaussianProcess {
            x: x.to_vec(),
            l,
            alpha,
            kernel,
            lml,
        })
    }

    /// Covariance vector `k* = [k(x_i, x*)]_i` between the training inputs and a
    /// test point.
    fn cross_covariance(&self, x_star: &[f64]) -> Vec<f64> {
        self.x
            .iter()
            .map(|xi| self.kernel.eval(xi, x_star))
            .collect()
    }

    /// Posterior mean and variance at a single test point `x_star`.
    ///
    /// The mean is `k*ᵀ α`. The variance is `k(x*, x*) − vᵀv` where `L v = k*`;
    /// it is clamped to be non-negative to guard against tiny negative values
    /// from floating-point round-off.
    pub fn predict(&self, x_star: &[f64]) -> (f64, f64) {
        let k_star = self.cross_covariance(x_star);
        let mean: f64 = k_star
            .iter()
            .zip(self.alpha.iter())
            .map(|(&ki, &ai)| ki * ai)
            .sum();

        let v = forward_substitution(&self.l, &k_star);
        let vtv: f64 = v.iter().map(|&vi| vi * vi).sum();
        let prior = self.kernel.eval(x_star, x_star);
        let var = (prior - vtv).max(0.0);

        (mean, var)
    }

    /// Posterior mean and variance at many test points.
    ///
    /// Equivalent to calling [`GaussianProcess::predict`] on each row of `xs`.
    pub fn predict_many(&self, xs: &[Vec<f64>]) -> Vec<(f64, f64)> {
        xs.iter().map(|x| self.predict(x)).collect()
    }

    /// Log marginal likelihood of the training data under the fitted model.
    ///
    /// `log p(y | X) = −½ yᵀα − Σ_i ln L_ii − (n/2) ln(2π)`, where `α = K⁻¹ y`
    /// and `L` is the Cholesky factor of `K`. Larger values indicate a better
    /// fit; this is the standard objective for kernel hyperparameter selection.
    pub fn log_marginal_likelihood(&self) -> f64 {
        self.lml
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dense reference solver used to cross-check the GP internals: solves
    /// `A x = b` for a small matrix `A` via Gauss-Jordan elimination with
    /// partial pivoting.
    fn dense_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = a.len();
        let mut m = a.to_vec();
        let mut rhs = b.to_vec();
        for col in 0..n
        {
            // Partial pivot.
            let mut piv = col;
            for r in (col + 1)..n
            {
                if m[r][col].abs() > m[piv][col].abs()
                {
                    piv = r;
                }
            }
            m.swap(col, piv);
            rhs.swap(col, piv);
            let d = m[col][col];
            assert!(d.abs() > 1e-300, "singular matrix in dense_solve");
            for v in m[col].iter_mut()
            {
                *v /= d;
            }
            rhs[col] /= d;
            let pivot_row = m[col].clone();
            let pivot_rhs = rhs[col];
            for r in 0..n
            {
                if r != col
                {
                    let f = m[r][col];
                    for (v, &p) in m[r].iter_mut().zip(pivot_row.iter())
                    {
                        *v -= f * p;
                    }
                    rhs[r] -= f * pivot_rhs;
                }
            }
        }
        rhs
    }

    fn build_k<Ker: Kernel>(x: &[Vec<f64>], kernel: &Ker, noise: f64) -> Vec<Vec<f64>> {
        let n = x.len();
        let mut k = vec![vec![0.0; n]; n];
        for (i, row) in k.iter_mut().enumerate()
        {
            for (j, cell) in row.iter_mut().enumerate()
            {
                *cell = kernel.eval(&x[i], &x[j]) + if i == j { noise } else { 0.0 };
            }
        }
        k
    }

    #[test]
    fn cholesky_reconstructs_matrix() {
        let a = vec![
            vec![4.0, 2.0, 2.0],
            vec![2.0, 5.0, 3.0],
            vec![2.0, 3.0, 6.0],
        ];
        let l = cholesky(&a).expect("PD matrix");
        // Reconstruct A = L Lᵀ.
        for i in 0..3
        {
            for j in 0..3
            {
                let s: f64 = (0..3).map(|k| l[i][k] * l[j][k]).sum();
                assert!((s - a[i][j]).abs() < 1e-12, "entry ({i},{j})");
            }
        }
    }

    #[test]
    fn cholesky_rejects_non_pd() {
        // Indefinite matrix.
        let a = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert!(cholesky(&a).is_none());
    }

    #[test]
    fn interpolation_at_training_points() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let y = vec![0.5, -0.3, 1.2, 0.7, -1.1];
        let kernel = Rbf {
            lengthscale: 1.3,
            variance: 1.0,
        };
        let gp = GaussianProcess::fit(&x, &y, kernel, 1e-10).unwrap();
        for (xi, &yi) in x.iter().zip(y.iter())
        {
            let (mean, var) = gp.predict(xi);
            assert!((mean - yi).abs() < 1e-4, "mean {mean} vs target {yi}");
            assert!(
                (0.0..1e-3).contains(&var),
                "variance at training point: {var}"
            );
        }
    }

    #[test]
    fn variance_rises_away_from_data() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y = vec![0.0, 1.0, 0.0];
        let variance = 1.0;
        let kernel = Rbf {
            lengthscale: 0.5,
            variance,
        };
        let gp = GaussianProcess::fit(&x, &y, kernel, 1e-8).unwrap();

        let (_, var_near) = gp.predict(&[1.0]);
        let (_, var_far) = gp.predict(&[20.0]);
        assert!(var_near < 1e-3);
        // Far from the data the predictive variance approaches the prior variance.
        assert!(var_far > var_near);
        assert!((var_far - variance).abs() < 1e-6, "var_far = {var_far}");
    }

    #[test]
    fn two_point_hand_computable() {
        // Exact oracle: build K densely, solve K α = y, and form the predictive
        // mean/variance directly; compare to the GP's Cholesky-based answer.
        let x = vec![vec![0.0], vec![1.0]];
        let y = vec![1.0, 2.0];
        let noise = 0.1;
        let kernel = Rbf {
            lengthscale: 1.0,
            variance: 1.0,
        };
        let gp = GaussianProcess::fit(&x, &y, kernel, noise).unwrap();

        let k = build_k(&x, &kernel, noise);
        let alpha = dense_solve(&k, &y);

        let x_star = [0.4_f64];
        let k_star: Vec<f64> = x.iter().map(|xi| kernel.eval(xi, &x_star)).collect();
        let mean_oracle: f64 = k_star.iter().zip(alpha.iter()).map(|(a, b)| a * b).sum();

        // var = k** − k*ᵀ K⁻¹ k*.
        let kinv_kstar = dense_solve(&k, &k_star);
        let quad: f64 = k_star
            .iter()
            .zip(kinv_kstar.iter())
            .map(|(a, b)| a * b)
            .sum();
        let var_oracle = kernel.eval(&x_star, &x_star) - quad;

        let (mean, var) = gp.predict(&x_star);
        assert!(
            (mean - mean_oracle).abs() < 1e-9,
            "mean {mean} vs {mean_oracle}"
        );
        assert!((var - var_oracle).abs() < 1e-9, "var {var} vs {var_oracle}");
    }

    #[test]
    fn lml_matches_dense_oracle() {
        let x = vec![vec![0.0], vec![1.0], vec![2.5]];
        let y = vec![0.2, -0.4, 1.1];
        let noise = 0.05;
        let kernel = Matern52 {
            lengthscale: 1.0,
            variance: 1.0,
        };
        let gp = GaussianProcess::fit(&x, &y, kernel, noise).unwrap();

        // Oracle LML with an independent α from the dense Gauss-Jordan solver.
        let k = build_k(&x, &kernel, noise);
        let alpha = dense_solve(&k, &y);
        let y_alpha: f64 = y.iter().zip(alpha.iter()).map(|(a, b)| a * b).sum();
        let l = cholesky(&k).unwrap();
        let log_diag: f64 = (0..3).map(|i| l[i][i].ln()).sum();
        let n = 3.0;
        let lml_oracle = -0.5 * y_alpha - log_diag - n / 2.0 * (2.0 * std::f64::consts::PI).ln();

        assert!(gp.log_marginal_likelihood().is_finite());
        assert!(
            (gp.log_marginal_likelihood() - lml_oracle).abs() < 1e-9,
            "{} vs {}",
            gp.log_marginal_likelihood(),
            lml_oracle
        );
    }

    #[test]
    fn lml_prefers_matching_lengthscale() {
        // Smooth data: y = sin(x). A length scale that matches the smoothness
        // should score higher than a very short one.
        let x: Vec<Vec<f64>> = (0..12).map(|i| vec![i as f64 * 0.5]).collect();
        let y: Vec<f64> = x.iter().map(|xi| xi[0].sin()).collect();

        let good = GaussianProcess::fit(
            &x,
            &y,
            Rbf {
                lengthscale: 1.5,
                variance: 1.0,
            },
            1e-6,
        )
        .unwrap();
        let bad = GaussianProcess::fit(
            &x,
            &y,
            Rbf {
                lengthscale: 0.05,
                variance: 1.0,
            },
            1e-6,
        )
        .unwrap();

        assert!(good.log_marginal_likelihood().is_finite());
        assert!(bad.log_marginal_likelihood().is_finite());
        assert!(
            good.log_marginal_likelihood() > bad.log_marginal_likelihood(),
            "good {} should exceed bad {}",
            good.log_marginal_likelihood(),
            bad.log_marginal_likelihood()
        );
    }

    #[test]
    fn multidimensional_inputs() {
        let x = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        ];
        let y = vec![0.0, 1.0, 1.0, 2.0];
        let kernel = Matern32 {
            lengthscale: 1.0,
            variance: 1.0,
        };
        let gp = GaussianProcess::fit(&x, &y, kernel, 1e-10).unwrap();

        for (xi, &yi) in x.iter().zip(y.iter())
        {
            let (mean, _) = gp.predict(xi);
            assert!(
                (mean - yi).abs() < 1e-4,
                "d=2 interpolation: {mean} vs {yi}"
            );
        }

        let preds = gp.predict_many(&x);
        assert_eq!(preds.len(), x.len());
        for (i, &(_, var)) in preds.iter().enumerate()
        {
            assert!(var >= 0.0, "variance {i} clamped");
        }
    }

    #[test]
    fn all_kernels_have_correct_prior_variance() {
        // k(a, a) must equal the signal variance for every kernel.
        let a = [1.0, 2.0, 3.0];
        let rbf = Rbf {
            lengthscale: 0.7,
            variance: 2.5,
        };
        let m32 = Matern32 {
            lengthscale: 0.7,
            variance: 2.5,
        };
        let m52 = Matern52 {
            lengthscale: 0.7,
            variance: 2.5,
        };
        assert!((rbf.eval(&a, &a) - 2.5).abs() < 1e-12);
        assert!((m32.eval(&a, &a) - 2.5).abs() < 1e-12);
        assert!((m52.eval(&a, &a) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn matern_formulas_at_unit_distance() {
        // Closed-form checks at r = 1, ℓ = 1.
        let a = [0.0];
        let b = [1.0];
        let m32 = Matern32 {
            lengthscale: 1.0,
            variance: 1.0,
        };
        let m52 = Matern52 {
            lengthscale: 1.0,
            variance: 1.0,
        };
        let s3 = 3.0_f64.sqrt();
        let s5 = 5.0_f64.sqrt();
        let exp_m32 = (1.0 + s3) * (-s3).exp();
        let exp_m52 = (1.0 + s5 + 5.0 / 3.0) * (-s5).exp();
        assert!((m32.eval(&a, &b) - exp_m32).abs() < 1e-12);
        assert!((m52.eval(&a, &b) - exp_m52).abs() < 1e-12);
    }

    #[test]
    fn err_shape_mismatch() {
        let x = vec![vec![0.0], vec![1.0]];
        let y = vec![1.0];
        let kernel = Rbf {
            lengthscale: 1.0,
            variance: 1.0,
        };
        let e = GaussianProcess::fit(&x, &y, kernel, 0.1).unwrap_err();
        assert_eq!(
            e,
            GpError::ShapeMismatch {
                inputs: 2,
                targets: 1
            }
        );
    }

    #[test]
    fn err_empty_data() {
        let x: Vec<Vec<f64>> = vec![];
        let y: Vec<f64> = vec![];
        let kernel = Rbf {
            lengthscale: 1.0,
            variance: 1.0,
        };
        assert_eq!(
            GaussianProcess::fit(&x, &y, kernel, 0.1).unwrap_err(),
            GpError::EmptyData
        );
    }

    #[test]
    fn err_non_pd_duplicate_points_zero_noise() {
        // Duplicate inputs with zero noise make K singular / non-PD.
        let x = vec![vec![0.0], vec![0.0]];
        let y = vec![1.0, 1.0];
        let kernel = Rbf {
            lengthscale: 1.0,
            variance: 1.0,
        };
        assert_eq!(
            GaussianProcess::fit(&x, &y, kernel, 0.0).unwrap_err(),
            GpError::NotPositiveDefinite
        );
    }

    #[test]
    fn err_inconsistent_dimensions() {
        let x = vec![vec![0.0, 1.0], vec![1.0]];
        let y = vec![1.0, 2.0];
        let kernel = Rbf {
            lengthscale: 1.0,
            variance: 1.0,
        };
        assert!(matches!(
            GaussianProcess::fit(&x, &y, kernel, 0.1).unwrap_err(),
            GpError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn error_display_is_nonempty() {
        let errs = [
            GpError::EmptyData,
            GpError::ShapeMismatch {
                inputs: 2,
                targets: 3,
            },
            GpError::NotPositiveDefinite,
        ];
        for e in &errs
        {
            assert!(!format!("{e}").is_empty());
        }
    }
}
