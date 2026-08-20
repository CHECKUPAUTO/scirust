//! Numerical analysis of discrete-time quadratic Lyapunov certificates.
//!
//! For the autonomous linear system `x[k+1] = A x[k]`, a symmetric positive-
//! definite matrix `P` satisfying
//!
//! ```text
//! Aᵀ P A - P = -Q
//! ```
//!
//! with symmetric positive-definite `Q` is a classical certificate of
//! asymptotic stability in exact arithmetic. This module provides two small,
//! deterministic numerical tools around that relation:
//!
//! - [`verify_discrete_lyapunov`] checks dimensions, finiteness, symmetry,
//!   positive definiteness, and the Lyapunov residual of caller-supplied
//!   `A`, `P`, and `Q`;
//! - [`solve_discrete_lyapunov`] constructs `P` by the convergent fixed-point
//!   series `P_{k+1} = Q + Aᵀ P_k A` and then applies the same verifier.
//!
//! The returned [`DiscreteLyapunovCertificate`] is a **numerical certificate
//! under the supplied floating-point tolerance**, not a formal proof. In
//! particular, callers must not silently promote it into a hard safety claim
//! for a nonlinear, uncertain, time-varying, or otherwise different system.

use scirust_estimation::Mat;
use std::fmt;

/// Absolute/relative tolerance used for symmetry and residual checks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LyapunovTolerance {
    /// Absolute tolerance floor. Must be finite and non-negative.
    pub abs: f64,
    /// Relative tolerance multiplier. Must be finite and non-negative.
    pub rel: f64,
}

impl LyapunovTolerance {
    /// Construct a tolerance pair.
    pub const fn new(abs: f64, rel: f64) -> Self {
        Self { abs, rel }
    }

    fn validate(self) -> Result<Self, LyapunovError> {
        if !self.abs.is_finite() || !self.rel.is_finite() || self.abs < 0.0 || self.rel < 0.0
        {
            return Err(LyapunovError::InvalidTolerance);
        }
        Ok(self)
    }

    fn bound(self, scale: f64) -> f64 {
        self.abs + self.rel * scale
    }
}

impl Default for LyapunovTolerance {
    fn default() -> Self {
        Self {
            abs: 1e-10,
            rel: 1e-10,
        }
    }
}

/// Options for the fixed-point discrete Lyapunov solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DiscreteLyapunovOptions {
    /// Numerical tolerance used for convergence and final verification.
    pub tolerance: LyapunovTolerance,
    /// Maximum fixed-point iterations before returning `DidNotConverge`.
    pub max_iterations: usize,
}

impl Default for DiscreteLyapunovOptions {
    fn default() -> Self {
        Self {
            tolerance: LyapunovTolerance::default(),
            max_iterations: 10_000,
        }
    }
}

/// Verified numerical Lyapunov relation for one discrete-time linear system.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteLyapunovCertificate {
    /// Positive-definite Lyapunov matrix used by the certificate.
    pub p: Mat,
    /// Maximum absolute element of `Aᵀ P A - P + Q`.
    pub residual_max_abs: f64,
    /// Absolute/relative bound against which the residual was accepted.
    pub residual_bound: f64,
}

/// Result of the fixed-point Lyapunov solver.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteLyapunovSolution {
    /// Numerically verified Lyapunov certificate.
    pub certificate: DiscreteLyapunovCertificate,
    /// Number of fixed-point updates performed.
    pub iterations: usize,
}

/// Failures produced by discrete Lyapunov analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum LyapunovError {
    /// A matrix must contain at least one state dimension.
    EmptyMatrix,
    /// A named matrix is not square.
    NonSquare {
        /// Matrix name (`A`, `P`, or `Q`).
        matrix: &'static str,
        /// Row count.
        rows: usize,
        /// Column count.
        cols: usize,
    },
    /// `A`, `P`, and/or `Q` do not share the same square dimension.
    DimensionMismatch,
    /// A named input contains NaN or infinity.
    NonFiniteInput {
        /// Matrix name (`A`, `P`, or `Q`).
        matrix: &'static str,
    },
    /// Absolute/relative tolerance is negative or non-finite.
    InvalidTolerance,
    /// The solver iteration budget is zero.
    ZeroIterationBudget,
    /// A matrix expected to be symmetric exceeds the requested tolerance.
    NotSymmetric {
        /// Matrix name (`P` or `Q`).
        matrix: &'static str,
        /// Maximum absolute antisymmetric element difference.
        max_asymmetry: f64,
        /// Allowed asymmetry bound.
        bound: f64,
    },
    /// A matrix expected to be positive definite failed Cholesky factorization.
    NotPositiveDefinite {
        /// Matrix name (`P` or `Q`).
        matrix: &'static str,
    },
    /// The Lyapunov residual exceeds the requested tolerance.
    ResidualTooLarge {
        /// Maximum absolute residual element.
        residual: f64,
        /// Allowed residual bound.
        bound: f64,
    },
    /// An intermediate fixed-point computation became NaN or infinite.
    NonFiniteComputation,
    /// Fixed-point iteration exhausted its budget without a verifiable residual.
    DidNotConverge {
        /// Number of attempted updates.
        iterations: usize,
        /// Last finite maximum absolute residual observed.
        residual: f64,
    },
}

impl fmt::Display for LyapunovError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Lyapunov analysis requires a non-empty state matrix"),
            Self::NonSquare { matrix, rows, cols } => {
                write!(f, "{matrix} must be square, got {rows}x{cols}")
            }
            Self::DimensionMismatch => write!(f, "A, P, and Q dimensions must match"),
            Self::NonFiniteInput { matrix } => {
                write!(f, "{matrix} contains a non-finite value")
            }
            Self::InvalidTolerance => {
                write!(f, "Lyapunov tolerance must be finite and non-negative")
            }
            Self::ZeroIterationBudget => write!(f, "Lyapunov solver iteration budget is zero"),
            Self::NotSymmetric {
                matrix,
                max_asymmetry,
                bound,
            } => write!(
                f,
                "{matrix} is not symmetric within tolerance: asymmetry {max_asymmetry} > {bound}"
            ),
            Self::NotPositiveDefinite { matrix } => {
                write!(f, "{matrix} is not positive definite")
            }
            Self::ResidualTooLarge { residual, bound } => write!(
                f,
                "discrete Lyapunov residual {residual} exceeds tolerance {bound}"
            ),
            Self::NonFiniteComputation => {
                write!(f, "discrete Lyapunov iteration produced a non-finite value")
            }
            Self::DidNotConverge {
                iterations,
                residual,
            } => write!(
                f,
                "discrete Lyapunov iteration did not converge in {iterations} updates (residual {residual})"
            ),
        }
    }
}

impl std::error::Error for LyapunovError {}

fn validate_square(name: &'static str, matrix: &Mat) -> Result<(), LyapunovError> {
    if matrix.rows == 0 || matrix.cols == 0
    {
        return Err(LyapunovError::EmptyMatrix);
    }
    if matrix.rows != matrix.cols
    {
        return Err(LyapunovError::NonSquare {
            matrix: name,
            rows: matrix.rows,
            cols: matrix.cols,
        });
    }
    if matrix.data.iter().any(|value| !value.is_finite())
    {
        return Err(LyapunovError::NonFiniteInput { matrix: name });
    }
    Ok(())
}

fn max_abs(matrix: &Mat) -> f64 {
    matrix
        .data
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()))
}

fn max_asymmetry(matrix: &Mat) -> f64 {
    let mut maximum = 0.0_f64;
    for row in 0..matrix.rows
    {
        for col in (row + 1)..matrix.cols
        {
            maximum = maximum.max((matrix.get(row, col) - matrix.get(col, row)).abs());
        }
    }
    maximum
}

fn validate_symmetric_positive_definite(
    name: &'static str,
    matrix: &Mat,
    tolerance: LyapunovTolerance,
) -> Result<(), LyapunovError> {
    let asymmetry = max_asymmetry(matrix);
    let bound = tolerance.bound(max_abs(matrix));
    if asymmetry > bound
    {
        return Err(LyapunovError::NotSymmetric {
            matrix: name,
            max_asymmetry: asymmetry,
            bound,
        });
    }
    if matrix.cholesky().is_none()
    {
        return Err(LyapunovError::NotPositiveDefinite { matrix: name });
    }
    Ok(())
}

fn lyapunov_residual(a: &Mat, p: &Mat, q: &Mat) -> Mat {
    a.t().matmul(p).matmul(a).sub(p).add(q)
}

fn residual_bound(
    a: &Mat,
    p: &Mat,
    q: &Mat,
    tolerance: LyapunovTolerance,
) -> f64 {
    let dynamic_term = a.t().matmul(p).matmul(a);
    let scale = max_abs(&dynamic_term).max(max_abs(p)).max(max_abs(q));
    tolerance.bound(scale)
}

/// Verify a caller-supplied quadratic certificate for `x[k+1] = A x[k]`.
///
/// This checks that `P` and `Q` are symmetric positive definite and that the
/// numerical residual `Aᵀ P A - P + Q` is no larger than the requested
/// absolute/relative tolerance. In exact arithmetic, positive-definite `P,Q`
/// satisfying the equality certify asymptotic stability of the autonomous
/// discrete-time linear system.
///
/// The result is a floating-point numerical certificate only; model validity
/// and any nonlinear/uncertain-system interpretation remain the caller's
/// responsibility.
pub fn verify_discrete_lyapunov(
    a: &Mat,
    p: &Mat,
    q: &Mat,
    tolerance: LyapunovTolerance,
) -> Result<DiscreteLyapunovCertificate, LyapunovError> {
    let tolerance = tolerance.validate()?;
    validate_square("A", a)?;
    validate_square("P", p)?;
    validate_square("Q", q)?;
    if a.rows != p.rows || a.rows != q.rows
    {
        return Err(LyapunovError::DimensionMismatch);
    }
    validate_symmetric_positive_definite("P", p, tolerance)?;
    validate_symmetric_positive_definite("Q", q, tolerance)?;

    let residual = lyapunov_residual(a, p, q);
    if residual.data.iter().any(|value| !value.is_finite())
    {
        return Err(LyapunovError::NonFiniteComputation);
    }
    let residual_max_abs = max_abs(&residual);
    let bound = residual_bound(a, p, q, tolerance);
    if residual_max_abs > bound
    {
        return Err(LyapunovError::ResidualTooLarge {
            residual: residual_max_abs,
            bound,
        });
    }

    Ok(DiscreteLyapunovCertificate {
        p: p.clone(),
        residual_max_abs,
        residual_bound: bound,
    })
}

/// Solve `P = Q + Aᵀ P A` by deterministic fixed-point iteration.
///
/// For asymptotically stable `A`, the series
/// `P = Q + AᵀQA + (Aᵀ)²QA² + ...` converges for positive-definite `Q`.
/// The solver starts from `P₀ = Q`, iterates that series, and accepts a result
/// only after [`verify_discrete_lyapunov`] validates the final matrix.
///
/// An unstable or extremely slow-to-converge system normally returns
/// [`LyapunovError::DidNotConverge`]; overflow during divergence is reported as
/// [`LyapunovError::NonFiniteComputation`].
pub fn solve_discrete_lyapunov(
    a: &Mat,
    q: &Mat,
    options: DiscreteLyapunovOptions,
) -> Result<DiscreteLyapunovSolution, LyapunovError> {
    let tolerance = options.tolerance.validate()?;
    if options.max_iterations == 0
    {
        return Err(LyapunovError::ZeroIterationBudget);
    }
    validate_square("A", a)?;
    validate_square("Q", q)?;
    if a.rows != q.rows
    {
        return Err(LyapunovError::DimensionMismatch);
    }
    validate_symmetric_positive_definite("Q", q, tolerance)?;

    let at = a.t();
    let mut p = q.clone();
    let mut last_residual = f64::INFINITY;

    for iteration in 1..=options.max_iterations
    {
        let next = q.add(&at.matmul(&p).matmul(a));
        if next.data.iter().any(|value| !value.is_finite())
        {
            return Err(LyapunovError::NonFiniteComputation);
        }

        let residual = lyapunov_residual(a, &next, q);
        if residual.data.iter().any(|value| !value.is_finite())
        {
            return Err(LyapunovError::NonFiniteComputation);
        }
        last_residual = max_abs(&residual);
        let bound = residual_bound(a, &next, q, tolerance);
        p = next;

        if last_residual <= bound
        {
            let certificate = verify_discrete_lyapunov(a, &p, q, tolerance)?;
            return Ok(DiscreteLyapunovSolution {
                certificate,
                iterations: iteration,
            });
        }
    }

    Err(LyapunovError::DidNotConverge {
        iterations: options.max_iterations,
        residual: last_residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strict_options() -> DiscreteLyapunovOptions {
        DiscreteLyapunovOptions {
            tolerance: LyapunovTolerance::new(1e-12, 1e-12),
            max_iterations: 10_000,
        }
    }

    #[test]
    fn scalar_solution_matches_closed_form() {
        // P = Q / (1 - a²) = 1 / 0.75 = 4/3.
        let a = Mat::new(1, 1, vec![0.5]);
        let q = Mat::new(1, 1, vec![1.0]);
        let solution = solve_discrete_lyapunov(&a, &q, strict_options()).unwrap();
        assert!((solution.certificate.p.data[0] - 4.0 / 3.0).abs() < 1e-11);
        assert!(solution.certificate.residual_max_abs <= solution.certificate.residual_bound);
    }

    #[test]
    fn diagonal_solution_matches_independent_closed_forms() {
        let a = Mat::diag(&[0.5, 0.2]);
        let q = Mat::diag(&[1.0, 2.0]);
        let solution = solve_discrete_lyapunov(&a, &q, strict_options()).unwrap();
        let p = &solution.certificate.p;
        assert!((p.get(0, 0) - 4.0 / 3.0).abs() < 1e-11);
        assert!((p.get(1, 1) - 2.0 / 0.96).abs() < 1e-11);
        assert!(p.get(0, 1).abs() < 1e-14);
        assert!(p.get(1, 0).abs() < 1e-14);
    }

    #[test]
    fn verifier_accepts_exact_scalar_certificate() {
        let a = Mat::new(1, 1, vec![0.5]);
        let p = Mat::new(1, 1, vec![4.0 / 3.0]);
        let q = Mat::new(1, 1, vec![1.0]);
        let certificate = verify_discrete_lyapunov(
            &a,
            &p,
            &q,
            LyapunovTolerance::new(1e-12, 1e-12),
        )
        .unwrap();
        assert!(certificate.residual_max_abs < 1e-15);
    }

    #[test]
    fn unstable_scalar_does_not_produce_a_certificate() {
        let a = Mat::new(1, 1, vec![1.1]);
        let q = Mat::new(1, 1, vec![1.0]);
        let options = DiscreteLyapunovOptions {
            tolerance: LyapunovTolerance::new(1e-12, 1e-12),
            max_iterations: 100,
        };
        assert!(matches!(
            solve_discrete_lyapunov(&a, &q, options),
            Err(LyapunovError::DidNotConverge { .. })
                | Err(LyapunovError::NonFiniteComputation)
        ));
    }

    #[test]
    fn verifier_rejects_large_residual() {
        let a = Mat::new(1, 1, vec![1.1]);
        let p = Mat::new(1, 1, vec![1.0]);
        let q = Mat::new(1, 1, vec![1.0]);
        assert!(matches!(
            verify_discrete_lyapunov(
                &a,
                &p,
                &q,
                LyapunovTolerance::new(1e-12, 1e-12)
            ),
            Err(LyapunovError::ResidualTooLarge { .. })
        ));
    }

    #[test]
    fn malformed_certificates_are_rejected() {
        let a = Mat::identity(2);
        let nonsymmetric = Mat::new(2, 2, vec![1.0, 0.1, 0.0, 1.0]);
        let q = Mat::identity(2);
        assert!(matches!(
            verify_discrete_lyapunov(&a, &nonsymmetric, &q, LyapunovTolerance::default()),
            Err(LyapunovError::NotSymmetric { matrix: "P", .. })
        ));

        let indefinite = Mat::diag(&[1.0, -1.0]);
        assert!(matches!(
            solve_discrete_lyapunov(&a, &indefinite, DiscreteLyapunovOptions::default()),
            Err(LyapunovError::NotPositiveDefinite { matrix: "Q" })
        ));

        let bad_shape = Mat::new(1, 2, vec![1.0, 0.0]);
        assert!(matches!(
            solve_discrete_lyapunov(&bad_shape, &Mat::identity(1), strict_options()),
            Err(LyapunovError::NonSquare { matrix: "A", .. })
        ));
    }
}
