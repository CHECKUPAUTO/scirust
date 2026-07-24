//! Classical linear partial correlation: QR residualization and Fisher-z
//! calibration.
//!
//! # What partial correlation measures
//!
//! The partial correlation `ρ(X, Y | Z)` is the Pearson correlation between
//! the parts of `X` and `Y` that a linear regression on `Z` cannot explain —
//! equivalently, the correlation of the residuals of `X ~ Z` and `Y ~ Z`
//! (with an intercept). It is a measure of *linear* conditional association.
//! It can be **zero even when `X` and `Y` are conditionally dependent** if
//! that dependence is nonlinear (see the crate's adversarial tests for an
//! explicit demonstration) — a linear statistic is not evidence about
//! nonlinear structure.
//!
//! # Classical (Gaussian-style) assumptions
//!
//! Fisher-z calibration ([`fisher_z_p_value`]) is an asymptotic
//! approximation: `atanh(ρ̂)` is approximately normally distributed with
//! variance `1 / (n - |Z| - 3)` when `(X, Y, Z)` are jointly approximately
//! Gaussian (or at least the linear-regression residuals are approximately
//! normal) and `n` is not too small relative to `|Z|`. It degrades under
//! heavy tails, non-normality, and small samples relative to `|Z|` — that is
//! exactly the regime [`crate::RobustCalibration`] and permutation calibration
//! exist for.

use crate::error::CausalError;
use scirust_solvers::Matrix;
use scirust_solvers::linalg::{qr_decompose, solve_qr_least_squares, svd};
use scirust_stats::{Distribution, Normal};

/// Fixed-order-accumulation Pearson correlation. Returns `None` if either
/// side has (numerically) zero variance — the caller has the variable-index
/// context needed to turn that into a [`CausalError::ZeroVariance`].
///
/// A result that mathematically must lie in `[-1, 1]` (Cauchy-Schwarz) is
/// clamped back into that range only for a floating-point overshoot of at
/// most `1e-9` — a materially larger overshoot is a bug, not drift, and is
/// deliberately left visible rather than hidden.
#[must_use]
pub(crate) fn pearson_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut covariance = 0.0;
    let mut variance_x = 0.0;
    let mut variance_y = 0.0;
    for i in 0..x.len()
    {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }

    const ZERO_VARIANCE_TOLERANCE: f64 = 1e-14;
    if variance_x <= ZERO_VARIANCE_TOLERANCE || variance_y <= ZERO_VARIANCE_TOLERANCE
    {
        return None;
    }

    let r = covariance / (variance_x.sqrt() * variance_y.sqrt());
    const OVERSHOOT_TOLERANCE: f64 = 1e-9;
    let r = if r > 1.0 && r <= 1.0 + OVERSHOOT_TOLERANCE
    {
        1.0
    }
    else if (-1.0 - OVERSHOOT_TOLERANCE..-1.0).contains(&r)
    {
        -1.0
    }
    else
    {
        r
    };
    Some(r)
}

/// The rank of `design` (an `n × p` matrix), via its singular values: the
/// count of singular values at least `relative_tolerance` of the largest one.
/// `design` is otherwise unused by the caller after this call, so this owns
/// the SVD needed purely for rank diagnosis, kept separate from the QR
/// factorization used for the actual least-squares solve (SVD's singular
/// values are basis/column-order-invariant, unlike an unpivoted QR's `R`
/// diagonal, so this is the robust way to detect rank deficiency regardless
/// of which columns are the dependent ones).
fn numerical_rank(design: &Matrix, relative_tolerance: f64) -> Result<usize, CausalError> {
    let factorization = svd(design).map_err(|e| CausalError::SolverFailure {
        detail: e.to_string(),
    })?;
    let largest = factorization.s.first().copied().unwrap_or(0.0);
    if largest <= 0.0
    {
        return Ok(0);
    }
    let threshold = relative_tolerance * largest;
    Ok(factorization.s.iter().filter(|&&sv| sv > threshold).count())
}

/// Regresses `target` on `[intercept, z_columns...]` via QR and returns the
/// residual `target - fitted` (row-major, same order as `target`).
///
/// # Errors
///
/// [`CausalError::RankDeficientConditioningSet`] if the design's numerical
/// rank (via SVD, see [`numerical_rank`]) is below its column count under
/// `relative_tolerance`; [`CausalError::SolverFailure`] if the underlying QR
/// solve fails for any other reason.
fn residualize(
    target: &[f64],
    z_columns: &[&[f64]],
    relative_tolerance: f64,
) -> Result<Vec<f64>, CausalError> {
    let n = target.len();
    let p = 1 + z_columns.len();

    let mut data = Vec::with_capacity(n * p);
    for row in 0..n
    {
        data.push(1.0); // intercept
        for column in z_columns
        {
            data.push(column[row]);
        }
    }
    let design = Matrix::from_row_major(n, p, data);

    let rank = numerical_rank(&design, relative_tolerance)?;
    if rank < p
    {
        return Err(CausalError::RankDeficientConditioningSet { rank, columns: p });
    }

    let qr = qr_decompose(design.clone()).map_err(|e| CausalError::SolverFailure {
        detail: e.to_string(),
    })?;
    let coefficients =
        solve_qr_least_squares(&qr, target).map_err(|e| CausalError::SolverFailure {
            detail: e.to_string(),
        })?;
    let fitted = design
        .matvec(&coefficients)
        .map_err(|e| CausalError::SolverFailure {
            detail: e.to_string(),
        })?;

    Ok(target.iter().zip(&fitted).map(|(t, f)| t - f).collect())
}

/// The classical partial-correlation statistic (residualize-then-correlate),
/// reducing to plain Pearson correlation when `z_columns` is empty.
///
/// `r` is `None` exactly when [`pearson_correlation`] finds zero variance on
/// either residual — `x_residual`/`y_residual` are still populated so the
/// caller (which knows the original variable indices) can attribute *which*
/// side was degenerate and report a precise
/// [`CausalError::ZeroVariance`].
pub(crate) struct ClassicalPartialCorrelationOutcome {
    pub r: Option<f64>,
    pub x_residual: Vec<f64>,
    pub y_residual: Vec<f64>,
    pub rank: usize,
}

/// Computes `ρ(x, y | z_columns)` via QR residualization (or plain Pearson
/// correlation when `z_columns` is empty — no regression is performed, and
/// `rank` is reported as `0`, since there is no conditioning design at all).
///
/// # Errors
///
/// [`CausalError::RankDeficientConditioningSet`] / [`CausalError::SolverFailure`]
/// from [`residualize`].
pub(crate) fn classical_partial_correlation(
    x: &[f64],
    y: &[f64],
    z_columns: &[&[f64]],
    relative_tolerance: f64,
) -> Result<ClassicalPartialCorrelationOutcome, CausalError> {
    let (x_residual, y_residual, rank) = if z_columns.is_empty()
    {
        (x.to_vec(), y.to_vec(), 0)
    }
    else
    {
        let x_residual = residualize(x, z_columns, relative_tolerance)?;
        let y_residual = residualize(y, z_columns, relative_tolerance)?;
        (x_residual, y_residual, 1 + z_columns.len())
    };

    let r = pearson_correlation(&x_residual, &y_residual);
    Ok(ClassicalPartialCorrelationOutcome {
        r,
        x_residual,
        y_residual,
        rank,
    })
}

/// Fisher-z calibration of a (partial) correlation `r` computed from `n`
/// samples with `conditioning_size` conditioning variables (`|Z|`, `0` for
/// plain Pearson correlation): `z = atanh(r) · sqrt(n − conditioning_size − 3)`,
/// two-sided `p = 2 · Φ̄(|z|)` where `Φ̄` is the standard normal survival
/// function.
///
/// Returns `None` — not an error — when `n - conditioning_size - 3 <= 0`: the
/// statistic itself is perfectly well-defined, but there are not enough
/// residual degrees of freedom for the asymptotic normal approximation to
/// mean anything. The caller reports this as
/// [`crate::IndependenceDecision::Inconclusive`], not a hard failure.
///
/// Also returns `None` for `|r| == 1` exactly (`atanh` diverges) — perfect
/// (anti)correlation is not itself an error, but its p-value is degenerate.
#[must_use]
pub(crate) fn fisher_z_p_value(r: f64, n: usize, conditioning_size: usize) -> Option<f64> {
    let degrees_of_freedom = n as f64 - conditioning_size as f64 - 3.0;
    if degrees_of_freedom <= 0.0
    {
        return None;
    }
    if r.abs() >= 1.0
    {
        return None;
    }
    let z = r.atanh() * degrees_of_freedom.sqrt();
    Some(2.0 * Normal::standard().sf(z.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_matches_hand_computation() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_correlation(&x, &y).unwrap();
        assert!(
            (r - 1.0).abs() < 1e-12,
            "perfect linear relation should give r=1, got {r}"
        );
    }

    #[test]
    fn pearson_rejects_zero_variance() {
        let x = [1.0, 1.0, 1.0, 1.0];
        let y = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(pearson_correlation(&x, &y), None);
    }

    #[test]
    fn pearson_sign_flip_on_negation() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0, 3.0, 1.0, -1.0, -3.0];
        let r = pearson_correlation(&x, &y).unwrap();
        assert!(r < 0.0);
        assert!((r + 1.0).abs() < 1e-9);
    }

    #[test]
    fn residualize_against_empty_z_is_not_called_directly_by_classical_partial_correlation() {
        // classical_partial_correlation short-circuits z_columns.is_empty()
        // rather than calling residualize with p=1 (intercept only); confirm
        // that short-circuit actually reduces to Pearson correlation.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = [2.0, 1.0, 4.0, 3.0, 6.0, 5.0];
        let direct = pearson_correlation(&x, &y).unwrap();
        let via_partial = classical_partial_correlation(&x, &y, &[], 1e-9).unwrap();
        assert!((direct - via_partial.r.unwrap()).abs() < 1e-12);
        assert_eq!(via_partial.rank, 0);
    }

    #[test]
    fn fisher_z_reduces_correctly_and_is_symmetric_in_sign() {
        let p_pos = fisher_z_p_value(0.6, 30, 0).unwrap();
        let p_neg = fisher_z_p_value(-0.6, 30, 0).unwrap();
        assert!(
            (p_pos - p_neg).abs() < 1e-12,
            "two-sided p-value must not depend on sign"
        );
        assert!(
            p_pos < 0.05,
            "r=0.6 at n=30 should be clearly significant, got p={p_pos}"
        );
    }

    #[test]
    fn fisher_z_returns_none_when_degrees_of_freedom_are_exhausted() {
        // n - |Z| - 3 <= 0
        assert_eq!(fisher_z_p_value(0.5, 5, 3), None);
        assert_eq!(fisher_z_p_value(0.5, 4, 1), None);
    }

    #[test]
    fn fisher_z_returns_none_at_the_correlation_boundary() {
        assert_eq!(fisher_z_p_value(1.0, 100, 0), None);
        assert_eq!(fisher_z_p_value(-1.0, 100, 0), None);
    }

    #[test]
    fn residualize_rejects_rank_deficient_design() {
        // z1 and z2 are identical columns: [intercept, z1, z2] has rank 2, not 3.
        let target = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let z1 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let z2 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = residualize(&target, &[&z1, &z2], 1e-9);
        assert!(matches!(
            result,
            Err(CausalError::RankDeficientConditioningSet {
                rank: 2,
                columns: 3
            })
        ));
    }

    #[test]
    fn residualize_recovers_zero_residual_for_an_exact_linear_relationship() {
        // target is an exact linear function of z: residual must be ~0.
        let z = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let target: Vec<f64> = z.iter().map(|v| 3.0 + 2.0 * v).collect();
        let residual = residualize(&target, &[&z], 1e-9).unwrap();
        for r in residual
        {
            assert!(r.abs() < 1e-9, "residual should be ~0, got {r}");
        }
    }
}
