//! Deterministic polynomial least-squares for the `U -> 0` intercept.
//!
//! The effective PPN estimators are fitted with a low-degree polynomial in the
//! (normalized) compactness and evaluated at zero compactness. Everything here
//! is deterministic: fixed sample order, fixed accumulation order, partial
//! pivoting with a retain-first tie rule, no randomness, and no hidden
//! regularization.

use super::error::PpnError;

/// Maximum supported extrapolation degree. Kept small: higher degrees make the
/// normal equations ill-conditioned for `U << 1` faster than they add value.
pub const MAX_DEGREE: usize = 4;

/// Reject a fit whose conditioning indicator (scaled minimum pivot magnitude)
/// falls below this.
const CONDITIONING_FLOOR: f64 = 1.0e-10;

/// At or above this, an accepted fit is classified [`ConditioningClass::WellConditioned`]
/// rather than [`ConditioningClass::Marginal`]. Set four orders of magnitude above
/// `CONDITIONING_FLOOR` — comfortably inside `f64` working precision (`~1e-15`),
/// while still flagging fits that are accepted yet far from ideal.
pub const CONDITIONING_MARGINAL_THRESHOLD: f64 = 1.0e-6;

/// A classification of a fit's conditioning indicator ([`PolynomialFit::conditioning`]).
///
/// There is no accepted-fit variant for "rejected": a fit whose indicator falls
/// below `CONDITIONING_FLOOR` is always rejected outright with
/// [`PpnError::IllConditionedFit`] before any [`PolynomialFit`] (or downstream
/// `ParameterEstimate`) is constructed — see [`fit_polynomial_intercept`].
/// `IllConditioned` documents that reject boundary so the full spectrum is named
/// and independently testable via [`classify_conditioning`]; it cannot appear
/// inside a successful extraction's diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditioningClass {
    /// Indicator at or above [`CONDITIONING_MARGINAL_THRESHOLD`].
    WellConditioned,
    /// Indicator at or above `CONDITIONING_FLOOR` but below
    /// [`CONDITIONING_MARGINAL_THRESHOLD`]: accepted, but close to the reject
    /// floor — treat estimates built on it with extra caution.
    Marginal,
    /// Indicator below `CONDITIONING_FLOOR`. Unreachable from a successful
    /// extraction (see the type-level documentation); named for completeness.
    IllConditioned,
}

/// Classify a raw conditioning indicator (see [`PolynomialFit::conditioning`]).
#[must_use]
pub fn classify_conditioning(indicator: f64) -> ConditioningClass {
    if indicator < CONDITIONING_FLOOR
    {
        ConditioningClass::IllConditioned
    }
    else if indicator < CONDITIONING_MARGINAL_THRESHOLD
    {
        ConditioningClass::Marginal
    }
    else
    {
        ConditioningClass::WellConditioned
    }
}

/// The result of a least-squares polynomial fit `y ≈ sum_k c_k x^k`.
#[derive(Debug, Clone, PartialEq)]
pub struct PolynomialFit {
    /// The intercept `c_0` (the value at `x = 0`).
    pub intercept: f64,
    /// All fitted coefficients `c_0 .. c_degree`.
    pub coefficients: Vec<f64>,
    /// Euclidean residual norm `|| V c - y ||`.
    pub residual_norm: f64,
    /// Conditioning indicator: the ratio of the smallest to the largest pivot
    /// magnitude in the elimination (near `1` well-conditioned, near `0` not).
    pub conditioning: f64,
}

/// Fit `ys` against a degree-`degree` polynomial in `xs` and return the fit,
/// whose `intercept` is the `x -> 0` extrapolation.
///
/// `xs` are rescaled by their maximum magnitude before fitting so the normal
/// equations stay well scaled; this leaves the intercept unchanged. Returns a
/// typed [`PpnError`] for an unsupported degree, too few samples, non-finite
/// inputs, or a singular / ill-conditioned system.
pub fn fit_polynomial_intercept(
    xs: &[f64],
    ys: &[f64],
    degree: usize,
) -> Result<PolynomialFit, PpnError> {
    if degree == 0 || degree > MAX_DEGREE
    {
        return Err(PpnError::UnsupportedExtrapolationOrder {
            order: degree,
            maximum: MAX_DEGREE,
        });
    }
    let unknowns = degree + 1;
    if xs.len() != ys.len() || xs.len() < unknowns
    {
        return Err(PpnError::InsufficientSamples {
            available: xs.len().min(ys.len()),
            required: unknowns,
        });
    }
    if xs.iter().chain(ys.iter()).any(|value| !value.is_finite())
    {
        return Err(PpnError::NonFiniteEstimate);
    }

    // Rescale x by its largest magnitude so the design columns are O(1).
    let scale = xs.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    if scale == 0.0
    {
        return Err(PpnError::IllConditionedFit { conditioning: 0.0 });
    }
    let scaled: Vec<f64> = xs.iter().map(|&x| x / scale).collect();

    // Vandermonde design matrix V (rows = samples, columns = powers 0..=degree).
    let design: Vec<Vec<f64>> = scaled
        .iter()
        .map(|&s| {
            let mut power = 1.0;
            let mut row = Vec::with_capacity(unknowns);
            for _ in 0..unknowns
            {
                row.push(power);
                power *= s;
            }
            row
        })
        .collect();

    // Normal equations: (V^T V) c = V^T y, formed with fixed accumulation order.
    let mut normal = vec![vec![0.0_f64; unknowns]; unknowns];
    let mut rhs = vec![0.0_f64; unknowns];
    for (row, &y) in design.iter().zip(ys.iter())
    {
        for (j, &row_j) in row.iter().enumerate()
        {
            rhs[j] += row_j * y;
            for (k, &row_k) in row.iter().enumerate()
            {
                normal[j][k] += row_j * row_k;
            }
        }
    }

    let (coefficients, conditioning) = solve_symmetric_system(normal, rhs)?;
    if conditioning < CONDITIONING_FLOOR
    {
        return Err(PpnError::IllConditionedFit { conditioning });
    }

    // Residual || V c - y ||, in the scaled coordinates (scale-invariant since it
    // is measured against the same y).
    let mut residual_squared = 0.0_f64;
    for (row, &y) in design.iter().zip(ys.iter())
    {
        let model: f64 = row
            .iter()
            .zip(coefficients.iter())
            .map(|(&basis, &coefficient)| basis * coefficient)
            .sum();
        let difference = model - y;
        residual_squared += difference * difference;
    }
    let residual_norm = residual_squared.sqrt();

    let intercept = coefficients[0];
    if !intercept.is_finite() || !residual_norm.is_finite()
    {
        return Err(PpnError::NonFiniteEstimate);
    }

    Ok(PolynomialFit {
        intercept,
        coefficients,
        residual_norm,
        conditioning,
    })
}

/// Solve `matrix * x = rhs` by Gaussian elimination with partial pivoting,
/// returning the solution and a conditioning indicator (smallest / largest pivot
/// magnitude). A zero pivot is [`PpnError::SingularFit`].
// Dense elimination and back-substitution read most clearly with explicit row
// and column indices.
#[allow(clippy::needless_range_loop)]
fn solve_symmetric_system(
    mut matrix: Vec<Vec<f64>>,
    mut rhs: Vec<f64>,
) -> Result<(Vec<f64>, f64), PpnError> {
    let size = rhs.len();
    let mut smallest_pivot = f64::INFINITY;
    let mut largest_pivot = 0.0_f64;

    for column in 0..size
    {
        // Partial pivot: largest-magnitude entry in this column at or below the
        // diagonal; retain the first on ties (deterministic).
        let mut pivot_row = column;
        let mut pivot_magnitude = matrix[column][column].abs();
        for row in (column + 1)..size
        {
            let magnitude = matrix[row][column].abs();
            if magnitude > pivot_magnitude
            {
                pivot_row = row;
                pivot_magnitude = magnitude;
            }
        }

        if pivot_magnitude == 0.0
        {
            return Err(PpnError::SingularFit);
        }
        smallest_pivot = smallest_pivot.min(pivot_magnitude);
        largest_pivot = largest_pivot.max(pivot_magnitude);

        if pivot_row != column
        {
            matrix.swap(pivot_row, column);
            rhs.swap(pivot_row, column);
        }

        let pivot = matrix[column][column];
        for row in (column + 1)..size
        {
            let factor = matrix[row][column] / pivot;
            for inner in column..size
            {
                matrix[row][inner] -= factor * matrix[column][inner];
            }
            rhs[row] -= factor * rhs[column];
        }
    }

    // Back-substitution.
    let mut solution = vec![0.0_f64; size];
    for row in (0..size).rev()
    {
        let mut value = rhs[row];
        for column in (row + 1)..size
        {
            value -= matrix[row][column] * solution[column];
        }
        solution[row] = value / matrix[row][row];
    }

    let conditioning = if largest_pivot > 0.0
    {
        smallest_pivot / largest_pivot
    }
    else
    {
        0.0
    };
    Ok((solution, conditioning))
}
