//! Internal numerical helpers shared across the interpolation methods.
//!
//! Nothing in this module is part of the public API.

use crate::error::InterpError;

/// Validate node arrays for a multi-point method.
///
/// Checks, in order: equal lengths, at least `min_points` points, all values
/// finite (rejecting NaN/±∞), and strictly increasing abscissae.
pub(crate) fn validate_nodes(xs: &[f64], ys: &[f64], min_points: usize) -> Result<(), InterpError> {
    if xs.len() != ys.len()
    {
        return Err(InterpError::LengthMismatch {
            xs: xs.len(),
            ys: ys.len(),
        });
    }
    if xs.len() < min_points
    {
        return Err(InterpError::TooFewPoints {
            got: xs.len(),
            need: min_points,
        });
    }
    for (i, &v) in xs.iter().enumerate()
    {
        if !v.is_finite()
        {
            return Err(InterpError::NonFinite { index: i });
        }
    }
    for (i, &v) in ys.iter().enumerate()
    {
        if !v.is_finite()
        {
            return Err(InterpError::NonFinite { index: i });
        }
    }
    for (i, w) in xs.windows(2).enumerate()
    {
        if w[1] <= w[0]
        {
            return Err(InterpError::NotStrictlyIncreasing { index: i + 1 });
        }
    }
    Ok(())
}

/// Locate the segment containing `x`.
///
/// Returns the left index `i` of the bracketing interval `[xs[i], xs[i + 1]]`,
/// clamped to `0..=n - 2` so that queries below `xs[0]` map to the first
/// segment and queries above `xs[n - 1]` map to the last one (the basis for
/// each method's extrapolation). Callers guarantee `xs.len() >= 2`.
pub(crate) fn find_segment(xs: &[f64], x: f64) -> usize {
    let n = xs.len();
    if x <= xs[0]
    {
        return 0;
    }
    if x >= xs[n - 1]
    {
        return n - 2;
    }
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1
    {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }
    lo
}

/// Evaluate a cubic Hermite segment.
///
/// The segment spans a node with value `y0` and slope `d0` at its left end and
/// value `y1`, slope `d1` at its right end, with width `h`. `dx` is
/// `x - x_left`; values of `dx` outside `[0, h]` extrapolate the same cubic.
pub(crate) fn hermite(y0: f64, y1: f64, d0: f64, d1: f64, h: f64, dx: f64) -> f64 {
    let t = dx / h;
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1
}

/// Solve a tridiagonal linear system with the Thomas algorithm.
///
/// `a` is the sub-diagonal (`a[0]` unused), `b` the main diagonal, `c` the
/// super-diagonal (`c[n - 1]` unused) and `d` the right-hand side. The systems
/// assembled by this crate are diagonally dominant, so no pivoting is needed.
pub(crate) fn thomas(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut cp = vec![0.0; n];
    let mut dp = vec![0.0; n];
    cp[0] = c[0] / b[0];
    dp[0] = d[0] / b[0];
    for i in 1..n
    {
        let m = b[i] - a[i] * cp[i - 1];
        cp[i] = if i < n - 1 { c[i] / m } else { 0.0 };
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m;
    }
    let mut x = vec![0.0; n];
    x[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev()
    {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    x
}
