//! Learning/adaptation module.
//!
//! Provides lightweight regression and pattern-detection utilities
//! that depend only on `std` and `serde`.

pub mod control;
pub mod finance;
pub mod optim;
pub mod pattern_miner;
pub mod time_series;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// PatternMemory
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternMemory {
    pub patterns: Vec<(Vec<f64>, f64)>,
}

impl Default for PatternMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternMemory {
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: Gaussian elimination with partial pivoting
// ---------------------------------------------------------------------------

/// Solve the linear system `A * x = b` **in-place** (A is consumed, b is
/// overwritten with the solution).  Returns `None` when the matrix is
/// singular (zero pivot after partial pivot).
fn gauss_elim(a: &mut [Vec<f64>], b: &mut [f64]) -> Option<()> {
    let n = a.len();
    // Forward elimination
    for col in 0..n {
        // Partial pivoting: find row with largest absolute value in this col
        let mut best = col;
        for row in (col + 1)..n {
            if a[row][col].abs() > a[best][col].abs() {
                best = row;
            }
        }
        if a[best][col].abs() < 1e-15 {
            return None; // singular
        }
        a.swap(col, best);
        b.swap(col, best);

        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            // Eliminate column `col` in this row
            #[allow(clippy::needless_range_loop)]
            for k in col..n {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution
    for col in (0..n).rev() {
        let pivot = a[col][col];
        let mut sum = b[col];
        for k in (col + 1)..n {
            sum -= a[col][k] * b[k];
        }
        b[col] = sum / pivot;
    }
    Some(())
}

// ---------------------------------------------------------------------------
// polynomial_fit
// ---------------------------------------------------------------------------

/// Fit a polynomial of the given `degree` to the points `(x[i], y[i])` using
/// ordinary least squares (Vandermonde + normal equations).
///
/// Returns the coefficient vector `[c0, c1, ..., c_{degree}]` where the
/// polynomial is `c0 + c1*x + c2*x^2 + ... + cd*x^degree`.
///
/// Returns an empty vector when either input is empty, the two slices have
/// different lengths, or the system is singular (e.g. not enough distinct
/// points).
///
/// # Exemple
/// ```
/// use scirust_learning::polynomial_fit;
/// let x = vec![0.0, 1.0, 2.0];
/// let y = vec![1.0, 2.0, 5.0]; // y = x^2 + 1
/// let coeffs = polynomial_fit(&x, &y, 2);
/// assert!((coeffs[0] - 1.0).abs() < 1e-10);
/// assert!((coeffs[1] - 0.0).abs() < 1e-10);
/// assert!((coeffs[2] - 1.0).abs() < 1e-10);
/// ```
pub fn polynomial_fit(x: &[f64], y: &[f64], degree: usize) -> Vec<f64> {
    if x.is_empty() || y.is_empty() || x.len() != y.len() {
        return vec![];
    }
    let n = x.len();
    let d = degree + 1; // number of coefficients

    // If we have fewer points than coefficients the system is under-determined.
    if n < d {
        return vec![];
    }

    // Build the normal-equation system: A^T * A * coeffs = A^T * y
    // where A is the Vandermonde matrix (n x d) with A[i][k] = x[i]^k.

    // We work with the Gram matrix G = A^T * A  (d x d) and rhs = A^T * y.
    // Pre-compute powers of each x to avoid repeated pow() calls.
    let mut powers: Vec<Vec<f64>> = Vec::with_capacity(n);
    for &xi in x {
        let mut row = Vec::with_capacity(d);
        let mut p = 1.0;
        for _ in 0..d {
            row.push(p);
            p *= xi;
        }
        powers.push(row);
    }

    // G[k][l] = sum_i x[i]^(k+l)
    let mut g: Vec<Vec<f64>> = vec![vec![0.0; d]; d];
    let mut rhs: Vec<f64> = vec![0.0; d];
    for i in 0..n {
        let yi = y[i];
        for k in 0..d {
            rhs[k] += powers[i][k] * yi;
            for l in 0..d {
                g[k][l] += powers[i][k] * powers[i][l];
            }
        }
    }

    let mut coeffs = rhs;
    if gauss_elim(&mut g, &mut coeffs).is_none() {
        return vec![];
    }
    coeffs
}

// ---------------------------------------------------------------------------
// linear_regression
// ---------------------------------------------------------------------------

/// Compute the slope and intercept of the best-fit line `y = slope * x + intercept`
/// using the covariance/variance formula.
///
/// Returns `(slope, intercept)`.
/// Returns `(0.0, 0.0)` when there are fewer than 2 points or slices have
/// different lengths.
///
/// # Exemple
/// ```
/// use scirust_learning::linear_regression;
/// let x = vec![1.0, 2.0, 3.0];
/// let y = vec![2.0, 4.0, 6.0];
/// let (slope, intercept) = linear_regression(&x, &y);
/// assert!((slope - 2.0).abs() < 1e-10);
/// assert!(intercept.abs() < 1e-10);
/// ```
pub fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
    if x.len() < 2 || y.len() < 2 || x.len() != y.len() {
        return (0.0, 0.0);
    }
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var += dx * dx;
    }
    if var.abs() < 1e-15 {
        return (0.0, mean_y);
    }
    let slope = cov / var;
    let intercept = mean_y - slope * mean_x;
    (slope, intercept)
}

// ---------------------------------------------------------------------------
// discover_patterns
// ---------------------------------------------------------------------------

/// Run basic pattern detection on a univariate time series.
///
/// Currently detects:
/// - **Moving-average crossover**: when a short-term MA crosses a long-term MA.
/// - **Volatility regime**: periods of high or low volatility relative to the
///   series-wide standard deviation.
///
/// Each detected pattern is returned as a human-readable string.
pub fn discover_patterns(data: &[f64]) -> Vec<String> {
    if data.len() < 10 {
        return vec!["Insufficient data: need at least 10 points".to_string()];
    }
    let mut signals: Vec<String> = Vec::new();

    // ---------- helper: simple moving average ----------
    let sma = |period: usize| -> Vec<f64> {
        if period == 0 || period > data.len() {
            return vec![];
        }
        let mut result = Vec::with_capacity(data.len() - period + 1);
        let mut sum: f64 = data[..period].iter().sum();
        result.push(sum / period as f64);
        for i in period..data.len() {
            sum += data[i] - data[i - period];
            result.push(sum / period as f64);
        }
        result
    };

    // ---------- MA crossover detection ----------
    let short_period = 3.min(data.len() - 1);
    let long_period = 8.min(data.len() - 1);
    // Ensure short < long
    let short_period = short_period.min(long_period - 1).max(1);
    let long_period = (short_period + 1).min(data.len() - 1);

    let short_ma = sma(short_period);
    let long_ma = sma(long_period);

    // Align: short_ma has len = N - short + 1, long_ma has len = N - long + 1
    // The earliest aligned index in `data` is `long_period - 1` (the first valid long MA point).
    // short_ma at index (i - short_period + 1) corresponds to the same window end as
    // long_ma at index (i - long_period + 1).
    let offset_short = long_period - short_period; // how far ahead short_ma is
    for i in 0..(long_ma.len().saturating_sub(1)) {
        let idx_short = i + offset_short;
        if idx_short == 0 || idx_short >= short_ma.len() {
            continue;
        }
        let prev_short = short_ma[idx_short - 1];
        let curr_short = short_ma[idx_short];
        let prev_long = long_ma[i];
        let curr_long = long_ma[i + 1];

        let prev_diff = prev_short - prev_long;
        let curr_diff = curr_short - curr_long;

        // Crossing: sign of diff changed
        if prev_diff.is_sign_negative() && curr_diff.is_sign_positive() {
            signals.push(format!(
                "MA crossover (bullish) at index {}: short MA crossed above long MA",
                long_period + i
            ));
        } else if prev_diff.is_sign_positive() && curr_diff.is_sign_negative() {
            signals.push(format!(
                "MA crossover (bearish) at index {}: short MA crossed below long MA",
                long_period + i
            ));
        }
    }

    // ---------- Volatility regime detection ----------
    let n = data.len();
    if n >= 20 {
        // Compute overall standard deviation
        let mean: f64 = data.iter().sum::<f64>() / n as f64;
        let variance: f64 = data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        let stddev = variance.sqrt();
        let window = 10;

        // Compute rolling volatility (standard deviation over sliding windows)
        let mut rolling_vol: Vec<f64> = Vec::new();
        for w_start in 0..=(n - window) {
            let slice = &data[w_start..w_start + window];
            let m: f64 = slice.iter().sum::<f64>() / window as f64;
            let v: f64 = slice.iter().map(|v| (v - m).powi(2)).sum::<f64>() / window as f64;
            rolling_vol.push(v.sqrt());
        }

        // Detect sustained high/low volatility regimes
        let high_threshold = stddev * 1.5;
        let low_threshold = stddev * 0.5;

        let mut in_high = false;
        let mut in_low = false;
        let mut regime_start = 0;

        for (i, vol) in rolling_vol.iter().enumerate() {
            let idx = window / 2 + i; // centre of the window
            if *vol > high_threshold && !in_high {
                in_high = true;
                in_low = false;
                regime_start = idx;
            } else if *vol < low_threshold && !in_low {
                in_low = true;
                in_high = false;
                regime_start = idx;
            } else if (in_high && *vol <= high_threshold) || (in_low && *vol >= low_threshold) {
                // Regime ended
                let regime_type = if in_high { "high" } else { "low" };
                signals.push(format!(
                    "Volatility regime ({}) from index {} to {}",
                    regime_type, regime_start, idx
                ));
                in_high = false;
                in_low = false;
            }
        }

        // Close any open regime at the end of the series
        if in_high {
            signals.push(format!(
                "Volatility regime (high) from index {} to {} (ongoing)",
                regime_start,
                n - 1
            ));
        } else if in_low {
            signals.push(format!(
                "Volatility regime (low) from index {} to {} (ongoing)",
                regime_start,
                n - 1
            ));
        }
    }

    if signals.is_empty() {
        signals.push("No significant patterns detected".to_string());
    }

    signals
}

// ---------------------------------------------------------------------------
// tensor::device
// ---------------------------------------------------------------------------

pub mod tensor {
    pub mod device {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
        pub enum Device {
            #[default]
            Cpu,
            Gpu,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_linear_regression_perfect_line() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect(); // y = 2x + 1
        let (slope, intercept) = linear_regression(&x, &y);
        assert!(approx_eq(slope, 2.0, 1e-10), "slope = {}", slope);
        assert!(
            approx_eq(intercept, 1.0, 1e-10),
            "intercept = {}",
            intercept
        );
    }

    #[test]
    fn test_linear_regression_noisy() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.1, 3.9, 6.2, 7.8, 10.1]; // approx y = 2x
        let (slope, intercept) = linear_regression(&x, &y);
        assert!(approx_eq(slope, 2.0, 0.2), "slope = {}", slope);
        assert!(approx_eq(intercept, 0.0, 0.3), "intercept = {}", intercept);
    }

    #[test]
    fn test_linear_regression_short_input() {
        assert_eq!(linear_regression(&[1.0], &[2.0]), (0.0, 0.0));
        assert_eq!(linear_regression(&[], &[]), (0.0, 0.0));
    }

    #[test]
    fn test_polynomial_fit_quadratic() {
        let x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let y: Vec<f64> = x.iter().map(|v| 3.0 * v * v + 2.0 * v + 1.0).collect();
        let coeffs = polynomial_fit(&x, &y, 2);
        assert_eq!(coeffs.len(), 3);
        assert!(approx_eq(coeffs[0], 1.0, 1e-8), "c0 = {}", coeffs[0]);
        assert!(approx_eq(coeffs[1], 2.0, 1e-8), "c1 = {}", coeffs[1]);
        assert!(approx_eq(coeffs[2], 3.0, 1e-8), "c2 = {}", coeffs[2]);
    }

    #[test]
    fn test_polynomial_fit_linear() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![5.0, 7.0, 9.0, 11.0];
        let coeffs = polynomial_fit(&x, &y, 1);
        assert_eq!(coeffs.len(), 2);
        assert!(approx_eq(coeffs[0], 5.0, 1e-8), "c0 = {}", coeffs[0]);
        assert!(approx_eq(coeffs[1], 2.0, 1e-8), "c1 = {}", coeffs[1]);
    }

    #[test]
    fn test_polynomial_fit_too_few_points() {
        assert!(polynomial_fit(&[1.0, 2.0], &[3.0, 4.0], 3).is_empty());
        assert!(polynomial_fit(&[], &[], 1).is_empty());
    }

    #[test]
    fn test_discover_patterns_insufficient() {
        let r = discover_patterns(&[1.0, 2.0, 3.0]);
        assert!(r[0].contains("Insufficient"));
    }

    #[test]
    fn test_discover_patterns_runs() {
        // A simple upward trend — no fancy patterns but should not crash
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let r = discover_patterns(&data);
        assert!(!r.is_empty());
    }
}
