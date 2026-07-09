//! Descriptive statistics over a slice of samples. All functions are pure and
//! deterministic; variance/standard deviation use the unbiased `n − 1` divisor.

/// Arithmetic mean. Returns `NaN` for an empty slice.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty()
    {
        return f64::NAN;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Unbiased sample variance (divisor `n − 1`). Returns `NaN` for fewer than two
/// samples. Uses a two-pass formula for numerical stability.
pub fn variance(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2
    {
        return f64::NAN;
    }
    let m = mean(data);
    data.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n as f64 - 1.0)
}

/// Unbiased sample standard deviation.
pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Standard error of the mean, `s / √n`.
pub fn std_error(data: &[f64]) -> f64 {
    std_dev(data) / (data.len() as f64).sqrt()
}

/// Linear-interpolated quantile for `p ∈ [0, 1]` (the common "type 7" rule).
/// Returns `NaN` for an empty slice.
pub fn quantile(data: &[f64], p: f64) -> f64 {
    if data.is_empty()
    {
        return f64::NAN;
    }
    let mut v: Vec<f64> = data.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = p.clamp(0.0, 1.0);
    let h = (v.len() as f64 - 1.0) * p;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    v[lo] + (h - lo as f64) * (v[hi] - v[lo])
}

/// Median (the 0.5 quantile).
pub fn median(data: &[f64]) -> f64 {
    quantile(data, 0.5)
}

/// Minimum, ignoring `NaN`. `NaN` if empty.
pub fn min(data: &[f64]) -> f64 {
    data.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Maximum, ignoring `NaN`. `NaN` if empty.
pub fn max(data: &[f64]) -> f64 {
    data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_hand_computed() {
        let d = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((mean(&d) - 5.0).abs() < 1e-12);
        // Unbiased variance of this classic set is 32/7.
        assert!((variance(&d) - 32.0 / 7.0).abs() < 1e-12);
        assert!((median(&d) - 4.5).abs() < 1e-12);
        assert_eq!(min(&d), 2.0);
        assert_eq!(max(&d), 9.0);
    }

    #[test]
    fn quantile_endpoints_and_interpolation() {
        let d = [1.0, 2.0, 3.0, 4.0];
        assert!((quantile(&d, 0.0) - 1.0).abs() < 1e-12);
        assert!((quantile(&d, 1.0) - 4.0).abs() < 1e-12);
        assert!((quantile(&d, 0.5) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn degenerate_inputs_are_nan_not_panics() {
        assert!(mean(&[]).is_nan());
        assert!(variance(&[1.0]).is_nan());
        assert!(median(&[]).is_nan());
    }
}
