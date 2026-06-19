//! Seasonal and cyclic pattern detection algorithms.
//!
//! Provides STL decomposition, autocorrelation-based period detection, Fourier
//! analysis, wavelet-like windowed transforms, seasonal adjustment, trend tests,
//! and change-point detection — all operating on `f64` time-series data.
//!
//! # Modules
//! - **stl** — Seasonal-Trend decomposition using Loess
//! - **detection** — ACF, PACF, periodogram, dominant frequency
//! - **cyclic** — Fourier analysis, windowed FFT, cycle length, phase
//! - **adjustment** — Moving-average decomposition, X-11 style, deseasonalize
//! - **trend** — Mann-Kendall test, Sen's slope, linear trend with CI
//! - **changepoint** — Seasonal CUSUM and seasonal break detection

use serde::{Deserialize, Serialize};

// Modules are defined inline below.

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Next power of two ≥ `n`.
#[inline]
fn next_pow2(n: usize) -> usize {
    if n <= 1
    {
        return 1;
    }
    1usize << (usize::BITS - (n - 1).leading_zeros())
}

/// Mean of a slice.
#[inline]
fn mean(data: &[f64]) -> f64 {
    if data.is_empty()
    {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Trimmed mean: discard `trim` elements from each end, then average.
#[inline]
fn trimmed_mean(data: &[f64], trim: usize) -> f64 {
    if data.len() <= 2 * trim
    {
        return mean(data);
    }
    let trimmed = &data[trim..data.len() - trim];
    trimmed.iter().sum::<f64>() / trimmed.len() as f64
}

/// Variance of a slice.
#[inline]
fn variance(data: &[f64]) -> f64 {
    if data.len() < 2
    {
        return 0.0;
    }
    let m = mean(data);
    data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / data.len() as f64
}

/// Standard deviation.
#[allow(dead_code)]
#[inline]
fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Pad a centered array (shorter than n) to full length by repeating edge values.
fn pad_centered(data: &mut Vec<f64>, target_len: usize) {
    let current_len = data.len();
    if current_len >= target_len
    {
        data.truncate(target_len);
        return;
    }
    let pad_each = (target_len - current_len) / 2;
    let mut left = Vec::with_capacity(pad_each);
    for _ in 0..pad_each
    {
        left.push(data[0]);
    }
    let mut right = Vec::with_capacity(target_len - current_len - pad_each);
    for _ in 0..target_len - current_len - pad_each
    {
        right.push(data[data.len() - 1]);
    }
    let mut result = left;
    result.append(data);
    result.extend(right);
    *data = result;
}

/// Quantile of a sorted slice (linear interpolation).
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty()
    {
        return 0.0;
    }
    if sorted.len() == 1
    {
        return sorted[0];
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Running (simple) moving average of `window` size.
fn moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || data.is_empty()
    {
        return Vec::new();
    }
    let n = data.len();
    if window > n
    {
        return vec![mean(data)];
    }
    let mut result = Vec::with_capacity(n - window + 1);
    let mut sum: f64 = data[..window].iter().sum();
    result.push(sum / window as f64);
    for i in window..n
    {
        sum += data[i] - data[i - window];
        result.push(sum / window as f64);
    }
    result
}

/// Centered moving average: for even `window`, average two successive MAs.
fn centered_moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || data.is_empty()
    {
        return Vec::new();
    }
    let n = data.len();
    if window > n
    {
        return vec![mean(data)];
    }
    if window % 2 == 1
    {
        return moving_average(data, window);
    }
    // Even window: compute MA(window), then MA(2) of that
    let ma = moving_average(data, window);
    moving_average(&ma, 2)
}

/// Linear regression: returns (slope, intercept).
fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
    assert_eq!(x.len(), y.len(), "x and y must have same length");
    let n = x.len() as f64;
    if n < 2.0
    {
        return (0.0, if n == 1.0 { y[0] } else { 0.0 });
    }
    let sx: f64 = x.iter().sum();
    let sy: f64 = y.iter().sum();
    let sxx: f64 = x.iter().map(|&v| v * v).sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < f64::EPSILON
    {
        return (0.0, sy / n);
    }
    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;
    (slope, intercept)
}

// ─── Result types ────────────────────────────────────────────────────────────

/// Result of STL decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STLResult {
    /// Original time series length.
    pub length: usize,
    /// Seasonal period.
    pub period: usize,
    /// Trend component.
    pub trend: Vec<f64>,
    /// Seasonal component.
    pub seasonal: Vec<f64>,
    /// Remainder (original - trend - seasonal).
    pub remainder: Vec<f64>,
}

/// Result of a seasonal detection analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodDetection {
    /// Detected dominant period (in samples).
    pub period: usize,
    /// Strength of the detected period (0.0 – 1.0).
    pub strength: f64,
    /// All candidate periods with their strengths.
    pub candidates: Vec<PeriodCandidate>,
}

/// A single candidate period from detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodCandidate {
    /// Period in samples.
    pub period: usize,
    /// Strength / confidence score.
    pub strength: f64,
}

/// Cyclic pattern information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CyclicPattern {
    /// Estimated cycle length in samples.
    pub cycle_length: f64,
    /// Phase offset in radians.
    pub phase: f64,
    /// Amplitude of the dominant cycle.
    pub amplitude: f64,
    /// All detected harmonics.
    pub harmonics: Vec<Harmonic>,
}

/// A detected harmonic component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Harmonic {
    /// Frequency index.
    pub frequency_index: usize,
    /// Frequency in cycles per sample.
    pub frequency: f64,
    /// Amplitude.
    pub amplitude: f64,
    /// Phase in radians.
    pub phase: f64,
}

/// Linear trend result with confidence intervals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendResult {
    /// Slope (change per time unit).
    pub slope: f64,
    /// Intercept.
    pub intercept: f64,
    /// Mann-Kendall p-value.
    pub p_value: f64,
    /// Mann-Kendall S statistic.
    pub s_statistic: f64,
    /// Mann-Kendall Z statistic.
    pub z_statistic: f64,
    /// Sen's slope estimate.
    pub sens_slope: f64,
    /// 95% CI for slope [lower, upper].
    pub slope_ci: [f64; 2],
    /// Whether trend is significant (p < 0.05).
    pub significant: bool,
    /// Trend direction.
    pub direction: TrendDirection,
}

/// Direction of a detected trend.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    NoTrend,
}

/// Change point result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangePoint {
    /// Index of the change point.
    pub index: usize,
    /// Confidence / magnitude of the change.
    pub magnitude: f64,
    /// Seasonal component of the change.
    pub seasonal_component: f64,
}

// ═════════════════════════════════════════════════════════════════════════════
// Module implementations
// ═════════════════════════════════════════════════════════════════════════════

pub mod stl {
    //! Seasonal-Trend decomposition using Loess (STL).
    //!
    //! Decomposes a time series `y` into `y = trend + seasonal + remainder`.
    //! Uses centered moving averages for trend extraction and iterative
    //! seasonal smoothing.

    use super::*;

    /// Configuration for STL decomposition.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct STLConfig {
        /// Seasonal period (e.g. 12 for monthly data).
        pub period: usize,
        /// Number of inner iterations (default: 2).
        pub inner_iterations: usize,
        /// Number of outer iterations for robustness (default: 0 = non-robust).
        pub outer_iterations: usize,
        /// Trend smoothing window (must be odd, 0 = auto).
        pub trend_window: usize,
        /// Seasonal smoothing window (must be odd, 0 = auto).
        pub seasonal_window: usize,
    }

    impl Default for STLConfig {
        fn default() -> Self {
            Self {
                period: 12,
                inner_iterations: 2,
                outer_iterations: 0,
                trend_window: 0,
                seasonal_window: 0,
            }
        }
    }

    /// Run STL decomposition on the given time series.
    pub fn stl_decompose(data: &[f64], config: &STLConfig) -> STLResult {
        let n = data.len();
        let period = config.period;
        assert!(period >= 2, "period must be >= 2");
        assert!(n >= 2 * period, "data length must be >= 2 * period");

        // Auto-select windows
        let trend_window = if config.trend_window == 0
        {
            let mut tw = (1.5 * period as f64 / (1.0 - 1.5 / period as f64)) as usize;
            if tw.is_multiple_of(2)
            {
                tw += 1;
            }
            tw.max(period + 1)
        }
        else
        {
            config.trend_window
        };

        let seasonal_window = if config.seasonal_window == 0
        {
            let mut sw = period.saturating_add(if period.is_multiple_of(2) { 1 } else { 0 });
            if sw % 2 == 0
            {
                sw += 1;
            }
            sw
        }
        else
        {
            config.seasonal_window
        };

        let mut trend = vec![0.0; n];
        let mut seasonal = vec![0.0; n];
        let mut remainder: Vec<f64> = data.to_vec();

        for _outer in 0..=config.outer_iterations
        {
            // Inner loop
            for _inner in 0..config.inner_iterations
            {
                // Step 1: Detrend — subtract current trend estimate
                let detrended: Vec<f64> = data
                    .iter()
                    .zip(trend.iter())
                    .map(|(&y, &t)| y - t)
                    .collect();

                // Step 2: Seasonal smoothing
                let cycle_subseries = extract_cycle_subseries(&detrended, period);
                let smoothed = smooth_cycle_subseries(&cycle_subseries, period, seasonal_window);
                seasonal = reconstruct_seasonal(&smoothed, n, period);

                // Step 3: Deseasonalize
                let deseasonalized: Vec<f64> = data
                    .iter()
                    .zip(seasonal.iter())
                    .map(|(&y, &s)| y - s)
                    .collect();

                // Step 4: Trend smoothing
                trend = centered_moving_average(&deseasonalized, trend_window);
                // Pad trend to full length
                pad_centered(&mut trend, n);
            }

            // Compute remainder
            remainder = data
                .iter()
                .zip(trend.iter().zip(seasonal.iter()))
                .map(|(&y, (&t, &s))| y - t - s)
                .collect();

            // Robustness weights (outer iterations)
            if config.outer_iterations > 0
            {
                let h = 6.0 * median_abs(&remainder);
                for r in remainder.iter_mut()
                {
                    if r.abs() > h
                    {
                        *r = if *r > 0.0 { h } else { -h };
                    }
                }
                // Re-weight the data for the next outer iteration
                // In a full implementation, we'd re-derive trend from weighted data
            }
        }

        STLResult {
            length: n,
            period,
            trend,
            seasonal,
            remainder,
        }
    }

    /// Extract the cycle subseries: for each position within a period,
    /// collect all values at that phase.
    fn extract_cycle_subseries(data: &[f64], period: usize) -> Vec<Vec<f64>> {
        let mut subseries = vec![Vec::new(); period];
        for (i, &val) in data.iter().enumerate()
        {
            subseries[i % period].push(val);
        }
        subseries
    }

    /// Smooth each subseries with a moving average, then center.
    fn smooth_cycle_subseries(subseries: &[Vec<f64>], period: usize, window: usize) -> Vec<f64> {
        let mut smoothed = Vec::with_capacity(period);
        for s in subseries.iter()
        {
            if s.is_empty()
            {
                smoothed.push(0.0);
                continue;
            }
            let ma = moving_average(s, window.min(s.len()));
            // Use the centered value
            let mid = ma.len() / 2;
            smoothed.push(if ma.is_empty() { 0.0 } else { ma[mid] });
        }
        // Center: subtract mean
        let m = mean(&smoothed);
        for v in smoothed.iter_mut()
        {
            *v -= m;
        }
        smoothed
    }

    /// Reconstruct full-length seasonal component from period-length smoothed values.
    fn reconstruct_seasonal(smoothed: &[f64], n: usize, period: usize) -> Vec<f64> {
        (0..n).map(|i| smoothed[i % period]).collect()
    }

    /// Median absolute deviation.
    fn median_abs(data: &[f64]) -> f64 {
        if data.is_empty()
        {
            return 0.0;
        }
        let mut sorted: Vec<f64> = data.iter().map(|x| x.abs()).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        quantile(&sorted, 0.5)
    }

    /// Extract trend component from STL result.
    pub fn extract_trend(result: &STLResult) -> &[f64] {
        &result.trend
    }

    /// Extract seasonal component from STL result.
    pub fn extract_seasonal(result: &STLResult) -> &[f64] {
        &result.seasonal
    }

    /// Compute remainder: original - trend - seasonal.
    pub fn compute_remainder(data: &[f64], result: &STLResult) -> Vec<f64> {
        data.iter()
            .zip(result.trend.iter().zip(result.seasonal.iter()))
            .map(|(&y, (&t, &s))| y - t - s)
            .collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn synthetic_seasonal(n: usize, period: usize) -> Vec<f64> {
            (0..n)
                .map(|i| {
                    let t = i as f64;
                    let trend = 0.01 * t;
                    let seasonal = (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
                    trend + seasonal
                })
                .collect()
        }

        #[test]
        fn stl_recovers_seasonal_and_trend() {
            let period = 12;
            let n = 240;
            let data = synthetic_seasonal(n, period);
            let config = STLConfig {
                period,
                ..Default::default()
            };
            let result = stl_decompose(&data, &config);

            assert_eq!(result.length, n);
            assert_eq!(result.period, period);

            // The remainder should be much smaller than the original signal
            let orig_var = variance(&data);
            let rem_var = variance(&result.remainder);
            assert!(
                rem_var < orig_var * 0.1,
                "remainder variance {} should be < 10% of original {}",
                rem_var,
                orig_var
            );
        }

        #[test]
        fn stl_trend_length_matches() {
            let period = 7;
            let n = 140;
            let data = synthetic_seasonal(n, period);
            let config = STLConfig {
                period,
                ..Default::default()
            };
            let result = stl_decompose(&data, &config);
            assert_eq!(result.trend.len(), n);
            assert_eq!(result.seasonal.len(), n);
            assert_eq!(result.remainder.len(), n);
        }

        #[test]
        #[allow(clippy::needless_range_loop)]
        fn stl_decomposition_sums_correctly() {
            let period = 12;
            let n = 120;
            let data = synthetic_seasonal(n, period);
            let config = STLConfig {
                period,
                inner_iterations: 3,
                ..Default::default()
            };
            let result = stl_decompose(&data, &config);

            for i in 0..n
            {
                let reconstructed = result.trend[i] + result.seasonal[i] + result.remainder[i];
                assert!(
                    (data[i] - reconstructed).abs() < 1e-10,
                    "decomposition doesn't sum at {}: {} vs {}",
                    i,
                    data[i],
                    reconstructed
                );
            }
        }
    }
}

pub mod detection {
    //! Seasonal period detection using autocorrelation, PACF, periodogram,
    //! and dominant frequency analysis.

    use scirust_signal::fft_real;

    use super::*;

    /// Compute the Autocorrelation Function (ACF) for lags 0..max_lag.
    ///
    /// Normalizes by the variance so that lag 0 equals 1.0.
    pub fn acf(data: &[f64], max_lag: usize) -> Vec<f64> {
        let n = data.len();
        if n < 2
        {
            return vec![1.0];
        }
        let max_lag = max_lag.min(n - 1);
        let m = mean(data);
        let var = data.iter().map(|&x| (x - m).powi(2)).sum::<f64>();
        if var < f64::EPSILON
        {
            return vec![1.0; max_lag + 1];
        }
        let centered: Vec<f64> = data.iter().map(|&x| x - m).collect();
        let mut result = Vec::with_capacity(max_lag + 1);
        for lag in 0..=max_lag
        {
            let mut sum = 0.0;
            for i in 0..n - lag
            {
                sum += centered[i] * centered[i + lag];
            }
            result.push(sum / var);
        }
        result
    }

    /// Compute the Partial Autocorrelation Function (PACF) using the
    /// Durbin-Levinson recursion.
    ///
    /// Returns PACF values for lags 1..=max_lag.
    pub fn pacf(data: &[f64], max_lag: usize) -> Vec<f64> {
        let acf_vals = acf(data, max_lag);
        let m = max_lag.min(acf_vals.len() - 1);
        if m == 0
        {
            return Vec::new();
        }

        let mut phi = vec![0.0; m + 1];
        let mut phi_prev = vec![0.0; m + 1];
        let mut pacf_vals = Vec::with_capacity(m);

        // k = 1
        phi[1] = acf_vals[1];
        pacf_vals.push(phi[1]);

        for k in 2..=m
        {
            // Compute numerator
            let mut num = acf_vals[k];
            for j in 1..k
            {
                num -= phi_prev[j] * acf_vals[k - j];
            }
            // Compute denominator
            let mut denom = 1.0;
            for j in 1..k
            {
                denom -= phi_prev[j] * acf_vals[j];
            }
            if denom.abs() < f64::EPSILON
            {
                pacf_vals.push(0.0);
                continue;
            }
            phi[k] = num / denom;
            pacf_vals.push(phi[k]);

            // Update phi_prev for next iteration
            for j in 1..k
            {
                phi[j] = phi_prev[j] - phi[k] * phi_prev[k - j];
            }
            phi_prev = phi.clone();
        }

        pacf_vals
    }

    /// Compute the periodogram (power spectrum) of the time series.
    ///
    /// Returns (frequencies, power) pairs for positive frequencies.
    /// Input is zero-padded to the next power of 2 for FFT.
    pub fn periodogram(data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = data.len();
        if n == 0
        {
            return (Vec::new(), Vec::new());
        }

        let fft_size = next_pow2(n);
        let mut padded: Vec<f64> = data.to_vec();
        padded.resize(fft_size, 0.0);

        // Remove mean
        let m = mean(data);
        for v in padded.iter_mut()
        {
            *v -= m;
        }

        let spectrum = fft_real(&padded);
        let n_f64 = fft_size as f64;
        let frequencies: Vec<f64> = (0..spectrum.len()).map(|k| k as f64 / n_f64).collect();
        let power: Vec<f64> = spectrum.iter().map(|c| c.mag_sq() / n_f64).collect();

        (frequencies, power)
    }

    /// Detect the dominant period from the periodogram.
    ///
    /// Finds the frequency bin (excluding DC) with the highest power,
    /// then returns `1/frequency` as the period.
    pub fn dominant_frequency(data: &[f64]) -> PeriodDetection {
        let (frequencies, power) = periodogram(data);
        if power.len() <= 1
        {
            return PeriodDetection {
                period: 0,
                strength: 0.0,
                candidates: Vec::new(),
            };
        }

        // Skip DC (bin 0), only look at positive frequencies up to Nyquist
        let nyquist = power.len() / 2;
        let mut candidates: Vec<(usize, f64)> = (1..=nyquist.min(power.len() - 1))
            .map(|k| (k, power[k]))
            .collect();

        // Sort by power descending
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let total_power: f64 = power[1..=nyquist.min(power.len() - 1)].iter().sum();

        let mut result_candidates = Vec::new();
        for &(k, pwr) in candidates.iter().take(5)
        {
            let freq = frequencies[k];
            let period = if freq > f64::EPSILON
            {
                (1.0 / freq).round() as usize
            }
            else
            {
                0
            };
            let strength = if total_power > f64::EPSILON
            {
                pwr / total_power
            }
            else
            {
                0.0
            };
            result_candidates.push(PeriodCandidate { period, strength });
        }

        let (dominant_k, dominant_power) = candidates[0];
        let dominant_freq = frequencies[dominant_k];
        let period = if dominant_freq > f64::EPSILON
        {
            (1.0 / dominant_freq).round() as usize
        }
        else
        {
            0
        };
        let strength = if total_power > f64::EPSILON
        {
            dominant_power / total_power
        }
        else
        {
            0.0
        };

        PeriodDetection {
            period,
            strength,
            candidates: result_candidates,
        }
    }

    /// Detect period using ACF peaks.
    ///
    /// Finds the first significant peak in the ACF, which corresponds
    /// to the dominant seasonal period.
    pub fn detect_period_acf(data: &[f64], max_lag: Option<usize>) -> PeriodDetection {
        let n = data.len();
        let max_lag = max_lag.unwrap_or(n / 2);
        let acf_vals = acf(data, max_lag);

        if acf_vals.len() < 3
        {
            return PeriodDetection {
                period: 0,
                strength: 0.0,
                candidates: Vec::new(),
            };
        }

        // Find peaks in ACF (local maxima above significance threshold)
        let threshold = 1.96 / (n as f64).sqrt(); // 95% confidence band
        let mut peaks: Vec<(usize, f64)> = Vec::new();

        for i in 1..acf_vals.len() - 1
        {
            if acf_vals[i] > acf_vals[i - 1]
                && acf_vals[i] > acf_vals[i + 1]
                && acf_vals[i] > threshold
            {
                peaks.push((i, acf_vals[i]));
            }
        }

        // Sort by ACF value
        peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let candidates: Vec<PeriodCandidate> = peaks
            .iter()
            .take(5)
            .map(|&(lag, val)| PeriodCandidate {
                period: lag,
                strength: val.abs(),
            })
            .collect();

        let (period, strength) = if let Some(&(lag, val)) = peaks.first()
        {
            (lag, val.abs())
        }
        else
        {
            (0, 0.0)
        };

        PeriodDetection {
            period,
            strength,
            candidates,
        }
    }

    /// Combined period detection: uses both ACF and periodogram,
    /// returns the consensus result.
    pub fn detect_period(data: &[f64]) -> PeriodDetection {
        let acf_result = detect_period_acf(data, None);
        let freq_result = dominant_frequency(data);

        // If both agree (within tolerance), high confidence
        if acf_result.period > 0 && freq_result.period > 0
        {
            let diff = (acf_result.period as f64 - freq_result.period as f64).abs();
            let tol = (acf_result.period as f64 * 0.1).max(1.0);
            if diff <= tol
            {
                return PeriodDetection {
                    period: ((acf_result.period + freq_result.period) / 2).max(1),
                    strength: ((acf_result.strength + freq_result.strength) / 2.0).min(1.0),
                    candidates: acf_result.candidates,
                };
            }
        }

        // Prefer ACF result if it has higher confidence
        if acf_result.strength >= freq_result.strength
        {
            acf_result
        }
        else
        {
            freq_result
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn sine_wave(n: usize, period: usize) -> Vec<f64> {
            (0..n)
                .map(|i| (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin())
                .collect()
        }

        #[test]
        fn acf_detects_period() {
            let n = 200;
            let period = 12;
            let data = sine_wave(n, period);
            let acf_vals = acf(&data, n / 2);

            // ACF at lag = period should be near 1.0
            assert!(
                acf_vals[period] > 0.9,
                "ACF at lag {} should be ~1, got {}",
                period,
                acf_vals[period]
            );
        }

        #[test]
        fn pacf_detects_autoregressive() {
            // AR(1) process: x[t] = 0.8*x[t-1] + noise
            let n = 500;
            let mut data = vec![0.0; n];
            let mut rng_state: u64 = 42;
            for i in 1..n
            {
                // Simple LCG pseudo-random
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2.0;
                data[i] = 0.8 * data[i - 1] + noise;
            }
            let pacf_vals = pacf(&data, 10);
            // PACF at lag 1 should be near 0.8
            assert!(
                (pacf_vals[0] - 0.8).abs() < 0.15,
                "PACF at lag 1 should be ~0.8, got {}",
                pacf_vals[0]
            );
            // PACF at lags > 1 should be small
            for &v in &pacf_vals[2..]
            {
                assert!(
                    v.abs() < 0.2,
                    "PACF at higher lag should be small, got {}",
                    v
                );
            }
        }

        #[test]
        fn periodogram_finds_sine_period() {
            let n = 256;
            let period = 16;
            let data = sine_wave(n, period);
            let result = dominant_frequency(&data);
            assert!(
                (result.period as isize - period as isize).abs() <= 1,
                "periodogram should detect period ~{}, got {}",
                period,
                result.period
            );
        }

        #[test]
        fn combined_detect_period() {
            let n = 240;
            let period = 24;
            let data: Vec<f64> = (0..n)
                .map(|i| {
                    (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
                        + 0.5 * (4.0 * std::f64::consts::PI * i as f64 / period as f64).sin()
                })
                .collect();
            let result = detect_period(&data);
            assert!(
                (result.period as isize - period as isize).abs() <= 2,
                "combined detection should find period ~{}, got {}",
                period,
                result.period
            );
        }
    }
}

pub mod cyclic {
    //! Cyclic pattern detection: Fourier analysis, windowed FFT (STFT),
    //! cycle length estimation, and phase detection.

    use scirust_signal::{Complex, fft_real, hanning};

    use super::*;

    /// Perform Fourier analysis to find dominant cyclic components.
    ///
    /// Returns the `CyclicPattern` with detected harmonics, amplitude, and phase.
    pub fn fourier_analysis(data: &[f64]) -> CyclicPattern {
        let n = data.len();
        if n < 4
        {
            return CyclicPattern {
                cycle_length: n as f64,
                phase: 0.0,
                amplitude: 0.0,
                harmonics: Vec::new(),
            };
        }

        let fft_size = next_pow2(n);
        let mut padded: Vec<f64> = data.to_vec();
        padded.resize(fft_size, 0.0);

        // Remove mean
        let m = mean(&padded);
        for v in padded.iter_mut()
        {
            *v -= m;
        }

        let spectrum = fft_real(&padded);
        let n_f64 = fft_size as f64;

        // Find dominant frequency (excluding DC)
        let nyquist = spectrum.len() / 2;
        let mut max_power = 0.0;
        let mut max_bin = 0usize;
        #[allow(clippy::needless_range_loop)]
        for k in 1..=nyquist
        {
            let power = spectrum[k].mag_sq();
            if power > max_power
            {
                max_power = power;
                max_bin = k;
            }
        }

        let amplitude = if max_bin < spectrum.len()
        {
            2.0 * spectrum[max_bin].mag() / n_f64
        }
        else
        {
            0.0
        };
        let phase = if max_bin < spectrum.len()
        {
            spectrum[max_bin].phase()
        }
        else
        {
            0.0
        };
        let cycle_length = if max_bin > 0
        {
            n_f64 / max_bin as f64
        }
        else
        {
            n as f64
        };

        // Extract harmonics
        let mut harmonics = Vec::new();
        let power_threshold = max_power * 0.05; // 5% of dominant
        #[allow(clippy::needless_range_loop)]
        for k in 1..=nyquist
        {
            let power = spectrum[k].mag_sq();
            if power > power_threshold
            {
                let freq = k as f64 / n_f64;
                harmonics.push(Harmonic {
                    frequency_index: k,
                    frequency: freq,
                    amplitude: 2.0 * spectrum[k].mag() / n_f64,
                    phase: spectrum[k].phase(),
                });
            }
        }
        harmonics.sort_by(|a, b| {
            b.amplitude
                .partial_cmp(&a.amplitude)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        CyclicPattern {
            cycle_length,
            phase,
            amplitude,
            harmonics,
        }
    }

    /// Windowed FFT (Short-Time Fourier Transform) for time-varying frequency analysis.
    ///
    /// Returns a matrix of complex spectra: `result[window_idx][freq_bin]`.
    /// Each window is multiplied by a Hanning window before FFT.
    pub fn windowed_fft(data: &[f64], window_size: usize, hop_size: usize) -> Vec<Vec<Complex>> {
        if data.len() < window_size || window_size == 0
        {
            return Vec::new();
        }

        let fft_size = next_pow2(window_size);
        let window = hanning(window_size);
        let num_windows = (data.len() - window_size) / hop_size + 1;
        let mut result = Vec::with_capacity(num_windows);

        for w in 0..num_windows
        {
            let start = w * hop_size;
            let mut segment: Vec<f64> = data[start..start + window_size]
                .iter()
                .zip(window.iter())
                .map(|(&x, &win)| x * win)
                .collect();
            segment.resize(fft_size, 0.0);

            // Remove mean
            let m = mean(&segment);
            for v in segment.iter_mut()
            {
                *v -= m;
            }

            let spectrum = fft_real(&segment);
            result.push(spectrum);
        }

        result
    }

    /// Estimate cycle length from the time series using zero-crossings and
    /// peak-to-peak analysis.
    pub fn estimate_cycle_length(data: &[f64]) -> f64 {
        if data.len() < 3
        {
            return data.len() as f64;
        }

        let m = mean(data);
        let centered: Vec<f64> = data.iter().map(|&x| x - m).collect();

        // Find zero crossings
        let mut crossings = Vec::new();
        for i in 1..centered.len()
        {
            if (centered[i - 1] >= 0.0 && centered[i] < 0.0)
                || (centered[i - 1] < 0.0 && centered[i] >= 0.0)
            {
                // Linear interpolation for precise crossing point
                let t = centered[i - 1] / (centered[i - 1] - centered[i]);
                crossings.push((i - 1) as f64 + t);
            }
        }

        if crossings.len() < 2
        {
            // Fallback: use peak-to-peak
            let mut peaks = Vec::new();
            for i in 1..centered.len() - 1
            {
                if centered[i] > centered[i - 1] && centered[i] > centered[i + 1]
                {
                    peaks.push(i);
                }
            }
            if peaks.len() >= 2
            {
                let diffs: Vec<f64> = peaks.windows(2).map(|w| (w[1] - w[0]) as f64).collect();
                return mean(&diffs);
            }
            return data.len() as f64;
        }

        // Average half-period from consecutive crossings
        let half_periods: Vec<f64> = crossings.windows(2).map(|w| w[1] - w[0]).collect();
        let avg_half = mean(&half_periods);
        2.0 * avg_half
    }

    /// Detect the phase of a known period in the time series.
    ///
    /// Returns phase in radians [0, 2π).
    pub fn detect_phase(data: &[f64], period: usize) -> f64 {
        if data.len() < period || period == 0
        {
            return 0.0;
        }

        // Compute phase via correlation with sine and cosine
        let mut sin_sum = 0.0;
        let mut cos_sum = 0.0;
        for (i, &val) in data.iter().enumerate()
        {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / period as f64;
            sin_sum += val * angle.sin();
            cos_sum += val * angle.cos();
        }

        let phase = sin_sum.atan2(cos_sum);
        // Normalize to [0, 2π)
        if phase < 0.0
        {
            phase + 2.0 * std::f64::consts::PI
        }
        else
        {
            phase
        }
    }

    /// Detect phase using FFT for higher accuracy.
    pub fn detect_phase_fft(data: &[f64], period: usize) -> f64 {
        if data.len() < period || period == 0
        {
            return 0.0;
        }

        let fft_size = next_pow2(data.len());
        let mut padded: Vec<f64> = data.to_vec();
        padded.resize(fft_size, 0.0);

        let m = mean(data);
        for v in padded.iter_mut()
        {
            *v -= m;
        }

        let spectrum = fft_real(&padded);
        let bin = (fft_size as f64 / period as f64).round() as usize;

        if bin > 0 && bin < spectrum.len()
        {
            let phase = spectrum[bin].phase();
            if phase < 0.0
            {
                phase + 2.0 * std::f64::consts::PI
            }
            else
            {
                phase
            }
        }
        else
        {
            0.0
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn sine_wave(n: usize, period: usize) -> Vec<f64> {
            (0..n)
                .map(|i| (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin())
                .collect()
        }

        #[test]
        fn fourier_finds_cycle_length() {
            let n = 256;
            let period = 20;
            let data = sine_wave(n, period);
            let pattern = fourier_analysis(&data);
            assert!(
                (pattern.cycle_length - period as f64).abs() < 2.0,
                "expected cycle_length ~{}, got {}",
                period,
                pattern.cycle_length
            );
            assert!(
                pattern.amplitude > 0.5,
                "amplitude should be ~1.0, got {}",
                pattern.amplitude
            );
        }

        #[test]
        fn windowed_fft_output_shape() {
            let n = 256;
            let data = sine_wave(n, 16);
            let result = windowed_fft(&data, 64, 32);
            assert!(!result.is_empty());
            // fft_real returns N/2+1 positive-frequency bins
            assert_eq!(result[0].len(), 33);
        }

        #[test]
        fn cycle_length_estimation() {
            let n = 200;
            let period = 24;
            let data = sine_wave(n, period);
            let est = estimate_cycle_length(&data);
            assert!(
                (est - period as f64).abs() < 3.0,
                "expected ~{}, got {}",
                period,
                est
            );
        }

        #[test]
        fn phase_detection_sine() {
            // cos wave: cos(x) = sin(x + π/2). The DFT phase at the fundamental
            // frequency for a pure cosine is 0 (since it aligns with the cos basis).
            let n = 128;
            let period = 32;
            let data: Vec<f64> = (0..n)
                .map(|i| (2.0 * std::f64::consts::PI * i as f64 / period as f64).cos())
                .collect();
            let phase = detect_phase(&data, period);
            // For cos(x), the DFT coefficient is real and positive → phase ≈ 0
            assert!(
                phase.abs() < 0.3 || (phase - 2.0 * std::f64::consts::PI).abs() < 0.3,
                "expected phase ~0 or ~2π for cosine, got {}",
                phase
            );
        }
    }
}

pub mod adjustment {
    //! Seasonal adjustment methods: moving-average decomposition, X-11 style
    //! adjustment, and deseasonalization.

    use super::*;

    /// Seasonal adjustment using moving-average decomposition.
    ///
    /// Decomposes `y = trend * seasonal * irregular` (multiplicative) or
    /// `y = trend + seasonal + irregular` (additive).
    pub fn moving_average_adjustment(data: &[f64], period: usize, additive: bool) -> Vec<f64> {
        let n = data.len();
        if n < 2 * period
        {
            return data.to_vec();
        }

        // Estimate trend via centered MA
        let trend = centered_moving_average(data, period);
        let mut trend_full = trend.clone();
        pad_centered(&mut trend_full, n);

        if additive
        {
            // Remove trend
            let detrended: Vec<f64> = data
                .iter()
                .zip(trend_full.iter())
                .map(|(&y, &t)| y - t)
                .collect();
            // Compute seasonal factors
            let factors = compute_seasonal_factors_additive(&detrended, period);
            // Remove seasonal
            data.iter()
                .zip(factors.iter().cycle())
                .map(|(&y, &s)| y - s)
                .collect()
        }
        else
        {
            // Multiplicative
            let detrended: Vec<f64> = data
                .iter()
                .zip(trend_full.iter())
                .map(|(&y, &t)| if t.abs() > f64::EPSILON { y / t } else { 1.0 })
                .collect();
            let factors = compute_seasonal_factors_multiplicative(&detrended, period);
            data.iter()
                .zip(factors.iter().cycle())
                .map(|(&y, &s)| if s.abs() > f64::EPSILON { y / s } else { y })
                .collect()
        }
    }

    /// X-11 style seasonal adjustment.
    ///
    /// Iterative procedure: estimate trend, compute seasonal factors,
    /// smooth factors, and remove. Typically converges in 2-3 iterations.
    pub fn x11_adjustment(data: &[f64], period: usize) -> Vec<f64> {
        let n = data.len();
        if n < 2 * period
        {
            return data.to_vec();
        }

        let mut current = data.to_vec();
        let iterations = 3;

        for _ in 0..iterations
        {
            // Step 1: Estimate trend (centered 2×period MA for even period)
            let trend_ma = if period.is_multiple_of(2)
            {
                let ma1 = moving_average(&current, period);
                let ma2 = moving_average(&ma1, 2);
                // Re-center
                moving_average(&ma2, 2)
            }
            else
            {
                moving_average(&current, period)
            };
            let mut trend = trend_ma.clone();
            pad_centered(&mut trend, n);

            // Step 2: Detrend (multiplicative)
            let detrended: Vec<f64> = current
                .iter()
                .zip(trend.iter())
                .map(|(&y, &t)| if t.abs() > f64::EPSILON { y / t } else { 1.0 })
                .collect();

            // Step 3: Compute unsmoothed seasonal factors
            let mut seasonal = vec![0.0; n];
            #[allow(clippy::needless_range_loop)]
            for i in 0..n
            {
                let phase = i % period;
                // Collect all values at this phase
                let mut phase_vals: Vec<f64> = detrended
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| j % period == phase)
                    .map(|(_, &v)| v)
                    .collect();
                phase_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                seasonal[i] = trimmed_mean(&phase_vals, phase_vals.len() / 4);
            }

            // Step 4: Normalize seasonal factors (each cycle sums to period)
            for cycle_start in (0..n).step_by(period)
            {
                let cycle_end = (cycle_start + period).min(n);
                let cycle_sum: f64 = seasonal[cycle_start..cycle_end].iter().sum();
                let adjustment = cycle_sum / period as f64;
                for s in seasonal[cycle_start..cycle_end].iter_mut()
                {
                    *s -= adjustment;
                }
            }

            // Step 5: Smooth seasonal factors with 3×period MA
            let smooth_window = 3 * period;
            if smooth_window < n
            {
                let smoothed = moving_average(&seasonal, smooth_window);
                let mut seasonal_smooth = vec![0.0; n];
                // Map smoothed back to seasonal positions
                for (i, &s) in smoothed.iter().enumerate()
                {
                    let base_idx = i + smooth_window / 2;
                    if base_idx < n
                    {
                        seasonal_smooth[base_idx] = s;
                    }
                }
                // Fill gaps with nearest
                let mut last = 0.0;
                for s in seasonal_smooth.iter_mut()
                {
                    if *s == 0.0
                    {
                        *s = last;
                    }
                    else
                    {
                        last = *s;
                    }
                }
                seasonal = seasonal_smooth;
            }

            // Step 6: Deseasonalize
            current = current
                .iter()
                .zip(seasonal.iter())
                .map(|(&y, &s)| {
                    if (1.0 + s).abs() > f64::EPSILON
                    {
                        y / (1.0 + s)
                    }
                    else
                    {
                        y
                    }
                })
                .collect();
        }

        current
    }

    /// Deseasonalize a time series by subtracting (additive) or dividing
    /// (multiplicative) the estimated seasonal component.
    pub fn deseasonalize(data: &[f64], period: usize, additive: bool) -> Vec<f64> {
        let n = data.len();
        if n < 2 * period
        {
            return data.to_vec();
        }

        // Estimate seasonal component
        let trend = centered_moving_average(data, period);
        let mut trend_full = trend.clone();
        pad_centered(&mut trend_full, n);

        if additive
        {
            let detrended: Vec<f64> = data
                .iter()
                .zip(trend_full.iter())
                .map(|(&y, &t)| y - t)
                .collect();
            let seasonal = compute_seasonal_factors_additive(&detrended, period);
            data.iter()
                .zip(seasonal.iter().cycle())
                .map(|(&y, &s)| y - s)
                .collect()
        }
        else
        {
            let detrended: Vec<f64> = data
                .iter()
                .zip(trend_full.iter())
                .map(|(&y, &t)| if t.abs() > f64::EPSILON { y / t } else { 1.0 })
                .collect();
            let seasonal = compute_seasonal_factors_multiplicative(&detrended, period);
            data.iter()
                .zip(seasonal.iter().cycle())
                .map(|(&y, &s)| if s.abs() > f64::EPSILON { y / s } else { y })
                .collect()
        }
    }

    fn compute_seasonal_factors_additive(detrended: &[f64], period: usize) -> Vec<f64> {
        let mut factors = vec![0.0; period];
        #[allow(clippy::needless_range_loop)]
        for p in 0..period
        {
            let vals: Vec<f64> = detrended
                .iter()
                .enumerate()
                .filter(|(i, _)| i % period == p)
                .map(|(_, &v)| v)
                .collect();
            factors[p] = mean(&vals);
        }
        // Normalize: subtract mean of factors
        let m = mean(&factors);
        for f in factors.iter_mut()
        {
            *f -= m;
        }
        factors
    }

    fn compute_seasonal_factors_multiplicative(detrended: &[f64], period: usize) -> Vec<f64> {
        let mut factors = vec![1.0; period];
        #[allow(clippy::needless_range_loop)]
        for p in 0..period
        {
            let vals: Vec<f64> = detrended
                .iter()
                .enumerate()
                .filter(|(i, _)| i % period == p)
                .map(|(_, &v)| v)
                .collect();
            factors[p] = mean(&vals);
        }
        // Normalize: divide by mean of factors
        let m = mean(&factors);
        if m.abs() > f64::EPSILON
        {
            for f in factors.iter_mut()
            {
                *f /= m;
            }
        }
        factors
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn seasonal_data(n: usize, period: usize) -> Vec<f64> {
            (0..n)
                .map(|i| {
                    let trend = 100.0 + 0.1 * i as f64;
                    let seasonal =
                        5.0 * (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
                    trend + seasonal
                })
                .collect()
        }

        #[test]
        fn ma_adjustment_reduces_seasonality() {
            let period = 12;
            let n = 240;
            let data = seasonal_data(n, period);
            let adjusted = moving_average_adjustment(&data, period, true);

            // Compute variance of seasonal component before/after
            let seasonal_before: Vec<f64> = data
                .iter()
                .enumerate()
                .map(|(i, &y)| {
                    let t = 100.0 + 0.1 * i as f64;
                    y - t
                })
                .collect();
            let seasonal_after: Vec<f64> = adjusted
                .iter()
                .enumerate()
                .map(|(i, &y)| {
                    let t = 100.0 + 0.1 * i as f64;
                    y - t
                })
                .collect();

            let var_before = variance(&seasonal_before);
            let var_after = variance(&seasonal_after);
            assert!(
                var_after < var_before * 0.5,
                "adjustment should reduce seasonal variance: {} vs {}",
                var_before,
                var_after
            );
        }

        #[test]
        fn x11_produces_valid_output() {
            let period = 12;
            let n = 240;
            let data = seasonal_data(n, period);
            let adjusted = x11_adjustment(&data, period);
            assert_eq!(adjusted.len(), n);
            // Adjusted should have lower variance in seasonal component
            for &v in &adjusted
            {
                assert!(v.is_finite(), "adjusted values must be finite");
            }
        }

        #[test]
        fn deseasonalize_removes_seasonal() {
            let period = 12;
            let n = 240;
            let data = seasonal_data(n, period);
            let adjusted = deseasonalize(&data, period, true);
            // The deseasonalized series should be smoother
            let var_original = variance(&data);
            let var_adjusted = variance(&adjusted);
            assert!(
                var_adjusted < var_original,
                "deseasonalized should be smoother: {} vs {}",
                var_adjusted,
                var_original
            );
        }
    }
}

pub mod trend {
    //! Trend detection: Mann-Kendall test, Sen's slope estimator,
    //! and linear trend with confidence intervals.

    use super::*;

    /// Perform the Mann-Kendall trend test.
    ///
    /// Returns the S statistic, Z statistic, and p-value.
    /// S > 0 indicates increasing trend, S < 0 indicates decreasing.
    pub fn mann_kendall(data: &[f64]) -> (f64, f64, f64) {
        let n = data.len();
        if n < 3
        {
            return (0.0, 0.0, 1.0);
        }

        let mut s = 0.0;
        for i in 0..n - 1
        {
            for j in i + 1..n
            {
                let diff = data[j] - data[i];
                if diff > 0.0
                {
                    s += 1.0;
                }
                else if diff < 0.0
                {
                    s -= 1.0;
                }
            }
        }

        // Variance of S (without ties correction for simplicity)
        let var_s = (n * (n - 1) * (2 * n + 5)) as f64 / 18.0;

        // Z statistic
        let z = if s > 0.0
        {
            (s - 1.0) / var_s.sqrt()
        }
        else if s < 0.0
        {
            (s + 1.0) / var_s.sqrt()
        }
        else
        {
            0.0
        };

        // Two-tailed p-value from normal distribution (approximation)
        let p = 2.0 * (1.0 - normal_cdf(z.abs()));

        (s, z, p)
    }

    /// Sen's slope estimator: median of all pairwise slopes.
    pub fn sens_slope(data: &[f64]) -> f64 {
        let n = data.len();
        if n < 2
        {
            return 0.0;
        }

        let mut slopes = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n - 1
        {
            for j in i + 1..n
            {
                let dx = (j - i) as f64;
                if dx > f64::EPSILON
                {
                    slopes.push((data[j] - data[i]) / dx);
                }
            }
        }

        if slopes.is_empty()
        {
            return 0.0;
        }

        slopes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        quantile(&slopes, 0.5)
    }

    /// Complete trend analysis with Mann-Kendall test, Sen's slope,
    /// linear regression, and 95% confidence intervals.
    pub fn trend_analysis(data: &[f64]) -> TrendResult {
        let n = data.len();
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();

        let (slope, intercept) = linear_regression(&x, data);
        let (s_stat, z_stat, p_val) = mann_kendall(data);
        let s_slope = sens_slope(data);

        // Confidence interval for slope via bootstrap-like approach
        // Using residual-based variance estimate
        let residuals: Vec<f64> = data
            .iter()
            .enumerate()
            .map(|(i, &y)| y - (slope * i as f64 + intercept))
            .collect();
        let residual_var = variance(&residuals);
        let x_var: f64 = x.iter().map(|&v| (v - mean(&x)).powi(2)).sum();
        let se_slope = if x_var > f64::EPSILON
        {
            (residual_var / x_var).sqrt()
        }
        else
        {
            0.0
        };

        // t-distribution critical value (approximation for df = n-2)
        let t_crit = if n > 2
        {
            // Approximate t critical value for 95% CI
            let df = n as f64 - 2.0;
            1.96 + 1.0 / df // rough approximation
        }
        else
        {
            1.96
        };

        let ci_lower = slope - t_crit * se_slope;
        let ci_upper = slope + t_crit * se_slope;

        let direction = if p_val < 0.05
        {
            if slope > 0.0
            {
                TrendDirection::Increasing
            }
            else
            {
                TrendDirection::Decreasing
            }
        }
        else
        {
            TrendDirection::NoTrend
        };

        TrendResult {
            slope,
            intercept,
            p_value: p_val,
            s_statistic: s_stat,
            z_statistic: z_stat,
            sens_slope: s_slope,
            slope_ci: [ci_lower, ci_upper],
            significant: p_val < 0.05,
            direction,
        }
    }

    /// Cumulative sum (CUSUM) for trend detection.
    ///
    /// Returns the cumulative sum of deviations from the mean.
    pub fn cusum(data: &[f64]) -> Vec<f64> {
        let m = mean(data);
        let mut result = Vec::with_capacity(data.len());
        let mut sum = 0.0;
        for &val in data
        {
            sum += val - m;
            result.push(sum);
        }
        result
    }

    /// Approximate CDF of standard normal distribution.
    fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
    }

    /// Error function approximation (Abramowitz & Stegun).
    fn erf(x: f64) -> f64 {
        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + 0.3275911 * x);
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t3 * t;
        let t5 = t4 * t;
        let poly = 0.254829592 * t - 0.284496736 * t2 + 1.421413741 * t3 - 1.453152027 * t4
            + 1.061405429 * t5;
        sign * (1.0 - poly * (-x * x).exp())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn increasing_trend() {
            let data: Vec<f64> = (0..100)
                .map(|i| i as f64 + 0.5 * (i as f64).sin())
                .collect();
            let result = trend_analysis(&data);
            assert!(
                result.slope > 0.0,
                "slope should be positive, got {}",
                result.slope
            );
            assert!(
                result.slope > 0.8,
                "slope should be ~1.0, got {}",
                result.slope
            );
            assert!(result.significant, "should be significant");
            assert_eq!(result.direction, TrendDirection::Increasing);
        }

        #[test]
        fn decreasing_trend() {
            let data: Vec<f64> = (0..100)
                .map(|i| 100.0 - i as f64 + 0.5 * (i as f64).sin())
                .collect();
            let result = trend_analysis(&data);
            assert!(result.slope < 0.0, "slope should be negative");
            assert!(result.significant, "should be significant");
            assert_eq!(result.direction, TrendDirection::Decreasing);
        }

        #[test]
        fn no_trend() {
            let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
            let result = trend_analysis(&data);
            assert!(
                result.slope.abs() < 0.1,
                "slope should be near zero, got {}",
                result.slope
            );
        }

        #[test]
        fn sens_slope_matches_linear() {
            let data: Vec<f64> = (0..50).map(|i| 2.0 * i as f64).collect();
            let result = trend_analysis(&data);
            assert!(
                (result.sens_slope - 2.0).abs() < 0.01,
                "Sen's slope should be ~2.0, got {}",
                result.sens_slope
            );
        }

        #[test]
        #[allow(clippy::needless_range_loop)]
        fn cusum_detects_shift() {
            let mut data = vec![0.0; 100];
            for i in 50..100
            {
                data[i] = 1.0;
            }
            let cs = cusum(&data);
            // CUSUM should increase after index 50
            assert!(cs[99] > cs[49], "CUSUM should detect the shift");
        }
    }
}

pub mod changepoint {
    //! Change-point detection with seasonality: seasonal CUSUM and
    //! seasonal break detection.

    use super::*;

    /// Seasonal CUSUM (Cumulative Sum) control chart.
    ///
    /// Detects changes in the seasonal pattern by computing separate
    /// CUSUM statistics for each phase within the period.
    pub fn seasonal_cusum(data: &[f64], period: usize, threshold: f64) -> Vec<ChangePoint> {
        let n = data.len();
        if n < 2 * period
        {
            return Vec::new();
        }

        // Estimate seasonal baseline
        let trend = centered_moving_average(data, period);
        let mut trend_full = trend.clone();
        pad_centered(&mut trend_full, n);

        let deseasonalized: Vec<f64> = data
            .iter()
            .zip(trend_full.iter())
            .map(|(&y, &t)| y - t)
            .collect();

        // Compute baseline mean for each phase
        let mut phase_means = vec![0.0; period];
        let mut phase_counts = vec![0usize; period];
        for (i, &val) in deseasonalized.iter().enumerate()
        {
            let phase = i % period;
            phase_means[phase] += val;
            phase_counts[phase] += 1;
        }
        for p in 0..period
        {
            if phase_counts[p] > 0
            {
                phase_means[p] /= phase_counts[p] as f64;
            }
        }

        // Compute CUSUM for each phase
        let mut cusum_vals = vec![0.0f64; n];
        for i in 0..n
        {
            let phase = i % period;
            let deviation = deseasonalized[i] - phase_means[phase];
            cusum_vals[i] = if i == 0
            {
                deviation
            }
            else
            {
                cusum_vals[i - 1] + deviation
            };
        }

        // Detect change points: where CUSUM exceeds threshold
        let mut change_points = Vec::new();
        let mut last_cp = 0usize;
        let mut max_abs = 0.0f64;
        let mut max_idx = 0usize;

        for i in 1..n
        {
            if (cusum_vals[i] - cusum_vals[last_cp]).abs() > max_abs
            {
                max_abs = (cusum_vals[i] - cusum_vals[last_cp]).abs();
                max_idx = i;
            }
            if (cusum_vals[i] - cusum_vals[last_cp]).abs() > threshold
            {
                let seasonal_comp = phase_means[max_idx % period];
                change_points.push(ChangePoint {
                    index: max_idx,
                    magnitude: max_abs,
                    seasonal_component: seasonal_comp,
                });
                last_cp = max_idx;
                max_abs = 0.0;
                max_idx = i;
            }
        }

        change_points
    }

    /// Detect seasonal breaks using moving-window variance comparison.
    ///
    /// Compares variance in a left window vs right window at each position.
    /// A significant ratio indicates a change point.
    pub fn seasonal_break_detection(
        data: &[f64],
        period: usize,
        window_size: usize,
        significance: f64,
    ) -> Vec<ChangePoint> {
        let n = data.len();
        let half_win = window_size / 2;
        if n < 2 * window_size + period
        {
            return Vec::new();
        }

        // Deseasonalize first
        let adjusted = super::adjustment::deseasonalize(data, period, true);

        let mut change_points = Vec::new();
        let mut last_cp = 0usize;

        for i in half_win + period..n - half_win - period
        {
            // Skip if too close to last change point
            if i - last_cp < period
            {
                continue;
            }

            let left = &adjusted[i - half_win..i];
            let right = &adjusted[i..i + half_win];

            let var_left = variance(left);
            let var_right = variance(right);

            // F-test for variance ratio
            let f_stat = if var_right > f64::EPSILON
            {
                var_left / var_right
            }
            else
            {
                f64::INFINITY
            };

            // Also check mean shift
            let mean_left = mean(left);
            let mean_right = mean(right);
            let pooled_std = ((var_left + var_right) / 2.0).sqrt();
            let t_stat = if pooled_std > f64::EPSILON
            {
                (mean_left - mean_right).abs() / (pooled_std * (2.0 / half_win as f64).sqrt())
            }
            else
            {
                0.0
            };

            // Combined criterion: significant mean shift or variance change
            let critical_t = 1.96 + significance; // rough threshold
            if t_stat > critical_t || (f_stat > 2.0 || f_stat < 0.5)
            {
                let magnitude = t_stat.max((f_stat - 1.0).abs());
                change_points.push(ChangePoint {
                    index: i,
                    magnitude,
                    seasonal_component: 0.0,
                });
                last_cp = i;
            }
        }

        // Merge nearby change points (within one period)
        merge_nearby(&mut change_points, period);
        change_points
    }

    /// Binary segmentation for seasonal change points.
    ///
    /// Recursively splits the series at the point of maximum likelihood ratio.
    pub fn seasonal_binary_segmentation(
        data: &[f64],
        period: usize,
        min_segment: usize,
        max_segments: usize,
    ) -> Vec<ChangePoint> {
        let n = data.len();
        if n < 2 * min_segment
        {
            return Vec::new();
        }

        let mut change_points = Vec::new();
        let mut segments: Vec<(usize, usize)> = vec![(0, n)];

        while change_points.len() < max_segments && !segments.is_empty()
        {
            let mut best_gain = 0.0f64;
            let mut best_segment_idx = 0usize;
            let mut best_split = 0usize;

            for (seg_idx, &(start, end)) in segments.iter().enumerate()
            {
                let seg_len = end - start;
                if seg_len < 2 * min_segment
                {
                    continue;
                }

                let segment = &data[start..end];
                let total_var = variance(segment);

                // Try each split point
                for split in (start + min_segment..end - min_segment).step_by(period.max(1))
                {
                    let left = &data[start..split];
                    let right = &data[split..end];

                    let left_var = variance(left);
                    let right_var = variance(right);
                    let left_w = left.len() as f64 / seg_len as f64;
                    let right_w = right.len() as f64 / seg_len as f64;

                    let pooled_var = left_w * left_var + right_w * right_var;
                    let gain = total_var - pooled_var;

                    if gain > best_gain
                    {
                        best_gain = gain;
                        best_segment_idx = seg_idx;
                        best_split = split;
                    }
                }
            }

            if best_gain > f64::EPSILON
            {
                let (start, end) = segments[best_segment_idx];
                segments.remove(best_segment_idx);
                segments.push((start, best_split));
                segments.push((best_split, end));

                change_points.push(ChangePoint {
                    index: best_split,
                    magnitude: best_gain,
                    seasonal_component: 0.0,
                });
            }
            else
            {
                break;
            }
        }

        change_points.sort_by_key(|cp| cp.index);
        change_points
    }

    /// Merge change points that are closer than `min_distance` apart.
    fn merge_nearby(change_points: &mut Vec<ChangePoint>, min_distance: usize) {
        if change_points.is_empty()
        {
            return;
        }
        change_points.sort_by_key(|cp| cp.index);
        let mut merged = vec![change_points[0].clone()];
        for cp in change_points.iter().skip(1)
        {
            let last = merged.last().unwrap();
            if cp.index - last.index >= min_distance
            {
                merged.push(cp.clone());
            }
            else if cp.magnitude > last.magnitude
            {
                *merged.last_mut().unwrap() = cp.clone();
            }
        }
        *change_points = merged;
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        #[allow(clippy::needless_range_loop)]
        fn cusum_detects_level_shift() {
            let mut data = vec![0.0; 200];
            // Add seasonal pattern
            for i in 0..200
            {
                data[i] = (2.0 * std::f64::consts::PI * i as f64 / 12.0).sin();
            }
            // Shift mean at index 100
            for i in 100..200
            {
                data[i] += 3.0;
            }

            let cps = seasonal_cusum(&data, 12, 3.0);
            assert!(!cps.is_empty(), "should detect at least one change point");
            // The change point should be near index 100
            let closest = cps
                .iter()
                .min_by_key(|cp| (cp.index as isize - 100).abs())
                .unwrap();
            assert!(
                (closest.index as isize - 100).abs() < 20,
                "change point should be near 100, got {}",
                closest.index
            );
        }

        #[test]
        #[allow(clippy::needless_range_loop)]
        fn break_detection_finds_shift() {
            let mut data = vec![0.0; 240];
            let period = 12;
            for i in 0..240
            {
                data[i] = (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
            }
            // Large shift at index 120
            for i in 120..240
            {
                data[i] += 5.0;
            }

            let cps = seasonal_break_detection(&data, period, 48, 0.0);
            assert!(!cps.is_empty(), "should detect at least one break");
            let closest = cps
                .iter()
                .min_by_key(|cp| (cp.index as isize - 120).abs())
                .unwrap();
            assert!(
                (closest.index as isize - 120).abs() < 20,
                "break should be near 120, got {}",
                closest.index
            );
        }

        #[test]
        #[allow(clippy::needless_range_loop)]
        fn binary_segmentation_finds_multiple() {
            let mut data = vec![0.0; 360];
            let period = 12;
            for i in 0..360
            {
                data[i] = (2.0 * std::f64::consts::PI * i as f64 / period as f64).sin();
            }
            // Two shifts
            for i in 120..240
            {
                data[i] += 3.0;
            }
            for i in 240..360
            {
                data[i] += 6.0;
            }

            let cps = seasonal_binary_segmentation(&data, period, 30, 5);
            assert!(
                cps.len() >= 2,
                "should find at least 2 change points, found {}",
                cps.len()
            );
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::adjustment::deseasonalize;
    use super::changepoint::seasonal_cusum;
    use super::cyclic::fourier_analysis;
    use super::detection::detect_period;
    use super::stl::STLConfig;
    use super::stl::stl_decompose;
    use super::trend::trend_analysis;

    /// Generate a synthetic time series with trend, seasonality, and noise.
    fn synthetic_series(n: usize, period: usize) -> Vec<f64> {
        let mut rng_state: u64 = 12345;
        (0..n)
            .map(|i| {
                let t = i as f64;
                let trend = 50.0 + 0.05 * t;
                let seasonal = 3.0 * (2.0 * std::f64::consts::PI * t / period as f64).sin()
                    + 1.5 * (4.0 * std::f64::consts::PI * t / period as f64).cos();
                // Simple noise
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = ((rng_state >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.5;
                trend + seasonal + noise
            })
            .collect()
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn full_pipeline() {
        let period = 12;
        let n = 240;
        let data = synthetic_series(n, period);

        // 1. Detect period
        let detection = detect_period(&data);
        assert!(
            (detection.period as isize - period as isize).abs() <= 3,
            "detected period: {}",
            detection.period
        );

        // 2. STL decomposition
        let config = STLConfig {
            period,
            inner_iterations: 2,
            ..Default::default()
        };
        let stl_result = stl_decompose(&data, &config);
        assert_eq!(stl_result.length, n);
        // Decomposition should sum correctly
        for i in 0..n
        {
            let reconstructed =
                stl_result.trend[i] + stl_result.seasonal[i] + stl_result.remainder[i];
            assert!(
                (data[i] - reconstructed).abs() < 1e-10,
                "STL decomposition error at {}",
                i
            );
        }

        // 3. Fourier analysis
        let cyclic = fourier_analysis(&stl_result.remainder);
        // Remainder should have low amplitude
        assert!(
            cyclic.amplitude < 2.0,
            "remainder amplitude: {}",
            cyclic.amplitude
        );

        // 4. Deseasonalize
        let adjusted = deseasonalize(&data, period, true);
        assert_eq!(adjusted.len(), n);

        // 5. Trend analysis
        let trend_result = trend_analysis(&data);
        assert!(trend_result.slope > 0.0, "should detect positive trend");
        assert!(
            trend_result.slope > 0.03,
            "slope should be ~0.05, got {}",
            trend_result.slope
        );

        // 6. Change point detection
        let mut data_with_shift = data.clone();
        for i in 180..n
        {
            data_with_shift[i] += 5.0;
        }
        let cps = seasonal_cusum(&data_with_shift, period, 5.0);
        assert!(!cps.is_empty(), "should detect change point");
    }

    #[test]
    fn edge_cases() {
        // Very short series
        let short_data = vec![1.0, 2.0, 3.0, 4.0];
        let result = trend_analysis(&short_data);
        assert!(result.slope > 0.0);

        // Constant series
        let constant = vec![5.0; 100];
        let result = trend_analysis(&constant);
        assert!(result.slope.abs() < f64::EPSILON);
        assert!(!result.significant);

        // Single value
        let single = vec![42.0];
        let result = trend_analysis(&single);
        assert_eq!(result.slope, 0.0);
    }
}
