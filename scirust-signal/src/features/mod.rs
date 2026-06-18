pub mod spectral;

/// Root Mean Square of a signal.
pub fn rms(signal: &[f64]) -> f64 {
    if signal.is_empty()
    {
        return 0.0;
    }
    let sum_sq: f64 = signal.iter().map(|&x| x * x).sum();
    f64::sqrt(sum_sq / signal.len() as f64)
}

/// Peak value (maximum absolute value).
pub fn peak(signal: &[f64]) -> f64 {
    signal.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()))
}

/// Peak-to-peak amplitude.
pub fn peak_to_peak(signal: &[f64]) -> f64 {
    if signal.is_empty()
    {
        return 0.0;
    }
    let min = signal.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = signal.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    max - min
}

/// Crest factor: peak / RMS.
/// Indicates impulsiveness; high values (> 4-5) suggest bearing defects.
pub fn crest_factor(signal: &[f64]) -> f64 {
    let r = rms(signal);
    if r < f64::EPSILON
    {
        return 0.0;
    }
    peak(signal) / r
}

/// Kurtosis (excess kurtosis: normal = 0).
/// High kurtosis (> 3) indicates impulsive signals typical of bearing faults.
pub fn kurtosis(signal: &[f64]) -> f64 {
    if signal.len() < 2
    {
        return 0.0;
    }
    let n = signal.len() as f64;
    let mean: f64 = signal.iter().sum::<f64>() / n;
    let m2: f64 = signal.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let m4: f64 = signal.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n;
    if m2 < f64::EPSILON
    {
        return 0.0;
    }
    m4 / (m2 * m2) - 3.0 // excess kurtosis
}

/// Skewness: asymmetry of the distribution.
/// Positive = right-skewed, negative = left-skewed.
pub fn skewness(signal: &[f64]) -> f64 {
    if signal.len() < 2
    {
        return 0.0;
    }
    let n = signal.len() as f64;
    let mean: f64 = signal.iter().sum::<f64>() / n;
    let m2: f64 = signal.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let m3: f64 = signal.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n;
    if m2 < f64::EPSILON
    {
        return 0.0;
    }
    m3 / m2.powf(1.5)
}

/// Zero-crossing rate: number of sign changes / (N-1).
/// Useful for voice/signal activity detection.
pub fn zero_crossing_rate(signal: &[f64]) -> f64 {
    if signal.len() < 2
    {
        return 0.0;
    }
    let count = signal
        .windows(2)
        .filter(|w| w[0].signum() != w[1].signum())
        .count();
    count as f64 / (signal.len() - 1) as f64
}

/// Autocorrelation at lag `k`.
pub fn autocorrelation(signal: &[f64], lag: usize) -> f64 {
    if lag >= signal.len()
    {
        return 0.0;
    }
    let n = signal.len() - lag;
    if n == 0
    {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..n
    {
        sum += signal[i] * signal[i + lag];
    }
    sum / n as f64
}

/// Autocorrelation for all lags 0..max_lag.
pub fn autocorrelation_full(signal: &[f64], max_lag: usize) -> Vec<f64> {
    (0..=max_lag.min(signal.len().saturating_sub(1)))
        .map(|lag| autocorrelation(signal, lag))
        .collect()
}

/// Signal energy: sum of squared samples.
pub fn energy(signal: &[f64]) -> f64 {
    signal.iter().map(|&x| x * x).sum()
}

/// Shannon entropy estimate via histogram (64 bins).
pub fn entropy(signal: &[f64]) -> f64 {
    if signal.is_empty()
    {
        return 0.0;
    }
    let n_bins = 64usize;
    let min = signal.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = signal.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max - min;
    if range < f64::EPSILON
    {
        return 0.0;
    }
    let mut bins = vec![0usize; n_bins];
    for &x in signal
    {
        let mut idx = ((x - min) / range * n_bins as f64) as usize;
        if idx >= n_bins
        {
            idx = n_bins - 1;
        }
        bins[idx] += 1;
    }
    let n = signal.len() as f64;
    let mut h = 0.0;
    for &count in &bins
    {
        if count > 0
        {
            let p = count as f64 / n;
            h -= p * p.log2();
        }
    }
    h
}

/// Returns a vector of standard time-domain features.
/// Order: [rms, crest_factor, kurtosis, skewness, zcr, entropy, peak_to_peak, energy]
pub fn time_features(signal: &[f64]) -> Vec<f64> {
    vec![
        rms(signal),
        crest_factor(signal),
        kurtosis(signal),
        skewness(signal),
        zero_crossing_rate(signal),
        entropy(signal),
        peak_to_peak(signal),
        energy(signal),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    #[test]
    fn test_rms() {
        let s = vec![1.0, 2.0, 3.0];
        // sqrt((1+4+9)/3) = sqrt(14/3) = sqrt(4.6667) ≈ 2.1602
        let expected = f64::sqrt(14.0 / 3.0);
        assert!((rms(&s) - expected).abs() < EPS);
    }

    #[test]
    fn test_zcr() {
        let s = vec![-1.0, 1.0, -1.0, 1.0];
        assert!((zero_crossing_rate(&s) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_autocorrelation() {
        let s = vec![1.0, 1.0, 1.0, 1.0];
        assert!((autocorrelation(&s, 0) - 1.0).abs() < EPS);
        assert!((autocorrelation(&s, 1) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_kurtosis_normalish() {
        // A simple sine wave has negative excess kurtosis
        let n = 128;
        let s: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let k = kurtosis(&s);
        // Sine waves have excess kurtosis around -1.5
        assert!(k < 0.0);
    }

    #[test]
    fn test_empty_inputs() {
        let empty: Vec<f64> = vec![];
        assert!((rms(&empty) - 0.0).abs() < EPS);
        assert!((crest_factor(&empty) - 0.0).abs() < EPS);
        assert!((kurtosis(&empty) - 0.0).abs() < EPS);
        assert!((entropy(&empty) - 0.0).abs() < EPS);
    }

    #[test]
    fn test_crest_factor_impulse() {
        // A signal with a large spike should have high crest factor
        let mut s = vec![0.1; 100];
        s[50] = 10.0;
        let cf = crest_factor(&s);
        assert!(cf > 3.0, "crest_factor too low: {}", cf);
    }
}
