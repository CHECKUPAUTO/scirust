use crate::Complex;

/// Power Spectral Density: `|X[k]|^2 / N` for each positive-frequency bin.
///
/// `spectrum` should be the output of `fft_real` (positive half only).
/// Returns PSD values in the same order.
pub fn psd(spectrum: &[Complex], n_total: usize) -> Vec<f64> {
    let n = n_total as f64;
    spectrum.iter().map(|c| c.mag_sq() / n).collect()
}

/// Spectral centroid: weighted mean of frequencies.
/// Higher values → brighter/higher-frequency content.
///
/// `spectrum` is the positive half-spectrum. `sample_rate` in Hz.
pub fn spectral_centroid(spectrum: &[Complex], sample_rate: f64) -> f64 {
    if spectrum.len() <= 1
    {
        return 0.0;
    }
    let n = spectrum.len() - 1; // positive bins excluding DC
    let mut numer = 0.0;
    let mut denom = 0.0;
    for (k, c) in spectrum.iter().enumerate().skip(1)
    {
        let freq = k as f64 * sample_rate / (2.0 * n as f64);
        let mag = c.mag();
        numer += freq * mag;
        denom += mag;
    }
    if denom < f64::EPSILON
    {
        return 0.0;
    }
    numer / denom
}

/// Spectral spread (standard deviation around centroid).
pub fn spectral_spread(spectrum: &[Complex], sample_rate: f64) -> f64 {
    let centroid = spectral_centroid(spectrum, sample_rate);
    if spectrum.len() <= 1
    {
        return 0.0;
    }
    let n = spectrum.len() - 1;
    let mut numer = 0.0;
    let mut denom = 0.0;
    for (k, c) in spectrum.iter().enumerate().skip(1)
    {
        let freq = k as f64 * sample_rate / (2.0 * n as f64);
        let mag = c.mag();
        let diff = freq - centroid;
        numer += diff * diff * mag;
        denom += mag;
    }
    if denom < f64::EPSILON
    {
        return 0.0;
    }
    f64::sqrt(numer / denom)
}

/// Spectral entropy (normalized, 0..1).
/// 0 = pure tone, 1 = white noise.
pub fn spectral_entropy(spectrum: &[Complex]) -> f64 {
    let n = spectrum.len();
    if n <= 1
    {
        return 0.0;
    }
    let total: f64 = spectrum.iter().map(|c| c.mag_sq()).sum();
    if total < f64::EPSILON
    {
        return 0.0;
    }
    let mut h = 0.0;
    for c in spectrum
    {
        let p = c.mag_sq() / total;
        if p > f64::EPSILON
        {
            h -= p * p.log2();
        }
    }
    let max_entropy = (n as f64).log2();
    if max_entropy < f64::EPSILON
    {
        return 0.0;
    }
    h / max_entropy
}

/// Spectral rolloff: frequency below which `ratio` (e.g. 0.85) of total energy is contained.
pub fn spectral_rolloff(spectrum: &[Complex], sample_rate: f64, ratio: f64) -> f64 {
    if spectrum.len() <= 1
    {
        return 0.0;
    }
    let total: f64 = spectrum.iter().map(|c| c.mag_sq()).sum();
    let threshold = total * ratio;
    let n = spectrum.len() - 1;
    let mut accum = 0.0;
    for (k, c) in spectrum.iter().enumerate()
    {
        accum += c.mag_sq();
        if accum >= threshold
        {
            return k as f64 * sample_rate / (2.0 * n as f64);
        }
    }
    sample_rate / 2.0
}

/// Band power: sum of squared magnitudes in a frequency range [low_hz, high_hz].
pub fn band_power(spectrum: &[Complex], sample_rate: f64, low_hz: f64, high_hz: f64) -> f64 {
    if spectrum.len() <= 1
    {
        return 0.0;
    }
    let n = spectrum.len() - 1;
    let nyquist = sample_rate / 2.0;
    let mut power = 0.0;
    for (k, c) in spectrum.iter().enumerate()
    {
        let freq = k as f64 * sample_rate / (2.0 * n as f64);
        if freq >= low_hz && freq <= high_hz.min(nyquist)
        {
            power += c.mag_sq();
        }
    }
    power
}

/// Spectral flatness: geometric mean / arithmetic mean of the spectrum.
/// 1.0 = white noise (flat), close to 0 = tonal.
pub fn spectral_flatness(spectrum: &[Complex]) -> f64 {
    let mags: Vec<f64> = spectrum.iter().map(|c| c.mag()).collect();
    let n = mags.len();
    if n == 0
    {
        return 0.0;
    }
    let am: f64 = mags.iter().sum::<f64>() / n as f64;
    if am < f64::EPSILON
    {
        return 0.0;
    }
    let gm: f64 = {
        let sum_log: f64 = mags
            .iter()
            .map(|&m| {
                if m > f64::EPSILON
                {
                    m.ln()
                }
                else
                {
                    f64::NEG_INFINITY
                }
            })
            .sum();
        if sum_log.is_finite()
        {
            (sum_log / n as f64).exp()
        }
        else
        {
            0.0
        }
    };
    gm / am
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn test_psd() {
        let spec = vec![
            Complex::new(0.0, 0.0),
            Complex::new(3.0, 4.0),
            Complex::new(1.0, 0.0),
        ];
        let psd_vals = psd(&spec, 4); // n_total used for scaling
        assert!((psd_vals[0] - 0.0).abs() < EPS);
        // |3+4i|^2 = 25, / 4 = 6.25
        assert!((psd_vals[1] - 6.25).abs() < EPS);
    }

    #[test]
    fn test_spectral_entropy_pure_tone() {
        // Pure tone: one bin has all energy → entropy ~ 0
        let mut spec = vec![Complex::zero(); 32];
        spec[4] = Complex::new(10.0, 0.0);
        let ent = spectral_entropy(&spec);
        assert!(ent < 0.01, "entropy {} should be near 0", ent);
    }

    #[test]
    fn test_band_power() {
        // spectrum with bins at 0, 1.33, 2.67, 4.0 Hz (sample_rate=8, N=8 → bin_spacing=1 Hz)
        // Use 9-point spectrum to get 1 Hz bin spacing: 8/(2*4)=1 Hz
        // Bin 0=0Hz, Bin 1=1Hz, Bin 2=2Hz, Bin 3=3Hz, Bin 4=4Hz
        let spec = vec![
            Complex::new(0.0, 0.0), // DC
            Complex::new(2.0, 0.0), // 1 Hz
            Complex::new(3.0, 0.0), // 2 Hz
            Complex::new(1.0, 0.0), // 3 Hz
            Complex::new(0.0, 0.0), // 4 Hz
        ];
        let bp = band_power(&spec, 8.0, 1.5, 2.5);
        assert!((bp - 9.0).abs() < EPS); // only 3.0^2 = 9 at bin 2 (2 Hz)
    }

    #[test]
    fn test_spectral_rolloff_empty_does_not_panic() {
        // Empty spectrum previously underflowed `spectrum.len() - 1`
        // (panic in debug, wrap-to-usize::MAX in release).
        let spec: Vec<Complex> = Vec::new();
        assert_eq!(spectral_rolloff(&spec, 8.0, 0.85), 0.0);

        // Single-bin (DC only) is also degenerate and must return 0.0.
        let dc = vec![Complex::new(1.0, 0.0)];
        assert_eq!(spectral_rolloff(&dc, 8.0, 0.85), 0.0);
    }

    #[test]
    fn test_band_power_empty_does_not_panic() {
        // Empty spectrum previously underflowed `spectrum.len() - 1`
        // (panic in debug, wrap-to-usize::MAX in release).
        let spec: Vec<Complex> = Vec::new();
        assert_eq!(band_power(&spec, 8.0, 1.0, 2.0), 0.0);

        // Single-bin (DC only) is also degenerate and must return 0.0.
        let dc = vec![Complex::new(1.0, 0.0)];
        assert_eq!(band_power(&dc, 8.0, 1.0, 2.0), 0.0);
    }
}
