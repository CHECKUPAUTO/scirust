use crate::Complex;
use crate::error::{SignalError, SignalResult};
use crate::fft::fft_real;

/// Power Spectral Density: `|X[k]|^2 / N` for each positive-frequency bin.
///
/// `spectrum` should be the output of `fft_real` (positive half only).
/// Returns PSD values in the same order.
pub fn psd(spectrum: &[Complex], n_total: usize) -> Vec<f64> {
    let n = n_total as f64;
    spectrum.iter().map(|c| c.mag_sq() / n).collect()
}

/// The result of a Welch power-spectral-density estimate.
///
/// Carries the segment count beside the numbers, because that is what a
/// caller needs in order to know how much the estimate can be trusted: the
/// variance of a Welch PSD falls roughly as `1/segments`, so a spectrum
/// reported without its `segments` is as uninterpretable as a mean reported
/// without its sample size. `dropped_samples` is reported for the same
/// reason — a record that does not divide evenly into segments loses its tail,
/// and silently keeping that to ourselves would overstate how much data went
/// into the answer.
#[derive(Debug, Clone, PartialEq)]
pub struct WelchPsd {
    /// The averaged power spectral density over the positive half-spectrum,
    /// on the same scale as [`psd`].
    pub psd: Vec<f64>,
    /// How many segments were averaged.
    pub segments: usize,
    /// Trailing samples that did not fill a whole segment and were not used.
    pub dropped_samples: usize,
}

/// Welch's method: the average of the periodograms of overlapping,
/// windowed segments.
///
/// A single periodogram ([`psd`] of one [`fft_real`]) is a **biased,
/// high-variance** estimator whose variance does not shrink with record
/// length — a longer record buys frequency resolution, not stability.
/// Averaging `K` segments cuts the standard deviation by roughly `√K`, at
/// the cost of the coarser resolution a shorter segment implies. That
/// trade-off is the whole method, and it is why both halves are reported:
/// [`WelchPsd::segments`] says how much averaging happened, and
/// `sample_rate / segment_len` is the resolution paid for it.
///
/// # Scale
///
/// The result is on the same scale as [`psd`], so the two are directly
/// comparable: the per-segment periodograms are divided by the window's
/// mean square power, which makes a rectangular window a no-op. A
/// single-segment rectangular call therefore reproduces
/// `psd(&fft_real(x), x.len())` exactly. Without that correction a
/// Hann-windowed estimate would sit ~3/8 of a rectangular one and the two
/// could not be compared at all.
///
/// # Dropped samples
///
/// Segments start at `0, hop, 2·hop, …` with `hop = segment_len - overlap`.
/// A trailing partial segment is **not** zero-padded — padding would invent
/// data and bias the estimate low — so it is dropped and counted in
/// [`WelchPsd::dropped_samples`].
///
/// # Errors
/// [`SignalError::InvalidInput`] if `segment_len` is not a power of two of at
/// least 2 (the radix-2 FFT requires it), if `overlap >= segment_len`, if
/// `window.len() != segment_len`, if the signal is shorter than one segment,
/// or if any sample or window coefficient is non-finite.
pub fn welch_psd(
    signal: &[f64],
    segment_len: usize,
    overlap: usize,
    window: &[f64],
) -> SignalResult<WelchPsd> {
    let bad = |m: String| Err(SignalError::InvalidInput(m));
    if segment_len < 2 || !segment_len.is_power_of_two()
    {
        return bad(format!(
            "segment_len must be a power of two of at least 2, got {segment_len}"
        ));
    }
    if overlap >= segment_len
    {
        return bad(format!(
            "overlap {overlap} must be less than segment_len {segment_len}"
        ));
    }
    if window.len() != segment_len
    {
        return bad(format!(
            "window has {} coefficients but segment_len is {segment_len}",
            window.len()
        ));
    }
    if signal.len() < segment_len
    {
        return bad(format!(
            "signal has {} samples, fewer than one segment of {segment_len}",
            signal.len()
        ));
    }
    if let Some(i) = signal.iter().position(|x| !x.is_finite())
    {
        return bad(format!("sample #{i} is not finite ({})", signal[i]));
    }
    if let Some(i) = window.iter().position(|w| !w.is_finite())
    {
        return bad(format!("window coefficient #{i} is not finite"));
    }

    // Mean square window power, the correction that puts every window on the
    // same scale. Guaranteed non-zero: a window of all zeros would make every
    // segment identically zero, which is not an estimate of anything.
    let window_power = window.iter().map(|w| w * w).sum::<f64>() / segment_len as f64;
    if window_power <= 0.0
    {
        return bad("window has zero total power".to_string());
    }

    let hop = segment_len - overlap;
    let segments = 1 + (signal.len() - segment_len) / hop;
    let used = (segments - 1) * hop + segment_len;

    let mut acc = vec![0.0_f64; segment_len / 2 + 1];
    let mut shaped = vec![0.0_f64; segment_len];
    for s in 0..segments
    {
        let start = s * hop;
        for (dst, (x, w)) in shaped
            .iter_mut()
            .zip(signal[start..start + segment_len].iter().zip(window))
        {
            *dst = x * w;
        }
        let spectrum = fft_real(&shaped);
        for (a, p) in acc.iter_mut().zip(psd(&spectrum, segment_len))
        {
            *a += p;
        }
    }

    let scale = 1.0 / (segments as f64 * window_power);
    for a in &mut acc
    {
        *a *= scale;
    }
    Ok(WelchPsd {
        psd: acc,
        segments,
        dropped_samples: signal.len() - used,
    })
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
mod welch_tests {
    use super::*;
    use crate::windows::hanning;
    use std::f64::consts::TAU;

    fn rect(n: usize) -> Vec<f64> {
        vec![1.0; n]
    }

    /// Deterministic zero-mean white noise, so the variance claims below are
    /// reproducible rather than flaky.
    fn noise(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                // SplitMix64, inlined to keep this module dependency-free.
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect()
    }

    /// Relative dispersion across non-DC bins — ≈1 for a single periodogram of
    /// noise, and ≈1/√K after averaging K segments.
    fn dispersion(psd: &[f64]) -> f64 {
        let bins = &psd[1..];
        let n = bins.len() as f64;
        let mean = bins.iter().sum::<f64>() / n;
        let var = bins.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n;
        var.sqrt() / mean
    }

    fn tone(freq: f64, rate: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (TAU * freq * i as f64 / rate).sin())
            .collect()
    }

    #[test]
    fn one_rectangular_segment_reproduces_the_plain_periodogram() {
        // The scale claim, checked exactly: Welch is the *average* of the same
        // quantity `psd` computes, so with one segment and no window it must
        // be that quantity bit for bit.
        let x = tone(64.0, 1024.0, 1024);
        let direct = psd(&fft_real(&x), 1024);
        let welch = welch_psd(&x, 1024, 0, &rect(1024)).unwrap();

        assert_eq!(welch.segments, 1);
        assert_eq!(welch.dropped_samples, 0);
        assert_eq!(welch.psd.len(), direct.len());
        for (i, (a, b)) in welch.psd.iter().zip(&direct).enumerate()
        {
            assert!((a - b).abs() < 1e-12, "bin {i}: {a} vs {b}");
        }
    }

    #[test]
    fn averaging_segments_reduces_the_variance_by_about_the_square_root() {
        // The property the method exists for. A single 256-sample periodogram
        // of noise has relative dispersion ≈1; averaging 16 non-overlapping
        // segments should cut it by roughly √16 = 4.
        let x = noise(4096, 0x5EED);
        let one = welch_psd(&x[..256], 256, 0, &rect(256)).unwrap();
        let many = welch_psd(&x, 256, 0, &rect(256)).unwrap();

        assert_eq!(one.segments, 1);
        assert_eq!(many.segments, 16);
        let (d1, d16) = (dispersion(&one.psd), dispersion(&many.psd));
        assert!(d1 > 0.7, "a single periodogram of noise is noisy: {d1}");
        assert!(
            d16 < d1 / 2.5,
            "16 segments should cut the dispersion by ~4: {d1} -> {d16}"
        );
    }

    #[test]
    fn more_segments_keep_helping() {
        // Monotone in the segment count, which is what distinguishes real
        // averaging from a longer single transform.
        let x = noise(16_384, 7);
        let d = |segments: usize| {
            let n = 256 * segments;
            dispersion(&welch_psd(&x[..n], 256, 0, &rect(256)).unwrap().psd)
        };
        let (d4, d64) = (d(4), d(64));
        assert!(d64 < d4, "64 segments must beat 4: {d4} -> {d64}");
    }

    #[test]
    fn a_longer_single_transform_does_not_help_which_is_the_whole_point() {
        // The contrast that makes the method worth having: 16x the samples in
        // *one* transform buys resolution, not stability.
        let x = noise(4096, 0x5EED);
        let short = dispersion(&welch_psd(&x[..256], 256, 0, &rect(256)).unwrap().psd);
        let long = dispersion(&welch_psd(&x, 4096, 0, &rect(4096)).unwrap().psd);
        assert!(
            long > 0.9 * short,
            "a single long periodogram must not be more stable: {short} -> {long}"
        );
    }

    #[test]
    fn a_hann_window_lands_on_the_same_scale_as_a_rectangular_one() {
        // Without the window-power correction a Hann estimate would sit at
        // ~3/8 of a rectangular one and the two could not be compared.
        let x = noise(4096, 11);
        let r = welch_psd(&x, 256, 0, &rect(256)).unwrap();
        let h = welch_psd(&x, 256, 0, &hanning(256)).unwrap();
        let total = |p: &[f64]| p.iter().sum::<f64>();
        let ratio = total(&h.psd) / total(&r.psd);
        assert!(
            (0.8..1.25).contains(&ratio),
            "windows must be comparable in scale, got ratio {ratio}"
        );
    }

    #[test]
    fn a_tone_still_lands_in_its_own_bin() {
        // Averaging must not smear the thing being measured.
        let x = tone(64.0, 1024.0, 4096);
        let w = welch_psd(&x, 1024, 512, &hanning(1024)).unwrap();
        let peak = w
            .psd
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, 64, "the tone must stay in bin 64");
        assert_eq!(w.segments, 7, "50% overlap over 4096 samples");
    }

    #[test]
    fn overlap_yields_more_segments_from_the_same_record() {
        let x = noise(4096, 3);
        let none = welch_psd(&x, 256, 0, &rect(256)).unwrap();
        let half = welch_psd(&x, 256, 128, &rect(256)).unwrap();
        assert_eq!(none.segments, 16);
        assert_eq!(half.segments, 31);
    }

    #[test]
    fn a_trailing_partial_segment_is_dropped_and_counted() {
        // Not zero-padded: padding would invent data and bias the estimate
        // low. Reported instead, so the omission is visible.
        let x = noise(300, 5);
        let w = welch_psd(&x, 256, 0, &rect(256)).unwrap();
        assert_eq!(w.segments, 1);
        assert_eq!(w.dropped_samples, 44);
    }

    #[test]
    fn the_estimate_is_bit_reproducible() {
        let x = noise(2048, 99);
        let a = welch_psd(&x, 256, 128, &hanning(256)).unwrap();
        let b = welch_psd(&x, 256, 128, &hanning(256)).unwrap();
        for (p, q) in a.psd.iter().zip(&b.psd)
        {
            assert_eq!(p.to_bits(), q.to_bits());
        }
    }

    #[test]
    fn invalid_parameters_are_errors_rather_than_panics() {
        let x = noise(1024, 1);
        // Not a power of two — the radix-2 FFT would assert.
        assert!(welch_psd(&x, 300, 0, &rect(300)).is_err());
        // Degenerate segment.
        assert!(welch_psd(&x, 1, 0, &rect(1)).is_err());
        // Overlap at or beyond the segment would make hop zero.
        assert!(welch_psd(&x, 256, 256, &rect(256)).is_err());
        assert!(welch_psd(&x, 256, 300, &rect(256)).is_err());
        // Window length mismatch.
        assert!(welch_psd(&x, 256, 0, &rect(128)).is_err());
        // Shorter than one segment.
        assert!(welch_psd(&x[..100], 256, 0, &rect(256)).is_err());
        // A window with no power estimates nothing.
        assert!(welch_psd(&x, 256, 0, &vec![0.0; 256]).is_err());
    }

    #[test]
    fn non_finite_input_is_named_rather_than_producing_nan() {
        let mut x = noise(1024, 2);
        x[513] = f64::NAN;
        let err = welch_psd(&x, 256, 0, &rect(256)).unwrap_err();
        assert!(err.to_string().contains("513"), "{err}");

        let mut w = rect(256);
        w[7] = f64::INFINITY;
        let err = welch_psd(&noise(1024, 2), 256, 0, &w).unwrap_err();
        assert!(err.to_string().contains('7'), "{err}");
    }
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
