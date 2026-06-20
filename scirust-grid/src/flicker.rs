//! Voltage flicker severity (simplified IEC 61000-4-15).
//!
//! Flicker is the visual annoyance of light fluctuating with the supply voltage.
//! The eye is most sensitive to modulation around **8.8 Hz**; the IEC flickermeter
//! captures this with a perceptibility weighting filter. This module implements
//! the essential physics — spectral analysis of the relative voltage fluctuation
//! `ΔV/V` weighted by a perceptibility curve peaking at 8.8 Hz — to produce a
//! flicker-sensation index.
//!
//! It is a **simplified** estimator (the certified IEC flickermeter adds squaring
//! demodulation, specific filter coefficients and the statistical `Pst`
//! evaluation); use it as a relative indicator, not a compliance instrument.

use scirust_signal::fft_real;

/// IEC-like perceptibility weight at frequency `f` (Hz): a log-Gaussian band-pass
/// peaking exactly at 8.8 Hz with unit gain there, falling off symmetrically in
/// log-frequency.
pub fn perceptibility_weight(f: f64) -> f64 {
    if f <= 0.0
    {
        return 0.0;
    }
    let f0 = 8.8;
    let sigma = 0.8;
    let z = (f / f0).ln() / sigma;
    (-0.5 * z * z).exp()
}

/// Flicker-sensation index from a relative voltage-fluctuation signal `ΔV/V`
/// sampled at `fs` Hz (`fluctuation.len()` a power of two): the RMS of the
/// fluctuation spectrum weighted by [`perceptibility_weight`].
pub fn flicker_severity(fluctuation: &[f64], fs: f64) -> f64 {
    let n = fluctuation.len();
    if n < 4
    {
        return 0.0;
    }
    let mean = fluctuation.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = fluctuation.iter().map(|&x| x - mean).collect();
    let spec = fft_real(&centered);
    let bin_hz = fs / n as f64;
    let mut power = 0.0;
    for (b, c) in spec.iter().enumerate()
    {
        let f = b as f64 * bin_hz;
        let w = perceptibility_weight(f);
        // |X|/N amplitude, weighted; sum of squared weighted amplitudes.
        let amp = c.mag() / n as f64;
        power += (w * amp).powi(2);
    }
    power.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    fn modulation(n: usize, fs: f64, fmod: f64, depth: f64) -> Vec<f64> {
        (0..n)
            .map(|i| depth * (2.0 * PI * fmod * i as f64 / fs).sin())
            .collect()
    }

    #[test]
    fn perceptibility_peaks_near_8_8_hz() {
        let p88 = perceptibility_weight(8.8);
        assert!(p88 >= perceptibility_weight(1.0));
        assert!(p88 >= perceptibility_weight(30.0));
        assert!((perceptibility_weight(8.8) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn flicker_is_highest_at_the_perceptible_frequency() {
        let (n, fs, depth) = (4096usize, 1000.0, 0.02);
        let at_88 = flicker_severity(&modulation(n, fs, 8.8, depth), fs);
        let at_1 = flicker_severity(&modulation(n, fs, 1.0, depth), fs);
        let at_30 = flicker_severity(&modulation(n, fs, 30.0, depth), fs);
        assert!(at_88 > at_1, "8.8Hz {at_88} vs 1Hz {at_1}");
        assert!(at_88 > at_30, "8.8Hz {at_88} vs 30Hz {at_30}");
    }

    #[test]
    fn deeper_modulation_means_more_flicker() {
        let (n, fs) = (4096usize, 1000.0);
        let shallow = flicker_severity(&modulation(n, fs, 8.8, 0.01), fs);
        let deep = flicker_severity(&modulation(n, fs, 8.8, 0.05), fs);
        assert!(deep > shallow);
        // A steady voltage flickers not at all.
        assert!(flicker_severity(&vec![0.0; n], fs) < 1e-12);
    }
}
