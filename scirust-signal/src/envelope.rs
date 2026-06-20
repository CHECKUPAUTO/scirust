//! Hilbert envelope analysis — the gold standard for rolling-element bearing
//! diagnosis.
//!
//! A bearing defect produces periodic impacts that **amplitude-modulate** a
//! high-frequency structural resonance. Demodulating that resonance with the
//! analytic-signal envelope (Hilbert transform) and taking its spectrum exposes
//! the bearing fault frequencies (BPFO/BPFI/BSF) as clear peaks — invisible in
//! the raw spectrum, where they are buried under the carrier.

use crate::complex::Complex;
use crate::fft::{fft, fft_real, ifft};

/// Analytic-signal envelope via the Hilbert transform (FFT method).
/// `signal.len()` must be a power of two.
pub fn hilbert_envelope(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let mut buf: Vec<Complex> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft(&mut buf);
    // Analytic-signal weighting: keep DC and Nyquist, double positive
    // frequencies, zero the negative ones.
    for (k, c) in buf.iter_mut().enumerate()
    {
        if k == 0 || (n.is_multiple_of(2) && k == n / 2)
        {
            // unchanged
        }
        else if k < n / 2
        {
            *c = *c * 2.0;
        }
        else
        {
            *c = Complex::new(0.0, 0.0);
        }
    }
    ifft(&mut buf);
    buf.iter().map(Complex::mag).collect()
}

/// Envelope spectrum: magnitude half-spectrum of the DC-removed Hilbert
/// envelope. Bearing fault frequencies show up here as peaks.
pub fn envelope_spectrum(signal: &[f64]) -> Vec<f64> {
    let env = hilbert_envelope(signal);
    let mean = env.iter().sum::<f64>() / env.len() as f64;
    let centered: Vec<f64> = env.iter().map(|&x| x - mean).collect();
    fft_real(&centered).iter().map(Complex::mag).collect()
}

/// Strongest modulation frequency (Hz) in the envelope spectrum, ignoring the
/// lowest `dc_guard_bins` bins (DC / drift).
pub fn dominant_envelope_freq(signal: &[f64], sample_rate: f64, dc_guard_bins: usize) -> f64 {
    let spec = envelope_spectrum(signal);
    let n = signal.len();
    let bin_hz = sample_rate / n as f64;
    let mut best_bin = dc_guard_bins;
    let mut best = 0.0;
    for (b, &m) in spec.iter().enumerate().skip(dc_guard_bins)
    {
        if m > best
        {
            best = m;
            best_bin = b;
        }
    }
    best_bin as f64 * bin_hz
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn envelope_recovers_bearing_modulation() {
        // A 2 kHz resonance amplitude-modulated by 80 Hz impacts (BPFO).
        let (n, sr, carrier, fault) = (8192usize, 8192.0, 2000.0, 80.0);
        let sig: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (1.0 + 0.8 * (2.0 * PI * fault * t).sin()) * (2.0 * PI * carrier * t).sin()
            })
            .collect();
        let f = dominant_envelope_freq(&sig, sr, 5);
        assert!(
            (f - fault).abs() < 2.0,
            "dominant envelope freq {f} (want {fault})"
        );
    }

    #[test]
    fn envelope_of_pure_tone_is_flat() {
        // A pure carrier has a constant envelope -> no modulation peak.
        let (n, sr) = (4096usize, 4096.0);
        let sig: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 500.0 * i as f64 / sr).sin())
            .collect();
        let env = hilbert_envelope(&sig);
        let mean = env.iter().sum::<f64>() / env.len() as f64;
        // Envelope ~ constant amplitude 1 (away from edges).
        let mid = &env[n / 4..3 * n / 4];
        for &e in mid
        {
            assert!((e - mean).abs() < 0.1, "envelope not flat: {e} vs {mean}");
        }
    }
}
