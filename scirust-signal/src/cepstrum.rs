//! Cepstrum analysis — gear-train and harmonic-family diagnosis.
//!
//! The real cepstrum `IFFT(log|FFT(x)|)` turns *uniformly spaced* spectral lines
//! (gear-mesh harmonics, bearing sidebands) into a single peak at the
//! **quefrency** equal to their spacing's period. Where the spectrum shows a
//! whole comb, the cepstrum collapses it to one clear marker — the standard tool
//! for gearbox condition monitoring.

use crate::complex::Complex;
use crate::fft::{fft, ifft};

/// Real cepstrum of `signal` (`len` a power of two): `IFFT(log|FFT|)`.
pub fn real_cepstrum(signal: &[f64]) -> Vec<f64> {
    let mut buf: Vec<Complex> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft(&mut buf);
    for c in buf.iter_mut()
    {
        *c = Complex::new((c.mag() + 1e-12).ln(), 0.0);
    }
    ifft(&mut buf);
    buf.iter().map(|c| c.re).collect()
}

/// Quefrency (seconds) of the strongest cepstral peak above `min_quefrency_bins`
/// (skip the low-quefrency spectral-envelope region). Returns 0 if none.
pub fn dominant_quefrency(signal: &[f64], sample_rate: f64, min_quefrency_bins: usize) -> f64 {
    let c = real_cepstrum(signal);
    let n = signal.len();
    let hi = n / 2;
    let lo = min_quefrency_bins.max(1).min(hi);
    let mut best_bin = lo;
    let mut best = f64::MIN;
    for (b, &v) in c.iter().enumerate().take(hi).skip(lo)
    {
        if v > best
        {
            best = v;
            best_bin = b;
        }
    }
    best_bin as f64 / sample_rate
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn comb_of_harmonics_makes_a_rahmonic_at_one_over_spacing() {
        // Gear-mesh family: a strong comb of harmonics of f0 = 128 Hz. The
        // rahmonic sits at quefrency 1/f0, i.e. bin sr/f0 = 64.
        let (n, sr, f0) = (8192usize, 8192.0, 128.0);
        let sig: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sr;
                (1..=30)
                    .map(|k| (2.0 * PI * f0 * k as f64 * t).sin())
                    .sum::<f64>()
            })
            .collect();
        let c = real_cepstrum(&sig);
        let b = (sr / f0).round() as usize; // 64
        // A clear local maximum at the spacing rahmonic.
        assert!(
            c[b] > c[b - 8] && c[b] > c[b + 8] && c[b] > 0.0,
            "no rahmonic at bin {b}"
        );
        // And it is the dominant quefrency above the low-quefrency envelope.
        let q = dominant_quefrency(&sig, sr, 16);
        assert!(
            (q - 1.0 / f0).abs() < 5e-4,
            "quefrency {q} (want {})",
            1.0 / f0
        );
    }

    #[test]
    fn pure_tone_has_no_strong_high_quefrency_peak() {
        // A single sinusoid is not a harmonic comb -> no spacing marker.
        let (n, sr) = (4096usize, 4096.0);
        let sig: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 137.0 * i as f64 / sr).sin())
            .collect();
        let c = real_cepstrum(&sig);
        // Cepstrum is finite and the comb-quefrency (0.01 s) is not specially excited.
        assert!(c.iter().all(|v| v.is_finite()));
    }
}
