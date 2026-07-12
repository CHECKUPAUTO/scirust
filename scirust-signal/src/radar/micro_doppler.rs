//! Micro-Doppler analysis — the time–frequency signature of target micro-motion.
//!
//! Beyond a target's bulk translation, its moving parts — helicopter or drone
//! rotor blades, a tank's treads, a walking person's limbs — each add a small,
//! often periodic, Doppler modulation on top of the body return. Resolved in a
//! **time–frequency** representation (a spectrogram of the slow-time signal),
//! these show up as a modulated ridge whose shape and cadence identify the
//! target class — the basis of non-cooperative target recognition (NCTR).
//!
//! This module builds the spectrogram on the crate's power-of-two
//! [`fft`](crate::fft) with a Hann analysis window, then extracts the standard
//! micro-Doppler descriptors: the instantaneous-Doppler **ridge**, the **bulk**
//! (mean) Doppler, the modulation **bandwidth**, and the micro-motion **cadence**
//! (its repetition frequency). Dependency-free.

use crate::complex::Complex;
use crate::fft::fft;
use crate::windows::hanning;

/// The magnitude **spectrogram** of a complex slow-time `signal`: a Hann-windowed
/// short-time Fourier transform with window length `win_len` (a power of two) and
/// step `hop`. Returns one magnitude spectrum (length `win_len`, natural FFT bin
/// order) per frame. Empty if `win_len` is not a power of two, `hop` is zero, or
/// the signal is shorter than one window.
pub fn spectrogram(signal: &[Complex], win_len: usize, hop: usize) -> Vec<Vec<f64>> {
    if win_len == 0 || !win_len.is_power_of_two() || hop == 0 || signal.len() < win_len
    {
        return Vec::new();
    }
    let window = hanning(win_len);
    let mut frames = Vec::new();
    let mut start = 0;
    while start + win_len <= signal.len()
    {
        let mut buf: Vec<Complex> = (0..win_len)
            .map(|i| signal[start + i] * window[i])
            .collect();
        fft(&mut buf);
        frames.push(buf.iter().map(|c| c.mag()).collect());
        start += hop;
    }
    frames
}

/// The signed Doppler frequency of each FFT bin, in the same natural bin order as
/// [`spectrogram`]: bins `0..N/2` are positive, `N/2..N` fold to negative
/// frequencies, scaled by `sample_rate / win_len`.
pub fn bin_frequencies(win_len: usize, sample_rate: f64) -> Vec<f64> {
    (0..win_len)
        .map(|k| {
            let kk = if 2 * k < win_len
            {
                k as f64
            }
            else
            {
                k as f64 - win_len as f64
            };
            kk * sample_rate / win_len as f64
        })
        .collect()
}

/// The micro-Doppler **ridge**: the peak (dominant-Doppler) frequency of each
/// spectrogram frame, tracing the instantaneous Doppler over time. `freqs` maps
/// bins to frequencies (from [`bin_frequencies`]).
pub fn ridge(spectrogram: &[Vec<f64>], freqs: &[f64]) -> Vec<f64> {
    spectrogram
        .iter()
        .filter_map(|frame| {
            frame
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(k, _)| freqs[k])
        })
        .collect()
}

/// The **bulk Doppler** — the mean of the ridge, i.e. the body's translational
/// Doppler with the (zero-mean) micro-modulation averaged out. `0.0` for an
/// empty ridge.
pub fn mean_doppler(ridge: &[f64]) -> f64 {
    if ridge.is_empty()
    {
        return 0.0;
    }
    ridge.iter().sum::<f64>() / ridge.len() as f64
}

/// The micro-Doppler **bandwidth** — the peak-to-peak extent of the ridge, a
/// measure of how far the micro-motion swings the Doppler. `0.0` for an empty
/// ridge.
pub fn doppler_bandwidth(ridge: &[f64]) -> f64 {
    if ridge.is_empty()
    {
        return 0.0;
    }
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &r in ridge
    {
        lo = lo.min(r);
        hi = hi.max(r);
    }
    hi - lo
}

/// The micro-motion **cadence** — the repetition frequency (Hz) of the ridge's
/// oscillation, found as the strongest non-zero-lag peak of its autocorrelation.
/// `frame_rate` is the spectrogram frame rate (`sample_rate / hop`). `None` for a
/// ridge too short to hold a period.
pub fn cadence(ridge: &[f64], frame_rate: f64) -> Option<f64> {
    let n = ridge.len();
    if n < 4 || frame_rate <= 0.0
    {
        return None;
    }
    let mean = mean_doppler(ridge);
    let d: Vec<f64> = ridge.iter().map(|&r| r - mean).collect();
    let max_lag = n / 2;
    if max_lag < 2
    {
        return None;
    }
    let r: Vec<f64> = (0..=max_lag)
        .map(|lag| (0..(n - lag)).map(|i| d[i] * d[i + lag]).sum())
        .collect();
    // The autocorrelation's main lobe descends from lag 0; skip past it to the
    // first local minimum, then the highest peak beyond is the fundamental period
    // (the naive global max would otherwise land inside the broad main lobe).
    let mut lag = 1;
    while lag < max_lag && r[lag] < r[lag - 1]
    {
        lag += 1;
    }
    let (mut best_lag, mut best) = (0usize, f64::NEG_INFINITY);
    for (l, &rl) in r.iter().enumerate().skip(lag)
    {
        if rl > best
        {
            best = rl;
            best_lag = l;
        }
    }
    if best_lag == 0 || best <= 0.0
    {
        return None;
    }
    Some(frame_rate / best_lag as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// A rotating-scatterer slow-time signal: bulk Doppler `f_b` plus a
    /// sinusoidal micro-Doppler of amplitude `f_max` at rotation rate `f_rot`.
    /// Instantaneous frequency is `f_b + f_max·cos(2π f_rot t)`.
    fn rotor_signal(n: usize, fs: f64, f_b: f64, f_max: f64, f_rot: f64) -> Vec<Complex> {
        (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let phase = 2.0 * PI * f_b * t + (f_max / f_rot) * (2.0 * PI * f_rot * t).sin();
                Complex::cis(phase)
            })
            .collect()
    }

    #[test]
    fn spectrogram_shape_and_guards() {
        let sig = rotor_signal(2048, 1024.0, 100.0, 50.0, 2.0);
        let (win, hop) = (128, 32);
        let spec = spectrogram(&sig, win, hop);
        assert_eq!(spec.len(), (2048 - win) / hop + 1);
        assert_eq!(spec[0].len(), win);
        // Guards.
        assert!(spectrogram(&sig, 100, 32).is_empty()); // not power of two
        assert!(spectrogram(&sig, 128, 0).is_empty()); // zero hop
        assert!(spectrogram(&sig[..64], 128, 32).is_empty()); // shorter than window
    }

    #[test]
    fn mean_doppler_recovers_the_bulk_motion() {
        let (fs, f_b, f_max, f_rot) = (1024.0, 100.0, 50.0, 2.0);
        let sig = rotor_signal(2048, fs, f_b, f_max, f_rot);
        let spec = spectrogram(&sig, 128, 32);
        let freqs = bin_frequencies(128, fs);
        let r = ridge(&spec, &freqs);
        assert!(
            (mean_doppler(&r) - f_b).abs() < 12.0,
            "mean {} vs {f_b}",
            mean_doppler(&r)
        );
    }

    #[test]
    fn bandwidth_reflects_the_micro_motion_and_is_zero_for_a_tone() {
        let fs = 1024.0;
        // Rotor: ridge swings ±f_max about f_b, so peak-to-peak ≈ 2·f_max.
        let sig = rotor_signal(2048, fs, 100.0, 50.0, 2.0);
        let freqs = bin_frequencies(128, fs);
        let bw = doppler_bandwidth(&ridge(&spectrogram(&sig, 128, 32), &freqs));
        assert!((bw - 100.0).abs() < 25.0, "bandwidth {bw} vs ~100");
        // A pure tone (no micro-motion) has a flat ridge ⇒ ~zero bandwidth.
        let tone = rotor_signal(2048, fs, 100.0, 0.0, 2.0);
        let bw_tone = doppler_bandwidth(&ridge(&spectrogram(&tone, 128, 32), &freqs));
        assert!(bw_tone < 10.0, "tone bandwidth {bw_tone}");
    }

    #[test]
    fn cadence_recovers_the_rotation_frequency() {
        let (fs, f_rot) = (1024.0, 2.0);
        let sig = rotor_signal(2048, fs, 100.0, 50.0, f_rot);
        let (win, hop) = (128, 32);
        let spec = spectrogram(&sig, win, hop);
        let freqs = bin_frequencies(win, fs);
        let r = ridge(&spec, &freqs);
        let frame_rate = fs / hop as f64;
        let c = cadence(&r, frame_rate).unwrap();
        assert!((c - f_rot).abs() < 0.5, "cadence {c} vs {f_rot}");
    }

    #[test]
    fn pure_tone_ridge_sits_at_its_frequency() {
        let fs = 1024.0;
        let tone = rotor_signal(1024, fs, 120.0, 0.0, 1.0);
        let freqs = bin_frequencies(256, fs);
        let r = ridge(&spectrogram(&tone, 256, 64), &freqs);
        assert!(!r.is_empty());
        for &f in &r
        {
            assert!((f - 120.0).abs() < 6.0, "ridge freq {f} vs 120");
        }
        assert!(cadence(&r, fs / 64.0).is_none() || doppler_bandwidth(&r) < 6.0);
    }

    #[test]
    fn empty_ridge_degenerates_gracefully() {
        assert_eq!(mean_doppler(&[]), 0.0);
        assert_eq!(doppler_bandwidth(&[]), 0.0);
        assert!(cadence(&[], 16.0).is_none());
    }
}
