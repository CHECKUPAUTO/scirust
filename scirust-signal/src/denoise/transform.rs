//! Transform-domain denoisers — Fourier filtering and wavelet shrinkage.
//!
//! The idea: move to a basis where *signal* is compact (few large coefficients)
//! and *noise* is spread out (many small coefficients), then attenuate the small
//! coefficients. Fourier suits stationary/tonal content; wavelets suit transients
//! and edges. Arbitrary-length inputs are reflection-padded to a power of two for
//! the radix-2 FFT / Haar transform and cropped back afterwards.

use super::{mad, next_pow2, pad_reflect_pow2};
use crate::Complex;
use crate::fft::{fft, ifft};
use core::f64::consts::SQRT_2;

/// Absolute physical frequency (Hz) of FFT bin `k` for an `n`-point transform.
fn bin_abs_freq(k: usize, n: usize, sample_rate: f64) -> f64 {
    let kk = if k <= n / 2 { k as f64 } else { (n - k) as f64 };
    kk * sample_rate / n as f64
}

/// Run a spectral gain function over the (padded) FFT of `signal` and return the
/// real inverse cropped to the original length. `gain(k, n, |X_k|²)` returns the
/// multiplier applied to bin `k`.
fn spectral_filter<F>(signal: &[f64], gain: F) -> Vec<f64>
where
    F: Fn(usize, usize, f64) -> f64,
{
    let n0 = signal.len();
    if n0 < 2
    {
        return signal.to_vec();
    }
    let padded = pad_reflect_pow2(signal);
    let n = padded.len();
    let mut buf: Vec<Complex> = padded.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft(&mut buf);
    for (k, c) in buf.iter_mut().enumerate()
    {
        let g = gain(k, n, c.mag_sq());
        *c = *c * g;
    }
    ifft(&mut buf);
    buf[..n0].iter().map(|c| c.re).collect()
}

/// Ideal (brick-wall) low-pass: keep every bin at or below `cutoff_hz`, zero the
/// rest. The bluntest transform-domain smoother; use it when the signal band and
/// the noise band are cleanly separated.
pub fn fft_lowpass(signal: &[f64], sample_rate: f64, cutoff_hz: f64) -> Vec<f64> {
    spectral_filter(signal, |k, n, _| {
        if bin_abs_freq(k, n, sample_rate) <= cutoff_hz
        {
            1.0
        }
        else
        {
            0.0
        }
    })
}

/// Ideal high-pass: keep every bin at or above `cutoff_hz`. Handy for stripping
/// slow baseline drift while leaving the fast signal untouched.
pub fn fft_highpass(signal: &[f64], sample_rate: f64, cutoff_hz: f64) -> Vec<f64> {
    spectral_filter(signal, |k, n, _| {
        if bin_abs_freq(k, n, sample_rate) >= cutoff_hz
        {
            1.0
        }
        else
        {
            0.0
        }
    })
}

/// Band-stop (notch) filter: zero the bins within `bandwidth` Hz of `center_hz`
/// (and its negative-frequency mirror). The standard cure for a single tonal
/// interferer such as mains hum.
pub fn notch_filter(signal: &[f64], sample_rate: f64, center_hz: f64, bandwidth: f64) -> Vec<f64> {
    let half_bw = bandwidth.abs() * 0.5;
    spectral_filter(signal, |k, n, _| {
        let f = bin_abs_freq(k, n, sample_rate);
        if (f - center_hz).abs() <= half_bw
        {
            0.0
        }
        else
        {
            1.0
        }
    })
}

/// Remove mains hum and its harmonics in one pass: notch `mains_hz`, `2·mains_hz`,
/// …, up to `n_harmonics`, each `bandwidth` Hz wide. Covers the 50/60 Hz power-line
/// interference that plagues ECG, EEG and industrial acquisition.
pub fn remove_mains_hum(
    signal: &[f64],
    sample_rate: f64,
    mains_hz: f64,
    n_harmonics: usize,
    bandwidth: f64,
) -> Vec<f64> {
    let half_bw = bandwidth.abs() * 0.5;
    let nyquist = sample_rate * 0.5;
    spectral_filter(signal, |k, n, _| {
        let f = bin_abs_freq(k, n, sample_rate);
        for h in 1..=n_harmonics
        {
            let center = mains_hz * h as f64;
            if center > nyquist
            {
                break;
            }
            if (f - center).abs() <= half_bw
            {
                return 0.0;
            }
        }
        1.0
    })
}

/// Wiener filter for additive white noise of known standard deviation. Applies the
/// minimum-mean-square gain `G_k = max(0, (|X_k|² − P_n) / |X_k|²)` per bin, where
/// the expected white-noise power per bin is `P_n = N·σ²`. Unlike a brick wall it
/// keeps partially-buried bins in proportion to their signal-to-noise ratio.
///
/// Pair it with [`super::detect::estimate_noise_std`] to get `noise_std` for free.
pub fn wiener_white(signal: &[f64], noise_std: f64) -> Vec<f64> {
    if noise_std <= 0.0
    {
        return signal.to_vec();
    }
    let n_pad = next_pow2(signal.len().max(2));
    let p_noise = n_pad as f64 * noise_std * noise_std;
    spectral_filter(signal, move |_, _, px| {
        if px <= 0.0
        {
            return 0.0;
        }
        ((px - p_noise) / px).max(0.0)
    })
}

/// Which shrinkage rule wavelet thresholding applies to detail coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdMode {
    /// Hard threshold: keep coefficients above the threshold, zero the rest.
    /// Preserves peak amplitude but can leave visible artefacts.
    Hard,
    /// Soft threshold: shrink every coefficient toward zero by the threshold.
    /// Smoother reconstruction; the Donoho-Johnstone default.
    Soft,
}

fn apply_threshold(x: f64, t: f64, mode: ThresholdMode) -> f64 {
    match mode
    {
        ThresholdMode::Hard =>
        {
            if x.abs() > t
            {
                x
            }
            else
            {
                0.0
            }
        },
        ThresholdMode::Soft =>
        {
            let shrunk = x.abs() - t;
            if shrunk > 0.0
            {
                x.signum() * shrunk
            }
            else
            {
                0.0
            }
        },
    }
}

/// Wavelet-shrinkage denoising (VisuShrink) on a Haar basis.
///
/// This is the flagship general-purpose denoiser. It performs a multi-level Haar
/// discrete wavelet transform, estimates the noise scale robustly from the finest
/// detail band (`σ = MAD / 0.6745`, Donoho-Johnstone), applies the *universal
/// threshold* `λ = σ·√(2 ln N)` to every detail coefficient, and reconstructs.
/// Small (noise) coefficients are removed while large (signal) coefficients — the
/// edges and transients — survive, so it denoises without the blur of a low-pass.
///
/// `levels = 0` selects a sensible depth automatically.
pub fn wavelet_denoise(signal: &[f64], levels: usize, mode: ThresholdMode) -> Vec<f64> {
    let n0 = signal.len();
    if n0 < 2
    {
        return signal.to_vec();
    }
    let padded = pad_reflect_pow2(signal);
    let n = padded.len();
    let max_levels = n.trailing_zeros() as usize;
    let levels_eff = if levels == 0
    {
        max_levels.saturating_sub(2).max(1)
    }
    else
    {
        levels.min(max_levels)
    };

    // Forward multi-level Haar DWT: `work[..cur_len]` holds the running approximation.
    let mut work = padded;
    let mut detail_coeffs: Vec<Vec<f64>> = Vec::with_capacity(levels_eff);
    let mut cur_len = n;
    for _ in 0..levels_eff
    {
        let half = cur_len / 2;
        let mut approx = vec![0.0; half];
        let mut detail = vec![0.0; half];
        for i in 0..half
        {
            let a = work[2 * i];
            let b = work[2 * i + 1];
            approx[i] = (a + b) / SQRT_2;
            detail[i] = (a - b) / SQRT_2;
        }
        work[..half].copy_from_slice(&approx);
        detail_coeffs.push(detail);
        cur_len = half;
    }

    // Robust noise scale from the finest detail band, then the universal threshold.
    let sigma = mad(&detail_coeffs[0]) / 0.6745;
    let thresh = sigma * (2.0 * (n as f64).ln()).sqrt();
    for detail in detail_coeffs.iter_mut()
    {
        for d in detail.iter_mut()
        {
            *d = apply_threshold(*d, thresh, mode);
        }
    }

    // Inverse DWT from the coarsest approximation upward.
    let mut approx = work[..cur_len].to_vec();
    for detail in detail_coeffs.iter().rev()
    {
        let half = detail.len();
        let mut rec = vec![0.0; 2 * half];
        for i in 0..half
        {
            let a = approx[i];
            let d = detail[i];
            rec[2 * i] = (a + d) / SQRT_2;
            rec[2 * i + 1] = (a - d) / SQRT_2;
        }
        approx = rec;
    }
    approx[..n0].to_vec()
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{Lcg, snr_db};
    use super::*;
    use core::f64::consts::PI;

    #[test]
    fn lowpass_removes_high_freq_noise() {
        let n = 512;
        let fs = 512.0;
        let mut rng = Lcg::new(11);
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / fs).sin())
            .collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.5 * rng.gauss()).collect();
        let out = fft_lowpass(&obs, fs, 20.0);
        assert!(snr_db(&clean, &out) > snr_db(&clean, &obs));
    }

    #[test]
    fn notch_kills_tonal_interference() {
        let n = 512;
        let fs = 512.0;
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / fs).sin())
            .collect();
        // 50 Hz interferer.
        let obs: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, &c)| c + 0.8 * (2.0 * PI * 50.0 * i as f64 / fs).sin())
            .collect();
        let out = notch_filter(&obs, fs, 50.0, 4.0);
        assert!(snr_db(&clean, &out) > snr_db(&clean, &obs) + 10.0);
    }

    #[test]
    fn mains_hum_and_harmonics_removed() {
        let n = 1024;
        let fs = 1000.0;
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 7.0 * i as f64 / fs).sin())
            .collect();
        let obs: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let t = i as f64 / fs;
                c + 0.6 * (2.0 * PI * 50.0 * t).sin() + 0.3 * (2.0 * PI * 100.0 * t).sin()
            })
            .collect();
        let out = remove_mains_hum(&obs, fs, 50.0, 3, 3.0);
        assert!(snr_db(&clean, &out) > snr_db(&clean, &obs) + 8.0);
    }

    #[test]
    fn wiener_reduces_white_noise() {
        let n = 512;
        let mut rng = Lcg::new(13);
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 4.0 * i as f64 / n as f64).sin())
            .collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.4 * rng.gauss()).collect();
        let out = wiener_white(&obs, 0.4);
        assert!(snr_db(&clean, &out) > snr_db(&clean, &obs));
    }

    #[test]
    fn wavelet_denoise_beats_raw() {
        let n = 512;
        let mut rng = Lcg::new(17);
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 3.0 * i as f64 / n as f64).sin())
            .collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.4 * rng.gauss()).collect();
        let out = wavelet_denoise(&obs, 0, ThresholdMode::Soft);
        assert_eq!(out.len(), n);
        assert!(snr_db(&clean, &out) > snr_db(&clean, &obs));
    }

    #[test]
    fn wavelet_preserves_step_edge() {
        // A step with noise: wavelets should keep the edge sharp.
        let n = 256;
        let mut rng = Lcg::new(19);
        let clean: Vec<f64> = (0..n).map(|i| if i < 128 { 0.0 } else { 2.0 }).collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.3 * rng.gauss()).collect();
        let out = wavelet_denoise(&obs, 0, ThresholdMode::Hard);
        assert!(snr_db(&clean, &out) > snr_db(&clean, &obs));
    }
}
