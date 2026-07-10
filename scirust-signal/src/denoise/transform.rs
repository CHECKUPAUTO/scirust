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

/// The orthogonal wavelet basis used by [`wavelet_denoise_with`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wavelet {
    /// Haar (Daubechies-1): piecewise-constant basis. Best for steps and abrupt
    /// changes; leaves blocky artefacts on smooth signals.
    Haar,
    /// Daubechies-4 (two vanishing moments): represents locally-linear signals
    /// compactly, so it denoises smooth data with far fewer artefacts than Haar.
    Db4,
}

impl Wavelet {
    /// Orthonormal analysis low-pass filter taps. The matching high-pass is the
    /// alternating flip `g[j] = (−1)^j · h[K−1−j]` (quadrature mirror).
    fn lowpass(self) -> Vec<f64> {
        match self
        {
            Wavelet::Haar => vec![1.0 / SQRT_2, 1.0 / SQRT_2],
            Wavelet::Db4 =>
            {
                let s3 = 3.0_f64.sqrt();
                let z = 4.0 * SQRT_2;
                vec![
                    (1.0 + s3) / z,
                    (3.0 + s3) / z,
                    (3.0 - s3) / z,
                    (1.0 - s3) / z,
                ]
            },
        }
    }
}

/// One periodized analysis step: split `x` (even length ≥ filter length) into
/// approximation and detail halves with the orthonormal filter pair derived
/// from `h`.
fn dwt_step(x: &[f64], h: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len();
    let half = n / 2;
    let k = h.len();
    let mut approx = vec![0.0; half];
    let mut detail = vec![0.0; half];
    for i in 0..half
    {
        let mut a = 0.0;
        let mut d = 0.0;
        for (j, &hj) in h.iter().enumerate()
        {
            let v = x[(2 * i + j) % n];
            let gj = if j % 2 == 0
            {
                h[k - 1 - j]
            }
            else
            {
                -h[k - 1 - j]
            };
            a += hj * v;
            d += gj * v;
        }
        approx[i] = a;
        detail[i] = d;
    }
    (approx, detail)
}

/// One periodized synthesis step — the exact transpose of [`dwt_step`], which for
/// an orthonormal filter bank is its inverse (perfect reconstruction).
fn idwt_step(approx: &[f64], detail: &[f64], h: &[f64]) -> Vec<f64> {
    let half = approx.len();
    let n = 2 * half;
    let k = h.len();
    let mut rec = vec![0.0; n];
    for i in 0..half
    {
        for (j, &hj) in h.iter().enumerate()
        {
            let gj = if j % 2 == 0
            {
                h[k - 1 - j]
            }
            else
            {
                -h[k - 1 - j]
            };
            rec[(2 * i + j) % n] += hj * approx[i] + gj * detail[i];
        }
    }
    rec
}

/// Wavelet-shrinkage denoising (VisuShrink) on a Haar basis.
///
/// This is the flagship general-purpose denoiser. It performs a multi-level
/// discrete wavelet transform, estimates the noise scale robustly from the finest
/// detail band (`σ = MAD / 0.6745`, Donoho-Johnstone), applies the *universal
/// threshold* `λ = σ·√(2 ln N)` to every detail coefficient, and reconstructs.
/// Small (noise) coefficients are removed while large (signal) coefficients — the
/// edges and transients — survive, so it denoises without the blur of a low-pass.
///
/// `levels = 0` selects a sensible depth automatically. For smooth signals prefer
/// [`wavelet_denoise_with`] and [`Wavelet::Db4`].
pub fn wavelet_denoise(signal: &[f64], levels: usize, mode: ThresholdMode) -> Vec<f64> {
    wavelet_denoise_with(signal, levels, mode, Wavelet::Haar)
}

/// [`wavelet_denoise`] on a caller-chosen orthogonal basis. Haar keeps steps
/// crisp; Daubechies-4 (two vanishing moments) fits smooth signals with far
/// fewer blocky artefacts.
pub fn wavelet_denoise_with(
    signal: &[f64],
    levels: usize,
    mode: ThresholdMode,
    wavelet: Wavelet,
) -> Vec<f64> {
    let n0 = signal.len();
    let h = wavelet.lowpass();
    if n0 < h.len()
    {
        return signal.to_vec();
    }
    let padded = pad_reflect_pow2(signal);
    let n = padded.len();
    let max_levels = n.trailing_zeros() as usize;
    let levels_req = if levels == 0
    {
        max_levels.saturating_sub(2).max(1)
    }
    else
    {
        levels.min(max_levels)
    };

    // Forward multi-level periodized DWT. Stop early if the running approximation
    // gets shorter than the filter (periodization would fold onto itself).
    let mut approx = padded;
    let mut detail_coeffs: Vec<Vec<f64>> = Vec::with_capacity(levels_req);
    for _ in 0..levels_req
    {
        if approx.len() < h.len().max(2)
        {
            break;
        }
        let (a, d) = dwt_step(&approx, &h);
        approx = a;
        detail_coeffs.push(d);
    }
    if detail_coeffs.is_empty()
    {
        return signal.to_vec();
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
    for detail in detail_coeffs.iter().rev()
    {
        approx = idwt_step(&approx, detail, &h);
    }
    approx[..n0].to_vec()
}

/// Power spectral subtraction with over-subtraction and a spectral floor
/// (Berouti-Schwartz-Makhoul 1979 — the power-domain refinement of Boll's
/// spectral subtraction) for additive white noise.
///
/// Subtracts the expected noise power from each bin's power and keeps the noisy
/// phase: gain `G_k = √(max(floor², 1 − over·P_n/|X_k|²))` with `P_n = N·σ²`.
/// `over_subtraction` (≥ 1, typically 1–3) trades residual "musical noise"
/// against signal distortion; `floor` (0..1, typically 0.01–0.1) keeps a little
/// of every bin to avoid the hollow, warbling artefact of hard zeroing. The
/// classic speech-enhancement front end, useful on any broadband-noise signal.
/// (Power subtraction attenuates low-SNR bins more gently than Boll's
/// magnitude-domain rule `1 − over·√P_n/|X_k|` would.)
pub fn spectral_subtraction(
    signal: &[f64],
    noise_std: f64,
    over_subtraction: f64,
    floor: f64,
) -> Vec<f64> {
    if noise_std <= 0.0
    {
        return signal.to_vec();
    }
    let n_pad = next_pow2(signal.len().max(2));
    let p_noise = n_pad as f64 * noise_std * noise_std;
    let over = over_subtraction.max(0.0);
    let fl = floor.clamp(0.0, 1.0);
    spectral_filter(signal, move |_, _, px| {
        if px <= 0.0
        {
            return fl;
        }
        (1.0 - over * p_noise / px).max(fl * fl).sqrt()
    })
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

    #[test]
    fn db4_filter_is_orthonormal() {
        let h = Wavelet::Db4.lowpass();
        let norm: f64 = h.iter().map(|&x| x * x).sum();
        assert!((norm - 1.0).abs() < 1.0e-12, "‖h‖² = {norm}");
        // Double-shift orthogonality: Σ h[j]·h[j+2] = 0.
        let shift2: f64 = h[0] * h[2] + h[1] * h[3];
        assert!(shift2.abs() < 1.0e-12, "shift-2 dot = {shift2}");
    }

    #[test]
    fn dwt_roundtrip_is_exact_for_both_wavelets() {
        let x: Vec<f64> = (0..64)
            .map(|i| (i as f64 * 0.37).sin() + 0.5 * (i as f64 * 0.11).cos())
            .collect();
        for wavelet in [Wavelet::Haar, Wavelet::Db4]
        {
            let h = wavelet.lowpass();
            // Single level.
            let (a, d) = dwt_step(&x, &h);
            let rec = idwt_step(&a, &d, &h);
            for (orig, r) in x.iter().zip(rec.iter())
            {
                assert!((orig - r).abs() < 1.0e-10, "{wavelet:?}: {orig} vs {r}");
            }
            // Three levels down and back.
            let (a1, d1) = dwt_step(&x, &h);
            let (a2, d2) = dwt_step(&a1, &h);
            let (a3, d3) = dwt_step(&a2, &h);
            let r2 = idwt_step(&a3, &d3, &h);
            let r1 = idwt_step(&r2, &d2, &h);
            let r0 = idwt_step(&r1, &d1, &h);
            for (orig, r) in x.iter().zip(r0.iter())
            {
                assert!((orig - r).abs() < 1.0e-10, "{wavelet:?} multilevel");
            }
        }
    }

    #[test]
    fn db4_beats_haar_on_smooth_signal() {
        let n = 512;
        let mut rng = Lcg::new(61);
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 3.0 * i as f64 / n as f64).sin())
            .collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.4 * rng.gauss()).collect();
        let haar = wavelet_denoise_with(&obs, 0, ThresholdMode::Soft, Wavelet::Haar);
        let db4 = wavelet_denoise_with(&obs, 0, ThresholdMode::Soft, Wavelet::Db4);
        let s_haar = snr_db(&clean, &haar);
        let s_db4 = snr_db(&clean, &db4);
        assert!(s_db4 > snr_db(&clean, &obs), "db4 must beat raw");
        assert!(s_db4 > s_haar, "db4 {s_db4} dB vs haar {s_haar} dB");
    }

    #[test]
    fn spectral_subtraction_parameters_are_live() {
        // With a huge over-subtraction factor every bin hits the spectral floor,
        // so the output must be exactly `floor × signal` — this pins BOTH knobs:
        // ignoring `over` leaves bins above the floor (non-uniform gain), and
        // flooring at `fl` instead of `fl²` rescales to √floor instead of floor.
        let mut rng = Lcg::new(71);
        let x: Vec<f64> = (0..256).map(|_| rng.gauss()).collect();
        let out = spectral_subtraction(&x, 1.0, 1.0e9, 0.5);
        for (a, b) in x.iter().zip(out.iter())
        {
            assert!(
                (0.5 * a - b).abs() < 1.0e-6,
                "expected 0.5·x, got {b} for x={a}"
            );
        }
        // And a larger over-subtraction must attenuate pure noise more.
        let mild = spectral_subtraction(&x, 1.0, 1.0, 0.05);
        let strong = spectral_subtraction(&x, 1.0, 3.0, 0.05);
        let rms = |v: &[f64]| (v.iter().map(|&s| s * s).sum::<f64>() / v.len() as f64).sqrt();
        assert!(
            rms(&strong) < rms(&mild),
            "over=3 should attenuate more: {} vs {}",
            rms(&strong),
            rms(&mild)
        );
    }

    #[test]
    fn spectral_subtraction_reduces_white_noise() {
        let n = 512;
        let mut rng = Lcg::new(67);
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 4.0 * i as f64 / n as f64).sin())
            .collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.4 * rng.gauss()).collect();
        let out = spectral_subtraction(&obs, 0.4, 1.0, 0.05);
        assert_eq!(out.len(), n);
        assert!(snr_db(&clean, &out) > snr_db(&clean, &obs));
    }
}
