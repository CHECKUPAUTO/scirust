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

pub(crate) fn apply_threshold(x: f64, t: f64, mode: ThresholdMode) -> f64 {
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
///
/// Naming follows the tap count (Daubechies-`K` has `K` taps and `K/2` vanishing
/// moments): `Db4` = 2 moments, `Db6` = 3, `Db8` = 4. More vanishing moments
/// represent smoother signals with fewer large coefficients — at the price of
/// wider support (more ringing around isolated discontinuities).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wavelet {
    /// Haar (Daubechies-2 in tap count): piecewise-constant basis. Best for steps
    /// and abrupt changes; leaves blocky artefacts on smooth signals.
    Haar,
    /// Daubechies-4 (two vanishing moments): represents locally-linear signals
    /// compactly, so it denoises smooth data with far fewer artefacts than Haar.
    Db4,
    /// Daubechies-6 (three vanishing moments): annihilates locally-quadratic
    /// structure; the workhorse for smooth data.
    Db6,
    /// Daubechies-8 (four vanishing moments): the smoothest basis here; best for
    /// very smooth signals, worst around sharp steps.
    Db8,
}

/// Daubechies-6 scaling taps (extremal phase), derived by spectral factorization
/// of the 3-vanishing-moment Daubechies polynomial; the identities `Σh = √2`,
/// `‖h‖ = 1`, double-shift orthogonality and 3 vanishing moments are re-verified
/// to ~1e-12 by unit test, which pins these constants independently.
const DB6_H: [f64; 6] = [
    0.3326705529500826,
    0.8068915093110924,
    0.4598775021184915,
    -0.1350110200102546,
    -0.0854412738820267,
    0.0352262918857095,
];

/// Daubechies-8 scaling taps (extremal phase); same derivation and the same
/// identity-based unit-test verification as [`DB6_H`], with 4 vanishing moments.
const DB8_H: [f64; 8] = [
    0.2303778133088964,
    0.7148465705529153,
    0.6308807679298589,
    -0.0279837694168594,
    -0.1870348117190928,
    0.0308413818355607,
    0.0328830116668852,
    -0.0105974017850690,
];

impl Wavelet {
    /// Orthonormal analysis low-pass filter taps. The matching high-pass is the
    /// alternating flip `g[j] = (−1)^j · h[K−1−j]` (quadrature mirror).
    pub(crate) fn lowpass(self) -> Vec<f64> {
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
            Wavelet::Db6 => DB6_H.to_vec(),
            Wavelet::Db8 => DB8_H.to_vec(),
        }
    }

    /// Number of vanishing moments of the wavelet (high-pass) filter.
    pub fn vanishing_moments(self) -> usize {
        match self
        {
            Wavelet::Haar => 1,
            Wavelet::Db4 => 2,
            Wavelet::Db6 => 3,
            Wavelet::Db8 => 4,
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

/// Forward multi-level periodized DWT of the reflection-padded signal. Returns
/// `(coarsest approximation, detail bands finest-first, padded length)`, or `None`
/// when the signal is too short to transform. Stops early if the running
/// approximation gets shorter than the filter (periodization would fold onto
/// itself).
#[allow(clippy::type_complexity)]
pub(crate) fn dwt_forward(
    signal: &[f64],
    levels: usize,
    h: &[f64],
) -> Option<(Vec<f64>, Vec<Vec<f64>>, usize)> {
    if signal.len() < h.len()
    {
        return None;
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
    let mut approx = padded;
    let mut detail_coeffs: Vec<Vec<f64>> = Vec::with_capacity(levels_req);
    for _ in 0..levels_req
    {
        if approx.len() < h.len().max(2)
        {
            break;
        }
        let (a, d) = dwt_step(&approx, h);
        approx = a;
        detail_coeffs.push(d);
    }
    if detail_coeffs.is_empty()
    {
        return None;
    }
    Some((approx, detail_coeffs, n))
}

/// Inverse multi-level DWT, cropped back to the original length.
pub(crate) fn dwt_inverse(
    mut approx: Vec<f64>,
    detail_coeffs: &[Vec<f64>],
    h: &[f64],
    n0: usize,
) -> Vec<f64> {
    for detail in detail_coeffs.iter().rev()
    {
        approx = idwt_step(&approx, detail, h);
    }
    approx[..n0].to_vec()
}

/// [`wavelet_denoise`] on a caller-chosen orthogonal basis. Haar keeps steps
/// crisp; the Daubechies bases (more vanishing moments) fit smooth signals with
/// far fewer blocky artefacts.
pub fn wavelet_denoise_with(
    signal: &[f64],
    levels: usize,
    mode: ThresholdMode,
    wavelet: Wavelet,
) -> Vec<f64> {
    let h = wavelet.lowpass();
    let Some((approx, mut detail_coeffs, n)) = dwt_forward(signal, levels, &h)
    else
    {
        return signal.to_vec();
    };

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
    dwt_inverse(approx, &detail_coeffs, &h, signal.len())
}

/// Wavelet denoising with a **per-level SURE threshold** (SureShrink,
/// Donoho-Johnstone 1995) and soft shrinkage.
///
/// The universal threshold `σ√(2 ln N)` used by [`wavelet_denoise_with`] is
/// minimax over the sparsest case and tends to over-smooth signals whose energy
/// spreads over many coefficients. SureShrink instead picks, in each detail band
/// separately, the threshold minimizing **Stein's Unbiased Risk Estimate**
/// `SURE(t) = m − 2·#{|uᵢ| ≤ t} + Σ min(uᵢ², t²)` (an unbiased estimate of the
/// soft-thresholding mean-square error that needs no clean reference), falling
/// back to the universal threshold in bands the Donoho-Johnstone sparsity test
/// flags as too sparse for SURE to be reliable (the "hybrid" scheme).
pub fn wavelet_denoise_sure(signal: &[f64], levels: usize, wavelet: Wavelet) -> Vec<f64> {
    let h = wavelet.lowpass();
    let Some((approx, mut detail_coeffs, _)) = dwt_forward(signal, levels, &h)
    else
    {
        return signal.to_vec();
    };

    let sigma = mad(&detail_coeffs[0]) / 0.6745;
    if sigma > 0.0
    {
        for detail in detail_coeffs.iter_mut()
        {
            let thresh = sigma * sure_threshold_normalized(detail, sigma);
            for d in detail.iter_mut()
            {
                *d = apply_threshold(*d, thresh, ThresholdMode::Soft);
            }
        }
    }
    dwt_inverse(approx, &detail_coeffs, &h, signal.len())
}

/// SureShrink hybrid threshold for one detail band, in units of `sigma`.
fn sure_threshold_normalized(detail: &[f64], sigma: f64) -> f64 {
    let m = detail.len();
    if m < 2
    {
        return 0.0;
    }
    let mf = m as f64;
    let t_univ = (2.0 * mf.ln()).sqrt();

    // Normalized magnitudes, ascending. Total order (NaN-safe): a partial_cmp
    // comparator is inconsistent on NaN and makes modern Rust sorts panic.
    let mut a: Vec<f64> = detail.iter().map(|&d| (d / sigma).abs()).collect();
    a.sort_by(|x, y| x.total_cmp(y));

    // Donoho-Johnstone sparsity test: too little energy above the noise floor
    // means SURE's variance dominates — fall back to the universal threshold.
    let energy: f64 = a.iter().map(|&u| u * u).sum();
    let s_d = (energy - mf) / mf;
    let gamma = mf.log2().powf(1.5) / mf.sqrt();
    if s_d <= gamma
    {
        return t_univ;
    }

    // SURE(t) over candidate thresholds t ∈ {0} ∪ {|uᵢ|} capped at the universal
    // threshold, using prefix sums of squares: for t = a[k] (k+1 values ≤ t),
    // SURE = m − 2(k+1) + prefix_sq[k+1] + (m−k−1)·t².
    let mut prefix_sq = vec![0.0; m + 1];
    for (i, &u) in a.iter().enumerate()
    {
        prefix_sq[i + 1] = prefix_sq[i] + u * u;
    }
    let mut best_t = 0.0;
    let mut best_risk = mf; // SURE(0) = m
    for (k, &t) in a.iter().enumerate()
    {
        if t > t_univ
        {
            break;
        }
        let risk = mf - 2.0 * (k + 1) as f64 + prefix_sq[k + 1] + (m - k - 1) as f64 * t * t;
        if risk < best_risk
        {
            best_risk = risk;
            best_t = t;
        }
    }
    best_t
}

/// **Cycle spinning** — translation-invariant denoising (Coifman-Donoho 1995).
///
/// The decimated wavelet transform is not shift-invariant: displacing the input by
/// a single sample changes which coefficients straddle a transient, so thresholding
/// leaves a different pattern of pseudo-Gibbs ripples for every alignment. Cycle
/// spinning averages the artefact away — "shift, denoise, unshift, average": for
/// `n_shifts` evenly spaced circular shifts `s_k = ⌊k·n/n_shifts⌋` the signal is
/// rotated left by `s_k`, denoised, rotated back right by `s_k`, and the results
/// are averaged. Averaging over *all* `n` shifts is equivalent to denoising in the
/// undecimated (stationary) wavelet transform; a handful of shifts (8–16) already
/// captures most of that gain at a fraction of the cost.
///
/// `denoiser` must be length-preserving, as every denoiser in this module is.
/// Repeated shift amounts (which arise when `n_shifts > signal.len()`) are
/// deduplicated so each distinct alignment contributes exactly once. With
/// `n_shifts <= 1` — or an input too short to rotate — this is exactly
/// `denoiser(signal)`.
///
/// Choosing `n_shifts`: prefer a count that is **odd** (or otherwise does not
/// divide `signal.len()`). When `n_shifts` divides `n` every shift is a multiple
/// of `n/n_shifts`; on the power-of-two lengths the wavelet transform works with,
/// such shifts are all highly even, the finest-scale coefficient alignments never
/// change, and most of the artefact-cancelling benefit is silently lost. An odd
/// count like 15 or 31 spreads the shifts across all dyadic alignments.
pub fn cycle_spin(
    signal: &[f64],
    n_shifts: usize,
    denoiser: impl Fn(&[f64]) -> Vec<f64>,
) -> Vec<f64> {
    let n = signal.len();
    if n_shifts <= 1 || n < 2
    {
        return denoiser(signal);
    }
    // Distinct, evenly spaced shift amounts s_k = ⌊k·n/n_shifts⌋. The sequence is
    // non-decreasing in k, so consecutive dedup removes all repeats.
    let mut shifts: Vec<usize> = (0..n_shifts)
        .map(|k| ((k as u128 * n as u128) / n_shifts as u128) as usize)
        .collect();
    shifts.dedup();
    let mut acc = vec![0.0; n];
    for &s in &shifts
    {
        // Rotate LEFT by s: rotated[i] = signal[(i + s) mod n].
        let mut rotated = Vec::with_capacity(n);
        rotated.extend_from_slice(&signal[s..]);
        rotated.extend_from_slice(&signal[..s]);
        let den = denoiser(&rotated);
        // Rotate RIGHT by s and accumulate: output sample j came from den[(j − s) mod n].
        for (j, a) in acc.iter_mut().enumerate()
        {
            *a += den[(j + n - s) % n];
        }
    }
    let scale = 1.0 / shifts.len() as f64;
    for a in acc.iter_mut()
    {
        *a *= scale;
    }
    acc
}

/// Translation-invariant wavelet denoising: [`cycle_spin`] applied to
/// [`wavelet_denoise_with`].
///
/// Use this instead of the plain decimated denoiser whenever the signal contains
/// sharp steps or isolated transients: the shift-averaging suppresses the
/// alignment-dependent pseudo-Gibbs oscillations that the decimated transform
/// leaves around discontinuities (Coifman-Donoho 1995), typically buying one to
/// three dB of SNR at edges for an `n_shifts`-fold cost. An **odd** `n_shifts` of
/// 15–31 is usually enough (see the [`cycle_spin`] note on why a divisor of the
/// padded length wastes shifts); `n_shifts <= 1` reduces to
/// `wavelet_denoise_with` exactly.
pub fn wavelet_denoise_ti(
    signal: &[f64],
    levels: usize,
    mode: ThresholdMode,
    wavelet: Wavelet,
    n_shifts: usize,
) -> Vec<f64> {
    cycle_spin(signal, n_shifts, |x| {
        wavelet_denoise_with(x, levels, mode, wavelet)
    })
}

/// Wavelet denoising with a **level-dependent noise scale** for colored noise
/// (Johnstone-Silverman 1997).
///
/// Classical VisuShrink ([`wavelet_denoise_with`]) estimates a *single* noise scale
/// from the finest detail band and applies one universal threshold everywhere. That
/// is correct for white noise, whose power is the same at every scale — but for
/// **colored** noise (pink, brown, AR-correlated) the noise power grows toward the
/// coarse scales, so the finest-band σ badly *under-thresholds* the coarse bands
/// and low-frequency noise survives. This is exactly the regime
/// [`super::denoise_auto`] routes to wavelets. The fix (Johnstone-Silverman): make
/// the wavelet transform whiten the noise per band and estimate a separate scale in
/// each — `σ_j = MAD(d_j)/0.6745`, threshold `λ_j = σ_j·√(2 ln N)` with `N` the
/// padded length. For truly white noise all the `σ_j` agree and this reduces to
/// the classical rule (up to estimation error).
///
/// Caveat: the per-band MAD assumes the signal is *sparse within each band* (steps,
/// bumps, transients). A sustained tone fills its resonant band with uniformly
/// large coefficients, so the band median mistakes the tone for noise and the
/// threshold wipes it out — for dense tonal content keep the global rule of
/// [`wavelet_denoise_with`] or [`wavelet_denoise_sure`].
pub fn wavelet_denoise_leveldep(
    signal: &[f64],
    levels: usize,
    mode: ThresholdMode,
    wavelet: Wavelet,
) -> Vec<f64> {
    let h = wavelet.lowpass();
    let Some((approx, mut detail_coeffs, n)) = dwt_forward(signal, levels, &h)
    else
    {
        return signal.to_vec();
    };

    let universal = (2.0 * (n as f64).ln()).sqrt();
    for detail in detail_coeffs.iter_mut()
    {
        // Robust per-band noise scale: MAD is immune to the minority of large
        // signal coefficients sharing the band with the noise.
        let sigma_j = mad(detail) / 0.6745;
        let thresh = sigma_j * universal;
        for d in detail.iter_mut()
        {
            *d = apply_threshold(*d, thresh, mode);
        }
    }
    dwt_inverse(approx, &detail_coeffs, &h, signal.len())
}

/// **BayesShrink** wavelet denoising (Chang-Yu-Vetterli 2000).
///
/// Instead of the one-size-fits-all universal threshold, BayesShrink models each
/// detail band as signal + noise with a (generalized) Gaussian signal prior and
/// picks the threshold that is near-optimal for the *estimated* signal strength of
/// that band: with `σ²` the noise variance (MAD on the finest band) and
/// `σ_y² = mean(d²)` the observed band variance, the signal scale is
/// `σ_x = √(max(σ_y² − σ², 0))` and the band threshold is `t_j = σ²/σ_x` (soft
/// shrinkage). Strong-signal bands get a small threshold and are barely touched;
/// bands indistinguishable from pure noise (`σ_x = 0`) are zeroed outright. In
/// practice this is markedly less aggressive than VisuShrink on dense signals
/// while remaining fully automatic — `levels = 0` picks the depth exactly like
/// the other wavelet denoisers here.
pub fn wavelet_denoise_bayes(signal: &[f64], levels: usize, wavelet: Wavelet) -> Vec<f64> {
    let h = wavelet.lowpass();
    let Some((approx, mut detail_coeffs, _)) = dwt_forward(signal, levels, &h)
    else
    {
        return signal.to_vec();
    };

    // Noise variance from the finest band (Donoho MAD estimator).
    let sigma = mad(&detail_coeffs[0]) / 0.6745;
    let sigma_sq = sigma * sigma;
    for detail in detail_coeffs.iter_mut()
    {
        let m = detail.len() as f64;
        let sigma_y_sq = detail.iter().map(|&d| d * d).sum::<f64>() / m.max(1.0);
        let sigma_x = (sigma_y_sq - sigma_sq).max(0.0).sqrt();
        if sigma_x > 0.0
        {
            let thresh = sigma_sq / sigma_x;
            for d in detail.iter_mut()
            {
                *d = apply_threshold(*d, thresh, ThresholdMode::Soft);
            }
        }
        else
        {
            // The band carries no detectable signal energy: kill it entirely
            // (equivalent to an infinite threshold).
            for d in detail.iter_mut()
            {
                *d = 0.0;
            }
        }
    }
    dwt_inverse(approx, &detail_coeffs, &h, signal.len())
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
    fn daubechies_filters_satisfy_their_defining_identities() {
        // Σh = √2, ‖h‖ = 1, double-shift orthogonality, and p vanishing moments —
        // together these pin the (extremal-phase) Daubechies constants, so this
        // test validates the hardcoded DB6/DB8 tables independently of their
        // derivation.
        for wavelet in [Wavelet::Haar, Wavelet::Db4, Wavelet::Db6, Wavelet::Db8]
        {
            let h = wavelet.lowpass();
            let k = h.len();
            let sum: f64 = h.iter().sum();
            assert!((sum - SQRT_2).abs() < 1.0e-10, "{wavelet:?}: Σh = {sum}");
            let norm: f64 = h.iter().map(|&x| x * x).sum();
            assert!((norm - 1.0).abs() < 1.0e-10, "{wavelet:?}: ‖h‖² = {norm}");
            for m in 1..k / 2
            {
                let dot: f64 = (0..k - 2 * m).map(|j| h[j] * h[j + 2 * m]).sum();
                assert!(
                    dot.abs() < 1.0e-10,
                    "{wavelet:?}: shift-{} dot = {dot}",
                    2 * m
                );
            }
            // Vanishing moments of the quadrature-mirror high-pass: Σ g[j]·j^p = 0.
            let g: Vec<f64> = (0..k)
                .map(|j| {
                    if j % 2 == 0
                    {
                        h[k - 1 - j]
                    }
                    else
                    {
                        -h[k - 1 - j]
                    }
                })
                .collect();
            for p in 0..wavelet.vanishing_moments()
            {
                let moment: f64 = g
                    .iter()
                    .enumerate()
                    .map(|(j, &gj)| gj * (j as f64).powi(p as i32))
                    .sum();
                assert!(moment.abs() < 1.0e-9, "{wavelet:?}: moment {p} = {moment}");
            }
        }
    }

    #[test]
    fn dwt_roundtrip_is_exact_for_both_wavelets() {
        let x: Vec<f64> = (0..64)
            .map(|i| (i as f64 * 0.37).sin() + 0.5 * (i as f64 * 0.11).cos())
            .collect();
        for wavelet in [Wavelet::Haar, Wavelet::Db4, Wavelet::Db6, Wavelet::Db8]
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
    fn db8_beats_db4_on_very_smooth_signal() {
        let n = 512;
        let mut rng = Lcg::new(83);
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 2.0 * i as f64 / n as f64).sin())
            .collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.4 * rng.gauss()).collect();
        let db4 = wavelet_denoise_with(&obs, 0, ThresholdMode::Soft, Wavelet::Db4);
        let db8 = wavelet_denoise_with(&obs, 0, ThresholdMode::Soft, Wavelet::Db8);
        assert!(
            snr_db(&clean, &db8) > snr_db(&clean, &obs),
            "db8 must beat raw"
        );
        assert!(
            snr_db(&clean, &db8) > snr_db(&clean, &db4) - 0.5,
            "db8 {} dB should not trail db4 {} dB on a smooth signal",
            snr_db(&clean, &db8),
            snr_db(&clean, &db4)
        );
    }

    #[test]
    fn sure_beats_universal_on_dense_signal() {
        // A signal with energy spread over many coefficients (two tones, one
        // fast): the universal threshold over-smooths it, SURE adapts down.
        let n = 1024;
        let mut rng = Lcg::new(89);
        let clean: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 5.0 * t).sin() + 0.7 * (2.0 * PI * 60.0 * t).sin()
            })
            .collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.3 * rng.gauss()).collect();
        let universal = wavelet_denoise_with(&obs, 0, ThresholdMode::Soft, Wavelet::Db8);
        let sure = wavelet_denoise_sure(&obs, 0, Wavelet::Db8);
        let s_sure = snr_db(&clean, &sure);
        let s_univ = snr_db(&clean, &universal);
        assert!(s_sure > snr_db(&clean, &obs), "SURE must beat raw");
        assert!(s_sure > s_univ, "SURE {s_sure} dB vs universal {s_univ} dB");
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

    fn noisy_step(n: usize, edge: usize, sigma: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Lcg::new(seed);
        let clean: Vec<f64> = (0..n).map(|i| if i < edge { 0.0 } else { 2.0 }).collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + sigma * rng.gauss()).collect();
        (clean, obs)
    }

    #[test]
    fn cycle_spin_one_shift_equals_plain_denoiser() {
        let (_, obs) = noisy_step(256, 100, 0.3, 23);
        let plain = wavelet_denoise(&obs, 0, ThresholdMode::Soft);
        let spun = cycle_spin(&obs, 1, |x| wavelet_denoise(x, 0, ThresholdMode::Soft));
        assert_eq!(
            spun, plain,
            "n_shifts = 1 must be exactly the plain denoiser"
        );
        let spun0 = cycle_spin(&obs, 0, |x| wavelet_denoise(x, 0, ThresholdMode::Soft));
        assert_eq!(
            spun0, plain,
            "n_shifts = 0 must be exactly the plain denoiser"
        );
    }

    #[test]
    fn cycle_spin_shift_bookkeeping_and_dedupe() {
        // With an identity denoiser the rotate-left / rotate-right pair must cancel
        // exactly for every shift, so the average returns the input.
        let x = [1.0, -2.0, 3.5, 0.25, -1.75];
        let calls = core::cell::Cell::new(0usize);
        let out = cycle_spin(&x, 5, |s| {
            calls.set(calls.get() + 1);
            s.to_vec()
        });
        assert_eq!(calls.get(), 5);
        for (a, b) in x.iter().zip(out.iter())
        {
            assert!(
                (a - b).abs() < 1.0e-12,
                "identity round-trip broke: {a} vs {b}"
            );
        }
        // n_shifts > n: the repeated shift amounts ⌊k·n/n_shifts⌋ must be
        // deduplicated, so a length-4 signal sees exactly 4 distinct alignments.
        let y = [1.0, 2.0, 3.0, 4.0];
        let calls = core::cell::Cell::new(0usize);
        let out = cycle_spin(&y, 8, |s| {
            calls.set(calls.get() + 1);
            s.to_vec()
        });
        assert_eq!(calls.get(), 4, "expected one call per distinct shift");
        for (a, b) in y.iter().zip(out.iter())
        {
            assert!((a - b).abs() < 1.0e-12);
        }
    }

    #[test]
    fn ti_beats_plain_wavelet_on_noisy_step() {
        // The step edges are where shift-variance artefacts live: pseudo-Gibbs
        // ripples around each edge depend on the alignment, and averaging over
        // shifts cancels them. Two constructive details keep the comparison
        // honest: the signal returns to its baseline (no wrap-around seam, which
        // the periodized transform would otherwise hand to shift 0 perfectly
        // aligned), and the edges sit at ODD indices so the decimated Haar has no
        // lucky dyadic alignment to hide behind. n_shifts = 31 is odd, so the
        // shifts cover all fine-scale alignments (see the cycle_spin doc note).
        let n = 256;
        let mut rng = Lcg::new(29);
        let clean: Vec<f64> = (0..n)
            .map(|i| if (81..173).contains(&i) { 2.0 } else { 0.0 })
            .collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.3 * rng.gauss()).collect();
        let plain = wavelet_denoise_with(&obs, 0, ThresholdMode::Hard, Wavelet::Haar);
        let ti = wavelet_denoise_ti(&obs, 0, ThresholdMode::Hard, Wavelet::Haar, 31);
        let s_raw = snr_db(&clean, &obs);
        let s_plain = snr_db(&clean, &plain);
        let s_ti = snr_db(&clean, &ti);
        assert!(s_plain > s_raw, "plain must beat raw to begin with");
        assert!(
            s_ti >= s_plain + 0.5,
            "cycle spinning gained only {:.2} dB (plain {s_plain:.2}, TI {s_ti:.2})",
            s_ti - s_plain
        );
    }

    #[test]
    fn ti_parameters_are_live() {
        let (_, obs) = noisy_step(256, 100, 0.3, 31);
        // n_shifts must matter: averaging 8 alignments cannot reproduce a single one.
        let one = wavelet_denoise_ti(&obs, 0, ThresholdMode::Hard, Wavelet::Haar, 1);
        let eight = wavelet_denoise_ti(&obs, 0, ThresholdMode::Hard, Wavelet::Haar, 8);
        assert_ne!(one, eight, "n_shifts is ignored");
        // With n_shifts = 1 the wrapper must equal wavelet_denoise_with for the SAME
        // (levels, mode, wavelet) — a levels/n_shifts transposition (both usize)
        // would compile silently and fail this equality.
        let direct = wavelet_denoise_with(&obs, 3, ThresholdMode::Hard, Wavelet::Db4);
        let wrapped = wavelet_denoise_ti(&obs, 3, ThresholdMode::Hard, Wavelet::Db4, 1);
        assert_eq!(wrapped, direct, "levels/mode/wavelet not plumbed through");
    }

    #[test]
    fn leveldep_beats_global_threshold_on_colored_noise() {
        // AR(1) noise, x_k = 0.9·x_{k−1} + w_k: its power grows toward low
        // frequencies, so the finest-band σ of classical VisuShrink badly
        // under-thresholds the coarse bands. Per-level scales fix exactly that.
        let n = 1024;
        let mut rng = Lcg::new(41);
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 3.0 * i as f64 / n as f64).sin())
            .collect();
        let mut ar = 0.0;
        let obs: Vec<f64> = clean
            .iter()
            .map(|&c| {
                ar = 0.9 * ar + 0.15 * rng.gauss();
                c + ar
            })
            .collect();
        let global = wavelet_denoise_with(&obs, 5, ThresholdMode::Soft, Wavelet::Db4);
        let leveldep = wavelet_denoise_leveldep(&obs, 5, ThresholdMode::Soft, Wavelet::Db4);
        let s_global = snr_db(&clean, &global);
        let s_leveldep = snr_db(&clean, &leveldep);
        assert!(
            s_leveldep > s_global,
            "level-dependent {s_leveldep:.2} dB must beat global {s_global:.2} dB"
        );
    }

    #[test]
    fn leveldep_matches_global_convention_on_white_noise() {
        // On truly white noise every band has the same σ, so the level-dependent
        // rule must stay in the same league as the classical one (same family,
        // just per-band estimation error) — and beat the raw input. The clean
        // signal is piecewise constant: sparse in every Haar band, which is the
        // regime the per-band MAD assumes (see the doc caveat).
        let n = 512;
        let mut rng = Lcg::new(43);
        let clean: Vec<f64> = (0..n)
            .map(|i| {
                if i < 170
                {
                    0.0
                }
                else if i < 340
                {
                    1.5
                }
                else
                {
                    0.5
                }
            })
            .collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.3 * rng.gauss()).collect();
        let out = wavelet_denoise_leveldep(&obs, 5, ThresholdMode::Soft, Wavelet::Haar);
        assert_eq!(out.len(), n);
        assert!(snr_db(&clean, &out) > snr_db(&clean, &obs));
    }

    #[test]
    fn bayes_beats_raw_and_tracks_universal_on_mixed_signal() {
        // Sine + step + white noise: the mixed case both rules must handle.
        // BayesShrink adapts per band, so it must beat the raw input and stay
        // within 1 dB of (or better than) the universal threshold.
        let n = 1024;
        let mut rng = Lcg::new(59);
        let clean: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (2.0 * PI * 4.0 * t).sin() + if i >= 600 { 1.5 } else { 0.0 }
            })
            .collect();
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.3 * rng.gauss()).collect();
        let bayes = wavelet_denoise_bayes(&obs, 0, Wavelet::Db4);
        let universal = wavelet_denoise_with(&obs, 0, ThresholdMode::Soft, Wavelet::Db4);
        let s_raw = snr_db(&clean, &obs);
        let s_bayes = snr_db(&clean, &bayes);
        let s_univ = snr_db(&clean, &universal);
        assert!(
            s_bayes > s_raw,
            "Bayes {s_bayes:.2} dB must beat raw {s_raw:.2} dB"
        );
        assert!(
            s_bayes > s_univ - 1.0,
            "Bayes {s_bayes:.2} dB trails universal {s_univ:.2} dB by more than 1 dB"
        );
    }

    #[test]
    fn bayes_kills_pure_noise_bands() {
        // On pure white noise σ_y² ≈ σ² in every band, so σ_x ≈ 0 and BayesShrink
        // should zero (or nearly zero) the details — the output variance must be a
        // small fraction of the input's.
        let mut rng = Lcg::new(61);
        let x: Vec<f64> = (0..512).map(|_| rng.gauss()).collect();
        let out = wavelet_denoise_bayes(&x, 0, Wavelet::Haar);
        let power = |v: &[f64]| v.iter().map(|&s| s * s).sum::<f64>() / v.len() as f64;
        assert!(
            power(&out) < 0.5 * power(&x),
            "pure noise barely attenuated: {} vs {}",
            power(&out),
            power(&x)
        );
    }

    #[test]
    fn new_wavelet_denoisers_degrade_gracefully_on_short_inputs() {
        // Module convention: degenerate inputs come back unchanged, never panic.
        // Db4 needs 4 samples, so lengths 0..=3 must be exact pass-through.
        for len in 0..4_usize
        {
            let x: Vec<f64> = (0..len).map(|i| i as f64 - 1.0).collect();
            let ld = wavelet_denoise_leveldep(&x, 0, ThresholdMode::Soft, Wavelet::Db4);
            assert_eq!(ld, x, "leveldep len {len}");
            let by = wavelet_denoise_bayes(&x, 0, Wavelet::Db4);
            assert_eq!(by, x, "bayes len {len}");
            let ti = wavelet_denoise_ti(&x, 0, ThresholdMode::Soft, Wavelet::Db4, 8);
            assert_eq!(ti, x, "ti len {len}");
        }
        // Haar transforms a length-2 input legitimately; the guarantee there is
        // only shape and finiteness for every short length.
        for len in 0..4_usize
        {
            let x: Vec<f64> = (0..len).map(|i| 1.5 * i as f64).collect();
            for out in [
                wavelet_denoise_leveldep(&x, 0, ThresholdMode::Hard, Wavelet::Haar),
                wavelet_denoise_bayes(&x, 0, Wavelet::Haar),
                wavelet_denoise_ti(&x, 0, ThresholdMode::Hard, Wavelet::Haar, 8),
            ]
            {
                assert_eq!(out.len(), len);
                assert!(out.iter().all(|v| v.is_finite()));
            }
        }
        // cycle_spin itself: empty and single-sample inputs take the plain path.
        let empty: [f64; 0] = [];
        assert!(cycle_spin(&empty, 8, |s| s.to_vec()).is_empty());
        assert_eq!(cycle_spin(&[7.0], 8, |s| s.to_vec()), vec![7.0]);
    }

    #[test]
    fn new_wavelet_denoisers_preserve_constant_signals() {
        // A constant has zero detail coefficients at every level: thresholding
        // (any rule) must leave it untouched up to round-off.
        let x = vec![3.5; 64];
        for out in [
            wavelet_denoise_leveldep(&x, 0, ThresholdMode::Soft, Wavelet::Db4),
            wavelet_denoise_leveldep(&x, 0, ThresholdMode::Hard, Wavelet::Haar),
            wavelet_denoise_bayes(&x, 0, Wavelet::Db4),
            wavelet_denoise_bayes(&x, 0, Wavelet::Haar),
            wavelet_denoise_ti(&x, 0, ThresholdMode::Soft, Wavelet::Db4, 8),
            wavelet_denoise_ti(&x, 0, ThresholdMode::Hard, Wavelet::Haar, 8),
        ]
        {
            assert_eq!(out.len(), x.len());
            for v in out.iter()
            {
                assert!((v - 3.5).abs() < 1.0e-9, "constant not preserved: {v}");
            }
        }
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
