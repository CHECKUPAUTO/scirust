//! Short-time (block) Wiener denoising — spectral gains that adapt over **time**, for
//! **non-stationary** noise.
//!
//! [`super::transform::wiener_white`] takes one FFT of the whole record and applies a single
//! gain per frequency bin, which implicitly assumes the noise statistics never change: a
//! noise burst in the second half raises the *average* bin power and drags the gain of the
//! quiet first half down with it. This module instead slices the signal into overlapping
//! Hann-windowed frames, computes a per-frame per-bin suppression gain, and resynthesizes by
//! weighted overlap-add. Each frame sees only its own neighbourhood, so the filter tightens
//! exactly when and where the noise is strong — the textbook short-time-Fourier enhancement
//! architecture of speech denoising, applicable to any 1-D signal.
//!
//! Four entry points, in increasing order of automation:
//!
//! * [`stft_wiener`] — per-frame Wiener gain against a known white-noise level;
//! * [`stft_wiener_dd`] — the same gain driven by the **decision-directed** a-priori-SNR
//!   estimator (Ephraim-Malah 1984), which suppresses the "musical noise" artefact;
//! * [`stft_wiener_auto`] — fully automatic: noise level, frame size, hop and smoothing all
//!   chosen from the signal;
//! * [`stft_wiener_tracked`] — no noise level at all: a per-bin noise floor is tracked from
//!   the minima of the smoothed periodogram (a lightweight minimum-statistics scheme after
//!   Martin 2001), which also handles **colored** non-stationary noise.
//!
//! When the noise really is stationary and white, the one-shot
//! [`super::transform::wiener_white`] is cheaper; reach for this module when the disturbance
//! drifts, ramps, or switches on and off over the record.

use super::{estimate_noise_std_helper, median, mirror_index, next_pow2};
use crate::Complex;
use crate::fft::{fft, ifft};
use crate::windows::hanning;

/// The analysis-modify-synthesis engine every denoiser in this module shares.
///
/// The signal is extended by `frame_len` samples of symmetric reflection on both ends (via
/// [`mirror_index`]) so every original sample receives full window coverage, sliced into
/// Hann-windowed frames of `frame_len` samples every `hop` samples, transformed with the
/// radix-2 FFT, multiplied per bin by the gains returned by
/// `gain_fn(frame_index, power_spectrum)`, inverse-transformed (real part kept), and
/// recombined by *weighted* overlap-add: each output sample accumulates `w[j]·y[j]` and is
/// normalized by the accumulated `w[j]²` (epsilon-guarded). That normalization makes unit
/// gain an exact identity over the original sample range, for any hop and any input length.
///
/// `frame_len` is rounded up to a power of two (the crate FFT asserts it) with a floor of 4;
/// `hop` is clamped to `[1, frame_len/2]` so no output sample falls into a dead zone of the
/// zero-endpoint Hann window. Only gains for bins `0..=frame_len/2` are read: the mirror bin
/// `frame_len−k` receives the same gain as bin `k`, which keeps the modified spectrum
/// conjugate-symmetric and the inverse transform real. A gain vector shorter than expected
/// is padded with unit gains. Inputs shorter than 2 samples come back unchanged.
fn stft_process<F>(signal: &[f64], frame_len: usize, hop: usize, mut gain_fn: F) -> Vec<f64>
where
    F: FnMut(usize, &[f64]) -> Vec<f64>,
{
    let n = signal.len();
    if n < 2
    {
        return signal.to_vec();
    }
    let frame_len = next_pow2(frame_len.max(4));
    let hop = hop.clamp(1, frame_len / 2);
    let window = hanning(frame_len);

    // Reflect `frame_len` samples on both ends so the first and last original samples sit
    // under the full body of at least one window instead of only its tapered edge.
    let padded_len = n + 2 * frame_len;
    let mut padded = Vec::with_capacity(padded_len);
    for i in 0..padded_len
    {
        padded.push(signal[mirror_index(i as isize - frame_len as isize, n)]);
    }

    let mut acc = vec![0.0; padded_len];
    let mut norm = vec![0.0; padded_len];
    let mut frame_index = 0;
    let mut t = 0;
    while t + frame_len <= padded_len
    {
        let mut buf: Vec<Complex> = (0..frame_len)
            .map(|j| Complex::new(window[j] * padded[t + j], 0.0))
            .collect();
        fft(&mut buf);
        let power: Vec<f64> = buf.iter().map(|c| c.mag_sq()).collect();
        let gains = gain_fn(frame_index, &power);
        // Apply the same gain to bin k and its mirror frame_len−k: conjugate symmetry is
        // preserved exactly, so the inverse transform is real up to round-off.
        for k in 0..=frame_len / 2
        {
            let g = gains.get(k).copied().unwrap_or(1.0);
            buf[k] = buf[k] * g;
            if k != 0 && k != frame_len - k
            {
                buf[frame_len - k] = buf[frame_len - k] * g;
            }
        }
        ifft(&mut buf);
        for j in 0..frame_len
        {
            acc[t + j] += window[j] * buf[j].re;
            norm[t + j] += window[j] * window[j];
        }
        frame_index += 1;
        t += hop;
    }

    (0..n)
        .map(|i| {
            let w = norm[frame_len + i];
            if w > 1.0e-12
            {
                acc[frame_len + i] / w
            }
            else
            {
                0.0
            }
        })
        .collect()
}

/// Expected white-noise power per FFT bin of a **windowed** frame: `P_n = σ²·Σ_j w_j²`.
///
/// The noise part of the frame spectrum is `N_k = Σ_j w_j·n_j·e^{−2πi·jk/N}`; with
/// `E[n_j·n_l] = σ²·δ_{jl}` and `|e^{−2πi·jk/N}| = 1`, the expected power is
/// `E|N_k|² = Σ_j w_j²·σ²` — the window's energy replaces the bare frame length `N` that
/// appears in the rectangular-window (whole-record) formula of
/// [`super::transform::wiener_white`], and the result is independent of the bin index.
fn windowed_noise_power(frame_len: usize, noise_std: f64) -> f64 {
    let wsq: f64 = hanning(frame_len).iter().map(|&w| w * w).sum();
    noise_std * noise_std * wsq
}

/// Short-time Wiener filter for additive white noise of known standard deviation.
///
/// Every Hann-windowed frame gets the per-bin minimum-mean-square gain
/// `G_k = max(0, 1 − P_n/|X_k|²)` with `P_n = σ²·Σ_j w_j²` (the window's energy replaces the
/// bare frame length of the rectangular-window whole-record formula, because the noise passes
/// through the analysis window before the FFT). Because the gain is recomputed per frame,
/// suppression adapts
/// over time: quiet stretches are left almost untouched while noisy stretches are attenuated
/// hard — exactly what the single global gain of [`super::transform::wiener_white`] cannot
/// do when the noise level drifts. `frame_len` is rounded up to a power of two (floor 4) and
/// `hop` is clamped to `[1, frame_len/2]`; `frame_len/4` is a good default hop. Inputs
/// shorter than 2 samples, or a non-positive `noise_std`, return the signal unchanged.
///
/// The per-frame gain reacts to every fluctuation of `|X_k|²`, so isolated noise bins that
/// happen to exceed `P_n` survive for a single frame each — the "musical noise" artefact.
/// If that matters, prefer [`stft_wiener_dd`].
pub fn stft_wiener(signal: &[f64], noise_std: f64, frame_len: usize, hop: usize) -> Vec<f64> {
    if signal.len() < 2 || noise_std <= 0.0
    {
        return signal.to_vec();
    }
    let nfft = next_pow2(frame_len.max(4));
    let p_noise = windowed_noise_power(nfft, noise_std);
    stft_process(signal, nfft, hop, move |_, px| {
        px.iter()
            .map(|&p| {
                if p > 0.0
                {
                    (1.0 - p_noise / p).max(0.0)
                }
                else
                {
                    0.0
                }
            })
            .collect()
    })
}

/// Shared decision-directed core: per frame, `floor_fn` supplies the per-bin noise power for
/// bins `0..=nfft/2`, and the Ephraim-Malah recursion turns it into Wiener gains.
fn stft_wiener_dd_with_floor<F>(
    signal: &[f64],
    nfft: usize,
    hop: usize,
    alpha: f64,
    mut floor_fn: F,
) -> Vec<f64>
where
    F: FnMut(usize, &[f64]) -> Vec<f64>,
{
    let alpha = alpha.clamp(0.0, 0.9999);
    let half = nfft / 2;
    let mut a_prev_sq = vec![0.0; half + 1];
    stft_process(signal, nfft, hop, move |i, px| {
        let p_noise = floor_fn(i, px);
        let mut gains = vec![0.0; half + 1];
        for k in 0..=half
        {
            let pn = p_noise[k].max(1.0e-30);
            // A-posteriori SNR of the current frame, and the decision-directed blend of the
            // previous frame's clean-amplitude estimate with the current ML estimate.
            let gamma = px[k] / pn;
            let xi = alpha * (a_prev_sq[k] / pn) + (1.0 - alpha) * (gamma - 1.0).max(0.0);
            let g = xi / (1.0 + xi);
            a_prev_sq[k] = g * g * px[k];
            gains[k] = g;
        }
        gains
    })
}

/// Short-time Wiener filter with the **decision-directed a-priori SNR** estimator
/// (Ephraim-Malah 1984), the standard cure for musical noise.
///
/// Per bin and per frame: the a-posteriori SNR is `γ = |X_k|²/P_n`; the a-priori SNR is the
/// recursion `ξ = α·(A²_prev/P_n) + (1−α)·max(γ − 1, 0)`; the gain is the Wiener rule
/// `G = ξ/(1+ξ)`; and `A = G·|X_k|` is stored for the next frame. `P_n = σ²·Σ_j w_j²` as in
/// [`stft_wiener`], and `alpha` is clamped to `[0, 1)` — 0.98 is the classic choice.
///
/// **Why this kills musical noise:** in the plain per-frame gain, a noise-only bin has `γ`
/// fluctuating around 1, so it randomly pops above the threshold for one frame at a time —
/// each pop is an isolated, short-lived spectral peak, perceived as a warbling of random
/// tones. The recursion low-pass-filters the *gain trajectory in time*: a one-frame `γ`
/// spike raises `ξ` by only `(1−α)` of its excess (and the `A_prev` feedback of a
/// just-suppressed bin is ~0), so brief outliers no longer punch through, while sustained
/// signal energy accumulates in `A` and drives the gain smoothly toward 1. With `alpha = 0`
/// the recursion degenerates to `ξ = max(γ−1, 0)`, whose gain `ξ/(1+ξ)` equals the plain
/// [`stft_wiener`] rule `max(0, 1 − P_n/|X_k|²)` exactly.
pub fn stft_wiener_dd(
    signal: &[f64],
    noise_std: f64,
    frame_len: usize,
    hop: usize,
    alpha: f64,
) -> Vec<f64> {
    if signal.len() < 2 || noise_std <= 0.0
    {
        return signal.to_vec();
    }
    let nfft = next_pow2(frame_len.max(4));
    let p_noise = vec![windowed_noise_power(nfft, noise_std); nfft / 2 + 1];
    stft_wiener_dd_with_floor(signal, nfft, hop, alpha, move |_, _| p_noise.clone())
}

/// Fully automatic short-time Wiener denoiser — one argument, no tuning.
///
/// The noise level comes from the robust finest-scale MAD estimator (the same
/// [`super::detect::estimate_noise_std`] backbone used across the module), the frame length
/// is `clamp(next_pow2(n/8), 32, 256)` — long enough for spectral resolution, short enough
/// that roughly eight frames span the record and the gains can track drift — the hop is the
/// standard `frame_len/4`, and the decision-directed smoothing of [`stft_wiener_dd`] runs
/// with the classic `alpha = 0.98`. Degenerate inputs (fewer than 2 samples, or a signal so
/// clean the noise estimate is zero) come back unchanged.
pub fn stft_wiener_auto(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n < 2
    {
        return signal.to_vec();
    }
    let noise_std = estimate_noise_std_helper(signal);
    if noise_std <= 0.0
    {
        return signal.to_vec();
    }
    let frame_len = next_pow2(n / 8).clamp(32, 256);
    stft_wiener_dd(signal, noise_std, frame_len, frame_len / 4, 0.98)
}

/// Blind short-time Wiener filter with a **tracked per-bin noise floor** — no `noise_std`
/// input, and the noise may be both colored and non-stationary.
///
/// The tracker is a lightweight variant of Martin's minimum-statistics estimator (Martin
/// 2001, *Noise power spectral density estimation based on optimal smoothing and minimum
/// statistics*): the periodogram of each frame is smoothed across time with an exponential
/// factor of 0.8, its per-bin **minimum over the last 8 frames** is taken as the noise-floor
/// estimate, and the systematic downward bias of a minimum is compensated by the standard
/// factor of ~1.5. The insight carried over from Martin: the smoothed spectrum in every bin
/// keeps dipping down to the noise level whenever the signal energy there fluctuates, so a
/// windowed minimum estimates the noise PSD *per bin* without any noise-only reference —
/// which is what makes colored noise tractable. This is deliberately **not** the full
/// algorithm: Martin's optimal time-varying smoothing, sub-window minima and SNR-dependent
/// bias compensation are replaced by fixed constants, and a frequency-local 9-bin median is
/// applied to the periodogram first so that spectrally sparse signal lines (a steady tone
/// never "pauses", the classic failure mode of pure minimum statistics) do not contaminate
/// their own floor estimate — colored noise varies smoothly across bins, so the median
/// tracks it while rejecting isolated lines. The resulting floor drives the same
/// decision-directed Wiener recursion as [`stft_wiener_dd`] (`alpha = 0.98`).
///
/// Prefer this entry point when the noise is colored (hums with wide skirts, rumble, AR-type
/// correlated noise) and/or its level drifts, and no calibration segment is available. For
/// known white noise, [`stft_wiener`] / [`stft_wiener_dd`] are sharper because their floor
/// is exact rather than tracked.
pub fn stft_wiener_tracked(signal: &[f64], frame_len: usize, hop: usize) -> Vec<f64> {
    if signal.len() < 2
    {
        return signal.to_vec();
    }
    let nfft = next_pow2(frame_len.max(4));
    let half = nfft / 2;
    const SMOOTH: f64 = 0.8; // exponential smoothing of the periodogram
    const BIAS: f64 = 1.5; // minimum-statistics bias compensation
    const MIN_WINDOW: usize = 8; // frames over which the minimum is tracked
    let mut smoothed: Vec<f64> = Vec::new();
    let mut history: Vec<Vec<f64>> = Vec::with_capacity(MIN_WINDOW);
    let mut next_slot = 0;
    stft_wiener_dd_with_floor(signal, nfft, hop, 0.98, move |_, px| {
        // Frequency-local median: reject sparse signal lines, keep the smooth noise shape.
        let mut q = vec![0.0; half + 1];
        for (k, qk) in q.iter_mut().enumerate()
        {
            let lo = k.saturating_sub(4);
            let hi = (k + 4).min(half);
            *qk = median(&px[lo..=hi]);
        }
        // Exponentially smoothed periodogram (initialized on the first frame).
        if smoothed.is_empty()
        {
            smoothed = q;
        }
        else
        {
            for (s, &qk) in smoothed.iter_mut().zip(q.iter())
            {
                *s = SMOOTH * *s + (1.0 - SMOOTH) * qk;
            }
        }
        // Ring buffer of the last MIN_WINDOW smoothed spectra; the per-bin minimum over it,
        // bias-compensated, is the tracked noise floor for this frame.
        if history.len() < MIN_WINDOW
        {
            history.push(smoothed.clone());
        }
        else
        {
            history[next_slot] = smoothed.clone();
            next_slot = (next_slot + 1) % MIN_WINDOW;
        }
        (0..=half)
            .map(|k| BIAS * history.iter().map(|h| h[k]).fold(f64::INFINITY, f64::min))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{Lcg, snr_db};
    use super::super::transform::wiener_white;
    use super::*;
    use core::f64::consts::PI;

    fn sine(n: usize, cycles: f64) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * cycles * i as f64 / n as f64).sin())
            .collect()
    }

    /// Gain-flicker proxy: variance of the first difference of an error signal. Musical
    /// noise is made of isolated one-frame spectral pops, which show up as a jittery,
    /// rapidly sign-flipping residual — i.e. a large first-difference variance.
    fn diff_var(e: &[f64]) -> f64 {
        let d: Vec<f64> = e.windows(2).map(|w| w[1] - w[0]).collect();
        let m = d.iter().sum::<f64>() / d.len() as f64;
        d.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / d.len() as f64
    }

    #[test]
    fn unit_gain_reconstructs_input_exactly() {
        // Weighted overlap-add with the accumulated-w² normalization must be an exact
        // identity over the original range — including both edges, thanks to the reflection
        // padding — for power-of-two, non-power-of-two and shorter-than-frame lengths, and
        // for a hop that does not divide the frame length.
        let mut rng = Lcg::new(3);
        for &n in &[512_usize, 500, 37]
        {
            let x: Vec<f64> = (0..n).map(|_| rng.gauss()).collect();
            for &hop in &[16_usize, 21]
            {
                let out = stft_process(&x, 64, hop, |_, px| vec![1.0; px.len()]);
                assert_eq!(out.len(), n);
                for (i, (&a, &b)) in x.iter().zip(out.iter()).enumerate()
                {
                    assert!(
                        (a - b).abs() < 1.0e-9,
                        "n={n} hop={hop} sample {i}: {a} vs {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn frame_len_rounds_up_to_a_power_of_two() {
        // The crate FFT asserts power-of-two sizes, so frame_len = 48 must silently become
        // 64 — observable through the power-spectrum length the gain callback receives —
        // and reconstruction must still be exact.
        let mut rng = Lcg::new(5);
        let x: Vec<f64> = (0..300).map(|_| rng.gauss()).collect();
        let mut seen = 0;
        let out = stft_process(&x, 48, 12, |_, px| {
            seen = px.len();
            vec![1.0; px.len()]
        });
        assert_eq!(seen, 64, "frame_len 48 should round up to 64");
        for (&a, &b) in x.iter().zip(out.iter())
        {
            assert!((a - b).abs() < 1.0e-9);
        }
        // And the public entry points must accept a non-power-of-two frame_len untroubled.
        let w = stft_wiener(&x, 0.5, 48, 12);
        assert_eq!(w.len(), x.len());
        assert!(w.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn stft_wiener_improves_snr_on_stationary_white_noise() {
        let n = 512;
        let mut rng = Lcg::new(13);
        let clean = sine(n, 32.0); // period 16 samples: bin 4 of every 64-sample frame
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.4 * rng.gauss()).collect();
        let out = stft_wiener(&obs, 0.4, 64, 16);
        assert_eq!(out.len(), n);
        assert!(
            snr_db(&clean, &out) > snr_db(&clean, &obs),
            "stft_wiener {:.2} dB must beat raw {:.2} dB",
            snr_db(&clean, &out),
            snr_db(&clean, &obs)
        );
    }

    #[test]
    fn stft_beats_global_wiener_on_ramped_noise() {
        // Non-stationarity is the whole point: white noise whose σ ramps linearly from 0.1
        // to 0.8 across the record, both filters given the same mid-scale σ = 0.45. The
        // block filter wins on two fronts. First, adaptivity: its gains track the ramp
        // frame by frame — the quiet start passes nearly untouched while the loud end is
        // suppressed against its own local spectrum — whereas the global gain only sees the
        // record-average bin power. Second, analysis quality on a realistic record: the
        // length is NOT a power of two and the tone is strong, so the global filter's
        // unwindowed reflection-padded FFT smears the sine across the spectrum (rectangular
        // leakage plus the phase corner at the padding seam) and then shaves those smeared
        // sidelobes off as if they were noise, while every Hann-windowed frame keeps the
        // tone compact. Both effects are exactly what short-time processing is for.
        let n = 640;
        let mut rng = Lcg::new(17);
        let clean: Vec<f64> = (0..n)
            .map(|i| 8.0 * (2.0 * PI * i as f64 / 8.0).sin()) // period 8: bin 8 per frame
            .collect();
        let obs: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let sigma = 0.1 + 0.7 * i as f64 / (n - 1) as f64;
                c + sigma * rng.gauss()
            })
            .collect();
        let block = stft_wiener(&obs, 0.45, 64, 16);
        let global = wiener_white(&obs, 0.45);
        let s_block = snr_db(&clean, &block);
        let s_global = snr_db(&clean, &global);
        assert!(
            s_block >= s_global + 1.5,
            "block Wiener {s_block:.2} dB must beat global Wiener {s_global:.2} dB by 1.5 dB"
        );
    }

    #[test]
    fn decision_directed_smooths_residual_without_losing_snr() {
        // The decision-directed recursion must not cost SNR (within 0.5 dB, usually a gain)
        // AND must produce a temporally smoother error — musical noise is flicker, measured
        // here as the first-difference variance of (output − clean).
        let n = 1024;
        let mut rng = Lcg::new(19);
        let clean = sine(n, 64.0);
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.5 * rng.gauss()).collect();
        let plain = stft_wiener(&obs, 0.5, 64, 16);
        let dd = stft_wiener_dd(&obs, 0.5, 64, 16, 0.98);
        let s_plain = snr_db(&clean, &plain);
        let s_dd = snr_db(&clean, &dd);
        assert!(
            s_dd >= s_plain - 0.5,
            "decision-directed {s_dd:.2} dB fell more than 0.5 dB below plain {s_plain:.2} dB"
        );
        let e_plain: Vec<f64> = plain.iter().zip(clean.iter()).map(|(o, c)| o - c).collect();
        let e_dd: Vec<f64> = dd.iter().zip(clean.iter()).map(|(o, c)| o - c).collect();
        assert!(
            diff_var(&e_dd) < diff_var(&e_plain),
            "dd flicker {} must be below plain flicker {}",
            diff_var(&e_dd),
            diff_var(&e_plain)
        );
    }

    #[test]
    fn tracked_handles_colored_nonstationary_noise_blind() {
        // Amplitude-modulated colored noise: an AR(1) process (pole 0.9) whose driving σ
        // ramps up 5× across the record, burying the sine (raw SNR is negative). The
        // tracked filter gets no noise level at all — its per-bin minimum-statistics floor
        // must follow both the AR(1) spectral shape and the amplitude ramp — and must still
        // gain ≥ 3 dB over the raw observation.
        let n = 2048;
        let mut rng = Lcg::new(23);
        let clean = sine(n, 128.0); // period 16 samples: bin 4 of every 64-sample frame
        let mut ar = 0.0;
        let obs: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let drive = 0.15 + 0.60 * i as f64 / (n - 1) as f64;
                ar = 0.9 * ar + drive * rng.gauss();
                c + ar
            })
            .collect();
        let out = stft_wiener_tracked(&obs, 64, 16);
        let s_raw = snr_db(&clean, &obs);
        let s_out = snr_db(&clean, &out);
        assert!(
            s_out >= s_raw + 3.0,
            "tracked {s_out:.2} dB must beat raw {s_raw:.2} dB by 3 dB"
        );
    }

    #[test]
    fn auto_improves_snr_blind() {
        let n = 1024;
        let mut rng = Lcg::new(29);
        let clean = sine(n, 64.0);
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.4 * rng.gauss()).collect();
        let out = stft_wiener_auto(&obs);
        assert_eq!(out.len(), n);
        assert!(
            snr_db(&clean, &out) > snr_db(&clean, &obs),
            "auto {:.2} dB must beat raw {:.2} dB",
            snr_db(&clean, &out),
            snr_db(&clean, &obs)
        );
    }

    #[test]
    fn dd_with_alpha_zero_matches_plain_wiener() {
        // With alpha = 0 the recursion is ξ = max(γ−1, 0), whose gain ξ/(1+ξ) is
        // algebraically the plain Wiener rule — the outputs must agree to round-off. This
        // also pins the alpha plumbing: a live alpha of 0.98 must NOT reproduce it.
        let n = 512;
        let mut rng = Lcg::new(31);
        let clean = sine(n, 32.0);
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.4 * rng.gauss()).collect();
        let plain = stft_wiener(&obs, 0.4, 64, 16);
        let dd0 = stft_wiener_dd(&obs, 0.4, 64, 16, 0.0);
        for (i, (&a, &b)) in plain.iter().zip(dd0.iter()).enumerate()
        {
            assert!((a - b).abs() < 1.0e-9, "sample {i}: plain {a} vs dd(0) {b}");
        }
        // Out-of-range alpha clamps into [0, 1): below-range equals alpha = 0 exactly.
        assert_eq!(stft_wiener_dd(&obs, 0.4, 64, 16, -2.0), dd0);
        let above = stft_wiener_dd(&obs, 0.4, 64, 16, 1.5);
        assert!(above.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn stft_parameters_are_live() {
        // Every knob must change the output — a transposed or ignored parameter (all of
        // frame_len/hop are usize) would compile silently.
        let n = 512;
        let mut rng = Lcg::new(37);
        let clean = sine(n, 32.0);
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.4 * rng.gauss()).collect();
        let base = stft_wiener(&obs, 0.4, 64, 16);
        assert_ne!(base, stft_wiener(&obs, 0.4, 64, 8), "hop is ignored");
        assert_ne!(
            base,
            stft_wiener(&obs, 0.4, 128, 16),
            "frame_len is ignored"
        );
        assert_ne!(base, stft_wiener(&obs, 0.2, 64, 16), "noise_std is ignored");
        assert_ne!(
            stft_wiener_dd(&obs, 0.4, 64, 16, 0.5),
            stft_wiener_dd(&obs, 0.4, 64, 16, 0.98),
            "alpha is ignored"
        );
        assert_ne!(
            stft_wiener_tracked(&obs, 64, 16),
            stft_wiener_tracked(&obs, 64, 8),
            "tracked hop is ignored"
        );
        assert_ne!(
            stft_wiener_tracked(&obs, 64, 16),
            stft_wiener_tracked(&obs, 128, 16),
            "tracked frame_len is ignored"
        );
    }

    #[test]
    fn degenerate_inputs_come_back_unchanged_or_sane() {
        // Module convention: degenerate inputs never panic; too-short inputs pass through.
        let empty: [f64; 0] = [];
        assert!(stft_wiener(&empty, 0.4, 64, 16).is_empty());
        assert!(stft_wiener_dd(&empty, 0.4, 64, 16, 0.98).is_empty());
        assert!(stft_wiener_auto(&empty).is_empty());
        assert!(stft_wiener_tracked(&empty, 64, 16).is_empty());
        assert_eq!(stft_wiener(&[2.5], 0.4, 64, 16), vec![2.5]);
        assert_eq!(stft_wiener_dd(&[2.5], 0.4, 64, 16, 0.98), vec![2.5]);
        assert_eq!(stft_wiener_auto(&[2.5]), vec![2.5]);
        assert_eq!(stft_wiener_tracked(&[2.5], 64, 16), vec![2.5]);
        for len in 2..4_usize
        {
            let x: Vec<f64> = (0..len).map(|i| i as f64 - 1.0).collect();
            for out in [
                stft_wiener(&x, 0.4, 64, 16),
                stft_wiener_dd(&x, 0.4, 64, 16, 0.98),
                stft_wiener_auto(&x),
                stft_wiener_tracked(&x, 64, 16),
            ]
            {
                assert_eq!(out.len(), len);
                assert!(out.iter().all(|v| v.is_finite()));
            }
        }
        // Non-positive noise level: nothing to subtract, exact pass-through.
        let mut rng = Lcg::new(41);
        let x: Vec<f64> = (0..100).map(|_| rng.gauss()).collect();
        assert_eq!(stft_wiener(&x, 0.0, 64, 16), x);
        assert_eq!(stft_wiener_dd(&x, -1.0, 64, 16, 0.98), x);
        // Pathological frame/hop requests are clamped, not fatal.
        let tiny = stft_wiener(&x, 0.3, 0, 0);
        assert_eq!(tiny.len(), x.len());
        assert!(tiny.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn constant_signal_is_essentially_preserved() {
        // A constant lives in the DC bin, whose power towers over any noise floor, so every
        // variant must return (very nearly) the constant.
        let x = vec![3.5; 200];
        for (name, out) in [
            ("stft_wiener", stft_wiener(&x, 0.1, 64, 16)),
            ("stft_wiener_dd", stft_wiener_dd(&x, 0.1, 64, 16, 0.98)),
            ("stft_wiener_auto", stft_wiener_auto(&x)),
            ("stft_wiener_tracked", stft_wiener_tracked(&x, 64, 16)),
        ]
        {
            assert_eq!(out.len(), x.len(), "{name}");
            for v in out.iter()
            {
                assert!((v - 3.5).abs() < 0.05, "{name}: constant became {v}");
            }
        }
    }
}
