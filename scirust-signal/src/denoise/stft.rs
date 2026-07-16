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
//! The batch entry points, in increasing order of automation:
//!
//! * [`stft_wiener`] — per-frame Wiener gain against a known white-noise level;
//! * [`stft_wiener_dd`] — the same gain driven by the **decision-directed** a-priori-SNR
//!   estimator (Ephraim-Malah 1984), which suppresses the "musical noise" artefact;
//! * [`stft_mmse_lsa`] — the **log-spectral-amplitude** MMSE gain (Ephraim-Malah 1985) on
//!   the same decision-directed recursion: the perceptual-scale optimum, gentler on weak
//!   signal than the Wiener rule and far less musical than the plain per-frame gain;
//! * [`stft_wiener_auto`] — fully automatic: noise level, frame size, hop and smoothing all
//!   chosen from the signal;
//! * [`stft_wiener_tracked`] — no noise level at all: a per-bin noise floor is tracked from
//!   the minima of the smoothed periodogram (a lightweight minimum-statistics scheme after
//!   Martin 2001), which also handles **colored** non-stationary noise;
//! * [`stft_wiener_tracked_ms`] — a fuller Martin-style tracker: adaptive optimal-smoothing
//!   surrogate, sub-window minima with bounded forgetting delay, and bias compensation.
//!
//! For real-time use, [`StreamingStftWiener`] runs the same analysis-modify-synthesis loop
//! **causally**, one sample per [`StreamingStftWiener::push`], with a reported delay of one
//! frame — the streaming counterpart of [`stft_wiener_dd`] / [`stft_wiener_tracked`].
//!
//! When the noise really is stationary and white, the one-shot
//! [`super::transform::wiener_white`] is cheaper; reach for this module when the disturbance
//! drifts, ramps, or switches on and off over the record.

use std::collections::VecDeque;

use super::streaming::StreamingDenoiser;
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

/// Exponential integral `E1(x) = ∫_x^∞ e^{−t}/t dt` for `x > 0` — the special function at the
/// heart of the Ephraim-Malah log-spectral-amplitude gain of [`stft_mmse_lsa`].
///
/// Two classical regimes (Abramowitz & Stegun, *Handbook of Mathematical Functions*, ch. 5):
///
/// * `x ≤ 1` — the convergent power series 5.1.11,
///   `E1(x) = −γ − ln x + Σ_{k≥1} (−1)^{k+1}·xᵏ/(k·k!)`, with `γ` the Euler-Mascheroni
///   constant. The terms shrink factorially, so a few dozen reach round-off.
/// * `x > 1` — the Stieltjes continued fraction 5.1.22,
///   `E1(x) = e^{−x}/(x + 1 − 1²/(x + 3 − 2²/(x + 5 − …)))`, evaluated bottom-up-free by the
///   modified Lentz algorithm (Numerical Recipes §6.3), which converges in a handful of
///   iterations and stays accurate arbitrarily deep into the tail.
///
/// Deterministic, ~1e-14 relative accuracy over the callers' clamped domain `[1e-10, 700]`.
/// Non-positive arguments (outside the supported domain) return `+∞`, the correct limit as
/// `x → 0⁺`.
fn expint_e1(x: f64) -> f64 {
    const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
    if x <= 0.0
    {
        return f64::INFINITY;
    }
    if x <= 1.0
    {
        let mut sum = 0.0;
        let mut term = 1.0; // running xᵏ/k!
        let mut sign = 1.0;
        for k in 1..=60
        {
            term *= x / k as f64;
            sum += sign * term / k as f64;
            if term / k as f64 <= 1.0e-17
            {
                break;
            }
            sign = -sign;
        }
        -EULER_GAMMA - x.ln() + sum
    }
    else
    {
        let mut b = x + 1.0;
        let mut c = 1.0e300;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=200
        {
            let a = -((i * i) as f64);
            b += 2.0;
            d = 1.0 / (a * d + b);
            c = b + a / c;
            let del = c * d;
            h *= del;
            if (del - 1.0).abs() < 1.0e-15
            {
                break;
            }
        }
        h * (-x).exp()
    }
}

/// Short-time **MMSE log-spectral-amplitude** estimator — Ephraim & Malah 1985, *Speech
/// enhancement using a minimum mean-square error log-spectral amplitude estimator* (IEEE
/// Trans. ASSP-33), the second and more celebrated of the two Ephraim-Malah estimators.
///
/// The decision-directed machinery is exactly that of [`stft_wiener_dd`]: per bin and per
/// frame, the a-posteriori SNR is `γ = |X_k|²/P_n`, the a-priori SNR is the recursion
/// `ξ = α·(A²_prev/P_n) + (1−α)·max(γ − 1, 0)`, and `P_n = σ²·Σ_j w_j²` is the windowed
/// white-noise power. What changes is the gain rule. Minimizing the mean-square error of the
/// **log**-amplitude — the perceptually meaningful scale for audio, and the natural scale for
/// any strictly positive spectral quantity — yields the closed form
///
/// `G = ξ/(1+ξ) · exp(½·E1(v))` with `v = ξγ/(1+ξ)`,
///
/// where `E1` is the exponential integral (evaluated by the private `expint_e1`; `v` is
/// clamped to
/// `[1e-10, 700]` before `E1`/`exp`, and `G` to `[0, 1]`). Since `E1(v) > 0`, the LSA gain
/// sits **above** the Wiener rule `ξ/(1+ξ)` at equal `(ξ, γ)` — pronounced exactly where `v`
/// is small, i.e. in low-SNR bins — so it is gentler on weak signal components the Wiener
/// gain would clip, while the decision-directed recursion keeps the gain trajectory smooth
/// (no musical noise). Its classic selling point is a lower, *smoother* residual noise floor
/// than the amplitude-domain estimators: measured against the plain per-frame [`stft_wiener`]
/// rule it leaves an order of magnitude less residual on noise-only stretches (the aggressive
/// decision-directed Wiener output of [`stft_wiener_dd`] is by construction a pointwise lower
/// envelope of this estimator's output, at the price of clipping weak signal harder).
///
/// `alpha` is clamped to `[0, 1)` — 0.98 is the classic choice; `frame_len` is rounded up to
/// a power of two (floor 4) and `hop` is clamped to `[1, frame_len/2]`. Inputs shorter than
/// 2 samples, or a non-positive `noise_std`, return the signal unchanged.
pub fn stft_mmse_lsa(
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
    let half = nfft / 2;
    let alpha = alpha.clamp(0.0, 0.9999);
    let p_noise = windowed_noise_power(nfft, noise_std).max(1.0e-30);
    let mut a_prev_sq = vec![0.0; half + 1];
    stft_process(signal, nfft, hop, move |_, px| {
        let mut gains = vec![0.0; half + 1];
        for k in 0..=half
        {
            let gamma = px[k] / p_noise;
            let xi = alpha * (a_prev_sq[k] / p_noise) + (1.0 - alpha) * (gamma - 1.0).max(0.0);
            let v = (xi * gamma / (1.0 + xi)).clamp(1.0e-10, 700.0);
            let g = (xi / (1.0 + xi) * (0.5 * expint_e1(v)).exp()).clamp(0.0, 1.0);
            a_prev_sq[k] = g * g * px[k];
            gains[k] = g;
        }
        gains
    })
}

/// Blind short-time Wiener filter with a **fuller minimum-statistics noise tracker** (after
/// Martin 2001, *Noise power spectral density estimation based on optimal smoothing and
/// minimum statistics*, IEEE Trans. SAP-9) — the upgrade of [`stft_wiener_tracked`]'s
/// fixed-constant scheme toward the real algorithm.
///
/// Three components move from fixed constants to Martin-style mechanisms:
///
/// 1. **Adaptive smoothing.** The periodogram is smoothed across time per bin with
///    `α_s(k,m) = α_max / (1 + (P̄(k,m−1)/P(k,m) − 1)²)`, clamped to `[0.3, 0.96]` — a
///    surrogate of Martin's optimal smoothing parameter (his eq. 12): when the spectrum is
///    stationary the ratio is ≈ 1 and the smoothing is heavy (long memory, low variance, a
///    deep reliable minimum); when the local power jumps the ratio leaves 1, the factor drops
///    and the tracker reacts within a frame or two instead of smearing the transition.
/// 2. **Sub-window minima.** The per-bin minimum is taken over `U = 4` sub-windows of
///    `V = 8` frames each (effective window `D = U·V = 32` frames): a completed sub-window's
///    minimum expires after at most `D + V` frames, so an obsolete deep minimum from a
///    quieter past is forgotten with bounded delay — the mechanism that lets the floor climb
///    after a step *up* in the noise level, which a single ever-growing minimum cannot do.
/// 3. **Bias compensation.** The minimum of a fluctuating spectral estimate underestimates
///    its mean, the more so the longer the window; the floor is multiplied by
///    `B_min = 1 + (D − 1)·0.02`, a simple monotone-in-`D` **surrogate** of Martin's
///    `B_min(D, Q̄eq)` table — not the exact table, whose values depend on the smoothed
///    estimate's equivalent degrees of freedom. The slope is calibrated to the heavy
///    (`α_s ≤ 0.96`) smoothing above, under which the 32 tracked frames are strongly
///    correlated and the raw minimum dips only mildly below the mean.
///
/// As in [`stft_wiener_tracked`], a frequency-local 9-bin median cleans the periodogram
/// first, so spectrally sparse signal lines (a steady tone never "pauses" — the classic
/// failure mode of pure minimum statistics) do not contaminate their own floor estimate. The
/// tracked floor then drives the same decision-directed Wiener recursion as
/// [`stft_wiener_dd`] (`alpha = 0.98`). Fully deterministic; inputs shorter than 2 samples
/// come back unchanged.
///
/// Prefer this over [`stft_wiener_tracked`] when the noise level *switches* (machinery
/// turning on, gain changes) rather than merely drifting: the sub-window forgetting adapts to
/// a step within ~`D + V` frames, while the light scheme's short 8-frame window is noisier
/// per estimate and its fixed smoothing reacts more sluggishly per bin.
pub fn stft_wiener_tracked_ms(signal: &[f64], frame_len: usize, hop: usize) -> Vec<f64> {
    if signal.len() < 2
    {
        return signal.to_vec();
    }
    let nfft = next_pow2(frame_len.max(4));
    let half = nfft / 2;
    const ALPHA_MAX: f64 = 0.96; // ceiling of the adaptive smoothing factor
    const ALPHA_MIN: f64 = 0.3; // floor: never trust a single frame entirely
    const SUBWINDOWS: usize = 4; // U sub-windows tracked
    const SUBWINDOW_FRAMES: usize = 8; // V frames per sub-window
    const EFFECTIVE: usize = SUBWINDOWS * SUBWINDOW_FRAMES; // D = U·V
    const BIAS: f64 = 1.0 + (EFFECTIVE - 1) as f64 * 0.02; // B_min surrogate (see docs)
    let mut p_smooth: Vec<f64> = Vec::new();
    let mut cur_min = vec![f64::INFINITY; half + 1];
    let mut sub_minima: Vec<Vec<f64>> = Vec::with_capacity(SUBWINDOWS);
    let mut next_slot = 0;
    let mut frames_in_sub = 0;
    stft_wiener_dd_with_floor(signal, nfft, hop, 0.98, move |_, px| {
        // Frequency-local median: reject sparse signal lines, keep the smooth noise shape.
        let mut q = vec![0.0; half + 1];
        for (k, qk) in q.iter_mut().enumerate()
        {
            let lo = k.saturating_sub(4);
            let hi = (k + 4).min(half);
            *qk = median(&px[lo..=hi]);
        }
        // (1) Adaptive smoothing: heavy while stationary, reactive across level jumps.
        if p_smooth.is_empty()
        {
            p_smooth = q;
        }
        else
        {
            for (s, &qk) in p_smooth.iter_mut().zip(q.iter())
            {
                let alpha_s = if qk > 0.0
                {
                    let dev = *s / qk - 1.0;
                    (ALPHA_MAX / (1.0 + dev * dev)).clamp(ALPHA_MIN, ALPHA_MAX)
                }
                else
                {
                    ALPHA_MIN
                };
                *s = alpha_s * *s + (1.0 - alpha_s) * qk;
            }
        }
        // (2) Running minimum of the current sub-window…
        for (m, &s) in cur_min.iter_mut().zip(p_smooth.iter())
        {
            if s < *m
            {
                *m = s;
            }
        }
        frames_in_sub += 1;
        // …and (3) the bias-compensated minimum over the stored sub-windows plus the live
        // one is the tracked noise floor for this frame.
        let floor: Vec<f64> = (0..=half)
            .map(|k| {
                let past = sub_minima
                    .iter()
                    .map(|h| h[k])
                    .fold(f64::INFINITY, f64::min);
                BIAS * cur_min[k].min(past)
            })
            .collect();
        if frames_in_sub == SUBWINDOW_FRAMES
        {
            if sub_minima.len() < SUBWINDOWS
            {
                sub_minima.push(cur_min.clone());
            }
            else
            {
                sub_minima[next_slot] = cur_min.clone();
                next_slot = (next_slot + 1) % SUBWINDOWS;
            }
            cur_min = vec![f64::INFINITY; half + 1];
            frames_in_sub = 0;
        }
        floor
    })
}

/// Causal, sample-by-sample **streaming short-time Wiener denoiser** — the real-time
/// counterpart of [`stft_wiener_dd`] (known white-noise level) and, in its blind mode, of
/// [`stft_wiener_tracked`]'s lightweight minimum-statistics floor.
///
/// A ring buffer holds the most recent `frame_len` samples; every `hop` new samples that
/// window is Hann-weighted, FFT-transformed, multiplied per bin by the decision-directed
/// Wiener gain (Ephraim-Malah 1984 recursion, state carried across frames),
/// inverse-transformed, and weighted-overlap-added into an output accumulator that
/// normalizes by the accumulated `w²` — the same weighted-overlap-add identity as this
/// module's batch engine (`stft_process`), made causal. The pipeline is **primed with one frame of zeros**
/// (the streaming stand-in for the batch engine's mirror padding, which would need future
/// samples): that keeps every emitted sample under a full complement of analysis windows, so
/// the `w²` normalization is always well conditioned — no output ever divides by the tiny
/// tapered edge of a lone window. Because every frame that can touch a sample starts no
/// later than that sample, a sample has received all of its window coverage `frame_len`
/// pushes after it went in — which is exactly the reported [`delay`](Self::delay).
///
/// # The delay / warm-up contract
///
/// `delay()` is `frame_len` (after rounding). Let `out[i]` be the value returned by the
/// `i`-th call to [`push`](Self::push) (0-based):
///
/// * **Warm-up:** for `i < frame_len`, `out[i]` is exactly `0.0` — those slots correspond to
///   the synthetic zero priming, and this implementation returns zeros (not pass-through)
///   during warm-up.
/// * **Steady state:** for `i ≥ frame_len`, `out[i]` is the fully-covered
///   weighted-overlap-add estimate of the input sample pushed at index `i − frame_len`.
///
/// The earliest estimates (`i < 2·frame_len`) come from frames that straddle the zero
/// priming and a cold-started recursion, so they show a brief fade-in transient; treat
/// `i = 2·frame_len` as the settled point. On a long record the interior then behaves like
/// the batch filter to within a fraction of a dB of SNR.
///
/// # Parameters
///
/// `frame_len` is rounded up to a power of two with a floor of 4, `hop` is clamped to
/// `[1, frame_len/2]`, `alpha` to `[0, 1)`. A positive `noise_std` fixes the per-bin floor at
/// the windowed white-noise power `σ²·Σ w²`; `noise_std ≤ 0` switches to **blind tracking**:
/// per frame, a 9-bin frequency median of the periodogram is smoothed over time (factor 0.8)
/// and the per-bin minimum over the last 8 frames, bias-compensated by 1.5, serves as the
/// noise floor — the same lightweight minimum-statistics scheme, with the same constants, as
/// [`stft_wiener_tracked`]. Memory is `O(frame_len)` and every operation is deterministic.
#[derive(Debug, Clone)]
pub struct StreamingStftWiener {
    frame_len: usize,
    hop: usize,
    alpha: f64,
    /// Fixed windowed white-noise floor; `0.0` selects the blind tracking mode.
    p_noise: f64,
    window: Vec<f64>,
    /// Absolute sample count in primed-stream coordinates: starts at `frame_len` (the zero
    /// priming) and grows by one per push.
    count: usize,
    /// The most recent `frame_len` primed-stream samples (zeros before the first push).
    input: VecDeque<f64>,
    /// Overlap-add accumulator ring (`2·frame_len` slots, indexed by absolute position).
    acc: Vec<f64>,
    /// Accumulated `w²` ring matching `acc`.
    norm: Vec<f64>,
    /// Decision-directed clean-amplitude-squared state, bins `0..=frame_len/2`.
    a_prev_sq: Vec<f64>,
    /// Tracking mode: exponentially smoothed median periodogram (empty until frame 1).
    smoothed: Vec<f64>,
    /// Tracking mode: ring of the last 8 smoothed spectra for the windowed minimum.
    history: Vec<Vec<f64>>,
    /// Tracking mode: ring-buffer write index into `history`.
    next_slot: usize,
}

impl StreamingStftWiener {
    /// Build a streaming denoiser; see the type docs for parameter handling. `noise_std ≤ 0`
    /// selects the blind noise-tracking mode.
    pub fn new(frame_len: usize, hop: usize, noise_std: f64, alpha: f64) -> Self {
        let frame_len = next_pow2(frame_len.max(4));
        let hop = hop.clamp(1, frame_len / 2);
        let alpha = alpha.clamp(0.0, 0.9999);
        let p_noise = if noise_std > 0.0
        {
            windowed_noise_power(frame_len, noise_std)
        }
        else
        {
            0.0
        };
        Self {
            frame_len,
            hop,
            alpha,
            p_noise,
            window: hanning(frame_len),
            count: frame_len,
            input: VecDeque::from(vec![0.0; frame_len]),
            acc: vec![0.0; 2 * frame_len],
            norm: vec![0.0; 2 * frame_len],
            a_prev_sq: vec![0.0; frame_len / 2 + 1],
            smoothed: Vec::new(),
            history: Vec::new(),
            next_slot: 0,
        }
    }

    /// Feed one sample; returns one output sample under the delay / warm-up contract of the
    /// type docs (`0.0` during the first `frame_len` pushes, then the estimate of the sample
    /// pushed `frame_len` calls earlier).
    pub fn push(&mut self, x: f64) -> f64 {
        self.input.pop_front();
        self.input.push_back(x);
        self.count += 1;
        // The frame starting at primed position s = count − frame_len is complete exactly
        // now; frames start every `hop` positions (the all-zero frame at s = 0 is skipped —
        // unit or zero gain on silence is the same silence).
        if (self.count - self.frame_len).is_multiple_of(self.hop)
        {
            self.process_frame();
        }
        // Emit primed position count − 1 − frame_len: every frame that touches it has been
        // processed (the last one starts at most at its own position), so its coverage is
        // complete. Its accumulator slot is drained for reuse one lap later.
        let p = self.count - 1 - self.frame_len;
        let slot = p % self.acc.len();
        let a = self.acc[slot];
        let w = self.norm[slot];
        self.acc[slot] = 0.0;
        self.norm[slot] = 0.0;
        if self.count <= 2 * self.frame_len
        {
            return 0.0; // warm-up: position p is still inside the zero priming
        }
        if w > 1.0e-12 { a / w } else { 0.0 }
    }

    /// Return to the exact just-constructed state, discarding all buffered samples and the
    /// decision-directed / noise-tracking state.
    pub fn reset(&mut self) {
        self.count = self.frame_len;
        self.input.clear();
        self.input.extend(std::iter::repeat_n(0.0, self.frame_len));
        self.acc.fill(0.0);
        self.norm.fill(0.0);
        self.a_prev_sq.fill(0.0);
        self.smoothed.clear();
        self.history.clear();
        self.next_slot = 0;
    }

    /// Group delay of the causal implementation: one full (rounded) frame. `out[i]` is the
    /// estimate of the input pushed at `i − delay()` once `i ≥ delay()`; see the type docs.
    pub fn delay(&self) -> usize {
        self.frame_len
    }

    /// Analyze the newest `frame_len` samples, apply the decision-directed Wiener gain, and
    /// overlap-add the result into the accumulator rings.
    fn process_frame(&mut self) {
        let n = self.frame_len;
        let half = n / 2;
        let off = self.input.len() - n;
        let mut buf: Vec<Complex> = (0..n)
            .map(|j| Complex::new(self.window[j] * self.input[off + j], 0.0))
            .collect();
        fft(&mut buf);
        let power: Vec<f64> = buf.iter().map(|c| c.mag_sq()).collect();
        let floor = if self.p_noise > 0.0
        {
            vec![self.p_noise; half + 1]
        }
        else
        {
            self.tracked_floor(&power)
        };
        for k in 0..=half
        {
            let pn = floor[k].max(1.0e-30);
            let gamma = power[k] / pn;
            let xi =
                self.alpha * (self.a_prev_sq[k] / pn) + (1.0 - self.alpha) * (gamma - 1.0).max(0.0);
            let g = xi / (1.0 + xi);
            self.a_prev_sq[k] = g * g * power[k];
            buf[k] = buf[k] * g;
            if k != 0 && k != n - k
            {
                buf[n - k] = buf[n - k] * g;
            }
        }
        ifft(&mut buf);
        let start = self.count - n;
        let ring = self.acc.len();
        for (j, (&w, b)) in self.window.iter().zip(buf.iter()).enumerate().take(n)
        {
            let slot = (start + j) % ring;
            self.acc[slot] += w * b.re;
            self.norm[slot] += w * w;
        }
    }

    /// Blind mode's per-frame noise floor: the lightweight minimum-statistics scheme of
    /// [`stft_wiener_tracked`] (9-bin frequency median → exponential smoothing 0.8 → per-bin
    /// minimum over the last 8 frames → bias compensation 1.5), causal by construction.
    fn tracked_floor(&mut self, px: &[f64]) -> Vec<f64> {
        const SMOOTH: f64 = 0.8;
        const BIAS: f64 = 1.5;
        const MIN_WINDOW: usize = 8;
        let half = self.frame_len / 2;
        let mut q = vec![0.0; half + 1];
        for (k, qk) in q.iter_mut().enumerate()
        {
            let lo = k.saturating_sub(4);
            let hi = (k + 4).min(half);
            *qk = median(&px[lo..=hi]);
        }
        if self.smoothed.is_empty()
        {
            self.smoothed = q;
        }
        else
        {
            for (s, &qk) in self.smoothed.iter_mut().zip(q.iter())
            {
                *s = SMOOTH * *s + (1.0 - SMOOTH) * qk;
            }
        }
        if self.history.len() < MIN_WINDOW
        {
            self.history.push(self.smoothed.clone());
        }
        else
        {
            self.history[self.next_slot] = self.smoothed.clone();
            self.next_slot = (self.next_slot + 1) % MIN_WINDOW;
        }
        (0..=half)
            .map(|k| {
                BIAS * self
                    .history
                    .iter()
                    .map(|h| h[k])
                    .fold(f64::INFINITY, f64::min)
            })
            .collect()
    }
}

// Forward the trait to the inherent methods, mirroring streaming.rs's
// `impl_streaming_denoiser!` expansion (the macro is private to that module).
impl StreamingDenoiser for StreamingStftWiener {
    fn push(&mut self, x: f64) -> f64 {
        StreamingStftWiener::push(self, x)
    }
    fn reset(&mut self) {
        StreamingStftWiener::reset(self);
    }
    fn delay(&self) -> usize {
        StreamingStftWiener::delay(self)
    }
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

    // -----------------------------------------------------------------------
    // expint_e1 / stft_mmse_lsa
    // -----------------------------------------------------------------------

    #[test]
    fn expint_e1_matches_tabulated_values() {
        // Reference values from the standard 10+-digit tables (Abramowitz & Stegun ch. 5).
        // The set spans the power-series branch (0.1, 0.5), its boundary (1.0) and the
        // continued-fraction branch (2, 5, 10).
        for (x, want) in [
            (0.1, 1.822_923_958_5),
            (0.5, 0.559_773_594_8),
            (1.0, 0.219_383_934_4),
            (2.0, 0.048_900_510_708),
            (5.0, 0.001_148_295_591),
            (10.0, 4.156_968_929_7e-6),
        ]
        {
            let got = expint_e1(x);
            assert!(
                ((got - want) / want).abs() < 1.0e-6,
                "E1({x}) = {got:e}, want {want:e}"
            );
        }
        // Limits: +∞ at the origin (and for the unsupported x ≤ 0), monotone decreasing.
        assert_eq!(expint_e1(0.0), f64::INFINITY);
        assert_eq!(expint_e1(-1.0), f64::INFINITY);
        assert!(expint_e1(1.0e-10) > expint_e1(0.5));
        assert!(expint_e1(700.0) > 0.0 && expint_e1(700.0) < 1.0e-300);
    }

    #[test]
    fn mmse_lsa_improves_snr_on_stationary_white_noise() {
        let n = 1024;
        let mut rng = Lcg::new(47);
        let clean = sine(n, 64.0); // period 16 samples: bin 4 of every 64-sample frame
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.5 * rng.gauss()).collect();
        let out = stft_mmse_lsa(&obs, 0.5, 64, 16, 0.98);
        assert_eq!(out.len(), n);
        assert!(
            snr_db(&clean, &out) > snr_db(&clean, &obs),
            "stft_mmse_lsa {:.2} dB must beat raw {:.2} dB",
            snr_db(&clean, &out),
            snr_db(&clean, &obs)
        );
    }

    #[test]
    fn mmse_lsa_leaves_less_residual_noise_than_the_plain_wiener_gain() {
        // Pure noise in, residue out — the low-SNR regime where every bin is noise-only.
        // The honest baseline for the LSA's low-residual claim is the plain per-frame
        // Wiener rule: against it the LSA residual floor is far lower and smoother (its
        // gain trajectory is decision-directed). The stft_wiener_dd output is *by
        // construction* a pointwise lower envelope of the LSA output — exp(½·E1(v)) ≥ 1
        // multiplies the same ξ/(1+ξ) under the same recursion — so that comparison would
        // be vacuous in the other direction.
        let n = 2048;
        let mut rng = Lcg::new(53);
        let noise: Vec<f64> = (0..n).map(|_| 0.5 * rng.gauss()).collect();
        let rms = |v: &[f64]| (v.iter().map(|&x| x * x).sum::<f64>() / v.len() as f64).sqrt();
        let lsa = stft_mmse_lsa(&noise, 0.5, 64, 16, 0.98);
        let plain = stft_wiener(&noise, 0.5, 64, 16);
        assert!(
            rms(&lsa) < 0.5 * rms(&plain),
            "LSA residual {:.4} must be well below the plain Wiener residual {:.4}",
            rms(&lsa),
            rms(&plain)
        );
        // And the suppression must be real in absolute terms too: at least 6 dB of the
        // noise gone.
        assert!(
            rms(&lsa) < 0.5 * rms(&noise),
            "LSA residual {:.4} vs input noise {:.4}",
            rms(&lsa),
            rms(&noise)
        );
    }

    #[test]
    fn mmse_lsa_parameters_are_live() {
        let n = 512;
        let mut rng = Lcg::new(83);
        let clean = sine(n, 32.0);
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.4 * rng.gauss()).collect();
        let base = stft_mmse_lsa(&obs, 0.4, 64, 16, 0.98);
        assert_ne!(
            base,
            stft_mmse_lsa(&obs, 0.4, 64, 16, 0.5),
            "alpha is ignored"
        );
        assert_ne!(
            base,
            stft_mmse_lsa(&obs, 0.4, 64, 8, 0.98),
            "hop is ignored"
        );
        assert_ne!(
            base,
            stft_mmse_lsa(&obs, 0.4, 128, 16, 0.98),
            "frame_len is ignored"
        );
        assert_ne!(
            base,
            stft_mmse_lsa(&obs, 0.2, 64, 16, 0.98),
            "noise_std is ignored"
        );
    }

    // -----------------------------------------------------------------------
    // stft_wiener_tracked_ms
    // -----------------------------------------------------------------------

    #[test]
    fn tracked_ms_handles_colored_nonstationary_noise_blind() {
        // Same fixture family as the stft_wiener_tracked test (AR(1) colored noise, pole
        // 0.9, driving σ ramping 5× across the record) with a different seed: the fuller
        // minimum-statistics tracker must beat raw by ≥ 3 dB AND hold the light tracker's
        // level to within 0.5 dB.
        let n = 2048;
        let mut rng = Lcg::new(43);
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
        let ms = stft_wiener_tracked_ms(&obs, 64, 16);
        let light = stft_wiener_tracked(&obs, 64, 16);
        let s_raw = snr_db(&clean, &obs);
        let s_ms = snr_db(&clean, &ms);
        let s_light = snr_db(&clean, &light);
        assert!(
            s_ms >= s_raw + 3.0,
            "tracked_ms {s_ms:.2} dB must beat raw {s_raw:.2} dB by 3 dB"
        );
        assert!(
            s_ms >= s_light - 0.5,
            "tracked_ms {s_ms:.2} dB fell more than 0.5 dB below tracked {s_light:.2} dB"
        );
    }

    #[test]
    fn tracked_ms_adapts_to_a_step_noise_level() {
        // The sub-window forgetting is what lets the floor climb after a step UP in the
        // noise level: σ jumps 0.2 → 0.8 at midrecord. The blind tracker must beat the
        // global Wiener given the mid-scale σ = 0.5 — the global gain both over-suppresses
        // the quiet half and under-suppresses the loud half, while the tracked floor is
        // re-settled on each level within D + V frames of the step.
        let n = 16384;
        let mut rng = Lcg::new(59);
        let clean = sine(n, 1024.0); // period 16 samples: bin 4 of every 64-sample frame
        let obs: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let sigma = if i < n / 2 { 0.2 } else { 0.8 };
                c + sigma * rng.gauss()
            })
            .collect();
        let ms = stft_wiener_tracked_ms(&obs, 64, 16);
        let global = wiener_white(&obs, 0.5);
        let s_ms = snr_db(&clean, &ms);
        let s_global = snr_db(&clean, &global);
        assert!(
            s_ms >= s_global + 1.0,
            "tracked_ms {s_ms:.2} dB must beat global wiener_white(0.5) {s_global:.2} dB \
             by 1 dB on a step noise level"
        );
    }

    #[test]
    fn tracked_ms_is_deterministic_and_parameters_are_live() {
        let n = 512;
        let mut rng = Lcg::new(89);
        let clean = sine(n, 32.0);
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.4 * rng.gauss()).collect();
        // Determinism: two runs are byte-identical (pure f64 arithmetic, no hidden state).
        assert_eq!(
            stft_wiener_tracked_ms(&obs, 64, 16),
            stft_wiener_tracked_ms(&obs, 64, 16)
        );
        assert_ne!(
            stft_wiener_tracked_ms(&obs, 64, 16),
            stft_wiener_tracked_ms(&obs, 64, 8),
            "hop is ignored"
        );
        assert_ne!(
            stft_wiener_tracked_ms(&obs, 64, 16),
            stft_wiener_tracked_ms(&obs, 128, 16),
            "frame_len is ignored"
        );
    }

    #[test]
    fn new_entry_points_degrade_gracefully() {
        // Module convention: degenerate inputs never panic; too-short inputs pass through.
        let empty: [f64; 0] = [];
        assert!(stft_mmse_lsa(&empty, 0.4, 64, 16, 0.98).is_empty());
        assert!(stft_wiener_tracked_ms(&empty, 64, 16).is_empty());
        assert_eq!(stft_mmse_lsa(&[2.5], 0.4, 64, 16, 0.98), vec![2.5]);
        assert_eq!(stft_wiener_tracked_ms(&[2.5], 64, 16), vec![2.5]);
        for len in 2..4_usize
        {
            let x: Vec<f64> = (0..len).map(|i| i as f64 - 1.0).collect();
            for out in [
                stft_mmse_lsa(&x, 0.4, 64, 16, 0.98),
                stft_wiener_tracked_ms(&x, 64, 16),
            ]
            {
                assert_eq!(out.len(), len);
                assert!(out.iter().all(|v| v.is_finite()));
            }
        }
        // Non-positive noise level: nothing to subtract, exact pass-through.
        let mut rng = Lcg::new(97);
        let x: Vec<f64> = (0..100).map(|_| rng.gauss()).collect();
        assert_eq!(stft_mmse_lsa(&x, 0.0, 64, 16, 0.98), x);
        assert_eq!(stft_mmse_lsa(&x, -1.0, 64, 16, 0.98), x);
        // Out-of-range alpha clamps, pathological frame/hop requests clamp — never fatal.
        for out in [
            stft_mmse_lsa(&x, 0.3, 0, 0, 2.0),
            stft_wiener_tracked_ms(&x, 0, 0),
        ]
        {
            assert_eq!(out.len(), x.len());
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn new_variants_essentially_preserve_a_constant() {
        // A constant lives in the DC bin, whose power towers over any noise floor.
        let x = vec![3.5; 256];
        for (name, out) in [
            ("stft_mmse_lsa", stft_mmse_lsa(&x, 0.1, 64, 16, 0.98)),
            ("stft_wiener_tracked_ms", stft_wiener_tracked_ms(&x, 64, 16)),
        ]
        {
            assert_eq!(out.len(), x.len(), "{name}");
            for v in out.iter()
            {
                assert!((v - 3.5).abs() < 0.05, "{name}: constant became {v}");
            }
        }
        // Streaming: past the zero-priming fade-in (settled from 2·delay by the documented
        // contract) the constant must come back.
        let mut f = StreamingStftWiener::new(64, 16, 0.1, 0.98);
        let out: Vec<f64> = x.iter().map(|&v| f.push(v)).collect();
        for (i, v) in out.iter().enumerate().skip(2 * f.delay())
        {
            assert!((v - 3.5).abs() < 0.05, "streaming sample {i}: {v}");
        }
    }

    // -----------------------------------------------------------------------
    // StreamingStftWiener
    // -----------------------------------------------------------------------

    #[test]
    fn streaming_stft_yields_one_finite_output_per_push() {
        for n in [0_usize, 1, 3, 50, 200]
        {
            let mut rng = Lcg::new(61);
            let sig: Vec<f64> = (0..n).map(|_| rng.gauss()).collect();
            for noise_std in [0.4, 0.0]
            {
                let mut f = StreamingStftWiener::new(64, 16, noise_std, 0.98);
                let out: Vec<f64> = sig.iter().map(|&x| f.push(x)).collect();
                assert_eq!(out.len(), n);
                assert!(
                    out.iter().all(|v| v.is_finite()),
                    "n={n} noise_std={noise_std}"
                );
            }
        }
    }

    #[test]
    fn streaming_stft_matches_batch_snr_after_delay_compensation() {
        // The delay contract: out[i] estimates x[i − delay]. Over the steady-state region
        // (warm-up and the decision-directed spin-up skipped) the causal filter must land
        // within 1.5 dB of the batch stft_wiener_dd at identical parameters — the only
        // differences are the borders (no mirror padding) and the cold-started recursion.
        let n = 4096;
        let mut rng = Lcg::new(67);
        let clean = sine(n, 256.0); // period 16 samples: bin 4 of every 64-sample frame
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.4 * rng.gauss()).collect();
        let mut f = StreamingStftWiener::new(64, 16, 0.4, 0.98);
        let d = f.delay();
        assert_eq!(d, 64);
        let out: Vec<f64> = obs.iter().map(|&x| f.push(x)).collect();
        let start = 4 * d;
        let reference = &clean[start - d..n - d];
        let s_stream = snr_db(reference, &out[start..]);
        let batch = stft_wiener_dd(&obs, 0.4, 64, 16, 0.98);
        let s_batch = snr_db(reference, &batch[start - d..n - d]);
        assert!(
            (s_stream - s_batch).abs() <= 1.5,
            "streaming {s_stream:.2} dB must be within 1.5 dB of batch {s_batch:.2} dB"
        );
        // And the warm-up is exactly the documented zeros.
        assert!(out[..d].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn streaming_stft_reset_reproduces_identical_outputs() {
        let mut rng = Lcg::new(71);
        let sig: Vec<f64> = (0..300).map(|_| rng.gauss()).collect();
        for noise_std in [0.5, 0.0]
        {
            let mut f = StreamingStftWiener::new(64, 16, noise_std, 0.98);
            let first: Vec<f64> = sig.iter().map(|&x| f.push(x)).collect();
            f.reset();
            let second: Vec<f64> = sig.iter().map(|&x| f.push(x)).collect();
            assert_eq!(
                first, second,
                "noise_std={noise_std}: not reproducible after reset()"
            );
        }
    }

    #[test]
    fn streaming_stft_parameters_are_live() {
        let n = 512;
        let mut rng = Lcg::new(73);
        let clean = sine(n, 32.0);
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.4 * rng.gauss()).collect();
        let run = |frame: usize, hop: usize, ns: f64, alpha: f64| -> Vec<f64> {
            let mut f = StreamingStftWiener::new(frame, hop, ns, alpha);
            obs.iter().map(|&x| f.push(x)).collect()
        };
        let base = run(64, 16, 0.4, 0.98);
        assert_ne!(base, run(64, 8, 0.4, 0.98), "hop is ignored");
        assert_ne!(base, run(128, 16, 0.4, 0.98), "frame_len is ignored");
        assert_ne!(base, run(64, 16, 0.2, 0.98), "noise_std is ignored");
        assert_ne!(base, run(64, 16, 0.4, 0.5), "alpha is ignored");
        // Constructor clamps mirror the batch conventions, observable through delay().
        assert_eq!(StreamingStftWiener::new(48, 16, 0.4, 0.98).delay(), 64);
        assert_eq!(StreamingStftWiener::new(0, 0, 0.4, 0.98).delay(), 4);
        let mut tiny = StreamingStftWiener::new(0, 0, 0.4, 0.98);
        let out: Vec<f64> = obs.iter().map(|&x| tiny.push(x)).collect();
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn streaming_stft_tracking_mode_improves_snr() {
        // noise_std ≤ 0 switches to the blind per-bin minimum-statistics floor; on the
        // stationary fixture the delay-compensated output must still clearly beat raw.
        let n = 4096;
        let mut rng = Lcg::new(79);
        let clean = sine(n, 256.0);
        let obs: Vec<f64> = clean.iter().map(|&c| c + 0.4 * rng.gauss()).collect();
        let mut f = StreamingStftWiener::new(64, 16, 0.0, 0.98);
        let d = f.delay();
        let out: Vec<f64> = obs.iter().map(|&x| f.push(x)).collect();
        let start = 4 * d;
        let reference = &clean[start - d..n - d];
        let s_stream = snr_db(reference, &out[start..]);
        let s_raw = snr_db(reference, &obs[start - d..n - d]);
        assert!(
            s_stream > s_raw + 2.0,
            "blind streaming {s_stream:.2} dB must beat raw {s_raw:.2} dB by 2 dB"
        );
    }

    #[test]
    fn streaming_stft_works_through_the_trait_object() {
        // The hand-written StreamingDenoiser forwarding impl must expose the exact inherent
        // behavior (and prove object safety alongside the streaming.rs filters).
        let mut rng = Lcg::new(101);
        let sig: Vec<f64> = (0..256).map(|_| rng.gauss()).collect();
        let mut inherent = StreamingStftWiener::new(64, 16, 0.3, 0.98);
        let expect: Vec<f64> = sig.iter().map(|&x| inherent.push(x)).collect();
        let mut boxed: Box<dyn StreamingDenoiser> =
            Box::new(StreamingStftWiener::new(64, 16, 0.3, 0.98));
        assert_eq!(boxed.delay(), 64);
        let got: Vec<f64> = sig.iter().map(|&x| boxed.push(x)).collect();
        assert_eq!(got, expect);
        boxed.reset();
        let again: Vec<f64> = sig.iter().map(|&x| boxed.push(x)).collect();
        assert_eq!(again, expect);
    }
}
