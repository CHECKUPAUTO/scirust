//! # Débruitage & détection de bruit — a broad, extensible noise-removal toolkit
//!
//! This module answers two questions at once:
//!
//! 1. **How do we make noise removal "exhaustive"?** Not by enumerating every
//!    algorithm ever published — that set is open-ended — but by fixing a small,
//!    closed **taxonomy of families** and covering each one. The selection engines
//!    ([`denoise_auto`], [`denoise_best`], [`denoise_cascade`]) map the detected
//!    noise character onto the matching family and apply a concrete routine (or a
//!    shortlist of candidates) from it. The families below cover the standard
//!    signal-processing literature end to end:
//!
//!    | Family | Idea | Best against |
//!    |--------|------|--------------|
//!    | Linear | LTI convolution (moving average, Gaussian, Savitzky-Golay, EMA) | broadband Gaussian, gentle smoothing |
//!    | Rank | order statistics (median, Hampel, α-trimmed mean) | impulsive / salt-and-pepper spikes |
//!    | Transform | Fourier / wavelet shrinkage (low-pass, notch — brick-wall *and* zero-phase IIR [`notch_iir`] — Wiener, wavelet threshold: universal / SURE / level-dependent / Bayes / block ([`wavelet_denoise_neighblock`]) / translation-invariant) | tonal interference, white & colored noise |
//!    | Transform × Adaptive | short-time (block) Wiener gains re-estimated per frame ([`stft_wiener`], [`stft_wiener_dd`], [`stft_wiener_auto`], [`stft_wiener_tracked`]) | **non-stationary** broadband noise: ramps, bursts, drifting colored floors |
//!    | Variational | penalized least squares (Tikhonov, Total Variation) | edge-preserving smoothing, baseline drift |
//!    | Adaptive | model / data-driven (Kalman RTS smoother, LMS/RLS line enhancers, non-local means [`nlm1d`]) | non-stationary noise, drifting tones, self-similar signals |
//!
//! 2. **How do we detect "any" noise on "any" signal?** By *characterizing* the
//!    noise with a fixed feature set rather than trying to recognize it by name.
//!    [`detect::classify`] estimates the noise level (robust MAD), impulsivity
//!    (kurtosis / crest factor), spectral flatness, periodicity (dominant spectral
//!    line prominence), low-frequency trend strength, and the `1/f` color slope,
//!    then a small decision tree maps that [`NoiseProfile`] onto a [`NoiseType`].
//!    [`denoise_auto`] closes the loop: detect → pick the matching family → apply.
//!
//! ## Three entry points, one philosophy
//!
//! * [`denoise_auto`] — **one-shot, rule-based**: classify once, apply the single
//!   family the decision tree names. Cheapest; right when one noise process
//!   dominates. Periodic interference gets the full multi-line treatment
//!   ([`detect_lines`] + [`harmonic_stack`] + zero-phase IIR notching).
//! * [`denoise_best`] — **tournament-validated**: run a shortlist of 3–4 candidate
//!   denoisers from the profile's family and keep the one with the best
//!   *reference-free* score (residual whiteness minus an over/under-denoising
//!   penalty). Costs a few denoiser runs; right when the classification is
//!   ambiguous or the stakes justify an empirical check.
//! * [`denoise_cascade`] — **mixed noise**: repeat detect → treat → re-detect so
//!   spikes, hum, drift and a broadband floor are each removed by the family built
//!   for them, with loop protection and an accept-or-roll-back guard that keeps a
//!   broadband stage only when what it removed is noise-like, not a signal tone.
//!
//! **Signal-dependent noise** (photon counting, speckle, gain-dependent sensors)
//! gets a conditional pre/post step: [`denoise_auto`] first runs the conservative
//! [`vst::detect_noise_model`] and, when a Poisson-like or multiplicative law is
//! identified, denoises in the matched variance-stabilized domain and returns
//! through a **bias-corrected** inverse (module [`vst`]: Anscombe with the
//! Mäkitalo-Foi exact unbiased inverse, signed log with Duan smearing). Operators
//! that genuinely *couple* the channels of a multichannel record live in
//! [`multichannel`] (spatial joint Wiener; vector median with its honestly
//! measured negative verdict), and bounded saturating maps for display/feature
//! use — with no inverse, by design — in [`compand`].
//!
//! For real-time / embedded use, where no future samples exist, the [`streaming`]
//! module provides causal sample-by-sample counterparts (moving average, median,
//! Hampel, EMA, Kalman) behind the [`StreamingDenoiser`] trait. The window filters
//! and the EMA are bit-identical to their batch versions on the interior of a
//! record, at the price of a reported group delay; the streaming Kalman is the
//! causal *forward* filter (no Rauch-Tung-Striebel backward pass, which would need
//! future samples), so it trades a little smoothing for zero look-ahead rather than
//! reproducing the batch smoother exactly.
//!
//! Everything is pure Rust over `f64` slices, no external dependencies, and every
//! routine is validated by a signal-to-noise-ratio improvement test on a synthetic
//! signal with a *known* clean reference.
//!
//! ## Example
//!
//! ```
//! use scirust_signal::denoise::{classify, denoise_auto, separate};
//!
//! // A clean tone plus deterministic broadband disturbance.
//! let clean: Vec<f64> = (0..256).map(|i| (i as f64 * 0.2).sin()).collect();
//! let noisy: Vec<f64> = clean
//!     .iter()
//!     .enumerate()
//!     .map(|(i, &c)| c + 0.3 * ((i.wrapping_mul(2654435761)) as f64).sin())
//!     .collect();
//!
//! // 1. Characterize the noise without naming it up front.
//! let profile = classify(&noisy, 256.0);
//! println!("noise looks like {:?} (σ ≈ {:.3})", profile.dominant, profile.noise_std);
//!
//! // 2. One call detects the noise character and removes it.
//! let result = denoise_auto(&noisy, 256.0);
//! assert_eq!(result.output.len(), noisy.len());
//!
//! // 3. Or split into information + noise, with a whiteness self-check that flags
//! //    whether structure leaked into the "noise".
//! let sep = separate(&noisy, 256.0);
//! assert_eq!(sep.signal_estimate.len(), noisy.len());
//! assert!(sep.residual_whiteness >= 0.0 && sep.residual_whiteness <= 1.0);
//! ```

pub mod adaptive;
pub mod autotune;
pub mod block;
pub mod cascade;
pub mod collab;
pub mod compand;
pub mod detect;
pub mod iir;
pub mod linear;
pub mod multichannel;
pub mod nlm;
pub mod pipeline;
pub mod rank;
pub mod stft;
pub mod streaming;
pub mod transform;
pub mod variational;
pub mod vst;
pub mod vst_bench;

pub use adaptive::{
    KalmanFit, kalman_smooth, kalman_smooth_auto, kalman_trend_smooth, lms_line_enhancer,
    rls_line_enhancer,
};
pub use autotune::{DenoiseCase, VstAutotuneResult, autotune_vst, default_vst_candidates};
pub use block::wavelet_denoise_neighblock;
pub use cascade::{CascadeResult, CascadeStage, denoise_cascade, denoise_cascade_auto};
pub use collab::{collab1d, collab1d_auto};
pub use compand::{SoftClipKind, soft_clip, soft_clip_robust};
pub use detect::{
    NoiseProfile, NoiseType, Separation, SpectralLine, classify, detect_lines, estimate_noise_std,
    estimate_snr_db, harmonic_stack, separate,
};
pub use iir::{BiquadState, filtfilt_sos, notch_iir, rbj_notch, remove_mains_hum_iir};
pub use linear::{exp_moving_average, gaussian_smooth, moving_average, savitzky_golay};
pub use multichannel::{MultichannelGateReport, phase2_gate_report, vector_median, wiener_spatial};
pub use nlm::{nlm1d, nlm1d_auto};
pub use pipeline::{
    WaveletRlsRtsParams, reference_noise_cancel, wavelet_rls_rts_smooth, wavelet_rls_rts_smooth_1d,
    wavelet_rls_rts_smooth_multiref,
};
pub use rank::{alpha_trimmed_mean, hampel_filter, impulse_mask, median_filter};
pub use stft::{
    StreamingStftWiener, stft_mmse_lsa, stft_wiener, stft_wiener_auto, stft_wiener_dd,
    stft_wiener_tracked, stft_wiener_tracked_ms,
};
pub use streaming::{
    StreamingDenoiser, StreamingEma, StreamingHampel, StreamingKalman, StreamingMedian,
    StreamingMovingAverage, StreamingNlm, StreamingVst,
};
pub use transform::{
    ThresholdMode, Wavelet, cycle_spin, fft_highpass, fft_lowpass, notch_filter, remove_mains_hum,
    spectral_subtraction, wavelet_denoise, wavelet_denoise_bayes, wavelet_denoise_leveldep,
    wavelet_denoise_sure, wavelet_denoise_ti, wavelet_denoise_with, wiener_white,
};
pub use variational::{
    tikhonov_smooth, total_variation, total_variation_exact, total_variation_norm,
};
pub use vst::{
    VstAutoResult, VstKind, anscombe_inverse_exact_unbiased, detect_noise_model, vst_denoise,
    vst_denoise_auto, vst_forward, vst_inverse_corrected, vst_inverse_naive,
};

/// Result of the automatic detect-then-denoise pipeline.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AutoResult {
    /// The noise characterization that drove the choice.
    pub profile: NoiseProfile,
    /// Which method was selected (human-readable).
    pub method: String,
    /// The denoised signal.
    pub output: Vec<f64>,
}

/// **Universal denoiser**: characterize the noise, pick the matching family, apply.
///
/// This is the "one call that handles any signal" entry point. It never fails:
/// for an unrecognized or clean signal it falls back to a light Savitzky-Golay
/// smooth. `sample_rate` is used only to report physical frequencies and to size
/// the notch filter; pass `1.0` if you work in normalized units.
///
/// ## Signal-dependent noise: the conditional VST step
///
/// Before classifying, [`vst::detect_noise_model`] checks (conservatively —
/// default is "no") whether the noise variance follows the signal level:
/// Poisson-like (`σ² ∝ level`, photon counting) or multiplicative (`σ ∝ level`,
/// speckle, sensor gain). When one of those laws is identified with high
/// confidence, the whole pipeline runs in the matched variance-stabilized domain
/// — forward VST, then this same detect-and-treat core on the now-homoscedastic
/// record, then the **bias-corrected** inverse (Mäkitalo-Foi exact unbiased
/// inverse for Anscombe, Duan smearing for the log) — and the method string
/// reports the sandwich. On ordinary additive-noise records the detector returns
/// `Identity` and nothing changes.
pub fn denoise_auto(signal: &[f64], sample_rate: f64) -> AutoResult {
    let kind = vst::detect_noise_model(signal);
    if kind != VstKind::Identity
    {
        let (fwd_name, inv_name) = match kind
        {
            VstKind::Anscombe => ("anscombe", "exact_unbiased_inverse"),
            VstKind::SignedLog => ("signed_log", "smearing_inverse"),
            // The conservative detector only ever returns the two kinds above,
            // but the sandwich is correct for any manually supplied kind too.
            _ => ("vst", "smearing_inverse"),
        };
        let transformed = vst_forward(kind, signal);
        let inner = denoise_auto_core(&transformed, sample_rate);
        if inner.output.len() == transformed.len()
        {
            let residuals: Vec<f64> = transformed
                .iter()
                .zip(inner.output.iter())
                .map(|(t, d)| t - d)
                .collect();
            let output = vst_inverse_corrected(kind, &inner.output, &residuals);
            return AutoResult {
                profile: inner.profile,
                method: format!("{fwd_name} ∘ {} ∘ {inv_name}", inner.method),
                output,
            };
        }
    }
    denoise_auto_core(signal, sample_rate)
}

/// The detect-and-treat core of [`denoise_auto`]: classify the (possibly
/// variance-stabilized) record, apply the single family the decision tree names.
fn denoise_auto_core(signal: &[f64], sample_rate: f64) -> AutoResult {
    let profile = detect::classify(signal, sample_rate);
    let (method, output): (String, Vec<f64>) = match profile.dominant
    {
        NoiseType::LowNoise => ("savitzky_golay(2, 5)".into(), savitzky_golay(signal, 2, 5)),
        NoiseType::Impulsive => (
            "hampel_filter(3, 3.0)".into(),
            hampel_filter(signal, 3, 3.0),
        ),
        NoiseType::Periodic => notch_detected_lines(signal, sample_rate, profile.dominant_freq_hz),
        NoiseType::Baseline =>
        {
            let base = tikhonov_smooth(signal, 1.0e4);
            let out: Vec<f64> = signal.iter().zip(base.iter()).map(|(s, b)| s - b).collect();
            ("baseline_removal (signal − tikhonov trend)".into(), out)
        },
        NoiseType::Gaussian =>
        {
            // Stationary broadband noise: the Wiener gain leaves the whitest residual.
            (
                "wiener_white".into(),
                wiener_white(signal, profile.noise_std),
            )
        },
        NoiseType::Colored =>
        {
            // Level-dependent thresholds are the Johnstone-Silverman answer to
            // colored noise; on records short enough to afford it, cycle spinning
            // additionally averages out the decimated transform's shift artefacts.
            if signal.len() <= 8192
            {
                (
                    "cycle_spin(15) × wavelet_denoise_leveldep(auto, soft, db4)".into(),
                    cycle_spin(signal, 15, |x| {
                        wavelet_denoise_leveldep(x, 0, ThresholdMode::Soft, Wavelet::Db4)
                    }),
                )
            }
            else
            {
                (
                    "wavelet_denoise_leveldep(auto, soft, db4)".into(),
                    wavelet_denoise_leveldep(signal, 0, ThresholdMode::Soft, Wavelet::Db4),
                )
            }
        },
    };
    AutoResult {
        profile,
        method,
        output,
    }
}

/// The shared periodic-interference treatment of [`denoise_auto`] and
/// [`cascade::denoise_cascade`]: peel up to five spectral lines with
/// [`detect_lines`], recognize a harmonic family with [`harmonic_stack`], and
/// notch with the zero-phase filters of [`iir`] / [`transform`] — one
/// [`remove_mains_hum_iir`] comb for a stack of ≥ 2 harmonically related lines
/// (sized to the *highest* detected harmonic, [`detect::harmonic_span`], so no
/// member is left un-notched), otherwise one notch per line. Notch bandwidths
/// follow the [`denoise_auto`] sizing rule `max(5 % of f, 4 FFT bins)`.
///
/// Two correctness guards:
///
/// * **Don't notch the signal.** A detected line within two bins of the
///   information component's own dominant tone ([`detect::signal_dominant_freq`])
///   is the *signal*, not an interferer; it is dropped from the notch list. If that
///   leaves nothing to notch (the Periodic verdict was really the signal's tone on a
///   quiet floor), a light broadband Wiener runs instead of self-notching.
/// * **Honest near-Nyquist treatment.** The zero-phase IIR notch degenerates to a
///   pass-through at or above Nyquist, so a line in the top 2 % of the band is
///   notched with the brick-wall [`transform::notch_filter`] (which *can* zero the
///   Nyquist bin) — the returned method string never claims a notch that did not run.
///
/// The returned method string lists every frequency actually notched.
pub(crate) fn notch_detected_lines(
    signal: &[f64],
    sample_rate: f64,
    fallback_hz: f64,
) -> (String, Vec<f64>) {
    let nyquist = sample_rate * 0.5;
    let bin = sample_rate / next_pow2(signal.len().max(2)) as f64;
    let bw_for = |f: f64| (f * 0.05).max(bin * 4.0);
    // Notch one line, staying honest near Nyquist where the IIR design is identity.
    let notch_one = |x: &[f64], f: f64| -> Vec<f64> {
        if f >= 0.98 * nyquist
        {
            notch_filter(x, sample_rate, f, bw_for(f))
        }
        else
        {
            notch_iir(x, sample_rate, f, bw_for(f))
        }
    };

    // Protect the signal's own dominant tone from being notched as "interference".
    let sig_tone = detect::signal_dominant_freq(signal, sample_rate);
    let protects = |f: f64| sig_tone.is_some_and(|fs| (f - fs).abs() <= 2.0 * bin);

    let mut lines = detect_lines(signal, sample_rate, 5);
    lines.retain(|l| !protects(l.freq_hz));

    if lines.is_empty()
    {
        // Nothing to notch, or the only line was the signal itself: a light broadband
        // Wiener removes the actual floor without destroying the tone.
        if fallback_hz <= 0.0 || protects(fallback_hz)
        {
            let out = wiener_white(signal, estimate_noise_std(signal));
            return (
                "periodic verdict resolved to signal tone; broadband wiener".into(),
                out,
            );
        }
        let out = notch_one(signal, fallback_hz);
        return (format!("notch @ {fallback_hz:.3} Hz"), out);
    }

    if let Some((f0, count)) = harmonic_stack(&lines)
    {
        // Notch every harmonic up to the highest detected one, not just `count`.
        let n_harm = detect::harmonic_span(&lines, f0);
        let out = remove_mains_hum_iir(signal, sample_rate, f0, n_harm, bw_for(f0));
        let freqs: Vec<String> = lines.iter().map(|l| format!("{:.3}", l.freq_hz)).collect();
        return (
            format!(
                "remove_mains_hum_iir @ {f0:.3} Hz × {n_harm} harmonics \
                 ({count} lines: {} Hz)",
                freqs.join(", ")
            ),
            out,
        );
    }
    let mut out = signal.to_vec();
    let mut freqs = Vec::with_capacity(lines.len());
    for line in &lines
    {
        out = notch_one(&out, line.freq_hz);
        freqs.push(format!("{:.3}", line.freq_hz));
    }
    (format!("notch @ {} Hz", freqs.join(", ")), out)
}

/// The winner of a [`denoise_best`] tournament, with the full scoreboard.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BestResult {
    /// The winning candidate's denoised signal (same length as the input).
    pub output: Vec<f64>,
    /// The winning candidate's name (human-readable, includes the parameters).
    pub method: String,
    /// The winning candidate's score (see [`denoise_best`] for the criterion).
    pub score: f64,
    /// Every candidate that competed, as `(name, score)` pairs in shortlist order.
    pub candidates: Vec<(String, f64)>,
}

/// **Tournament denoiser**: run a shortlist of candidate methods from the noise
/// profile's family and keep the output with the best *reference-free* score.
///
/// Where [`denoise_auto`] trusts the decision tree's single pick, this entry point
/// validates it empirically — useful when the classification is ambiguous (e.g.
/// Gaussian vs. colored) or when a few extra denoiser runs are cheap relative to
/// getting it wrong.
///
/// ## The scoring criterion (no clean reference required)
///
/// `score = W(x − ŷ) − 0.5·min(1, |σ_res/σ̂ − 1|)`, where
///
/// * `W` is the residual whiteness self-check shared with [`separate`] (fraction
///   of autocorrelation lags of the removed component inside the `±1.96/√N`
///   white-noise band): a correct denoiser removes *structureless* residue, so
///   whiter is better;
/// * `σ_res` is the standard deviation of the removed component and `σ̂ =`
///   [`estimate_noise_std`]`(x)` the robust noise level of the input. Whiteness
///   alone is gameable from both sides — the identity map removes nothing (its
///   empty residual is trivially "white") and an over-smoother removes signal
///   along with the noise — so the penalty term punishes any mismatch between how
///   much was removed and how much noise is actually there: `σ_res ≪ σ̂` is
///   under-denoising, `σ_res ≫ σ̂` is over-smoothing, both capped at half a point.
///
/// Ties go to the earlier candidate in the shortlist (deterministic). Degrades
/// gracefully: on degenerate inputs every candidate is an identity copy and the
/// result is a copy of the input.
pub fn denoise_best(signal: &[f64], sample_rate: f64) -> BestResult {
    let profile = detect::classify(signal, sample_rate);
    let sigma_hat = profile.noise_std;

    // Shortlist 3-4 candidates from the profile's family.
    let mut pool: Vec<(String, Vec<f64>)> = Vec::new();
    match profile.dominant
    {
        NoiseType::Gaussian =>
        {
            pool.push(("wiener_white".into(), wiener_white(signal, sigma_hat)));
            pool.push(("stft_wiener_auto".into(), stft_wiener_auto(signal)));
            pool.push((
                "wavelet_denoise_ti(auto, soft, db4, 15)".into(),
                wavelet_denoise_ti(signal, 0, ThresholdMode::Soft, Wavelet::Db4, 15),
            ));
            pool.push((
                "kalman_smooth_auto".into(),
                kalman_smooth_auto(signal).output,
            ));
        },
        NoiseType::Colored =>
        {
            pool.push((
                "wavelet_denoise_leveldep(auto, soft, db4)".into(),
                wavelet_denoise_leveldep(signal, 0, ThresholdMode::Soft, Wavelet::Db4),
            ));
            pool.push((
                "cycle_spin(15) × wavelet_denoise_leveldep(auto, soft, db4)".into(),
                cycle_spin(signal, 15, |x| {
                    wavelet_denoise_leveldep(x, 0, ThresholdMode::Soft, Wavelet::Db4)
                }),
            ));
            pool.push((
                "kalman_smooth_auto".into(),
                kalman_smooth_auto(signal).output,
            ));
        },
        NoiseType::Impulsive =>
        {
            pool.push((
                "hampel_filter(3, 3.0)".into(),
                hampel_filter(signal, 3, 3.0),
            ));
            pool.push(("median_filter(3)".into(), median_filter(signal, 3)));
            pool.push((
                "alpha_trimmed_mean(3, 0.25)".into(),
                alpha_trimmed_mean(signal, 3, 0.25),
            ));
        },
        NoiseType::Periodic =>
        {
            // All three candidates are genuine noise *removers*. (A line enhancer is
            // deliberately excluded: it returns the predicted narrowband component —
            // the interference plus any tonal signal — not the cleaned signal, and
            // its broadband prediction error would fool the whiteness score into
            // crowning a fully un-notched output.)
            let bin = sample_rate / next_pow2(signal.len().max(2)) as f64;
            let f = profile.dominant_freq_hz;
            let bw = (f * 0.05).max(bin * 4.0);
            // notch_detected_lines already routes a harmonic stack to
            // remove_mains_hum_iir and isolated lines to per-line notches.
            pool.push(notch_detected_lines(signal, sample_rate, f));
            pool.push((
                format!("notch_iir @ {f:.3} Hz"),
                notch_iir(signal, sample_rate, f, bw),
            ));
            pool.push((
                format!("notch_filter (brick-wall) @ {f:.3} Hz"),
                notch_filter(signal, sample_rate, f, bw),
            ));
        },
        NoiseType::Baseline =>
        {
            let tik = tikhonov_smooth(signal, 1.0e4);
            pool.push((
                "baseline_removal (signal − tikhonov trend)".into(),
                signal.iter().zip(tik.iter()).map(|(s, b)| s - b).collect(),
            ));
            // Kalman trend: only the variance *ratios* matter, so (0, 1e-6, 1)
            // is a scale-free "very stiff trend" configuration.
            let trend = kalman_trend_smooth(signal, 0.0, 1.0e-6, 1.0);
            pool.push((
                "baseline_removal (signal − kalman trend)".into(),
                signal
                    .iter()
                    .zip(trend.iter())
                    .map(|(s, b)| s - b)
                    .collect(),
            ));
            // High-pass cutoff at the classifier's trend-band edge (≈ np/256 bins).
            let np = next_pow2(signal.len().max(2));
            let cutoff = (np as f64 / 256.0).max(2.0) * sample_rate / np as f64;
            pool.push((
                format!("fft_highpass @ {cutoff:.3} Hz"),
                fft_highpass(signal, sample_rate, cutoff),
            ));
        },
        NoiseType::LowNoise =>
        {
            // The identity goes first: on a low-noise verdict "do nothing" is the
            // least-harm default, and score ties resolve to the earlier candidate.
            pool.push(("passthrough".into(), signal.to_vec()));
            pool.push(("savitzky_golay(2, 5)".into(), savitzky_golay(signal, 2, 5)));
            pool.push((
                "kalman_smooth_auto".into(),
                kalman_smooth_auto(signal).output,
            ));
        },
    }

    // Score every candidate; keep the best (ties → earlier entry).
    let mut candidates: Vec<(String, f64)> = Vec::with_capacity(pool.len());
    let mut best_idx = 0;
    let mut best_score = f64::NEG_INFINITY;
    let mut outputs: Vec<Vec<f64>> = Vec::with_capacity(pool.len());
    for (i, (name, out)) in pool.into_iter().enumerate()
    {
        let residual: Vec<f64> = signal.iter().zip(out.iter()).map(|(s, o)| s - o).collect();
        let whiteness = detect::whiteness_of(&residual);
        let sigma_res = detect::std_of(&residual);
        let penalty = if sigma_hat > 0.0
        {
            0.5 * (sigma_res / sigma_hat - 1.0).abs().min(1.0)
        }
        else if sigma_res > 0.0
        {
            // The input carries no measurable noise: anything removed is signal.
            0.5
        }
        else
        {
            0.0
        };
        let score = whiteness - penalty;
        if score > best_score
        {
            best_score = score;
            best_idx = i;
        }
        candidates.push((name, score));
        outputs.push(out);
    }
    BestResult {
        output: outputs.swap_remove(best_idx),
        method: candidates[best_idx].0.clone(),
        score: candidates[best_idx].1,
        candidates,
    }
}

// ---------------------------------------------------------------------------
// Shared helpers used across the submodules.
// ---------------------------------------------------------------------------

/// Reflect an out-of-range index back into `0..n` (symmetric / edge-repeated
/// mirror), so windowed filters can run over signal borders without shrinking.
pub(crate) fn mirror_index(i: isize, n: usize) -> usize {
    if n <= 1
    {
        return 0;
    }
    let n_i = n as isize;
    let period = 2 * n_i;
    let mut m = i.rem_euclid(period);
    if m >= n_i
    {
        m = period - 1 - m;
    }
    m as usize
}

/// Median of a slice (clones and sorts). Returns 0.0 for an empty slice.
///
/// Sorts with [`f64::total_cmp`] — a *total* order. A `partial_cmp`-based comparator
/// is inconsistent when the slice contains NaN (NaN compares "equal" to everything
/// while the finite values still order among themselves), which modern Rust sorts
/// detect and **panic** on; `total_cmp` orders NaN deterministically instead, so a
/// stray NaN degrades gracefully rather than crashing the rank filters that build on
/// this. For all-finite input the ordering is identical to `partial_cmp`.
pub(crate) fn median(values: &[f64]) -> f64 {
    if values.is_empty()
    {
        return 0.0;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1
    {
        v[n / 2]
    }
    else
    {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Median absolute deviation: `median(|x − median(x)|)`. A robust, breakdown-heavy
/// scale estimator immune to a minority of outliers.
pub(crate) fn mad(values: &[f64]) -> f64 {
    if values.is_empty()
    {
        return 0.0;
    }
    let med = median(values);
    let dev: Vec<f64> = values.iter().map(|&x| (x - med).abs()).collect();
    median(&dev)
}

/// Robust noise-σ estimate (Donoho MAD on the finest Haar detail band). Shared by
/// [`detect::estimate_noise_std`] and any denoiser that needs a noise scale.
pub(crate) fn estimate_noise_std_helper(signal: &[f64]) -> f64 {
    let n = signal.len();
    if n < 2
    {
        return 0.0;
    }
    let half = n / 2;
    let mut detail = Vec::with_capacity(half);
    for i in 0..half
    {
        detail.push((signal[2 * i] - signal[2 * i + 1]) / core::f64::consts::SQRT_2);
    }
    mad(&detail) / 0.6745
}

/// Smallest power of two `>= n` (at least 1).
pub(crate) fn next_pow2(n: usize) -> usize {
    if n <= 1
    {
        return 1;
    }
    n.next_power_of_two()
}

/// Right-pad a signal by symmetric reflection up to the next power of two, so the
/// power-of-two FFT and Haar transform can process arbitrary-length inputs with a
/// smooth (non-discontinuous) boundary. Returns the original slice unchanged when
/// it is already a power of two.
pub(crate) fn pad_reflect_pow2(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let target = next_pow2(n);
    if target == n
    {
        return signal.to_vec();
    }
    let mut out = Vec::with_capacity(target);
    out.extend_from_slice(signal);
    for i in 0..(target - n)
    {
        let idx = mirror_index((n + i) as isize, n);
        out.push(signal[idx]);
    }
    out
}

#[cfg(test)]
pub(crate) mod testutil {
    use core::f64::consts::PI;

    /// Deterministic 64-bit LCG so noise tests are reproducible without a `rand`
    /// dependency.
    pub(crate) struct Lcg(u64);

    impl Lcg {
        pub(crate) fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        /// Uniform in [0, 1).
        pub(crate) fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        /// Standard normal via Box-Muller.
        pub(crate) fn gauss(&mut self) -> f64 {
            let u1 = self.uniform().max(1.0e-12);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
        }
    }

    /// Signal-to-noise ratio in dB of an estimate `est` against a clean reference.
    pub(crate) fn snr_db(clean: &[f64], est: &[f64]) -> f64 {
        let sig: f64 = clean.iter().map(|&x| x * x).sum();
        let err: f64 = clean
            .iter()
            .zip(est.iter())
            .map(|(&c, &e)| (c - e) * (c - e))
            .sum();
        10.0 * (sig / err.max(1.0e-30)).log10()
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::Lcg;
    use super::*;
    use core::f64::consts::PI;

    fn noisy_sine(n: usize) -> Vec<f64> {
        let mut rng = Lcg::new(79);
        (0..n)
            .map(|i| (2.0 * PI * 4.0 * i as f64 / n as f64).sin() + 0.3 * rng.gauss())
            .collect()
    }

    #[test]
    fn denoise_best_scoreboard_is_consistent() {
        // Plumbing invariants on a broadband-noise fixture: candidates exist, the
        // winner is one of them with the maximal score, and the length is kept.
        let obs = noisy_sine(512);
        let best = denoise_best(&obs, 512.0);
        assert_eq!(best.output.len(), obs.len());
        assert!(best.output.iter().all(|v| v.is_finite()));
        assert!(best.candidates.len() >= 3, "{:?}", best.candidates);
        assert!(
            best.candidates
                .iter()
                .any(|(name, score)| *name == best.method && *score == best.score),
            "winner {} not on the scoreboard {:?}",
            best.method,
            best.candidates
        );
        for (name, score) in &best.candidates
        {
            assert!(score.is_finite(), "{name} got a non-finite score");
            assert!(
                best.score >= *score,
                "winner {} ({}) trails {name} ({score})",
                best.method,
                best.score
            );
        }
    }

    #[test]
    fn denoise_best_improves_snr_on_impulsive_noise() {
        use super::testutil::snr_db;
        let n = 512;
        let mut rng = Lcg::new(211);
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 4.0 * i as f64 / n as f64).sin())
            .collect();
        let mut obs: Vec<f64> = clean.iter().map(|&c| c + 0.05 * rng.gauss()).collect();
        // Aperiodic (Bernoulli-placed) spikes — genuine impulsive noise, which the
        // classifier routes to the rank family (a periodic train would be treated as
        // a legitimate signal feature; see detect::periodic_impulse_train).
        for v in obs.iter_mut()
        {
            if rng.uniform() < 1.0 / 37.0
            {
                *v += 8.0;
            }
        }
        let best = denoise_best(&obs, 512.0);
        let s_raw = snr_db(&clean, &obs);
        let s_best = snr_db(&clean, &best.output);
        assert!(
            s_best > s_raw + 3.0,
            "winner {} gained only {:.2} dB ({s_raw:.2} → {s_best:.2})",
            best.method,
            s_best - s_raw
        );
    }

    #[test]
    fn denoise_best_degrades_gracefully() {
        let empty: [f64; 0] = [];
        let best = denoise_best(&empty, 1000.0);
        assert!(best.output.is_empty());
        assert!(!best.candidates.is_empty());
        for len in 1..4_usize
        {
            let x: Vec<f64> = (0..len).map(|i| i as f64 - 1.0).collect();
            let best = denoise_best(&x, 1000.0);
            assert_eq!(best.output.len(), len);
            assert!(best.output.iter().all(|v| v.is_finite()));
        }
        let c = vec![3.5; 64];
        let best = denoise_best(&c, 1000.0);
        assert_eq!(best.output, c, "a constant must survive the tournament");
    }

    #[test]
    fn denoise_auto_applies_corrected_vst_on_poisson_noise() {
        use super::testutil::snr_db;
        // Low-count Poisson: intensity λ_i ∈ [1, 11], sampled with Knuth's
        // algorithm on the deterministic LCG. The 40-cycle carrier keeps the
        // intensity out of the classifier's trend band — a slower sine would be
        // "detrended" away as Baseline on both arms, measuring nothing.
        let n = 4096;
        let mut rng = Lcg::new(7);
        let mut poisson = |lambda: f64| -> f64 {
            let l = (-lambda).exp();
            let mut k = 0u32;
            let mut p = 1.0;
            loop
            {
                k += 1;
                p *= rng.uniform();
                if p <= l
                {
                    break;
                }
            }
            (k - 1) as f64
        };
        let clean: Vec<f64> = (0..n)
            .map(|i| 6.0 + 5.0 * (2.0 * PI * 40.0 * i as f64 / n as f64).sin())
            .collect();
        let obs: Vec<f64> = clean.iter().map(|&l| poisson(l)).collect();

        let auto = denoise_auto(&obs, 1.0);
        assert!(
            auto.method.contains("anscombe") && auto.method.contains("exact_unbiased_inverse"),
            "expected the VST sandwich in the method string, got: {}",
            auto.method
        );
        // The stabilized pipeline must beat the same core run on the raw record.
        let core = denoise_auto_core(&obs, 1.0);
        let s_vst = snr_db(&clean, &auto.output);
        let s_raw = snr_db(&clean, &core.output);
        assert!(
            s_vst >= s_raw + 0.5,
            "VST sandwich {s_vst:.2} dB must beat the raw core {s_raw:.2} dB by ≥ 0.5 dB"
        );
    }

    #[test]
    fn denoise_auto_is_unchanged_on_additive_noise() {
        // Zero-mean additive fixture: the conservative detector must return
        // Identity, so the public entry point is byte-identical to the core.
        let obs = noisy_sine(2048);
        let auto = denoise_auto(&obs, 512.0);
        let core = denoise_auto_core(&obs, 512.0);
        assert_eq!(auto.output, core.output);
        assert_eq!(auto.method, core.method);
    }

    #[test]
    fn auto_v2_notches_the_whole_harmonic_stack() {
        use super::testutil::snr_db;
        // A 7 Hz signal buried under a 50/100/150 Hz mains stack. n = 2072 makes
        // the classifier's trimmed residual core exactly 2048 samples, so all
        // three harmonics sit on periodogram bins at fs = 1024.
        let n = 2072;
        let fs = 1024.0;
        let clean: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 7.0 * i as f64 / fs).sin())
            .collect();
        let obs: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let t = i as f64 / fs;
                c + 0.6 * (2.0 * PI * 50.0 * t).sin()
                    + 0.3 * (2.0 * PI * 100.0 * t).sin()
                    + 0.2 * (2.0 * PI * 150.0 * t).sin()
            })
            .collect();
        let auto = denoise_auto(&obs, fs);
        assert_eq!(auto.profile.dominant, NoiseType::Periodic);
        // The stack must be recognized and the method string must say so, listing
        // the fundamental.
        assert!(
            auto.method.contains("remove_mains_hum_iir") && auto.method.contains("50.000"),
            "method: {}",
            auto.method
        );
        // The OLD v1 behavior — a single brick-wall notch on the dominant line —
        // computed inline for comparison: it leaves the other two harmonics.
        let bin = fs / next_pow2(n) as f64;
        let f_dom = auto.profile.dominant_freq_hz;
        let old = notch_filter(&obs, fs, f_dom, (f_dom * 0.05).max(bin * 4.0));
        let s_new = snr_db(&clean, &auto.output);
        let s_old = snr_db(&clean, &old);
        assert!(
            s_new >= s_old + 3.0,
            "harmonic-aware notching {s_new:.2} dB must beat the old single notch \
             {s_old:.2} dB by ≥ 3 dB"
        );
    }

    #[test]
    fn auto_v2_periodic_method_lists_isolated_line_frequencies() {
        // Two unrelated tones: no stack, so each detected line gets its own
        // zero-phase notch and the method string names both frequencies.
        let n = 2072;
        let fs = 1024.0;
        let mut rng = Lcg::new(223);
        let obs: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 4.0 * t).sin()
                    + 0.8 * (2.0 * PI * 50.0 * t).sin()
                    + 0.8 * (2.0 * PI * 120.0 * t).sin()
                    + 0.05 * rng.gauss()
            })
            .collect();
        let auto = denoise_auto(&obs, fs);
        assert_eq!(auto.profile.dominant, NoiseType::Periodic);
        assert!(
            auto.method.contains("notch")
                && auto.method.contains("50.000")
                && auto.method.contains("120.000"),
            "method: {}",
            auto.method
        );
        // Both tones must actually be gone: the output is closer to the tone-free
        // signal than the input by a wide margin.
        use super::testutil::snr_db;
        let reference: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 4.0 * t).sin()
            })
            .collect();
        assert!(snr_db(&reference, &auto.output) > snr_db(&reference, &obs) + 8.0);
    }
}
