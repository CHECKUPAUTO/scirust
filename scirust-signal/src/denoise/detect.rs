//! Noise detection, characterization and **separation**.
//!
//! ## Detecting "any" noise without naming it
//!
//! You cannot recognize every noise process by name — the set is open. What you
//! *can* do is measure a fixed panel of descriptors that, together, pin down the
//! nature of the disturbance:
//!
//! * **level** — a robust noise standard deviation from the finest wavelet detail
//!   band (`σ = MAD / 0.6745`, immune to a minority of outliers);
//! * **impulsivity** — excess kurtosis and crest factor of the high-pass residual
//!   (heavy tails ⇒ spikes / salt-and-pepper), *gated on aperiodicity* so a periodic
//!   high-crest train (an ECG's QRS complexes) reads as a signal feature, not noise;
//! * **spectral flatness** — geometric-over-arithmetic mean of the power spectrum
//!   (≈1 ⇒ white; ≪1 ⇒ tonal or colored);
//! * **periodicity** — prominence of the strongest spectral line over the median
//!   (a mains hum or interferer stands far above the floor);
//! * **trend strength** — fraction of AC energy in the lowest band (baseline drift);
//! * **color** — the `1/f` slope of the log-log spectrum (white ≈ 0, pink ≈ −1,
//!   brown ≈ −2).
//!
//! [`classify`] rolls these into a [`NoiseProfile`] and a [`NoiseType`] verdict.
//!
//! ## Separating noise from information — and *proving* it
//!
//! "What is noise and what is information" is not an absolute label; it is defined
//! *relative to a model*. The operational definition used here: information is the
//! part a denoiser can predict as structure; noise is the residual left over. So
//! [`separate`] runs the auto-denoiser, calls its output the information estimate,
//! and the residual the noise estimate.
//!
//! The crucial extra step is **falsifying** that split: a correct separation leaves
//! a residual with *no remaining structure* — it must be white. [`separate`]
//! therefore runs a whiteness test on the residual (normalized autocorrelation vs
//! the `±1.96/√N` white-noise confidence band). If too many lags fall outside the
//! band, structure has leaked into the "noise" (the model under-fit and threw away
//! information), and [`Separation::leaked_structure`] flags it. That test is what
//! turns a plausible split into a checkable one.
//!
//! ## Known limitation: tonal interference deep inside the signal band
//!
//! The information/noise split is a 9-sample moving average, i.e. a *normalized*
//! cutoff near `0.05·fs`. A tonal interferer **below** that cutoff (e.g. 50 Hz mains
//! on a record sampled at 100 kHz) lands on the *information* side of the split,
//! where it is indistinguishable, without priors, from a legitimate low-frequency
//! signal component: every statistic derived from the split (smooth/residual energy
//! ratio, leak amplitude) is a deterministic function of `f/fs` and carries no
//! signal-vs-interference information. The automatic pipelines therefore do **not**
//! notch tones in that region — the self-notch protection errs on the side of
//! keeping them. When the interference frequency is known (mains hum almost always
//! is), call [`super::remove_mains_hum_iir`] or [`super::notch_iir`] explicitly
//! before the automatic pipeline.

use super::linear::moving_average;
use super::{estimate_noise_std_helper, pad_reflect_pow2};
use crate::Complex;
use crate::fft::{fft, fft_real, ifft};
use serde::{Deserialize, Serialize};

/// The dominant character of the noise on a signal, as judged by [`classify`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseType {
    /// Little to no noise detected relative to the signal.
    LowNoise,
    /// Broadband, roughly-white additive noise (flat spectrum, light tails).
    Gaussian,
    /// Impulsive noise: spikes, dropouts, salt-and-pepper (heavy-tailed residual).
    /// Only *aperiodic* spikes qualify — a **periodic** high-crest train (an ECG's
    /// QRS complexes, an engine's knock, any repeated transient) is a legitimate
    /// signal feature and is deliberately *not* classified Impulsive.
    Impulsive,
    /// Periodic interference: a strong tonal line such as mains hum.
    Periodic,
    /// Colored `1/f` noise (pink/brown): energy concentrated at low frequencies.
    Colored,
    /// Baseline wander / slow drift dominating the low-frequency band.
    Baseline,
}

/// A fixed panel of noise descriptors plus the [`NoiseType`] verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseProfile {
    /// Robust noise standard deviation (Donoho MAD estimator).
    pub noise_std: f64,
    /// Estimated signal-to-noise ratio in dB.
    pub snr_db: f64,
    /// Excess kurtosis of the high-pass residual (impulsivity; 0 = Gaussian).
    pub residual_kurtosis: f64,
    /// Crest factor (peak / RMS) of the residual (impulsivity).
    pub crest_factor: f64,
    /// Spectral flatness / Wiener entropy in [0, 1] (1 = white, 0 = pure tone).
    pub spectral_flatness: f64,
    /// `1/f` color exponent from the log-log spectrum (white ≈ 0, pink ≈ −1).
    pub spectral_slope: f64,
    /// Frequency (Hz) of the strongest spectral line, if any.
    pub dominant_freq_hz: f64,
    /// Prominence of that line: peak PSD divided by the median PSD.
    pub peak_prominence: f64,
    /// Fisher's g statistic: strongest periodogram ordinate over the total. A
    /// number-of-bins-aware test for a single dominant spectral line, robust where
    /// raw prominence is fooled by the natural spread of white-noise ordinates.
    pub line_dominance: f64,
    /// Fraction of AC energy in the lowest band (baseline-drift strength).
    pub trend_strength: f64,
    /// The classifier's verdict.
    pub dominant: NoiseType,
}

/// One spectral line found by [`detect_lines`]: a periodogram peak of the high-pass
/// residual that passes Fisher's g-test against the white-noise null.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpectralLine {
    /// Frequency of the line in Hz (bin center of the residual periodogram).
    pub freq_hz: f64,
    /// Fisher's g of this line at the moment it was peeled: its peak ordinate
    /// divided by the sum of the ordinates still standing (previous lines removed).
    pub fisher_g: f64,
    /// Fraction of the total residual AC power carried by the peeled peak region
    /// (peak bin ± 2 bins), relative to the residual spectrum *before* any peeling.
    pub power_ratio: f64,
}

/// A signal/noise decomposition with a self-check on its own validity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Separation {
    /// The estimated information (structure) component.
    pub signal_estimate: Vec<f64>,
    /// The estimated noise component (observed − information).
    pub noise_estimate: Vec<f64>,
    /// Which denoiser produced the split.
    pub method: String,
    /// Standard deviation of the extracted noise.
    pub noise_std: f64,
    /// Signal-to-noise ratio in dB of the decomposition.
    pub snr_db: f64,
    /// Whiteness of the residual in [0, 1]: fraction of autocorrelation lags inside
    /// the white-noise confidence band. 1.0 ⇒ the residual is indistinguishable
    /// from white noise (ideal separation).
    pub residual_whiteness: f64,
    /// True when the residual still carries structure — a warning that the
    /// separation discarded information along with the noise.
    pub leaked_structure: bool,
    /// The noise characterization that drove the method choice.
    pub profile: NoiseProfile,
}

/// Robust noise standard-deviation estimate via the finest-scale Haar detail band:
/// `σ = MAD(detail) / 0.6745` (Donoho-Johnstone). Robust to a minority of impulses
/// and to smooth signal content, because differencing adjacent samples cancels the
/// signal while the noise survives.
pub fn estimate_noise_std(signal: &[f64]) -> f64 {
    estimate_noise_std_helper(signal)
}

/// Estimate the signal-to-noise ratio (dB) given a noise standard deviation, using
/// `SNR = 10·log10((var − σ²)/σ²)` with the signal power inferred as the excess of
/// the total variance over the noise variance.
pub fn estimate_snr_db(signal: &[f64], noise_std: f64) -> f64 {
    let n = signal.len();
    if n == 0 || noise_std <= 0.0
    {
        return f64::INFINITY;
    }
    let mean = signal.iter().sum::<f64>() / n as f64;
    let var = signal.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    let noise_pow = noise_std * noise_std;
    let sig_pow = (var - noise_pow).max(var * 1.0e-6);
    10.0 * (sig_pow / noise_pow).log10()
}

/// Characterize the noise on a signal: measure the descriptor panel and return a
/// [`NoiseProfile`] with the [`NoiseType`] verdict. `sample_rate` is used only to
/// report the dominant frequency in Hz; pass `1.0` for normalized units.
pub fn classify(signal: &[f64], sample_rate: f64) -> NoiseProfile {
    let n = signal.len();
    let noise_std = estimate_noise_std(signal);
    if n < 8
    {
        return NoiseProfile {
            noise_std,
            snr_db: estimate_snr_db(signal, noise_std),
            residual_kurtosis: 0.0,
            crest_factor: 0.0,
            spectral_flatness: 1.0,
            spectral_slope: 0.0,
            dominant_freq_hz: 0.0,
            peak_prominence: 0.0,
            line_dominance: 0.0,
            trend_strength: 0.0,
            dominant: NoiseType::LowNoise,
        };
    }

    let mean = signal.iter().sum::<f64>() / n as f64;
    let var = signal.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
    let rms = var.sqrt();
    let snr_db = estimate_snr_db(signal, noise_std);

    // Convention: the slow, low-frequency part is treated as *information*; noise is
    // what remains once it is subtracted. So the impulsivity, periodicity, flatness
    // and color descriptors are all measured on the high-pass residual, which keeps
    // additive noise and narrowband interference while discarding the smooth signal.
    let core = highpass_residual_core(signal);
    let residual_kurtosis = crate::features::kurtosis(&core);
    let crest_factor = crate::features::crest_factor(&core);
    let res_rms = std_of(&core);

    // Residual spectrum (mean-removed, reflection-padded) for the noise descriptors.
    let (res_psd, np) = residual_psd(&core);

    let spectral_flatness = spectral_flatness_of(&res_psd[1..]);
    let spectral_slope = loglog_slope(&res_psd);
    let (dom_bin, peak_prominence) = dominant_line(&res_psd, 2);
    let dominant_freq_hz = dom_bin as f64 * sample_rate / np as f64;

    // Fisher's g-test for a single dominant line: g = max ordinate / sum of ordinates
    // over the positive-frequency band. Its white-noise critical value shrinks with
    // the bin count `m`, so it is not fooled by the wide spread of white ordinates
    // the way a raw max/median prominence is.
    let line_dominance = fisher_g(&res_psd[1..]);
    let g_crit = fisher_g_critical(res_psd.len().saturating_sub(1));

    // Trend strength is, by contrast, measured on the *raw* spectrum's lowest band:
    // it is precisely the low-frequency energy the residual threw away as "signal".
    let raw_centered: Vec<f64> = signal.iter().map(|&x| x - mean).collect();
    let raw_psd: Vec<f64> = fft_real(&pad_reflect_pow2(&raw_centered))
        .iter()
        .map(|c| c.mag_sq())
        .collect();
    let trend_band = (raw_psd.len() / 128).clamp(2, raw_psd.len().max(2));
    let ac_energy: f64 = raw_psd[1..].iter().sum();
    let low_energy: f64 = raw_psd[1..trend_band.min(raw_psd.len())].iter().sum();
    let trend_strength = if ac_energy > 0.0
    {
        low_energy / ac_energy
    }
    else
    {
        0.0
    };

    // Robust edge score: the step-vs-drift discriminator for the Baseline gate
    // below (see [`step_edge_score`]). Only meaningful within the population that
    // reaches that gate (trend_strength high): drift-dominated records measure
    // 2–9, step/edge records 17–550.
    let edge_score = step_edge_score(signal);

    // Decision tree, ordered from most to least specific. Structured disturbances
    // (spikes, tones, drift) are tested before the level-based low-noise gate, since
    // they can dominate even when the broadband σ is small.
    //
    // The Baseline gate is calibrated on drift-vs-signal power: a drift at equal
    // power with the signal (0 dB) measures trend_strength ≈ 0.49 on the reference
    // fixtures while strongly colored AR(0.9) noise stays near 0.05, so 0.45 catches
    // the realistic drift regime the old 0.6 gate missed (0.6 required the drift to
    // carry ~1.5× the signal's power) with a wide safety margin against colored
    // noise. The edge-score guard keeps step/edge records — whose low-frequency
    // energy mimics a drift — out of the detrending branch, where subtracting a
    // "trend" would smear (at high SNR, essentially erase) their edges. When the
    // guard vetoes, the record falls through to the broadband branches: the drift
    // may go untreated, but the signal is never destroyed.
    //
    // The LowNoise gate at 5 % of the RMS routes essentially-noiseless records
    // (three pure tones plus a weak wander measure σ̂/rms ≈ 3–4 %) to the gentle
    // Savitzky-Golay touch instead of Colored: with no broadband floor, the
    // residual's spectral slope is measured on tone-leakage skirts, not on noise,
    // and the wavelet machinery it would trigger has nothing legitimate to remove.
    // The impulsive gate additionally requires the high-crest events to be
    // *aperiodic*: a periodic spike train (an ECG's QRS complexes, an engine's knock,
    // any repeated transient) is a legitimate signal feature, not impulsive noise, and
    // filtering it away as spikes would destroy signal. [`periodic_impulse_train`]
    // vetoes the verdict for such records, which then fall through to the broadband /
    // baseline branches. Random impulsive noise (salt-and-pepper, drop-outs, electrode
    // pops) is aperiodic and still classified Impulsive.
    let dominant =
        if residual_kurtosis > 4.0 && crest_factor > 5.0 && !periodic_impulse_train(&core)
        {
            NoiseType::Impulsive
        }
        else if line_dominance > g_crit.max(0.1)
            && res_rms > 0.05 * rms.max(1.0e-12)
            && dominant_freq_hz > 0.0
        {
            NoiseType::Periodic
        }
        else if trend_strength > 0.45 && edge_score < 13.0
        {
            NoiseType::Baseline
        }
        else if noise_std < 0.05 * rms.max(1.0e-12)
        {
            NoiseType::LowNoise
        }
        else if spectral_slope < -0.5
        {
            NoiseType::Colored
        }
        else
        {
            NoiseType::Gaussian
        };

    NoiseProfile {
        noise_std,
        snr_db,
        residual_kurtosis,
        crest_factor,
        spectral_flatness,
        spectral_slope,
        dominant_freq_hz,
        peak_prominence,
        line_dominance,
        trend_strength,
        dominant,
    }
}

/// Fisher's g statistic: the largest periodogram ordinate divided by the sum of all
/// ordinates. Under a white-noise null its distribution depends only on the number
/// of bins, which makes it a principled single-line periodicity test.
fn fisher_g(psd: &[f64]) -> f64 {
    let sum: f64 = psd.iter().sum();
    if sum <= 0.0
    {
        return 0.0;
    }
    let max = psd.iter().cloned().fold(0.0_f64, f64::max);
    max / sum
}

/// White-noise critical value of Fisher's g over `m_bins` periodogram ordinates at a
/// significance of 1e-4, from the leading term of Fisher's exact null distribution
/// `P(g > x) ≈ m·(1 − x)^{m−1}`. Shared by [`classify`] and [`detect_lines`] so both
/// call "line" at exactly the same evidence level.
fn fisher_g_critical(m_bins: usize) -> f64 {
    let m = m_bins.max(2) as f64;
    1.0 - (1.0e-4 / m).powf(1.0 / (m - 1.0))
}

/// The high-pass residual all the noise descriptors are measured on: the signal
/// minus its 9-point moving average, with the borders trimmed (reflection padding at
/// the smoother's edges can turn a large low-frequency swing into a couple of
/// spurious residual spikes that would otherwise masquerade as impulsive noise).
/// Callers guarantee `signal.len() >= 8` so the trimmed core is never empty.
fn highpass_residual_core(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let smooth = moving_average(signal, 9);
    let residual: Vec<f64> = signal
        .iter()
        .zip(smooth.iter())
        .map(|(s, m)| s - m)
        .collect();
    let trim = 12.min(n / 4);
    residual[trim..n - trim].to_vec()
}

/// Minimum repetition period (samples) the periodic-feature detector considers — a
/// spike train tighter than this is quasi-continuous, not a train of "rare peaks".
const PERIODIC_MIN_PERIOD: usize = 8;
/// Normalized energy-envelope autocorrelation above which the sparse, high-crest
/// events of the residual are judged a **legitimate periodic feature** — a
/// repetitive transient such as an ECG QRS train — rather than random impulsive
/// noise. Calibrated by measurement (`fixture is real ECG record 100`): a real QRS
/// train autocorrelates at ≈ 0.3–0.4 at its beat lag (≈ 73 bpm, robust to
/// heart-rate variability), a strictly periodic synthetic spike train far higher,
/// while aperiodic (Bernoulli-placed) impulses stay ≈ 0.2 — so a 0.30 gate separates
/// a periodic *signal* feature from impulsive *noise* with margin on both sides.
const PERIODIC_IMPULSE_AUTOCORR: f64 = 0.30;

/// Is the high-crest content of `core` a **periodic train** (a legitimate repetitive
/// feature — heart beats, engine knocks, repeated transients) rather than random
/// impulsive noise?
///
/// The impulse-energy envelope `core² − mean(core²)` of a periodic spike train
/// autocorrelates strongly at its repetition lag; random impulses do not. The linear
/// autocorrelation is computed via FFT (zero-padding to `≥ 2n` makes the circular
/// correlation equal the linear one over lags `0..n`) and normalized by its zero-lag
/// value; the maximum over repetition lags `[PERIODIC_MIN_PERIOD, n/3]` — at least
/// three periods must fit — is compared to [`PERIODIC_IMPULSE_AUTOCORR`]. Used to
/// **veto the Impulsive verdict** when the spikes are the *signal*, so a legitimate
/// periodic feature (an ECG's QRS complexes) is not filtered away as noise.
fn periodic_impulse_train(core: &[f64]) -> bool {
    let n = core.len();
    if n < 3 * PERIODIC_MIN_PERIOD
    {
        return false;
    }
    let mean_e = core.iter().map(|&x| x * x).sum::<f64>() / n as f64;
    // Linear autocorrelation of the zero-mean energy envelope via FFT.
    let m = (2 * n).next_power_of_two();
    let mut buf: Vec<Complex> = core
        .iter()
        .map(|&x| Complex::new(x * x - mean_e, 0.0))
        .collect();
    buf.resize(m, Complex::new(0.0, 0.0));
    fft(&mut buf);
    for c in buf.iter_mut()
    {
        *c = Complex::new(c.mag_sq(), 0.0);
    }
    ifft(&mut buf);
    let ac0 = buf[0].re;
    if ac0 <= 0.0
    {
        return false;
    }
    let lag_max = (n / 3).max(PERIODIC_MIN_PERIOD);
    let best = buf[PERIODIC_MIN_PERIOD..=lag_max]
        .iter()
        .map(|c| c.re / ac0)
        .fold(0.0_f64, f64::max);
    best > PERIODIC_IMPULSE_AUTOCORR
}

/// Periodogram of a (non-empty) residual core: mean-removed, reflection-padded to a
/// power of two, squared-magnitude of the positive-frequency FFT bins. Returns the
/// PSD ordinates and the padded length `np` (bin `k` sits at `k·sample_rate/np` Hz).
fn residual_psd(core: &[f64]) -> (Vec<f64>, usize) {
    let core_mean = core.iter().sum::<f64>() / core.len() as f64;
    let core_centered: Vec<f64> = core.iter().map(|&x| x - core_mean).collect();
    let core_padded = pad_reflect_pow2(&core_centered);
    let np = core_padded.len();
    let psd: Vec<f64> = fft_real(&core_padded).iter().map(|c| c.mag_sq()).collect();
    (psd, np)
}

/// Detect up to `max_lines` spectral lines by **iterative peeling** of the high-pass
/// residual periodogram — the multi-line extension of the single-line Fisher g-test
/// [`classify`] performs (Fisher 1929, *Tests of significance in harmonic analysis*;
/// peeling after Siegel 1980's observation that secondary lines hide under the
/// primary's dominance).
///
/// The residual PSD is built exactly as in [`classify`] (9-point moving-average
/// high-pass, trimmed borders, mean-removed, reflection-padded periodogram). Then,
/// repeatedly: the dominant ordinate (excluding DC and bin 1, like [`classify`]) is
/// tested with Fisher's g against the same white-noise critical value [`classify`]
/// uses; if it passes, the line is recorded and the peak bin ± 2 bins are zeroed —
/// the stored spectrum covers positive frequencies only, so this implicitly removes
/// the conjugate mirror bins as well — and g is recomputed on the ordinates still
/// standing. The loop stops at `max_lines`, or as soon as the strongest remaining
/// ordinate no longer beats the critical value (no line without evidence).
///
/// Degrades gracefully: fewer than 8 samples, `max_lines = 0`, or a constant signal
/// yield an empty vector. Feed the result to [`harmonic_stack`] to recognize a
/// mains-hum-style harmonic family, or notch each line individually.
pub fn detect_lines(signal: &[f64], sample_rate: f64, max_lines: usize) -> Vec<SpectralLine> {
    let n = signal.len();
    if n < 8 || max_lines == 0
    {
        return Vec::new();
    }
    let core = highpass_residual_core(signal);
    let (mut psd, np) = residual_psd(&core);
    let g_crit = fisher_g_critical(psd.len().saturating_sub(1));
    let total_power: f64 = psd[1..].iter().sum();
    if total_power <= 0.0
    {
        return Vec::new();
    }

    let mut lines = Vec::new();
    for _ in 0..max_lines
    {
        let remaining: f64 = psd[1..].iter().sum();
        if remaining <= 0.0
        {
            break;
        }
        let (bin, _) = dominant_line(&psd, 2);
        if bin == 0 || psd[bin] <= 0.0
        {
            break;
        }
        let g = psd[bin] / remaining;
        if g <= g_crit
        {
            break;
        }
        let lo = bin.saturating_sub(2).max(1);
        let hi = (bin + 2).min(psd.len() - 1);
        let peak_power: f64 = psd[lo..=hi].iter().sum();
        lines.push(SpectralLine {
            freq_hz: bin as f64 * sample_rate / np as f64,
            fisher_g: g,
            power_ratio: peak_power / total_power,
        });
        for p in psd[lo..=hi].iter_mut()
        {
            *p = 0.0;
        }
    }
    lines
}

/// Highest harmonic index a [`harmonic_stack`] test will consider. Beyond ~12 the
/// per-harmonic tolerance window (`±2 %·k·f0`) grows wide enough to swallow the
/// spacing between multiples of `f0`, so *any* high line would match *some*
/// fundamental — the failure mode that lets unrelated tones form spurious families.
/// Capping `k` keeps the test to the low harmonics real interferers actually excite.
const HARMONIC_MAX_INDEX: usize = 12;

/// Relative tolerance for calling a line the `k`-th harmonic of `f0`
/// (`|f − k·f0| ≤ HARMONIC_TOL·k·f0`), i.e. a 2 % deviation from the ideal multiple.
const HARMONIC_TOL: f64 = 0.02;

/// The harmonic index `k` of a line at `freq` relative to fundamental `f0`, or
/// `None` when the line is not a bounded (`1 ≤ k ≤ `[`HARMONIC_MAX_INDEX`]) integer
/// multiple within [`HARMONIC_TOL`]. Shared by [`harmonic_stack`] and the notch
/// router so both agree on exactly which lines belong to a family.
pub(crate) fn harmonic_index_of(freq: f64, f0: f64) -> Option<usize> {
    if !f0.is_finite() || f0 <= 0.0 || !freq.is_finite() || freq <= 0.0
    {
        return None;
    }
    let kf = (freq / f0).round();
    if kf < 1.0 || kf > HARMONIC_MAX_INDEX as f64
    {
        return None;
    }
    if (freq - kf * f0).abs() <= HARMONIC_TOL * kf * f0
    {
        Some(kf as usize)
    }
    else
    {
        None
    }
}

/// The highest harmonic index of `f0` matched by any line — the number of harmonics
/// [`super::remove_mains_hum_iir`] must notch to cover the whole detected family
/// (`n_harmonics = k_max`, *not* the line count: a missing-fundamental {100, 150}
/// stack needs 3 harmonics of `f0 = 50`, though only 2 lines were seen).
pub(crate) fn harmonic_span(lines: &[SpectralLine], f0: f64) -> usize {
    lines
        .iter()
        .filter_map(|l| harmonic_index_of(l.freq_hz, f0))
        .max()
        .unwrap_or(1)
}

/// Recognize a **harmonically related subset** of detected lines — the fingerprint
/// of mains hum and of any nonlinearly distorted interferer, whose energy sits at
/// integer multiples of one fundamental.
///
/// Every line's frequency is tried as a candidate fundamental `f0`, and so is its
/// half (`f0/2` recovers a stack whose fundamental was notched out or buried — the
/// "missing fundamental": lines at 100 and 150 Hz share `f0 = 50`). For each
/// candidate the lines matching an integer multiple within [`HARMONIC_TOL`] are
/// grouped by their harmonic index ([`harmonic_index_of`]); the returned count is
/// the number of **distinct** harmonic indices hit.
///
/// A candidate is only accepted when it clears three guards that keep unrelated
/// tones from forming phantom families:
///
/// * **at least two distinct harmonic indices** — several leakage-skirt lines all
///   at `k = 1` around one peak are one line, not a stack;
/// * **a low harmonic present** (`min k ≤ 2`) — a real family includes its
///   fundamental or its octave, not only high harmonics that any small `f0` can fit;
/// * **bounded `k`** ([`harmonic_index_of`]) — so a far-off line cannot be declared
///   the 39th harmonic of a signal remnant.
///
/// Among the survivors the largest distinct-index count wins; ties prefer the
/// **larger** fundamental, so {50, 100, 150} reports `f0 = 50` rather than the
/// equally-consistent sub-harmonic 25. Returns `None` when no candidate relates two
/// or more lines through a low, bounded harmonic family (e.g. unrelated tones at 50
/// and 137 Hz, or leakage skirts around a single peak).
pub fn harmonic_stack(lines: &[SpectralLine]) -> Option<(f64, usize)> {
    let mut best: Option<(f64, usize)> = None;
    for line in lines
    {
        for f0 in [line.freq_hz, 0.5 * line.freq_hz]
        {
            if !f0.is_finite() || f0 <= 0.0
            {
                continue;
            }
            // Distinct harmonic indices this candidate explains.
            let mut indices: Vec<usize> = lines
                .iter()
                .filter_map(|l| harmonic_index_of(l.freq_hz, f0))
                .collect();
            indices.sort_unstable();
            indices.dedup();
            let count = indices.len();
            let has_low = indices.first().is_some_and(|&k| k <= 2);
            if count >= 2
                && has_low
                && best.is_none_or(|(bf, bc)| count > bc || (count == bc && f0 > bf))
            {
                best = Some((f0, count));
            }
        }
    }
    best
}

/// The dominant tone of the signal's **information** component — the strongest
/// spectral peak of the moving-average-smoothed signal (the part this module treats
/// as structure, not noise), returned only when it is unambiguously dominant: its
/// Fisher-g fraction (peak bin power over the summed positive-frequency band) must
/// exceed 0.3. That fraction stays below ~0.15 for broadband smoothed noise across
/// record lengths but reaches 0.5–0.6 for a tone at or above 0 dB, so the test is
/// scale-robust where a raw peak/median prominence is not. The
/// periodic-interference router uses this to refuse to notch a line that *is* the
/// signal's own tone rather than an interferer. Returns `None` when the smooth part
/// has no clearly dominant line (a broadband information signal needs no protection).
pub(crate) fn signal_dominant_freq(signal: &[f64], sample_rate: f64) -> Option<f64> {
    let n = signal.len();
    if n < 8
    {
        return None;
    }
    let smooth = moving_average(signal, 9);
    let mean = smooth.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = smooth.iter().map(|&x| x - mean).collect();
    let padded = pad_reflect_pow2(&centered);
    let np = padded.len();
    let psd: Vec<f64> = fft_real(&padded).iter().map(|c| c.mag_sq()).collect();
    let (bin, _) = dominant_line(&psd, 2);
    if bin == 0 || bin >= psd.len()
    {
        return None;
    }
    let band_sum: f64 = psd[2..].iter().sum();
    if band_sum <= 0.0 || psd[bin] / band_sum < 0.3
    {
        return None;
    }
    Some(bin as f64 * sample_rate / np as f64)
}

/// Separate a signal into information and noise, then falsify the split with a
/// residual whiteness test. See the module docs for the reasoning.
pub fn separate(signal: &[f64], sample_rate: f64) -> Separation {
    let auto = super::denoise_auto(signal, sample_rate);
    let signal_estimate = auto.output;
    let noise_estimate: Vec<f64> = signal
        .iter()
        .zip(signal_estimate.iter())
        .map(|(s, e)| s - e)
        .collect();

    let noise_std = std_of(&noise_estimate);
    let sig_pow = mean_square(&signal_estimate);
    let noise_pow = mean_square(&noise_estimate).max(1.0e-30);
    let snr_db = 10.0 * (sig_pow / noise_pow).max(1.0e-30).log10();

    let residual_whiteness = whiteness_of(&noise_estimate);
    let leaked_structure = residual_whiteness < 0.9;

    Separation {
        signal_estimate,
        noise_estimate,
        method: auto.method,
        noise_std,
        snr_db,
        residual_whiteness,
        leaked_structure,
        profile: auto.profile,
    }
}

// ---------------------------------------------------------------------------
// Descriptor helpers.
// ---------------------------------------------------------------------------

/// Robust **edge score** — the step-vs-drift discriminator behind the Baseline gate.
///
/// The score is `max|Δ| / (1.4826·MAD(Δ))` where `Δ` is the first difference of the
/// median(5)-prefiltered signal: the median filter suppresses broadband noise while
/// keeping a step's jump concentrated in one or two samples, so an edge shows up as
/// a *single* huge derivative outlier that a max-statistic sees undiluted (a
/// kurtosis-based test dilutes a lone spike by `1/n` and goes blind below ~30 dB
/// SNR — measured, it missed the reference step at 10–20 dB while this score reads
/// 17–73 there). A genuine wander differentiates into another smooth waveform:
/// drift-dominated records measure 2–9 across the calibration fixtures.
///
/// The score is only meaningful **within the high-`trend_strength` population** the
/// Baseline gate examines: on pure broadband noise the median plateaus shrink the
/// MAD and the score idles near 14, but such records never reach the gate
/// (`trend_strength ≈ 0`). The gate threshold of 13 sits between the drift
/// population (≤ 9.4) and the step population (≥ 17.5).
fn step_edge_score(signal: &[f64]) -> f64 {
    if signal.len() < 8
    {
        return 0.0;
    }
    let smoothed = super::rank::median_filter(signal, 2);
    let diff: Vec<f64> = smoothed.windows(2).map(|w| w[1] - w[0]).collect();
    let dm = super::median(&diff);
    let scale = 1.4826 * super::mad(&diff);
    if scale <= 0.0
    {
        return 0.0;
    }
    diff.iter().map(|&d| (d - dm).abs()).fold(0.0, f64::max) / scale
}

/// Whether a component is dominated by a narrowband (tonal) line, judged by very low
/// spectral flatness of its own mean-removed periodogram. Used by the cascade to
/// tell a broadband stage that removed *noise* (flat, colored or white → accept)
/// from one that removed a *signal tone* (a spike in the spectrum → roll back).
pub(crate) fn is_tonal(x: &[f64], _sample_rate: f64) -> bool {
    if x.len() < 8
    {
        return false;
    }
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let centered: Vec<f64> = x.iter().map(|&v| v - mean).collect();
    let padded = pad_reflect_pow2(&centered);
    let psd: Vec<f64> = fft_real(&padded).iter().map(|c| c.mag_sq()).collect();
    if psd.len() < 2
    {
        return false;
    }
    spectral_flatness_of(&psd[1..]) < 0.06
}

fn spectral_flatness_of(psd: &[f64]) -> f64 {
    if psd.is_empty()
    {
        return 1.0;
    }
    let mut log_sum = 0.0;
    let mut arith = 0.0;
    let mut count = 0.0;
    for &p in psd
    {
        let pp = p.max(1.0e-30);
        log_sum += pp.ln();
        arith += pp;
        count += 1.0;
    }
    let geo = (log_sum / count).exp();
    let arith = arith / count;
    if arith <= 0.0
    {
        1.0
    }
    else
    {
        (geo / arith).clamp(0.0, 1.0)
    }
}

fn loglog_slope(psd: &[f64]) -> f64 {
    // Least-squares slope of ln(PSD) against ln(bin) over positive bins.
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut m = 0.0;
    for (k, &p) in psd.iter().enumerate().skip(1)
    {
        if p <= 0.0
        {
            continue;
        }
        let x = (k as f64).ln();
        let y = p.ln();
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
        m += 1.0;
    }
    if m < 2.0
    {
        return 0.0;
    }
    let denom = m * sxx - sx * sx;
    if denom.abs() < 1.0e-30
    {
        return 0.0;
    }
    (m * sxy - sx * sy) / denom
}

fn dominant_line(psd: &[f64], low_cut: usize) -> (usize, f64) {
    let start = low_cut.min(psd.len().saturating_sub(1)).max(1);
    if start >= psd.len()
    {
        return (0, 0.0);
    }
    let mut max_val = 0.0;
    let mut max_bin = start;
    for (k, &p) in psd.iter().enumerate().skip(start)
    {
        if p > max_val
        {
            max_val = p;
            max_bin = k;
        }
    }
    let med = super::median(&psd[start..]);
    let prominence = if med > 0.0 { max_val / med } else { 0.0 };
    (max_bin, prominence)
}

/// Whiteness of a residual in [0, 1]: the fraction of autocorrelation lags
/// (up to `min(n/4, 40)`) that stay inside the `±1.96/√N` white-noise confidence
/// band. 1.0 ⇒ indistinguishable from white noise. This is the single shared
/// implementation behind [`separate`]'s self-check, the cascade's stage-progress
/// criterion and the tournament score of [`super::denoise_best`].
pub(crate) fn whiteness_of(residual: &[f64]) -> f64 {
    let n = residual.len();
    let max_lag = (n / 4).clamp(1, 40);
    let ac = normalized_autocorr(residual, max_lag);
    let band = 1.96 / (n.max(1) as f64).sqrt();
    let within = (1..=max_lag).filter(|&l| ac[l].abs() < band).count();
    within as f64 / max_lag as f64
}

fn normalized_autocorr(x: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0; max_lag + 1];
    if n == 0
    {
        return out;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let c0: f64 = x.iter().map(|&v| (v - mean) * (v - mean)).sum();
    out[0] = 1.0;
    if c0 <= 0.0
    {
        return out;
    }
    for lag in 1..=max_lag.min(n.saturating_sub(1))
    {
        let mut s = 0.0;
        for i in 0..(n - lag)
        {
            s += (x[i] - mean) * (x[i + lag] - mean);
        }
        out[lag] = s / c0;
    }
    out
}

/// Population standard deviation of a slice (0.0 when empty). Shared with the
/// tournament scorer in the parent module.
pub(crate) fn std_of(x: &[f64]) -> f64 {
    let n = x.len();
    if n == 0
    {
        return 0.0;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    (x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n as f64).sqrt()
}

fn mean_square(x: &[f64]) -> f64 {
    if x.is_empty()
    {
        return 0.0;
    }
    x.iter().map(|&v| v * v).sum::<f64>() / x.len() as f64
}

#[cfg(test)]
mod tests {
    use super::super::testutil::Lcg;
    use super::*;
    use core::f64::consts::PI;

    fn base_sine(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| (2.0 * PI * 4.0 * i as f64 / n as f64).sin())
            .collect()
    }

    #[test]
    fn detects_gaussian_noise() {
        let mut rng = Lcg::new(101);
        let obs: Vec<f64> = base_sine(512)
            .iter()
            .map(|&c| c + 0.3 * rng.gauss())
            .collect();
        let p = classify(&obs, 512.0);
        assert!(matches!(
            p.dominant,
            NoiseType::Gaussian | NoiseType::Colored
        ));
        assert!(
            p.noise_std > 0.1 && p.noise_std < 0.6,
            "std {}",
            p.noise_std
        );
    }

    #[test]
    fn detects_impulsive_noise() {
        // Genuine impulsive noise is *aperiodic* (salt-and-pepper, drop-outs,
        // electrode pops occur at random times) — placed here by a Bernoulli draw so
        // the impulse-energy envelope has no periodic autocorrelation and the verdict
        // is Impulsive (cf. `periodic_spike_train_is_a_legitimate_feature`, where a
        // *periodic* train is instead recognized as signal).
        let mut rng = Lcg::new(103);
        let mut obs: Vec<f64> = base_sine(512)
            .iter()
            .map(|&c| c + 0.05 * rng.gauss())
            .collect();
        let mut spikes = 0;
        for v in obs.iter_mut()
        {
            if rng.uniform() < 1.0 / 37.0
            {
                *v += 8.0;
                spikes += 1;
            }
        }
        assert!(spikes >= 6, "fixture drew too few spikes: {spikes}");
        let p = classify(&obs, 512.0);
        assert_eq!(p.dominant, NoiseType::Impulsive);
    }

    #[test]
    fn periodic_spike_train_is_a_legitimate_feature() {
        // A *periodic* high-crest train — an ECG's QRS complexes, an engine's knock,
        // any repeated transient — is signal, not impulsive noise: the impulsive gate
        // must be vetoed by the energy-envelope periodicity test so the feature is not
        // filtered away. Same spikes as the aperiodic fixture above, but on a regular
        // 37-sample grid.
        let mut rng = Lcg::new(103);
        let mut obs: Vec<f64> = base_sine(512)
            .iter()
            .map(|&c| c + 0.05 * rng.gauss())
            .collect();
        for (i, v) in obs.iter_mut().enumerate()
        {
            if i % 37 == 0
            {
                *v += 8.0;
            }
        }
        let p = classify(&obs, 512.0);
        assert_ne!(
            p.dominant,
            NoiseType::Impulsive,
            "a periodic spike train must not be filtered as impulsive noise"
        );
    }

    #[test]
    fn detects_periodic_interference() {
        let n = 1024;
        let fs = 1000.0;
        let clean = base_sine(n);
        let obs: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, &c)| c + 1.0 * (2.0 * PI * 60.0 * i as f64 / fs).sin())
            .collect();
        let p = classify(&obs, fs);
        assert_eq!(p.dominant, NoiseType::Periodic);
        assert!(
            (p.dominant_freq_hz - 60.0).abs() < 3.0,
            "freq {}",
            p.dominant_freq_hz
        );
    }

    #[test]
    fn detects_baseline_drift() {
        let n = 512;
        let clean = base_sine(n);
        // A slow one-cycle wander (bin 1) with large amplitude dominates the low band.
        let obs: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, &c)| c + 6.0 * (2.0 * PI * 1.0 * i as f64 / n as f64).sin())
            .collect();
        let p = classify(&obs, 512.0);
        assert_eq!(p.dominant, NoiseType::Baseline);
    }

    #[test]
    fn detects_baseline_drift_at_equal_power() {
        // The realistic regime the old 0.6 gate missed: a drift carrying the *same*
        // power as the signal (0 dB) measures trend_strength ≈ 0.49 and must now be
        // caught (gate 0.45), while a weak drift (amp 0.5, ts ≈ 0.20) still is not.
        let n = 2048;
        let fs = 1000.0;
        let mut rng = Lcg::new(5);
        let obs: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 5.0 * t).sin() + 1.0 * (2.0 * PI * 0.7 * t).sin() + 0.1 * rng.gauss()
            })
            .collect();
        assert_eq!(classify(&obs, fs).dominant, NoiseType::Baseline);
    }

    #[test]
    fn colored_noise_stays_out_of_the_baseline_gate() {
        // AR(0.9) is the strongest "legitimately low-frequency" noise the module
        // targets; its trend_strength (~0.05) must stay far below the 0.45 gate.
        let n = 2048;
        let fs = 1000.0;
        let mut rng = Lcg::new(9);
        let mut w = 0.0;
        let obs: Vec<f64> = (0..n)
            .map(|i| {
                w = 0.9 * w + 0.3 * rng.gauss();
                (2.0 * PI * 5.0 * (i as f64 / fs)).sin() + w
            })
            .collect();
        let p = classify(&obs, fs);
        assert!(p.trend_strength < 0.45, "ts = {}", p.trend_strength);
        assert_ne!(p.dominant, NoiseType::Baseline);
    }

    #[test]
    fn step_edges_are_not_misread_as_drift() {
        // A step + ramp record concentrates energy in the low band exactly like a
        // drift (trend_strength ≈ 0.9), but its median-prefiltered derivative has a
        // single huge outlier (edge score 17–550 vs 2–9 for genuine wanders): the
        // edge-score guard must keep it out of the detrending branch — including at
        // MODERATE noise levels (10–20 dB), where a kurtosis-based guard is blind
        // because one spike dilutes by 1/n. Detrending a step record erases it
        // (measured: −19 dB at 20 dB input before the guard).
        let n = 2048;
        for sigma in [0.05, 0.15, 0.45]
        {
            let mut rng = Lcg::new(7);
            let obs: Vec<f64> = (0..n)
                .map(|i| {
                    let ramp = i as f64 / n as f64;
                    (if i < n / 2 { 0.0 } else { 2.0 }) + ramp + sigma * rng.gauss()
                })
                .collect();
            let p = classify(&obs, 1000.0);
            assert_ne!(
                p.dominant,
                NoiseType::Baseline,
                "step/ramp at σ = {sigma} must not be detrended as a baseline wander \
                 (ts = {:.2})",
                p.trend_strength
            );
        }
    }

    #[test]
    fn estimate_noise_std_is_accurate() {
        let mut rng = Lcg::new(107);
        let true_sigma = 0.25;
        let obs: Vec<f64> = base_sine(2048)
            .iter()
            .map(|&c| c + true_sigma * rng.gauss())
            .collect();
        let est = estimate_noise_std(&obs);
        assert!((est - true_sigma).abs() < 0.06, "est {est} vs {true_sigma}");
    }

    #[test]
    fn separation_residual_is_white_for_gaussian() {
        let mut rng = Lcg::new(109);
        let obs: Vec<f64> = base_sine(512)
            .iter()
            .map(|&c| c + 0.3 * rng.gauss())
            .collect();
        let sep = separate(&obs, 512.0);
        assert_eq!(sep.signal_estimate.len(), obs.len());
        assert_eq!(sep.noise_estimate.len(), obs.len());
        // Reconstruction identity: signal + noise == observed.
        for (i, &o) in obs.iter().enumerate()
        {
            assert!((sep.signal_estimate[i] + sep.noise_estimate[i] - o).abs() < 1.0e-9);
        }
        assert!(
            sep.residual_whiteness > 0.7,
            "whiteness {}",
            sep.residual_whiteness
        );
    }

    #[test]
    fn separation_flags_leaked_structure() {
        // If we deliberately under-denoise a very noisy tonal signal, the residual
        // should retain structure. Here a strong 60 Hz tone is the "noise"; the
        // separator notches it, leaving a near-white residual — so instead we test
        // that a clean-ish signal with correlated (colored) noise is caught.
        let mut rng = Lcg::new(111);
        let n = 512;
        // Colored noise: integrate white noise (random walk) → strongly correlated.
        let mut walk = 0.0;
        let obs: Vec<f64> = base_sine(n)
            .iter()
            .map(|&c| {
                walk += 0.15 * rng.gauss();
                c + walk
            })
            .collect();
        let sep = separate(&obs, 512.0);
        // Whiteness is a well-defined number in [0,1] either way.
        assert!(sep.residual_whiteness >= 0.0 && sep.residual_whiteness <= 1.0);
    }

    /// Two tones + a slow signal + a small white floor, sized so the trimmed
    /// residual core (n − 24 samples) is exactly 2048 and both tones sit ON a
    /// periodogram bin (50 Hz = bin 100, 120 Hz = bin 240 at fs = 1024), which
    /// keeps the peeling crisp and the frequency readout exact. The noise floor
    /// matters: without it the tiny high-pass remnant of the 4 Hz *signal* would
    /// stand alone in an otherwise empty spectrum and be reported as a third line.
    fn two_tone_fixture(fs: f64) -> Vec<f64> {
        let n = 2072;
        let mut rng = Lcg::new(211);
        (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                (2.0 * PI * 4.0 * t).sin()
                    + 0.8 * (2.0 * PI * 50.0 * t).sin()
                    + 0.8 * (2.0 * PI * 120.0 * t).sin()
                    + 0.05 * rng.gauss()
            })
            .collect()
    }

    #[test]
    fn detect_lines_finds_two_separated_tones() {
        let fs = 1024.0;
        let obs = two_tone_fixture(fs);
        let lines = detect_lines(&obs, fs, 5);
        assert!(lines.len() >= 2, "found only {} lines", lines.len());
        let bin = fs / 2048.0;
        for target in [50.0, 120.0]
        {
            assert!(
                lines
                    .iter()
                    .any(|l| (l.freq_hz - target).abs() <= 2.0 * bin),
                "no line within 2 bins of {target} Hz: {lines:?}"
            );
        }
        // And nothing but the two injected tones is reported: every line must sit
        // on one of them (the peeler must not invent lines out of the noise floor).
        for l in &lines
        {
            assert!(
                (l.freq_hz - 50.0).abs() <= 2.0 * bin || (l.freq_hz - 120.0).abs() <= 2.0 * bin,
                "spurious line at {} Hz",
                l.freq_hz
            );
        }
        for l in &lines
        {
            assert!(l.fisher_g > 0.0 && l.fisher_g <= 1.0, "g = {}", l.fisher_g);
            assert!(
                l.power_ratio > 0.0 && l.power_ratio <= 1.0,
                "power_ratio = {}",
                l.power_ratio
            );
        }
        // Two isolated tones are NOT a harmonic stack (120 is not a multiple of 50).
        assert!(harmonic_stack(&lines).is_none());
    }

    #[test]
    fn detect_lines_parameters_are_live() {
        let fs = 1024.0;
        let obs = two_tone_fixture(fs);
        // max_lines caps the peeling.
        assert_eq!(detect_lines(&obs, fs, 1).len(), 1);
        assert!(detect_lines(&obs, fs, 0).is_empty());
        // sample_rate scales the reported frequencies: doubling fs doubles them.
        let base = detect_lines(&obs, fs, 1);
        let doubled = detect_lines(&obs, 2.0 * fs, 1);
        assert!(
            (doubled[0].freq_hz - 2.0 * base[0].freq_hz).abs() < 1.0e-9,
            "{} vs {}",
            doubled[0].freq_hz,
            base[0].freq_hz
        );
    }

    #[test]
    fn detect_lines_degrades_gracefully() {
        let empty: [f64; 0] = [];
        assert!(detect_lines(&empty, 1000.0, 5).is_empty());
        for len in 1..8_usize
        {
            let x: Vec<f64> = (0..len).map(|i| i as f64).collect();
            assert!(detect_lines(&x, 1000.0, 5).is_empty(), "len {len}");
        }
        // A constant has a zero residual spectrum: no line, no panic.
        assert!(detect_lines(&vec![3.5; 64], 1000.0, 5).is_empty());
    }

    #[test]
    fn detect_lines_agrees_with_classify_on_a_periodic_verdict() {
        // Whenever classify says Periodic, the peeler (same PSD, same critical
        // value) must find at least the line classify saw, at the same frequency.
        let n = 1024;
        let fs = 1000.0;
        let obs: Vec<f64> = base_sine(n)
            .iter()
            .enumerate()
            .map(|(i, &c)| c + 1.0 * (2.0 * PI * 60.0 * i as f64 / fs).sin())
            .collect();
        let p = classify(&obs, fs);
        assert_eq!(p.dominant, NoiseType::Periodic);
        let lines = detect_lines(&obs, fs, 5);
        assert!(!lines.is_empty());
        assert!(
            (lines[0].freq_hz - p.dominant_freq_hz).abs() < 1.0e-9,
            "peeler {} Hz vs classify {} Hz",
            lines[0].freq_hz,
            p.dominant_freq_hz
        );
    }

    #[test]
    fn harmonic_stack_finds_fundamental_and_rejects_unrelated() {
        let mk = |f: f64| SpectralLine {
            freq_hz: f,
            fisher_g: 0.3,
            power_ratio: 0.2,
        };
        // A full 50/100/150 stack: f0 = 50 with all three lines matched. The
        // sub-harmonic 25 relates the same three lines; the tie must go to the
        // larger fundamental.
        let (f0, count) = harmonic_stack(&[mk(50.0), mk(100.0), mk(150.0)]).unwrap();
        assert!((f0 - 50.0).abs() < 1.0e-9, "f0 = {f0}");
        assert_eq!(count, 3);
        // Order must not matter.
        let (f0, count) = harmonic_stack(&[mk(150.0), mk(50.0), mk(100.0)]).unwrap();
        assert!((f0 - 50.0).abs() < 1.0e-9);
        assert_eq!(count, 3);
        // Missing fundamental: 100 and 150 only relate through the f0/2 candidate.
        let (f0, count) = harmonic_stack(&[mk(100.0), mk(150.0)]).unwrap();
        assert!((f0 - 50.0).abs() < 1.0e-9, "f0 = {f0}");
        assert_eq!(count, 2);
        // Unrelated tones: no fundamental relates 50 and 137 within 2 %.
        assert!(harmonic_stack(&[mk(50.0), mk(137.0)]).is_none());
        // Degenerate inputs.
        assert!(harmonic_stack(&[]).is_none());
        assert!(harmonic_stack(&[mk(50.0)]).is_none());
        // The 2 % tolerance is live: 101 Hz is a near-multiple of 50 (1 % off),
        // 103 Hz is not (3 % off).
        assert!(harmonic_stack(&[mk(50.0), mk(101.0)]).is_some());
        assert!(harmonic_stack(&[mk(50.0), mk(103.0)]).is_none());
    }

    #[test]
    fn harmonic_stack_rejects_spurious_families() {
        let mk = |f: f64| SpectralLine {
            freq_hz: f,
            fisher_g: 0.3,
            power_ratio: 0.2,
        };
        // A low signal remnant (7 Hz) and an unrelated interferer (137 Hz): the old
        // unbounded-k matcher paired them at f0 = 3.5 (7 = 2·3.5, 137 ≈ 39·3.5) and
        // let the router notch the 7 Hz signal. The k ≤ 12 cap forbids k = 39.
        assert!(harmonic_stack(&[mk(137.0), mk(50.0), mk(7.0)]).is_none());
        // Two unrelated tones cannot be "related" through a tiny fundamental.
        assert!(harmonic_stack(&[mk(2.0), mk(97.0), mk(151.0)]).is_none());
        // Leakage skirts around one peak (all at k = 1) are one line, not a stack.
        assert!(harmonic_stack(&[mk(119.1), mk(120.6), mk(122.1), mk(123.5)]).is_none());
        // A genuine odd-harmonic family (symmetric distortion) is still recognized.
        let (f0, count) = harmonic_stack(&[mk(50.0), mk(150.0), mk(250.0)]).unwrap();
        assert!((f0 - 50.0).abs() < 1.0e-9, "f0 = {f0}");
        assert_eq!(count, 3);
    }

    #[test]
    fn harmonic_span_covers_every_detected_line() {
        let mk = |f: f64| SpectralLine {
            freq_hz: f,
            fisher_g: 0.3,
            power_ratio: 0.2,
        };
        // Missing-fundamental {100, 150}: the stack count is 2 but three harmonics of
        // f0 = 50 must be notched to reach the 150 Hz line.
        let lines = [mk(100.0), mk(150.0)];
        let (f0, _) = harmonic_stack(&lines).unwrap();
        assert_eq!(harmonic_span(&lines, f0), 3);
    }

    #[test]
    fn signal_dominant_freq_finds_a_tone_and_ignores_broadband() {
        let n = 1024;
        let fs = 1000.0;
        let tone: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 8.0 * i as f64 / fs).sin())
            .collect();
        let f = signal_dominant_freq(&tone, fs).expect("a clear tone must be found");
        assert!((f - 8.0).abs() < 3.0, "dominant {f} Hz");
        // Broadband white noise has no dominant information tone to protect.
        let mut rng = Lcg::new(202);
        let noise: Vec<f64> = (0..n).map(|_| rng.gauss()).collect();
        assert!(signal_dominant_freq(&noise, fs).is_none());
    }
}
