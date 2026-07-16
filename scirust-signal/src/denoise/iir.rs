//! Zero-phase IIR notch filtering — RBJ biquad design plus forward-backward application.
//!
//! [`super::transform::notch_filter`] removes a tonal interferer by zeroing FFT bins — a
//! brick-wall filter. That works when the tone sits exactly on a bin, but a real-world hum on an
//! arbitrary-length record almost never does: an off-bin tone leaks energy across the whole
//! spectrum, so zeroing a couple of bins leaves a sizeable residual, and the hard spectral edge
//! rings in the time domain (Gibbs). A recursive (IIR) notch has no frequency grid: its zero pair
//! is placed *exactly* at the requested frequency regardless of record length, and its smooth
//! magnitude response does not ring.
//!
//! A causal IIR filter, however, delays and phase-shifts the signal. The classic cure is
//! forward-backward filtering ([`filtfilt_sos`], MATLAB/scipy `filtfilt`): run the filter once
//! forward and once backward so the phase distortion cancels exactly and the magnitude response
//! is squared (a −40 dB notch becomes −80 dB). Two implementation details make this work on
//! finite records: odd (anti-symmetric) reflection padding at both ends, and starting each pass
//! from the filter's step-response steady state scaled to the first padded sample (scipy's
//! `lfilter_zi` trick). The latter is what keeps the startup transient of a *high-Q* notch —
//! whose impulse response is far longer than any reasonable padding — out of the output.
//!
//! Use [`notch_iir`] / [`remove_mains_hum_iir`] as drop-in, higher-precision replacements for
//! [`super::transform::notch_filter`] / [`super::transform::remove_mains_hum`] whenever the
//! interferer frequency is known but need not align with an FFT bin. For real-time processing,
//! where the backward pass is impossible, [`BiquadState`] provides the causal streaming form (at
//! the price of the filter's group delay).

use crate::filter::Biquad;
use core::f64::consts::PI;

/// The pass-through section returned when a design request is out of its validity domain.
const IDENTITY: Biquad = Biquad {
    b0: 1.0,
    b1: 0.0,
    b2: 0.0,
    a1: 0.0,
    a2: 0.0,
};

/// Design a notch biquad with the RBJ recipe (Robert Bristow-Johnson, *Cookbook formulae for
/// audio EQ biquad filter coefficients*): `w0 = 2π·center_hz/sample_rate`,
/// `alpha = sin(w0)/(2q)`, numerator `[1, −2cos(w0), 1]`, denominator
/// `[1+alpha, −2cos(w0), 1−alpha]`, normalized so `a0 = 1`.
///
/// The zeros sit exactly on the unit circle at `±w0` — the gain at `center_hz` is exactly zero —
/// and the poles sit just inside at the same angle, so `q = center_hz/bandwidth` yields a −3 dB
/// notch width of `bandwidth` Hz. The design is valid for `0 < center_hz < sample_rate/2` and
/// `q > 0` (all arguments finite); outside that domain the function degrades gracefully to the
/// identity (pass-through) biquad instead of returning an unstable or degenerate section.
pub fn rbj_notch(sample_rate: f64, center_hz: f64, q: f64) -> Biquad {
    let valid = sample_rate.is_finite()
        && center_hz.is_finite()
        && q.is_finite()
        && sample_rate > 0.0
        && center_hz > 0.0
        && center_hz < 0.5 * sample_rate
        && q > 0.0;
    if !valid
    {
        return IDENTITY;
    }
    let w0 = 2.0 * PI * center_hz / sample_rate;
    let alpha = w0.sin() / (2.0 * q);
    let cw = w0.cos();
    let a0 = 1.0 + alpha;
    Biquad {
        b0: 1.0 / a0,
        b1: -2.0 * cw / a0,
        b2: 1.0 / a0,
        a1: -2.0 * cw / a0,
        a2: (1.0 - alpha) / a0,
    }
}

/// Steady-state Direct Form II Transposed state of a section driven by a unit step — the closed
/// form of `scipy.signal.lfilter_zi` for one a0-normalized biquad. Scaling it by the first input
/// sample starts the filter "as if the input had always been there", which suppresses the startup
/// transient that would otherwise leak a high-Q notch's long impulse response into the output.
fn lfilter_zi(sec: &Biquad) -> (f64, f64) {
    let den = 1.0 + sec.a1 + sec.a2;
    if !den.is_finite() || den.abs() < 1.0e-12
    {
        // A pole pinned at z = 1 has no step steady state; fall back to rest.
        return (0.0, 0.0);
    }
    let h1 = (sec.b0 + sec.b1 + sec.b2) / den;
    let z2 = sec.b2 - sec.a2 * h1;
    let z1 = sec.b1 - sec.a1 * h1 + z2;
    (z1, z2)
}

/// One causal DF2T pass of a single section over `x`, with the internal state initialized to the
/// step steady state ([`lfilter_zi`]) scaled by the first sample.
fn df2t_steady(sec: &Biquad, x: &[f64]) -> Vec<f64> {
    let (z1, z2) = lfilter_zi(sec);
    let x0 = x.first().copied().unwrap_or(0.0);
    let mut s1 = z1 * x0;
    let mut s2 = z2 * x0;
    let mut y = Vec::with_capacity(x.len());
    for &xi in x
    {
        let yi = sec.b0 * xi + s1;
        s1 = sec.b1 * xi + s2 - sec.a1 * yi;
        s2 = sec.b2 * xi - sec.a2 * yi;
        y.push(yi);
    }
    y
}

/// Zero-phase forward-backward filtering of a second-order-section cascade — the semantics of
/// `scipy.signal.sosfiltfilt` (MATLAB `filtfilt`).
///
/// The signal is extended on both ends by an odd (anti-symmetric, point-reflected) extension of
/// length `padlen = min(n − 1, max(24, 9·sections.len()))`, which continues the signal without a
/// jump in value or slope. The cascade is applied forward, the result is reversed, filtered
/// again, reversed back, and the padding cropped, so every frequency component returns with zero
/// phase shift and the squared magnitude of the cascade's response. Each pass of each section
/// starts from its step-response steady state scaled by the first sample of its input (the
/// `lfilter_zi` initialization) — for sharp notches this matters far more than the padding, whose
/// reasonable lengths are much shorter than a high-Q impulse response.
///
/// Degrades gracefully: an empty cascade or a signal shorter than two samples is returned
/// unchanged. The output always has the length of the input.
pub fn filtfilt_sos(sections: &[Biquad], signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n < 2 || sections.is_empty()
    {
        return signal.to_vec();
    }
    let padlen = (n - 1).min(24usize.max(9 * sections.len()));

    // Odd extension: reflect through the end points so value and slope stay continuous.
    let mut ext = Vec::with_capacity(n + 2 * padlen);
    for j in (1..=padlen).rev()
    {
        ext.push(2.0 * signal[0] - signal[j]);
    }
    ext.extend_from_slice(signal);
    for j in 1..=padlen
    {
        ext.push(2.0 * signal[n - 1] - signal[n - 1 - j]);
    }

    for sec in sections
    {
        ext = df2t_steady(sec, &ext);
    }
    ext.reverse();
    for sec in sections
    {
        ext = df2t_steady(sec, &ext);
    }
    ext.reverse();
    ext[padlen..padlen + n].to_vec()
}

/// Zero-phase IIR notch: [`rbj_notch`] designed with `q = center_hz/bandwidth`, applied through
/// [`filtfilt_sos`]. The precise replacement for the brick-wall
/// [`super::transform::notch_filter`] — same `(center_hz, bandwidth)` parameter convention, so
/// callers can swap one for the other — but with the null placed exactly at `center_hz` even when
/// that frequency falls between FFT bins, and with no Gibbs ringing.
///
/// Degrades gracefully to a copy of the input on degenerate signals (length < 2) and on invalid
/// parameters (`bandwidth ≤ 0`, or a `(sample_rate, center_hz)` pair outside the [`rbj_notch`]
/// validity domain).
pub fn notch_iir(signal: &[f64], sample_rate: f64, center_hz: f64, bandwidth: f64) -> Vec<f64> {
    if signal.len() < 2 || !bandwidth.is_finite() || bandwidth <= 0.0
    {
        return signal.to_vec();
    }
    let sec = rbj_notch(sample_rate, center_hz, center_hz / bandwidth);
    if sec == IDENTITY
    {
        return signal.to_vec();
    }
    filtfilt_sos(&[sec], signal)
}

/// Remove mains hum and its harmonics with a single zero-phase pass over a cascade of RBJ
/// notches at `h·mains_hz` for `h = 1..=n_harmonics` — the IIR counterpart of
/// [`super::transform::remove_mains_hum`], same parameter convention.
///
/// Each harmonic keeps the same *absolute* −3 dB width of `bandwidth` Hz, i.e. a per-harmonic
/// quality factor `q = (h·mains_hz)/bandwidth`. Harmonics at or above 99 % of the Nyquist
/// frequency are skipped: that close to the band edge the bilinear-warped notch degenerates, and
/// any aliased hum folds back below Nyquist anyway. Degrades gracefully to a copy of the input
/// when no valid notch remains (degenerate signal, `n_harmonics = 0`, or invalid
/// `mains_hz`/`bandwidth`/`sample_rate`).
pub fn remove_mains_hum_iir(
    signal: &[f64],
    sample_rate: f64,
    mains_hz: f64,
    n_harmonics: usize,
    bandwidth: f64,
) -> Vec<f64> {
    let valid = sample_rate.is_finite()
        && mains_hz.is_finite()
        && bandwidth.is_finite()
        && sample_rate > 0.0
        && mains_hz > 0.0
        && bandwidth > 0.0;
    if signal.len() < 2 || !valid
    {
        return signal.to_vec();
    }
    let nyquist = 0.5 * sample_rate;
    let mut cascade = Vec::new();
    for h in 1..=n_harmonics
    {
        let center = mains_hz * h as f64;
        if center >= 0.99 * nyquist
        {
            break;
        }
        let sec = rbj_notch(sample_rate, center, center / bandwidth);
        if sec != IDENTITY
        {
            cascade.push(sec);
        }
    }
    if cascade.is_empty()
    {
        return signal.to_vec();
    }
    filtfilt_sos(&cascade, signal)
}

/// A biquad in causal streaming form: Direct Form II Transposed state fed one sample at a time.
///
/// Unlike [`filtfilt_sos`] this is *causal* — it can run in real time on a live stream, but the
/// section's full group delay and phase distortion apply (a notch smears its surroundings in
/// time). Feeding every sample of a slice through [`BiquadState::push`] reproduces
/// [`Biquad::filter`] exactly, sample for sample.
#[derive(Debug, Clone)]
pub struct BiquadState {
    /// The section's coefficients (`a0`-normalized).
    coeffs: Biquad,
    /// First DF2T delay-line register.
    s1: f64,
    /// Second DF2T delay-line register.
    s2: f64,
}

impl BiquadState {
    /// Wrap a section with zeroed (rest) state — the same initial condition as
    /// [`Biquad::filter`].
    pub fn new(coeffs: Biquad) -> Self {
        Self {
            coeffs,
            s1: 0.0,
            s2: 0.0,
        }
    }

    /// Process one input sample and return the corresponding output sample (DF2T recursion:
    /// `y = b0·x + s1`, `s1 ← b1·x + s2 − a1·y`, `s2 ← b2·x − a2·y`).
    pub fn push(&mut self, x: f64) -> f64 {
        let y = self.coeffs.b0 * x + self.s1;
        self.s1 = self.coeffs.b1 * x + self.s2 - self.coeffs.a1 * y;
        self.s2 = self.coeffs.b2 * x - self.coeffs.a2 * y;
        y
    }

    /// Clear the delay line back to the rest state [`BiquadState::new`] started from.
    pub fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::snr_db;
    use super::super::transform;
    use super::*;
    use core::f64::consts::FRAC_1_SQRT_2;

    fn tone(n: usize, fs: f64, freq: f64, amp: f64) -> Vec<f64> {
        (0..n)
            .map(|i| amp * (2.0 * PI * freq * i as f64 / fs).sin())
            .collect()
    }

    fn rms(x: &[f64]) -> f64 {
        (x.iter().map(|&v| v * v).sum::<f64>() / x.len().max(1) as f64).sqrt()
    }

    /// Magnitude of a biquad's frequency response at `freq` Hz — evaluates
    /// `|H(e^{jw})|` directly from the coefficients, independent of any filtering code.
    fn biquad_gain(sec: &Biquad, fs: f64, freq: f64) -> f64 {
        let w = 2.0 * PI * freq / fs;
        let (c1, s1) = (w.cos(), w.sin());
        let (c2, s2) = ((2.0 * w).cos(), (2.0 * w).sin());
        let nr = sec.b0 + sec.b1 * c1 + sec.b2 * c2;
        let ni = -(sec.b1 * s1 + sec.b2 * s2);
        let dr = 1.0 + sec.a1 * c1 + sec.a2 * c2;
        let di = -(sec.a1 * s1 + sec.a2 * s2);
        ((nr * nr + ni * ni) / (dr * dr + di * di)).sqrt()
    }

    #[test]
    fn rbj_notch_response_matches_design() {
        // q = 25 at 50 Hz ⇒ −3 dB width of 2 Hz. This pins center AND q: a swapped or
        // ignored parameter moves the null or the band edges.
        let fs = 1000.0;
        let sec = rbj_notch(fs, 50.0, 25.0);
        assert!(biquad_gain(&sec, fs, 50.0) < 1.0e-9, "null at center");
        assert!(
            (biquad_gain(&sec, fs, 0.0) - 1.0).abs() < 1.0e-9,
            "unity DC"
        );
        assert!((biquad_gain(&sec, fs, 25.0) - 1.0).abs() < 0.01);
        assert!((biquad_gain(&sec, fs, 100.0) - 1.0).abs() < 0.01);
        for edge in [49.0, 51.0]
        {
            let g = biquad_gain(&sec, fs, edge);
            assert!(
                (g - FRAC_1_SQRT_2).abs() < 0.05,
                "-3 dB at band edge {edge} Hz, got {g}"
            );
        }
        // q is live: a low-q (wide) notch attenuates 51 Hz much harder.
        let wide = rbj_notch(fs, 50.0, 2.0);
        assert!(biquad_gain(&wide, fs, 51.0) < 0.2);
    }

    #[test]
    fn notch_iir_kills_center_and_passes_octave_neighbors() {
        let fs = 1000.0;
        let n = 4000;
        let core = 500..3500;
        // > 40 dB steady-state attenuation at the notch center.
        let at_center = tone(n, fs, 50.0, 1.0);
        let out = notch_iir(&at_center, fs, 50.0, 2.0);
        let att =
            20.0 * (rms(&at_center[core.clone()]) / rms(&out[core.clone()]).max(1.0e-300)).log10();
        assert!(att > 40.0, "attenuation {att} dB");
        // Within 1 % of unity gain one octave away on either side.
        for freq in [25.0, 100.0]
        {
            let x = tone(n, fs, freq, 1.0);
            let y = notch_iir(&x, fs, 50.0, 2.0);
            let ratio = rms(&y[core.clone()]) / rms(&x[core.clone()]);
            assert!((ratio - 1.0).abs() < 0.01, "gain {ratio} at {freq} Hz");
        }
    }

    #[test]
    fn far_tone_returns_with_zero_lag_and_unit_amplitude() {
        let fs = 1000.0;
        let n = 2000;
        let x = tone(n, fs, 10.0, 1.0);
        let y = notch_iir(&x, fs, 50.0, 2.0);
        // Zero phase: the cross-correlation against the input peaks at lag 0.
        let mut best_lag = isize::MIN;
        let mut best = f64::MIN;
        for lag in -25_isize..=25
        {
            let mut c = 0.0;
            for (i, &xi) in x.iter().enumerate()
            {
                let j = i as isize + lag;
                if j >= 0 && (j as usize) < n
                {
                    c += xi * y[j as usize];
                }
            }
            if c > best
            {
                best = c;
                best_lag = lag;
            }
        }
        assert_eq!(best_lag, 0, "output is delayed by {best_lag} samples");
        // Unit amplitude away from the edges.
        let ratio = rms(&y[200..1800]) / rms(&x[200..1800]);
        assert!((ratio - 1.0).abs() < 0.01, "amplitude ratio {ratio}");
    }

    #[test]
    fn off_bin_interferer_iir_beats_fft_bin_zeroing() {
        // n = 500 is NOT a power of two, and 50.3 Hz falls between the padded FFT's
        // bins (fs/512 ≈ 1.95 Hz grid), so the brick-wall notch leaks; the IIR null
        // sits exactly on 50.3 Hz no matter the record length.
        let n = 500;
        let fs = 1000.0;
        let clean = tone(n, fs, 5.0, 1.0);
        let obs: Vec<f64> = clean
            .iter()
            .enumerate()
            .map(|(i, &c)| c + (2.0 * PI * 50.3 * i as f64 / fs).sin())
            .collect();
        let iir = notch_iir(&obs, fs, 50.3, 4.0);
        let fft = transform::notch_filter(&obs, fs, 50.3, 4.0);
        let s_raw = snr_db(&clean, &obs);
        let s_iir = snr_db(&clean, &iir);
        let s_fft = snr_db(&clean, &fft);
        assert!(s_iir > s_raw, "iir {s_iir} dB must beat raw {s_raw} dB");
        assert!(s_iir > s_fft, "iir {s_iir} dB must beat fft {s_fft} dB");
    }

    #[test]
    fn zi_initialization_suppresses_edge_transients() {
        // A pure passband tone through a high-Q notch: the first samples must already
        // match the clean tone. Without the steady-state (lfilter_zi) initialization
        // the notch's ~160-sample transient dwarfs the 24-sample padding.
        let fs = 1000.0;
        let clean = tone(1000, fs, 10.0, 1.0);
        let out = notch_iir(&clean, fs, 50.0, 2.0);
        for (i, (&o, &c)) in out.iter().zip(clean.iter()).take(20).enumerate()
        {
            assert!(
                (o - c).abs() < 0.05,
                "sample {i}: {o} deviates from {c} by more than 5 %"
            );
        }
    }

    #[test]
    fn mains_hum_and_harmonics_removed_iir() {
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
                c + 0.6 * (2.0 * PI * 50.0 * t).sin()
                    + 0.3 * (2.0 * PI * 100.0 * t).sin()
                    + 0.2 * (2.0 * PI * 150.0 * t).sin()
            })
            .collect();
        let out = remove_mains_hum_iir(&obs, fs, 50.0, 3, 3.0);
        assert_eq!(out.len(), n);
        assert!(snr_db(&clean, &out) > snr_db(&clean, &obs) + 8.0);
    }

    #[test]
    fn notch_iir_bandwidth_is_live() {
        // A 53 Hz tone survives a 2 Hz-wide notch at 50 Hz but dies in a 20 Hz one —
        // ignoring `bandwidth` (or transposing it with `center_hz`) breaks this.
        let fs = 1000.0;
        let x = tone(2000, fs, 53.0, 1.0);
        let narrow = notch_iir(&x, fs, 50.0, 2.0);
        let wide = notch_iir(&x, fs, 50.0, 20.0);
        let r_narrow = rms(&narrow[300..1700]);
        let r_wide = rms(&wide[300..1700]);
        assert!(
            r_narrow > 5.0 * r_wide,
            "narrow {r_narrow} vs wide {r_wide}"
        );
    }

    #[test]
    fn mains_hum_harmonic_count_is_live() {
        let fs = 1000.0;
        let hum100 = tone(2000, fs, 100.0, 1.0);
        let r_in = rms(&hum100[300..1700]);
        let one = remove_mains_hum_iir(&hum100, fs, 50.0, 1, 3.0);
        let two = remove_mains_hum_iir(&hum100, fs, 50.0, 2, 3.0);
        let r_one = rms(&one[300..1700]) / r_in;
        let r_two = rms(&two[300..1700]) / r_in;
        assert!(r_one > 0.95, "1 harmonic must keep 100 Hz: ratio {r_one}");
        assert!(r_two < 0.05, "2 harmonics must kill 100 Hz: ratio {r_two}");
    }

    #[test]
    fn mains_hum_skips_harmonics_near_nyquist() {
        // fs = 200 ⇒ Nyquist 100 Hz: harmonics 2..4 of 50 Hz all exceed 0.99·Nyquist
        // and must be skipped without panicking or destabilizing the cascade.
        let fs = 200.0;
        let x = tone(400, fs, 30.0, 1.0);
        let out = remove_mains_hum_iir(&x, fs, 50.0, 4, 2.0);
        assert_eq!(out.len(), x.len());
        assert!(out.iter().all(|v| v.is_finite()));
        let ratio = rms(&out[60..340]) / rms(&x[60..340]);
        assert!(ratio > 0.9, "30 Hz passband tone kept: ratio {ratio}");
    }

    #[test]
    fn biquad_state_matches_batch_filter_and_resets() {
        let sec = rbj_notch(1000.0, 50.0, 10.0);
        let x: Vec<f64> = (0..200)
            .map(|i| (i as f64 * 0.13).sin() + 0.5 * (i as f64 * 0.031).cos())
            .collect();
        let batch = sec.filter(&x);
        let mut state = BiquadState::new(sec);
        let streamed: Vec<f64> = x.iter().map(|&v| state.push(v)).collect();
        assert_eq!(streamed, batch, "push must equal Biquad::filter exactly");
        // reset() restores the initial (rest) state: the second run is identical.
        state.reset();
        let again: Vec<f64> = x.iter().map(|&v| state.push(v)).collect();
        assert_eq!(again, batch);
    }

    #[test]
    fn degenerate_inputs_come_back_unchanged() {
        let empty: [f64; 0] = [];
        assert!(notch_iir(&empty, 1000.0, 50.0, 2.0).is_empty());
        assert!(remove_mains_hum_iir(&empty, 1000.0, 50.0, 3, 2.0).is_empty());
        assert!(filtfilt_sos(&[rbj_notch(1000.0, 50.0, 10.0)], &empty).is_empty());
        for n in 1..=3
        {
            let x: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
            for out in [
                notch_iir(&x, 1000.0, 50.0, 2.0),
                remove_mains_hum_iir(&x, 1000.0, 50.0, 3, 2.0),
                filtfilt_sos(&[rbj_notch(1000.0, 50.0, 10.0)], &x),
            ]
            {
                assert_eq!(out.len(), n);
                assert!(out.iter().all(|v| v.is_finite()));
            }
        }
        // An empty cascade is the identity.
        let x = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(filtfilt_sos(&[], &x), x.to_vec());
    }

    #[test]
    fn constant_signal_passes_unchanged() {
        let x = vec![3.5; 200];
        for out in [
            notch_iir(&x, 1000.0, 50.0, 2.0),
            remove_mains_hum_iir(&x, 1000.0, 50.0, 3, 3.0),
        ]
        {
            assert_eq!(out.len(), x.len());
            for &v in &out
            {
                assert!((v - 3.5).abs() < 1.0e-9, "constant drifted to {v}");
            }
        }
    }

    #[test]
    fn invalid_design_parameters_degrade_to_identity() {
        for sec in [
            rbj_notch(1000.0, 0.0, 5.0),
            rbj_notch(1000.0, -3.0, 5.0),
            rbj_notch(1000.0, 500.0, 5.0), // at Nyquist
            rbj_notch(1000.0, 600.0, 5.0), // above Nyquist
            rbj_notch(1000.0, 50.0, 0.0),
            rbj_notch(1000.0, 50.0, -1.0),
            rbj_notch(0.0, 50.0, 5.0),
            rbj_notch(1000.0, f64::NAN, 5.0),
            rbj_notch(f64::INFINITY, 50.0, 5.0),
        ]
        {
            assert_eq!(sec, IDENTITY);
        }
        // The convenience wrappers pass the signal through untouched in those cases.
        let x = tone(64, 1000.0, 10.0, 1.0);
        assert_eq!(notch_iir(&x, 1000.0, 50.0, 0.0), x);
        assert_eq!(notch_iir(&x, 1000.0, 700.0, 4.0), x);
        assert_eq!(remove_mains_hum_iir(&x, 1000.0, 50.0, 0, 3.0), x);
        assert_eq!(remove_mains_hum_iir(&x, 1000.0, -50.0, 3, 3.0), x);
        assert_eq!(remove_mains_hum_iir(&x, 1000.0, 50.0, 3, 0.0), x);
    }
}
