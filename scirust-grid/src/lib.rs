//! # scirust-grid — power-system / smart-grid analytics
//!
//! Deterministic, pure-Rust estimators for grid monitoring and protection:
//! grid **frequency** and **RoCoF** (rate of change of frequency), **synchro-
//! phasor** magnitude/phase, **THD** (total harmonic distortion), a
//! RoCoF-based **islanding** check, three-phase **symmetrical components** /
//! voltage-unbalance, IEC 61000-4-30 **voltage-event** classification, and —
//! the layer that ties this together into wide-area protection — weighted
//! least-squares **state estimation** with bad-data detection
//! ([`state_estimation`]) and mho-characteristic **distance relay** zone
//! logic ([`distance_relay`]). All built on the SciRust FFT/linear-algebra
//! stack, so a run is bit-reproducible — the determinism a protection relay
//! needs.
//!
//! Commercial use of the original frequency/RoCoF/synchrophasor/THD
//! analytics is gated by [`GridModule`]: unlock the module against a signed
//! entitlement ([`scirust_license`]) before running them. The raw functions
//! remain available for noncommercial use under the dual license; the newer
//! modules ([`power_quality`], [`symmetrical`], [`flicker`],
//! [`state_estimation`], [`distance_relay`]) are ungated, like
//! [`power_quality`] and [`symmetrical`] before them.

use scirust_signal::{Complex, fft_real, hanning};
use serde::{Deserialize, Serialize};

pub mod distance_relay;
pub mod flicker;
pub mod license;
pub mod power_quality;
pub mod state_estimation;
pub mod symmetrical;
pub use distance_relay::{
    DistanceRelay, RelayZone, TripDecision, apparent_impedance, mho_operates,
};
pub use flicker::{flicker_severity, perceptibility_weight};
pub use license::GridModule;
pub use power_quality::{EventSpan, VoltageEvent, classify_voltage, cycle_rms, detect_events};
pub use state_estimation::{
    BadDataReport, GridError, StateEstimate, chi_squared_test, largest_normalized_residual_test,
    wls_state_estimate,
};
pub use symmetrical::{symmetrical_components, voltage_unbalance_factor};

/// Windowed half-spectrum of `signal` (Hann window, low leakage).
fn windowed_spectrum(signal: &[f64]) -> Vec<Complex> {
    let win = hanning(signal.len());
    let windowed: Vec<f64> = signal.iter().zip(&win).map(|(&x, &w)| x * w).collect();
    fft_real(&windowed)
}

/// Magnitude at integer bin `b` (0 outside range).
fn mag_at(spec: &[Complex], b: isize) -> f64 {
    if b < 0 || b as usize >= spec.len()
    {
        0.0
    }
    else
    {
        spec[b as usize].mag()
    }
}

/// Estimate the dominant frequency (Hz) near `nominal_hz` to sub-bin accuracy
/// via parabolic interpolation of the windowed spectrum peak.
///
/// `signal.len()` must be a power of two. Searches within ±`search_hz` of
/// `nominal_hz`.
pub fn grid_frequency(signal: &[f64], sample_rate: f64, nominal_hz: f64, search_hz: f64) -> f64 {
    let n = signal.len();
    let spec = windowed_spectrum(signal);
    let bin_hz = sample_rate / n as f64;
    let lo = ((nominal_hz - search_hz) / bin_hz).floor().max(1.0) as usize;
    let hi = (((nominal_hz + search_hz) / bin_hz).ceil() as usize).min(spec.len() - 2);

    // Peak bin in the search band.
    let mut peak = lo;
    let mut best = mag_at(&spec, lo as isize);
    for b in lo..=hi
    {
        let m = mag_at(&spec, b as isize);
        if m > best
        {
            best = m;
            peak = b;
        }
    }
    // Parabolic interpolation using the two neighbouring bins.
    let m0 = mag_at(&spec, peak as isize - 1);
    let m1 = mag_at(&spec, peak as isize);
    let m2 = mag_at(&spec, peak as isize + 1);
    let denom = m0 - 2.0 * m1 + m2;
    let delta = if denom.abs() > 1e-20
    {
        0.5 * (m0 - m2) / denom
    }
    else
    {
        0.0
    };
    (peak as f64 + delta) * bin_hz
}

/// Rate of change of frequency (Hz/s) from a least-squares line through a
/// sequence of frequency estimates spaced `dt` apart.
pub fn rocof(freqs: &[f64], dt: f64) -> f64 {
    let n = freqs.len();
    if n < 2 || dt <= 0.0
    {
        return 0.0;
    }
    let nf = n as f64;
    let mean_t = (n - 1) as f64 * dt / 2.0;
    let mean_f = freqs.iter().sum::<f64>() / nf;
    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &f) in freqs.iter().enumerate()
    {
        let t = i as f64 * dt - mean_t;
        num += t * (f - mean_f);
        den += t * t;
    }
    if den > 0.0 { num / den } else { 0.0 }
}

/// Synchrophasor (magnitude, phase in radians) of the component at `freq_hz`,
/// estimated by a single-bin DFT of the raw signal (rectangular window so the
/// phase is unbiased on an integer-period capture).
pub fn synchrophasor(signal: &[f64], sample_rate: f64, freq_hz: f64) -> (f64, f64) {
    let n = signal.len();
    let w = 2.0 * core::f64::consts::PI * freq_hz / sample_rate;
    let (mut re, mut im) = (0.0, 0.0);
    for (k, &x) in signal.iter().enumerate()
    {
        let ang = w * k as f64;
        re += x * ang.cos();
        im -= x * ang.sin();
    }
    let mag = 2.0 * (re * re + im * im).sqrt() / n as f64;
    (mag, im.atan2(re))
}

/// Total Harmonic Distortion: `sqrt(Σ_{h≥2} A_h²) / A_1`, using windowed peak
/// magnitudes at the first `n_harmonics` multiples of `fundamental_hz`.
pub fn thd(signal: &[f64], sample_rate: f64, fundamental_hz: f64, n_harmonics: usize) -> f64 {
    let n = signal.len();
    let spec = windowed_spectrum(signal);
    let bin_hz = sample_rate / n as f64;
    let amp_at = |f: f64| -> f64 {
        let b = (f / bin_hz).round() as isize;
        // peak over ±1 bin to catch the windowed main lobe
        mag_at(&spec, b - 1)
            .max(mag_at(&spec, b))
            .max(mag_at(&spec, b + 1))
    };
    let a1 = amp_at(fundamental_hz);
    if a1 <= 0.0
    {
        return 0.0;
    }
    let mut harm_sq = 0.0;
    for h in 2..=n_harmonics
    {
        let a = amp_at(fundamental_hz * h as f64);
        harm_sq += a * a;
    }
    harm_sq.sqrt() / a1
}

/// RoCoF-based islanding detection (IEEE 1547-style): an island is declared when
/// `|RoCoF|` exceeds `rocof_limit` (Hz/s) or the frequency leaves
/// `[f_min, f_max]`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct IslandingDetector {
    pub rocof_limit: f64,
    pub f_min: f64,
    pub f_max: f64,
}

impl IslandingDetector {
    /// Whether the present frequency / RoCoF indicate an islanding condition.
    pub fn is_islanding(&self, freq_hz: f64, rocof_hz_s: f64) -> bool {
        rocof_hz_s.abs() > self.rocof_limit || freq_hz < self.f_min || freq_hz > self.f_max
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    fn sine(n: usize, sr: f64, freq: f64, amp: f64, phase: f64) -> Vec<f64> {
        (0..n)
            .map(|i| amp * (2.0 * PI * freq * i as f64 / sr + phase).sin())
            .collect()
    }

    #[test]
    fn recovers_off_nominal_frequency() {
        let (n, sr) = (4096usize, 4096.0);
        let sig = sine(n, sr, 50.2, 1.0, 0.0);
        let f = grid_frequency(&sig, sr, 50.0, 2.0);
        assert!((f - 50.2).abs() < 0.05, "frequency {f} (want 50.2)");
    }

    #[test]
    fn rocof_recovers_a_frequency_ramp() {
        // Frequency drops at 0.5 Hz/s.
        let freqs: Vec<f64> = (0..10).map(|k| 50.0 - 0.5 * k as f64 * 0.1).collect();
        let r = rocof(&freqs, 0.1);
        assert!((r - (-0.5)).abs() < 1e-9, "RoCoF {r}");
    }

    #[test]
    fn thd_recovers_known_harmonics() {
        let (n, sr) = (4096usize, 4096.0);
        // Fundamental 50 Hz + 5% 3rd + 3% 5th harmonic (on-bin).
        let mut sig = sine(n, sr, 50.0, 1.0, 0.0);
        for (i, s) in sig.iter_mut().enumerate()
        {
            let t = i as f64 / sr;
            *s += 0.05 * (2.0 * PI * 150.0 * t).sin();
            *s += 0.03 * (2.0 * PI * 250.0 * t).sin();
        }
        let d = thd(&sig, sr, 50.0, 7);
        let want = (0.05_f64.powi(2) + 0.03_f64.powi(2)).sqrt();
        assert!((d - want).abs() < 5e-3, "THD {d} (want {want})");
    }

    #[test]
    fn synchrophasor_magnitude_and_phase() {
        let (n, sr) = (4096usize, 4096.0);
        let sig = sine(n, sr, 50.0, 2.0, 0.7);
        let (mag, phase) = synchrophasor(&sig, sr, 50.0);
        assert!((mag - 2.0).abs() < 1e-2, "mag {mag}");
        // sin(θ) = cos(θ − π/2): the measured phase is offset by −π/2 from a cos ref.
        assert!((phase - (0.7 - PI / 2.0)).abs() < 1e-2, "phase {phase}");
    }

    #[test]
    fn islanding_on_rocof_or_band() {
        let d = IslandingDetector {
            rocof_limit: 0.5,
            f_min: 49.5,
            f_max: 50.5,
        };
        assert!(!d.is_islanding(50.0, 0.1));
        assert!(d.is_islanding(50.0, 0.8)); // RoCoF trip
        assert!(d.is_islanding(49.0, 0.0)); // under-frequency
    }
}
