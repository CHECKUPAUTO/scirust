//! Radar measurement accuracy — Cramér–Rao lower bounds.
//!
//! The estimators elsewhere in this crate (matched-filter delay in
//! [`super::matched_filter`], Doppler in [`super::doppler`], monopulse angle in
//! [`super::monopulse`]) each produce a measurement; this module gives the
//! *theoretical floor* on how precise any unbiased estimate can be — the
//! **Cramér–Rao lower bound (CRLB)**. The three headline results, all scaling as
//! `1/√SNR`:
//!
//! - **delay / range** `σ_τ = 1/(2π·β_rms·√(2·SNR))`, `σ_R = (c/2)·σ_τ` — sharper
//!   with wider (RMS) signal bandwidth `β_rms`;
//! - **Doppler / velocity** `σ_fd = 1/(2π·T_rms·√(2·SNR))`, `σ_v = (λ/2)·σ_fd` —
//!   sharper with a longer coherent dwell (RMS duration `T_rms`);
//! - **angle** `σ_θ = θ_3dB/(k_m·√(2·SNR))` — the monopulse accuracy, sharper for
//!   a narrow beam and a steep difference-pattern slope `k_m`.
//!
//! These are the numbers a radar link budget must close to *meet an accuracy
//! spec*, the complement to the detection budget in [`super::range_equation`].
//! `SNR` is linear (a power ratio, not dB). Dependency-free.

use std::f64::consts::PI;

/// Speed of light (m/s).
const C_LIGHT: f64 = 2.997_924_58e8;

/// The **RMS bandwidth** `β_rms = B/√12` of a flat (rectangular, LFM-like)
/// spectrum of width `bandwidth` (Hz) — the second moment of the spectrum that
/// drives the delay accuracy. `0` for a non-positive bandwidth.
pub fn rms_bandwidth_lfm(bandwidth: f64) -> f64 {
    if bandwidth <= 0.0
    {
        return 0.0;
    }
    bandwidth / 12.0_f64.sqrt()
}

/// The **RMS duration** `T_rms = T/√12` of a rectangular pulse of length
/// `pulse_duration` (s) — the time-domain analogue that drives the Doppler
/// accuracy. `0` for a non-positive duration.
pub fn rms_duration_rect(pulse_duration: f64) -> f64 {
    if pulse_duration <= 0.0
    {
        return 0.0;
    }
    pulse_duration / 12.0_f64.sqrt()
}

/// The **delay (time) CRLB** `σ_τ = 1/(2π·β_rms·√(2·SNR))` (s) for RMS bandwidth
/// `rms_bandwidth` (Hz) and linear `snr`. `+∞` (no information) for a non-positive
/// SNR or bandwidth.
pub fn delay_crlb(snr: f64, rms_bandwidth: f64) -> f64 {
    if snr <= 0.0 || rms_bandwidth <= 0.0
    {
        return f64::INFINITY;
    }
    1.0 / (2.0 * PI * rms_bandwidth * (2.0 * snr).sqrt())
}

/// The **range CRLB** `σ_R = (c/2)·σ_τ` (m) — the delay bound carried to range.
/// `+∞` for a non-positive SNR or bandwidth.
pub fn range_crlb(snr: f64, rms_bandwidth: f64) -> f64 {
    (C_LIGHT / 2.0) * delay_crlb(snr, rms_bandwidth)
}

/// The **Doppler-frequency CRLB** `σ_fd = 1/(2π·T_rms·√(2·SNR))` (Hz) for RMS
/// pulse duration `rms_duration` (s) and linear `snr`. `+∞` for a non-positive
/// SNR or duration.
pub fn doppler_crlb(snr: f64, rms_duration: f64) -> f64 {
    if snr <= 0.0 || rms_duration <= 0.0
    {
        return f64::INFINITY;
    }
    1.0 / (2.0 * PI * rms_duration * (2.0 * snr).sqrt())
}

/// The **radial-velocity CRLB** `σ_v = (λ/2)·σ_fd` (m/s) — the Doppler bound
/// carried to velocity at `wavelength` (m). `+∞` for a non-positive SNR,
/// duration, or wavelength.
pub fn velocity_crlb(snr: f64, rms_duration: f64, wavelength: f64) -> f64 {
    if snr <= 0.0 || rms_duration <= 0.0 || wavelength <= 0.0
    {
        return f64::INFINITY;
    }
    (wavelength / 2.0) * doppler_crlb(snr, rms_duration)
}

/// The **angle CRLB** `σ_θ = θ_3dB/(k_m·√(2·SNR))` (rad) — the single-dwell
/// monopulse accuracy for beamwidth `beamwidth` (rad), difference-pattern
/// (monopulse) slope `monopulse_slope`, and linear `snr`. `+∞` for a non-positive
/// SNR, beamwidth, or slope.
pub fn angle_crlb(snr: f64, beamwidth: f64, monopulse_slope: f64) -> f64 {
    if snr <= 0.0 || beamwidth <= 0.0 || monopulse_slope <= 0.0
    {
        return f64::INFINITY;
    }
    beamwidth / (monopulse_slope * (2.0 * snr).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_bandwidth_and_duration_of_a_flat_spectrum() {
        let b = 10.0e6;
        assert!((rms_bandwidth_lfm(b) - b / 12.0_f64.sqrt()).abs() < 1e-3);
        let t = 1e-3;
        assert!((rms_duration_rect(t) - t / 12.0_f64.sqrt()).abs() < 1e-15);
        assert_eq!(rms_bandwidth_lfm(0.0), 0.0);
        assert_eq!(rms_duration_rect(-1.0), 0.0);
    }

    #[test]
    fn delay_crlb_matches_the_closed_form() {
        let (snr, brms) = (100.0_f64, 5.0e6);
        let expected = 1.0 / (2.0 * PI * brms * (2.0 * snr).sqrt());
        assert!((delay_crlb(snr, brms) - expected).abs() / expected < 1e-12);
        // Sub-nanosecond timing at 5 MHz RMS bandwidth, 20 dB SNR.
        assert!(delay_crlb(snr, brms) < 1e-8);
    }

    #[test]
    fn range_is_c_over_two_times_delay_and_snr_scales() {
        let (snr, brms) = (50.0, 4.0e6);
        assert!((range_crlb(snr, brms) - (C_LIGHT / 2.0) * delay_crlb(snr, brms)).abs() < 1e-12);
        // A realistic sub-metre range accuracy.
        assert!(range_crlb(snr, brms) < 1.0);
        // CRLB ∝ 1/√SNR: quadrupling SNR halves it.
        assert!((delay_crlb(4.0 * snr, brms) / delay_crlb(snr, brms) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn doppler_and_velocity_crlb_match_and_convert() {
        let (snr, trms, lam) = (100.0_f64, 2e-4, 0.03);
        let expected = 1.0 / (2.0 * PI * trms * (2.0 * snr).sqrt());
        assert!((doppler_crlb(snr, trms) - expected).abs() / expected < 1e-12);
        // velocity = (λ/2)·doppler.
        assert!(
            (velocity_crlb(snr, trms, lam) - (lam / 2.0) * doppler_crlb(snr, trms)).abs() < 1e-15
        );
        // ∝ 1/√SNR.
        assert!((doppler_crlb(4.0 * snr, trms) / doppler_crlb(snr, trms) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn angle_crlb_scaling() {
        let (snr, bw, km) = (100.0_f64, 3.0_f64.to_radians(), 1.6);
        let expected = bw / (km * (2.0 * snr).sqrt());
        assert!((angle_crlb(snr, bw, km) - expected).abs() / expected < 1e-12);
        // ∝ beamwidth, ∝ 1/√SNR, ∝ 1/slope.
        assert!(angle_crlb(snr, 6.0_f64.to_radians(), km) > angle_crlb(snr, bw, km));
        assert!((angle_crlb(4.0 * snr, bw, km) / angle_crlb(snr, bw, km) - 0.5).abs() < 1e-9);
        assert!(angle_crlb(snr, bw, 3.2) < angle_crlb(snr, bw, km));
    }

    #[test]
    fn accuracy_sharpens_with_snr_bandwidth_and_dwell() {
        let (snr, brms, trms, lam, bw, km) = (100.0, 5e6, 2e-4, 0.03, 0.05, 1.6);
        // More SNR sharpens every measurement.
        assert!(delay_crlb(2.0 * snr, brms) < delay_crlb(snr, brms));
        assert!(doppler_crlb(2.0 * snr, trms) < doppler_crlb(snr, trms));
        assert!(angle_crlb(2.0 * snr, bw, km) < angle_crlb(snr, bw, km));
        // Wider bandwidth sharpens range; longer dwell sharpens velocity.
        assert!(range_crlb(snr, 2.0 * brms) < range_crlb(snr, brms));
        assert!(velocity_crlb(snr, 2.0 * trms, lam) < velocity_crlb(snr, trms, lam));
    }

    #[test]
    fn degenerate_inputs_are_safe() {
        assert!(delay_crlb(0.0, 5e6).is_infinite());
        assert!(delay_crlb(100.0, 0.0).is_infinite());
        assert!(range_crlb(-1.0, 5e6).is_infinite());
        assert!(doppler_crlb(100.0, 0.0).is_infinite());
        assert!(velocity_crlb(0.0, 2e-4, 0.03).is_infinite());
        // Zero wavelength must not produce 0·∞ = NaN.
        let v = velocity_crlb(100.0, 2e-4, 0.0);
        assert!(v.is_infinite() && !v.is_nan());
        assert!(angle_crlb(100.0, 0.0, 1.6).is_infinite());
        assert!(angle_crlb(100.0, 0.05, 0.0).is_infinite());
        assert_eq!(rms_bandwidth_lfm(0.0), 0.0);
        assert_eq!(rms_duration_rect(0.0), 0.0);
    }
}
