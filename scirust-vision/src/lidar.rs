//! Laser / LIDAR ranging: pulsed time-of-flight and CW phase-shift.
//!
//! A laser rangefinder recovers distance from the round-trip travel time of
//! light. Two schemes dominate. A **pulsed** (direct time-of-flight) rangefinder
//! fires a short pulse and times the echo: the one-way range is `R = c·t/2`, the
//! range resolution is set by the pulse width `τ` as `ΔR = c·τ/2`, and — as in a
//! pulsed radar — a finite pulse-repetition frequency caps the unambiguous range
//! at `R_ua = c/(2·PRF)`, beyond which the echo of one pulse is confused with the
//! next. A **continuous-wave (CW)** rangefinder instead amplitude-modulates the
//! beam at `f_m` and measures the phase lag `φ` of the returning envelope: the
//! range is `R = c·φ/(4π·f_m)`, which wraps every `2π`, so its unambiguous range
//! is `R_ua = c/(2·f_m)` — a phase of `2π` maps to exactly that distance. This
//! module gives those closed-form conversions and their ambiguity limits.
//! Dependency-free.

use std::f64::consts::PI;

/// Speed of light `c` (m/s).
const C_LIGHT: f64 = 2.997_924_58e8;

/// One-way **range from round-trip time-of-flight** `R = c·t/2` (m): light covers
/// the target distance twice, so half the measured travel time `tof` (s) yields
/// the range.
pub fn range_from_time_of_flight(tof: f64) -> f64 {
    C_LIGHT * tof / 2.0
}

/// Round-trip **time-of-flight for a range** `t = 2·R/c` (s): the time light needs
/// to reach the target at range `range` (m) and return. Inverse of
/// [`range_from_time_of_flight`].
pub fn time_of_flight(range: f64) -> f64 {
    2.0 * range / C_LIGHT
}

/// The **range resolution** `ΔR = c·τ/2` (m) of a pulsed rangefinder, set by the
/// transmitted pulse width `pulse_width` `τ` (s): two targets closer than `ΔR`
/// return overlapping echoes and cannot be separated.
pub fn range_resolution(pulse_width: f64) -> f64 {
    C_LIGHT * pulse_width / 2.0
}

/// The **pulsed unambiguous range** `R_ua = c/(2·PRF)` (m): the maximum range whose
/// echo still returns before the next pulse is fired at pulse-repetition frequency
/// `prf` (Hz). `f64::INFINITY` for a non-positive PRF (a single pulse is never
/// ambiguous).
pub fn max_unambiguous_range_pulsed(prf: f64) -> f64 {
    if prf <= 0.0
    {
        return f64::INFINITY;
    }
    C_LIGHT / (2.0 * prf)
}

/// The **CW phase-shift range** `R = c·φ/(4π·f_m)` (m): distance recovered from the
/// phase lag `phase` `φ` (rad) of a beam amplitude-modulated at `mod_freq` `f_m`
/// (Hz). The phase wraps every `2π`, so the measurement is unique only within
/// [`max_unambiguous_range_cw`].
pub fn range_from_phase(phase: f64, mod_freq: f64) -> f64 {
    C_LIGHT * phase / (4.0 * PI * mod_freq)
}

/// The **CW unambiguous range** `R_ua = c/(2·f_m)` (m): the range at which the
/// phase lag of a beam modulated at `mod_freq` `f_m` (Hz) reaches `2π` and wraps.
/// `f64::INFINITY` for a non-positive modulation frequency.
pub fn max_unambiguous_range_cw(mod_freq: f64) -> f64 {
    if mod_freq <= 0.0
    {
        return f64::INFINITY;
    }
    C_LIGHT / (2.0 * mod_freq)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_and_time_of_flight_round_trip() {
        // R -> t -> R must return the original range exactly (inverse maps).
        for &range in &[1.0_f64, 150.0, 1_500.0, 42_000.0]
        {
            let tof = time_of_flight(range);
            assert!(
                (range_from_time_of_flight(tof) - range).abs() < 1e-9,
                "range {range}"
            );
        }
        // A known 1 µs round trip is 149.896... m one-way (c/2 · 1e-6).
        let expect = C_LIGHT * 1e-6 / 2.0;
        assert!((range_from_time_of_flight(1e-6) - expect).abs() < 1e-9);
    }

    #[test]
    fn range_resolution_is_half_c_tau() {
        // ΔR = c·τ/2: a 10 ns pulse resolves ~1.5 m.
        let tau = 10e-9;
        assert!((range_resolution(tau) - C_LIGHT * tau / 2.0).abs() < 1e-12);
        // Resolution scales linearly with pulse width.
        assert!((range_resolution(2.0 * tau) - 2.0 * range_resolution(tau)).abs() < 1e-12);
    }

    #[test]
    fn pulsed_unambiguous_range_is_inverse_prf() {
        // R_ua = c/(2·PRF): a higher PRF shortens the unambiguous range.
        assert!((max_unambiguous_range_pulsed(1000.0) - C_LIGHT / 2000.0).abs() < 1e-6);
        assert!(max_unambiguous_range_pulsed(2000.0) < max_unambiguous_range_pulsed(1000.0));
        // The echo of a target at R_ua arrives exactly at the next pulse interval.
        let prf = 10_000.0;
        let r_ua = max_unambiguous_range_pulsed(prf);
        assert!((time_of_flight(r_ua) - 1.0 / prf).abs() < 1e-15);
    }

    #[test]
    fn phase_range_inverts_a_known_range() {
        // Build the phase a target at R produces, then recover R from it.
        let (mod_freq, range) = (10e6_f64, 7.5);
        let phase = 4.0 * PI * mod_freq * range / C_LIGHT;
        assert!((range_from_phase(phase, mod_freq) - range).abs() < 1e-9);
        // Range grows linearly with measured phase.
        assert!((range_from_phase(2.0 * phase, mod_freq) - 2.0 * range).abs() < 1e-9);
    }

    #[test]
    fn phase_of_two_pi_is_the_cw_unambiguous_range() {
        // A full 2π phase lag maps to exactly R_ua = c/(2·f_m).
        for &mod_freq in &[1e6_f64, 10e6, 75e6]
        {
            let at_2pi = range_from_phase(2.0 * PI, mod_freq);
            assert!(
                (at_2pi - max_unambiguous_range_cw(mod_freq)).abs() < 1e-6,
                "mod_freq {mod_freq}"
            );
        }
    }

    #[test]
    fn cw_unambiguous_range_is_inverse_mod_freq() {
        // R_ua = c/(2·f_m): a higher modulation frequency shortens the window.
        assert!((max_unambiguous_range_cw(15e6) - C_LIGHT / 30e6).abs() < 1e-9);
        assert!(max_unambiguous_range_cw(30e6) < max_unambiguous_range_cw(15e6));
    }

    #[test]
    fn degenerate_inputs_guard_to_infinity_without_nan() {
        // Non-positive PRF / modulation frequency => infinite (never ambiguous).
        assert!(max_unambiguous_range_pulsed(0.0).is_infinite());
        assert!(max_unambiguous_range_pulsed(-5.0).is_infinite());
        assert!(max_unambiguous_range_cw(0.0).is_infinite());
        assert!(max_unambiguous_range_cw(-1.0).is_infinite());
        // No path produces NaN.
        for &v in &[0.0_f64, -1.0, 1000.0]
        {
            assert!(!max_unambiguous_range_pulsed(v).is_nan());
            assert!(!max_unambiguous_range_cw(v).is_nan());
        }
        // Zero range and zero time-of-flight are mutual inverses at the origin.
        assert_eq!(range_from_time_of_flight(0.0), 0.0);
        assert_eq!(time_of_flight(0.0), 0.0);
    }
}
