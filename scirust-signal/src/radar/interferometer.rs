//! Phase-comparison (interferometric) monopulse angle estimation.
//!
//! Where amplitude-comparison monopulse ([`super::monopulse`]) reads a target's
//! off-boresight angle from the *ratio* of two squinted beams, a **phase
//! interferometer** reads it from the *phase difference* between two antenna
//! elements separated by a baseline `d`. A plane wave from angle `θ` reaches the
//! far element with an extra path `d·sin θ`, i.e. a phase lead
//! `Δφ = 2π·d·sin θ/λ`; measuring `Δφ` and inverting gives
//! `θ = arcsin(Δφ·λ/(2π·d))`. A wide baseline steepens `Δφ` (finer angle
//! accuracy) but wraps the phase past `±π` sooner, so the **unambiguous field of
//! view** `arcsin(λ/2d)` shrinks — the interferometer's resolution/ambiguity
//! trade-off. Built on the crate's [`Complex`](crate::complex::Complex);
//! dependency-free.

use crate::complex::Complex;
use std::f64::consts::PI;

/// The interferometric **phase difference** `Δφ = 2π·d·sin θ/λ` (rad) between the
/// two elements for a target at `theta` (rad from boresight), baseline `baseline`
/// and wavelength `wavelength` (same units). May exceed `±π` for a wide baseline
/// (an ambiguous measurement).
pub fn phase_difference(theta: f64, baseline: f64, wavelength: f64) -> f64 {
    if wavelength == 0.0
    {
        return 0.0;
    }
    2.0 * PI * baseline * theta.sin() / wavelength
}

/// Wrap a phase to the principal interval `(−π, π]`.
pub fn wrap_phase(phase: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let w = (phase + PI).rem_euclid(two_pi) - PI;
    if w <= -PI { w + two_pi } else { w }
}

/// Recover the off-boresight **angle** (rad) from a measured phase difference
/// `phase_diff`, inverting `Δφ = 2π·d·sin θ/λ`: `θ = arcsin(Δφ·λ/(2π·d))`. The
/// argument to `arcsin` is clamped to `[−1, 1]`; `0` for a degenerate array.
pub fn angle_from_phase(phase_diff: f64, baseline: f64, wavelength: f64) -> f64 {
    if baseline <= 0.0 || wavelength <= 0.0
    {
        return 0.0;
    }
    let s = (phase_diff * wavelength / (2.0 * PI * baseline)).clamp(-1.0, 1.0);
    s.asin()
}

/// The measured phase difference from the two element voltages `near`, `far`:
/// `arg(far·conj(near))`, wrapped to `(−π, π]`. This is what a receiver actually
/// observes, and it aliases when the true `Δφ` exceeds `±π`.
pub fn phase_from_signals(near: Complex, far: Complex) -> f64 {
    (far * near.conj()).phase()
}

/// The **unambiguous field of view** `±arcsin(λ/2d)` (rad): the largest
/// off-boresight angle whose phase difference stays within `±π`, so it can be
/// inverted without ambiguity. A half-wavelength baseline (`d = λ/2`) is
/// unambiguous over the full `±90°`.
pub fn unambiguous_angle(baseline: f64, wavelength: f64) -> f64 {
    if baseline <= 0.0
    {
        return PI / 2.0;
    }
    let s = wavelength / (2.0 * baseline);
    if s >= 1.0 { PI / 2.0 } else { s.asin() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_is_zero_on_boresight_and_odd() {
        let (d, lambda) = (0.02, 0.03);
        assert!(phase_difference(0.0, d, lambda).abs() < 1e-15);
        let p = phase_difference(0.3, d, lambda);
        assert!((phase_difference(-0.3, d, lambda) + p).abs() < 1e-12);
        assert!(p > 0.0);
    }

    #[test]
    fn estimate_inverts_the_phase_within_the_unambiguous_field() {
        // A half-wavelength baseline is unambiguous over the whole field.
        let (d, lambda) = (0.015, 0.03); // d = λ/2
        for &theta in &[-1.2, -0.5, 0.0, 0.4, 1.0]
        {
            let dphi = phase_difference(theta, d, lambda);
            let est = angle_from_phase(dphi, d, lambda);
            assert!((est - theta).abs() < 1e-9, "{est} vs {theta}");
        }
    }

    #[test]
    fn measured_phase_from_signals_recovers_the_difference() {
        // Two element voltages differing by a known phase.
        let near = Complex::cis(0.3);
        let far = Complex::cis(0.3 + 0.8);
        assert!((phase_from_signals(near, far) - 0.8).abs() < 1e-12);
        // For a target at θ (within the unambiguous field), the observed phase
        // inverts back to the angle.
        let (d, lambda, theta) = (0.015, 0.03, 0.6);
        let v0 = Complex::cis(0.0);
        let v1 = Complex::cis(phase_difference(theta, d, lambda));
        let measured = phase_from_signals(v0, v1);
        assert!((angle_from_phase(measured, d, lambda) - theta).abs() < 1e-9);
    }

    #[test]
    fn wide_baseline_narrows_the_unambiguous_field() {
        let lambda = 0.03;
        // d = λ/2 ⇒ ±90°.
        assert!((unambiguous_angle(0.5 * lambda, lambda) - PI / 2.0).abs() < 1e-9);
        // d = λ ⇒ arcsin(1/2) = 30°.
        assert!((unambiguous_angle(lambda, lambda) - (0.5_f64).asin()).abs() < 1e-9);
        // A wider baseline gives a narrower unambiguous field.
        assert!(unambiguous_angle(2.0 * lambda, lambda) < unambiguous_angle(lambda, lambda));
    }

    #[test]
    fn wide_baseline_measurement_aliases_outside_the_field() {
        // A baseline of 2λ has an unambiguous field of arcsin(1/4) ≈ 14.5°.
        // A target beyond it produces a wrapped phase that the receiver cannot
        // disambiguate, so the naive estimate differs from the truth.
        let (d, lambda, theta) = (0.06, 0.03, 0.5); // d = 2λ, θ = 0.5 rad ≈ 28.6°
        assert!(theta > unambiguous_angle(d, lambda));
        let true_phase = phase_difference(theta, d, lambda);
        assert!(
            true_phase.abs() > PI,
            "phase should exceed ±π: {true_phase}"
        );
        let measured = wrap_phase(true_phase);
        assert!((measured - wrap_phase(true_phase)).abs() < 1e-12);
        // The wrapped measurement maps to a different (aliased) angle.
        assert!((angle_from_phase(measured, d, lambda) - theta).abs() > 1e-3);
    }

    #[test]
    fn wrap_phase_maps_into_the_principal_interval() {
        assert!((wrap_phase(0.3) - 0.3).abs() < 1e-12);
        assert!((wrap_phase(PI + 0.1) - (-PI + 0.1)).abs() < 1e-12);
        assert!((wrap_phase(-PI - 0.1) - (PI - 0.1)).abs() < 1e-12);
    }

    #[test]
    fn degenerate_arrays_are_safe() {
        assert_eq!(angle_from_phase(1.0, 0.0, 0.03), 0.0);
        assert_eq!(phase_difference(0.5, 0.02, 0.0), 0.0);
        assert!((unambiguous_angle(0.0, 0.03) - PI / 2.0).abs() < 1e-12);
    }
}
