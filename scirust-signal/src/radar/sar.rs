//! Synthetic-aperture radar (SAR) azimuth compression.
//!
//! A real antenna of along-track length `D` has an azimuth beamwidth `λ/D`, so at
//! range `R` its ground footprint — and hence its cross-range resolution — is
//! `λR/D`, coarse at long range. **SAR** recovers fine cross-range resolution by
//! synthesising a long aperture from the platform's motion: as the radar flies
//! past a point target at closest-approach range `R₀`, the two-way range history
//! traces a parabola `R(x) ≈ R₀ + (x−x₀)²/(2R₀)`, imprinting a quadratic phase
//! (an **azimuth chirp**) on the slow-time return. Matched-filtering that chirp —
//! exactly the pulse-compression trick of [`super::matched_filter`], now in the
//! along-track dimension — focuses the target to a sharp peak. The synthesised
//! aperture spans `L_sa = λR/D`, and the focused cross-range resolution is the
//! celebrated `δ_az = D/2`: **independent of range**, and finer for a *smaller*
//! antenna. Built on the crate's [`Complex`](crate::complex::Complex) and the
//! existing matched filter; dependency-free.

use super::matched_filter::cross_correlate;
use crate::complex::Complex;
use std::f64::consts::PI;

/// The **synthetic aperture length** `L_sa = λ·R/D` — the along-track distance
/// over which a point target stays within the real antenna's beam (beamwidth
/// `λ/D`), and so the span the coherent integration synthesises. `0` for a
/// degenerate antenna.
pub fn synthetic_aperture_length(wavelength: f64, range: f64, antenna_length: f64) -> f64 {
    if antenna_length <= 0.0
    {
        return 0.0;
    }
    wavelength * range / antenna_length
}

/// The focused **azimuth (cross-range) resolution** `δ_az = D/2` — the SAR
/// hallmark: set only by the physical antenna length, independent of range and
/// wavelength. `0` for a degenerate antenna.
pub fn azimuth_resolution(antenna_length: f64) -> f64 {
    if antenna_length <= 0.0
    {
        return 0.0;
    }
    0.5 * antenna_length
}

/// The **azimuth Doppler bandwidth** `B_d = 2v/D` swept across the synthetic
/// aperture — the slow-time bandwidth the azimuth matched filter compresses. `0`
/// for a degenerate antenna.
pub fn azimuth_doppler_bandwidth(velocity: f64, antenna_length: f64) -> f64 {
    if antenna_length <= 0.0
    {
        return 0.0;
    }
    2.0 * velocity / antenna_length
}

/// The **azimuth FM (chirp) rate** `k_a = 2v²/(λ·R)` (Hz/s) — the Doppler rate of
/// the quadratic phase history as the platform flies past a target at range
/// `range`. `0` for degenerate geometry.
pub fn azimuth_chirp_rate(velocity: f64, wavelength: f64, range: f64) -> f64 {
    if wavelength <= 0.0 || range <= 0.0
    {
        return 0.0;
    }
    2.0 * velocity * velocity / (wavelength * range)
}

/// The slow-time **azimuth phase history** of a unit point target at closest range
/// `range` and along-track offset `target_x`, sampled at platform along-track
/// `positions`. Uses the far-field parabolic approximation
/// `R(x) − R₀ ≈ (x − x₀)²/(2R)`, giving `s(x) = exp(−j·2π·(x − x₀)²/(λ·R))`
/// (unit amplitude). For degenerate geometry (`λ ≤ 0` or `R ≤ 0`) returns a safe
/// all-ones vector of the same length.
pub fn azimuth_history(
    range: f64,
    target_x: f64,
    wavelength: f64,
    positions: &[f64],
) -> Vec<Complex> {
    if wavelength <= 0.0 || range <= 0.0
    {
        return vec![Complex::new(1.0, 0.0); positions.len()];
    }
    let k = 2.0 * PI / (wavelength * range);
    positions
        .iter()
        .map(|&x| {
            let d = x - target_x;
            Complex::cis(-k * d * d)
        })
        .collect()
}

/// The azimuth **reference chirp** (matched-filter replica) for a target at
/// closest range `range`, sampled at `positions`: the phase history of a target
/// at `target_x = 0`. Cross-correlating a return against this focuses it.
pub fn azimuth_reference(range: f64, wavelength: f64, positions: &[f64]) -> Vec<Complex> {
    azimuth_history(range, 0.0, wavelength, positions)
}

/// **Azimuth compression**: cross-correlate the slow-time `signal` with the
/// `reference` chirp and return the focused magnitude profile. A point target
/// collapses to a peak at the lag encoding its along-track position (locate it
/// with [`super::matched_filter::peak_lag`] on the raw correlation).
pub fn focus_azimuth(signal: &[Complex], reference: &[Complex]) -> Vec<f64> {
    cross_correlate(signal, reference)
        .iter()
        .map(|c| c.mag())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::matched_filter::peak_lag;
    use super::*;

    #[test]
    fn closed_form_resolution_aperture_and_bandwidth() {
        let (lam, d) = (0.03, 1.5);
        // Cross-range resolution D/2 — no range dependence at all.
        assert!((azimuth_resolution(d) - 0.75).abs() < 1e-12);
        // Synthetic aperture λR/D grows with range.
        assert!((synthetic_aperture_length(lam, 10_000.0, d) - lam * 10_000.0 / d).abs() < 1e-9);
        assert!(
            synthetic_aperture_length(lam, 20_000.0, d)
                > synthetic_aperture_length(lam, 10_000.0, d)
        );
        // Azimuth Doppler bandwidth 2v/D.
        assert!((azimuth_doppler_bandwidth(200.0, d) - 400.0 / d).abs() < 1e-9);
    }

    #[test]
    fn chirp_rate_scales_with_velocity_squared_and_inverse_range() {
        let (lam, r) = (0.03, 10_000.0);
        assert!((azimuth_chirp_rate(200.0, lam, r) - 2.0 * 200.0 * 200.0 / (lam * r)).abs() < 1e-6);
        // ∝ v².
        assert!(
            (azimuth_chirp_rate(400.0, lam, r) / azimuth_chirp_rate(200.0, lam, r) - 4.0).abs()
                < 1e-9
        );
        // ∝ 1/R.
        assert!(
            (azimuth_chirp_rate(200.0, lam, 2.0 * r) / azimuth_chirp_rate(200.0, lam, r) - 0.5)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn azimuth_history_matches_the_parabolic_phase() {
        let positions = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let (r, lam) = (1000.0, 0.03);
        let s = azimuth_history(r, 0.0, lam, &positions);
        for (i, &x) in positions.iter().enumerate()
        {
            assert!((s[i].mag() - 1.0).abs() < 1e-12);
            let expected = Complex::cis(-2.0 * PI * x * x / (lam * r));
            assert!((s[i].re - expected.re).abs() < 1e-12);
            assert!((s[i].im - expected.im).abs() < 1e-12);
        }
        // Offsetting the target shifts the parabola vertex.
        let s2 = azimuth_history(r, 1.0, lam, &positions);
        let e = Complex::cis(-2.0 * PI * 1.0 / (lam * r)); // at x=0, (0−1)² = 1
        assert!((s2[2].re - e.re).abs() < 1e-12 && (s2[2].im - e.im).abs() < 1e-12);
    }

    /// A well-sampled aperture whose per-sample phase step stays below π (no
    /// aliasing) so the phase-difference oracles are exact.
    fn aperture(lam: f64, r: f64, d: f64, dx: f64) -> Vec<f64> {
        let l = synthetic_aperture_length(lam, r, d);
        let n = (l / dx) as usize;
        (0..n).map(|i| -0.5 * l + i as f64 * dx).collect()
    }

    #[test]
    fn azimuth_history_is_a_linear_fm_chirp() {
        let (lam, r, d, dx) = (0.03, 10_000.0, 2.0, 0.5);
        let positions = aperture(lam, r, d, dx);
        let s = azimuth_history(r, 0.0, lam, &positions);
        // Consecutive phase increments; a chirp has a constant second difference.
        let dphi: Vec<f64> = (1..s.len())
            .map(|i| (s[i] * s[i - 1].conj()).phase())
            .collect();
        let expected = -4.0 * PI * dx * dx / (lam * r);
        for i in 1..dphi.len()
        {
            assert!(
                ((dphi[i] - dphi[i - 1]) - expected).abs() < 1e-9,
                "not a linear FM at {i}"
            );
        }
    }

    #[test]
    fn matched_filter_focuses_a_point_target_at_its_position() {
        let (lam, r, d, dx) = (0.03, 10_000.0, 2.0, 0.5);
        let positions = aperture(lam, r, d, dx);
        // Target offset by an integer number of samples along-track.
        let p = 10isize;
        let target_x = p as f64 * dx;
        let signal = azimuth_history(r, target_x, lam, &positions);
        let reference = azimuth_reference(r, lam, &positions);
        let corr = cross_correlate(&signal, &reference);
        // The azimuth chirp compresses to a peak at the target's along-track lag.
        assert_eq!(peak_lag(&corr, reference.len()), Some(p));
    }

    #[test]
    fn focusing_resolves_two_separated_targets() {
        let (lam, r, d, dx) = (0.03, 10_000.0, 2.0, 0.5);
        let positions = aperture(lam, r, d, dx);
        let half = 20isize; // ±10 m — many resolution cells (δ_az = D/2 = 1 m) apart
        let s1 = azimuth_history(r, -(half as f64) * dx, lam, &positions);
        let s2 = azimuth_history(r, (half as f64) * dx, lam, &positions);
        let signal: Vec<Complex> = s1.iter().zip(&s2).map(|(&a, &b)| a + b).collect();
        let reference = azimuth_reference(r, lam, &positions);
        let prof = focus_azimuth(&signal, &reference);
        let zero = reference.len() - 1; // lag 0
        let left = zero - half as usize;
        let right = zero + half as usize;
        // Two focused peaks with a valley between them ⇒ the targets are resolved.
        assert!(
            prof[left] > prof[zero] && prof[right] > prof[zero],
            "not resolved: L={} mid={} R={}",
            prof[left],
            prof[zero],
            prof[right]
        );
    }

    #[test]
    fn degenerate_inputs_are_safe() {
        assert!(azimuth_history(1000.0, 0.0, 0.03, &[]).is_empty());
        // Degenerate geometry ⇒ safe unit-phase, no NaN or division by zero.
        let g = azimuth_history(-1.0, 0.0, 0.03, &[0.0, 1.0]);
        assert_eq!(g.len(), 2);
        assert!(g.iter().all(|c| (c.mag() - 1.0).abs() < 1e-12));
        assert_eq!(synthetic_aperture_length(0.03, 1000.0, 0.0), 0.0);
        assert_eq!(azimuth_resolution(-1.0), 0.0);
        assert_eq!(azimuth_doppler_bandwidth(100.0, 0.0), 0.0);
        assert_eq!(azimuth_chirp_rate(100.0, 0.0, 1000.0), 0.0);
        assert!(focus_azimuth(&[], &[Complex::zero()]).is_empty());
    }
}
