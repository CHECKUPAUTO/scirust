//! Low-order Zernike wavefront aberrations and the Maréchal Strehl estimate.
//!
//! An imaging system's wavefront error — the deviation of the exit-pupil
//! wavefront from a perfect reference sphere — decomposes over the unit disk
//! into the orthogonal Zernike polynomials. This module gives the four
//! workhorse low-order modes in their Noll-normalized form (unit RMS over the
//! disk): **defocus**, **astigmatism**, **coma**, and **spherical** aberration,
//! each a function of the normalized pupil radius `r ∈ [0, 1]` and, for the
//! azimuthal modes, the pupil angle `θ` (rad). Because the modes are
//! orthonormal, the aggregate RMS wavefront error of a mixture is simply the
//! quadrature (root-sum-square) of the per-mode coefficients, each expressed in
//! waves. The [`strehl_marechal`] estimate then turns that RMS into a
//! peak-intensity ratio via the extended-Maréchal approximation
//! `S ≈ exp[−(2π·σ)²]`, whose `σ = λ/14` (≈ 0.0714 waves) marks the classic
//! Strehl ≈ 0.8 diffraction-limited criterion. Dependency-free.

use std::f64::consts::PI;

/// Noll-normalized **defocus** `Z(2, 0) = √3·(2r² − 1)` at normalized pupil
/// radius `r ∈ [0, 1]`. Rotationally symmetric; RMS 1 over the unit disk.
pub fn defocus(r: f64) -> f64 {
    3.0_f64.sqrt() * (2.0 * r * r - 1.0)
}

/// Noll-normalized **astigmatism** `Z(2, 2) = √6·r²·cos(2θ)` at pupil radius
/// `r ∈ [0, 1]` and angle `theta` (rad). RMS 1 over the unit disk.
pub fn astigmatism(r: f64, theta: f64) -> f64 {
    6.0_f64.sqrt() * r * r * (2.0 * theta).cos()
}

/// Noll-normalized **coma** `Z(3, 1) = √8·(3r³ − 2r)·cos(θ)` at pupil radius
/// `r ∈ [0, 1]` and angle `theta` (rad). RMS 1 over the unit disk.
pub fn coma(r: f64, theta: f64) -> f64 {
    8.0_f64.sqrt() * (3.0 * r * r * r - 2.0 * r) * theta.cos()
}

/// Noll-normalized **spherical** aberration `Z(4, 0) = √5·(6r⁴ − 6r² + 1)` at
/// pupil radius `r ∈ [0, 1]`. Rotationally symmetric; RMS 1 over the unit disk.
pub fn spherical(r: f64) -> f64 {
    5.0_f64.sqrt() * (6.0 * r.powi(4) - 6.0 * r * r + 1.0)
}

/// Aggregate **RMS wavefront error** of an orthonormal Zernike mixture: the
/// quadrature sum `√(Σ cᵢ²)` of the modal coefficients `coeffs` (each in waves).
/// Because the modes are orthonormal, cross terms vanish and the total RMS is
/// exactly this root-sum-square. Returns `0.0` for an empty coefficient slice.
pub fn rms_wavefront_error(coeffs: &[f64]) -> f64 {
    if coeffs.is_empty()
    {
        return 0.0;
    }
    coeffs.iter().map(|c| c * c).sum::<f64>().sqrt()
}

/// The extended-**Maréchal Strehl** approximation `S ≈ exp[−(2π·σ)²]`, mapping
/// an RMS wavefront error `rms_wfe_waves` (in waves) to the peak-intensity ratio
/// of the aberrated point spread against the diffraction-limited one. `S → 1` as
/// the error vanishes and decreases monotonically; `σ = λ/14` (≈ 0.0714 waves)
/// gives `S ≈ 0.8`, the conventional diffraction-limited threshold.
pub fn strehl_marechal(rms_wfe_waves: f64) -> f64 {
    (-(2.0 * PI * rms_wfe_waves).powi(2)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Numerically integrate `f(r, θ)` over the unit disk with the polar area
    /// element `r dr dθ` and divide by the disk area `π`, i.e. the disk mean of
    /// `f`. With `f = Z²` this is the mean-square (≈ 1 for a Noll mode); with
    /// `f = Zₐ·Z_b` it is the orthogonality inner product (≈ 0 for `a ≠ b`).
    fn disk_average<F>(f: F) -> f64
    where
        F: Fn(f64, f64) -> f64,
    {
        let (nr, nt) = (400usize, 400usize);
        let dr = 1.0 / nr as f64;
        let dt = 2.0 * PI / nt as f64;
        let mut sum = 0.0;
        for i in 0..nr
        {
            let r = (i as f64 + 0.5) * dr;
            for j in 0..nt
            {
                let theta = (j as f64 + 0.5) * dt;
                sum += f(r, theta) * r * dr * dt;
            }
        }
        sum / PI
    }

    #[test]
    fn closed_form_values_at_reference_points() {
        let s3 = 3.0_f64.sqrt();
        let s5 = 5.0_f64.sqrt();
        let s6 = 6.0_f64.sqrt();
        let s8 = 8.0_f64.sqrt();
        // Defocus: −√3 at center, +√3 at the rim, zero at r = 1/√2.
        assert!((defocus(0.0) + s3).abs() < 1e-12);
        assert!((defocus(1.0) - s3).abs() < 1e-12);
        assert!(defocus(0.5_f64.sqrt()).abs() < 1e-12);
        // Spherical: √5 at both center and rim, negative dip in between.
        assert!((spherical(0.0) - s5).abs() < 1e-12);
        assert!((spherical(1.0) - s5).abs() < 1e-12);
        assert!(spherical(0.5) < 0.0);
        // Astigmatism at the rim: √6·cos(2θ).
        assert!((astigmatism(1.0, 0.0) - s6).abs() < 1e-12);
        assert!((astigmatism(1.0, PI / 2.0) + s6).abs() < 1e-12);
        assert!(astigmatism(1.0, PI / 4.0).abs() < 1e-12);
        // Coma at the rim, θ = 0: √8·(3 − 2) = √8.
        assert!((coma(1.0, 0.0) - s8).abs() < 1e-12);
    }

    #[test]
    fn modes_are_noll_normalized() {
        // Each Noll mode has unit mean-square (RMS = 1) over the unit disk.
        assert!((disk_average(|r, _| defocus(r).powi(2)) - 1.0).abs() < 1e-2);
        assert!((disk_average(|r, _| spherical(r).powi(2)) - 1.0).abs() < 1e-2);
        assert!((disk_average(|r, t| astigmatism(r, t).powi(2)) - 1.0).abs() < 1e-2);
        assert!((disk_average(|r, t| coma(r, t).powi(2)) - 1.0).abs() < 1e-2);
    }

    #[test]
    fn distinct_modes_are_orthogonal() {
        // Radial orthogonality (both m = 0) and azimuthal orthogonality.
        assert!(disk_average(|r, _| defocus(r) * spherical(r)).abs() < 1e-2);
        assert!(disk_average(|r, t| defocus(r) * astigmatism(r, t)).abs() < 1e-2);
        assert!(disk_average(|r, t| astigmatism(r, t) * coma(r, t)).abs() < 1e-2);
    }

    #[test]
    fn rms_is_quadrature_sum_with_empty_guard() {
        // Orthonormality ⇒ the aggregate RMS is the root-sum-square of the modes.
        assert!((rms_wavefront_error(&[0.03, 0.04]) - 0.05).abs() < 1e-12);
        assert!((rms_wavefront_error(&[0.1, 0.1, 0.1]) - 3.0_f64.sqrt() / 10.0).abs() < 1e-12);
        // A single mode passes its coefficient through unchanged.
        assert!((rms_wavefront_error(&[0.07]) - 0.07).abs() < 1e-12);
        // Degenerate input: no modes ⇒ no error.
        assert_eq!(rms_wavefront_error(&[]), 0.0);
    }

    #[test]
    fn strehl_is_unity_at_zero_and_monotonically_decreasing() {
        assert!((strehl_marechal(0.0) - 1.0).abs() < 1e-15);
        let mut prev = strehl_marechal(0.0);
        for k in 1..=10
        {
            let s = strehl_marechal(k as f64 * 0.01);
            assert!(s < prev, "Strehl must decrease with wavefront error");
            assert!(s > 0.0 && s <= 1.0);
            prev = s;
        }
    }

    #[test]
    fn marechal_criterion_gives_diffraction_limited_strehl() {
        // The λ/14 RMS criterion yields the classic diffraction-limited Strehl:
        // exp[−(2π/14)²] ≈ 0.8176, the exponential form of the ≈ 0.8 rule.
        let s = strehl_marechal(1.0 / 14.0);
        assert!((s - 0.8176).abs() < 1e-3, "strehl at λ/14 = {s}");
        assert!(s > 0.8);
    }

    #[test]
    fn azimuthal_modes_follow_their_cosine_symmetry() {
        // Astigmatism has period π in θ; coma has period 2π and is antisymmetric.
        let (r, theta) = (0.6_f64, 0.37_f64);
        assert!((astigmatism(r, theta) - astigmatism(r, theta + PI)).abs() < 1e-12);
        assert!((coma(r, theta) + coma(r, theta + PI)).abs() < 1e-12);
        // Both azimuthal modes vanish identically at the pupil center.
        assert_eq!(astigmatism(0.0, theta), 0.0);
        assert_eq!(coma(0.0, theta), 0.0);
    }
}
