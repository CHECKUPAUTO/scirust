//! Atmospheric optical turbulence and adaptive-optics metrics.
//!
//! The [`atmosphere`](crate::atmosphere) module models how the path *attenuates*
//! contrast (Beer–Lambert). But even a perfectly transmitting path *blurs* a
//! long-range EO/IR image, because refractive-index turbulence (strength set by
//! the structure constant `Cn²`) randomises the wavefront. This module supplies
//! the standard closed-form descriptors of that effect:
//!
//! - the **Fried parameter** `r₀` — the aperture diameter over which the wavefront
//!   stays coherent to ~1 rad, `r₀ = (0.423·k²·Cn²·L)^(−3/5)` for a plane wave;
//! - the **seeing angle** `≈ 0.98·λ/r₀`, the long-exposure blur that replaces the
//!   diffraction limit `λ/D` once `D > r₀`;
//! - the **Strehl ratio** `[1 + (D/r₀)^(5/3)]^(−6/5)`, the peak-intensity fraction
//!   an uncorrected aperture keeps;
//! - the **Greenwood frequency** `0.426·v/r₀`, the temporal bandwidth an adaptive
//!   optics loop must run at to track the turbulence;
//! - the number of correction **degrees of freedom** `(D/r₀)²`; and
//! - the **Rytov variance** `1.23·Cn²·k^(7/6)·L^(11/6)`, the weak-turbulence
//!   log-intensity scintillation that fades a laser beam.
//!
//! These are the quantities that size an EO/IR telescope, a laser designator, or
//! an adaptive-optics system against a given turbulence profile. Dependency-free.

use std::f64::consts::PI;

/// The optical **wavenumber** `k = 2π/λ`.
fn wavenumber(wavelength: f64) -> f64 {
    2.0 * PI / wavelength
}

/// The **Fried parameter** (atmospheric coherence length) `r₀ =
/// (0.423·k²·Cn²·L)^(−3/5)` for a plane wave through uniform turbulence of
/// structure constant `cn2` (m^(−2/3)) over `path_length` (m) at `wavelength`
/// (m). `r₀` is the aperture over which the wavefront stays coherent; a larger
/// telescope than `r₀` is turbulence-limited, not diffraction-limited. Returns
/// `+∞` (perfect coherence) for no turbulence or degenerate geometry.
pub fn fried_parameter(cn2: f64, wavelength: f64, path_length: f64) -> f64 {
    if cn2 <= 0.0 || wavelength <= 0.0 || path_length <= 0.0
    {
        return f64::INFINITY;
    }
    let k = wavenumber(wavelength);
    (0.423 * k * k * cn2 * path_length).powf(-3.0 / 5.0)
}

/// The long-exposure **seeing angle** `≈ 0.98·λ/r₀` (rad) — the turbulence-limited
/// angular blur (FWHM) that supersedes the diffraction limit `λ/D` once the
/// aperture exceeds `r₀`. Zero as `r₀ → ∞` (no turbulence ⇒ diffraction-limited);
/// `+∞` for a non-positive `r0`.
pub fn seeing_angle(wavelength: f64, r0: f64) -> f64 {
    if r0 <= 0.0
    {
        return f64::INFINITY;
    }
    if wavelength <= 0.0
    {
        return 0.0;
    }
    0.98 * wavelength / r0
}

/// The **Strehl ratio** `S = [1 + (D/r₀)^(5/3)]^(−6/5)` — the fraction of the
/// diffraction-limited peak intensity an *uncorrected* aperture of diameter
/// `aperture_diameter` retains through turbulence of coherence length `r0`.
/// `S → 1` for `D ≪ r₀` (diffraction-limited) and falls toward 0 as `D/r₀` grows.
/// `1` for a degenerate aperture, `0` for a non-positive `r0`.
pub fn strehl_ratio(aperture_diameter: f64, r0: f64) -> f64 {
    if aperture_diameter <= 0.0
    {
        return 1.0;
    }
    if r0 <= 0.0
    {
        return 0.0;
    }
    let x = (aperture_diameter / r0).powf(5.0 / 3.0);
    (1.0 + x).powf(-6.0 / 5.0)
}

/// The **Greenwood frequency** `f_G = 0.426·v/r₀` (Hz) — the temporal bandwidth an
/// adaptive-optics control loop must exceed to correct turbulence blowing across
/// the aperture at wind speed `wind_speed` (m/s), coherence length `r0`. Higher
/// for faster wind or smaller `r₀`. `+∞` for a non-positive `r0`.
pub fn greenwood_frequency(wind_speed: f64, r0: f64) -> f64 {
    if r0 <= 0.0
    {
        return f64::INFINITY;
    }
    0.426 * wind_speed.abs() / r0
}

/// The number of adaptive-optics **degrees of freedom** `≈ (D/r₀)²` — the count of
/// independent turbulence cells (and hence corrector actuators / wavefront-sensor
/// subapertures) across an aperture of diameter `aperture_diameter`. `+∞` for a
/// non-positive `r0`.
pub fn degrees_of_freedom(aperture_diameter: f64, r0: f64) -> f64 {
    if r0 <= 0.0
    {
        return f64::INFINITY;
    }
    let ratio = aperture_diameter / r0;
    ratio * ratio
}

/// The **Rytov variance** `σ_R² = 1.23·Cn²·k^(7/6)·L^(11/6)` — the weak-turbulence
/// variance of log-amplitude, i.e. the **scintillation** (intensity twinkling)
/// that fades a laser beam over a path of length `path_length` (m). `σ_R² ≳ 1`
/// signals the onset of strong-fluctuation (saturated) scintillation. Scales with
/// `Cn²`, steeply with path length, and with `k^(7/6)` (shorter wavelengths
/// scintillate more). `0` for degenerate inputs.
pub fn rytov_variance(cn2: f64, wavelength: f64, path_length: f64) -> f64 {
    if cn2 <= 0.0 || wavelength <= 0.0 || path_length <= 0.0
    {
        return 0.0;
    }
    let k = wavenumber(wavelength);
    1.23 * cn2 * k.powf(7.0 / 6.0) * path_length.powf(11.0 / 6.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fried_parameter_matches_the_closed_form() {
        let (cn2, lam, l) = (1e-14, 0.5e-6, 1000.0);
        let k = 2.0 * PI / lam;
        let expected = (0.423 * k * k * cn2 * l).powf(-3.0 / 5.0);
        let r0 = fried_parameter(cn2, lam, l);
        assert!((r0 - expected).abs() / expected < 1e-12);
        // A realistic ~2 cm coherence length at visible wavelengths, 1 km path.
        assert!(r0 > 0.001 && r0 < 1.0, "r0 = {r0}");
    }

    #[test]
    fn fried_parameter_scales_as_lambda_to_the_six_fifths() {
        let (cn2, l) = (1e-14, 1000.0);
        let r0a = fried_parameter(cn2, 0.5e-6, l);
        let r0b = fried_parameter(cn2, 1.0e-6, l);
        // r₀ ∝ λ^(6/5): doubling the wavelength scales r₀ by 2^(6/5).
        assert!((r0b / r0a - 2.0_f64.powf(6.0 / 5.0)).abs() < 1e-9);
        // Stronger turbulence and a longer path both shrink r₀.
        assert!(fried_parameter(2e-14, 0.5e-6, l) < r0a);
        assert!(fried_parameter(cn2, 0.5e-6, 2000.0) < r0a);
        // No turbulence ⇒ infinite coherence.
        assert!(fried_parameter(0.0, 0.5e-6, l).is_infinite());
    }

    #[test]
    fn seeing_angle_and_crossover_with_diffraction() {
        let lam = 0.5e-6;
        let r0 = 0.05; // 5 cm
        let seeing = seeing_angle(lam, r0);
        assert!((seeing - 0.98 * lam / r0).abs() < 1e-15);
        // A 1 m telescope (D ≫ r₀) is seeing-limited: its blur exceeds λ/D.
        let diffraction = lam / 1.0;
        assert!(seeing > diffraction);
        // No turbulence (r₀ → ∞) ⇒ seeing collapses to the diffraction limit (0).
        assert_eq!(seeing_angle(lam, f64::INFINITY), 0.0);
    }

    #[test]
    fn strehl_ratio_limits_and_monotonicity() {
        let r0 = 0.1;
        // D ≪ r₀ ⇒ nearly diffraction-limited (Strehl → 1).
        let s_small = strehl_ratio(0.001, r0);
        assert!(s_small > 0.99 && s_small <= 1.0);
        // D = r₀ ⇒ S = [1 + 1]^(−6/5) = 2^(−6/5).
        let s_eq = strehl_ratio(r0, r0);
        assert!((s_eq - 2.0_f64.powf(-6.0 / 5.0)).abs() < 1e-12);
        // Monotonically decreasing in D/r₀.
        assert!(strehl_ratio(0.2, r0) < strehl_ratio(0.1, r0));
        assert!(strehl_ratio(1.0, r0) < strehl_ratio(0.2, r0));
        // Bounded in (0, 1].
        for &d in &[0.01, 0.1, 0.5, 2.0]
        {
            let s = strehl_ratio(d, r0);
            assert!(s > 0.0 && s <= 1.0);
        }
        // No turbulence ⇒ Strehl 1.
        assert!((strehl_ratio(1.0, f64::INFINITY) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn greenwood_frequency_scaling() {
        let r0 = 0.1;
        let f1 = greenwood_frequency(10.0, r0);
        assert!((f1 - 0.426 * 10.0 / r0).abs() < 1e-12);
        // ∝ wind speed.
        assert!((greenwood_frequency(20.0, r0) / f1 - 2.0).abs() < 1e-12);
        // ∝ 1/r₀ — smaller coherence needs a faster AO loop.
        assert!(greenwood_frequency(10.0, 0.05) > f1);
    }

    #[test]
    fn rytov_variance_scaling() {
        let (cn2, lam, l) = (1e-15, 1.0e-6, 5000.0_f64);
        let k = 2.0 * PI / lam;
        let expected = 1.23 * cn2 * k.powf(7.0 / 6.0) * l.powf(11.0 / 6.0);
        let sr = rytov_variance(cn2, lam, l);
        assert!((sr - expected).abs() / expected < 1e-12);
        // ∝ Cn².
        assert!((rytov_variance(2e-15, lam, l) / sr - 2.0).abs() < 1e-9);
        // ∝ L^(11/6).
        assert!((rytov_variance(cn2, lam, 2.0 * l) / sr - 2.0_f64.powf(11.0 / 6.0)).abs() < 1e-9);
        // Shorter wavelength scintillates more (∝ k^(7/6) ∝ λ^(−7/6)).
        assert!(rytov_variance(cn2, 0.5e-6, l) > sr);
    }

    #[test]
    fn degrees_of_freedom_and_guards() {
        let r0 = 0.1;
        // (D/r₀)².
        assert!((degrees_of_freedom(1.0, r0) - 100.0).abs() < 1e-9);
        assert!((degrees_of_freedom(0.3, r0) - 9.0).abs() < 1e-9);
        // Degenerate / non-positive inputs are safe (no NaN).
        assert!(fried_parameter(1e-14, 0.5e-6, 0.0).is_infinite());
        assert_eq!(strehl_ratio(0.0, r0), 1.0);
        assert_eq!(strehl_ratio(1.0, 0.0), 0.0);
        assert_eq!(degrees_of_freedom(0.0, r0), 0.0);
        assert_eq!(rytov_variance(-1.0, 0.5e-6, 100.0), 0.0);
        assert!(seeing_angle(0.5e-6, 0.0).is_infinite());
    }
}
