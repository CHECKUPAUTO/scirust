//! Atmospheric transmission and the EO/IR range budget.
//!
//! Radiometry ([`radiometry`](crate::radiometry)) and sensitivity (NETD) fix how
//! much target–background contrast a sensor needs at its *aperture*. The
//! atmosphere between target and sensor attenuates that contrast, so what the
//! sensor can actually detect depends on the path. This module models the path
//! with the **Beer–Lambert law** — transmittance `τ = e^{−αR}` for an extinction
//! coefficient `α` over range `R` — plus the standard tie-ins: extinction from
//! meteorological **visibility** (Koschmieder), the **contrast-transmission** law
//! that carries an intrinsic contrast to its apparent value at range, and the
//! **range budget** that turns an intrinsic NETD into the target ΔT required to
//! see through the path at a given range.
//!
//! [`radiometry`]: crate::radiometry
//! Dependency-free.

/// Koschmieder constant `−ln(0.02)`: the 2 % contrast threshold linking
/// meteorological visibility to the extinction coefficient.
const KOSCHMIEDER: f64 = 3.912_023_005_428_146;

/// **Beer–Lambert** path transmittance `τ = e^{−α·R}` for extinction coefficient
/// `extinction` (per metre) over `path_length` (m). Negative inputs are clamped
/// to zero, giving unit transmittance.
pub fn transmittance(extinction: f64, path_length: f64) -> f64 {
    let tau = (-extinction.max(0.0) * path_length.max(0.0)).exp();
    tau.clamp(0.0, 1.0)
}

/// The **optical depth** `τ_opt = α·R` of a path — the dimensionless exponent of
/// the Beer–Lambert law, so that `transmittance = e^{−optical_depth}`.
pub fn optical_depth(extinction: f64, path_length: f64) -> f64 {
    extinction.max(0.0) * path_length.max(0.0)
}

/// The total extinction coefficient of an absorbing-and-scattering atmosphere,
/// `α = α_abs + α_scat` — extinction is additive over independent loss
/// mechanisms.
pub fn extinction(absorption: f64, scattering: f64) -> f64 {
    absorption + scattering
}

/// The extinction coefficient (per metre) implied by a meteorological
/// `visibility` (m) via **Koschmieder's law** `α = 3.912 / V` — the range at
/// which a black target's contrast falls to the 2 % threshold. `None` for a
/// non-positive visibility.
pub fn extinction_from_visibility(visibility: f64) -> Option<f64> {
    if visibility <= 0.0
    {
        return None;
    }
    Some(KOSCHMIEDER / visibility)
}

/// The extinction coefficient recovered from a measured `transmittance` over
/// `path_length`, `α = −ln(τ)/R`. `None` for a non-positive path or a
/// transmittance outside `(0, 1]`.
pub fn extinction_from_transmittance(transmittance: f64, path_length: f64) -> Option<f64> {
    if path_length <= 0.0 || transmittance <= 0.0 || transmittance > 1.0
    {
        return None;
    }
    Some(-transmittance.ln() / path_length)
}

/// The **apparent contrast** at the sensor after atmospheric attenuation,
/// Koschmieder's contrast-transmission law `C = C₀·e^{−αR}` — an intrinsic
/// target contrast `intrinsic_contrast` seen through `extinction` over
/// `path_length`.
pub fn apparent_contrast(intrinsic_contrast: f64, extinction: f64, path_length: f64) -> f64 {
    intrinsic_contrast * transmittance(extinction, path_length)
}

/// The **range budget**: the intrinsic target–background ΔT required to detect a
/// target through the atmosphere at `path_length`, given the sensor's intrinsic
/// `netd`. Since the path attenuates the thermal signal by `τ`, the target must
/// exceed `NETD / τ` at the aperture — so the required ΔT grows with range.
/// `f64::INFINITY` if the path is fully opaque.
pub fn required_delta_t(netd: f64, extinction: f64, path_length: f64) -> f64 {
    let tau = transmittance(extinction, path_length);
    if tau <= 0.0
    {
        return f64::INFINITY;
    }
    netd / tau
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transmittance_is_unity_at_zero_and_decays_monotonically() {
        assert!((transmittance(1e-4, 0.0) - 1.0).abs() < 1e-12);
        assert!((transmittance(0.0, 5000.0) - 1.0).abs() < 1e-12); // clear path
        let near = transmittance(1e-3, 1000.0);
        let far = transmittance(1e-3, 5000.0);
        assert!(far < near && near < 1.0, "{near} vs {far}");
        // Closed form τ = e^{−αR}.
        assert!((near - (-1.0_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn beer_lambert_is_multiplicative_over_segments() {
        // τ(α, R₁+R₂) = τ(α, R₁)·τ(α, R₂): transmittance composes along the path.
        let a = 4e-4;
        let (r1, r2) = (1200.0, 800.0);
        let whole = transmittance(a, r1 + r2);
        let split = transmittance(a, r1) * transmittance(a, r2);
        assert!((whole - split).abs() < 1e-12, "{whole} vs {split}");
    }

    #[test]
    fn optical_depth_and_transmittance_are_inverse() {
        let (a, r) = (7e-4, 2500.0);
        let od = optical_depth(a, r);
        assert!((od - a * r).abs() < 1e-12);
        assert!((transmittance(a, r) - (-od).exp()).abs() < 1e-12);
        // Recovering the extinction from the transmittance round-trips.
        let recovered = extinction_from_transmittance(transmittance(a, r), r).unwrap();
        assert!((recovered - a).abs() < 1e-12, "{recovered} vs {a}");
    }

    #[test]
    fn koschmieder_visibility_hits_the_two_percent_threshold() {
        // At the meteorological visibility, transmittance is the 2 % contrast
        // threshold by definition.
        let v = 10_000.0;
        let a = extinction_from_visibility(v).unwrap();
        assert!(
            (transmittance(a, v) - 0.02).abs() < 1e-6,
            "{}",
            transmittance(a, v)
        );
        assert!(extinction_from_visibility(0.0).is_none());
    }

    #[test]
    fn additive_extinction_and_apparent_contrast() {
        // Extinction sums; apparent contrast follows C = C₀·e^{−αR}.
        let a = extinction(3e-4, 5e-4);
        assert!((a - 8e-4).abs() < 1e-12);
        let c = apparent_contrast(1.0, a, 1250.0);
        assert!((c - (-1.0_f64).exp()).abs() < 1e-12, "{c}");
        assert!(c < 1.0);
    }

    #[test]
    fn required_delta_t_grows_with_range() {
        // The range budget: needed ΔT = NETD/τ, rising with range as τ falls.
        let (netd, a) = (0.03, 5e-4);
        let near = required_delta_t(netd, a, 0.0);
        let far = required_delta_t(netd, a, 4000.0);
        assert!(
            (near - netd).abs() < 1e-12,
            "at zero range, needed ΔT = NETD"
        );
        assert!(far > near, "{near} -> {far}");
        // Explicitly NETD / τ.
        assert!((far - netd / transmittance(a, 4000.0)).abs() < 1e-12);
    }
}
