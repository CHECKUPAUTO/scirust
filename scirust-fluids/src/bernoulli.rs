//! Bernoulli-equation relations for steady incompressible flow, and the
//! classical flow-measurement devices derived from it (Pitot tube,
//! Venturi meter, orifice).

use crate::error::{FluidsError, finite, in_range, non_negative, positive};

/// Dynamic pressure `q = ρ V²/2` \[Pa\].
pub fn dynamic_pressure(density: f64, speed: f64) -> Result<f64, FluidsError> {
    positive("density", density)?;
    non_negative("speed", speed)?;
    Ok(0.5 * density * speed * speed)
}

/// Stagnation (total) pressure `p₀ = p + ρ V²/2` \[Pa\] of an
/// incompressible stream.
pub fn stagnation_pressure(
    static_pressure: f64,
    density: f64,
    speed: f64,
) -> Result<f64, FluidsError> {
    finite("static_pressure", static_pressure)?;
    Ok(static_pressure + dynamic_pressure(density, speed)?)
}

/// Speed from a Pitot-static reading, `V = √(2 Δp / ρ)`.
///
/// * `delta_p` — stagnation minus static pressure \[Pa\], ≥ 0
pub fn pitot_speed(delta_p: f64, density: f64) -> Result<f64, FluidsError> {
    non_negative("delta_p", delta_p)?;
    positive("density", density)?;
    Ok((2.0 * delta_p / density).sqrt())
}

/// Torricelli efflux speed from a free surface, `V = √(2 g h)`.
pub fn torricelli_speed(gravity: f64, head: f64) -> Result<f64, FluidsError> {
    positive("gravity", gravity)?;
    non_negative("head", head)?;
    Ok((2.0 * gravity * head).sqrt())
}

/// Static pressure at station 2 from the Bernoulli equation between two
/// stations of the same streamline (steady, incompressible, lossless):
///
/// `p₂ = p₁ + ρ (V₁² − V₂²)/2 + ρ g (z₁ − z₂)` \[Pa\].
#[allow(clippy::too_many_arguments)]
pub fn bernoulli_pressure(
    p1: f64,
    v1: f64,
    z1: f64,
    v2: f64,
    z2: f64,
    density: f64,
    gravity: f64,
) -> Result<f64, FluidsError> {
    finite("p1", p1)?;
    non_negative("v1", v1)?;
    finite("z1", z1)?;
    non_negative("v2", v2)?;
    finite("z2", z2)?;
    positive("density", density)?;
    positive("gravity", gravity)?;
    Ok(p1 + 0.5 * density * (v1 * v1 - v2 * v2) + density * gravity * (z1 - z2))
}

/// Volumetric flow through a **Venturi** meter \[m³/s\]:
///
/// `Q = C_d A₂ √( 2 Δp / (ρ (1 − (A₂/A₁)²)) )`
///
/// * `area_inlet` — upstream area A₁ \[m²\], must exceed `area_throat`
/// * `area_throat` — throat area A₂ \[m²\], > 0
/// * `delta_p` — upstream minus throat static pressure \[Pa\], ≥ 0
/// * `discharge_coeff` — C_d, in `(0, 1]` (≈ 0.95–0.99 for a Venturi)
pub fn venturi_flow(
    area_inlet: f64,
    area_throat: f64,
    delta_p: f64,
    density: f64,
    discharge_coeff: f64,
) -> Result<f64, FluidsError> {
    positive("area_inlet", area_inlet)?;
    positive("area_throat", area_throat)?;
    non_negative("delta_p", delta_p)?;
    positive("density", density)?;
    in_range("discharge_coeff", discharge_coeff, f64::MIN_POSITIVE, 1.0)?;
    if area_throat >= area_inlet
    {
        return Err(FluidsError::OutOfRange {
            name: "area_throat",
            value: area_throat,
            min: 0.0,
            max: area_inlet,
        });
    }
    let beta2 = area_throat / area_inlet;
    let q =
        discharge_coeff * area_throat * (2.0 * delta_p / (density * (1.0 - beta2 * beta2))).sqrt();
    Ok(q)
}

/// Volumetric flow through a sharp-edged **orifice** discharging from a
/// large reservoir (upstream velocity neglected) \[m³/s\]:
///
/// `Q = C_d A √(2 Δp / ρ)`
///
/// * `discharge_coeff` — C_d, in `(0, 1]` (≈ 0.61 for a sharp edge)
pub fn orifice_flow(
    area: f64,
    delta_p: f64,
    density: f64,
    discharge_coeff: f64,
) -> Result<f64, FluidsError> {
    positive("area", area)?;
    non_negative("delta_p", delta_p)?;
    positive("density", density)?;
    in_range("discharge_coeff", discharge_coeff, f64::MIN_POSITIVE, 1.0)?;
    Ok(discharge_coeff * area * (2.0 * delta_p / density).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stagnation_minus_static_is_dynamic() {
        let p0 = stagnation_pressure(101_325.0, 1.225, 50.0).unwrap();
        let q = dynamic_pressure(1.225, 50.0).unwrap();
        assert!((p0 - 101_325.0 - q).abs() < 1e-9);
        // and the Pitot inversion recovers the speed
        let v = pitot_speed(q, 1.225).unwrap();
        assert!((v - 50.0).abs() < 1e-12);
    }

    #[test]
    fn torricelli_5m_head() {
        // √(2·9.81·5) = 9.9045 m/s.
        let v = torricelli_speed(9.81, 5.0).unwrap();
        assert!((v - 9.904544).abs() < 1e-5);
    }

    #[test]
    fn bernoulli_recovers_hydrostatics() {
        // No flow: Δp = ρ g Δz.
        let p2 = bernoulli_pressure(100_000.0, 0.0, 10.0, 0.0, 0.0, 1000.0, 9.81).unwrap();
        assert!((p2 - (100_000.0 + 1000.0 * 9.81 * 10.0)).abs() < 1e-9);
    }

    #[test]
    fn bernoulli_conserves_total_head() {
        // Horizontal nozzle: 2 m/s → 8 m/s, water.
        let p2 = bernoulli_pressure(200_000.0, 2.0, 0.0, 8.0, 0.0, 1000.0, 9.81).unwrap();
        assert!((p2 - (200_000.0 + 500.0 * (4.0 - 64.0))).abs() < 1e-9);
    }

    #[test]
    fn venturi_textbook_case() {
        // A₁ = 0.01 m², A₂ = 0.005 m², Δp = 1000 Pa, water, C_d = 1:
        // Q = 0.005·√(2000/(1000·0.75)) = 8.16497e-3 m³/s.
        let q = venturi_flow(0.01, 0.005, 1000.0, 1000.0, 1.0).unwrap();
        assert!((q - 8.164966e-3).abs() < 1e-8, "Q = {q}");
    }

    #[test]
    fn orifice_flow_scales_with_cd() {
        let q1 = orifice_flow(1e-4, 5000.0, 1000.0, 1.0).unwrap();
        let q061 = orifice_flow(1e-4, 5000.0, 1000.0, 0.61).unwrap();
        assert!((q061 - 0.61 * q1).abs() < 1e-15);
    }

    #[test]
    fn venturi_rejects_throat_wider_than_inlet() {
        assert!(venturi_flow(0.005, 0.01, 1000.0, 1000.0, 1.0).is_err());
        assert!(venturi_flow(0.01, 0.005, 1000.0, 1000.0, 1.5).is_err());
    }
}
