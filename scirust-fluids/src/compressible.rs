//! Compressible flow of a calorically perfect gas: isentropic relations
//! and normal-shock jump conditions.
//!
//! All ratio functions take the Mach number and the heat-capacity ratio
//! `γ > 1`; they are exact closed forms (NACA Report 1135 conventions).

use crate::error::{FluidsError, in_range, non_negative, positive};

fn check_gamma(gamma: f64) -> Result<f64, FluidsError> {
    in_range("gamma", gamma, 1.0 + 1e-12, 2.0)
}

/// Speed of sound of a perfect gas, `a = √(γ R T)` \[m/s\].
///
/// * `r` — specific gas constant \[J/(kg·K)\], > 0
/// * `temperature` — static temperature \[K\], > 0
pub fn speed_of_sound(gamma: f64, r: f64, temperature: f64) -> Result<f64, FluidsError> {
    check_gamma(gamma)?;
    positive("r", r)?;
    positive("temperature", temperature)?;
    Ok((gamma * r * temperature).sqrt())
}

/// Isentropic stagnation-to-static temperature ratio
/// `T₀/T = 1 + (γ−1)/2 · M²`.
pub fn isentropic_temperature_ratio(mach: f64, gamma: f64) -> Result<f64, FluidsError> {
    non_negative("mach", mach)?;
    check_gamma(gamma)?;
    Ok(1.0 + 0.5 * (gamma - 1.0) * mach * mach)
}

/// Isentropic stagnation-to-static pressure ratio
/// `p₀/p = (T₀/T)^{γ/(γ−1)}`.
pub fn isentropic_pressure_ratio(mach: f64, gamma: f64) -> Result<f64, FluidsError> {
    let tr = isentropic_temperature_ratio(mach, gamma)?;
    Ok(tr.powf(gamma / (gamma - 1.0)))
}

/// Isentropic stagnation-to-static density ratio
/// `ρ₀/ρ = (T₀/T)^{1/(γ−1)}`.
pub fn isentropic_density_ratio(mach: f64, gamma: f64) -> Result<f64, FluidsError> {
    let tr = isentropic_temperature_ratio(mach, gamma)?;
    Ok(tr.powf(1.0 / (gamma - 1.0)))
}

/// Isentropic area ratio to the sonic throat, `A/A*`, for `M > 0`:
///
/// `A/A* = (1/M) [ (2/(γ+1)) (1 + (γ−1)/2 · M²) ]^{(γ+1)/(2(γ−1))}`
pub fn area_ratio(mach: f64, gamma: f64) -> Result<f64, FluidsError> {
    positive("mach", mach)?;
    check_gamma(gamma)?;
    let gm1 = gamma - 1.0;
    let gp1 = gamma + 1.0;
    let t = 2.0 / gp1 * (1.0 + 0.5 * gm1 * mach * mach);
    Ok(t.powf(gp1 / (2.0 * gm1)) / mach)
}

/// Mach number `M₂` downstream of a normal shock (`M₁ > 1`):
///
/// `M₂² = ((γ−1) M₁² + 2) / (2 γ M₁² − (γ−1))`
pub fn normal_shock_mach(m1: f64, gamma: f64) -> Result<f64, FluidsError> {
    in_range("m1", m1, 1.0, f64::MAX)?;
    check_gamma(gamma)?;
    let m1sq = m1 * m1;
    let m2sq = ((gamma - 1.0) * m1sq + 2.0) / (2.0 * gamma * m1sq - (gamma - 1.0));
    Ok(m2sq.sqrt())
}

/// Static pressure ratio `p₂/p₁` across a normal shock:
/// `p₂/p₁ = (2 γ M₁² − (γ−1)) / (γ+1)`.
pub fn normal_shock_pressure_ratio(m1: f64, gamma: f64) -> Result<f64, FluidsError> {
    in_range("m1", m1, 1.0, f64::MAX)?;
    check_gamma(gamma)?;
    Ok((2.0 * gamma * m1 * m1 - (gamma - 1.0)) / (gamma + 1.0))
}

/// Static density ratio `ρ₂/ρ₁` across a normal shock:
/// `ρ₂/ρ₁ = (γ+1) M₁² / ((γ−1) M₁² + 2)`.
pub fn normal_shock_density_ratio(m1: f64, gamma: f64) -> Result<f64, FluidsError> {
    in_range("m1", m1, 1.0, f64::MAX)?;
    check_gamma(gamma)?;
    let m1sq = m1 * m1;
    Ok((gamma + 1.0) * m1sq / ((gamma - 1.0) * m1sq + 2.0))
}

/// Static temperature ratio `T₂/T₁` across a normal shock
/// (= pressure ratio / density ratio, by the perfect-gas law).
pub fn normal_shock_temperature_ratio(m1: f64, gamma: f64) -> Result<f64, FluidsError> {
    Ok(normal_shock_pressure_ratio(m1, gamma)? / normal_shock_density_ratio(m1, gamma)?)
}

/// Stagnation-pressure ratio `p₀₂/p₀₁` across a normal shock (the
/// shock's irreversibility: 1 at M₁ = 1, decreasing for stronger shocks):
///
/// `p₀₂/p₀₁ = [ (γ+1)M₁² / ((γ−1)M₁²+2) ]^{γ/(γ−1)} · [ (γ+1) / (2γM₁²−(γ−1)) ]^{1/(γ−1)}`
pub fn normal_shock_stagnation_pressure_ratio(m1: f64, gamma: f64) -> Result<f64, FluidsError> {
    in_range("m1", m1, 1.0, f64::MAX)?;
    check_gamma(gamma)?;
    let a = normal_shock_density_ratio(m1, gamma)?.powf(gamma / (gamma - 1.0));
    let b = (1.0 / normal_shock_pressure_ratio(m1, gamma)?).powf(1.0 / (gamma - 1.0));
    Ok(a * b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sea_level_speed_of_sound() {
        // ISA sea level: γ = 1.4, R = 287.052874, T = 288.15 K → 340.29 m/s.
        let a = speed_of_sound(1.4, 287.052_874, 288.15).unwrap();
        assert!((a - 340.294).abs() < 5e-3, "a = {a}");
    }

    #[test]
    fn isentropic_ratios_at_mach_1_and_2() {
        // NACA 1135, γ = 1.4. M = 1: T₀/T = 1.2, p₀/p = 1.8929.
        assert!((isentropic_temperature_ratio(1.0, 1.4).unwrap() - 1.2).abs() < 1e-12);
        assert!((isentropic_pressure_ratio(1.0, 1.4).unwrap() - 1.892_929).abs() < 1e-5);
        // M = 2: T₀/T = 1.8, p₀/p = 7.8245, ρ₀/ρ = 4.3469.
        assert!((isentropic_temperature_ratio(2.0, 1.4).unwrap() - 1.8).abs() < 1e-12);
        assert!((isentropic_pressure_ratio(2.0, 1.4).unwrap() - 7.824_45).abs() < 1e-4);
        assert!((isentropic_density_ratio(2.0, 1.4).unwrap() - 4.346_92).abs() < 1e-4);
    }

    #[test]
    fn area_ratio_exact_values() {
        // A/A* = 1 at M = 1 exactly; A/A* = 1.6875 at M = 2, γ = 1.4 (exact
        // rational value); ≈ 2.9635 at M = 0.2.
        assert!((area_ratio(1.0, 1.4).unwrap() - 1.0).abs() < 1e-12);
        assert!((area_ratio(2.0, 1.4).unwrap() - 1.6875).abs() < 1e-10);
        assert!((area_ratio(0.2, 1.4).unwrap() - 2.9635).abs() < 1e-4);
    }

    #[test]
    fn normal_shock_mach_2_exact_fractions() {
        // γ = 1.4, M₁ = 2 gives exact rationals:
        // M₂ = √(1/3), p₂/p₁ = 4.5, ρ₂/ρ₁ = 8/3, T₂/T₁ = 1.6875.
        assert!((normal_shock_mach(2.0, 1.4).unwrap() - (1.0f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((normal_shock_pressure_ratio(2.0, 1.4).unwrap() - 4.5).abs() < 1e-12);
        assert!((normal_shock_density_ratio(2.0, 1.4).unwrap() - 8.0 / 3.0).abs() < 1e-12);
        assert!((normal_shock_temperature_ratio(2.0, 1.4).unwrap() - 1.6875).abs() < 1e-12);
        // NACA 1135 table: p₀₂/p₀₁ ≈ 0.72087 at M₁ = 2.
        let s = normal_shock_stagnation_pressure_ratio(2.0, 1.4).unwrap();
        assert!((s - 0.720_87).abs() < 2e-4, "p02/p01 = {s}");
    }

    #[test]
    fn shock_becomes_degenerate_at_mach_1() {
        for f in [
            normal_shock_mach,
            normal_shock_pressure_ratio,
            normal_shock_density_ratio,
            normal_shock_temperature_ratio,
            normal_shock_stagnation_pressure_ratio,
        ]
        {
            assert!((f(1.0, 1.4).unwrap() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn rejects_subsonic_shock_and_bad_gamma() {
        assert!(normal_shock_mach(0.8, 1.4).is_err());
        assert!(isentropic_pressure_ratio(1.0, 1.0).is_err());
        assert!(speed_of_sound(1.4, -287.0, 300.0).is_err());
    }
}
