//! Convection correlations for external and natural (free) flows.
//!
//! Complements the internal-flow correlations of [`crate::heat_transfer`]
//! (Dittus–Boelter, laminar constants). All correlations return the mean
//! Nusselt number; `h = Nu·k/L` recovers the convection coefficient.
//! Validity ranges are enforced, not just documented.

use crate::error::{ThermoError, in_range, non_negative, positive};

/// Rayleigh number `Ra = g β ΔT L³/(ν α)` for natural convection.
///
/// * `gravity` — g \[m/s²\], > 0
/// * `beta` — volumetric expansion coefficient \[1/K\], > 0 (1/T for an
///   ideal gas)
/// * `delta_t` — |surface − fluid| temperature difference \[K\], ≥ 0
/// * `length` — characteristic length \[m\], > 0
/// * `kin_viscosity` — ν \[m²/s\], > 0
/// * `thermal_diffusivity` — α \[m²/s\], > 0
pub fn rayleigh(
    gravity: f64,
    beta: f64,
    delta_t: f64,
    length: f64,
    kin_viscosity: f64,
    thermal_diffusivity: f64,
) -> Result<f64, ThermoError> {
    positive("gravity", gravity)?;
    positive("beta", beta)?;
    non_negative("delta_t", delta_t)?;
    positive("length", length)?;
    positive("kin_viscosity", kin_viscosity)?;
    positive("thermal_diffusivity", thermal_diffusivity)?;
    Ok(gravity * beta * delta_t * length.powi(3) / (kin_viscosity * thermal_diffusivity))
}

/// Mean Nusselt number of a **laminar flat plate**,
/// `Nu = 0.664 Re_L^{1/2} Pr^{1/3}` (exact result of the Blasius layer
/// with the Pohlhausen 1/3-power Prandtl correction).
///
/// Validity (enforced): `Re_L ∈ (0, 5×10⁵]`, `Pr ≥ 0.6`.
pub fn nusselt_flat_plate_laminar(re_l: f64, pr: f64) -> Result<f64, ThermoError> {
    in_range("re_l", re_l, f64::MIN_POSITIVE, 5.0e5)?;
    in_range("pr", pr, 0.6, f64::MAX)?;
    Ok(0.664 * re_l.sqrt() * pr.powf(1.0 / 3.0))
}

/// Mean Nusselt number of a **mixed laminar–turbulent flat plate**
/// (transition at `Re = 5×10⁵`):
/// `Nu = (0.037 Re_L^{4/5} − 871) Pr^{1/3}`.
///
/// Validity (enforced): `Re_L ∈ [5×10⁵, 10⁸]`, `Pr ∈ [0.6, 60]`. By
/// construction it joins the laminar correlation continuously at the
/// transition Reynolds number.
pub fn nusselt_flat_plate_mixed(re_l: f64, pr: f64) -> Result<f64, ThermoError> {
    in_range("re_l", re_l, 5.0e5, 1.0e8)?;
    in_range("pr", pr, 0.6, 60.0)?;
    Ok((0.037 * re_l.powf(0.8) - 871.0) * pr.powf(1.0 / 3.0))
}

/// Mean Nusselt number of a **cylinder in cross-flow**,
/// **Churchill–Bernstein** correlation:
///
/// `Nu = 0.3 + 0.62 Re^{1/2} Pr^{1/3} / [1 + (0.4/Pr)^{2/3}]^{1/4}
///        · [1 + (Re/282000)^{5/8}]^{4/5}`
///
/// Single comprehensive equation for all `Re·Pr ≥ 0.2` (enforced).
pub fn nusselt_cylinder_churchill_bernstein(re: f64, pr: f64) -> Result<f64, ThermoError> {
    positive("re", re)?;
    positive("pr", pr)?;
    if re * pr < 0.2
    {
        return Err(ThermoError::OutOfRange {
            name: "re*pr",
            value: re * pr,
            min: 0.2,
            max: f64::MAX,
        });
    }
    let core =
        0.62 * re.sqrt() * pr.powf(1.0 / 3.0) / (1.0 + (0.4 / pr).powf(2.0 / 3.0)).powf(0.25);
    let high_re = (1.0 + (re / 282_000.0).powf(0.625)).powf(0.8);
    Ok(0.3 + core * high_re)
}

/// Mean Nusselt number of a **sphere**, **Ranz–Marshall** correlation
/// `Nu = 2 + 0.6 Re^{1/2} Pr^{1/3}`, which tends to the exact pure-
/// conduction limit `Nu = 2` as `Re → 0`.
///
/// Validity (enforced): `Re ∈ [0, 5×10⁴]`, `Pr ∈ [0.6, 380]`.
pub fn nusselt_sphere_ranz_marshall(re: f64, pr: f64) -> Result<f64, ThermoError> {
    in_range("re", re, 0.0, 5.0e4)?;
    in_range("pr", pr, 0.6, 380.0)?;
    Ok(2.0 + 0.6 * re.sqrt() * pr.powf(1.0 / 3.0))
}

/// Mean Nusselt number of natural convection on a **vertical plate**,
/// **Churchill–Chu** correlation (valid over the entire Ra range):
///
/// `Nu = { 0.825 + 0.387 Ra^{1/6} / [1 + (0.492/Pr)^{9/16}]^{8/27} }²`
///
/// Validity (enforced): `Ra ∈ (0, 10¹²]`, `Pr > 0`.
pub fn nusselt_vertical_plate_churchill_chu(ra: f64, pr: f64) -> Result<f64, ThermoError> {
    in_range("ra", ra, f64::MIN_POSITIVE, 1.0e12)?;
    positive("pr", pr)?;
    let f = (1.0 + (0.492 / pr).powf(9.0 / 16.0)).powf(8.0 / 27.0);
    let root = 0.825 + 0.387 * ra.powf(1.0 / 6.0) / f;
    Ok(root * root)
}

/// Mean Nusselt number of natural convection around a long
/// **horizontal cylinder**, **Churchill–Chu** correlation:
///
/// `Nu = { 0.60 + 0.387 Ra^{1/6} / [1 + (0.559/Pr)^{9/16}]^{8/27} }²`
///
/// Validity (enforced): `Ra ∈ (0, 10¹²]`, `Pr > 0`.
pub fn nusselt_horizontal_cylinder_churchill_chu(ra: f64, pr: f64) -> Result<f64, ThermoError> {
    in_range("ra", ra, f64::MIN_POSITIVE, 1.0e12)?;
    positive("pr", pr)?;
    let f = (1.0 + (0.559 / pr).powf(9.0 / 16.0)).powf(8.0 / 27.0);
    let root = 0.60 + 0.387 * ra.powf(1.0 / 6.0) / f;
    Ok(root * root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_plate_laminar_is_colburn_analogy() {
        // Nu = (C_f/2) Re Pr^{1/3} with the Blasius mean C_f = 1.328/√Re
        // gives exactly the 0.664 coefficient.
        let (re, pr) = (2.0e5, 0.9);
        let nu = nusselt_flat_plate_laminar(re, pr).unwrap();
        let cf_mean = 1.328 / re.sqrt();
        assert!((nu - 0.5 * cf_mean * re * pr.powf(1.0 / 3.0)).abs() / nu < 1e-12);
    }

    #[test]
    fn flat_plate_mixed_joins_laminar_at_transition() {
        // The 871 constant is built so both formulas agree at Re = 5e5.
        let pr = 0.7;
        let lam = nusselt_flat_plate_laminar(5.0e5, pr).unwrap();
        let mix = nusselt_flat_plate_mixed(5.0e5, pr).unwrap();
        assert!((lam - mix).abs() / lam < 2e-3, "lam = {lam}, mix = {mix}");
    }

    #[test]
    fn churchill_bernstein_known_behaviour() {
        // Low-Re limit tends to the 0.3 constant.
        let low = nusselt_cylinder_churchill_bernstein(0.3, 0.7).unwrap();
        assert!(low > 0.3 && low < 1.2, "Nu = {low}");
        // Spot value at Re = 1e4, Pr = 0.7 (hand-evaluated): Nu ≈ 53.3.
        let nu = nusselt_cylinder_churchill_bernstein(1.0e4, 0.7).unwrap();
        assert!((nu - 53.3).abs() < 1.0, "Nu = {nu}");
        // Monotone in Re.
        assert!(
            nusselt_cylinder_churchill_bernstein(2.0e4, 0.7).unwrap() > nu,
            "not monotone in Re"
        );
    }

    #[test]
    fn sphere_conduction_limit_is_two() {
        // Exact analytic limit of a sphere in a still medium.
        assert!((nusselt_sphere_ranz_marshall(0.0, 0.7).unwrap() - 2.0).abs() < 1e-15);
        assert!(nusselt_sphere_ranz_marshall(100.0, 0.7).unwrap() > 2.0);
    }

    #[test]
    fn churchill_chu_vertical_plate_spot_value() {
        // Air (Pr = 0.7) at Ra = 1e9 (edge of laminar free convection):
        // Nu ≈ 123 (hand-evaluated; Incropera quotes ≈ 120–130).
        let nu = nusselt_vertical_plate_churchill_chu(1.0e9, 0.7).unwrap();
        assert!((110.0..135.0).contains(&nu), "Nu = {nu}");
        // Monotone in Ra.
        assert!(nusselt_vertical_plate_churchill_chu(1.0e10, 0.7).unwrap() > nu);
    }

    #[test]
    fn horizontal_cylinder_below_vertical_plate() {
        // Same Ra, Pr: the cylinder correlation's smaller constant gives
        // a smaller Nu — a fixed cross-relation of the two formulas.
        let (ra, pr) = (1.0e7, 0.7);
        let cyl = nusselt_horizontal_cylinder_churchill_chu(ra, pr).unwrap();
        let plate = nusselt_vertical_plate_churchill_chu(ra, pr).unwrap();
        assert!(cyl < plate, "cyl = {cyl}, plate = {plate}");
    }

    #[test]
    fn rayleigh_ideal_gas_case() {
        // Air at 300 K, ΔT = 20 K, L = 0.5 m, ν = 1.6e-5, α = 2.25e-5:
        // Ra = 9.81·(1/300)·20·0.125/(1.6e-5·2.25e-5) ≈ 2.27e8.
        let ra = rayleigh(9.81, 1.0 / 300.0, 20.0, 0.5, 1.6e-5, 2.25e-5).unwrap();
        assert!((ra - 2.27e8).abs() / 2.27e8 < 5e-3, "Ra = {ra}");
    }

    #[test]
    fn rejects_out_of_validity() {
        assert!(nusselt_flat_plate_laminar(1.0e6, 0.7).is_err()); // turbulent
        assert!(nusselt_flat_plate_mixed(1.0e5, 0.7).is_err()); // laminar
        assert!(nusselt_cylinder_churchill_bernstein(0.1, 0.7).is_err()); // Re·Pr < 0.2
        assert!(nusselt_sphere_ranz_marshall(1.0e5, 0.7).is_err());
        assert!(nusselt_vertical_plate_churchill_chu(1.0e13, 0.7).is_err());
        assert!(rayleigh(9.81, -1.0, 20.0, 0.5, 1.6e-5, 2.25e-5).is_err());
    }
}
