//! Steady-state heat transfer: conduction resistances, convection,
//! radiation, log-mean temperature difference and effectiveness-NTU
//! heat-exchanger relations.

use crate::error::{ThermoError, finite, in_range, non_negative, positive};

/// Stefan–Boltzmann constant σ \[W/(m²·K⁴)\] (CODATA 2018, exact).
pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

/// Conductive thermal resistance of a **plane wall**,
/// `R = L/(k A)` \[K/W\].
pub fn resistance_plane(conductivity: f64, area: f64, thickness: f64) -> Result<f64, ThermoError> {
    positive("conductivity", conductivity)?;
    positive("area", area)?;
    positive("thickness", thickness)?;
    Ok(thickness / (conductivity * area))
}

/// Conductive thermal resistance of a **cylindrical shell** of length L,
/// `R = ln(r_out/r_in)/(2π k L)` \[K/W\].
pub fn resistance_cylinder(
    conductivity: f64,
    length: f64,
    r_in: f64,
    r_out: f64,
) -> Result<f64, ThermoError> {
    positive("conductivity", conductivity)?;
    positive("length", length)?;
    positive("r_in", r_in)?;
    positive("r_out", r_out)?;
    if r_out <= r_in
    {
        return Err(ThermoError::OutOfRange {
            name: "r_out",
            value: r_out,
            min: r_in,
            max: f64::MAX,
        });
    }
    Ok((r_out / r_in).ln() / (2.0 * std::f64::consts::PI * conductivity * length))
}

/// Convective thermal resistance of a surface, `R = 1/(h A)` \[K/W\].
pub fn resistance_convection(h: f64, area: f64) -> Result<f64, ThermoError> {
    positive("h", h)?;
    positive("area", area)?;
    Ok(1.0 / (h * area))
}

/// Heat flow through resistances **in series**,
/// `q = ΔT / ΣR` \[W\] (ΔT may be negative: heat then flows the other way).
pub fn heat_flow_series(resistances: &[f64], delta_t: f64) -> Result<f64, ThermoError> {
    finite("delta_t", delta_t)?;
    if resistances.is_empty()
    {
        return Err(ThermoError::NonPositive {
            name: "resistances.len()",
            value: 0.0,
        });
    }
    let mut total = 0.0;
    for &r in resistances
    {
        positive("resistance", r)?;
        total += r;
    }
    Ok(delta_t / total)
}

/// Net radiative exchange between a small grey surface at `T_s` and
/// large isothermal surroundings at `T_sur`:
/// `q = ε σ A (T_s⁴ − T_sur⁴)` \[W\].
pub fn radiation_exchange(
    emissivity: f64,
    area: f64,
    t_surface: f64,
    t_surroundings: f64,
) -> Result<f64, ThermoError> {
    in_range("emissivity", emissivity, 0.0, 1.0)?;
    positive("area", area)?;
    positive("t_surface", t_surface)?;
    positive("t_surroundings", t_surroundings)?;
    Ok(emissivity * STEFAN_BOLTZMANN * area * (t_surface.powi(4) - t_surroundings.powi(4)))
}

/// Log-mean temperature difference of a heat exchanger \[K\]:
/// `LMTD = (ΔT₁ − ΔT₂)/ln(ΔT₁/ΔT₂)`, with the exact limit `ΔT` when the
/// two end differences are (nearly) equal. Both end differences must be
/// strictly positive.
pub fn lmtd(delta_t1: f64, delta_t2: f64) -> Result<f64, ThermoError> {
    positive("delta_t1", delta_t1)?;
    positive("delta_t2", delta_t2)?;
    // Relative closeness switch keeps the function smooth and avoids the
    // 0/0 form without any user-visible discontinuity.
    if (delta_t1 - delta_t2).abs() <= 1e-12 * delta_t1.max(delta_t2)
    {
        return Ok(0.5 * (delta_t1 + delta_t2));
    }
    Ok((delta_t1 - delta_t2) / (delta_t1 / delta_t2).ln())
}

/// Effectiveness of a **counter-flow** heat exchanger from `NTU ≥ 0`
/// and capacity ratio `C_r ∈ [0, 1]`:
///
/// `ε = (1 − e^{−NTU(1−C_r)}) / (1 − C_r e^{−NTU(1−C_r)})`,
/// with the exact limit `ε = NTU/(1+NTU)` at `C_r = 1`.
pub fn effectiveness_counterflow(ntu: f64, cr: f64) -> Result<f64, ThermoError> {
    non_negative("ntu", ntu)?;
    in_range("cr", cr, 0.0, 1.0)?;
    if (1.0 - cr).abs() <= 1e-12
    {
        return Ok(ntu / (1.0 + ntu));
    }
    let e = (-ntu * (1.0 - cr)).exp();
    Ok((1.0 - e) / (1.0 - cr * e))
}

/// Effectiveness of a **parallel-flow** heat exchanger:
/// `ε = (1 − e^{−NTU(1+C_r)}) / (1 + C_r)`.
pub fn effectiveness_parallel(ntu: f64, cr: f64) -> Result<f64, ThermoError> {
    non_negative("ntu", ntu)?;
    in_range("cr", cr, 0.0, 1.0)?;
    Ok((1.0 - (-ntu * (1.0 + cr)).exp()) / (1.0 + cr))
}

/// **Dittus–Boelter** Nusselt correlation for fully developed turbulent
/// pipe flow, `Nu = 0.023 Re^{0.8} Pr^n` with `n = 0.4` when the fluid
/// is being heated and `n = 0.3` when cooled.
///
/// Validity (enforced): `Re ∈ [10⁴, 10⁷]`, `Pr ∈ [0.6, 160]`.
pub fn nusselt_dittus_boelter(re: f64, pr: f64, heating: bool) -> Result<f64, ThermoError> {
    in_range("re", re, 1.0e4, 1.0e7)?;
    in_range("pr", pr, 0.6, 160.0)?;
    let n = if heating { 0.4 } else { 0.3 };
    Ok(0.023 * re.powf(0.8) * pr.powf(n))
}

/// Nusselt number of fully developed **laminar** pipe flow with a
/// constant wall temperature: `Nu = 3.66` (exact asymptotic value).
pub fn nusselt_laminar_constant_wall_t() -> f64 {
    3.66
}

/// Nusselt number of fully developed **laminar** pipe flow with a
/// constant wall heat flux: `Nu = 4.36` (exact value 48/11).
pub fn nusselt_laminar_constant_flux() -> f64 {
    48.0 / 11.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_wall_textbook_case() {
        // 20 cm brick wall (k = 0.72), 10 m², ΔT = 20 K:
        // R = 0.2/(0.72·10) = 0.027778 K/W → q = 720 W.
        let r = resistance_plane(0.72, 10.0, 0.2).unwrap();
        let q = heat_flow_series(&[r], 20.0).unwrap();
        assert!((q - 720.0).abs() < 1e-9, "q = {q}");
    }

    #[test]
    fn composite_wall_with_convection() {
        // Wall + two convective films; series resistances simply add.
        let r1 = resistance_convection(10.0, 2.0).unwrap(); // 0.05
        let r2 = resistance_plane(1.0, 2.0, 0.1).unwrap(); // 0.05
        let r3 = resistance_convection(20.0, 2.0).unwrap(); // 0.025
        let q = heat_flow_series(&[r1, r2, r3], 25.0).unwrap();
        assert!((q - 25.0 / 0.125).abs() < 1e-9, "q = {q}");
    }

    #[test]
    fn cylinder_resistance_value() {
        // k = 50, L = 1 m, r 0.05→0.10: R = ln2/(2π·50) = 2.2064e-3 K/W.
        let r = resistance_cylinder(50.0, 1.0, 0.05, 0.10).unwrap();
        assert!((r - 2.206_36e-3).abs() < 1e-7, "R = {r}");
    }

    #[test]
    fn radiation_of_black_body_at_500k() {
        // εσT⁴ per m² with T_sur → tiny: q ≈ 5.67e-8·500⁴ = 3544 W/m².
        let q = radiation_exchange(1.0, 1.0, 500.0, 1.0e-3).unwrap();
        assert!((q - 3_543.98).abs() < 0.5, "q = {q}");
        // Sign flips when the surroundings are hotter.
        assert!(radiation_exchange(1.0, 1.0, 300.0, 500.0).unwrap() < 0.0);
    }

    #[test]
    fn lmtd_60_20_case() {
        // (60−20)/ln 3 = 36.4096 K (standard textbook value).
        let dt = lmtd(60.0, 20.0).unwrap();
        assert!((dt - 36.409_6).abs() < 1e-4, "LMTD = {dt}");
        // Equal end differences: the limit is that difference.
        assert!((lmtd(15.0, 15.0).unwrap() - 15.0).abs() < 1e-12);
        // Symmetry.
        assert!((lmtd(60.0, 20.0).unwrap() - lmtd(20.0, 60.0).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn ntu_known_values() {
        // Counterflow NTU = 1, C_r = 0.5: ε = 0.564734 (Incropera tables).
        let e = effectiveness_counterflow(1.0, 0.5).unwrap();
        assert!((e - 0.564_73).abs() < 1e-4, "eps = {e}");
        // C_r = 1 limit: NTU/(1+NTU) = 0.5 at NTU = 1.
        assert!((effectiveness_counterflow(1.0, 1.0).unwrap() - 0.5).abs() < 1e-12);
        // C_r = 0 (one fluid at constant T): both layouts give 1 − e^{−NTU}.
        let e0c = effectiveness_counterflow(2.0, 0.0).unwrap();
        let e0p = effectiveness_parallel(2.0, 0.0).unwrap();
        assert!((e0c - (1.0 - (-2.0f64).exp())).abs() < 1e-12);
        assert!((e0p - e0c).abs() < 1e-12);
        // Counterflow beats parallel flow at equal NTU, C_r.
        assert!(
            effectiveness_counterflow(2.0, 0.8).unwrap()
                > effectiveness_parallel(2.0, 0.8).unwrap()
        );
    }

    #[test]
    fn dittus_boelter_water_heating() {
        // Re = 1e5, Pr = 7 (water), heating:
        // Nu = 0.023·1e5^0.8·7^0.4 ≈ 501.
        let nu = nusselt_dittus_boelter(1.0e5, 7.0, true).unwrap();
        assert!((nu - 501.0).abs() / nu < 2e-2, "Nu = {nu}");
        // Cooling exponent is smaller, so Nu is smaller for Pr > 1.
        assert!(nusselt_dittus_boelter(1.0e5, 7.0, false).unwrap() < nu);
    }

    #[test]
    fn laminar_nusselt_constants() {
        assert!((nusselt_laminar_constant_wall_t() - 3.66).abs() < 1e-12);
        assert!((nusselt_laminar_constant_flux() - 4.363_64).abs() < 1e-5);
    }

    #[test]
    fn rejects_invalid_domains() {
        assert!(resistance_cylinder(50.0, 1.0, 0.1, 0.05).is_err());
        assert!(lmtd(-5.0, 20.0).is_err());
        assert!(effectiveness_counterflow(-1.0, 0.5).is_err());
        assert!(nusselt_dittus_boelter(5000.0, 7.0, true).is_err()); // Re too low
        assert!(heat_flow_series(&[], 10.0).is_err());
        assert!(radiation_exchange(1.2, 1.0, 300.0, 280.0).is_err());
    }
}
