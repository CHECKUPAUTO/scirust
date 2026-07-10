//! Ideal (air-standard) thermodynamic power and refrigeration cycles.
//!
//! All efficiencies are dimensionless in `[0, 1)`; temperatures are
//! absolute \[K\].

use crate::error::{ThermoError, in_range, positive};

fn check_gamma(gamma: f64) -> Result<f64, ThermoError> {
    in_range("gamma", gamma, 1.0 + 1e-12, 2.0)
}

/// Carnot efficiency `η = 1 − T_cold/T_hot` (absolute temperatures,
/// `0 < T_cold < T_hot`).
pub fn carnot_efficiency(t_hot: f64, t_cold: f64) -> Result<f64, ThermoError> {
    positive("t_hot", t_hot)?;
    positive("t_cold", t_cold)?;
    if t_cold >= t_hot
    {
        return Err(ThermoError::OutOfRange {
            name: "t_cold",
            value: t_cold,
            min: 0.0,
            max: t_hot,
        });
    }
    Ok(1.0 - t_cold / t_hot)
}

/// Carnot coefficient of performance of a **refrigerator**,
/// `COP = T_cold/(T_hot − T_cold)`.
pub fn carnot_cop_refrigerator(t_hot: f64, t_cold: f64) -> Result<f64, ThermoError> {
    carnot_efficiency(t_hot, t_cold)?; // same validation
    Ok(t_cold / (t_hot - t_cold))
}

/// Carnot coefficient of performance of a **heat pump**,
/// `COP = T_hot/(T_hot − T_cold)` (= refrigerator COP + 1).
pub fn carnot_cop_heat_pump(t_hot: f64, t_cold: f64) -> Result<f64, ThermoError> {
    carnot_efficiency(t_hot, t_cold)?;
    Ok(t_hot / (t_hot - t_cold))
}

/// Air-standard **Otto** cycle efficiency,
/// `η = 1 − r^{1−γ}` for compression ratio `r > 1`.
pub fn otto_efficiency(compression_ratio: f64, gamma: f64) -> Result<f64, ThermoError> {
    in_range(
        "compression_ratio",
        compression_ratio,
        1.0 + 1e-12,
        f64::MAX,
    )?;
    check_gamma(gamma)?;
    Ok(1.0 - compression_ratio.powf(1.0 - gamma))
}

/// Air-standard **Diesel** cycle efficiency for compression ratio
/// `r > 1` and cut-off ratio `r_c > 1`:
///
/// `η = 1 − (1/r^{γ−1}) · (r_c^γ − 1)/(γ (r_c − 1))`
pub fn diesel_efficiency(
    compression_ratio: f64,
    cutoff_ratio: f64,
    gamma: f64,
) -> Result<f64, ThermoError> {
    in_range(
        "compression_ratio",
        compression_ratio,
        1.0 + 1e-12,
        f64::MAX,
    )?;
    in_range("cutoff_ratio", cutoff_ratio, 1.0 + 1e-12, compression_ratio)?;
    check_gamma(gamma)?;
    let r = compression_ratio;
    let rc = cutoff_ratio;
    Ok(1.0 - (rc.powf(gamma) - 1.0) / (gamma * (rc - 1.0) * r.powf(gamma - 1.0)))
}

/// Air-standard **Brayton** (gas-turbine) cycle efficiency,
/// `η = 1 − r_p^{−(γ−1)/γ}` for pressure ratio `r_p > 1`.
pub fn brayton_efficiency(pressure_ratio: f64, gamma: f64) -> Result<f64, ThermoError> {
    in_range("pressure_ratio", pressure_ratio, 1.0 + 1e-12, f64::MAX)?;
    check_gamma(gamma)?;
    Ok(1.0 - pressure_ratio.powf(-(gamma - 1.0) / gamma))
}

/// Energy balance of an ideal (isentropic) **Rankine** steam cycle,
/// all quantities specific \[J/kg\] except the dimensionless efficiency
/// and exit quality.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankineCycle {
    /// Pump work input `w_p = v_f (p_boiler − p_condenser)` \[J/kg\].
    pub pump_work: f64,
    /// Heat added in the boiler `q_in = h₃ − h₂` \[J/kg\].
    pub boiler_heat: f64,
    /// Turbine work output `w_t = h₃ − h₄` \[J/kg\].
    pub turbine_work: f64,
    /// Heat rejected in the condenser `q_out = h₄ − h₁` \[J/kg\].
    pub condenser_heat: f64,
    /// Net work `w_t − w_p` \[J/kg\].
    pub net_work: f64,
    /// Thermal efficiency `w_net / q_in`.
    pub efficiency: f64,
    /// Steam quality at the turbine exit (1.0 if the exhaust is still
    /// superheated).
    pub exit_quality: f64,
}

/// Ideal (isentropic) **Rankine** cycle on the IAPWS-IF97 water
/// properties of [`crate::steam`]:
///
/// 1 → 2 isentropic pumping of saturated liquid from `p_condenser` to
/// `p_boiler` (incompressible-liquid work `v Δp`); 2 → 3 isobaric heating
/// to superheated steam at `t_turbine_inlet`; 3 → 4 isentropic expansion
/// back to `p_condenser` (wet or, for very hot inlets, still-superheated
/// exhaust — solved by deterministic bisection in either case);
/// 4 → 1 isobaric condensation.
///
/// * `p_boiler` — boiler pressure \[Pa\], > `p_condenser`
/// * `p_condenser` — condenser pressure \[Pa\], within region 4 and with
///   `T_sat ≤ 623.15 K`
/// * `t_turbine_inlet` — turbine inlet temperature \[K\]; the point
///   `(t, p_boiler)` must lie in IF97 region 2 (superheated)
pub fn rankine_ideal(
    p_boiler: f64,
    p_condenser: f64,
    t_turbine_inlet: f64,
) -> Result<RankineCycle, ThermoError> {
    positive("p_boiler", p_boiler)?;
    positive("p_condenser", p_condenser)?;
    if p_boiler <= p_condenser
    {
        return Err(ThermoError::OutOfRange {
            name: "p_boiler",
            value: p_boiler,
            min: p_condenser,
            max: f64::MAX,
        });
    }
    let t_cond = crate::steam::saturation_temperature(p_condenser)?;
    in_range(
        "t_sat(p_condenser)",
        t_cond,
        273.15,
        crate::steam::T_MAX_REGION1,
    )?;

    // State 1: saturated liquid at the condenser pressure.
    let liq = crate::steam::saturated_liquid(t_cond)?;
    let vap = crate::steam::saturated_vapor(t_cond)?;
    // State 2: after the ideal pump.
    let pump_work = liq.v * (p_boiler - p_condenser);
    let h2 = liq.h + pump_work;
    // State 3: superheated steam at the boiler exit.
    let st3 = crate::steam::region2(t_turbine_inlet, p_boiler)?;
    // State 4: isentropic expansion to the condenser pressure.
    let h4 = if st3.s <= vap.s
    {
        // Wet exhaust: interpolate inside the dome at t_cond.
        let x = (st3.s - liq.s) / (vap.s - liq.s);
        if x < 0.0
        {
            return Err(ThermoError::OutOfRange {
                name: "t_turbine_inlet",
                value: t_turbine_inlet,
                min: t_cond,
                max: crate::steam::T_MAX_REGION2,
            });
        }
        liq.h + x * (vap.h - liq.h)
    }
    else
    {
        // Superheated exhaust: find T with s(T, p_cond) = s₃ by
        // deterministic bisection (s is strictly increasing in T).
        let (mut lo, mut hi) = (t_cond, crate::steam::T_MAX_REGION2);
        for _ in 0..200
        {
            let mid = 0.5 * (lo + hi);
            if crate::steam::region2(mid, p_condenser)?.s > st3.s
            {
                hi = mid;
            }
            else
            {
                lo = mid;
            }
        }
        crate::steam::region2(0.5 * (lo + hi), p_condenser)?.h
    };
    let exit_quality = if st3.s <= vap.s
    {
        (st3.s - liq.s) / (vap.s - liq.s)
    }
    else
    {
        1.0
    };

    let boiler_heat = st3.h - h2;
    let turbine_work = st3.h - h4;
    let condenser_heat = h4 - liq.h;
    let net_work = turbine_work - pump_work;
    Ok(RankineCycle {
        pump_work,
        boiler_heat,
        turbine_work,
        condenser_heat,
        net_work,
        efficiency: net_work / boiler_heat,
        exit_quality,
    })
}

/// Energy balance of a **real** (irreversible) Rankine cycle: same
/// layout as [`RankineCycle`], plus the ideal-cycle efficiency for
/// direct comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankineCycleReal {
    /// Actual pump work input \[J/kg\].
    pub pump_work: f64,
    /// Heat added in the boiler \[J/kg\].
    pub boiler_heat: f64,
    /// Actual turbine work output \[J/kg\].
    pub turbine_work: f64,
    /// Heat rejected in the condenser \[J/kg\].
    pub condenser_heat: f64,
    /// Net work `w_t − w_p` \[J/kg\].
    pub net_work: f64,
    /// Thermal efficiency `w_net / q_in`.
    pub efficiency: f64,
    /// Steam quality at the turbine exit (1.0 if still superheated).
    pub exit_quality: f64,
    /// Efficiency of the corresponding ideal (isentropic) cycle, for
    /// direct comparison.
    pub ideal_efficiency: f64,
}

/// Real (irreversible) **Rankine** cycle: same layout as
/// [`rankine_ideal`], but the turbine and pump fall short of their
/// isentropic work by the given efficiencies.
///
/// The turbine actually delivers `η_t (h₃ − h₄ₛ)` rather than the full
/// isentropic drop, so it leaves a higher-enthalpy (drier) exhaust; its
/// exact state at `p_condenser` is located from that actual enthalpy —
/// a direct quality formula inside the two-phase dome, or a
/// deterministic bisection on `h` (monotone in T) if still superheated.
/// The pump correspondingly consumes `w_p,ideal / η_p`.
///
/// * `turbine_efficiency`, `pump_efficiency` — isentropic efficiencies,
///   each in `(0, 1]`
///
/// Other arguments are as [`rankine_ideal`].
pub fn rankine_real(
    p_boiler: f64,
    p_condenser: f64,
    t_turbine_inlet: f64,
    turbine_efficiency: f64,
    pump_efficiency: f64,
) -> Result<RankineCycleReal, ThermoError> {
    in_range(
        "turbine_efficiency",
        turbine_efficiency,
        f64::MIN_POSITIVE,
        1.0,
    )?;
    in_range("pump_efficiency", pump_efficiency, f64::MIN_POSITIVE, 1.0)?;

    let ideal = rankine_ideal(p_boiler, p_condenser, t_turbine_inlet)?;

    let t_cond = crate::steam::saturation_temperature(p_condenser)?;
    let liq = crate::steam::saturated_liquid(t_cond)?;
    let vap = crate::steam::saturated_vapor(t_cond)?;
    let st3 = crate::steam::region2(t_turbine_inlet, p_boiler)?;

    let pump_work = ideal.pump_work / pump_efficiency;
    let h2 = liq.h + pump_work;

    let turbine_work = turbine_efficiency * ideal.turbine_work;
    let h4 = st3.h - turbine_work;
    let exit_quality = if h4 <= vap.h
    {
        (h4 - liq.h) / (vap.h - liq.h)
    }
    else
    {
        1.0
    };

    let boiler_heat = st3.h - h2;
    let condenser_heat = h4 - liq.h;
    let net_work = turbine_work - pump_work;
    Ok(RankineCycleReal {
        pump_work,
        boiler_heat,
        turbine_work,
        condenser_heat,
        net_work,
        efficiency: net_work / boiler_heat,
        exit_quality,
        ideal_efficiency: ideal.efficiency,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carnot_textbook_values() {
        // 600 K / 300 K → 50 %; COPs: fridge 1, heat pump 2.
        assert!((carnot_efficiency(600.0, 300.0).unwrap() - 0.5).abs() < 1e-15);
        assert!((carnot_cop_refrigerator(600.0, 300.0).unwrap() - 1.0).abs() < 1e-15);
        assert!((carnot_cop_heat_pump(600.0, 300.0).unwrap() - 2.0).abs() < 1e-15);
        // Heat-pump COP = fridge COP + 1, always.
        let (th, tc) = (310.0, 275.0);
        let cop_r = carnot_cop_refrigerator(th, tc).unwrap();
        let cop_h = carnot_cop_heat_pump(th, tc).unwrap();
        assert!((cop_h - cop_r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn otto_r8_is_56_5_percent() {
        // Classic result: r = 8, γ = 1.4 → η = 1 − 8^{−0.4} ≈ 0.5647.
        let eta = otto_efficiency(8.0, 1.4).unwrap();
        assert!((eta - 0.5647).abs() < 1e-4, "eta = {eta}");
    }

    #[test]
    fn diesel_r18_rc2_is_63_2_percent() {
        // Cengel & Boles example: r = 18, r_c = 2, γ = 1.4 → η ≈ 0.632.
        let eta = diesel_efficiency(18.0, 2.0, 1.4).unwrap();
        assert!((eta - 0.6316).abs() < 2e-4, "eta = {eta}");
    }

    #[test]
    fn diesel_tends_to_otto_as_cutoff_shrinks() {
        // lim r_c → 1⁺ of the Diesel efficiency is the Otto efficiency.
        let eta_d = diesel_efficiency(12.0, 1.0 + 1e-6, 1.4).unwrap();
        let eta_o = otto_efficiency(12.0, 1.4).unwrap();
        assert!((eta_d - eta_o).abs() < 1e-5);
        // and Diesel is strictly below Otto at the same r for r_c > 1.
        assert!(diesel_efficiency(12.0, 2.0, 1.4).unwrap() < eta_o);
    }

    #[test]
    fn brayton_rp10_is_48_2_percent() {
        // r_p = 10, γ = 1.4 → η = 1 − 10^{−2/7} ≈ 0.4821.
        let eta = brayton_efficiency(10.0, 1.4).unwrap();
        assert!((eta - 0.4821).abs() < 2e-4, "eta = {eta}");
    }

    #[test]
    fn rejects_degenerate_cycles() {
        assert!(carnot_efficiency(300.0, 300.0).is_err());
        assert!(carnot_efficiency(300.0, 400.0).is_err());
        assert!(otto_efficiency(1.0, 1.4).is_err());
        assert!(diesel_efficiency(18.0, 20.0, 1.4).is_err()); // r_c > r
        assert!(brayton_efficiency(0.5, 1.4).is_err());
    }

    #[test]
    fn rankine_cengel_textbook_example() {
        // Cengel & Boles ex. 10-1: boiler 3 MPa, turbine inlet 350 °C,
        // condenser 75 kPa → η ≈ 26.0 %, exit quality ≈ 0.886.
        let cy = rankine_ideal(3.0e6, 75.0e3, 623.15).unwrap();
        assert!(
            (cy.efficiency - 0.260).abs() < 3e-3,
            "eta = {}",
            cy.efficiency
        );
        assert!(
            (cy.exit_quality - 0.886).abs() < 4e-3,
            "x4 = {}",
            cy.exit_quality
        );
        // Pump work is tiny next to turbine work (backwork ratio < 1 %).
        assert!(cy.pump_work / cy.turbine_work < 0.01);
    }

    #[test]
    fn rankine_first_and_second_law() {
        let cy = rankine_ideal(8.0e6, 10.0e3, 753.15).unwrap();
        // First law around the cycle: q_in − q_out = w_net (exact).
        assert!((cy.boiler_heat - cy.condenser_heat - cy.net_work).abs() / cy.net_work < 1e-12);
        // Second law: below Carnot between the extreme temperatures.
        let t_cond = crate::steam::saturation_temperature(10.0e3).unwrap();
        let eta_carnot = carnot_efficiency(753.15, t_cond).unwrap();
        assert!(cy.efficiency < eta_carnot);
        assert!(cy.efficiency > 0.0 && cy.exit_quality <= 1.0);
    }

    #[test]
    fn rankine_lower_condenser_pressure_helps() {
        // Textbook fact: dropping the condenser pressure raises η.
        let hi = rankine_ideal(3.0e6, 100.0e3, 623.15).unwrap();
        let lo = rankine_ideal(3.0e6, 10.0e3, 623.15).unwrap();
        assert!(lo.efficiency > hi.efficiency);
    }

    #[test]
    fn rankine_superheated_exhaust_branch() {
        // Very hot inlet at modest boiler pressure: the isentropic
        // expansion ends still-superheated; quality is reported as 1
        // and the enthalpy comes from the bisection branch.
        let cy = rankine_ideal(0.5e6, 200.0e3, 1000.0).unwrap();
        assert!((cy.exit_quality - 1.0).abs() < 1e-12);
        assert!(cy.turbine_work > 0.0 && cy.efficiency > 0.0);
    }

    #[test]
    fn rankine_rejects_bad_configurations() {
        // Boiler below condenser pressure.
        assert!(rankine_ideal(50.0e3, 75.0e3, 623.15).is_err());
        // Turbine inlet not superheated at the boiler pressure
        // (T below saturation at 3 MPa → region-2 validation fails).
        assert!(rankine_ideal(3.0e6, 75.0e3, 400.0).is_err());
        // Condenser pressure outside region 4.
        assert!(rankine_ideal(3.0e6, 100.0, 623.15).is_err());
    }

    #[test]
    fn rankine_real_at_unit_efficiency_matches_ideal() {
        // η_t = η_p = 1 must reproduce the ideal cycle exactly.
        let (pb, pc, t) = (3.0e6, 75.0e3, 623.15);
        let ideal = rankine_ideal(pb, pc, t).unwrap();
        let real = rankine_real(pb, pc, t, 1.0, 1.0).unwrap();
        assert!((real.efficiency - ideal.efficiency).abs() < 1e-9);
        assert!((real.exit_quality - ideal.exit_quality).abs() < 1e-9);
        assert!((real.pump_work - ideal.pump_work).abs() < 1e-6);
        assert!((real.turbine_work - ideal.turbine_work).abs() < 1e-6);
    }

    #[test]
    fn rankine_real_irreversibility_effects() {
        // Typical isentropic efficiencies (85 %): real efficiency must
        // fall below the ideal one, and the wetter-turbine physics
        // reverses — a less-effective expansion leaves MORE enthalpy in
        // the exhaust, hence a HIGHER (drier) exit quality.
        let (pb, pc, t) = (3.0e6, 75.0e3, 623.15);
        let ideal = rankine_ideal(pb, pc, t).unwrap();
        let real = rankine_real(pb, pc, t, 0.85, 0.85).unwrap();
        assert!(
            real.efficiency < real.ideal_efficiency,
            "real={}, ideal={}",
            real.efficiency,
            real.ideal_efficiency
        );
        assert!((real.ideal_efficiency - ideal.efficiency).abs() < 1e-9);
        assert!(
            real.exit_quality > ideal.exit_quality,
            "real x4={}, ideal x4={}",
            real.exit_quality,
            ideal.exit_quality
        );
        // Degraded pump needs more work; degraded turbine yields less.
        assert!(real.pump_work > ideal.pump_work);
        assert!(real.turbine_work < ideal.turbine_work);
        // First law still closes exactly for the real cycle.
        assert!(
            (real.boiler_heat - real.condenser_heat - real.net_work).abs() / real.net_work < 1e-12
        );
    }

    #[test]
    fn rankine_real_rejects_bad_efficiencies() {
        let (pb, pc, t) = (3.0e6, 75.0e3, 623.15);
        assert!(rankine_real(pb, pc, t, 0.0, 0.85).is_err());
        assert!(rankine_real(pb, pc, t, 0.85, 0.0).is_err());
        assert!(rankine_real(pb, pc, t, 1.1, 0.85).is_err());
        assert!(rankine_real(pb, pc, t, 0.85, -0.5).is_err());
    }
}
