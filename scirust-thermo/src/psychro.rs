//! Psychrometrics of moist air (ASHRAE Fundamentals formulations).
//!
//! Temperatures in K, pressures in Pa, humidity ratio in kg water per kg
//! dry air, enthalpy in J per kg **dry air**.

use crate::error::{ThermoError, in_range, non_negative, positive};

/// Specific gas constant of dry air used by the ASHRAE moist-air
/// relations \[J/(kg·K)\].
pub const R_DRY_AIR: f64 = 287.042;

/// Ratio of molar masses water/dry-air used in the humidity ratio
/// (ASHRAE value 0.621945).
pub const EPSILON_W: f64 = 0.621_945;

/// Lower bound of the saturation-pressure correlations \[K\].
pub const T_MIN: f64 = 173.15;
/// Upper bound of the saturation-pressure correlations \[K\].
pub const T_MAX: f64 = 473.15;

/// Saturation pressure of water vapour \[Pa\] over **ice** below
/// 273.15 K and over **liquid water** above (Hyland–Wexler, ASHRAE
/// Fundamentals eqs. 5–6). Valid for `T ∈ [173.15, 473.15] K`.
pub fn saturation_pressure(t: f64) -> Result<f64, ThermoError> {
    in_range("t", t, T_MIN, T_MAX)?;
    let ln_p = if t < 273.15
    {
        // Over ice, 173.15–273.15 K.
        -5.674_535_9e3 / t + 6.392_524_7 - 9.677_843e-3 * t
            + 6.221_570_1e-7 * t * t
            + 2.074_782_5e-9 * t * t * t
            - 9.484_024e-13 * t * t * t * t
            + 4.163_501_9 * t.ln()
    }
    else
    {
        // Over liquid water, 273.15–473.15 K.
        -5.800_220_6e3 / t + 1.391_499_3 - 4.864_023_9e-2 * t + 4.176_476_8e-5 * t * t
            - 1.445_209_3e-8 * t * t * t
            + 6.545_967_3 * t.ln()
    };
    Ok(ln_p.exp())
}

/// Humidity ratio from total pressure and water-vapour partial pressure,
/// `W = 0.621945 p_w/(p − p_w)` \[kg/kg dry air\]. Requires `p_w < p`.
pub fn humidity_ratio(pressure: f64, p_vapor: f64) -> Result<f64, ThermoError> {
    positive("pressure", pressure)?;
    non_negative("p_vapor", p_vapor)?;
    if p_vapor >= pressure
    {
        return Err(ThermoError::OutOfRange {
            name: "p_vapor",
            value: p_vapor,
            min: 0.0,
            max: pressure,
        });
    }
    Ok(EPSILON_W * p_vapor / (pressure - p_vapor))
}

/// Humidity ratio of air at relative humidity `rh ∈ [0, 1]` and dry-bulb
/// temperature `t` \[K\] under total pressure `pressure` \[Pa\].
pub fn humidity_ratio_from_rh(pressure: f64, t: f64, rh: f64) -> Result<f64, ThermoError> {
    in_range("rh", rh, 0.0, 1.0)?;
    let pw = rh * saturation_pressure(t)?;
    humidity_ratio(pressure, pw)
}

/// Relative humidity `φ = p_w / p_ws(t)` from the vapour partial
/// pressure and the dry-bulb temperature (may exceed 1 for supersaturated
/// input; that is reported, not clamped).
pub fn relative_humidity(p_vapor: f64, t: f64) -> Result<f64, ThermoError> {
    non_negative("p_vapor", p_vapor)?;
    Ok(p_vapor / saturation_pressure(t)?)
}

/// Dew-point temperature \[K\]: the temperature whose saturation
/// pressure equals the given vapour partial pressure. Inverted from
/// [`saturation_pressure`] by deterministic bisection (the curve is
/// strictly increasing), so identical inputs give bit-identical results.
pub fn dew_point(p_vapor: f64) -> Result<f64, ThermoError> {
    positive("p_vapor", p_vapor)?;
    let p_lo = saturation_pressure(T_MIN)?;
    let p_hi = saturation_pressure(T_MAX)?;
    if !(p_lo..=p_hi).contains(&p_vapor)
    {
        return Err(ThermoError::OutOfRange {
            name: "p_vapor",
            value: p_vapor,
            min: p_lo,
            max: p_hi,
        });
    }
    let (mut lo, mut hi) = (T_MIN, T_MAX);
    for _ in 0..200
    {
        let mid = 0.5 * (lo + hi);
        if saturation_pressure(mid)? > p_vapor
        {
            hi = mid;
        }
        else
        {
            lo = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// Specific enthalpy of moist air \[J per kg **dry air**\], ASHRAE
/// formulation (reference: dry air and liquid water at 0 °C):
/// `h = 1006 t + W (2 501 000 + 1860 t)` with `t` in °C.
pub fn moist_air_enthalpy(t: f64, w: f64) -> Result<f64, ThermoError> {
    in_range("t", t, T_MIN, T_MAX)?;
    non_negative("w", w)?;
    let tc = t - 273.15;
    Ok(1006.0 * tc + w * (2.501e6 + 1860.0 * tc))
}

/// Specific volume of moist air \[m³ per kg **dry air**\]:
/// `v = R_da T (1 + 1.607858 W)/p`.
pub fn specific_volume(pressure: f64, t: f64, w: f64) -> Result<f64, ThermoError> {
    positive("pressure", pressure)?;
    positive("t", t)?;
    non_negative("w", w)?;
    Ok(R_DRY_AIR * t * (1.0 + 1.607_858 * w) / pressure)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saturation_pressure_ashrae_table_values() {
        // ASHRAE Fundamentals ch. 1 table: p_ws in kPa.
        // 0 °C: 0.6112 ; 20 °C: 2.3392 ; 25 °C: 3.1699 ; 50 °C: 12.3513.
        for &(t, p_kpa) in &[
            (273.15, 0.6112),
            (293.15, 2.3392),
            (298.15, 3.1699),
            (323.15, 12.3513),
        ]
        {
            let p = saturation_pressure(t).unwrap() / 1000.0;
            assert!((p - p_kpa).abs() / p_kpa < 2e-3, "p_ws({t}) = {p} kPa");
        }
        // Boiling point: p_ws(100 °C) ≈ 101.4 kPa.
        let p100 = saturation_pressure(373.15).unwrap();
        assert!((p100 - 101_420.0).abs() < 300.0, "p_ws(100C) = {p100}");
        // Over ice at −20 °C: 0.10326 kPa (ASHRAE table).
        let pm20 = saturation_pressure(253.15).unwrap() / 1000.0;
        assert!((pm20 - 0.103_26).abs() / 0.103_26 < 3e-3, "p = {pm20}");
    }

    #[test]
    fn standard_room_air_case() {
        // 25 °C, 50 % RH, 101.325 kPa: W ≈ 0.00988 kg/kg (ASHRAE example),
        // h ≈ 50.3 kJ/kg dry air.
        let w = humidity_ratio_from_rh(101_325.0, 298.15, 0.5).unwrap();
        assert!((w - 0.009_88).abs() < 1e-4, "W = {w}");
        let h = moist_air_enthalpy(298.15, w).unwrap();
        assert!((h - 50_300.0).abs() < 300.0, "h = {h}");
    }

    #[test]
    fn dew_point_inverts_saturation() {
        for &t in &[260.0, 273.15, 285.0, 300.0, 350.0]
        {
            let td = dew_point(saturation_pressure(t).unwrap()).unwrap();
            assert!((td - t).abs() < 1e-8, "roundtrip at {t}: {td}");
        }
        // Known point: 50 % RH at 25 °C dew point ≈ 13.9 °C.
        let pw = 0.5 * saturation_pressure(298.15).unwrap();
        let td = dew_point(pw).unwrap() - 273.15;
        assert!((td - 13.86).abs() < 0.1, "t_dp = {td}");
    }

    #[test]
    fn dry_air_specific_volume_matches_ideal_gas() {
        // W = 0 reduces to dry-air ideal gas.
        let v = specific_volume(101_325.0, 288.15, 0.0).unwrap();
        assert!((v - R_DRY_AIR * 288.15 / 101_325.0).abs() < 1e-15);
        // Moist air is lighter: bigger specific volume.
        assert!(specific_volume(101_325.0, 288.15, 0.01).unwrap() > v);
    }

    #[test]
    fn relative_humidity_roundtrip() {
        let pw = 1500.0;
        let rh = relative_humidity(pw, 295.0).unwrap();
        let w = humidity_ratio_from_rh(101_325.0, 295.0, rh).unwrap();
        assert!((w - humidity_ratio(101_325.0, pw).unwrap()).abs() < 1e-12);
    }

    #[test]
    fn rejects_out_of_domain() {
        assert!(saturation_pressure(100.0).is_err());
        assert!(saturation_pressure(500.0).is_err());
        assert!(humidity_ratio(101_325.0, 101_325.0).is_err());
        assert!(humidity_ratio_from_rh(101_325.0, 298.15, 1.5).is_err());
        assert!(dew_point(0.0).is_err());
    }
}
