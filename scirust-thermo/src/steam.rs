//! Water/steam saturation line — IAPWS-IF97, region 4.
//!
//! Clean-room implementation of the region-4 saturation-pressure and
//! saturation-temperature equations of the IAPWS Industrial Formulation
//! 1997 (both are exact closed-form inverses of the same quadratic-in-β
//! relation, so the pair round-trips to machine precision). Oracle
//! values are the verification tables 35 and 36 of the IF97 release.

use crate::error::{ThermoError, in_range};

/// Triple-point temperature of water \[K\].
pub const T_TRIPLE: f64 = 273.16;
/// Critical temperature of water \[K\].
pub const T_CRITICAL: f64 = 647.096;
/// Critical pressure of water \[Pa\].
pub const P_CRITICAL: f64 = 22.064e6;
/// Saturation pressure at 273.15 K, the lower bound of the region \[Pa\].
pub const P_MIN: f64 = 611.212_677;

/// IF97 region-4 coefficients n₁…n₁₀.
const N: [f64; 10] = [
    0.116_705_214_527_67e4,
    -0.724_213_167_032_06e6,
    -0.170_738_469_400_92e2,
    0.120_208_247_024_70e5,
    -0.323_255_503_223_33e7,
    0.149_151_086_135_30e2,
    -0.482_326_573_615_91e4,
    0.405_113_405_420_57e6,
    -0.238_555_575_678_49,
    0.650_175_348_447_98e3,
];

/// Saturation (vapour) pressure of water at temperature `t` \[K\],
/// returned in Pa. Valid for `t ∈ [273.15, 647.096] K`.
pub fn saturation_pressure(t: f64) -> Result<f64, ThermoError> {
    in_range("t", t, 273.15, T_CRITICAL)?;
    let theta = t + N[8] / (t - N[9]);
    let a = theta * theta + N[0] * theta + N[1];
    let b = N[2] * theta * theta + N[3] * theta + N[4];
    let c = N[5] * theta * theta + N[6] * theta + N[7];
    let base = 2.0 * c / (-b + (b * b - 4.0 * a * c).sqrt());
    let p_mpa = base * base * base * base;
    Ok(p_mpa * 1.0e6)
}

/// Saturation temperature of water at pressure `p` \[Pa\], returned
/// in K. Valid for `p ∈ [611.213 Pa, 22.064 MPa]`.
pub fn saturation_temperature(p: f64) -> Result<f64, ThermoError> {
    in_range("p", p, P_MIN, P_CRITICAL)?;
    let beta = (p / 1.0e6).powf(0.25);
    let e = beta * beta + N[2] * beta + N[5];
    let f = N[0] * beta * beta + N[3] * beta + N[6];
    let g = N[1] * beta * beta + N[4] * beta + N[7];
    let d = 2.0 * g / (-f - (f * f - 4.0 * e * g).sqrt());
    let s = N[9] + d;
    Ok(0.5 * (s - (s * s - 4.0 * (N[8] + N[9] * d)).sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn if97_table_35_saturation_pressure() {
        // Official IF97 verification values (MPa).
        for &(t, p_mpa) in &[
            (300.0, 0.353_658_941e-2),
            (500.0, 0.263_889_776e1),
            (600.0, 0.123_443_146e2),
        ]
        {
            let p = saturation_pressure(t).unwrap() / 1.0e6;
            assert!((p - p_mpa).abs() / p_mpa < 1e-8, "psat({t}) = {p} MPa");
        }
    }

    #[test]
    fn if97_table_36_saturation_temperature() {
        // Official IF97 verification values (K).
        for &(p_mpa, t) in &[
            (0.1, 0.372_755_919e3),
            (1.0, 0.453_035_632e3),
            (10.0, 0.584_149_488e3),
        ]
        {
            let ts = saturation_temperature(p_mpa * 1.0e6).unwrap();
            assert!((ts - t).abs() / t < 1e-8, "Tsat({p_mpa} MPa) = {ts} K");
        }
    }

    #[test]
    fn boiling_point_at_one_atmosphere() {
        // 101.325 kPa → 99.974 °C (IF97; the modern boiling point of
        // water is slightly below 100 °C).
        let ts = saturation_temperature(101_325.0).unwrap() - 273.15;
        assert!((ts - 99.974).abs() < 5e-3, "t_boil = {ts} °C");
    }

    #[test]
    fn endpoints_are_consistent() {
        // Critical point: psat(Tc) = pc.
        let pc = saturation_pressure(T_CRITICAL).unwrap();
        assert!((pc - P_CRITICAL).abs() / P_CRITICAL < 1e-6, "pc = {pc}");
        // Triple point: psat(273.16) ≈ 611.657 Pa.
        let pt = saturation_pressure(T_TRIPLE).unwrap();
        assert!((pt - 611.657).abs() < 0.1, "pt = {pt}");
    }

    #[test]
    fn pressure_temperature_roundtrip() {
        // The two closed forms are exact inverses: round-trip to ~1e-9 K.
        for &t in &[273.15, 300.0, 373.15, 450.0, 550.0, 640.0]
        {
            let ts = saturation_temperature(saturation_pressure(t).unwrap()).unwrap();
            assert!((ts - t).abs() < 1e-6, "roundtrip at {t}: {ts}");
        }
    }

    #[test]
    fn monotone_on_the_whole_line() {
        let mut prev = saturation_pressure(273.15).unwrap();
        let mut t = 274.0;
        while t < T_CRITICAL
        {
            let p = saturation_pressure(t).unwrap();
            assert!(p > prev, "not increasing at {t}");
            prev = p;
            t += 1.0;
        }
    }

    #[test]
    fn rejects_out_of_region() {
        assert!(saturation_pressure(250.0).is_err());
        assert!(saturation_pressure(700.0).is_err());
        assert!(saturation_temperature(100.0).is_err());
        assert!(saturation_temperature(30.0e6).is_err());
    }
}
