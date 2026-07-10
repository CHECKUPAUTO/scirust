//! Water/steam properties — IAPWS-IF97, regions 1, 2 and 4.
//!
//! Clean-room implementation of the IAPWS Industrial Formulation 1997:
//!
//! - **Region 4** (saturation line): closed-form `p_sat(T)` and
//!   `T_sat(p)`, exact inverses of the same quadratic-in-β relation.
//! - **Region 1** (compressed/subcooled liquid): Gibbs-energy equation
//!   giving v, h, u, s, cp, cv and the speed of sound.
//! - **Region 2** (superheated steam): ideal + residual Gibbs-energy
//!   equation, same properties, bounded above by the saturation line and
//!   the region-2/3 **B23** parabola.
//!
//! All public interfaces are SI (K, Pa, J/kg, J/(kg·K), m³/kg, m/s);
//! the IF97-native MPa/kJ units are internal. Oracle values are the
//! official verification tables 5, 15, 35 and 36 of the IF97 release.

use crate::error::{ThermoError, in_range};

/// Specific gas constant of water used by IF97 \[J/(kg·K)\].
pub const R_WATER: f64 = 461.526;
/// Triple-point temperature of water \[K\].
pub const T_TRIPLE: f64 = 273.16;
/// Critical temperature of water \[K\].
pub const T_CRITICAL: f64 = 647.096;
/// Critical pressure of water \[Pa\].
pub const P_CRITICAL: f64 = 22.064e6;
/// Saturation pressure at 273.15 K, the lower bound of region 4 \[Pa\].
pub const P_MIN: f64 = 611.212_677;
/// Upper pressure bound of regions 1 and 2 \[Pa\].
pub const P_MAX: f64 = 100.0e6;
/// Upper temperature bound of region 1 (and of the saturated-state
/// helpers) \[K\].
pub const T_MAX_REGION1: f64 = 623.15;
/// Upper temperature bound of region 2 \[K\].
pub const T_MAX_REGION2: f64 = 1073.15;

/// IF97 region-4 coefficients n₁…n₁₀.
const N4: [f64; 10] = [
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

/// IF97 region-1 exponents I₁…I₃₄ (Table 2).
const R1_I: [i32; 34] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 8, 8, 21, 23, 29,
    30, 31, 32,
];
/// IF97 region-1 exponents J₁…J₃₄ (Table 2).
const R1_J: [i32; 34] = [
    -2, -1, 0, 1, 2, 3, 4, 5, -9, -7, -1, 0, 1, 3, -3, 0, 1, 3, 17, -4, 0, 6, -5, -2, 10, -8, -11,
    -6, -29, -31, -38, -39, -40, -41,
];
/// IF97 region-1 coefficients n₁…n₃₄ (Table 2).
const R1_N: [f64; 34] = [
    0.14632971213167,
    -0.84548187169114,
    -3.756360367204,
    3.3855169168385,
    -0.95791963387872,
    0.15772038513228,
    -0.016616417199501,
    0.00081214629983568,
    0.00028319080123804,
    -0.00060706301565874,
    -0.018990068218419,
    -0.032529748770505,
    -0.021841717175414,
    -5.283835796993e-05,
    -0.00047184321073267,
    -0.00030001780793026,
    4.7661393906987e-05,
    -4.4141845330846e-06,
    -7.2694996297594e-16,
    -3.1679644845054e-05,
    -2.8270797985312e-06,
    -8.5205128120103e-10,
    -2.2425281908e-06,
    -6.5171222895601e-07,
    -1.4341729937924e-13,
    -4.0516996860117e-07,
    -1.2734301741641e-09,
    -1.7424871230634e-10,
    -6.8762131295531e-19,
    1.4478307828521e-20,
    2.6335781662795e-23,
    -1.1947622640071e-23,
    1.8228094581404e-24,
    -9.3537087292458e-26,
];

/// IF97 region-2 ideal-gas exponents J⁰₁…J⁰₉ (Table 10).
const R2_J0: [i32; 9] = [0, 1, -5, -4, -3, -2, -1, 2, 3];
/// IF97 region-2 ideal-gas coefficients n⁰₁…n⁰₉ (Table 10).
const R2_N0: [f64; 9] = [
    -9.6927686500217,
    10.086655968018,
    -0.005608791128302,
    0.071452738081455,
    -0.40710498223928,
    1.4240819171444,
    -4.383951131945,
    -0.28408632460772,
    0.021268463753307,
];

/// IF97 region-2 residual exponents I₁…I₄₃ (Table 11).
const R2_I: [i32; 43] = [
    1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 6, 6, 6, 7, 7, 7, 8, 8, 9, 10, 10, 10,
    16, 16, 18, 20, 20, 20, 21, 22, 23, 24, 24, 24,
];
/// IF97 region-2 residual exponents J₁…J₄₃ (Table 11).
const R2_J: [i32; 43] = [
    0, 1, 2, 3, 6, 1, 2, 4, 7, 36, 0, 1, 3, 6, 35, 1, 2, 3, 7, 3, 16, 35, 0, 11, 25, 8, 36, 13, 4,
    10, 14, 29, 50, 57, 20, 35, 48, 21, 53, 39, 26, 40, 58,
];
/// IF97 region-2 residual coefficients n₁…n₄₃ (Table 11).
const R2_N: [f64; 43] = [
    -0.0017731742473213,
    -0.017834862292358,
    -0.045996013696365,
    -0.057581259083432,
    -0.05032527872793,
    -3.3032641670203e-05,
    -0.00018948987516315,
    -0.0039392777243355,
    -0.043797295650573,
    -2.6674547914087e-05,
    2.0481737692309e-08,
    4.3870667284435e-07,
    -3.227767723857e-05,
    -0.0015033924542148,
    -0.040668253562649,
    -7.8847309559367e-10,
    1.2790717852285e-08,
    4.8225372718507e-07,
    2.2922076337661e-06,
    -1.6714766451061e-11,
    -0.0021171472321355,
    -23.895741934104,
    -5.905956432427e-18,
    -1.2621808899101e-06,
    -0.038946842435739,
    1.1256211360459e-11,
    -8.2311340897998,
    1.9809712802088e-08,
    1.0406965210174e-19,
    -1.0234747095929e-13,
    -1.0018179379511e-09,
    -8.0882908646985e-11,
    0.10693031879409,
    -0.33662250574171,
    8.9185845355421e-25,
    3.0629316876232e-13,
    -4.2002467698208e-06,
    -5.9056029685639e-26,
    3.7826947613457e-06,
    -1.2768608934681e-15,
    7.3087610595061e-29,
    5.5414715350778e-17,
    -9.436970724121e-07,
];

/// Thermodynamic state of water or steam at a (T, p) point, SI units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SteamState {
    /// Specific volume \[m³/kg\].
    pub v: f64,
    /// Specific enthalpy \[J/kg\].
    pub h: f64,
    /// Specific internal energy \[J/kg\].
    pub u: f64,
    /// Specific entropy \[J/(kg·K)\].
    pub s: f64,
    /// Isobaric specific heat \[J/(kg·K)\].
    pub cp: f64,
    /// Isochoric specific heat \[J/(kg·K)\].
    pub cv: f64,
    /// Speed of sound \[m/s\].
    pub w: f64,
}

/// Saturation (vapour) pressure of water at temperature `t` \[K\],
/// returned in Pa. Valid for `t ∈ [273.15, 647.096] K` (IF97 region 4).
pub fn saturation_pressure(t: f64) -> Result<f64, ThermoError> {
    in_range("t", t, 273.15, T_CRITICAL)?;
    let theta = t + N4[8] / (t - N4[9]);
    let a = theta * theta + N4[0] * theta + N4[1];
    let b = N4[2] * theta * theta + N4[3] * theta + N4[4];
    let c = N4[5] * theta * theta + N4[6] * theta + N4[7];
    let base = 2.0 * c / (-b + (b * b - 4.0 * a * c).sqrt());
    let p_mpa = base * base * base * base;
    Ok(p_mpa * 1.0e6)
}

/// Saturation temperature of water at pressure `p` \[Pa\], returned
/// in K. Valid for `p ∈ [611.213 Pa, 22.064 MPa]` (IF97 region 4).
pub fn saturation_temperature(p: f64) -> Result<f64, ThermoError> {
    in_range("p", p, P_MIN, P_CRITICAL)?;
    let beta = (p / 1.0e6).powf(0.25);
    let e = beta * beta + N4[2] * beta + N4[5];
    let f = N4[0] * beta * beta + N4[3] * beta + N4[6];
    let g = N4[1] * beta * beta + N4[4] * beta + N4[7];
    let d = 2.0 * g / (-f - (f * f - 4.0 * e * g).sqrt());
    let s = N4[9] + d;
    Ok(0.5 * (s - (s * s - 4.0 * (N4[8] + N4[9] * d)).sqrt()))
}

/// Pressure of the region-2/region-3 **B23** boundary at temperature
/// `t` \[K\], returned in Pa (IF97 eq. 5). Valid for
/// `t ∈ [623.15, 863.15] K`.
pub fn b23_pressure(t: f64) -> Result<f64, ThermoError> {
    in_range("t", t, 623.15, 863.15)?;
    let p_mpa = 0.34805185628969e3 - 0.11671859879975e1 * t + 0.10192970039326e-2 * t * t;
    Ok(p_mpa * 1.0e6)
}

/// Dimensionless Gibbs energy of region 1 and its derivatives
/// `(g, g_π, g_ππ, g_τ, g_ττ, g_πτ)` at reduced coordinates.
fn gibbs1(pi: f64, tau: f64) -> (f64, f64, f64, f64, f64, f64) {
    let (mut g, mut gp, mut gpp, mut gt, mut gtt, mut gpt) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let x = 7.1 - pi;
    let y = tau - 1.222;
    for k in 0..R1_N.len()
    {
        let (i, j, n) = (R1_I[k], R1_J[k], R1_N[k]);
        let xi = x.powi(i);
        let yj = y.powi(j);
        g += n * xi * yj;
        gp -= n * f64::from(i) * x.powi(i - 1) * yj;
        gpp += n * f64::from(i) * f64::from(i - 1) * x.powi(i - 2) * yj;
        gt += n * f64::from(j) * xi * y.powi(j - 1);
        gtt += n * f64::from(j) * f64::from(j - 1) * xi * y.powi(j - 2);
        gpt -= n * f64::from(i) * f64::from(j) * x.powi(i - 1) * y.powi(j - 1);
    }
    (g, gp, gpp, gt, gtt, gpt)
}

/// Dimensionless Gibbs energy of region 2 (ideal + residual) and its
/// derivatives `(g, g_π, g_ππ, g_τ, g_ττ, g_πτ)`, plus the residual
/// parts `(g_π^r, g_ππ^r, g_πτ^r)` needed by the cv/w formulas.
#[allow(clippy::type_complexity)]
fn gibbs2(pi: f64, tau: f64) -> ((f64, f64, f64, f64, f64, f64), (f64, f64, f64)) {
    let mut g0 = pi.ln();
    let g0p = 1.0 / pi;
    let g0pp = -1.0 / (pi * pi);
    let (mut g0t, mut g0tt) = (0.0, 0.0);
    for k in 0..R2_N0.len()
    {
        let (j, n) = (R2_J0[k], R2_N0[k]);
        g0 += n * tau.powi(j);
        g0t += n * f64::from(j) * tau.powi(j - 1);
        g0tt += n * f64::from(j) * f64::from(j - 1) * tau.powi(j - 2);
    }
    let (mut gr, mut grp, mut grpp, mut grt, mut grtt, mut grpt) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let y = tau - 0.5;
    for k in 0..R2_N.len()
    {
        let (i, j, n) = (R2_I[k], R2_J[k], R2_N[k]);
        let pi_i = pi.powi(i);
        let yj = y.powi(j);
        gr += n * pi_i * yj;
        grp += n * f64::from(i) * pi.powi(i - 1) * yj;
        grpp += n * f64::from(i) * f64::from(i - 1) * pi.powi(i - 2) * yj;
        grt += n * f64::from(j) * pi_i * y.powi(j - 1);
        grtt += n * f64::from(j) * f64::from(j - 1) * pi_i * y.powi(j - 2);
        grpt += n * f64::from(i) * f64::from(j) * pi.powi(i - 1) * y.powi(j - 1);
    }
    (
        (
            g0 + gr,
            g0p + grp,
            g0pp + grpp,
            g0t + grt,
            g0tt + grtt,
            grpt,
        ),
        (grp, grpp, grpt),
    )
}

/// Properties of **compressed / subcooled liquid water** (IF97
/// region 1) at temperature `t` \[K\] and pressure `p` \[Pa\].
///
/// Validity (enforced): `273.15 ≤ t ≤ 623.15 K` and
/// `p_sat(t) ≤ p ≤ 100 MPa`.
pub fn region1(t: f64, p: f64) -> Result<SteamState, ThermoError> {
    in_range("t", t, 273.15, T_MAX_REGION1)?;
    in_range("p", p, saturation_pressure(t)?, P_MAX)?;
    let p_mpa = p / 1.0e6;
    let pi = p_mpa / 16.53;
    let tau = 1386.0 / t;
    let (g, gp, gpp, gt, gtt, gpt) = gibbs1(pi, tau);
    // R in kJ/(kg·K) for the IF97-native property formulas.
    let r = R_WATER / 1000.0;
    let v = pi * gp * r * t / p_mpa / 1000.0;
    let h = tau * gt * r * t;
    let s = r * (tau * gt - g);
    let cp = -r * tau * tau * gtt;
    let a = gp - tau * gpt;
    let cv = r * (-tau * tau * gtt + a * a / gpp);
    let w = (r * t * 1000.0 * gp * gp / (a * a / (tau * tau * gtt) - gpp)).sqrt();
    Ok(SteamState {
        v,
        h: h * 1000.0,
        u: (h - p_mpa * 1000.0 * v) * 1000.0,
        s: s * 1000.0,
        cp: cp * 1000.0,
        cv: cv * 1000.0,
        w,
    })
}

/// Properties of **superheated steam** (IF97 region 2) at temperature
/// `t` \[K\] and pressure `p` \[Pa\].
///
/// Validity (enforced): `273.15 ≤ t ≤ 1073.15 K`, `0 < p ≤ 100 MPa`,
/// and `p` at or below the region's upper boundary — the saturation
/// pressure for `t ≤ 623.15 K`, the B23 parabola for
/// `623.15 < t ≤ 863.15 K`.
pub fn region2(t: f64, p: f64) -> Result<SteamState, ThermoError> {
    in_range("t", t, 273.15, T_MAX_REGION2)?;
    let p_top = if t <= T_MAX_REGION1
    {
        saturation_pressure(t)?
    }
    else if t <= 863.15
    {
        b23_pressure(t)?
    }
    else
    {
        P_MAX
    };
    in_range("p", p, f64::MIN_POSITIVE, p_top.min(P_MAX))?;
    let p_mpa = p / 1.0e6;
    let pi = p_mpa;
    let tau = 540.0 / t;
    let ((g, gp, _, gt, gtt, _), (grp, grpp, grpt)) = gibbs2(pi, tau);
    let r = R_WATER / 1000.0;
    let v = pi * gp * r * t / p_mpa / 1000.0;
    let h = tau * gt * r * t;
    let s = r * (tau * gt - g);
    let cp = -r * tau * tau * gtt;
    // IF97 Table 12: a uses the residual part only, gtt the total.
    let a = 1.0 + pi * grp - tau * pi * grpt;
    let cv = r * (-tau * tau * gtt - a * a / (1.0 - pi * pi * grpp));
    let w = (r * t * 1000.0 * (1.0 + 2.0 * pi * grp + pi * pi * grp * grp)
        / (1.0 - pi * pi * grpp + a * a / (tau * tau * gtt)))
        .sqrt();
    Ok(SteamState {
        v,
        h: h * 1000.0,
        u: (h - p_mpa * 1000.0 * v) * 1000.0,
        s: s * 1000.0,
        cp: cp * 1000.0,
        cv: cv * 1000.0,
        w,
    })
}

/// Saturated-liquid state at temperature `t` \[K\] (region 1 evaluated
/// on the saturation line). Valid for `273.15 ≤ t ≤ 623.15 K`.
pub fn saturated_liquid(t: f64) -> Result<SteamState, ThermoError> {
    region1(t, saturation_pressure(t)?)
}

/// Saturated-vapour state at temperature `t` \[K\] (region 2 evaluated
/// on the saturation line). Valid for `273.15 ≤ t ≤ 623.15 K`.
pub fn saturated_vapor(t: f64) -> Result<SteamState, ThermoError> {
    region2(t, saturation_pressure(t)?)
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
    fn if97_table_5_region_1() {
        // Official IF97 verification table 5 (v, h, u, s, cp, w).
        let cases = [
            (
                300.0,
                3.0e6,
                (
                    0.100215168e-2,
                    0.115331273e3,
                    0.112324818e3,
                    0.392294792,
                    0.417301218e1,
                    0.150773921e4,
                ),
            ),
            (
                300.0,
                80.0e6,
                (
                    0.971180894e-3,
                    0.184142828e3,
                    0.106448356e3,
                    0.368563852,
                    0.401008987e1,
                    0.163469054e4,
                ),
            ),
            (
                500.0,
                3.0e6,
                (
                    0.120241800e-2,
                    0.975542239e3,
                    0.971934985e3,
                    0.258041912e1,
                    0.465580682e1,
                    0.124071337e4,
                ),
            ),
        ];
        for &(t, p, (v, h, u, s, cp, w)) in &cases
        {
            let st = region1(t, p).unwrap();
            assert!((st.v - v).abs() / v < 1e-8, "v({t},{p}) = {}", st.v);
            assert!(
                (st.h - h * 1000.0).abs() / (h * 1000.0) < 1e-8,
                "h = {}",
                st.h
            );
            assert!(
                (st.u - u * 1000.0).abs() / (u * 1000.0) < 1e-8,
                "u = {}",
                st.u
            );
            assert!(
                (st.s - s * 1000.0).abs() / (s * 1000.0) < 1e-8,
                "s = {}",
                st.s
            );
            assert!(
                (st.cp - cp * 1000.0).abs() / (cp * 1000.0) < 1e-8,
                "cp = {}",
                st.cp
            );
            assert!((st.w - w).abs() / w < 1e-8, "w = {}", st.w);
        }
        // cv from the reference implementation's own doctest.
        let st = region1(300.0, 80.0e6).unwrap();
        assert!(
            (st.cv - 3917.36606).abs() / 3917.36606 < 1e-8,
            "cv = {}",
            st.cv
        );
    }

    #[test]
    fn if97_table_15_region_2() {
        // Official IF97 verification table 15 (v, h, u, s, cp, w).
        let cases = [
            (
                300.0,
                0.0035e6,
                (
                    0.394913866e2,
                    0.254991145e4,
                    0.241169160e4,
                    0.852238967e1,
                    0.191300162e1,
                    0.427920172e3,
                ),
            ),
            (
                700.0,
                0.0035e6,
                (
                    0.923015898e2,
                    0.333568375e4,
                    0.301262819e4,
                    0.101749996e2,
                    0.208141274e1,
                    0.644289068e3,
                ),
            ),
            (
                700.0,
                30.0e6,
                (
                    0.542946619e-2,
                    0.263149474e4,
                    0.246861076e4,
                    0.517540298e1,
                    0.103505092e2,
                    0.480386523e3,
                ),
            ),
        ];
        for &(t, p, (v, h, u, s, cp, w)) in &cases
        {
            let st = region2(t, p).unwrap();
            assert!((st.v - v).abs() / v < 1e-8, "v({t},{p}) = {}", st.v);
            assert!(
                (st.h - h * 1000.0).abs() / (h * 1000.0) < 1e-8,
                "h = {}",
                st.h
            );
            assert!(
                (st.u - u * 1000.0).abs() / (u * 1000.0) < 1e-8,
                "u = {}",
                st.u
            );
            assert!(
                (st.s - s * 1000.0).abs() / (s * 1000.0) < 1e-8,
                "s = {}",
                st.s
            );
            assert!(
                (st.cp - cp * 1000.0).abs() / (cp * 1000.0) < 1e-8,
                "cp = {}",
                st.cp
            );
            assert!((st.w - w).abs() / w < 1e-8, "w = {}", st.w);
        }
        // cv from the reference implementation's own doctest.
        let st = region2(700.0, 0.0035e6).unwrap();
        assert!(
            (st.cv - 1619.78333).abs() / 1619.78333 < 1e-8,
            "cv = {}",
            st.cv
        );
    }

    #[test]
    fn b23_verification_pair() {
        // IF97 eq. 5/6 verification: T = 623.15 K ↔ p = 16.5291643 MPa.
        let p = b23_pressure(623.15).unwrap();
        assert!((p - 16.5291643e6).abs() / 16.5291643e6 < 1e-8, "p23 = {p}");
    }

    #[test]
    fn steam_tables_at_100_celsius() {
        // Classic steam-table values at t_sat = 100 °C (p ≈ 101.42 kPa):
        // hf ≈ 419.1, hg ≈ 2675.6 kJ/kg, sf ≈ 1.3069, sg ≈ 7.3541 kJ/kg/K,
        // vg ≈ 1.6719 m³/kg.
        let t = 373.15;
        let liq = saturated_liquid(t).unwrap();
        let vap = saturated_vapor(t).unwrap();
        assert!((liq.h - 419_100.0).abs() < 200.0, "hf = {}", liq.h);
        assert!((vap.h - 2_675_600.0).abs() < 1000.0, "hg = {}", vap.h);
        assert!((liq.s - 1306.9).abs() < 2.0, "sf = {}", liq.s);
        assert!((vap.s - 7354.1).abs() < 3.0, "sg = {}", vap.s);
        assert!((vap.v - 1.6719).abs() < 5e-3, "vg = {}", vap.v);
        assert!((liq.v - 1.0435e-3).abs() < 5e-6, "vf = {}", liq.v);
    }

    #[test]
    fn phase_equilibrium_consistency() {
        // Along the saturation line the Gibbs energies of the two phases
        // must (nearly) coincide, and h_fg = T·s_fg follows. Regions 1
        // and 2 are independent fits, so agreement is to ~1e-4 relative.
        for &t in &[280.0, 320.0, 373.15, 450.0, 550.0, 620.0]
        {
            let liq = saturated_liquid(t).unwrap();
            let vap = saturated_vapor(t).unwrap();
            let hfg = vap.h - liq.h;
            let sfg = vap.s - liq.s;
            assert!(
                (hfg - t * sfg).abs() / hfg < 1e-3,
                "Clausius mismatch at {t}: hfg = {hfg}, T·sfg = {}",
                t * sfg
            );
            let gf = liq.h - t * liq.s;
            let gg = vap.h - t * vap.s;
            assert!((gf - gg).abs() / hfg < 1e-3, "Gibbs mismatch at {t}");
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
        // Region 1: superheated point (p < psat) must be rejected.
        assert!(region1(500.0, 1.0e6).is_err());
        assert!(region1(650.0, 30.0e6).is_err()); // T beyond region 1
        // Region 2: compressed-liquid point (p > psat) must be rejected.
        assert!(region2(300.0, 1.0e6).is_err());
        assert!(region2(700.0, 40.0e6).is_err()); // above B23 → region 3
        assert!(region2(1100.0, 1.0e6).is_err()); // T beyond region 2
        assert!(saturated_liquid(640.0).is_err()); // beyond 623.15 K
    }
}
