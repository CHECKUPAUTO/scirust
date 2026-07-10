//! Official IAPWS-IF97 **backward** equations: `T(p,h)` and `T(p,s)`
//! for regions 1 and 2.
//!
//! [`crate::steam::region1`] and [`crate::steam::region2`] take
//! `(T, p)` and compute `h`/`s`/etc. forward. Given `(p, h)` or `(p, s)`
//! instead — the natural outputs of an energy balance, e.g. a real
//! turbine's actual exit enthalpy — the *forward* equations would need
//! an iterative solve for `T` (as [`crate::cycles::rankine_real`] does,
//! by bisection). IF97 also publishes dedicated closed-form backward
//! correlations, fitted directly against the forward equations to
//! within their own tight tolerance band, for when repeated bisection
//! is a performance bottleneck. Region 2's backward equations split
//! into three fitted sub-regions (2a/2b/2c) purely to keep each
//! polynomial small; the split boundaries are internal implementation
//! detail, not user-facing.
//!
//! All public interfaces are SI (Pa, J/kg, J/(kg·K), K); the IF97-native
//! MPa/kJ units are internal. Oracle values are the worked examples of
//! the official backward-equation releases (IF97 eqs. 11, 13, 22–27).

use crate::error::{ThermoError, in_range};

/// IF97 region-1 backward `T(p,h)` exponents (Table 6).
const B1_PH_I: [i32; 20] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6];
const B1_PH_J: [i32; 20] = [
    0, 1, 2, 6, 22, 32, 0, 1, 2, 3, 4, 10, 32, 10, 32, 10, 32, 32, 32, 32,
];
const B1_PH_N: [f64; 20] = [
    -238.72489924521,
    404.21188637945,
    113.49746881718,
    -5.8457616048039,
    -0.0001528548241314,
    -1.0866707695377e-06,
    -13.391744872602,
    43.211039183559,
    -54.010067170506,
    30.535892203916,
    -6.5964749423638,
    0.0093965400878363,
    1.157364750534e-07,
    -2.5858641282073e-05,
    -4.0644363084799e-09,
    6.6456186191635e-08,
    8.0670734103027e-11,
    -9.3477771213947e-13,
    5.8265442020601e-15,
    -1.5020185953503e-17,
];

/// IF97 region-1 backward `T(p,s)` exponents (Table 8).
const B1_PS_I: [i32; 20] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4];
const B1_PS_J: [i32; 20] = [
    0, 1, 2, 3, 11, 31, 0, 1, 2, 3, 12, 31, 0, 1, 2, 9, 31, 10, 32, 32,
];
const B1_PS_N: [f64; 20] = [
    174.78268058307,
    34.806930892873,
    6.5292584978455,
    0.33039981775489,
    -1.9281382923196e-07,
    -2.4909197244573e-23,
    -0.26107636489332,
    0.22592965981586,
    -0.064256463395226,
    0.0078876289270526,
    3.5672110607366e-10,
    1.7332496994895e-24,
    0.00056608900654837,
    -0.00032635483139717,
    4.4778286690632e-05,
    -5.1322156908507e-10,
    -4.2522657042207e-26,
    2.6400441360689e-13,
    7.8124600459723e-29,
    -3.0732199903668e-31,
];

/// IF97 region-2a backward `T(p,h)` exponents (Table 20).
const B2A_PH_I: [i32; 34] = [
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6,
    6, 7,
];
const B2A_PH_J: [i32; 34] = [
    0, 1, 2, 3, 7, 20, 0, 1, 2, 3, 7, 9, 11, 18, 44, 0, 2, 7, 36, 38, 40, 42, 44, 24, 44, 12, 32,
    44, 32, 36, 42, 34, 44, 28,
];
const B2A_PH_N: [f64; 34] = [
    1089.8952318288,
    849.51654495535,
    -107.81748091826,
    33.153654801263,
    -7.4232016790248,
    11.765048724356,
    1.844574935579,
    -4.1792700549624,
    6.2478196935812,
    -17.344563108114,
    -200.58176862096,
    271.96065473796,
    -455.11318285818,
    3091.9688604755,
    252266.40357872,
    -0.0061707422868339,
    -0.31078046629583,
    11.670873077107,
    128127984.04046,
    -985549096.23276,
    2822454697.3002,
    -3594897141.0703,
    1722734991.3197,
    -13551.334240775,
    12848734.66465,
    1.3865724283226,
    235988.32556514,
    -13105236.545054,
    7399.9835474766,
    -551966.9703006,
    3715408.5996233,
    19127.72923966,
    -415351.64835634,
    -62.459855192507,
];

/// IF97 region-2b backward `T(p,h)` exponents (Table 21).
const B2B_PH_I: [i32; 38] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5,
    5, 6, 7, 7, 9, 9,
];
const B2B_PH_J: [i32; 38] = [
    0, 1, 2, 12, 18, 24, 28, 40, 0, 2, 6, 12, 18, 24, 28, 40, 2, 8, 18, 40, 1, 2, 12, 24, 2, 12,
    18, 24, 28, 40, 18, 24, 40, 28, 2, 28, 1, 40,
];
const B2B_PH_N: [f64; 38] = [
    1489.5041079516,
    743.07798314034,
    -97.708318797837,
    2.4742464705674,
    -0.63281320016026,
    1.1385952129658,
    -0.47811863648625,
    0.0085208123431544,
    0.93747147377932,
    3.3593118604916,
    3.3809355601454,
    0.16844539671904,
    0.73875745236695,
    -0.47128737436186,
    0.15020273139707,
    -0.002176411421975,
    -0.021810755324761,
    -0.10829784403677,
    -0.046333324635812,
    7.1280351959551e-05,
    0.00011032831789999,
    0.00018955248387902,
    0.0030891541160537,
    0.0013555504554949,
    2.8640237477456e-07,
    -1.0779857357512e-05,
    -7.6462712454814e-05,
    1.4052392818316e-05,
    -3.1083814331434e-05,
    -1.0302738212103e-06,
    2.821728163504e-07,
    1.2704902271945e-06,
    7.3803353468292e-08,
    -1.1030139238909e-08,
    -8.1456365207833e-14,
    -2.5180545682962e-11,
    -1.7565233969407e-18,
    8.6934156344163e-15,
];

/// IF97 region-2c backward `T(p,h)` exponents (Table 22).
const B2C_PH_I: [i32; 23] = [
    -7, -7, -6, -6, -5, -5, -2, -2, -1, -1, 0, 0, 1, 1, 2, 6, 6, 6, 6, 6, 6, 6, 6,
];
const B2C_PH_J: [i32; 23] = [
    0, 4, 0, 2, 0, 2, 0, 1, 0, 2, 0, 1, 4, 8, 4, 0, 1, 4, 10, 12, 16, 20, 22,
];
const B2C_PH_N: [f64; 23] = [
    -3236839855524.2,
    7326335090218.1,
    358250899454.47,
    -583401318515.9,
    -10783068217.47,
    20825544563.171,
    610747.83564516,
    859777.2253558,
    -25745.72360417,
    31081.088422714,
    1208.2315865936,
    482.19755109255,
    3.7966001272486,
    -10.842984880077,
    -0.04536417267666,
    1.4559115658698e-13,
    1.126159740723e-12,
    -1.7804982240686e-11,
    1.2324579690832e-07,
    -1.1606921130984e-06,
    2.7846367088554e-05,
    -0.00059270038474176,
    0.0012918582991878,
];

/// IF97 region-2a backward `T(p,s)` exponents on `Pr` (Table 25).
/// Unlike every other backward correlation in this module, these are
/// **quarter-integer** powers (`-1.5 ..= 1.5`), not plain integers —
/// hence `f64` and [`poly_frac`] (`powf`) rather than [`poly`] (`powi`).
const B2A_PS_I: [f64; 46] = [
    -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.25, -1.25, -1.25, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
    -0.75, -0.75, -0.5, -0.5, -0.5, -0.5, -0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25, 0.5,
    0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 1.0, 1.0, 1.25, 1.25, 1.5, 1.5,
];
const B2A_PS_J: [i32; 46] = [
    -24, -23, -19, -13, -11, -10, -19, -15, -6, -26, -21, -17, -16, -9, -8, -15, -14, -26, -13, -9,
    -7, -27, -25, -11, -6, 1, 4, 8, 11, 0, 1, 5, 6, 10, 14, 16, 0, 4, 9, 17, 7, 18, 3, 15, 5, 18,
];
const B2A_PS_N: [f64; 46] = [
    -392359.83861984,
    515265.7382727,
    40482.443161048,
    -321.93790923902,
    96.961424218694,
    -22.867846371773,
    -449429.14124357,
    -5011.8336020166,
    0.35684463560015,
    44235.33584819,
    -13673.388811708,
    421632.60207864,
    22516.925837475,
    474.42144865646,
    -149.31130797647,
    -197811.26320452,
    -23554.39947076,
    -19070.616302076,
    55375.669883164,
    3829.3691437363,
    -603.91860580567,
    1936.3102620331,
    4266.064369861,
    -5978.0638872718,
    -704.01463926862,
    338.36784107553,
    20.862786635187,
    0.033834172656196,
    -4.3124428414893e-05,
    166.53791356412,
    -139.86292055898,
    -0.78849547999872,
    0.072132411753872,
    -0.0059754839398283,
    -1.2141358953904e-05,
    2.3227096733871e-07,
    -10.538463566194,
    2.0718925496502,
    -0.072193155260427,
    2.074988708112e-07,
    -0.018340657911379,
    2.9036272348696e-07,
    0.21037527893619,
    0.00025681239729999,
    -0.012799002933781,
    -8.2198102652018e-06,
];

/// IF97 region-2b backward `T(p,s)` exponents (Table 26).
const B2B_PS_I: [i32; 44] = [
    -6, -6, -5, -5, -4, -4, -4, -3, -3, -3, -3, -2, -2, -2, -2, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0,
    0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5,
];
const B2B_PS_J: [i32; 44] = [
    0, 11, 0, 11, 0, 1, 11, 0, 1, 11, 12, 0, 1, 6, 10, 0, 1, 5, 8, 9, 0, 1, 2, 4, 5, 6, 9, 0, 1, 2,
    3, 7, 8, 0, 1, 5, 0, 1, 3, 0, 1, 0, 1, 2,
];
const B2B_PS_N: [f64; 44] = [
    316876.65083497,
    20.864175881858,
    -398593.99803599,
    -21.816058518877,
    223697.85194242,
    -2784.1703445817,
    9.920743607148,
    -75197.512299157,
    2970.8605951158,
    -3.4406878548526,
    0.38815564249115,
    17511.29508575,
    -1423.7112854449,
    1.0943803364167,
    0.89971619308495,
    -3375.9740098958,
    471.62885818355,
    -1.9188241993679,
    0.41078580492196,
    -0.33465378172097,
    1387.0034777505,
    -406.63326195838,
    41.72734715961,
    2.1932549434532,
    -1.0320050009077,
    0.35882943516703,
    0.0052511453726066,
    12.838916450705,
    -2.8642437219381,
    0.56912683664855,
    -0.099962954584931,
    -0.0032632037778459,
    0.00023320922576723,
    -0.1533480985745,
    0.029072288239902,
    0.00037534702741167,
    0.0017296691702411,
    -0.00038556050844504,
    -3.5017712292608e-05,
    -1.4566393631492e-05,
    5.6420857267269e-06,
    4.1286150074605e-08,
    -2.0684671118824e-08,
    1.6409393674725e-09,
];

/// IF97 region-2c backward `T(p,s)` exponents (Table 27).
const B2C_PS_I: [i32; 30] = [
    -2, -2, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 7, 7,
];
const B2C_PS_J: [i32; 30] = [
    0, 1, 0, 0, 1, 2, 3, 0, 1, 3, 4, 0, 1, 2, 0, 1, 5, 0, 1, 4, 0, 1, 2, 0, 1, 0, 1, 3, 4, 5,
];
const B2C_PS_N: [f64; 30] = [
    909.68501005365,
    2404.566708842,
    -591.6232638713,
    541.45404128074,
    -270.98308411192,
    979.76525097926,
    -469.66772959435,
    14.399274604723,
    -19.104204230429,
    5.3299167111971,
    -21.252975375934,
    -0.3114733441376,
    0.60334840894623,
    -0.042764839702509,
    0.0058185597255259,
    -0.014597008284753,
    0.0056631175631027,
    -7.6155864584577e-05,
    0.00022440342919332,
    -1.2561095013413e-05,
    6.3323132660934e-07,
    -2.0541989675375e-06,
    3.6405370390082e-08,
    -2.9759897789215e-09,
    1.0136618529763e-08,
    5.9925719692351e-12,
    -2.0677870105164e-11,
    -2.0874278181886e-11,
    1.0162166825089e-10,
    -1.6429828281347e-10,
];

fn poly(i: &[i32], j: &[i32], n: &[f64], x: f64, y: f64) -> f64 {
    let mut sum = 0.0;
    for k in 0..n.len()
    {
        sum += n[k] * x.powi(i[k]) * y.powi(j[k]);
    }
    sum
}

/// Same as [`poly`], but for the one correlation ([`B2A_PS_I`]) whose
/// exponent on `x` is a quarter-integer rather than a plain integer.
fn poly_frac(i: &[f64], j: &[i32], n: &[f64], x: f64, y: f64) -> f64 {
    let mut sum = 0.0;
    for k in 0..n.len()
    {
        sum += n[k] * x.powf(i[k]) * y.powi(j[k]);
    }
    sum
}

/// Region-2 b/c sub-region boundary, enthalpy as a function of
/// pressure \[kJ/kg\] given `p` \[MPa\] (IF97 eq. 21, the inverse of
/// [`p_2bc`]).
fn hbc_p(p: f64) -> f64 {
    2652.6571908428 + ((p - 4.5257578905948) / 1.2809002730136e-4).sqrt()
}

/// Clamp a region-2 backward temperature to at least the saturation
/// temperature at `p_mpa`, matching the official equations' own
/// correction near the region-2/4 boundary. Silently skipped if
/// `p_mpa` falls outside region 4's own validity (the correction does
/// not apply there).
fn clamp_to_saturation(t: f64, p_mpa: f64) -> f64 {
    if p_mpa <= crate::steam::P_CRITICAL / 1.0e6
    {
        if let Ok(t_sat) = crate::steam::saturation_temperature(p_mpa * 1.0e6)
        {
            return t.max(t_sat);
        }
    }
    t
}

/// Region-1 **backward** equation `T(p, h)` \[K\]: temperature of
/// compressed/subcooled liquid water from its pressure `p` \[Pa\] and
/// specific enthalpy `h` \[J/kg\], without iterating the forward
/// equation. Valid over region 1's own `(p, h)` domain (IF97 eq. 11).
pub fn region1_t_ph(p: f64, h: f64) -> Result<f64, ThermoError> {
    in_range("p", p, f64::MIN_POSITIVE, crate::steam::P_MAX)?;
    let pr = p / 1.0e6;
    let nu = h / 1000.0 / 2500.0 + 1.0;
    Ok(poly(&B1_PH_I, &B1_PH_J, &B1_PH_N, pr, nu))
}

/// Region-1 **backward** equation `T(p, s)` \[K\]: temperature of
/// compressed/subcooled liquid water from its pressure `p` \[Pa\] and
/// specific entropy `s` \[J/(kg·K)\] (IF97 eq. 13).
pub fn region1_t_ps(p: f64, s: f64) -> Result<f64, ThermoError> {
    in_range("p", p, f64::MIN_POSITIVE, crate::steam::P_MAX)?;
    let pr = p / 1.0e6;
    let sigma = s / 1000.0 + 2.0;
    Ok(poly(&B1_PS_I, &B1_PS_J, &B1_PS_N, pr, sigma))
}

/// Region-2 **backward** equation `T(p, h)` \[K\]: temperature of
/// superheated steam from its pressure `p` \[Pa\] and specific
/// enthalpy `h` \[J/kg\] (IF97 eqs. 22–24), dispatched across the three
/// fitted sub-regions 2a/2b/2c and clamped to the saturation
/// temperature (region 2 is only meaningful above it).
pub fn region2_t_ph(p: f64, h: f64) -> Result<f64, ThermoError> {
    in_range("p", p, f64::MIN_POSITIVE, crate::steam::P_MAX)?;
    let pr = p / 1.0e6;
    let nu = h / 1000.0 / 2000.0;
    let t = if pr <= 4.0
    {
        poly(&B2A_PH_I, &B2A_PH_J, &B2A_PH_N, pr, nu - 2.1)
    }
    else if pr <= 6.546699678 || h / 1000.0 >= hbc_p(pr)
    {
        poly(&B2B_PH_I, &B2B_PH_J, &B2B_PH_N, pr - 2.0, nu - 2.6)
    }
    else
    {
        poly(&B2C_PH_I, &B2C_PH_J, &B2C_PH_N, pr + 25.0, nu - 1.8)
    };
    Ok(clamp_to_saturation(t, pr))
}

/// Region-2 **backward** equation `T(p, s)` \[K\]: temperature of
/// superheated steam from its pressure `p` \[Pa\] and specific entropy
/// `s` \[J/(kg·K)\] (IF97 eqs. 25–27), dispatched across the three
/// fitted sub-regions and clamped to the saturation temperature.
#[allow(clippy::approx_constant)] // 0.7853 is an IF97-native scaling constant, not an approximation of π/4
pub fn region2_t_ps(p: f64, s: f64) -> Result<f64, ThermoError> {
    in_range("p", p, f64::MIN_POSITIVE, crate::steam::P_MAX)?;
    let pr = p / 1.0e6;
    let s_kj = s / 1000.0;
    let t = if pr <= 4.0
    {
        poly_frac(&B2A_PS_I, &B2A_PS_J, &B2A_PS_N, pr, s_kj / 2.0 - 2.0)
    }
    else if s_kj >= 5.85
    {
        poly(&B2B_PS_I, &B2B_PS_J, &B2B_PS_N, pr, 10.0 - s_kj / 0.7853)
    }
    else
    {
        poly(&B2C_PS_I, &B2C_PS_J, &B2C_PS_N, pr, 2.0 - s_kj / 2.9251)
    };
    Ok(clamp_to_saturation(t, pr))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn if97_region1_backward_worked_examples() {
        // Official IF97 backward-equation worked examples.
        let t = region1_t_ph(3.0e6, 500.0e3).unwrap();
        assert!((t - 391.798509).abs() / 391.798509 < 1e-8, "T = {t}");
        let t = region1_t_ph(80.0e6, 1500.0e3).unwrap();
        assert!((t - 611.041229).abs() / 611.041229 < 1e-8, "T = {t}");

        let t = region1_t_ps(3.0e6, 500.0).unwrap();
        assert!((t - 307.842258).abs() / 307.842258 < 1e-8, "T = {t}");
        let t = region1_t_ps(80.0e6, 3000.0).unwrap();
        assert!((t - 565.899909).abs() / 565.899909 < 1e-8, "T = {t}");
    }

    #[test]
    fn if97_region2_backward_worked_examples_t_ph() {
        // Sub-region 2a.
        let t = region2_t_ph(0.001e6, 3000.0e3).unwrap();
        assert!((t - 534.433241).abs() / 534.433241 < 1e-8, "2a T = {t}");
        let t = region2_t_ph(3.0e6, 4000.0e3).unwrap();
        assert!((t - 1010.77577).abs() / 1010.77577 < 1e-8, "2a T = {t}");
        // Sub-region 2b.
        let t = region2_t_ph(5.0e6, 4000.0e3).unwrap();
        assert!((t - 1015.31583).abs() / 1015.31583 < 1e-8, "2b T = {t}");
        let t = region2_t_ph(25.0e6, 3500.0e3).unwrap();
        assert!((t - 875.279054).abs() / 875.279054 < 1e-8, "2b T = {t}");
        // Sub-region 2c.
        let t = region2_t_ph(40.0e6, 2700.0e3).unwrap();
        assert!((t - 743.056411).abs() / 743.056411 < 1e-8, "2c T = {t}");
        let t = region2_t_ph(60.0e6, 3200.0e3).unwrap();
        assert!((t - 882.756860).abs() / 882.756860 < 1e-6, "2c T = {t}");
    }

    #[test]
    fn if97_region2_backward_worked_examples_t_ps() {
        // Sub-region 2a.
        let t = region2_t_ps(0.1e6, 7500.0).unwrap();
        assert!((t - 399.517097).abs() / 399.517097 < 1e-8, "2a T = {t}");
        let t = region2_t_ps(2.5e6, 8000.0).unwrap();
        assert!((t - 1039.84917).abs() / 1039.84917 < 1e-8, "2a T = {t}");
        // Sub-region 2b.
        let t = region2_t_ps(8.0e6, 6000.0).unwrap();
        assert!((t - 600.484040).abs() / 600.484040 < 1e-8, "2b T = {t}");
        let t = region2_t_ps(90.0e6, 6000.0).unwrap();
        assert!((t - 1038.01126).abs() / 1038.01126 < 1e-8, "2b T = {t}");
        // Sub-region 2c.
        let t = region2_t_ps(20.0e6, 5750.0).unwrap();
        assert!((t - 697.992849).abs() / 697.992849 < 1e-8, "2c T = {t}");
        let t = region2_t_ps(80.0e6, 5750.0).unwrap();
        assert!((t - 949.017998).abs() / 949.017998 < 1e-8, "2c T = {t}");
    }

    #[test]
    fn backward_agrees_with_forward_round_trip() {
        // Region 1: pick a (T,p), evaluate forward, invert with the
        // backward equation, and recover T closely (backward equations
        // are independent fits, accurate to within ~1 mK-1K per IF97,
        // not bit-exact inverses).
        for &(t, p) in &[(300.0, 3.0e6), (450.0, 50.0e6), (600.0, 90.0e6)]
        {
            let st = crate::steam::region1(t, p).unwrap();
            let t_ph = region1_t_ph(p, st.h).unwrap();
            let t_ps = region1_t_ps(p, st.s).unwrap();
            assert!(
                (t_ph - t).abs() < 0.03,
                "T_ph({p},{}) = {t_ph}, want {t}",
                st.h
            );
            assert!(
                (t_ps - t).abs() < 0.03,
                "T_ps({p},{}) = {t_ps}, want {t}",
                st.s
            );
        }

        // Region 2, spanning all three sub-regions.
        for &(t, p) in &[
            (400.0, 0.1e6),  // low p: sub-region 2a
            (700.0, 10.0e6), // moderate p: dispatched by h vs h_bc(p)
            (750.0, 40.0e6), // high p, still below the B23 boundary at 750 K
        ]
        {
            let st = crate::steam::region2(t, p).unwrap();
            let t_ph = region2_t_ph(p, st.h).unwrap();
            let t_ps = region2_t_ps(p, st.s).unwrap();
            assert!(
                (t_ph - t).abs() < 0.05,
                "T_ph({p},{}) = {t_ph}, want {t}",
                st.h
            );
            assert!(
                (t_ps - t).abs() < 0.05,
                "T_ps({p},{}) = {t_ps}, want {t}",
                st.s
            );
        }
    }

    #[test]
    fn region2_dispatch_matches_named_subregions() {
        // These three points were chosen (via the doctest oracle) to
        // land in each labelled sub-region; check the public dispatch
        // functions reproduce those same values (i.e. picked the same
        // branch as the reference).
        assert!((region2_t_ph(0.001e6, 3000.0e3).unwrap() - 534.433241).abs() < 1e-4);
        assert!((region2_t_ph(5.0e6, 4000.0e3).unwrap() - 1015.31583).abs() < 1e-4);
        assert!((region2_t_ph(60.0e6, 3200.0e3).unwrap() - 882.756860).abs() < 1e-3);
    }

    #[test]
    fn rejects_out_of_domain() {
        assert!(region1_t_ph(0.0, 500.0e3).is_err());
        assert!(region1_t_ph(200.0e6, 500.0e3).is_err());
        assert!(region2_t_ph(-1.0, 3000.0e3).is_err());
        assert!(region2_t_ps(200.0e6, 7500.0).is_err());
    }
}
