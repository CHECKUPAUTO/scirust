//! Thermal-infrared radiometry and sensor performance metrics.
//!
//! The [`optics`](crate::optics) module characterises an imager's *spatial*
//! response (PSF, MTF). This module adds its *radiometric* and *sensitivity*
//! counterparts — the physics that decides how small a temperature difference an
//! EO/IR sensor can see:
//!
//! - **Radiometry** — Planck's law for spectral radiance, the Stefan–Boltzmann
//!   total exitance `σT⁴` and its derivative, Wien's peak-wavelength law, and
//!   band-integrated radiance / thermal contrast obtained by quadrature.
//! - **Sensitivity** — **NETD** (noise-equivalent temperature difference: the
//!   ΔT that produces a signal equal to the detector noise) from the sensor's
//!   f-number, detector specific-detectivity `D*`, and in-band thermal contrast,
//!   and **MRTD** (minimum resolvable temperature difference), the NETD-over-MTF
//!   trade-off that combines this thermal sensitivity with the [`optics`] spatial
//!   response into the headline thermal-imager spec.
//!
//! [`optics`]: crate::optics
//! Dependency-free.

use std::f64::consts::PI;

/// Planck constant `h` (J·s).
const PLANCK_H: f64 = 6.626_070_15e-34;
/// Speed of light `c` (m/s).
const C_LIGHT: f64 = 2.997_924_58e8;
/// Boltzmann constant `k_B` (J/K).
const K_B: f64 = 1.380_649e-23;
/// Stefan–Boltzmann constant `σ` (W·m⁻²·K⁻⁴).
const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;
/// Wien displacement constant `b` (m·K).
const WIEN_B: f64 = 2.897_771_955e-3;

/// Spectral radiance of a blackbody, Planck's law
/// `L(λ, T) = 2hc²/λ⁵ · 1/(exp(hc/λk_BT) − 1)` in W·m⁻²·sr⁻¹·m⁻¹, for wavelength
/// `wavelength` (m) and temperature `temperature` (K).
pub fn planck_radiance(wavelength: f64, temperature: f64) -> f64 {
    if wavelength <= 0.0 || temperature <= 0.0
    {
        return 0.0;
    }
    let c1 = 2.0 * PLANCK_H * C_LIGHT * C_LIGHT;
    let x = PLANCK_H * C_LIGHT / (wavelength * K_B * temperature);
    c1 / (wavelength.powi(5) * (x.exp() - 1.0))
}

/// The temperature derivative `∂L/∂T` of the Planck spectral radiance, in
/// W·m⁻²·sr⁻¹·m⁻¹·K⁻¹. Analytic: `∂L/∂T = L · x·eˣ / (T·(eˣ − 1))` with
/// `x = hc/λk_BT`.
pub fn planck_radiance_dt(wavelength: f64, temperature: f64) -> f64 {
    if wavelength <= 0.0 || temperature <= 0.0
    {
        return 0.0;
    }
    let x = PLANCK_H * C_LIGHT / (wavelength * K_B * temperature);
    let ex = x.exp();
    planck_radiance(wavelength, temperature) * x * ex / (temperature * (ex - 1.0))
}

/// The hemispherical radiant exitance of a blackbody, Stefan–Boltzmann
/// `M = σT⁴` (W·m⁻²).
pub fn radiant_exitance(temperature: f64) -> f64 {
    STEFAN_BOLTZMANN * temperature.powi(4)
}

/// The temperature derivative of the radiant exitance, `dM/dT = 4σT³`
/// (W·m⁻²·K⁻¹).
pub fn exitance_derivative(temperature: f64) -> f64 {
    4.0 * STEFAN_BOLTZMANN * temperature.powi(3)
}

/// The wavelength of peak spectral exitance, Wien's displacement law
/// `λ_peak = b/T` (m).
pub fn peak_wavelength(temperature: f64) -> f64 {
    if temperature <= 0.0
    {
        return f64::INFINITY;
    }
    WIEN_B / temperature
}

/// Trapezoidal integral of `f` over `[lo, hi]` with `n` intervals.
fn integrate(lo: f64, hi: f64, n: usize, f: impl Fn(f64) -> f64) -> f64 {
    if n == 0 || hi <= lo
    {
        return 0.0;
    }
    let step = (hi - lo) / n as f64;
    let mut sum = 0.5 * (f(lo) + f(hi));
    for i in 1..n
    {
        sum += f(lo + i as f64 * step);
    }
    sum * step
}

/// The in-band radiance `∫_{λ₁}^{λ₂} L(λ, T) dλ` (W·m⁻²·sr⁻¹) by `n`-interval
/// quadrature over the wavelength band `[lambda_lo, lambda_hi]` (m).
pub fn band_radiance(lambda_lo: f64, lambda_hi: f64, temperature: f64, n: usize) -> f64 {
    integrate(lambda_lo, lambda_hi, n, |l| planck_radiance(l, temperature))
}

/// The in-band **thermal contrast** `∫_{λ₁}^{λ₂} ∂L/∂T dλ`
/// (W·m⁻²·sr⁻¹·K⁻¹) — the rate at which in-band radiance rises with target
/// temperature, the quantity that drives an infrared sensor's sensitivity.
pub fn thermal_contrast(lambda_lo: f64, lambda_hi: f64, temperature: f64, n: usize) -> f64 {
    integrate(lambda_lo, lambda_hi, n, |l| {
        planck_radiance_dt(l, temperature)
    })
}

/// **NETD** — the noise-equivalent temperature difference (K), the target/
/// background ΔT that produces a signal equal to the detector noise:
///
/// `NETD = 4·F² · √Δf / (π · √A_d · τ_o · D* · (∂L/∂T)_band)`
///
/// from the optics f-number `f_number`, detector area `detector_area` (m²),
/// noise-equivalent bandwidth `noise_bandwidth` (Hz), specific detectivity
/// `d_star` (m·√Hz·W⁻¹), optical transmission `tau_optics`, and the in-band
/// thermal contrast `contrast` (from [`thermal_contrast`]). Smaller is better.
/// Non-positive/degenerate inputs yield `f64::INFINITY`.
pub fn netd(
    f_number: f64,
    detector_area: f64,
    noise_bandwidth: f64,
    d_star: f64,
    tau_optics: f64,
    contrast: f64,
) -> f64 {
    let denom = PI * detector_area.sqrt() * tau_optics * d_star * contrast;
    if denom <= 0.0 || detector_area <= 0.0
    {
        return f64::INFINITY;
    }
    4.0 * f_number * f_number * noise_bandwidth.sqrt() / denom
}

/// **MRTD** — the minimum resolvable temperature difference (K) at a spatial
/// frequency whose system MTF is `mtf`: the thermal-sensitivity/resolution
/// trade-off `MRTD = k · NETD / MTF`, where the perception factor `k` folds the
/// observer SNR threshold and the eye/temporal integration geometry. As the MTF
/// rolls off toward zero the resolvable ΔT diverges. `mtf ≤ 0` yields
/// `f64::INFINITY`.
pub fn mrtd(netd: f64, mtf: f64, perception: f64) -> f64 {
    if mtf <= 0.0
    {
        return f64::INFINITY;
    }
    perception * netd / mtf
}

#[cfg(test)]
mod tests {
    use super::*;

    // Common infrared bands (m): MWIR 3–5 µm, LWIR 8–12 µm.
    const LWIR: (f64, f64) = (8e-6, 12e-6);

    #[test]
    fn planck_integral_recovers_stefan_boltzmann() {
        // Integrating spectral radiance over (almost) all wavelengths and
        // multiplying by π (Lambertian hemisphere) recovers M = σT⁴.
        let t = 300.0;
        let l_total = band_radiance(1e-7, 2e-4, t, 20_000);
        let m = PI * l_total;
        let expect = radiant_exitance(t);
        assert!((m - expect).abs() / expect < 1e-3, "{m} vs {expect}");
    }

    #[test]
    fn exitance_and_its_derivative_match_closed_forms() {
        let t = 320.0;
        assert!((radiant_exitance(t) - STEFAN_BOLTZMANN * t.powi(4)).abs() < 1e-9);
        // dM/dT = 4σT³ = 4·M/T; also matches a central finite difference.
        let analytic = exitance_derivative(t);
        assert!((analytic - 4.0 * radiant_exitance(t) / t).abs() < 1e-9);
        let fd = (radiant_exitance(t + 0.01) - radiant_exitance(t - 0.01)) / 0.02;
        assert!(
            (analytic - fd).abs() / analytic < 1e-4,
            "{analytic} vs {fd}"
        );
    }

    #[test]
    fn wien_peak_shifts_inversely_with_temperature() {
        // Hotter ⇒ shorter peak wavelength; and the Planck curve peaks there.
        let (cool, hot) = (300.0, 600.0);
        let (pc, ph) = (peak_wavelength(cool), peak_wavelength(hot));
        assert!(
            (pc / ph - 2.0).abs() < 1e-9,
            "peak should halve when T doubles"
        );
        let peak = peak_wavelength(cool);
        let at_peak = planck_radiance(peak, cool);
        assert!(at_peak > planck_radiance(0.5 * peak, cool));
        assert!(at_peak > planck_radiance(2.0 * peak, cool));
    }

    #[test]
    fn planck_dt_matches_a_finite_difference() {
        let (l, t) = (10e-6, 300.0);
        let analytic = planck_radiance_dt(l, t);
        let fd = (planck_radiance(l, t + 0.01) - planck_radiance(l, t - 0.01)) / 0.02;
        assert!(analytic > 0.0);
        assert!(
            (analytic - fd).abs() / analytic < 1e-4,
            "{analytic} vs {fd}"
        );
    }

    #[test]
    fn thermal_contrast_is_positive_and_grows_with_temperature() {
        let cool = thermal_contrast(LWIR.0, LWIR.1, 300.0, 400);
        let warm = thermal_contrast(LWIR.0, LWIR.1, 330.0, 400);
        assert!(cool > 0.0 && warm > cool, "contrast {cool} -> {warm}");
    }

    #[test]
    fn netd_obeys_its_scaling_laws() {
        let base = netd(2.0, 4e-10, 1e5, 3e10, 0.8, 5.0);
        assert!(base.is_finite() && base > 0.0);
        // ∝ F²: doubling the f-number quadruples NETD.
        assert!((netd(4.0, 4e-10, 1e5, 3e10, 0.8, 5.0) / base - 4.0).abs() < 1e-9);
        // ∝ 1/D*: doubling detectivity halves NETD.
        assert!((netd(2.0, 4e-10, 1e5, 6e10, 0.8, 5.0) / base - 0.5).abs() < 1e-9);
        // ∝ 1/contrast: doubling thermal contrast halves NETD.
        assert!((netd(2.0, 4e-10, 1e5, 3e10, 0.8, 10.0) / base - 0.5).abs() < 1e-9);
        // ∝ 1/√A_d: quadrupling detector area halves NETD.
        assert!((netd(2.0, 16e-10, 1e5, 3e10, 0.8, 5.0) / base - 0.5).abs() < 1e-9);
        // Degenerate contrast ⇒ infinite NETD.
        assert!(netd(2.0, 4e-10, 1e5, 3e10, 0.8, 0.0).is_infinite());
    }

    #[test]
    fn mrtd_rises_as_the_mtf_falls() {
        let netd_val = 0.05;
        // At full MTF, MRTD = k·NETD.
        assert!((mrtd(netd_val, 1.0, 2.0) - 0.1).abs() < 1e-12);
        // Halving the MTF doubles the resolvable ΔT.
        assert!((mrtd(netd_val, 0.5, 2.0) - 0.2).abs() < 1e-12);
        // At the MTF cutoff the resolvable ΔT diverges.
        assert!(mrtd(netd_val, 0.0, 2.0).is_infinite());
    }
}
