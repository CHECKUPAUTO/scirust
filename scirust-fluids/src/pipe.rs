//! Internal (pipe) flow: friction factors and pressure losses.
//!
//! Friction factors are **Darcy** friction factors (Moody chart
//! convention, 4× the Fanning factor). The head/pressure-loss relations
//! use the Darcy–Weisbach equation.

use crate::error::{FluidsError, in_range, non_negative, positive};

/// Reynolds number below which pipe flow is treated as laminar.
pub const RE_LAMINAR_MAX: f64 = 2300.0;
/// Reynolds number above which pipe flow is treated as fully turbulent.
pub const RE_TURBULENT_MIN: f64 = 4000.0;

/// Hydraulic diameter `D_h = 4 A / P` of an arbitrary cross-section.
///
/// * `area` — flow cross-section area \[m²\], > 0
/// * `wetted_perimeter` — wetted perimeter \[m\], > 0
pub fn hydraulic_diameter(area: f64, wetted_perimeter: f64) -> Result<f64, FluidsError> {
    positive("area", area)?;
    positive("wetted_perimeter", wetted_perimeter)?;
    Ok(4.0 * area / wetted_perimeter)
}

/// Laminar Darcy friction factor `f = 64 / Re` (circular pipe, Re > 0).
pub fn friction_laminar(re: f64) -> Result<f64, FluidsError> {
    positive("re", re)?;
    Ok(64.0 / re)
}

/// Turbulent Darcy friction factor from the implicit **Colebrook–White**
/// equation, solved deterministically by Newton iteration on `x = 1/√f`:
///
/// `1/√f = −2 log₁₀( ε/(3.7 D) + 2.51/(Re √f) )`
///
/// * `re` — Reynolds number, must be ≥ [`RE_TURBULENT_MIN`]
/// * `rel_roughness` — relative roughness ε/D, in `[0, 0.1]`
///
/// The iteration starts from the explicit Swamee–Jain estimate and is a
/// fixed, input-independent algorithm: identical inputs give identical
/// outputs on every platform. Converges to |Δx| < 1e-14 in a handful of
/// steps over the whole validity domain.
pub fn friction_colebrook(re: f64, rel_roughness: f64) -> Result<f64, FluidsError> {
    in_range("re", re, RE_TURBULENT_MIN, f64::MAX)?;
    in_range("rel_roughness", rel_roughness, 0.0, 0.1)?;

    let a = rel_roughness / 3.7;
    let b = 2.51 / re;
    // Swamee–Jain starting point (guard the smooth-pipe log argument).
    let mut x = 1.0 / friction_swamee_jain(re, rel_roughness)?.sqrt();
    let ln10 = std::f64::consts::LN_10;
    for _ in 0..50
    {
        let arg = a + b * x;
        let g = x + 2.0 * arg.log10();
        let dg = 1.0 + 2.0 * b / (ln10 * arg);
        let dx = g / dg;
        x -= dx;
        if dx.abs() < 1e-14
        {
            return Ok(1.0 / (x * x));
        }
    }
    Err(FluidsError::NoConvergence {
        what: "Colebrook-White friction factor",
    })
}

/// Explicit **Haaland** approximation of the Colebrook equation
/// (±1.5 % over the Moody chart):
/// `1/√f = −1.8 log₁₀( (ε/(3.7 D))^1.11 + 6.9/Re )`.
pub fn friction_haaland(re: f64, rel_roughness: f64) -> Result<f64, FluidsError> {
    in_range("re", re, RE_TURBULENT_MIN, f64::MAX)?;
    in_range("rel_roughness", rel_roughness, 0.0, 0.1)?;
    let inv_sqrt = -1.8 * ((rel_roughness / 3.7).powf(1.11) + 6.9 / re).log10();
    Ok(1.0 / (inv_sqrt * inv_sqrt))
}

/// Explicit **Swamee–Jain** approximation of the Colebrook equation:
/// `f = 0.25 / log₁₀( ε/(3.7 D) + 5.74/Re^0.9 )²`.
pub fn friction_swamee_jain(re: f64, rel_roughness: f64) -> Result<f64, FluidsError> {
    in_range("re", re, RE_TURBULENT_MIN, f64::MAX)?;
    in_range("rel_roughness", rel_roughness, 0.0, 0.1)?;
    let log = (rel_roughness / 3.7 + 5.74 / re.powf(0.9)).log10();
    Ok(0.25 / (log * log))
}

/// Darcy friction factor over the whole Reynolds range.
///
/// * `Re ≤ 2300` — laminar, `f = 64/Re`;
/// * `Re ≥ 4000` — turbulent, Colebrook–White;
/// * `2300 < Re < 4000` — critical zone, where no established law exists:
///   the value is a **documented deterministic linear blend in Re**
///   between the laminar value at 2300 and the Colebrook value at 4000.
pub fn friction_factor(re: f64, rel_roughness: f64) -> Result<f64, FluidsError> {
    positive("re", re)?;
    in_range("rel_roughness", rel_roughness, 0.0, 0.1)?;
    if re <= RE_LAMINAR_MAX
    {
        return friction_laminar(re);
    }
    if re >= RE_TURBULENT_MIN
    {
        return friction_colebrook(re, rel_roughness);
    }
    let f_lam = friction_laminar(RE_LAMINAR_MAX)?;
    let f_turb = friction_colebrook(RE_TURBULENT_MIN, rel_roughness)?;
    let t = (re - RE_LAMINAR_MAX) / (RE_TURBULENT_MIN - RE_LAMINAR_MAX);
    Ok(f_lam + t * (f_turb - f_lam))
}

/// Darcy–Weisbach pressure drop `Δp = f (L/D) ρ V²/2` \[Pa\].
///
/// * `friction` — Darcy friction factor, > 0
/// * `length` — pipe length L \[m\], > 0
/// * `diameter` — pipe (hydraulic) diameter D \[m\], > 0
/// * `density` — fluid density ρ \[kg/m³\], > 0
/// * `speed` — mean flow speed V \[m/s\], ≥ 0
pub fn darcy_pressure_drop(
    friction: f64,
    length: f64,
    diameter: f64,
    density: f64,
    speed: f64,
) -> Result<f64, FluidsError> {
    positive("friction", friction)?;
    positive("length", length)?;
    positive("diameter", diameter)?;
    positive("density", density)?;
    non_negative("speed", speed)?;
    Ok(friction * length / diameter * 0.5 * density * speed * speed)
}

/// Darcy–Weisbach head loss `h_f = f (L/D) V²/(2 g)` \[m of fluid column\].
pub fn darcy_head_loss(
    friction: f64,
    length: f64,
    diameter: f64,
    speed: f64,
    gravity: f64,
) -> Result<f64, FluidsError> {
    positive("friction", friction)?;
    positive("length", length)?;
    positive("diameter", diameter)?;
    non_negative("speed", speed)?;
    positive("gravity", gravity)?;
    Ok(friction * length / diameter * speed * speed / (2.0 * gravity))
}

/// Minor (fitting) pressure loss `Δp = K ρ V²/2` \[Pa\] for a loss
/// coefficient `K ≥ 0` (elbow, valve, entrance, …).
pub fn minor_loss(k: f64, density: f64, speed: f64) -> Result<f64, FluidsError> {
    non_negative("k", k)?;
    positive("density", density)?;
    non_negative("speed", speed)?;
    Ok(k * 0.5 * density * speed * speed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hydraulic_diameter_of_circle_is_diameter() {
        let d = 0.08;
        let area = std::f64::consts::PI * d * d / 4.0;
        let per = std::f64::consts::PI * d;
        let dh = hydraulic_diameter(area, per).unwrap();
        assert!((dh - d).abs() < 1e-15);
    }

    #[test]
    fn laminar_moody_value() {
        // Moody chart: Re = 1000 → f = 0.064.
        assert!((friction_laminar(1000.0).unwrap() - 0.064).abs() < 1e-15);
    }

    #[test]
    fn colebrook_satisfies_its_own_equation() {
        // Airtight oracle: the returned f must satisfy the implicit
        // Colebrook-White equation to near machine precision.
        for &(re, eps) in &[
            (4.0e3, 0.0),
            (1.0e5, 0.0),
            (1.0e5, 1.0e-4),
            (1.0e6, 1.0e-3),
            (1.0e8, 5.0e-2),
        ]
        {
            let f = friction_colebrook(re, eps).unwrap();
            let lhs = 1.0 / f.sqrt();
            let rhs = -2.0 * (eps / 3.7 + 2.51 / (re * f.sqrt())).log10();
            assert!((lhs - rhs).abs() < 1e-10, "residual at Re={re}, eps={eps}");
        }
    }

    #[test]
    fn colebrook_smooth_pipe_known_value() {
        // Smooth pipe, Re = 1e5: Moody/Colebrook give f ≈ 0.01799.
        let f = friction_colebrook(1.0e5, 0.0).unwrap();
        assert!((f - 0.01799).abs() < 1e-4, "f = {f}");
    }

    #[test]
    fn explicit_approximations_close_to_colebrook() {
        for &(re, eps) in &[(1.0e4, 1.0e-4), (1.0e5, 1.0e-3), (1.0e7, 1.0e-2)]
        {
            let f = friction_colebrook(re, eps).unwrap();
            let fh = friction_haaland(re, eps).unwrap();
            let fs = friction_swamee_jain(re, eps).unwrap();
            assert!((fh - f).abs() / f < 0.03, "Haaland off at Re={re}");
            assert!((fs - f).abs() / f < 0.03, "Swamee-Jain off at Re={re}");
        }
    }

    #[test]
    fn friction_factor_is_continuous_at_regime_edges() {
        let eps = 1e-4;
        let f_lam = friction_factor(RE_LAMINAR_MAX, eps).unwrap();
        let f_blend_lo = friction_factor(RE_LAMINAR_MAX + 1e-6, eps).unwrap();
        assert!((f_lam - f_blend_lo).abs() < 1e-6);
        let f_turb = friction_factor(RE_TURBULENT_MIN, eps).unwrap();
        let f_blend_hi = friction_factor(RE_TURBULENT_MIN - 1e-6, eps).unwrap();
        assert!((f_turb - f_blend_hi).abs() < 1e-6);
    }

    #[test]
    fn darcy_weisbach_textbook_case() {
        // f = 0.02, L = 100 m, D = 0.1 m, water ρ = 1000, V = 2 m/s:
        // Δp = 0.02·1000·0.5·1000·4 = 40 000 Pa; h_f = Δp/(ρ g).
        let dp = darcy_pressure_drop(0.02, 100.0, 0.1, 1000.0, 2.0).unwrap();
        assert!((dp - 40_000.0).abs() < 1e-9);
        let hf = darcy_head_loss(0.02, 100.0, 0.1, 2.0, 9.81).unwrap();
        assert!((hf - 40_000.0 / (1000.0 * 9.81)).abs() < 1e-12);
    }

    #[test]
    fn minor_loss_elbow() {
        // K = 0.9 elbow, water at 3 m/s: Δp = 0.9·0.5·1000·9 = 4050 Pa.
        let dp = minor_loss(0.9, 1000.0, 3.0).unwrap();
        assert!((dp - 4050.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_out_of_domain() {
        assert!(friction_colebrook(1000.0, 0.0).is_err()); // laminar Re
        assert!(friction_colebrook(1.0e5, 0.5).is_err()); // roughness too big
        assert!(friction_factor(0.0, 0.0).is_err());
    }
}
