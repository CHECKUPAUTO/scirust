//! Dimensionless groups of fluid mechanics.
//!
//! All functions take SI inputs, validate them, and return the
//! dimensionless number. Velocities are *speeds* (magnitudes, ≥ 0).

use crate::error::{FluidsError, non_negative, positive};

/// Reynolds number `Re = ρ V L / μ`.
///
/// * `density` — fluid density ρ \[kg/m³\], > 0
/// * `speed` — characteristic speed V \[m/s\], ≥ 0
/// * `length` — characteristic length L \[m\], > 0
/// * `dyn_viscosity` — dynamic viscosity μ \[Pa·s\], > 0
pub fn reynolds(
    density: f64,
    speed: f64,
    length: f64,
    dyn_viscosity: f64,
) -> Result<f64, FluidsError> {
    positive("density", density)?;
    non_negative("speed", speed)?;
    positive("length", length)?;
    positive("dyn_viscosity", dyn_viscosity)?;
    Ok(density * speed * length / dyn_viscosity)
}

/// Reynolds number from the kinematic viscosity, `Re = V L / ν`.
pub fn reynolds_kinematic(speed: f64, length: f64, kin_viscosity: f64) -> Result<f64, FluidsError> {
    non_negative("speed", speed)?;
    positive("length", length)?;
    positive("kin_viscosity", kin_viscosity)?;
    Ok(speed * length / kin_viscosity)
}

/// Prandtl number `Pr = cp μ / k`.
///
/// * `cp` — isobaric specific heat \[J/(kg·K)\], > 0
/// * `dyn_viscosity` — dynamic viscosity μ \[Pa·s\], > 0
/// * `conductivity` — thermal conductivity k \[W/(m·K)\], > 0
pub fn prandtl(cp: f64, dyn_viscosity: f64, conductivity: f64) -> Result<f64, FluidsError> {
    positive("cp", cp)?;
    positive("dyn_viscosity", dyn_viscosity)?;
    positive("conductivity", conductivity)?;
    Ok(cp * dyn_viscosity / conductivity)
}

/// Mach number `Ma = V / a`.
pub fn mach(speed: f64, sound_speed: f64) -> Result<f64, FluidsError> {
    non_negative("speed", speed)?;
    positive("sound_speed", sound_speed)?;
    Ok(speed / sound_speed)
}

/// Froude number `Fr = V / √(g L)`.
pub fn froude(speed: f64, gravity: f64, length: f64) -> Result<f64, FluidsError> {
    non_negative("speed", speed)?;
    positive("gravity", gravity)?;
    positive("length", length)?;
    Ok(speed / (gravity * length).sqrt())
}

/// Weber number `We = ρ V² L / σ`.
///
/// * `surface_tension` — σ \[N/m\], > 0
pub fn weber(
    density: f64,
    speed: f64,
    length: f64,
    surface_tension: f64,
) -> Result<f64, FluidsError> {
    positive("density", density)?;
    non_negative("speed", speed)?;
    positive("length", length)?;
    positive("surface_tension", surface_tension)?;
    Ok(density * speed * speed * length / surface_tension)
}

/// Péclet number `Pe = V L / α` (α = thermal or mass diffusivity).
pub fn peclet(speed: f64, length: f64, diffusivity: f64) -> Result<f64, FluidsError> {
    non_negative("speed", speed)?;
    positive("length", length)?;
    positive("diffusivity", diffusivity)?;
    Ok(speed * length / diffusivity)
}

/// Strouhal number `St = f L / V` (vortex-shedding frequency f).
pub fn strouhal(frequency: f64, length: f64, speed: f64) -> Result<f64, FluidsError> {
    non_negative("frequency", frequency)?;
    positive("length", length)?;
    positive("speed", speed)?;
    Ok(frequency * length / speed)
}

/// Nusselt number `Nu = h L / k` (from a known convection coefficient).
pub fn nusselt(h: f64, length: f64, conductivity: f64) -> Result<f64, FluidsError> {
    positive("h", h)?;
    positive("length", length)?;
    positive("conductivity", conductivity)?;
    Ok(h * length / conductivity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reynolds_water_pipe() {
        // Water at 20 °C in a 5 cm pipe at 1 m/s: Re = 998·1·0.05/1.002e-3.
        let re = reynolds(998.0, 1.0, 0.05, 1.002e-3).unwrap();
        assert!((re - 49_800.4).abs() / re < 1e-3, "Re = {re}");
    }

    #[test]
    fn kinematic_matches_dynamic() {
        let re1 = reynolds(1.2, 10.0, 0.3, 1.8e-5).unwrap();
        let re2 = reynolds_kinematic(10.0, 0.3, 1.8e-5 / 1.2).unwrap();
        assert!((re1 - re2).abs() / re1 < 1e-12);
    }

    #[test]
    fn prandtl_air() {
        // Air at 300 K: cp ≈ 1007, μ ≈ 1.846e-5, k ≈ 0.0263 → Pr ≈ 0.707.
        let pr = prandtl(1007.0, 1.846e-5, 0.0263).unwrap();
        assert!((pr - 0.7068).abs() < 5e-3, "Pr = {pr}");
    }

    #[test]
    fn froude_and_mach() {
        let fr = froude(3.0, 9.81, 1.0).unwrap();
        assert!((fr - 3.0 / 9.81f64.sqrt()).abs() < 1e-12);
        let ma = mach(340.29, 340.29).unwrap();
        assert!((ma - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(reynolds(-1.0, 1.0, 1.0, 1.0).is_err());
        assert!(reynolds(1.0, -1.0, 1.0, 1.0).is_err());
        assert!(mach(f64::NAN, 340.0).is_err());
        assert!(weber(1.0, 1.0, 1.0, 0.0).is_err());
    }
}
