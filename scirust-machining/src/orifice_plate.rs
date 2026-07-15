//! Débitmètre à plaque à **orifice** (diaphragme, norme ISO 5167) : débit
//! volumique à partir de la perte de pression au diaphragme.
//!
//! ```text
//! rapport de diamètres   β  = d₀/D
//! facteur d'approche     E  = 1/√(1 − β⁴)
//! débit volumique        Q  = Cd·ε·A₀·√(2·ΔP/(ρ·(1 − β⁴)))
//!                           = Cd·ε·A₀·E·√(2·ΔP/ρ)
//! ```
//!
//! `d₀` diamètre de l'orifice (m), `D` diamètre de la conduite (m), `β` rapport
//! de diamètres (sans dimension), `A₀` aire de l'orifice (m²), `ΔP` perte de
//! pression mesurée (Pa), `ρ` masse volumique amont (kg/m³), `Cd` coefficient de
//! décharge (sans dimension), `ε` facteur de détente/d'expansibilité (sans
//! dimension), `Q` débit volumique (m³/s), `E` facteur de vitesse d'approche.
//!
//! **Convention** : SI cohérent. **Limite honnête** : le coefficient de décharge
//! `Cd` et le facteur d'expansibilité `ε` sont **fournis** par l'appelant (issus
//! de la norme ISO 5167, dépendant du type de prise de pression, du nombre de
//! Reynolds et de `β`) ; écoulement établi en régime permanent ; `β < 1`. Aucune
//! valeur physique ou normative n'est inventée ici.

/// Rapport de diamètres `β = bore_diameter / pipe_diameter` (sans dimension).
///
/// Panique si `pipe_diameter <= 0`, `bore_diameter <= 0`, ou
/// `bore_diameter >= pipe_diameter`.
pub fn orifice_beta_ratio(bore_diameter: f64, pipe_diameter: f64) -> f64 {
    assert!(
        pipe_diameter > 0.0 && bore_diameter > 0.0,
        "les diamètres doivent être strictement positifs"
    );
    assert!(
        bore_diameter < pipe_diameter,
        "l'orifice doit être plus petit que la conduite (0 < d₀ < D)"
    );
    bore_diameter / pipe_diameter
}

/// Facteur de vitesse d'approche `E = 1/√(1 − β⁴)` (sans dimension).
///
/// Corrige la vitesse résiduelle du fluide en amont du diaphragme.
///
/// Panique si `beta_ratio` est hors de `[0, 1[`.
pub fn orifice_velocity_of_approach_factor(beta_ratio: f64) -> f64 {
    assert!((0.0..1.0).contains(&beta_ratio), "β doit être dans [0, 1[");
    1.0 / (1.0 - beta_ratio.powi(4)).sqrt()
}

/// Débit volumique `Q = Cd·ε·A₀·√(2·ΔP/(ρ·(1 − β⁴)))` (m³/s).
///
/// Panique si `density <= 0`, `bore_area < 0`, `pressure_drop < 0`, ou
/// `beta_ratio` hors de `[0, 1[`.
pub fn orifice_flow_rate(
    discharge_coefficient: f64,
    expansibility_factor: f64,
    bore_area: f64,
    beta_ratio: f64,
    pressure_drop: f64,
    density: f64,
) -> f64 {
    assert!(density > 0.0, "ρ doit être strictement positive");
    assert!(bore_area >= 0.0, "A₀ doit être positive ou nulle");
    assert!(pressure_drop >= 0.0, "ΔP doit être positive ou nulle");
    assert!((0.0..1.0).contains(&beta_ratio), "β doit être dans [0, 1[");
    let approach = orifice_velocity_of_approach_factor(beta_ratio);
    discharge_coefficient
        * expansibility_factor
        * bore_area
        * approach
        * (2.0 * pressure_drop / density).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use core::f64::consts::PI;

    #[test]
    fn beta_of_a_typical_orifice() {
        // d₀ = 50 mm dans D = 100 mm → β = 0,5.
        assert_relative_eq!(orifice_beta_ratio(0.050, 0.100), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn approach_factor_is_one_at_zero_beta() {
        // Sans contraction (β = 0), aucune correction : E = 1.
        assert_relative_eq!(
            orifice_velocity_of_approach_factor(0.0),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn flow_factors_out_the_approach_factor() {
        // Identité : Q = Cd·ε·A₀·E·√(2·ΔP/ρ), avec E le facteur d'approche.
        let (cd, eps, a0, beta, dp, rho) = (0.61, 1.0, 2.0e-3, 0.6, 1500.0, 1000.0);
        let e = orifice_velocity_of_approach_factor(beta);
        let manual = cd * eps * a0 * e * (2.0_f64 * dp / rho).sqrt();
        assert_relative_eq!(
            orifice_flow_rate(cd, eps, a0, beta, dp, rho),
            manual,
            max_relative = 1e-12
        );
    }

    #[test]
    fn flow_rises_with_root_pressure_drop() {
        // Q ∝ √ΔP : quadrupler ΔP double le débit.
        let q1 = orifice_flow_rate(0.61, 1.0, 2.0e-3, 0.5, 1000.0, 1000.0);
        let q2 = orifice_flow_rate(0.61, 1.0, 2.0e-3, 0.5, 4000.0, 1000.0);
        assert_relative_eq!(q2 / q1, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn realistic_water_metering_case() {
        // Diaphragme d₀ = 50 mm dans D = 100 mm (β = 0,5), eau ρ = 1000 kg/m³,
        // Cd = 0,61, ε = 1 (incompressible), ΔP = 1000 Pa.
        // A₀ = π/4·0,05² = 1,9634954e-3 m².
        // 1 − β⁴ = 0,9375 ; √(2·1000/(1000·0,9375)) = √2,133333 = 1,4605935.
        // Q = 0,61·1,9634954e-3·1,4605935 = 1,7493998e-3 m³/s.
        let a0 = PI / 4.0 * 0.050_f64.powi(2);
        let q = orifice_flow_rate(0.61, 1.0, a0, 0.5, 1000.0, 1000.0);
        assert_relative_eq!(q, 1.7493998e-3, max_relative = 1e-6);
    }

    #[test]
    #[should_panic(expected = "β doit être dans [0, 1[")]
    fn beta_at_least_one_panics() {
        orifice_velocity_of_approach_factor(1.0);
    }
}
