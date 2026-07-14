//! Contrainte circonférentielle dans un anneau/jante mince en rotation (volant) :
//! contrainte de traction, vitesse périphérique et vitesse d'éclatement.
//!
//! ```text
//! contrainte circonférentielle   σ = ρ·v²
//! vitesse périphérique            v = 2·π·N·r / 60
//! vitesse d'éclatement (rad/s)    ω_burst = √(σ_adm/(ρ·r²))
//! ```
//!
//! `ρ` masse volumique du matériau de la jante (kg/m³), `v` vitesse
//! périphérique de la jante (m/s), `σ` contrainte circonférentielle de traction
//! (Pa), `N` vitesse de rotation (tr/min), `r` rayon moyen de la jante (m),
//! `σ_adm` contrainte admissible du matériau (Pa), `ω_burst` vitesse angulaire
//! d'éclatement (rad/s).
//!
//! **Convention** : SI cohérent. **Limite honnête** : modèle de jante MINCE, la
//! contrainte circonférentielle est supposée uniforme dans l'épaisseur ; on
//! néglige la contrainte radiale, la flexion, ainsi que l'effet des bras et du
//! moyeu (couplage jante-rayons). Les constantes matériaux (masse volumique,
//! contrainte admissible, coefficient de sécurité) sont FOURNIES par l'appelant
//! et ne sont jamais supposées ici. Complète le module `flywheel` (énergie).

use core::f64::consts::PI;

/// Contrainte circonférentielle de traction dans une jante mince `σ = ρ·v²` (Pa).
///
/// `density` en kg/m³, `rim_speed` vitesse périphérique en m/s.
///
/// Panique si `density < 0`.
pub fn rim_hoop_stress(density: f64, rim_speed: f64) -> f64 {
    assert!(density >= 0.0, "la masse volumique doit être positive");
    density * rim_speed * rim_speed
}

/// Vitesse périphérique de la jante `v = 2·π·N·r / 60` (m/s) à partir des tr/min.
///
/// `rpm` en tr/min, `radius` rayon moyen en m.
///
/// Panique si `rpm < 0` ou `radius < 0`.
pub fn rim_speed_from_rpm(rpm: f64, radius: f64) -> f64 {
    assert!(rpm >= 0.0, "la vitesse de rotation doit être positive");
    assert!(radius >= 0.0, "le rayon doit être positif");
    2.0 * PI * rpm * radius / 60.0
}

/// Vitesse angulaire d'éclatement `ω_burst = √(σ_adm/(ρ·r²))` (rad/s).
///
/// Vitesse à laquelle la contrainte circonférentielle atteint la contrainte
/// admissible, à partir de `σ = ρ·(ω·r)² = σ_adm`.
///
/// `allowable_stress` en Pa, `density` en kg/m³, `radius` en m.
///
/// Panique si `allowable_stress < 0`, `density <= 0` ou `radius <= 0`.
pub fn rotating_burst_speed_rad(allowable_stress: f64, density: f64, radius: f64) -> f64 {
    assert!(
        allowable_stress >= 0.0,
        "la contrainte admissible doit être positive"
    );
    assert!(
        density > 0.0,
        "la masse volumique doit être strictement positive"
    );
    assert!(radius > 0.0, "le rayon doit être strictement positif");
    (allowable_stress / (density * radius * radius)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn hoop_stress_scales_with_speed_squared() {
        // σ ∝ v² : doubler la vitesse quadruple la contrainte.
        let rho = 7850.0_f64;
        let s1 = rim_hoop_stress(rho, 30.0);
        let s2 = rim_hoop_stress(rho, 60.0);
        assert_relative_eq!(s2 / s1, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn rim_speed_matches_omega_r() {
        // v = 2πN/60 · r doit égaler ω·r avec ω = 2πN/60.
        let (rpm, r) = (3000.0_f64, 0.4_f64);
        let omega = 2.0 * PI * rpm / 60.0;
        assert_relative_eq!(rim_speed_from_rpm(rpm, r), omega * r, epsilon = 1e-12);
    }

    #[test]
    fn burst_speed_is_inverse_of_hoop_stress() {
        // À ω_burst, la contrainte circonférentielle vaut exactement σ_adm.
        let (sigma, rho, r) = (250e6_f64, 7850.0_f64, 0.4_f64);
        let omega = rotating_burst_speed_rad(sigma, rho, r);
        let v = omega * r;
        assert_relative_eq!(rim_hoop_stress(rho, v), sigma, epsilon = 1.0);
    }

    #[test]
    fn burst_speed_scales_inversely_with_radius() {
        // ω_burst ∝ 1/r : doubler le rayon divise par 2 la vitesse d'éclatement.
        let (sigma, rho) = (250e6_f64, 7850.0_f64);
        let w1 = rotating_burst_speed_rad(sigma, rho, 0.3);
        let w2 = rotating_burst_speed_rad(sigma, rho, 0.6);
        assert_relative_eq!(w1 / w2, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn realistic_steel_flywheel_case() {
        // Jante acier ρ=7850 kg/m³ tournant à v=100 m/s :
        // σ = 7850 · 100² = 78,5 MPa.
        let sigma = rim_hoop_stress(7850.0, 100.0);
        assert_relative_eq!(sigma, 78.5e6, epsilon = 1.0);
    }

    #[test]
    #[should_panic(expected = "la masse volumique doit être strictement positive")]
    fn burst_speed_rejects_zero_density() {
        rotating_burst_speed_rad(250e6, 0.0, 0.4);
    }
}
