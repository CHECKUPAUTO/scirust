//! Pignon-crémaillère — conversion rotation ↔ translation : rayon primitif,
//! vitesse linéaire, course par tour et effort.
//!
//! ```text
//! rayon primitif    r = m·Z/2
//! vitesse linéaire  v = ω·r
//! course par tour   Δx = π·m·Z          (circonférence primitive)
//! effort            F = C/r
//! ```
//!
//! `m` module (m), `Z` nombre de dents du pignon, `r` rayon primitif (m), `ω`
//! vitesse angulaire (rad/s), `v` vitesse de la crémaillère (m/s), `C` couple au
//! pignon (N·m), `F` effort sur la crémaillère (N). Un tour de pignon déplace la
//! crémaillère de sa circonférence primitive.
//!
//! **Convention** : SI cohérent. **Limite honnête** : cinématique/statique
//! **idéale** du contact pignon-crémaillère (roulement sans glissement au primitif,
//! rendement unitaire) ; le module `m` et le nombre de dents `Z` sont fournis par
//! l'appelant.

use core::f64::consts::PI;

/// Rayon primitif du pignon `r = m·Z/2` (m).
pub fn pinion_pitch_radius(module_m: f64, teeth: u32) -> f64 {
    module_m * teeth as f64 / 2.0
}

/// Vitesse linéaire de la crémaillère `v = ω·r` (m/s).
pub fn linear_velocity(omega_rad_s: f64, pitch_radius: f64) -> f64 {
    omega_rad_s * pitch_radius
}

/// Course de la crémaillère par tour de pignon `Δx = π·m·Z` (m).
pub fn travel_per_revolution(module_m: f64, teeth: u32) -> f64 {
    PI * module_m * teeth as f64
}

/// Effort sur la crémaillère `F = C/r` (N).
///
/// Panique si `pitch_radius <= 0`.
pub fn force_from_torque(torque: f64, pitch_radius: f64) -> f64 {
    assert!(
        pitch_radius > 0.0,
        "le rayon primitif doit être strictement positif"
    );
    torque / pitch_radius
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn pitch_radius_and_travel() {
        // m=2 mm, Z=20 → r=20 mm ; course/tour = π·2·20 = 40π mm.
        assert_relative_eq!(pinion_pitch_radius(0.002, 20), 0.020, epsilon = 1e-12);
        assert_relative_eq!(
            travel_per_revolution(0.002, 20),
            PI * 0.040,
            epsilon = 1e-12
        );
    }

    #[test]
    fn travel_equals_pitch_circumference() {
        // Δx doit valoir 2π·r.
        let r = pinion_pitch_radius(0.002, 20);
        assert_relative_eq!(
            travel_per_revolution(0.002, 20),
            2.0 * PI * r,
            epsilon = 1e-12
        );
    }

    #[test]
    fn linear_velocity_from_rotation() {
        // ω=10 rad/s, r=20 mm → v = 0,2 m/s.
        assert_relative_eq!(linear_velocity(10.0, 0.020), 0.2, epsilon = 1e-12);
    }

    #[test]
    fn force_from_pinion_torque() {
        // C=5 N·m, r=20 mm → F = 250 N.
        assert_relative_eq!(force_from_torque(5.0, 0.020), 250.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "rayon primitif")]
    fn zero_radius_force_panics() {
        force_from_torque(5.0, 0.0);
    }
}
