//! Barre de **torsion** (ressort de torsion à section circulaire pleine) —
//! inertie polaire, raideur angulaire, contrainte de cisaillement et rotation.
//!
//! ```text
//! inertie polaire     J = π·d⁴/32                (section circulaire pleine)
//! raideur angulaire   k = G·J/L                  (couple par radian)
//! contrainte cisaill. τ = 16·T/(π·d³)            (fibre extérieure)
//! rotation            φ = T·L/(G·J)              (petites rotations)
//! ```
//!
//! `d` diamètre de la barre (m), `L` longueur active (m), `G` module de
//! cisaillement (Pa), `T` couple appliqué (N·m), `J` inertie polaire (m⁴),
//! `k` raideur angulaire (N·m/rad), `τ` contrainte de cisaillement (Pa),
//! `φ` rotation (rad).
//!
//! **Convention** : SI cohérent, rotations en rad. **Limite honnête** : barre
//! cylindrique pleine, matériau élastique linéaire isotrope (le module de
//! cisaillement `G` est **fourni** par l'appelant, jamais supposé), petites
//! rotations, et pas de concentration de contrainte aux extrémités (encastrements,
//! congés, cannelures) — ces facteurs restent à la charge de l'appelant.

use core::f64::consts::PI;

/// Inertie polaire d'une section circulaire pleine `J = π·d⁴/32` (m⁴).
///
/// Panique si `diameter <= 0`.
pub fn torsion_bar_polar_inertia(diameter: f64) -> f64 {
    assert!(
        diameter > 0.0,
        "le diamètre de la barre doit être strictement positif"
    );
    PI * diameter.powi(4) / 32.0
}

/// Raideur angulaire `k = G·J/L` (N·m par radian), avec `J = π·d⁴/32`.
///
/// Panique si `diameter <= 0` ou `length <= 0`.
pub fn torsion_bar_rate(shear_modulus: f64, diameter: f64, length: f64) -> f64 {
    assert!(
        length > 0.0,
        "la longueur active doit être strictement positive"
    );
    shear_modulus * torsion_bar_polar_inertia(diameter) / length
}

/// Contrainte de cisaillement en fibre extérieure `τ = 16·T/(π·d³)` (Pa).
///
/// Panique si `diameter <= 0`.
pub fn torsion_bar_shear_stress(torque: f64, diameter: f64) -> f64 {
    assert!(
        diameter > 0.0,
        "le diamètre de la barre doit être strictement positif"
    );
    16.0 * torque / (PI * diameter.powi(3))
}

/// Rotation sous couple `φ = T·L/(G·J)` (rad), avec `J = π·d⁴/32`.
///
/// Panique si `shear_modulus <= 0`, `diameter <= 0` ou `length <= 0`.
pub fn torsion_bar_angle(torque: f64, shear_modulus: f64, diameter: f64, length: f64) -> f64 {
    assert!(
        shear_modulus > 0.0,
        "le module de cisaillement doit être strictement positif"
    );
    torque * length / (shear_modulus * torsion_bar_polar_inertia(diameter))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn polar_inertia_scales_with_fourth_power() {
        // J = π·d⁴/32 : doubler d multiplie J par 16.
        let j1 = torsion_bar_polar_inertia(0.020);
        let j2 = torsion_bar_polar_inertia(0.040);
        assert_relative_eq!(j2 / j1, 16.0, epsilon = 1e-9);
        assert_relative_eq!(j1, PI * 0.020_f64.powi(4) / 32.0, epsilon = 1e-15);
    }

    #[test]
    fn rate_and_angle_are_inverse() {
        // k = G·J/L et φ = T/k : le couple reconstruit T = k·φ.
        let k = torsion_bar_rate(80e9, 0.020, 1.0);
        let phi = torsion_bar_angle(50.0, 80e9, 0.020, 1.0);
        assert_relative_eq!(k * phi, 50.0, max_relative = 1e-9);
    }

    #[test]
    fn angle_matches_torque_over_rate() {
        // φ = T·L/(G·J) doit égaler T/k puisque k = G·J/L.
        let k = torsion_bar_rate(80e9, 0.025, 0.8);
        let phi = torsion_bar_angle(120.0, 80e9, 0.025, 0.8);
        assert_relative_eq!(phi, 120.0 / k, max_relative = 1e-12);
    }

    #[test]
    fn shear_stress_scales_linearly_with_torque() {
        // τ = 16·T/(π·d³) : linéaire en T, inversement en d³.
        let t1 = torsion_bar_shear_stress(100.0, 0.020);
        let t2 = torsion_bar_shear_stress(300.0, 0.020);
        assert_relative_eq!(t2 / t1, 3.0, epsilon = 1e-12);
        assert_relative_eq!(t1, 16.0 * 100.0 / (PI * 0.020_f64.powi(3)), epsilon = 1e-6);
    }

    #[test]
    fn realistic_steel_bar() {
        // Acier G = 80 GPa, d = 20 mm, L = 1 m, T = 200 N·m.
        // J = π·0.02⁴/32 = π·1.6e-7/32 = 1.570796e-8 m⁴.
        let j = torsion_bar_polar_inertia(0.020);
        assert_relative_eq!(j, 1.570_796_326_79e-8, max_relative = 1e-9);
        // k = G·J/L = 80e9·1.570796e-8 = 1256.637 N·m/rad.
        let k = torsion_bar_rate(80e9, 0.020, 1.0);
        assert_relative_eq!(k, 1_256.637_061_44, max_relative = 1e-9);
        // φ = T/k = 200/1256.637 = 0.159155 rad.
        let phi = torsion_bar_angle(200.0, 80e9, 0.020, 1.0);
        assert_relative_eq!(phi, 0.159_154_943_09, max_relative = 1e-9);
        // τ = 16·200/(π·0.02³) = 3200/(π·8e-6) = 127.324 MPa.
        let tau = torsion_bar_shear_stress(200.0, 0.020);
        assert_relative_eq!(tau, 127.323_954_47e6, max_relative = 1e-9);
    }

    #[test]
    #[should_panic(expected = "le diamètre de la barre doit être strictement positif")]
    fn zero_diameter_panics() {
        torsion_bar_polar_inertia(0.0);
    }
}
