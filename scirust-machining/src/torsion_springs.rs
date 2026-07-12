//! Ressorts de **torsion** hélicoïdaux — raideur angulaire, rotation et
//! contrainte de flexion du fil (le fil travaille en **flexion**, non en torsion).
//!
//! ```text
//! raideur angulaire   k = E·d⁴/(64·D·n)      (moment par radian)
//! rotation            θ = M/k
//! contrainte flexion  σ = 32·M/(π·d³)         (× facteur de courbure Kb)
//! ```
//!
//! `E` module de Young (Pa), `d` diamètre du fil (m), `D` diamètre moyen
//! d'enroulement (m), `n` nombre de spires actives, `M` moment appliqué (N·m),
//! `θ` rotation (rad). Contrairement au ressort de compression, l'énergie est
//! stockée en **flexion** du fil.
//!
//! **Convention** : SI cohérent, rotations en rad. **Limite honnête** : théorie
//! de flexion pure du fil (petites rotations) ; le facteur de correction de
//! courbure `Kb` (contrainte en fibre intérieure) est laissé à l'appelant, tout
//! comme le frottement inter-spires et la variation de `D` en charge.

use core::f64::consts::PI;

/// Raideur angulaire `k = E·d⁴/(64·D·n)` (N·m par radian).
///
/// Panique si `coil_diameter <= 0`, `active_coils <= 0`.
pub fn angular_rate(
    youngs_modulus: f64,
    wire_diameter: f64,
    coil_diameter: f64,
    active_coils: f64,
) -> f64 {
    assert!(
        coil_diameter > 0.0 && active_coils > 0.0,
        "D > 0 et n > 0 requis"
    );
    youngs_modulus * wire_diameter.powi(4) / (64.0 * coil_diameter * active_coils)
}

/// Rotation sous moment `θ = M/k` (rad).
///
/// Panique si `angular_rate <= 0`.
pub fn angular_deflection(moment: f64, angular_rate: f64) -> f64 {
    assert!(
        angular_rate > 0.0,
        "la raideur angulaire doit être strictement positive"
    );
    moment / angular_rate
}

/// Contrainte de flexion nominale du fil `σ = 32·M/(π·d³)` (Pa).
///
/// Panique si `wire_diameter <= 0`.
pub fn bending_stress(moment: f64, wire_diameter: f64) -> f64 {
    assert!(
        wire_diameter > 0.0,
        "le diamètre du fil doit être strictement positif"
    );
    32.0 * moment / (PI * wire_diameter.powi(3))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn rate_and_deflection_are_inverse() {
        // k = E·d⁴/(64·D·n) ; θ = M/k doit redonner M = k·θ.
        let k = angular_rate(200e9, 0.003, 0.020, 5.0);
        assert_relative_eq!(
            k,
            200e9 * 0.003f64.powi(4) / (64.0 * 0.020 * 5.0),
            epsilon = 1e-9
        );
        let theta = angular_deflection(2.0, k);
        assert_relative_eq!(k * theta, 2.0, max_relative = 1e-9);
    }

    #[test]
    fn stiffer_with_thicker_wire() {
        // La raideur varie en d⁴ : doubler d multiplie k par 16.
        let k1 = angular_rate(200e9, 0.003, 0.020, 5.0);
        let k2 = angular_rate(200e9, 0.006, 0.020, 5.0);
        assert_relative_eq!(k2 / k1, 16.0, epsilon = 1e-9);
    }

    #[test]
    fn bending_stress_scales_with_moment() {
        // σ = 32·M/(π·d³) : linéaire en M.
        assert_relative_eq!(
            bending_stress(2.0, 0.003),
            32.0 * 2.0 / (PI * 0.003f64.powi(3)),
            epsilon = 1e-3
        );
    }

    #[test]
    #[should_panic(expected = "D > 0")]
    fn zero_coil_diameter_panics() {
        angular_rate(200e9, 0.003, 0.0, 5.0);
    }
}
