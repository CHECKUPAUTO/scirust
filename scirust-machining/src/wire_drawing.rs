//! Mise en forme — **tréfilage** (étirage de fil/barre) : réduction de section,
//! contrainte et effort d'étirage, réduction maximale par passe.
//!
//! ```text
//! réduction de section r = (A0 − Af)/A0
//! déformation vraie     ε = ln(A0/Af)
//! contrainte d'étirage (idéale) σd = Ȳ·ln(A0/Af)
//! effort d'étirage      F = σd·Af
//! réduction max (idéale) r_max = 1 − 1/e ≈ 0,632   (σd = Ȳ)
//! ```
//!
//! `A0`/`Af` aires entrée/sortie (m²), `Ȳ` contrainte d'écoulement **moyenne**
//! (Pa), `σd` contrainte d'étirage (Pa), `F` effort de traction sur le fil (N).
//! La contrainte d'étirage ne doit pas dépasser la limite d'écoulement du fil en
//! sortie, d'où la réduction maximale par passe.
//!
//! **Convention** : SI cohérent. **Limite honnête** : modèle de déformation
//! **homogène** (idéal, sans frottement ni déformation redondante) — le vrai
//! effort est supérieur (facteurs de Sachs/frottement d'angle de filière). `Ȳ`
//! (courbe d'écrouissage) est fourni par l'appelant.

use core::f64::consts::E;

/// Réduction maximale de section idéale par passe `1 − 1/e ≈ 0,632`.
pub const MAX_REDUCTION_IDEAL: f64 = 1.0 - 1.0 / E;

/// Réduction de section `r = (A0 − Af)/A0`.
///
/// Panique si `initial_area <= 0`.
pub fn area_reduction(initial_area: f64, final_area: f64) -> f64 {
    assert!(
        initial_area > 0.0,
        "l'aire d'entrée doit être strictement positive"
    );
    (initial_area - final_area) / initial_area
}

/// Déformation vraie `ε = ln(A0/Af)`.
///
/// Panique si `initial_area <= 0` ou `final_area <= 0`.
pub fn drawing_true_strain(initial_area: f64, final_area: f64) -> f64 {
    assert!(
        initial_area > 0.0 && final_area > 0.0,
        "A0 > 0 et Af > 0 requis"
    );
    (initial_area / final_area).ln()
}

/// Contrainte d'étirage idéale `σd = Ȳ·ln(A0/Af)` (Pa).
pub fn drawing_stress(avg_flow_stress: f64, initial_area: f64, final_area: f64) -> f64 {
    avg_flow_stress * drawing_true_strain(initial_area, final_area)
}

/// Effort d'étirage `F = σd·Af` (N).
pub fn drawing_force(drawing_stress: f64, final_area: f64) -> f64 {
    drawing_stress * final_area
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn reduction_and_strain() {
        // A0=100, Af=80 mm² → r=0,2 ; ε = ln(1,25).
        assert_relative_eq!(area_reduction(100e-6, 80e-6), 0.2, epsilon = 1e-9);
        assert_relative_eq!(
            drawing_true_strain(100e-6, 80e-6),
            1.25f64.ln(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn ideal_max_reduction_is_63_percent() {
        // À r_max, ε=1 → σd = Ȳ (limite de faisabilité idéale).
        assert_relative_eq!(
            MAX_REDUCTION_IDEAL,
            1.0 - 1.0 / core::f64::consts::E,
            epsilon = 1e-12
        );
        // Une réduction de 63,2 % donne A0/Af = e, donc ε = 1.
        let af = 1.0 - MAX_REDUCTION_IDEAL; // Af/A0
        assert_relative_eq!(drawing_true_strain(1.0, af), 1.0, max_relative = 1e-9);
    }

    #[test]
    fn drawing_stress_below_flow_stress_for_small_reduction() {
        // Réduction modérée → σd < Ȳ (étirage possible).
        let sd = drawing_stress(400e6, 100e-6, 80e-6);
        assert!(sd < 400e6);
        assert_relative_eq!(sd, 400e6 * 1.25f64.ln(), epsilon = 1e-3);
    }

    #[test]
    fn force_is_stress_times_exit_area() {
        let sd = drawing_stress(400e6, 100e-6, 80e-6);
        assert_relative_eq!(drawing_force(sd, 80e-6), sd * 80e-6, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "A0 > 0")]
    fn zero_area_strain_panics() {
        drawing_true_strain(0.0, 80e-6);
    }
}
