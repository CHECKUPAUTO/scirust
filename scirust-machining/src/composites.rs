//! Matériaux composites — **règle des mélanges** : bornes de module de **Voigt**
//! (iso-déformation) et de **Reuss** (iso-contrainte), masse volumique et
//! résistance longitudinale.
//!
//! ```text
//! Voigt (parallèle, // fibres)  E_L = Vf·Ef + (1 − Vf)·Em      (borne haute)
//! Reuss (série, ⟂ fibres)       E_T = 1/(Vf/Ef + (1 − Vf)/Em)  (borne basse)
//! masse volumique               ρ = Vf·ρf + (1 − Vf)·ρm
//! résistance longitudinale      σ_L = Vf·σf + (1 − Vf)·σm
//! ```
//!
//! `Vf` fraction volumique de fibres (`0`–`1`), `Ef`/`Em` modules fibre/matrice
//! (Pa), `ρf`/`ρm` masses volumiques, `σf`/`σm` résistances. Le module réel est
//! encadré par Voigt (chargement parallèle aux fibres) et Reuss (perpendiculaire).
//!
//! **Convention** : SI cohérent. **Limite honnête** : règle des mélanges pour un
//! **pli unidirectionnel** ; Voigt/Reuss sont des **bornes** (Hill), le
//! comportement transverse réel est plus proche de modèles semi-empiriques
//! (Halpin-Tsai). Adhérence fibre-matrice parfaite supposée.

/// Module de **Voigt** (parallèle) `E_L = Vf·Ef + (1 − Vf)·Em` (Pa).
///
/// Panique si `fiber_fraction` hors `[0, 1]`.
pub fn voigt_modulus(fiber_fraction: f64, fiber_modulus: f64, matrix_modulus: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&fiber_fraction),
        "Vf doit être dans [0, 1]"
    );
    fiber_fraction * fiber_modulus + (1.0 - fiber_fraction) * matrix_modulus
}

/// Module de **Reuss** (série) `E_T = 1/(Vf/Ef + (1 − Vf)/Em)` (Pa).
///
/// Panique si `fiber_fraction` hors `[0, 1]` ou si un module est `<= 0`.
pub fn reuss_modulus(fiber_fraction: f64, fiber_modulus: f64, matrix_modulus: f64) -> f64 {
    assert!(
        (0.0..=1.0).contains(&fiber_fraction),
        "Vf doit être dans [0, 1]"
    );
    assert!(
        fiber_modulus > 0.0 && matrix_modulus > 0.0,
        "Ef > 0 et Em > 0 requis"
    );
    1.0 / (fiber_fraction / fiber_modulus + (1.0 - fiber_fraction) / matrix_modulus)
}

/// Masse volumique du composite `ρ = Vf·ρf + (1 − Vf)·ρm`.
///
/// Panique si `fiber_fraction` hors `[0, 1]`.
pub fn rule_of_mixtures_density(
    fiber_fraction: f64,
    fiber_density: f64,
    matrix_density: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&fiber_fraction),
        "Vf doit être dans [0, 1]"
    );
    fiber_fraction * fiber_density + (1.0 - fiber_fraction) * matrix_density
}

/// Résistance longitudinale `σ_L = Vf·σf + (1 − Vf)·σm` (Pa).
///
/// Panique si `fiber_fraction` hors `[0, 1]`.
pub fn longitudinal_strength(
    fiber_fraction: f64,
    fiber_strength: f64,
    matrix_strength: f64,
) -> f64 {
    assert!(
        (0.0..=1.0).contains(&fiber_fraction),
        "Vf doit être dans [0, 1]"
    );
    fiber_fraction * fiber_strength + (1.0 - fiber_fraction) * matrix_strength
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn voigt_is_upper_bound_reuss_lower() {
        // Verre/époxy : Ef=72 GPa, Em=3,5 GPa, Vf=0,6.
        let voigt = voigt_modulus(0.6, 72e9, 3.5e9);
        let reuss = reuss_modulus(0.6, 72e9, 3.5e9);
        assert!(voigt > reuss);
        assert_relative_eq!(voigt, 0.6 * 72e9 + 0.4 * 3.5e9, epsilon = 1e-3);
    }

    #[test]
    fn pure_matrix_and_pure_fiber_limits() {
        // Vf=0 → module de la matrice ; Vf=1 → module de la fibre (Voigt et Reuss).
        assert_relative_eq!(voigt_modulus(0.0, 72e9, 3.5e9), 3.5e9, epsilon = 1e-3);
        assert_relative_eq!(reuss_modulus(1.0, 72e9, 3.5e9), 72e9, max_relative = 1e-12);
    }

    #[test]
    fn density_rule_of_mixtures() {
        // Vf=0,6, ρf=2540, ρm=1200 → ρ = 0,6·2540 + 0,4·1200 = 2004 kg/m³.
        assert_relative_eq!(
            rule_of_mixtures_density(0.6, 2540.0, 1200.0),
            2004.0,
            epsilon = 1e-6
        );
    }

    #[test]
    fn strength_grows_with_fiber_content() {
        assert!(
            longitudinal_strength(0.7, 2000e6, 60e6) > longitudinal_strength(0.3, 2000e6, 60e6)
        );
    }

    #[test]
    #[should_panic(expected = "Vf doit être dans")]
    fn invalid_fraction_panics() {
        voigt_modulus(1.5, 72e9, 3.5e9);
    }
}
