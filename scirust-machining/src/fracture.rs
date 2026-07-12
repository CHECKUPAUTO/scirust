//! Mécanique de la **rupture** — facteur d'intensité de contrainte, taille
//! critique de fissure, marge vis-à-vis de la ténacité et contrainte de
//! **Griffith**.
//!
//! ```text
//! intensité de contrainte  K = Y·σ·√(π·a)
//! taille critique          a_c = (1/π)·(K_Ic/(Y·σ))²
//! coefficient de sécurité  n = K_Ic/K
//! Griffith (fragile)       σ_f = √(2·E·γs/(π·a))
//! ```
//!
//! `K` facteur d'intensité de contrainte (mode I, Pa·√m), `Y` facteur de forme
//! (géométrie, ~1,12 pour une fissure de bord), `σ` contrainte nominale (Pa),
//! `a` longueur de fissure (m), `K_Ic` ténacité (Pa·√m), `E` module de Young,
//! `γs` énergie de surface (J/m²).
//!
//! **Convention** : SI cohérent (`K` en Pa·√m). **Limite honnête** : mécanique
//! **linéaire élastique** de la rupture (LEFM), petite zone plastique, mode I ;
//! `Y` et `K_Ic` dépendent de la géométrie et du matériau et sont fournis par
//! l'appelant. La rupture survient quand `K ≥ K_Ic`.

use core::f64::consts::PI;

/// Facteur d'intensité de contrainte `K = Y·σ·√(π·a)` (Pa·√m).
///
/// Panique si `crack_length < 0`.
pub fn stress_intensity(shape_factor: f64, stress: f64, crack_length: f64) -> f64 {
    assert!(
        crack_length >= 0.0,
        "la longueur de fissure doit être positive"
    );
    shape_factor * stress * (PI * crack_length).sqrt()
}

/// Taille de fissure critique `a_c = (1/π)·(K_Ic/(Y·σ))²` (m).
///
/// Panique si `shape_factor*stress <= 0`.
pub fn critical_crack_length(fracture_toughness: f64, shape_factor: f64, stress: f64) -> f64 {
    let denom = shape_factor * stress;
    assert!(denom > 0.0, "Y·σ doit être strictement positif");
    let ratio = fracture_toughness / denom;
    ratio * ratio / PI
}

/// Coefficient de sécurité vis-à-vis de la rupture brutale `n = K_Ic/K`.
///
/// Panique si `applied_k <= 0`.
pub fn fracture_safety_factor(fracture_toughness: f64, applied_k: f64) -> f64 {
    assert!(
        applied_k > 0.0,
        "le facteur d'intensité appliqué doit être positif"
    );
    fracture_toughness / applied_k
}

/// Contrainte de rupture fragile de **Griffith** `σ_f = √(2·E·γs/(π·a))` (Pa).
///
/// Panique si `crack_length <= 0`.
pub fn griffith_stress(youngs_modulus: f64, surface_energy: f64, crack_length: f64) -> f64 {
    assert!(
        crack_length > 0.0,
        "la longueur de fissure doit être strictement positive"
    );
    (2.0 * youngs_modulus * surface_energy / (PI * crack_length)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn stress_intensity_grows_with_crack() {
        // K ∝ √a : quadrupler la fissure double K.
        let k1 = stress_intensity(1.0, 100e6, 0.001);
        let k2 = stress_intensity(1.0, 100e6, 0.004);
        assert_relative_eq!(k2 / k1, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn critical_length_reaches_toughness() {
        // À a = a_c, le K appliqué doit exactement valoir K_Ic.
        let (kic, y, sigma) = (30e6, 1.12, 200e6);
        let ac = critical_crack_length(kic, y, sigma);
        assert_relative_eq!(stress_intensity(y, sigma, ac), kic, max_relative = 1e-9);
    }

    #[test]
    fn safety_factor_from_toughness() {
        // K_Ic=50, K=25 → n=2.
        assert_relative_eq!(fracture_safety_factor(50e6, 25e6), 2.0, epsilon = 1e-12);
    }

    #[test]
    fn griffith_stress_falls_with_crack_length() {
        // σ_f ∝ 1/√a : quadrupler a divise σ_f par 2.
        let s1 = griffith_stress(70e9, 1.0, 1e-6);
        let s2 = griffith_stress(70e9, 1.0, 4e-6);
        assert_relative_eq!(s1 / s2, 2.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "Y·σ")]
    fn zero_stress_critical_length_panics() {
        critical_crack_length(30e6, 1.12, 0.0);
    }
}
