//! Dilatation thermique et contrainte de bridage d'une pièce chauffée ou
//! refroidie sous encastrement.
//!
//! ```text
//! allongement linéique    dL = L·α·ΔT                (m)
//! dilatation volumique    dV = V·3·α·ΔT              (m³, isotrope)
//! déformation libre       ε  = α·ΔT                  (sans dimension)
//! contrainte de bridage   σ  = E·α·ΔT                (Pa, encastrement total)
//! ```
//!
//! `L` longueur initiale (m), `V` volume initial (m³), `α` coefficient de
//! dilatation thermique **linéique** (K⁻¹), `ΔT` écart de température (K = °C),
//! `E` module de Young (Pa), `dL` allongement (m), `dV` variation de volume (m³),
//! `ε` déformation libre (sans dimension), `σ` contrainte de bridage (Pa,
//! positive en compression quand `ΔT > 0` par convention de valeur absolue).
//!
//! **Convention** : unités SI ; `ΔT` est un **écart** de température (K = °C),
//! signé (positif en chauffage, négatif en refroidissement).
//! **Limite honnête** : le coefficient de dilatation linéique `α` et le module
//! `E` sont **fournis par l'appelant** — aucune valeur « par défaut » de
//! matériau, de procédé ou de constante physique n'est inventée. On suppose `α`
//! **constant** sur toute la plage de température (pas de dépendance en `T`), la
//! dilatation volumique **isotrope** (`3·α`), et la contrainte de bridage un
//! **encastrement parfait** restant dans le **domaine élastique** (pas de
//! plastification, de flambage ni de relaxation).

/// Allongement par dilatation thermique linéique `dL = L·α·ΔT` (m).
///
/// `ΔT` est signé : `dL > 0` en chauffage (`ΔT > 0`), `dL < 0` en refroidissement.
///
/// Panique si `length < 0`.
pub fn thermal_linear_expansion(
    length: f64,
    expansion_coefficient: f64,
    temperature_change: f64,
) -> f64 {
    assert!(length >= 0.0, "longueur L ≥ 0 requise");
    length * expansion_coefficient * temperature_change
}

/// Variation de volume par dilatation thermique isotrope `dV = V·3·α·ΔT` (m³),
/// où le coefficient volumique vaut `3·α` (matériau isotrope, `α` linéique).
///
/// Panique si `volume < 0`.
pub fn thermal_volumetric_expansion(
    volume: f64,
    expansion_coefficient: f64,
    temperature_change: f64,
) -> f64 {
    assert!(volume >= 0.0, "volume V ≥ 0 requis");
    volume * 3.0 * expansion_coefficient * temperature_change
}

/// Déformation thermique libre `ε = α·ΔT` (sans dimension), dilatation d'une
/// pièce non contrainte rapportée à sa longueur.
///
/// Aucune restriction de signe : `ε` suit le signe de `α·ΔT`. Ne panique jamais.
pub fn thermal_free_strain(expansion_coefficient: f64, temperature_change: f64) -> f64 {
    expansion_coefficient * temperature_change
}

/// Contrainte de bridage sous encastrement total `σ = E·α·ΔT` (Pa) : contrainte
/// développée lorsqu'une pièce empêchée de se dilater subit l'écart `ΔT`.
///
/// `ΔT` est signé : `σ` change de signe entre chauffage (compression) et
/// refroidissement (traction).
///
/// Panique si `youngs_modulus <= 0`.
pub fn thermal_constrained_stress(
    youngs_modulus: f64,
    expansion_coefficient: f64,
    temperature_change: f64,
) -> f64 {
    assert!(youngs_modulus > 0.0, "module de Young E > 0 requis");
    youngs_modulus * expansion_coefficient * temperature_change
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn linear_expansion_realistic_case() {
        // Barre d'acier L=2 m, α=12e-6 K⁻¹, ΔT=50 K.
        // dL = 2·12e-6·50 = 1.2e-3 m = 1.2 mm.
        let dl = thermal_linear_expansion(2.0, 12e-6, 50.0);
        assert_relative_eq!(dl, 1.2e-3, max_relative = 1e-12);
    }

    #[test]
    fn volumetric_is_three_times_free_strain_times_volume() {
        // dV = V·3·α·ΔT = 3·V·ε : identité entre dilatation volumique et
        // déformation libre. V=1 m³, α=12e-6, ΔT=50 K.
        let eps = thermal_free_strain(12e-6, 50.0);
        let dv = thermal_volumetric_expansion(1.0, 12e-6, 50.0);
        assert_relative_eq!(dv, 3.0 * 1.0 * eps, max_relative = 1e-12);
    }

    #[test]
    fn linear_expansion_equals_length_times_free_strain() {
        // dL = L·ε : cohérence entre allongement et déformation libre.
        let eps = thermal_free_strain(12e-6, 50.0);
        let dl = thermal_linear_expansion(2.0, 12e-6, 50.0);
        assert_relative_eq!(dl, 2.0 * eps, max_relative = 1e-12);
    }

    #[test]
    fn constrained_stress_realistic_case() {
        // Acier bridé : E=210 GPa, α=12e-6 K⁻¹, ΔT=50 K.
        // σ = 210e9·12e-6·50 = 210e9·6e-4 = 1.26e8 Pa = 126 MPa.
        let sigma = thermal_constrained_stress(210e9, 12e-6, 50.0);
        assert_relative_eq!(sigma, 126e6, max_relative = 1e-12);
    }

    #[test]
    fn constrained_stress_is_modulus_times_free_strain() {
        // σ = E·ε : la contrainte de bridage est le module fois la déformation
        // libre empêchée (loi de Hooke sur la dilatation contrariée).
        let eps = thermal_free_strain(12e-6, 50.0);
        let sigma = thermal_constrained_stress(210e9, 12e-6, 50.0);
        assert_relative_eq!(sigma, 210e9 * eps, max_relative = 1e-12);
    }

    #[test]
    fn sign_reverses_on_cooling() {
        // Réciprocité de signe : refroidir (-ΔT) inverse allongement et contrainte.
        let dl_heat = thermal_linear_expansion(2.0, 12e-6, 50.0);
        let dl_cool = thermal_linear_expansion(2.0, 12e-6, -50.0);
        assert_relative_eq!(dl_cool, -dl_heat, max_relative = 1e-12);
        let s_heat = thermal_constrained_stress(210e9, 12e-6, 50.0);
        let s_cool = thermal_constrained_stress(210e9, 12e-6, -50.0);
        assert_relative_eq!(s_cool, -s_heat, max_relative = 1e-12);
    }

    #[test]
    #[should_panic(expected = "E > 0")]
    fn non_positive_modulus_panics() {
        thermal_constrained_stress(0.0, 12e-6, 50.0);
    }
}
