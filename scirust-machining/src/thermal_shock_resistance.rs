//! Résistance au choc thermique — paramètres de **Kingery** `R` et `R'`
//! (écart de température maximal supportable lors d'une trempe sévère).
//!
//! ```text
//! paramètre R   R  = σ·(1 - ν)/(E·α)          (K, ΔT max en trempe sévère)
//! paramètre R'  R' = R·k                        (W·m⁻¹, avec conductivité)
//! ΔT max        ΔT_max = R                       (identité, réécriture de R)
//! ```
//!
//! `σ` résistance à la traction du matériau (Pa), `ν` coefficient de Poisson
//! (sans dimension), `E` module de Young (Pa), `α` coefficient de dilatation
//! thermique linéique (K⁻¹), `k` conductivité thermique (W·m⁻¹·K⁻¹), `R` premier
//! paramètre de résistance au choc thermique (K), `R'` second paramètre (W·m⁻¹),
//! `ΔT_max` écart de température maximal admissible (K).
//!
//! **Convention** : unités SI ; les températures sont des **écarts** (K = °C).
//! **Limite honnête** : `R` est le paramètre de **Kingery** valable pour une
//! trempe **sévère** (nombre de Biot infini, choc surfacique instantané) ;
//! toutes les propriétés (σ, ν, E, α, k) sont **fournies par l'appelant** — aucune
//! valeur « par défaut » n'est inventée. Ces paramètres classent des matériaux
//! entre eux ; ils **ne prédisent pas** la fissuration réelle d'une pièce
//! (géométrie, gradient, défauts et tenacité ne sont pas modélisés).

/// Premier paramètre de résistance au choc thermique de Kingery
/// `R = σ·(1 - ν)/(E·α)` (K).
///
/// Panique si `tensile_strength < 0`, `poisson_ratio ∉ [0, 0.5[`,
/// `youngs_modulus <= 0` ou `thermal_expansion <= 0`.
pub fn tsr_parameter_r(
    tensile_strength: f64,
    poisson_ratio: f64,
    youngs_modulus: f64,
    thermal_expansion: f64,
) -> f64 {
    assert!(tensile_strength >= 0.0, "σ ≥ 0 requis");
    assert!((0.0..0.5).contains(&poisson_ratio), "ν ∈ [0, 0.5[ requis");
    assert!(youngs_modulus > 0.0, "E > 0 requis");
    assert!(thermal_expansion > 0.0, "α > 0 requis");
    tensile_strength * (1.0 - poisson_ratio) / (youngs_modulus * thermal_expansion)
}

/// Second paramètre de résistance au choc thermique `R' = R·k` (W·m⁻¹),
/// pondéré par la conductivité thermique.
///
/// Panique si `r_parameter < 0` ou `thermal_conductivity < 0`.
pub fn tsr_parameter_r_prime(r_parameter: f64, thermal_conductivity: f64) -> f64 {
    assert!(
        r_parameter >= 0.0 && thermal_conductivity >= 0.0,
        "R ≥ 0 et k ≥ 0 requis"
    );
    r_parameter * thermal_conductivity
}

/// Écart de température maximal admissible en trempe sévère
/// `ΔT_max = R` (K) — réécriture explicite du paramètre `R`.
///
/// Panique si `r_parameter < 0`.
pub fn tsr_max_temperature_difference(r_parameter: f64) -> f64 {
    assert!(r_parameter >= 0.0, "R ≥ 0 requis");
    r_parameter
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn parameter_r_realistic_case() {
        // Cas chiffré : σ=200 MPa, ν=0.25, E=300 GPa, α=10e-6 K⁻¹.
        // R = 200e6·0.75/(300e9·10e-6) = 150e6/3e6 = 50 K.
        let r = tsr_parameter_r(200e6, 0.25, 300e9, 10e-6);
        assert_relative_eq!(r, 50.0, max_relative = 1e-12);
    }

    #[test]
    fn parameter_r_proportional_to_strength() {
        // R ∝ σ : doubler la résistance double R.
        let r1 = tsr_parameter_r(200e6, 0.25, 300e9, 10e-6);
        let r2 = tsr_parameter_r(400e6, 0.25, 300e9, 10e-6);
        assert_relative_eq!(r2 / r1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn parameter_r_inversely_proportional_to_expansion() {
        // R ∝ 1/α : doubler α divise R par deux.
        let r1 = tsr_parameter_r(200e6, 0.25, 300e9, 10e-6);
        let r2 = tsr_parameter_r(200e6, 0.25, 300e9, 20e-6);
        assert_relative_eq!(r1 / r2, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn parameter_r_prime_is_r_times_conductivity() {
        // R'=R·k : R=50 K, k=40 W·m⁻¹·K⁻¹ → R'=2000 W·m⁻¹.
        let r = tsr_parameter_r(200e6, 0.25, 300e9, 10e-6);
        assert_relative_eq!(tsr_parameter_r_prime(r, 40.0), 2000.0, max_relative = 1e-12);
    }

    #[test]
    fn max_temperature_difference_equals_r() {
        // Identité : ΔT_max = R.
        let r = tsr_parameter_r(200e6, 0.25, 300e9, 10e-6);
        assert_relative_eq!(tsr_max_temperature_difference(r), r, max_relative = 1e-15);
    }

    #[test]
    #[should_panic(expected = "α > 0")]
    fn zero_expansion_panics() {
        tsr_parameter_r(200e6, 0.25, 300e9, 0.0);
    }
}
