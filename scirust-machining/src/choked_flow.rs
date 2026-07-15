//! Écoulement compressible en tuyère — blocage sonique (col amorcé) d'un gaz parfait.
//!
//! ```text
//! rapport de pression critique      p*/p0 = (2/(γ+1))^(γ/(γ-1))
//! rapport de température critique    T*/T0 = 2/(γ+1)
//! blocage                           amorcé si  pb/p0 ≤ p*/p0
//! débit massique bloqué             ṁ = A·p0·√(γ/(R·T0))·(2/(γ+1))^((γ+1)/(2·(γ-1)))
//! ```
//!
//! `γ` rapport des chaleurs spécifiques (sans dimension, γ > 1), `p0` pression
//! d'arrêt (génératrice) (Pa), `T0` température d'arrêt (K), `pb` contre-pression
//! aval (Pa), `A` section du col (m²), `R` constante spécifique du gaz
//! (J·kg⁻¹·K⁻¹), `ṁ` débit massique au col (kg/s), `p*`/`T*` conditions soniques
//! au col.
//!
//! **Convention** : SI cohérent. **Limite honnête** : gaz parfait, écoulement
//! **isentropique 1D** ; une fois le col amorcé, `ṁ` est **indépendant de la
//! contre-pression aval**. Les propriétés physiques du gaz (`γ`, `R`) et les
//! conditions de procédé (`p0`, `T0`, `A`, `pb`) sont **fournies par l'appelant**
//! — aucune valeur matériau, fluide ou procédé n'est supposée par défaut.

/// Rapport de pression critique au col `p*/p0 = (2/(γ+1))^(γ/(γ-1))` (sans dimension).
///
/// Panique si `gamma <= 1`.
pub fn choked_critical_pressure_ratio(gamma: f64) -> f64 {
    assert!(
        gamma > 1.0,
        "le rapport des chaleurs spécifiques doit être strictement supérieur à 1"
    );
    (2.0_f64 / (gamma + 1.0)).powf(gamma / (gamma - 1.0))
}

/// Rapport de température critique au col `T*/T0 = 2/(γ+1)` (sans dimension).
///
/// Panique si `gamma <= 1`.
pub fn choked_critical_temperature_ratio(gamma: f64) -> f64 {
    assert!(
        gamma > 1.0,
        "le rapport des chaleurs spécifiques doit être strictement supérieur à 1"
    );
    2.0 / (gamma + 1.0)
}

/// Indique si le col est amorcé : `true` si `pb/p0 ≤ p*/p0` (sans dimension).
///
/// Panique si `back_pressure < 0`, `stagnation_pressure <= 0` ou `gamma <= 1`.
pub fn choked_is_choked(back_pressure: f64, stagnation_pressure: f64, gamma: f64) -> bool {
    assert!(
        back_pressure >= 0.0,
        "la contre-pression doit être positive ou nulle"
    );
    assert!(
        stagnation_pressure > 0.0,
        "la pression d'arrêt doit être strictement positive"
    );
    assert!(
        gamma > 1.0,
        "le rapport des chaleurs spécifiques doit être strictement supérieur à 1"
    );
    back_pressure / stagnation_pressure <= choked_critical_pressure_ratio(gamma)
}

/// Débit massique bloqué au col
/// `ṁ = A·p0·√(γ/(R·T0))·(2/(γ+1))^((γ+1)/(2·(γ-1)))` (kg/s).
///
/// Panique si `stagnation_pressure < 0`, `throat_area < 0`,
/// `stagnation_temperature <= 0`, `gas_constant <= 0` ou `gamma <= 1`.
pub fn choked_mass_flow(
    stagnation_pressure: f64,
    throat_area: f64,
    stagnation_temperature: f64,
    gamma: f64,
    gas_constant: f64,
) -> f64 {
    assert!(
        stagnation_pressure >= 0.0,
        "la pression d'arrêt doit être positive ou nulle"
    );
    assert!(
        throat_area >= 0.0,
        "la section du col doit être positive ou nulle"
    );
    assert!(
        stagnation_temperature > 0.0,
        "la température d'arrêt doit être strictement positive"
    );
    assert!(
        gas_constant > 0.0,
        "la constante du gaz doit être strictement positive"
    );
    assert!(
        gamma > 1.0,
        "le rapport des chaleurs spécifiques doit être strictement supérieur à 1"
    );
    let exponent = (gamma + 1.0) / (2.0 * (gamma - 1.0));
    throat_area
        * stagnation_pressure
        * (gamma / (gas_constant * stagnation_temperature)).sqrt()
        * (2.0_f64 / (gamma + 1.0)).powf(exponent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn critical_pressure_ratio_air() {
        // γ = 1,4 → γ/(γ-1) = 3,5, donc p*/p0 = (5/6)^3,5.
        assert_relative_eq!(
            choked_critical_pressure_ratio(1.4),
            (5.0_f64 / 6.0).powf(3.5),
            epsilon = 1e-12
        );
    }

    #[test]
    fn critical_temperature_ratio_air() {
        // γ = 1,4 → T*/T0 = 2/2,4 = 5/6.
        assert_relative_eq!(
            choked_critical_temperature_ratio(1.4),
            5.0 / 6.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn is_choked_threshold() {
        // Au seuil : pb/p0 exactement égal au rapport critique → amorcé (≤).
        let ratio = choked_critical_pressure_ratio(1.4);
        let p0 = 1.0e6;
        assert!(choked_is_choked(ratio * p0, p0, 1.4));
        // Juste en dessous du rapport → amorcé.
        assert!(choked_is_choked(0.5 * ratio * p0, p0, 1.4));
        // Au dessus du rapport → non amorcé.
        assert!(!choked_is_choked(0.99 * p0, p0, 1.4));
    }

    #[test]
    fn mass_flow_proportional_to_area_and_pressure() {
        // ṁ linéaire en A et en p0 (facteurs indépendants).
        let base = choked_mass_flow(2.0e6, 1.0e-3, 300.0, 1.4, 287.0);
        let double_area = choked_mass_flow(2.0e6, 2.0e-3, 300.0, 1.4, 287.0);
        let double_pressure = choked_mass_flow(4.0e6, 1.0e-3, 300.0, 1.4, 287.0);
        assert_relative_eq!(double_area, 2.0 * base, epsilon = 1e-9);
        assert_relative_eq!(double_pressure, 2.0 * base, epsilon = 1e-9);
    }

    #[test]
    fn mass_flow_reference_case() {
        // γ = 1,4 → exposant (γ+1)/(2(γ-1)) = 2,4/0,8 = 3.
        // Reconstruction indépendante avec powi(3) et √(γ/(R·T0)).
        let expected = (1.4_f64 / (287.0 * 350.0)).sqrt() * (5.0_f64 / 6.0).powi(3);
        assert_relative_eq!(
            choked_mass_flow(1.0, 1.0, 350.0, 1.4, 287.0),
            expected,
            epsilon = 1e-12
        );
    }

    #[test]
    fn mass_flow_scales_with_inverse_sqrt_temperature() {
        // ṁ ∝ 1/√T0 : quadrupler T0 divise le débit par 2.
        let base = choked_mass_flow(1.0e6, 1.0e-3, 300.0, 1.33, 296.0);
        let hot = choked_mass_flow(1.0e6, 1.0e-3, 1200.0, 1.33, 296.0);
        assert_relative_eq!(hot, base / 2.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "strictement supérieur à 1")]
    fn gamma_below_one_panics() {
        choked_critical_pressure_ratio(1.0);
    }
}
