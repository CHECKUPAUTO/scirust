//! **Module Peltier** — réfrigération thermoélectrique d'un module à effet
//! Peltier : puissance froide extraite, puissance électrique consommée,
//! coefficient de performance et écart de température maximal.
//!
//! ```text
//! puissance froide      Qc = S·I·Tc − ½·I²·R − K·ΔT
//! puissance électrique   Pe = S·I·ΔT + I²·R
//! coeff. de performance  COP = Qc/Pe
//! écart max              ΔT_max = ½·Z·Tc²
//! ```
//!
//! `S` coefficient de Seebeck du couple (V·K⁻¹), `I` courant traversant le
//! module (A), `Tc` température de la face froide (K), `R` résistance électrique
//! interne (Ω), `K` conductance thermique du module (W·K⁻¹), `ΔT = Th − Tc`
//! écart de température entre faces chaude et froide (K), `Qc` puissance froide
//! extraite à la face froide (W), `Pe` puissance électrique consommée (W), `COP`
//! coefficient de performance (sans dimension), `Z` facteur de mérite (K⁻¹),
//! `ΔT_max` écart de température maximal atteignable à Qc = 0 (K).
//!
//! **Convention** : SI, **températures en KELVIN**. **Limite honnête** : modèle
//! **à propriétés constantes**, l'**effet Thomson est négligé** et la moitié de
//! l'effet Joule est supposée revenir à la face froide. Le coefficient de
//! Seebeck du couple `S`, la résistance électrique `R`, la conductance thermique
//! `K` et le facteur de mérite `Z` sont **fournis par l'appelant** (mesurés ou
//! issus de la fiche du module) ; aucune valeur matériau/procédé « par défaut »
//! n'est inventée.

/// Puissance froide extraite à la face froide
/// `Qc = S·I·Tc − ½·I²·R − K·ΔT`.
///
/// Panique si `seebeck_coefficient <= 0`, `current < 0`,
/// `cold_side_temperature <= 0`, `electrical_resistance < 0` ou
/// `thermal_conductance < 0`.
pub fn peltier_cooling_power(
    seebeck_coefficient: f64,
    current: f64,
    cold_side_temperature: f64,
    electrical_resistance: f64,
    thermal_conductance: f64,
    temperature_difference: f64,
) -> f64 {
    assert!(
        seebeck_coefficient > 0.0,
        "le coefficient de Seebeck S doit être positif"
    );
    assert!(current >= 0.0, "le courant I doit être positif ou nul");
    assert!(
        cold_side_temperature > 0.0,
        "la température de face froide Tc doit être positive (kelvin)"
    );
    assert!(
        electrical_resistance >= 0.0,
        "la résistance électrique R doit être positive ou nulle"
    );
    assert!(
        thermal_conductance >= 0.0,
        "la conductance thermique K doit être positive ou nulle"
    );
    seebeck_coefficient * current * cold_side_temperature
        - 0.5 * current * current * electrical_resistance
        - thermal_conductance * temperature_difference
}

/// Puissance électrique consommée par le module
/// `Pe = S·I·ΔT + I²·R`.
///
/// Panique si `seebeck_coefficient <= 0`, `current < 0` ou
/// `electrical_resistance < 0`.
pub fn peltier_electrical_power(
    seebeck_coefficient: f64,
    current: f64,
    temperature_difference: f64,
    electrical_resistance: f64,
) -> f64 {
    assert!(
        seebeck_coefficient > 0.0,
        "le coefficient de Seebeck S doit être positif"
    );
    assert!(current >= 0.0, "le courant I doit être positif ou nul");
    assert!(
        electrical_resistance >= 0.0,
        "la résistance électrique R doit être positive ou nulle"
    );
    seebeck_coefficient * current * temperature_difference
        + current * current * electrical_resistance
}

/// Coefficient de performance en froid `COP = Qc/Pe`.
///
/// Panique si `electrical_power <= 0`.
pub fn peltier_cop(cooling_power: f64, electrical_power: f64) -> f64 {
    assert!(
        electrical_power > 0.0,
        "la puissance électrique Pe doit être positive (dénominateur non nul)"
    );
    cooling_power / electrical_power
}

/// Écart de température maximal atteignable (à Qc = 0)
/// `ΔT_max = ½·Z·Tc²`.
///
/// Panique si `figure_of_merit < 0` ou `cold_side_temperature <= 0`.
pub fn peltier_max_temperature_difference(figure_of_merit: f64, cold_side_temperature: f64) -> f64 {
    assert!(
        figure_of_merit >= 0.0,
        "le facteur de mérite Z doit être positif ou nul"
    );
    assert!(
        cold_side_temperature > 0.0,
        "la température de face froide Tc doit être positive (kelvin)"
    );
    0.5 * figure_of_merit * cold_side_temperature * cold_side_temperature
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn cooling_power_realistic_case() {
        // S=0,05 V/K, I=5 A, Tc=300 K, R=2 Ω, K=0,5 W/K, ΔT=40 K.
        // Qc = 0,05·5·300 − 0,5·25·2 − 0,5·40 = 75 − 25 − 20 = 30 W.
        let qc = peltier_cooling_power(0.05, 5.0, 300.0, 2.0, 0.5, 40.0);
        assert_relative_eq!(qc, 30.0, epsilon = 1e-9);
    }

    #[test]
    fn electrical_power_realistic_case() {
        // Pe = S·I·ΔT + I²·R = 0,05·5·40 + 25·2 = 10 + 50 = 60 W.
        let pe = peltier_electrical_power(0.05, 5.0, 40.0, 2.0);
        assert_relative_eq!(pe, 60.0, epsilon = 1e-9);
    }

    #[test]
    fn cop_is_ratio_and_reconstructs_cooling_power() {
        // COP = Qc/Pe = 30/60 = 0,5 ; identité Qc = COP·Pe.
        let qc = peltier_cooling_power(0.05, 5.0, 300.0, 2.0, 0.5, 40.0);
        let pe = peltier_electrical_power(0.05, 5.0, 40.0, 2.0);
        let cop = peltier_cop(qc, pe);
        assert_relative_eq!(cop, 0.5, epsilon = 1e-9);
        assert_relative_eq!(cop * pe, qc, epsilon = 1e-9);
    }

    #[test]
    fn max_temperature_difference_scales_with_tc_squared() {
        // ΔT_max ∝ Tc² : doubler Tc quadruple l'écart maximal.
        let d1 = peltier_max_temperature_difference(0.001_6, 150.0);
        let d2 = peltier_max_temperature_difference(0.001_6, 300.0);
        assert_relative_eq!(d2, 4.0 * d1, epsilon = 1e-9);
    }

    #[test]
    fn max_temperature_difference_realistic_case() {
        // Z=0,0016 K⁻¹, Tc=300 K → ΔT_max = 0,5·0,0016·90 000 = 72 K.
        let dtmax = peltier_max_temperature_difference(0.001_6, 300.0);
        assert_relative_eq!(dtmax, 72.0, epsilon = 1e-9);
    }

    #[test]
    fn cooling_power_decreases_with_conduction() {
        // Augmenter la conductance K réduit la puissance froide extraite.
        let qc_low = peltier_cooling_power(0.05, 5.0, 300.0, 2.0, 0.5, 40.0);
        let qc_high = peltier_cooling_power(0.05, 5.0, 300.0, 2.0, 1.0, 40.0);
        assert!(qc_high < qc_low);
        // Écart = (K_high − K_low)·ΔT = 0,5·40 = 20 W.
        assert_relative_eq!(qc_low - qc_high, 20.0, epsilon = 1e-9);
    }

    #[test]
    #[should_panic(expected = "la puissance électrique Pe doit être positive")]
    fn zero_electrical_power_panics() {
        peltier_cop(30.0, 0.0);
    }
}
