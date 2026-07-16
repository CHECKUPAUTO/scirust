//! **Essai d'isolement (mégohmmètre)** — module d'indicateurs d'un essai
//! d'isolement en courant continu : résistance d'isolement mesurée à partir de
//! la tension d'essai et du courant de fuite, indice de polarisation, rapport
//! d'absorption diélectrique et correction de la résistance à une température de
//! référence.
//!
//! ```text
//! résistance d'isolement       R_iso = U_essai / I_fuite
//! indice de polarisation       PI  = R_10min / R_1min
//! rapport d'absorption diél.   DAR = R_60s / R_30s
//! correction en température     R_ref = R_mes · k^((T_mes − T_ref) / 10)
//! ```
//!
//! `R_iso` résistance d'isolement (Ω), `U_essai` tension d'essai continue
//! appliquée (V), `I_fuite` courant de fuite mesuré (A), `PI` indice de
//! polarisation (sans dimension), `R_10min` résistance lue à 10 min (Ω),
//! `R_1min` résistance lue à 1 min (Ω), `DAR` rapport d'absorption diélectrique
//! (sans dimension), `R_60s` résistance lue à 60 s (Ω), `R_30s` résistance lue à
//! 30 s (Ω), `R_ref` résistance ramenée à la température de référence (Ω),
//! `R_mes` résistance mesurée (Ω), `k` coefficient de correction par tranche de
//! 10 °C (sans dimension), `T_mes` température de mesure (°C), `T_ref`
//! température de référence (°C).
//!
//! **Convention** : SI ; résistances en Ω, tension en V, courant en A,
//! températures en degrés Celsius. **Limite honnête** : mesure d'isolement en
//! **courant continu** ; la tension d'essai et le courant de fuite sont
//! **fournis par l'appelant** (relevé du mégohmmètre). L'indice de polarisation
//! (`PI > 2` souvent jugé bon) et le rapport d'absorption diélectrique
//! caractérisent l'**humidité et le vieillissement** de l'isolant à partir de
//! seuils purement **indicatifs**. La correction en température utilise un
//! coefficient `k` **fourni** par l'appelant (l'isolement chute quand la
//! température augmente). Ces valeurs sont **indicatives** et ne constituent pas
//! un diagnostic ; le module ne fait qu'appliquer les définitions aux grandeurs
//! fournies.

/// Résistance d'isolement `R_iso = applied_voltage / leakage_current` (Ω),
/// loi d'Ohm appliquée à la tension d'essai continue et au courant de fuite.
///
/// Panique si `applied_voltage <= 0` ou si `leakage_current <= 0` (tension
/// d'essai non physique ou division par zéro).
pub fn instest_insulation_resistance(applied_voltage: f64, leakage_current: f64) -> f64 {
    assert!(
        applied_voltage > 0.0,
        "la tension d'essai applied_voltage doit être strictement positive"
    );
    assert!(
        leakage_current > 0.0,
        "le courant de fuite leakage_current doit être strictement positif"
    );
    applied_voltage / leakage_current
}

/// Indice de polarisation `PI = resistance_10min / resistance_1min` (sans
/// dimension), rapport des résistances d'isolement lues à 10 min et à 1 min
/// (`PI > 2` est souvent jugé bon, seuil indicatif).
///
/// Panique si `resistance_10min < 0` ou si `resistance_1min <= 0` (résistance
/// non physique ou division par zéro).
pub fn instest_polarization_index(resistance_10min: f64, resistance_1min: f64) -> f64 {
    assert!(
        resistance_10min >= 0.0,
        "la résistance à 10 min resistance_10min doit être ≥ 0"
    );
    assert!(
        resistance_1min > 0.0,
        "la résistance à 1 min resistance_1min doit être strictement positive"
    );
    resistance_10min / resistance_1min
}

/// Rapport d'absorption diélectrique `DAR = resistance_60s / resistance_30s`
/// (sans dimension), rapport des résistances d'isolement lues à 60 s et à 30 s.
///
/// Panique si `resistance_60s < 0` ou si `resistance_30s <= 0` (résistance non
/// physique ou division par zéro).
pub fn instest_dielectric_absorption_ratio(resistance_60s: f64, resistance_30s: f64) -> f64 {
    assert!(
        resistance_60s >= 0.0,
        "la résistance à 60 s resistance_60s doit être ≥ 0"
    );
    assert!(
        resistance_30s > 0.0,
        "la résistance à 30 s resistance_30s doit être strictement positive"
    );
    resistance_60s / resistance_30s
}

/// Résistance d'isolement corrigée à la température de référence
/// `R_ref = measured_resistance · temperature_coefficient^((measured_temperature
/// − reference_temperature) / 10)` (Ω), le coefficient `temperature_coefficient`
/// étant donné par tranche de 10 °C et fourni par l'appelant.
///
/// Panique si `measured_resistance < 0` ou si `temperature_coefficient <= 0`
/// (résistance non physique ou base d'exponentiation non physique).
pub fn instest_temperature_corrected_resistance(
    measured_resistance: f64,
    temperature_coefficient: f64,
    measured_temperature: f64,
    reference_temperature: f64,
) -> f64 {
    assert!(
        measured_resistance >= 0.0,
        "la résistance mesurée measured_resistance doit être ≥ 0"
    );
    assert!(
        temperature_coefficient > 0.0,
        "le coefficient temperature_coefficient doit être strictement positif"
    );
    let exponent = (measured_temperature - reference_temperature) / 10.0;
    measured_resistance * temperature_coefficient.powf(exponent)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn insulation_resistance_reciprocal_of_ohms_law() {
        // Réciprocité loi d'Ohm : U = 5000 V, I_fuite = 1 µA = 1e-6 A →
        //   R = 5000 / 1e-6 = 5e9 Ω = 5000 MΩ. En repartant de R et I on
        //   retrouve U.
        let r = instest_insulation_resistance(5000.0, 1.0e-6);
        assert_relative_eq!(r, 5.0e9, epsilon = 1e-3);
        assert_relative_eq!(r * 1.0e-6, 5000.0, epsilon = 1e-3);
    }

    #[test]
    fn insulation_resistance_scales_inversely_with_leakage() {
        // Proportionnalité inverse : à tension fixée, doubler le courant de
        // fuite divise la résistance d'isolement par deux.
        let r1 = instest_insulation_resistance(1000.0, 2.0e-6);
        let r2 = instest_insulation_resistance(1000.0, 4.0e-6);
        assert_relative_eq!(r2, r1 / 2.0, epsilon = 1e-3);
    }

    #[test]
    fn polarization_index_unity_when_stable() {
        // Cas limite : isolement stable dans le temps (R identique à 1 min et
        //   10 min) → PI = 1.
        assert_relative_eq!(
            instest_polarization_index(1000.0e6, 1000.0e6),
            1.0,
            epsilon = 1e-9
        );
        // Cas chiffré : R_10min = 5000 MΩ, R_1min = 1000 MΩ → PI = 5.
        let pi = instest_polarization_index(5000.0e6, 1000.0e6);
        assert_relative_eq!(pi, 5.0, epsilon = 1e-9);
        assert!(pi > 2.0);
    }

    #[test]
    fn dielectric_absorption_ratio_numeric() {
        // Cas chiffré : R_60s = 1200 MΩ, R_30s = 800 MΩ →
        //   DAR = 1200 / 800 = 1,5.
        let dar = instest_dielectric_absorption_ratio(1200.0e6, 800.0e6);
        assert_relative_eq!(dar, 1.5, epsilon = 1e-9);
    }

    #[test]
    fn temperature_correction_identity_and_numeric() {
        // Cas limite : à la température de référence (exposant nul), la
        //   correction laisse la résistance inchangée.
        assert_relative_eq!(
            instest_temperature_corrected_resistance(500.0e6, 2.0, 20.0, 20.0),
            500.0e6,
            epsilon = 1e-3
        );
        // Cas chiffré : R_mes = 100 MΩ, k = 2, T_mes = 40 °C, T_ref = 20 °C →
        //   exposant = (40 − 20) / 10 = 2 ; 2^2 = 4 → R_ref = 100e6 · 4 = 4e8 Ω.
        let r_ref = instest_temperature_corrected_resistance(100.0e6, 2.0, 40.0, 20.0);
        assert_relative_eq!(r_ref, 4.0e8, epsilon = 1e-3);
    }

    #[test]
    #[should_panic(expected = "le courant de fuite leakage_current doit être strictement positif")]
    fn zero_leakage_current_panics() {
        instest_insulation_resistance(5000.0, 0.0);
    }
}
