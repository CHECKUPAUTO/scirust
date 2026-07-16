//! **Relais de protection à maximum de courant (temps inverse IEC)** — multiple du
//! réglage (PSM) à partir du courant de défaut et du rapport de transformateur de
//! courant, puis temps de déclenchement selon les courbes normalisées IEC 60255
//! (standard inverse, très inverse, extrêmement inverse).
//!
//! ```text
//! multiple du réglage          PSM = I_f / (K_TC · I_r)
//! standard inverse (SI)        t = TMS · 0,14 / (PSM^0,02 − 1)
//! très inverse (VI)            t = TMS · 13,5 / (PSM − 1)
//! extrêmement inverse (EI)     t = TMS · 80 / (PSM² − 1)
//! ```
//!
//! `I_f` courant de défaut (efficace, A) vu au primaire du transformateur de
//! courant, `K_TC` rapport de transformation du transformateur de courant (TC)
//! primaire/secondaire (sans dimension, `> 0`), `I_r` courant de réglage (plug
//! setting) rapporté au secondaire du TC (A, `> 0`), `PSM` multiple du réglage
//! (plug setting multiplier, sans dimension), `TMS` réglage du multiplicateur de
//! temps (time multiplier setting, sans dimension, `>= 0`) et `t` temps de
//! déclenchement du relais (s). Les constantes 0,14 / 0,02 / 13,5 / 80 sont
//! celles des courbes normalisées IEC 60255.
//!
//! **Convention** : SI ; courants efficaces en A, temps en secondes ; grandeurs
//! sans dimension pour PSM, K_TC et TMS ; **régime établi** (le courant de défaut
//! est supposé constant pendant la temporisation).
//! **Limite honnête** : relais à temps inverse selon les courbes **normalisées
//! IEC 60255** (constantes de courbe standard, non modifiables ici) ; les réglages
//! (`PSM` via `I_r` et `TMS`) ainsi que le rapport de TC (`K_TC`) sont **fournis
//! par l'appelant** — aucune valeur « par défaut » n'est inventée. Le PSM doit
//! être **strictement supérieur à 1** pour obtenir un déclenchement temporisé
//! fini (au réglage exact, `PSM = 1`, le dénominateur s'annule). Ce module ne
//! modélise **pas** la sélectivité du réseau (coordination entre étages,
//! intervalle de discrimination) ni les temps de retombée : cela reste à la
//! charge de l'appelant. Arithmétique **réelle** (f64), pas de représentation
//! complexe.

/// Multiple du réglage (plug setting multiplier) `PSM = I_f / (K_TC · I_r)`.
///
/// `fault_current` est le courant de défaut au primaire du TC (A), `
/// current_transformer_ratio` le rapport de transformation du TC (`K_TC`, sans
/// dimension) et `plug_setting_current` le courant de réglage au secondaire du TC
/// (`I_r`, A) ; le résultat est le multiple du réglage (sans dimension).
///
/// Panique si `fault_current < 0`, si `current_transformer_ratio <= 0` ou si
/// `plug_setting_current <= 0`.
pub fn ocr_plug_setting_multiplier(
    fault_current: f64,
    current_transformer_ratio: f64,
    plug_setting_current: f64,
) -> f64 {
    assert!(fault_current >= 0.0, "I_f ≥ 0 requis");
    assert!(current_transformer_ratio > 0.0, "K_TC > 0 requis");
    assert!(plug_setting_current > 0.0, "I_r > 0 requis");
    fault_current / (current_transformer_ratio * plug_setting_current)
}

/// Temps de déclenchement, courbe **standard inverse (SI)** IEC 60255
/// `t = TMS · 0,14 / (PSM^0,02 − 1)`.
///
/// `plug_setting_multiplier` est le multiple du réglage (`PSM`, sans dimension) et
/// `time_multiplier_setting` le réglage du multiplicateur de temps (`TMS`, sans
/// dimension) ; le résultat est le temps de déclenchement en secondes.
///
/// Panique si `plug_setting_multiplier <= 1` (dénominateur nul ou négatif : pas de
/// déclenchement temporisé) ou si `time_multiplier_setting < 0`.
pub fn ocr_iec_standard_inverse_time(
    plug_setting_multiplier: f64,
    time_multiplier_setting: f64,
) -> f64 {
    assert!(plug_setting_multiplier > 1.0, "PSM > 1 requis");
    assert!(time_multiplier_setting >= 0.0, "TMS ≥ 0 requis");
    time_multiplier_setting * 0.14 / (plug_setting_multiplier.powf(0.02) - 1.0)
}

/// Temps de déclenchement, courbe **très inverse (VI)** IEC 60255
/// `t = TMS · 13,5 / (PSM − 1)`.
///
/// `plug_setting_multiplier` est le multiple du réglage (`PSM`, sans dimension) et
/// `time_multiplier_setting` le réglage du multiplicateur de temps (`TMS`, sans
/// dimension) ; le résultat est le temps de déclenchement en secondes.
///
/// Panique si `plug_setting_multiplier <= 1` (dénominateur nul ou négatif : pas de
/// déclenchement temporisé) ou si `time_multiplier_setting < 0`.
pub fn ocr_iec_very_inverse_time(
    plug_setting_multiplier: f64,
    time_multiplier_setting: f64,
) -> f64 {
    assert!(plug_setting_multiplier > 1.0, "PSM > 1 requis");
    assert!(time_multiplier_setting >= 0.0, "TMS ≥ 0 requis");
    time_multiplier_setting * 13.5 / (plug_setting_multiplier - 1.0)
}

/// Temps de déclenchement, courbe **extrêmement inverse (EI)** IEC 60255
/// `t = TMS · 80 / (PSM² − 1)`.
///
/// `plug_setting_multiplier` est le multiple du réglage (`PSM`, sans dimension) et
/// `time_multiplier_setting` le réglage du multiplicateur de temps (`TMS`, sans
/// dimension) ; le résultat est le temps de déclenchement en secondes.
///
/// Panique si `plug_setting_multiplier <= 1` (dénominateur nul ou négatif : pas de
/// déclenchement temporisé) ou si `time_multiplier_setting < 0`.
pub fn ocr_iec_extremely_inverse_time(
    plug_setting_multiplier: f64,
    time_multiplier_setting: f64,
) -> f64 {
    assert!(plug_setting_multiplier > 1.0, "PSM > 1 requis");
    assert!(time_multiplier_setting >= 0.0, "TMS ≥ 0 requis");
    time_multiplier_setting * 80.0 / (plug_setting_multiplier * plug_setting_multiplier - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn psm_realistic_value() {
        // Cas chiffré : défaut 4000 A, TC 400/1 (K_TC = 400), réglage I_r = 1 A
        // → PSM = 4000 / (400 · 1) = 10.
        let psm = ocr_plug_setting_multiplier(4000.0, 400.0, 1.0);
        assert_relative_eq!(psm, 10.0, epsilon = 1e-9);
    }

    #[test]
    fn psm_scales_with_fault_current() {
        // Proportionnalité : doubler le courant de défaut double le PSM.
        let p1 = ocr_plug_setting_multiplier(2000.0, 400.0, 1.0);
        let p2 = ocr_plug_setting_multiplier(4000.0, 400.0, 1.0);
        assert_relative_eq!(p2 / p1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn standard_inverse_realistic_value() {
        // Cas chiffré : PSM = 10, TMS = 0,5, courbe SI.
        // t = 0,5 · 0,14 / (10^0,02 − 1) = 0,5 · 0,14 / 0,047128548… ≈ 1,485297 s.
        let t = ocr_iec_standard_inverse_time(10.0, 0.5);
        assert_relative_eq!(t, 1.485297, epsilon = 1e-3);
    }

    #[test]
    fn very_and_extremely_inverse_exact_values() {
        // PSM = 10, TMS = 0,5.
        // VI : 0,5 · 13,5 / (10 − 1) = 6,75 / 9 = 0,75 s.
        assert_relative_eq!(ocr_iec_very_inverse_time(10.0, 0.5), 0.75, epsilon = 1e-9);
        // EI : 0,5 · 80 / (100 − 1) = 40 / 99 ≈ 0,404040… s.
        assert_relative_eq!(
            ocr_iec_extremely_inverse_time(10.0, 0.5),
            40.0 / 99.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn time_is_proportional_to_tms() {
        // Toutes les courbes sont linéaires en TMS (à PSM fixé).
        let psm = 5.0_f64;
        assert_relative_eq!(
            ocr_iec_standard_inverse_time(psm, 1.0),
            2.0 * ocr_iec_standard_inverse_time(psm, 0.5),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            ocr_iec_very_inverse_time(psm, 1.0),
            2.0 * ocr_iec_very_inverse_time(psm, 0.5),
            epsilon = 1e-12
        );
        assert_relative_eq!(
            ocr_iec_extremely_inverse_time(psm, 1.0),
            2.0 * ocr_iec_extremely_inverse_time(psm, 0.5),
            epsilon = 1e-12
        );
    }

    #[test]
    fn zero_tms_gives_instantaneous_zero_time() {
        // Cas limite TMS = 0 : temps de déclenchement nul sur toutes les courbes.
        assert_relative_eq!(
            ocr_iec_standard_inverse_time(3.0, 0.0),
            0.0,
            epsilon = 1e-12
        );
        assert_relative_eq!(ocr_iec_very_inverse_time(3.0, 0.0), 0.0, epsilon = 1e-12);
        assert_relative_eq!(
            ocr_iec_extremely_inverse_time(3.0, 0.0),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    #[should_panic(expected = "PSM > 1 requis")]
    fn psm_at_setting_panics() {
        // Au réglage exact (PSM = 1), le dénominateur s'annule → panique.
        ocr_iec_standard_inverse_time(1.0, 0.5);
    }
}
