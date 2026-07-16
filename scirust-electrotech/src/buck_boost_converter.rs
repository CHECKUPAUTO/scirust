//! **Hacheur inverseur (buck-boost, abaisseur-élévateur) en conduction continue**
//! — magnitude de la tension de sortie (polarité **inversée**), rapport cyclique
//! requis, ondulation du courant d'inductance et test d'élévation.
//!
//! ```text
//! magnitude de sortie   |V_out| = V_in · D / (1 − D)
//! rapport cyclique       D       = |V_out| / (V_in + |V_out|)
//! ondulation de courant  ΔI_L    = V_in · D / (L · f_sw)
//! régime d'élévation     |V_out| > V_in  ⟺  D > 0,5
//! ```
//!
//! `V_in` tension d'entrée (V), `|V_out|` **magnitude** de la tension de sortie
//! moyenne (V ; la sortie réelle est de polarité **inversée**, `V_out = −|V_out|`),
//! `D` rapport cyclique (sans dimension, `0 < D < 1`), `ΔI_L` ondulation
//! crête-à-crête du courant dans l'inductance (A), `L` inductance de stockage (H),
//! `f_sw` fréquence de découpage (Hz). En régime permanent, `|V_out| = V_in · D /
//! (1 − D)` et `D = |V_out| / (V_in + |V_out|)` sont réciproques. Le hacheur
//! inverseur abaisse la tension pour `D < 0,5` et l'élève pour `D > 0,5` (avec
//! `|V_out| = V_in` exactement à `D = 0,5`).
//!
//! **Convention** : SI ; tensions en V, courants en A, inductance en H, fréquence
//! en Hz. **Limite honnête** : conduction **continue** (CCM), **régime
//! permanent**, interrupteur, diode, inductance et condensateur supposés
//! **idéaux** (résistances série, chutes directes et pertes de commutation
//! négligées ; rendement unitaire). La sortie est de **polarité inversée** : on
//! ne renvoie que sa **magnitude** `|V_out|`. La magnitude tend vers l'infini
//! quand `D → 1` (comportement non physique, borné en pratique par les pertes
//! réelles). La tension d'entrée `V_in`, le rapport cyclique `D` (`0 < D < 1`),
//! la magnitude de sortie visée `|V_out|` et les valeurs de composants (`L`,
//! `f_sw`) sont **fournis par l'appelant** (réseau, commande, choix de
//! dimensionnement) — aucune valeur « par défaut » n'est inventée.

/// Magnitude de la tension de sortie d'un hacheur inverseur
/// `|V_out| = V_in · D / (1 − D)` (V ; la sortie réelle est **inversée**).
///
/// Panique si `input_voltage < 0` ou si `duty_cycle` n'est pas dans `]0, 1[`.
pub fn buckboost_output_voltage_magnitude(input_voltage: f64, duty_cycle: f64) -> f64 {
    assert!(
        input_voltage >= 0.0,
        "la tension d'entrée V_in doit être ≥ 0"
    );
    assert!(
        duty_cycle > 0.0 && duty_cycle < 1.0,
        "le rapport cyclique D doit être dans ]0, 1[ (conduction continue)"
    );
    input_voltage * duty_cycle / (1.0 - duty_cycle)
}

/// Rapport cyclique requis pour obtenir la magnitude `|V_out|` à partir de
/// `V_in`, `D = |V_out| / (V_in + |V_out|)` (sans dimension).
///
/// Panique si `input_voltage <= 0` ou si `output_voltage_magnitude < 0`.
pub fn buckboost_duty_for_output(input_voltage: f64, output_voltage_magnitude: f64) -> f64 {
    assert!(
        input_voltage > 0.0,
        "la tension d'entrée V_in doit être strictement positive"
    );
    assert!(
        output_voltage_magnitude >= 0.0,
        "la magnitude de sortie |V_out| doit être ≥ 0"
    );
    output_voltage_magnitude / (input_voltage + output_voltage_magnitude)
}

/// Ondulation crête-à-crête du courant d'inductance
/// `ΔI_L = V_in · D / (L · f_sw)` (A).
///
/// Panique si `input_voltage < 0`, si `duty_cycle` n'est pas dans `]0, 1[`, si
/// `inductance <= 0` ou si `switching_frequency <= 0`.
pub fn buckboost_inductor_ripple_current(
    input_voltage: f64,
    duty_cycle: f64,
    inductance: f64,
    switching_frequency: f64,
) -> f64 {
    assert!(
        input_voltage >= 0.0,
        "la tension d'entrée V_in doit être ≥ 0"
    );
    assert!(
        duty_cycle > 0.0 && duty_cycle < 1.0,
        "le rapport cyclique D doit être dans ]0, 1[ (conduction continue)"
    );
    assert!(
        inductance > 0.0,
        "l'inductance L doit être strictement positive"
    );
    assert!(
        switching_frequency > 0.0,
        "la fréquence de découpage f_sw doit être strictement positive"
    );
    input_voltage * duty_cycle / (inductance * switching_frequency)
}

/// Indique si le hacheur inverseur fonctionne en **élévation** (`|V_out| > V_in`),
/// c'est-à-dire `D > 0,5` (à `D = 0,5`, `|V_out| = V_in`).
///
/// Panique si `duty_cycle` n'est pas dans `]0, 1[`.
pub fn buckboost_is_step_up(duty_cycle: f64) -> bool {
    assert!(
        duty_cycle > 0.0 && duty_cycle < 1.0,
        "le rapport cyclique D doit être dans ]0, 1[ (conduction continue)"
    );
    duty_cycle > 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn output_and_duty_are_reciprocal() {
        // Réciprocité : D = |V_out| / (V_in + |V_out|) inverse
        // |V_out| = V_in · D / (1 − D).
        let v_in = 12.0_f64;
        let duty = 0.6_f64;
        let v_mag = buckboost_output_voltage_magnitude(v_in, duty);
        assert_relative_eq!(
            buckboost_duty_for_output(v_in, v_mag),
            duty,
            epsilon = 1e-12
        );
    }

    #[test]
    fn output_magnitude_scales_linearly_with_input() {
        // Proportionnalité : à D fixé, doubler V_in double |V_out|.
        let m1 = buckboost_output_voltage_magnitude(10.0, 0.4);
        let m2 = buckboost_output_voltage_magnitude(20.0, 0.4);
        assert_relative_eq!(m2 / m1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn output_magnitude_reference_case() {
        // Cas chiffré : V_in = 12 V, D = 0,6, 1 − D = 0,4.
        //   |V_out| = 12 · 0,6 / 0,4 = 7,2 / 0,4 = 18 V.
        let v_mag = buckboost_output_voltage_magnitude(12.0, 0.6);
        assert_relative_eq!(v_mag, 18.0, epsilon = 1e-12);
    }

    #[test]
    fn unity_gain_at_half_duty() {
        // Cas limite : à D = 0,5, |V_out| = V_in · 0,5 / 0,5 = V_in.
        let v_in = 24.0_f64;
        let v_mag = buckboost_output_voltage_magnitude(v_in, 0.5);
        assert_relative_eq!(v_mag, v_in, epsilon = 1e-12);
        // Frontière d'élévation : D = 0,5 n'est pas encore une élévation.
        assert!(!buckboost_is_step_up(0.5));
        assert!(buckboost_is_step_up(0.5 + 1e-6));
        assert!(!buckboost_is_step_up(0.5 - 1e-6));
    }

    #[test]
    fn inductor_ripple_reference_case() {
        // Cas chiffré : V_in = 12 V, D = 0,6, L = 100 µH = 100e-6 H,
        //   f_sw = 100 kHz = 100e3 Hz. L · f_sw = 100e-6 · 100e3 = 10.
        //   ΔI_L = 12 · 0,6 / 10 = 7,2 / 10 = 0,72 A.
        let ripple = buckboost_inductor_ripple_current(12.0, 0.6, 100e-6, 100e3);
        assert_relative_eq!(ripple, 0.72, epsilon = 1e-9);
    }

    #[test]
    fn output_magnitude_diverges_as_duty_approaches_one() {
        // Cas limite : quand D → 1, |V_out| = V_in · D / (1 − D) → +∞ (non physique).
        let v_mag = buckboost_output_voltage_magnitude(12.0, 0.99);
        assert!(v_mag > 1000.0, "|V_out| doit diverger quand D → 1");
    }

    #[test]
    #[should_panic(expected = "le rapport cyclique D doit être dans ]0, 1[")]
    fn output_magnitude_rejects_duty_one() {
        buckboost_output_voltage_magnitude(12.0, 1.0);
    }
}
