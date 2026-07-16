//! **Convertisseur flyback (à isolement) en conduction continue** — tension de
//! sortie, rapport cyclique requis, tension réfléchie au primaire et contrainte
//! de tension sur l'interrupteur (hors pic de fuite).
//!
//! ```text
//! tension de sortie (CCM)   V_out = V_in · n · D / (1 − D)
//! rapport cyclique requis    D     = V_out / (V_out + V_in · n)
//! tension réfléchie          V_r   = V_out / n
//! contrainte interrupteur    V_sw  = V_in + V_r
//! ```
//!
//! `V_in` tension d'entrée (V), `V_out` tension de sortie moyenne (V), `n`
//! rapport de spires secondaire/primaire `n = Ns/Np` (sans dimension, `n > 0`),
//! `D` rapport cyclique (sans dimension, `0 < D < 1`), `V_r` tension réfléchie
//! au primaire (V), `V_sw` contrainte de tension à l'état bloqué sur
//! l'interrupteur (V). En régime permanent, `V_out = V_in · n · D / (1 − D)` et
//! `D = V_out / (V_out + V_in · n)` sont réciproques. La contrainte sur
//! l'interrupteur additionne la tension d'entrée et la tension réfléchie
//! `V_sw = V_in + V_out / n`.
//!
//! **Convention** : SI ; tensions en V, rapports (`n`, `D`) sans dimension.
//! **Limite honnête** : convertisseur flyback en conduction **continue** (CCM),
//! **régime permanent**, transformateur et interrupteur supposés **idéaux**
//! (résistances série, chutes directes, pertes de commutation et courant
//! magnétisant négligés ; rendement unitaire). Le rapport de spires `n = Ns/Np`
//! est **fourni par l'appelant** (choix du transformateur). La contrainte de
//! tension sur l'interrupteur `V_sw` **ignore la surtension** due à l'inductance
//! de fuite (le pic de fuite et le snubber d'écrêtage sont à dimensionner par
//! l'appelant). L'**isolement galvanique** est assuré par le transformateur. La
//! tension de sortie diverge quand `D → 1` (non physique, borné en pratique par
//! les pertes réelles). Les tensions `V_in`, `V_out` et les rapports `n`, `D`
//! sont **fournis par l'appelant** — aucune valeur « par défaut » n'est inventée.

/// Tension de sortie d'un convertisseur flyback en conduction continue
/// `V_out = V_in · n · D / (1 − D)` (V), avec `n = Ns/Np`.
///
/// Panique si `input_voltage < 0`, si `turns_ratio_secondary_primary <= 0` ou si
/// `duty_cycle` n'est pas dans `]0, 1[`.
pub fn flyback_output_voltage_ccm(
    input_voltage: f64,
    turns_ratio_secondary_primary: f64,
    duty_cycle: f64,
) -> f64 {
    assert!(
        input_voltage >= 0.0,
        "la tension d'entrée V_in doit être ≥ 0"
    );
    assert!(
        turns_ratio_secondary_primary > 0.0,
        "le rapport de spires n = Ns/Np doit être strictement positif"
    );
    assert!(
        duty_cycle > 0.0 && duty_cycle < 1.0,
        "le rapport cyclique D doit être dans ]0, 1[ (conduction continue)"
    );
    input_voltage * turns_ratio_secondary_primary * duty_cycle / (1.0 - duty_cycle)
}

/// Rapport cyclique requis pour obtenir `V_out` à partir de `V_in` et de `n`,
/// `D = V_out / (V_out + V_in · n)` (sans dimension).
///
/// Panique si `input_voltage < 0`, si `output_voltage < 0`, si
/// `turns_ratio_secondary_primary <= 0` ou si le dénominateur
/// `V_out + V_in · n` est nul (`V_out = 0` et `V_in = 0`).
pub fn flyback_duty_for_output(
    input_voltage: f64,
    output_voltage: f64,
    turns_ratio_secondary_primary: f64,
) -> f64 {
    assert!(
        input_voltage >= 0.0,
        "la tension d'entrée V_in doit être ≥ 0"
    );
    assert!(
        output_voltage >= 0.0,
        "la tension de sortie V_out doit être ≥ 0"
    );
    assert!(
        turns_ratio_secondary_primary > 0.0,
        "le rapport de spires n = Ns/Np doit être strictement positif"
    );
    let denominator = output_voltage + input_voltage * turns_ratio_secondary_primary;
    assert!(
        denominator > 0.0,
        "le dénominateur V_out + V_in · n doit être strictement positif"
    );
    output_voltage / denominator
}

/// Tension réfléchie au primaire d'un convertisseur flyback
/// `V_r = V_out / n` (V), avec `n = Ns/Np`.
///
/// Panique si `output_voltage < 0` ou si `turns_ratio_secondary_primary <= 0`.
pub fn flyback_reflected_voltage(output_voltage: f64, turns_ratio_secondary_primary: f64) -> f64 {
    assert!(
        output_voltage >= 0.0,
        "la tension de sortie V_out doit être ≥ 0"
    );
    assert!(
        turns_ratio_secondary_primary > 0.0,
        "le rapport de spires n = Ns/Np doit être strictement positif"
    );
    output_voltage / turns_ratio_secondary_primary
}

/// Contrainte de tension à l'état bloqué sur l'interrupteur d'un flyback
/// `V_sw = V_in + V_r` (V), **hors** pic de tension dû à l'inductance de fuite.
///
/// Panique si `input_voltage < 0` ou si `reflected_voltage < 0`.
pub fn flyback_switch_voltage_stress(input_voltage: f64, reflected_voltage: f64) -> f64 {
    assert!(
        input_voltage >= 0.0,
        "la tension d'entrée V_in doit être ≥ 0"
    );
    assert!(
        reflected_voltage >= 0.0,
        "la tension réfléchie V_r doit être ≥ 0"
    );
    input_voltage + reflected_voltage
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn output_and_duty_are_reciprocal() {
        // Réciprocité : D = V_out / (V_out + V_in · n) inverse
        // V_out = V_in · n · D / (1 − D).
        let v_in = 24.0_f64;
        let n = 0.5_f64;
        let duty = 0.4_f64;
        let v_out = flyback_output_voltage_ccm(v_in, n, duty);
        assert_relative_eq!(
            flyback_duty_for_output(v_in, v_out, n),
            duty,
            epsilon = 1e-12
        );
    }

    #[test]
    fn output_scales_linearly_with_turns_ratio() {
        // Proportionnalité : à V_in et D fixés, doubler n double V_out.
        let v1 = flyback_output_voltage_ccm(24.0, 0.5, 0.4);
        let v2 = flyback_output_voltage_ccm(24.0, 1.0, 0.4);
        assert_relative_eq!(v2 / v1, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn output_reference_case() {
        // Cas chiffré : V_in = 24 V, n = 0,5, D = 0,4, 1 − D = 0,6.
        //   V_out = 24 · 0,5 · 0,4 / 0,6 = 4,8 / 0,6 = 8 V.
        let v_out = flyback_output_voltage_ccm(24.0, 0.5, 0.4);
        assert_relative_eq!(v_out, 8.0, epsilon = 1e-12);
    }

    #[test]
    fn reflected_and_switch_stress_reference_case() {
        // Cas chiffré : V_out = 8 V, n = 0,5 ⇒ V_r = 8 / 0,5 = 16 V.
        //   V_sw = V_in + V_r = 24 + 16 = 40 V (hors pic de fuite).
        let v_r = flyback_reflected_voltage(8.0, 0.5);
        assert_relative_eq!(v_r, 16.0, epsilon = 1e-12);
        let v_sw = flyback_switch_voltage_stress(24.0, v_r);
        assert_relative_eq!(v_sw, 40.0, epsilon = 1e-12);
    }

    #[test]
    fn output_diverges_as_duty_approaches_one() {
        // Cas limite : quand D → 1, V_out = V_in · n · D / (1 − D) → +∞.
        let v_out = flyback_output_voltage_ccm(24.0, 0.5, 0.99);
        assert!(v_out > 1000.0, "V_out doit diverger quand D → 1");
    }

    #[test]
    fn reflected_voltage_inverts_scaling() {
        // Identité : V_r = V_out / n renverse le facteur n de la tension de sortie.
        // À partir de V_out = V_in · n · D / (1 − D), on a
        // V_r = V_in · D / (1 − D), indépendant de n.
        let v_out = flyback_output_voltage_ccm(24.0, 0.5, 0.4);
        let v_r = flyback_reflected_voltage(v_out, 0.5);
        // V_in · D / (1 − D) = 24 · 0,4 / 0,6 = 9,6 / 0,6 = 16 V.
        assert_relative_eq!(v_r, 16.0, epsilon = 1e-12);
    }

    #[test]
    #[should_panic(expected = "le rapport cyclique D doit être dans ]0, 1[")]
    fn output_rejects_duty_one() {
        flyback_output_voltage_ccm(24.0, 0.5, 1.0);
    }
}
